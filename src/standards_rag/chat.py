"""Citation-enforced chat orchestration for standards RAG."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import quote

from standards_rag.citation_validation import validate_answer_citations
from standards_rag.conversation_store import ConversationStore
from standards_rag.models import Citation, StandardDocument
from standards_rag.retrieval import InMemoryStandardsStore, SearchResult, resolve_document_pdf_path
from standards_rag.units import convert_measurement, extract_measurements, format_conversion

if TYPE_CHECKING:
    from standards_rag.video import VideoTranscriptStore

SPECIFIC_STANDARD_RE = re.compile(
    r"\b(?:ASTM\s*)?[A-Z]\d{2,5}(?:/[A-Z]\d{2,5})?-\d{2,4}[A-Z]?\b|"
    r"\bISO\s+\d+(?:[-:]\d+)*(?::\d{4})?\b|"
    r"\bBS\s+(?:EN\s+)?\d+(?:[-:]\d+)*(?::\d{4})?\b",
    re.IGNORECASE,
)
PARTIAL_SUPPORT_SCORE_THRESHOLD = 0.08
DOMAIN_ANCHOR_TERMS = {
    "geotextile",
    "geosynthetic",
    "geomembrane",
    "gcl",
    "soil",
    "tensile",
    "shear",
    "flux",
    "permeability",
    "transmissivity",
    "puncture",
    "compaction",
    "stabilization",
    "liner",
    "drainage",
    "astm",
    "standard",
}


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    citations: list[Citation]
    unsupported: bool = False
    needs_clarification: bool = False
    follow_up_suggestions: list[str] = field(default_factory=list)
    videos: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "unsupported": self.unsupported,
            "needs_clarification": self.needs_clarification,
            "follow_up_suggestions": self.follow_up_suggestions,
            "videos": self.videos,
        }


@dataclass
class _ConversationTurn:
    question: str
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class _MethodFamilyRoute:
    requested_families: frozenset[str] = frozenset()

    @property
    def active(self) -> bool:
        return bool(self.requested_families)


class StandardsRagEngine:
    """RAG chat engine that refuses answers without retrieved evidence."""

    def __init__(
        self,
        store: InMemoryStandardsStore,
        *,
        top_k: int = 6,
        min_score: float = 0.005,
        answer_rewriter: Callable[[str, str, list[Citation]], str] | None = None,
        conversation_store: ConversationStore | None = None,
        video_store: "VideoTranscriptStore | None" = None,
        title_generator: Callable[[str, str], str] | None = None,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.answer_rewriter = answer_rewriter
        self.conversation_store = conversation_store
        self.video_store = video_store
        self.title_generator = title_generator
        self._history: dict[str, list[_ConversationTurn]] = {}

    def ask(
        self,
        question: str,
        *,
        conversation_id: str = "default",
        unit_preference: str | None = None,
        user_id: str | None = None,
    ) -> ChatResponse:
        clean_question = question.strip()
        if user_id and self.conversation_store:
            self._hydrate_history_from_store(user_id, conversation_id)
        if not clean_question:
            return ChatResponse(
                answer="Please ask a standards-related question.",
                citations=[],
                needs_clarification=True,
            )

        if not self.store.chunks:
            return self._remember(
                conversation_id,
                clean_question,
                ChatResponse(
                    answer=(
                        "No standards are loaded in this instance yet (the search index is empty). "
                        "Build or point to an index, then restart the API:\n"
                        "- Run `standards-rag ingest <pdf-or-text-folder> --out data/index/standards-index.json` "
                        "from the project root, or set `STANDARDS_INDEX_PATH` in `.env` to your JSON index file.\n"
                        "Until documents are ingested, every question will return no retrieval hits."
                    ),
                    citations=[],
                    needs_clarification=True,
                    follow_up_suggestions=[
                        "After ingesting, ask again using words that appear in the standard (designation, topic).",
                    ],
                ),
            )

        if self._should_clarify(clean_question):
            return self._remember(
                conversation_id,
                clean_question,
                ChatResponse(
                    answer=(
                        "Can you narrow that down to a material, test method, topic, or standards "
                        "body? I need enough detail to retrieve the right standard before answering."
                    ),
                    citations=[],
                    needs_clarification=True,
                    follow_up_suggestions=[
                        "Which standards mention fly ash stabilization?",
                        "Compare compaction requirements across the loaded ASTM standards.",
                    ],
                ),
            )

        if self._is_out_of_scope_query(clean_question):
            return self._remember(
                conversation_id,
                clean_question,
                ChatResponse(
                    answer=(
                        "I could not find support for that request in the loaded standards "
                        "collection. The query also appears outside the geosynthetics standards "
                        "scope, so I should not answer it without citations."
                    ),
                    citations=[],
                    unsupported=True,
                    follow_up_suggestions=[
                        "Ask about geotextile/geosynthetic test methods in the loaded ASTM set.",
                        "Include an exact standard designation like D4595-24 or D5887-23.",
                    ],
                ),
                user_id=user_id,
                unit_preference=unit_preference,
            )

        query, scoped_document_ids = self._contextual_query(clean_question, conversation_id)
        route = _parse_method_family_route(clean_question)
        routed_document_ids = _document_ids_for_route(route, self.store.documents)
        search_document_ids = _merge_document_filters(scoped_document_ids, routed_document_ids)
        results = self._search_with_relaxation(query, search_document_ids)
        if _is_applicability_question(clean_question):
            results = self._plan_applicability_retrieval(
                clean_question,
                query,
                scoped_document_ids,
                route=route,
                routed_document_ids=routed_document_ids,
                seed_results=results,
            )
            if route.active:
                results = self._hydrate_applicability_facets(
                    clean_question,
                    route,
                    scoped_document_ids,
                    results,
                )

        if not results:
            n_chunks = len(self.store.chunks)
            hint = (
                f"(The loaded index has {n_chunks} text chunks, but nothing matched this query strongly enough.) "
                if n_chunks
                else ""
            )
            return self._remember(
                conversation_id,
                clean_question,
                ChatResponse(
                    answer=(
                        "I could not find support for that in the loaded standards collection. "
                        "I should not answer this from general knowledge without a cited source.\n\n"
                        f"{hint}"
                        "Tips: try naming an ASTM designation in the question, use terms that appear in the "
                        "standard's scope or title, or broaden the topic. If you expected different documents, "
                        "confirm `STANDARDS_INDEX_PATH` and re-ingest."
                    ),
                    citations=[],
                    unsupported=True,
                    follow_up_suggestions=[
                        "Paste the exact standard designation (e.g. D7762-18) into your question.",
                        "Ask what a specific loaded standard covers.",
                    ],
                ),
                user_id=user_id,
                unit_preference=unit_preference,
            )

        if self._needs_partial_clarification(clean_question, results):
            return self._remember(
                conversation_id,
                clean_question,
                self._partial_support_response(results),
                user_id=user_id,
                unit_preference=unit_preference,
            )

        if _is_applicability_question(clean_question):
            response = self._answer_applicability(
                clean_question,
                results,
                unit_preference,
                route=route,
            )
        elif _is_compare_question(clean_question):
            response = self._answer_comparison(clean_question, results, unit_preference)
        elif _is_context_meaning_question(clean_question, results):
            response = self._answer_context_meanings(clean_question, results, unit_preference)
        elif _is_find_question(clean_question):
            response = self._answer_find(clean_question, results)
        else:
            response = self._answer_direct(clean_question, results, unit_preference)

        response = self._maybe_rewrite_answer(clean_question, response)
        response = self._verify_citations_in_answer(response)
        response = self._attach_videos(clean_question, response)
        return self._remember(
            conversation_id,
            clean_question,
            response,
            user_id=user_id,
            unit_preference=unit_preference,
        )

    def _search_with_relaxation(
        self, query: str, document_ids: set[str] | None
    ) -> list[SearchResult]:
        """Return lexical hits, relaxing score threshold and document scope when needed."""
        relax_min = float(os.getenv("STANDARDS_SEARCH_MIN_SCORE_RELAXED", "0.0"))
        relax_k = max(self.top_k * 3, 16)

        def run(dids: set[str] | None, min_s: float, tk: int) -> list[SearchResult]:
            if hasattr(self.store, "search_with_citation_rerank"):
                return self.store.search_with_citation_rerank(
                    query, top_k=tk, document_ids=dids, min_score=min_s
                )
            return self.store.search(query, top_k=tk, document_ids=dids, min_score=min_s)

        hits = run(document_ids, self.min_score, self.top_k)
        if hits:
            return hits
        hits = run(document_ids, relax_min, relax_k)
        if hits:
            return hits
        if document_ids is None:
            return []
        hits = run(None, self.min_score, self.top_k)
        if hits:
            return hits
        return run(None, relax_min, relax_k)

    def _hydrate_applicability_facets(
        self,
        question: str,
        route: _MethodFamilyRoute,
        scoped_document_ids: set[str] | None,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Issue targeted retrieves for material-exclusion and design-limitation facets per routed method."""
        merged = list(results)
        seen_chunks = {r.chunk.chunk_id for r in merged}
        for base in _ordered_primary_bases(route):
            doc_scope_full = _document_ids_for_canonical_base(self.store, base)
            if scoped_document_ids is not None:
                doc_scope = doc_scope_full & scoped_document_ids
                if not doc_scope:
                    doc_scope = doc_scope_full
            else:
                doc_scope = doc_scope_full
            if not doc_scope:
                continue
            for facet_name, queries in APPLICABILITY_FACET_QUERIES.get(base, ()):
                if _facet_already_satisfied(merged, doc_scope, base, facet_name):
                    continue
                for q_suffix in queries:
                    targeted_q = (
                        f"{question}\nRetrieve {base}: {facet_name.replace('_', ' ')} evidence. "
                        f"{q_suffix}"
                    )
                    hits = self.store.search(
                        targeted_q,
                        top_k=6,
                        document_ids=doc_scope,
                        min_score=0.003,
                    )
                    for hit in hits:
                        if hit.chunk.chunk_id not in seen_chunks:
                            merged.append(hit)
                            seen_chunks.add(hit.chunk.chunk_id)
                    if _facet_already_satisfied(merged, doc_scope, base, facet_name):
                        break
        return merged

    def _verify_citations_in_answer(self, response: ChatResponse) -> ChatResponse:
        if response.unsupported or response.needs_clarification or not response.citations:
            return response
        if response.answer.startswith("There are multiple context-dependent meanings/usages"):
            return response

        answer, citations = validate_answer_citations(
            response.answer, response.citations, self.store.chunks
        )
        return ChatResponse(
            answer=answer,
            citations=citations,
            unsupported=response.unsupported,
            needs_clarification=response.needs_clarification,
            follow_up_suggestions=response.follow_up_suggestions,
        )

    def _maybe_rewrite_answer(self, question: str, response: ChatResponse) -> ChatResponse:
        if not self.answer_rewriter:
            return response
        if response.unsupported or response.needs_clarification or not response.citations:
            return response
        if response.answer.startswith("There are multiple context-dependent meanings/usages"):
            return response

        try:
            rewritten = self.answer_rewriter(response.answer, question, response.citations)
        except Exception:
            return response

        if not rewritten.strip():
            return response

        cleaned = _strip_trailing_sources_footer(rewritten.strip())
        return ChatResponse(
            answer=cleaned,
            citations=response.citations,
            unsupported=response.unsupported,
            needs_clarification=response.needs_clarification,
            follow_up_suggestions=response.follow_up_suggestions,
        )

    def _plan_applicability_retrieval(
        self,
        question: str,
        query: str,
        scoped_document_ids: set[str] | None,
        route: _MethodFamilyRoute,
        routed_document_ids: set[str] | None,
        *,
        seed_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Retrieve wider candidates, then rerank for applicability/exclusion/design-limit facets."""
        merged_ids = _merge_document_filters(scoped_document_ids, routed_document_ids)
        wide_k = max(self.top_k * 8, 24)
        wide_min = max(self.min_score * 0.5, 0.003)
        candidates = self.store.search(
            query,
            top_k=wide_k,
            document_ids=merged_ids,
            min_score=wide_min,
        )
        if not candidates:
            candidates = self.store.search(
                query, top_k=wide_k, document_ids=merged_ids, min_score=0.0
            )
        if route.active:
            broad = self.store.search(
                query,
                top_k=wide_k,
                document_ids=scoped_document_ids,
                min_score=wide_min,
            )
            if not broad:
                broad = self.store.search(
                    query, top_k=wide_k, document_ids=scoped_document_ids, min_score=0.0
                )
            candidates = _merge_search_results(candidates, broad)
        if not candidates:
            candidates = seed_results
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda item: _applicability_rank(question, item, route=route),
            reverse=True,
        )
        selected: list[SearchResult] = []
        per_document: dict[str, int] = {}
        for result in ranked:
            document_id = result.document.document_id
            count = per_document.get(document_id, 0)
            if count >= 3:
                continue
            per_document[document_id] = count + 1
            selected.append(result)
            if len(selected) >= max(self.top_k * 3, 12):
                break
        return selected or seed_results

    def _answer_applicability(
        self,
        question: str,
        results: list[SearchResult],
        unit_preference: str | None,
        route: _MethodFamilyRoute,
    ) -> ChatResponse:
        del question
        markers: dict[str, int] = {}
        cited_results: list[SearchResult] = []

        def marker_for(result: SearchResult) -> int:
            idx = markers.get(result.chunk.chunk_id)
            if idx is not None:
                return idx
            cited_results.append(result)
            idx = len(cited_results)
            markers[result.chunk.chunk_id] = idx
            return idx

        grouped_map: dict[str, list[SearchResult]] = defaultdict(list)
        for result in results:
            grouped_map[result.document.document_id].append(result)
        grouped = {
            doc_id: sorted(chunks, key=lambda r: r.score, reverse=True)
            for doc_id, chunks in grouped_map.items()
        }

        if route.active:
            return self._answer_applicability_routed(
                results, grouped, marker_for, route, unit_preference, cited_results
            )

        missing: list[str] = []

        applicable_rows: list[tuple[SearchResult, SearchResult, str | None]] = []
        not_direct_rows: list[tuple[SearchResult, str]] = []
        for group_results in grouped.values():
            best = group_results[0]
            family = _method_family_for_standard(best.document.standard_id)
            label = _METHOD_FAMILY_LABELS.get(family) if family else None

            scope_hit = _best_section_hit(group_results, "scope")
            significance_hit = _best_section_hit(group_results, "significance")
            if scope_hit and significance_hit:
                applicable_rows.append((scope_hit, significance_hit, label))
                continue

            if family is None:
                not_direct_rows.append(
                    (
                        best,
                        "measured property family is not mapped to the parsed objective",
                    )
                )
                continue

            missing_bits: list[str] = []
            if not scope_hit:
                missing_bits.append("Scope")
            if not significance_hit:
                missing_bits.append("Significance and Use")
            not_direct_rows.append(
                (
                    best,
                    f"missing required support from {' + '.join(missing_bits)} sections",
                )
            )

        applicable_rows = applicable_rows[:4]

        def _legacy_exclusion_ok(result: SearchResult) -> bool:
            if not _supports_exclusion_claim(result):
                return False
            text = result.chunk.text
            if _PROCEDURE_ARTIFACT_TERMS_RE.search(text) and not _material_anchor_hit(text.lower()):
                return False
            return True

        exclusions = [r for r in results if _legacy_exclusion_ok(r)][:6]
        design_limits = [r for r in results if _supports_design_limit_claim(r)][:4]

        lines: list[str] = ["Applicable method(s):"]
        if applicable_rows:
            for scope_hit, significance_hit, label in applicable_rows:
                descriptor = f" ({label})" if label else ""
                lines.append(
                    f"- {scope_hit.document.standard_id}{descriptor}: Scope says "
                    f"{_evidence_sentence(scope_hit)} [{marker_for(scope_hit)}]; "
                    f"Significance and Use says {_evidence_sentence(significance_hit)} "
                    f"[{marker_for(significance_hit)}]"
                )
        else:
            missing.append("Applicable methods with Scope + Significance and Use support")

        if not_direct_rows:
            lines.append("\nNot directly responsive:")
            for result, reason in not_direct_rows[:4]:
                lines.append(
                    f"- {result.document.standard_id}: {_evidence_sentence(result)} "
                    f"[{marker_for(result)}] ({reason})."
                )

        lines.append("\nExclusions / non-applicability:")
        if exclusions:
            for result in exclusions:
                lines.append(
                    f"- {result.document.standard_id}: {_evidence_sentence(result)} [{marker_for(result)}]"
                )
        else:
            missing.append("Explicit exclusions or non-applicability statements")
            lines.append("- Retrieved excerpts do not establish explicit exclusions.")

        lines.append("\nWhy values may not be directly interchangeable with field design values:")
        if design_limits:
            for result in design_limits:
                lines.append(
                    f"- {result.document.standard_id}: {_evidence_sentence(result)} [{marker_for(result)}]"
                )
        else:
            missing.append("Design-limit / index-vs-field limitation statements")
            lines.append(
                "- Retrieved excerpts do not establish a direct design-transfer statement for the selected methods."
            )

        if missing:
            lines.append("\nMissing support:")
            for item in missing:
                lines.append(f"- {item}.")

        answer = "\n".join(lines)
        unit_note = _unit_note(cited_results, unit_preference)
        if unit_note:
            answer += f"\n\nUnit note: {unit_note}"

        citations = _citations_from_results(cited_results)
        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Ask me to narrow this to one standard only.",
                "Ask for exact scope text for each cited method.",
            ],
        )

    def _answer_applicability_routed(
        self,
        results: list[SearchResult],
        grouped: dict[str, list[SearchResult]],
        marker_for: Callable[[SearchResult], int],
        route: _MethodFamilyRoute,
        unit_preference: str | None,
        cited_results: list[SearchResult],
    ) -> ChatResponse:
        """Applicability template when method-family routing is active; keeps exclusions on primary methods only."""
        primary_bases = _ordered_primary_bases(route)

        missing: list[str] = []
        applicable_rows: list[tuple[SearchResult, SearchResult, str | None]] = []
        related_not_primary: list[tuple[SearchResult, str]] = []
        wrong_property: list[tuple[SearchResult, str]] = []

        for base in primary_bases:
            flat = _flatten_grouped_for_canonical_base(grouped, self.store, base)
            if not flat:
                missing.append(f"{base} applicability (no retrieved excerpts)")
                continue
            family = _family_for_canonical_base(base)
            label = _METHOD_FAMILY_LABELS.get(family) if family else None
            scope_hit = _best_section_hit(flat, "scope") or _fallback_section_hit(flat, "scope")
            significance_hit = _best_section_hit(flat, "significance") or _fallback_section_hit(
                flat, "significance"
            )
            if scope_hit and significance_hit:
                applicable_rows.append((scope_hit, significance_hit, label))
            else:
                bits: list[str] = []
                if not scope_hit:
                    bits.append("Scope")
                if not significance_hit:
                    bits.append("Significance and Use")
                missing.append(f"{base} applicability (missing {' + '.join(bits)})")

        for group_results in grouped.values():
            best = group_results[0]
            canon = _canonical_method_base(best.document.standard_id)
            family = _method_family_for_standard(best.document.standard_id)
            if canon in primary_bases:
                continue
            if family is None:
                continue
            if family == "drainage_transmissivity":
                related_not_primary.append(
                    (
                        best,
                        f"measures {_METHOD_FAMILY_LABELS[family]} (continuous in-plane flow paths), "
                        f"not saturated GCL through-thickness index flux or soil–geosynthetic direct shear",
                    )
                )
            else:
                wrong_property.append(
                    (
                        best,
                        f"addresses {_METHOD_FAMILY_LABELS.get(family, 'a different measured property')}, "
                        f"not the parsed primary objective for this question",
                    )
                )

        applicable_rows.sort(
            key=lambda row: _base_route_order(_canonical_method_base(row[0].document.standard_id), route)
        )

        exclusion_picks: list[SearchResult] = []
        for base in primary_bases:
            pool = [
                r
                for r in results
                if _canonical_method_base(r.document.standard_id) == base
                and _material_exclusion_evidence_hit(r, base)
            ]
            if not pool:
                missing.append(f"{base} explicit material applicability / exclusions in retrieved excerpts")
                continue
            best_ex = max(
                pool,
                key=lambda r: (_material_exclusion_strength(r, base), r.score),
            )
            exclusion_picks.append(best_ex)

        lines: list[str] = ["Applicable method(s):"]
        if applicable_rows:
            for scope_hit, significance_hit, label in applicable_rows:
                descriptor = f" ({label})" if label else ""
                if scope_hit is significance_hit:
                    scope_txt = _top_scoring_sentence(
                        scope_hit.chunk.text,
                        ("1.", "scope", "covers", "1.1 ", "1.2 "),
                    )
                    sig_txt = _top_scoring_sentence(
                        significance_hit.chunk.text,
                        (
                            "significance",
                            "5.",
                            "laboratory",
                            "index",
                            "representative",
                            "intended",
                            "interchangeable",
                            "dependence",
                        ),
                    )
                else:
                    scope_txt = _evidence_sentence(scope_hit)
                    sig_txt = _evidence_sentence(significance_hit)
                lines.append(
                    f"- {scope_hit.document.standard_id}{descriptor}: Scope says "
                    f"{scope_txt} [{marker_for(scope_hit)}]; Significance and Use says "
                    f"{sig_txt} [{marker_for(significance_hit)}]"
                )
        else:
            missing.append("Applicable primary methods with Scope + Significance and Use support")

        lines.append("\nExclusions / non-applicability (from applicable methods only):")
        if exclusion_picks:
            for result in exclusion_picks:
                canon = _canonical_method_base(result.document.standard_id)
                lines.append(
                    f"- {result.document.standard_id} ({canon} material limits): "
                    f"{_faceted_sentence_for_exclusion_material(result, canon)} [{marker_for(result)}]"
                )
        else:
            missing.append("Explicit applicability / material exclusions from the applicable methods")
            lines.append("- Retrieved excerpts do not yet establish explicit material exclusions for listed methods.")

        lines.append("\nWhy values may not translate directly into field design values:")
        design_any = False
        for base in primary_bases:
            dl = _best_design_limit_result_for_base(results, base)
            if dl:
                design_any = True
                canon = _canonical_method_base(dl.document.standard_id)
                quote = _faceted_sentence_for_design_limit(dl, base)
                lines.append(f"- {dl.document.standard_id} ({canon}): {quote} [{marker_for(dl)}]")
            else:
                missing.append(
                    f"{base} laboratory-to-field limitation (normal stress / displacement controls or index flux vs design flux)"
                )
        if not design_any:
            lines.append(
                "- Retrieved excerpts do not yet establish method-specific laboratory-to-field transfer limits."
            )

        if related_not_primary:
            lines.append("\nRelated but not primary for this objective:")
            for result, reason in related_not_primary[:4]:
                lines.append(
                    f"- {result.document.standard_id}: {_evidence_sentence(result)} [{marker_for(result)}] ({reason})."
                )

        if wrong_property:
            lines.append("\nNot addressing this question's primary measured properties:")
            for result, reason in wrong_property[:4]:
                lines.append(
                    f"- {result.document.standard_id}: {_evidence_sentence(result)} [{marker_for(result)}] ({reason})."
                )

        if missing:
            lines.append("\nMissing support:")
            for item in missing:
                lines.append(f"- {item}.")

        answer = "\n".join(lines)
        unit_note = _unit_note(cited_results, unit_preference)
        if unit_note:
            answer += f"\n\nUnit note: {unit_note}"

        citations = _citations_from_results(cited_results)
        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Ask me to narrow this to one standard only.",
                "Ask for exact scope text for each cited method.",
            ],
        )

    def _answer_direct(
        self,
        question: str,
        results: list[SearchResult],
        unit_preference: str | None,
    ) -> ChatResponse:
        citations = _citations_from_results(results[:3])
        evidence_lines = [
            f"{_evidence_sentence(result)} [{index}]"
            for index, result in enumerate(results[:3], start=1)
        ]
        answer = "The loaded standards support the following answer:\n\n" + "\n".join(
            f"- {line}" for line in evidence_lines
        )

        unit_note = _unit_note(results, unit_preference)
        if unit_note:
            answer += f"\n\nUnit note: {unit_note}"

        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Do you want this compared against another loaded standard?",
                "Should I return the values in SI or US customary units?",
            ],
        )

    def _answer_find(self, question: str, results: list[SearchResult]) -> ChatResponse:
        del question
        by_document: dict[str, SearchResult] = {}
        for result in results:
            by_document.setdefault(result.document.document_id, result)
        citations = _citations_from_results(by_document.values())
        lines = []
        for index, result in enumerate(by_document.values(), start=1):
            document = result.document
            lines.append(
                f"- {document.standard_id}: {document.title}. "
                f"Relevant passage: {_evidence_sentence(result)} [{index}]"
            )
        answer = "I found these relevant loaded standards:\n\n" + "\n".join(lines)
        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Ask me to compare these standards.",
                "Ask for the relevant section from one standard.",
            ],
        )

    def _answer_comparison(
        self,
        question: str,
        results: list[SearchResult],
        unit_preference: str | None,
    ) -> ChatResponse:
        del question
        grouped: dict[str, list[SearchResult]] = {}
        for result in results:
            grouped.setdefault(result.document.document_id, []).append(result)

        if len(grouped) < 2:
            return self._answer_direct(
                "comparison requested but only one supporting standard was found",
                results,
                unit_preference,
            )

        comparison_lines = []
        cited_results: list[SearchResult] = []
        for index, group_results in enumerate(grouped.values(), start=1):
            best = group_results[0]
            cited_results.append(best)
            comparison_lines.append(
                f"- {best.document.standard_id}: {_evidence_sentence(best)} [{index}]"
            )

        citations = _citations_from_results(cited_results)
        answer = (
            "Here is the source-backed comparison across the loaded standards:\n\n"
            + "\n".join(comparison_lines)
        )
        unit_note = _unit_note(cited_results, unit_preference)
        if unit_note:
            answer += f"\n\nUnit note: {unit_note}"
        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Ask which difference matters for your use case.",
                "Ask for only the sections that mention a specific value or unit.",
            ],
        )

    def _answer_context_meanings(
        self,
        question: str,
        results: list[SearchResult],
        unit_preference: str | None,
    ) -> ChatResponse:
        focus = _focus_term(question)
        filtered_results = _results_for_focus_term(results, focus)
        if len({result.document.document_id for result in filtered_results}) >= 2:
            results = filtered_results

        grouped: dict[str, list[SearchResult]] = {}
        for result in results:
            grouped.setdefault(result.document.document_id, []).append(result)

        bullet_lines: list[str] = []
        cited_results: list[SearchResult] = []
        for group_results in grouped.values():
            best = max(
                group_results,
                key=lambda item: _question_sentence_overlap(question, _evidence_sentence(item)),
            )
            overlap = _question_sentence_overlap(question, _evidence_sentence(best))
            if overlap < 2:
                continue
            cited_results.append(best)
            sentence = _evidence_sentence(best)
            bullet_lines.append(f"- {best.document.standard_id}: {sentence}")
            if len(bullet_lines) == 4:
                break

        if len(cited_results) < 2:
            fallback = list(grouped.values())[:2]
            cited_results = [group[0] for group in fallback]
            bullet_lines = [
                f"- {result.document.standard_id}: {_evidence_sentence(result)}"
                for result in cited_results
            ]

        citations = _citations_from_results(cited_results)
        numbered_lines = [f"{line} [{index}]" for index, line in enumerate(bullet_lines, start=1)]
        answer = (
            f"There are multiple context-dependent meanings/usages for `{focus}` in the loaded standards.\n\n"
            "Here are the different contexts and what each standard says:\n\n"
            + "\n".join(numbered_lines)
        )
        unit_note = _unit_note(cited_results, unit_preference)
        if unit_note:
            answer += f"\n\nUnit note: {unit_note}"
        return ChatResponse(
            answer=answer,
            citations=citations,
            follow_up_suggestions=[
                "Ask me to focus on one of these standards only.",
                "Ask which context applies best for your specific test setup.",
            ],
        )

    def _contextual_query(
        self, question: str, conversation_id: str
    ) -> tuple[str, set[str] | None]:
        history = self._history.get(conversation_id, [])
        if not history or SPECIFIC_STANDARD_RE.search(question):
            return question, None

        if not _looks_like_follow_up(question):
            return question, None

        previous = history[-1]
        previous_document_ids = {citation.document_id for citation in previous.citations}
        previous_standards = " ".join(citation.standard_id for citation in previous.citations)
        return f"{previous.question}\nFollow-up: {question}\nPrior standards: {previous_standards}", (
            previous_document_ids or None
        )

    def _should_clarify(self, question: str) -> bool:
        content_terms = [
            term
            for term in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(term) > 2
            and term
            not in {
                "standard",
                "standards",
                "requirement",
                "requirements",
                "what",
                "which",
                "tell",
                "about",
            }
        ]
        return len(content_terms) == 0 and not SPECIFIC_STANDARD_RE.search(question)

    def _is_out_of_scope_query(self, question: str) -> bool:
        if SPECIFIC_STANDARD_RE.search(question):
            return False
        if _looks_like_follow_up(question):
            return False
        if (
            _is_applicability_question(question)
            or _is_compare_question(question)
            or _is_find_question(question)
            or _is_context_meaning_question(question, [])
        ):
            return False
        terms = {
            token.lower()
            for token in re.findall(r"[a-zA-Z]{3,}", question)
            if token.lower() not in {"what", "which", "when", "where", "how", "does", "mean"}
        }
        if not terms:
            return False
        return len(terms & DOMAIN_ANCHOR_TERMS) == 0

    def _remember(
        self,
        conversation_id: str,
        question: str,
        response: ChatResponse,
        *,
        user_id: str | None = None,
        unit_preference: str | None = None,
    ) -> ChatResponse:
        self._history.setdefault(conversation_id, []).append(
            _ConversationTurn(question=question, answer=response.answer, citations=response.citations)
        )
        if user_id and self.conversation_store and not response.needs_clarification:
            self.conversation_store.append_turn(
                user_id,
                conversation_id,
                question=question,
                answer=response.answer,
                citations=[citation.to_dict() for citation in response.citations],
                unit_preference=unit_preference,
                title_generator=self.title_generator,
            )
        return response

    def _attach_videos(self, question: str, response: ChatResponse) -> ChatResponse:
        """Cross-reference the question/answer with the video transcript store."""
        if self.video_store is None or len(self.video_store) == 0:
            return response
        if response.needs_clarification:
            return response

        from standards_rag.video import video_request_explicit

        explicit = video_request_explicit(question)
        # Lower the bar (and always surface the top hit) when a video is requested.
        min_score = 0.03 if explicit else 0.18
        top_k = 2 if explicit else 1
        matches = self.video_store.search(question, top_k=top_k, min_score=min_score)
        if not matches:
            return response

        from dataclasses import replace

        return replace(response, videos=[match.to_dict() for match in matches])

    def _hydrate_history_from_store(self, user_id: str, conversation_id: str) -> None:
        if conversation_id in self._history or not self.conversation_store:
            return
        record = self.conversation_store.get_conversation(user_id, conversation_id)
        if record is None:
            return
        turns: list[_ConversationTurn] = []
        pending_question: str | None = None
        for message in record.messages:
            if message.role == "user":
                pending_question = message.text
            elif message.role == "assistant" and pending_question:
                turns.append(
                    _ConversationTurn(
                        question=pending_question,
                        answer=message.text,
                        citations=[
                            Citation.from_dict(citation)
                            for citation in message.citations
                            if isinstance(citation, dict)
                        ],
                    )
                )
                pending_question = None
        if turns:
            self._history[conversation_id] = turns

    def _needs_partial_clarification(self, question: str, results: list[SearchResult]) -> bool:
        if not results or SPECIFIC_STANDARD_RE.search(question):
            return False
        if _is_applicability_question(question) or _is_compare_question(question):
            return False
        top_score = max(result.score for result in results)
        return top_score < PARTIAL_SUPPORT_SCORE_THRESHOLD

    def _partial_support_response(self, results: list[SearchResult]) -> ChatResponse:
        candidates = _best_by_document(results)
        lines = []
        suggestions = []
        for result in candidates[:4]:
            document = result.document
            lines.append(
                f"- {document.standard_id}: {document.title} "
                f"(best match score {result.score:.3f})"
            )
            suggestions.append(
                f"Ask specifically about {document.standard_id} and the exact test topic you mean."
            )
        answer = (
            "I found only partial support in the loaded standards for that question. "
            "Can you clarify which standard, material, or test topic you mean?\n\n"
            "Possible matches:\n"
            + "\n".join(lines)
        )
        return ChatResponse(
            answer=answer,
            citations=[],
            needs_clarification=True,
            follow_up_suggestions=suggestions[:4],
        )


def _citations_from_results(results: list[SearchResult] | tuple[SearchResult, ...] | object) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for result in results:  # type: ignore[union-attr]
        key = result.chunk.chunk_id
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_id=result.document.document_id,
                standard_id=result.document.standard_id,
                title=result.document.title,
                chunk_id=result.chunk.chunk_id,
                page_start=result.chunk.page_start,
                page_end=result.chunk.page_end,
                section=result.chunk.section,
                quote=_evidence_sentence(result, max_chars=240),
                pdf_url=_document_pdf_url(result.document),
            )
        )
    return citations


def _evidence_sentence(result: SearchResult, *, max_chars: int = 360) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text.replace("\n", " "))
    preferred_terms = {
        term
        for term in result.matched_terms
        if not re.match(r"^[a-z]?\d", term)
        and term
        not in {"standard", "method", "practice", "loaded"}
    }
    best = ""
    best_score = 0
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(1 for term in preferred_terms if term in lowered)
        if score > best_score:
            best = sentence.strip()
            best_score = score
    if not best:
        best = result.chunk.text.replace("\n", " ").strip()
    if len(best) > max_chars:
        return best[: max_chars - 3].rstrip() + "..."
    return best


def _top_scoring_sentence(text: str, keywords: tuple[str, ...], *, max_chars: int = 360) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    best = ""
    best_score = -1
    for raw in sentences:
        sentence = raw.strip()
        if not sentence:
            continue
        lowered = sentence.lower()
        score = sum(1 for term in keywords if term in lowered)
        if score > best_score:
            best_score = score
            best = sentence
    if not best:
        best = text.replace("\n", " ").strip()
    if len(best) > max_chars:
        return best[: max_chars - 3].rstrip() + "..."
    return best


def _faceted_sentence_for_design_limit(result: SearchResult, canonical_base: str) -> str:
    match canonical_base:
        case "D5321":
            keywords = (
                "normal stress",
                "displacement",
                "moisture",
                "drainage",
                "interchange",
                "laboratory",
                "field",
                "shear",
            )
        case "D5887":
            keywords = (
                "index",
                "flux",
                "design",
                "prescribed",
                "representative",
                "intended",
                "permeabil",
                "boundary",
            )
        case _:
            return _evidence_sentence(result)
    return _top_scoring_sentence(result.chunk.text, keywords)


def _faceted_sentence_for_exclusion_material(result: SearchResult, canonical_base: str) -> str:
    match canonical_base:
        case "D5321":
            keywords = ("exclude", "excluded", "gcl", "geosynthetic clay", "clay liner")
        case "D5887":
            keywords = (
                "exclude",
                "excluded",
                "geotextile",
                "backing",
                "geomembrane",
                "geofilm",
                "polymer",
                "applies only",
            )
        case _:
            return _evidence_sentence(result)
    return _top_scoring_sentence(result.chunk.text, keywords)


def _strip_trailing_sources_footer(text: str) -> str:
    """Remove a trailing 'Sources:' bibliography if the model still emits one."""
    for marker in ("\n\nSources:", "\r\n\r\nSources:"):
        if marker in text:
            return text.rsplit(marker, 1)[0].rstrip()
    return text


def _document_pdf_url(document: StandardDocument) -> str | None:
    if resolve_document_pdf_path(document) is None:
        return None
    return f"/documents/{quote(document.document_id, safe='')}/pdf"


def _chunk_section_type(result: SearchResult) -> str:
    return str(result.chunk.metadata.get("section_type", "other"))


def _best_by_document(results: list[SearchResult]) -> list[SearchResult]:
    by_document: dict[str, SearchResult] = {}
    neutral_route = _MethodFamilyRoute()
    for result in results:
        current = by_document.get(result.document.document_id)
        if current is None or _applicability_rank(
            "",
            result,
            route=neutral_route,
        ) > _applicability_rank(
            "",
            current,
            route=neutral_route,
        ):
            by_document[result.document.document_id] = result
    return list(by_document.values())


def _merge_search_results(
    primary: list[SearchResult], secondary: list[SearchResult]
) -> list[SearchResult]:
    merged = list(primary)
    seen = {result.chunk.chunk_id for result in primary}
    for result in secondary:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        merged.append(result)
    return merged


def _best_section_hit(results: list[SearchResult], section_type: str) -> SearchResult | None:
    candidates = [result for result in results if _section_matches(result, section_type)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.score)


def _fallback_section_hit(flat: list[SearchResult], wanted: str) -> SearchResult | None:
    """Prefer when chunking merges Scope and Significance so metadata cannot split them."""
    scored: list[tuple[float, SearchResult]] = []
    for result in flat:
        text = result.chunk.text
        text_lower = text.lower()
        if wanted == "scope":
            score = 0.0
            if "1." in text and "scope" in text_lower:
                score += 3.0
            if "this test method" in text_lower:
                score += 1.0
            for clause in ("1.1 ", "1.2 ", "1.3 "):
                if clause in text:
                    score += 1.5
        elif wanted == "significance":
            score = 0.0
            if "significance and use" in text_lower:
                score += 4.0
            if "5." in text:
                score += 2.0
            for needle in ("laboratory", "index flux", "not representative", "not intended"):
                if needle in text_lower:
                    score += 1.0
        else:
            score = 0.0
        if score > 0:
            scored.append((score + result.score, result))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _section_matches(result: SearchResult, section_type: str) -> bool:
    section = _chunk_section_type(result)
    heading = (result.chunk.heading or "").lower()
    text = result.chunk.text.lower()[:320]
    if section_type == "scope":
        return section in {"scope", "summary"} or "scope" in heading or "scope" in text
    if section_type == "significance":
        return (
            section == "significance"
            or "significance and use" in heading
            or "significance and use" in text
            or "significance" in heading
        )
    return section == section_type


def _supports_exclusion_claim(result: SearchResult) -> bool:
    text = result.chunk.text.lower()
    return any(
        marker in text
        for marker in (
            "does not apply",
            "do not apply",
            "excluded",
            "exclude",
            "limited to",
            "only applies",
            "not intended",
        )
    )


def _supports_design_limit_claim(result: SearchResult) -> bool:
    text = result.chunk.text.lower()
    return any(
        marker in text
        for marker in (
            "index",
            "not representative",
            "in-service",
            "field condition",
            "design",
            "test condition",
            "laboratory",
        )
    )


def _is_drainage_query(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in ("drainage", "transmissivity", "in-plane flow", "geonet", "geocomposite")
    )


_METHOD_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "interface_stability": (
        "interface stability",
        "interface shear",
        "shear",
        "sliding",
    ),
    "gcl_hydraulic_barrier": (
        "gcl hydraulic barrier",
        "hydraulic barrier",
        "barrier performance",
        "flux",
        "permeability",
    ),
    "drainage_transmissivity": (
        "drainage",
        "transmissivity",
        "in-plane flow",
    ),
    "stress_crack_durability": (
        "stress crack",
        "slow crack growth",
        "polyolefin geomembrane",
        "durability",
    ),
}

_METHOD_FAMILY_TO_STANDARDS: dict[str, tuple[str, ...]] = {
    "interface_stability": ("D5321",),
    "gcl_hydraulic_barrier": ("D5887",),
    "drainage_transmissivity": ("D4716",),
    "stress_crack_durability": ("D5397",),
}

_METHOD_FAMILY_LABELS: dict[str, str] = {
    "interface_stability": "interface stability / shear / sliding",
    "gcl_hydraulic_barrier": "GCL hydraulic barrier / flux / permeability",
    "drainage_transmissivity": "drainage / transmissivity / in-plane flow",
    "stress_crack_durability": "stress crack / slow crack growth durability",
}

# Per canonical designation prefix (for example ``D5321``, ``D5887``).
APPLICABILITY_FACET_QUERIES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "D5321": (
        (
            "material_exclusion",
            (
                "GCL geosynthetic clay liner excluded scope applicability exclusion",
                "exclude limitations applicability non applicability",
            ),
        ),
        (
            "design_limitation",
            (
                "normal stress displacement moisture drainage shear laboratory rate magnitude field design",
                "interface shear dependence test condition representative interchangeable",
            ),
        ),
    ),
    "D5887": (
        (
            "material_exclusion",
            (
                "geotextile backing geomembrane geofilm polymer coated excluded applicability scope limitation",
                "GCL backing types exclusion only applies excluded",
            ),
        ),
        (
            "design_limitation",
            (
                "index flux prescribed permeameter saturated laboratory field composite liner design flux",
                "index test not representative design hydraulic conductivity boundary conditions",
            ),
        ),
    ),
}

_FAMILY_ORDER = (
    "interface_stability",
    "gcl_hydraulic_barrier",
    "drainage_transmissivity",
    "stress_crack_durability",
)


def _ordered_primary_bases(route: _MethodFamilyRoute) -> list[str]:
    bases: list[str] = []
    seen: set[str] = set()
    for family in _FAMILY_ORDER:
        if family not in route.requested_families:
            continue
        for base in _METHOD_FAMILY_TO_STANDARDS.get(family, ()):
            if base not in seen:
                seen.add(base)
                bases.append(base)
    return bases


def _facet_already_satisfied(
    merged: list[SearchResult], doc_scope: set[str], canonical_base: str, facet_name: str
) -> bool:
    for r in merged:
        if r.document.document_id not in doc_scope:
            continue
        if _canonical_method_base(r.document.standard_id) != canonical_base:
            continue
        if facet_name == "material_exclusion" and _material_exclusion_evidence_hit(r, canonical_base):
            return True
        if facet_name == "design_limitation" and _design_limitation_evidence_hit(r, canonical_base):
            return True
    return False


_PROCEDURE_ARTIFACT_TERMS_RE = re.compile(
    r"\b(grip|gripping|clamp|clamping)\b",
    re.IGNORECASE,
)


def _material_exclusion_evidence_hit(result: SearchResult, canonical_base: str) -> bool:
    """True when the excerpt looks like explicit material applicability / exclusion wording."""
    if not _supports_exclusion_claim(result):
        return False
    text = result.chunk.text.lower()
    if _PROCEDURE_ARTIFACT_TERMS_RE.search(text) and not _material_anchor_hit(text):
        return False

    applicators_ph = ["limited to", "only applies", "only to", "applies only", "exclude", "excluded"]

    def _clause_ok(snippet: str) -> bool:
        return any(marker in snippet for marker in applicators_ph)

    match canonical_base:
        case "D5321":
            if not _clause_ok(text):
                return False
            return any(
                phrase in text
                for phrase in ("gcl", "geosynthetic clay", "clay liners", "clay liner", "liner (gcl)")
            )
        case "D5887":
            if not _clause_ok(text):
                return False
            geo_back = "geotextile" in text and ("back" in text or "backing" in text)
            return geo_back or any(
                k in text for k in ("geomembrane", "geofilm", "polymer-coated", "polymer coated", "thin film")
            )
        case _:
            return False


def _material_exclusion_strength(result: SearchResult, canonical_base: str) -> int:
    t = result.chunk.text.lower()
    score = 0
    for token in ("exclude", "excluded", "only applies", "only to", "applies only", "limited to"):
        if token in t:
            score += 2
    if canonical_base == "D5321":
        for ph in ("gcl", "geosynthetic clay", "clay liner", "clay liners"):
            if ph in t:
                score += 4
    elif canonical_base == "D5887":
        if "geomembrane" in t:
            score += 4
        if "geofilm" in t:
            score += 3
        if "geotextile" in t and ("back" in t or "backing" in t):
            score += 3
        if "polymer" in t:
            score += 2
    return score


def _material_anchor_hit(text: str) -> bool:
    return any(
        k in text
        for k in (
            "gcl",
            "geosynthetic clay",
            "geomembrane",
            "geotextile",
            "backing",
            "geofilm",
            "polymer",
            "clay liner",
        )
    )


def _design_limitation_evidence_hit(result: SearchResult, canonical_base: str) -> bool:
    """True when the excerpt states test/design transfer limits appropriate to the routed method."""
    text = result.chunk.text.lower()

    match canonical_base:
        case "D5321":
            knobs = ("normal stress", "displacement", "moisture", "drainage", "laboratory", "field")
            return sum(1 for k in knobs if k in text) >= 2
        case "D5887":
            has_index_flux = "index" in text and ("flux" in text or "permeab" in text)
            transfers = ("not representative" in text or "not intended" in text or "prescribed" in text)
            return (has_index_flux and transfers) or (
                ("index" in text or "flux" in text) and ("design" in text or "field" in text)
            )
        case _:
            return False


def _best_design_limit_result_for_base(
    results: list[SearchResult], canonical_base: str
) -> SearchResult | None:
    doc_hits = [
        r
        for r in results
        if _canonical_method_base(r.document.standard_id) == canonical_base
        and _design_limitation_evidence_hit(r, canonical_base)
    ]
    if not doc_hits:
        return None
    return max(doc_hits, key=lambda r: (_design_specificity_rank(r, canonical_base), r.score))


def _design_specificity_rank(result: SearchResult, canonical_base: str) -> int:
    """Prefer stronger design-transfer language over generic 'design' mentions."""
    text = result.chunk.text.lower()
    score = 0
    if canonical_base == "D5321":
        for term in (
            "normal stress",
            "displacement rate",
            "displacement magnitude",
            "moisture",
            "drainage",
            "interchange",
            "laboratory",
            "field",
        ):
            if term in text:
                score += 2
    elif canonical_base == "D5887":
        for term in ("index flux", "index test", "prescribed", "not representative", "design flux", "composite"):
            if term in text:
                score += 2
        if "index" in text:
            score += 1
    return score


_STANDARD_TO_METHOD_FAMILY: dict[str, str] = {
    standard: family
    for family, standards in _METHOD_FAMILY_TO_STANDARDS.items()
    for standard in standards
}


def _standard_base_id(standard_id: str) -> str:
    cleaned = standard_id.upper().replace("ASTM", "").strip()
    cleaned = cleaned.lstrip("-_ ")
    if cleaned and cleaned[0] == "-" and "-" in cleaned[1:]:
        cleaned = cleaned[1:]
    return cleaned.split("-", 1)[0]


def _canonical_method_base(standard_id: str) -> str:
    """Map designations such as ``D5321/D5321M-20`` onto the routed family prefix ``D5321``."""
    head = _standard_base_id(standard_id)
    head = head.lstrip("-_./ ")
    return head.split("/", 1)[0] if "/" in head else head


def _method_family_for_standard(standard_id: str) -> str | None:
    return _STANDARD_TO_METHOD_FAMILY.get(_canonical_method_base(standard_id))


def _parse_method_family_route(question: str) -> _MethodFamilyRoute:
    lowered = question.lower()
    requested: set[str] = set()
    for family, terms in _METHOD_FAMILY_KEYWORDS.items():
        if any(term in lowered for term in terms):
            requested.add(family)

    explicit_ids = {_canonical_method_base(match) for match in SPECIFIC_STANDARD_RE.findall(question)}
    for base in explicit_ids:
        family = _STANDARD_TO_METHOD_FAMILY.get(base)
        if family:
            requested.add(family)
    return _MethodFamilyRoute(requested_families=frozenset(requested))


def _document_ids_for_route(
    route: _MethodFamilyRoute, documents: dict[str, StandardDocument]
) -> set[str] | None:
    if not route.active:
        return None
    allowed = {
        std
        for family in route.requested_families
        for std in _METHOD_FAMILY_TO_STANDARDS.get(family, ())
    }
    doc_ids = {
        doc.document_id
        for doc in documents.values()
        if _canonical_method_base(doc.standard_id) in allowed
    }
    return doc_ids or None


def _document_ids_for_canonical_base(
    store: InMemoryStandardsStore, canonical_base: str
) -> set[str]:
    return {
        doc.document_id
        for doc in store.documents.values()
        if _canonical_method_base(doc.standard_id) == canonical_base
    }


def _flatten_grouped_for_canonical_base(
    grouped: dict[str, list[SearchResult]],
    store: InMemoryStandardsStore,
    canonical_base: str,
) -> list[SearchResult]:
    doc_ids = _document_ids_for_canonical_base(store, canonical_base)
    hits = [hit for doc_id, grp in grouped.items() for hit in grp if doc_id in doc_ids]
    return sorted(hits, key=lambda r: r.score, reverse=True)


def _family_for_canonical_base(canonical_base: str) -> str | None:
    for fam, bases in _METHOD_FAMILY_TO_STANDARDS.items():
        if canonical_base in bases:
            return fam
    return None


def _base_route_order(canonical_base: str, route: _MethodFamilyRoute) -> int:
    order = _ordered_primary_bases(route)
    return order.index(canonical_base) if canonical_base in order else 99


def _merge_document_filters(
    scoped_document_ids: set[str] | None, routed_document_ids: set[str] | None
) -> set[str] | None:
    if scoped_document_ids is None:
        return routed_document_ids
    if routed_document_ids is None:
        return scoped_document_ids
    merged = scoped_document_ids & routed_document_ids
    return merged or routed_document_ids


def _applicability_rank(question: str, result: SearchResult, *, route: _MethodFamilyRoute) -> float:
    """Rank suitability for applicability/exclusion/design-limit answers."""
    del question
    score = result.score
    section_type = _chunk_section_type(result)
    if section_type in {"scope", "summary", "significance", "terminology"}:
        score += 1.25
    if section_type in {"report", "calculation", "procedure", "precision"}:
        score -= 0.35

    family = _method_family_for_standard(result.document.standard_id)
    if route.active:
        if family in route.requested_families:
            score += 1.25
        elif family is not None:
            score -= 1.1
        else:
            score -= 0.55

    if _supports_exclusion_claim(result):
        score += 0.4
    if _supports_design_limit_claim(result):
        score += 0.35
    return score


def _is_applicability_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in (
            "which astm methods",
            "which method",
            "would apply",
            "applicable",
            "exclude",
            "not applicable",
            "interchangeable",
            "field design",
            "design limitation",
            "hydraulic barrier performance",
            "interface stability",
        )
    )


def _is_compare_question(question: str) -> bool:
    lowered = question.lower()
    return any(word in lowered for word in ["compare", "difference", "differences", "versus", " vs "])


def _is_find_question(question: str) -> bool:
    lowered = question.lower()
    return any(phrase in lowered for phrase in ["find", "which standard", "what standard", "relevant"])


def _is_context_meaning_question(question: str, results: list[SearchResult]) -> bool:
    lowered = question.lower()

    # Only trigger on explicit phrases that genuinely ask about definitions/contexts
    # across multiple standards. A single-letter variable alone is NOT enough.
    explicit_markers = [
        "in different contexts",
        "across standards",
        "different meanings",
        "multiple meanings",
        "how does it differ",
        "how do they differ",
    ]
    if any(marker in lowered for marker in explicit_markers):
        docs = {r.document.document_id for r in results[:6]}
        return len(docs) >= 2

    return False


def _looks_like_follow_up(question: str) -> bool:
    lowered = question.lower()
    phrase_markers = ("what about", "how about")
    if any(marker in lowered for marker in phrase_markers):
        return True

    word_markers = ("that", "those", "them", "it", "same", "their", "there")
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in word_markers)


def _focus_term(question: str) -> str:
    quoted = re.search(r"['\"]([^'\"]{1,40})['\"]", question)
    if quoted:
        return quoted.group(1)
    variable = re.search(
        r"\bvariable\s+([A-Za-z])(?:\b|[_^]|\d)",
        question,
        re.IGNORECASE,
    )
    if variable:
        return variable.group(1)
    of_var = re.search(r"\bof\s+([a-zA-Z])\b", question, re.IGNORECASE)
    if of_var:
        return of_var.group(1)
    single_letters = re.findall(r"\b([a-zA-Z])\b", question)
    for letter in single_letters:
        if letter.lower() not in {"s", "t", "m", "d", "a"}:
            return letter
    terms = [token for token in re.findall(r"[a-zA-Z]{3,}", question.lower())]
    return terms[0] if terms else "the term"


def _results_for_focus_term(results: list[SearchResult], focus: str) -> list[SearchResult]:
    lowered_focus = focus.lower()
    if lowered_focus == "the term":
        return results
    filtered = []
    for result in results:
        sentence = _evidence_sentence(result).lower()
        text = result.chunk.text.lower()
        if lowered_focus in sentence or lowered_focus in text:
            filtered.append(result)
    return filtered


def _question_sentence_overlap(question: str, sentence: str) -> int:
    stop = {"the", "for", "and", "with", "that", "this", "what", "when", "where", "there", "should"}
    q_terms = {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", question.lower())
        if token not in stop
    }
    s_terms = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
    return len(q_terms & s_terms)


def _unit_note(results: list[SearchResult], unit_preference: str | None) -> str | None:
    if not unit_preference:
        return None
    conversions: list[str] = []
    for result in results:
        for measurement in extract_measurements(result.chunk.text):
            converted = convert_measurement(measurement, unit_preference)
            if converted:
                conversions.append(format_conversion(measurement, converted))
        if conversions:
            break
    if not conversions:
        return None
    return "; ".join(conversions[:3]) + ". Conversions are derived from the cited values."
