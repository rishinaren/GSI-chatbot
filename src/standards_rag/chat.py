"""Citation-enforced chat orchestration for standards RAG."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from standards_rag.models import Citation
from standards_rag.retrieval import InMemoryStandardsStore, SearchResult
from standards_rag.units import convert_measurement, extract_measurements, format_conversion

SPECIFIC_STANDARD_RE = re.compile(
    r"\b(?:ASTM\s*)?[A-Z]\d{2,5}(?:/[A-Z]\d{2,5})?-\d{2,4}\b|"
    r"\bISO\s+\d+(?:[-:]\d+)*(?::\d{4})?\b|"
    r"\bBS\s+(?:EN\s+)?\d+(?:[-:]\d+)*(?::\d{4})?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    citations: list[Citation]
    retrieved_documents: list[str]
    unsupported: bool = False
    needs_clarification: bool = False
    follow_up_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieved_documents": self.retrieved_documents,
            "unsupported": self.unsupported,
            "needs_clarification": self.needs_clarification,
            "follow_up_suggestions": self.follow_up_suggestions,
        }


@dataclass
class _ConversationTurn:
    question: str
    answer: str
    citations: list[Citation]


class StandardsRagEngine:
    """RAG chat engine that refuses answers without retrieved evidence."""

    def __init__(
        self,
        store: InMemoryStandardsStore,
        *,
        top_k: int = 6,
        min_score: float = 0.01,
        answer_rewriter: Callable[[str, str, list[Citation]], str] | None = None,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.answer_rewriter = answer_rewriter
        self._history: dict[str, list[_ConversationTurn]] = {}

    def ask(
        self,
        question: str,
        *,
        conversation_id: str = "default",
        unit_preference: str | None = None,
    ) -> ChatResponse:
        clean_question = question.strip()
        if not clean_question:
            return ChatResponse(
                answer="Please ask a standards-related question.",
                citations=[],
                retrieved_documents=[],
                needs_clarification=True,
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
                    retrieved_documents=[],
                    needs_clarification=True,
                    follow_up_suggestions=[
                        "Which standards mention fly ash stabilization?",
                        "Compare compaction requirements across the loaded ASTM standards.",
                    ],
                ),
            )

        query, scoped_document_ids = self._contextual_query(clean_question, conversation_id)
        results = self.store.search(
            query,
            top_k=self.top_k,
            document_ids=scoped_document_ids,
            min_score=self.min_score,
        )
        if not results and scoped_document_ids:
            results = self.store.search(query, top_k=self.top_k, min_score=self.min_score)

        if not results:
            return self._remember(
                conversation_id,
                clean_question,
                ChatResponse(
                    answer=(
                        "I could not find support for that in the loaded standards collection. "
                        "I should not answer this from general knowledge without a cited source."
                    ),
                    citations=[],
                    retrieved_documents=[],
                    unsupported=True,
                ),
            )

        if _is_compare_question(clean_question):
            response = self._answer_comparison(clean_question, results, unit_preference)
        elif _is_context_meaning_question(clean_question, results):
            response = self._answer_context_meanings(clean_question, results, unit_preference)
        elif _is_find_question(clean_question):
            response = self._answer_find(clean_question, results)
        else:
            response = self._answer_direct(clean_question, results, unit_preference)

        response = self._maybe_rewrite_answer(clean_question, response)
        return self._remember(conversation_id, clean_question, response)

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

        return ChatResponse(
            answer=rewritten.strip(),
            citations=response.citations,
            retrieved_documents=response.retrieved_documents,
            unsupported=response.unsupported,
            needs_clarification=response.needs_clarification,
            follow_up_suggestions=response.follow_up_suggestions,
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

        answer += "\n\nSources:\n" + _format_sources(citations)
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved_documents=_document_labels(results),
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
        answer = (
            "I found these relevant loaded standards:\n\n"
            + "\n".join(lines)
            + "\n\nSources:\n"
            + _format_sources(citations)
        )
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved_documents=_document_labels(results),
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
        answer += "\n\nSources:\n" + _format_sources(citations)
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved_documents=_document_labels(results),
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
        answer += "\n\nSources:\n" + _format_sources(citations)
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved_documents=_document_labels(results),
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

    def _remember(
        self, conversation_id: str, question: str, response: ChatResponse
    ) -> ChatResponse:
        self._history.setdefault(conversation_id, []).append(
            _ConversationTurn(question=question, answer=response.answer, citations=response.citations)
        )
        return response


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


def _format_sources(citations: list[Citation]) -> str:
    return "\n".join(f"[{index}] {citation.format()}" for index, citation in enumerate(citations, 1))


def _document_labels(results: list[SearchResult]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for result in results:
        label = f"{result.document.standard_id} - {result.document.title}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


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
    return any(
        marker in lowered
        for marker in [
            "that",
            "those",
            "them",
            "it",
            "same",
            "what about",
            "how about",
            "their",
            "there",
        ]
    )


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
