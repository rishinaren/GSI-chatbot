"""Dependency-light retrieval store for standards RAG."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from standards_rag.models import SourceChunk, StandardDocument

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)?")


def resolve_document_pdf_path(document: StandardDocument) -> Path | None:
    """Return an absolute path to the on-disk PDF for this document, if it is safe to serve.

    When ``STANDARDS_PDF_ROOT`` is set, the resolved path must lie under that directory.
    When unset (typical local dev), any existing ``.pdf`` at ``source_path`` is allowed.
    """
    if not document.source_path:
        return None
    path = Path(document.source_path).expanduser()
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    if path.suffix.lower() != ".pdf":
        return None
    resolved = path.resolve()
    root_raw = os.getenv("STANDARDS_PDF_ROOT")
    if root_raw:
        root = Path(root_raw).expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
    return resolved

# Headings / early body lines that usually carry applicability vs procedural content.
_SCOPE_SECTION_HINT = re.compile(
    r"\b(scope|summary\s+of\s+test|significance\s+and\s+use|terminology|definitions|"
    r"referenced\s+documents|significance|applicable|applicability)\b",
    re.IGNORECASE,
)
_PROC_SECTION_HINT = re.compile(
    r"\b(procedure|calculation|report|precision|bias|apparatus|specimen|summary\s+of\s+test\s+method)\b",
    re.IGNORECASE,
)


def section_intent_boost(query: str, chunk: SourceChunk) -> float:
    """Boost chunks whose heading/text match the query intent (scope vs procedure)."""
    q = query.lower()
    wants_scope = any(
        k in q
        for k in (
            "which",
            "apply",
            "applicable",
            "applicability",
            "exclude",
            "exclusion",
            "scope",
            "limitation",
            "design",
            "field",
            "interchange",
            "landfill",
            "liner",
            "when would",
            "appropriate",
            "methods would",
            "evaluate",
        )
    )
    wants_proc = any(
        k in q
        for k in (
            "procedure",
            "calculate",
            "calculation",
            "equation",
            "report",
            "precision",
            "specimen",
            "apparatus",
            "step",
        )
    )
    if not wants_scope and not wants_proc:
        return 0.0

    label = f"{chunk.heading or ''} {chunk.text[:400]}"
    section_type = str(chunk.metadata.get("section_type", "other"))
    boost = 0.0
    if wants_scope and _SCOPE_SECTION_HINT.search(label):
        boost += 0.32
    if wants_scope and section_type in {"scope", "summary", "significance", "terminology"}:
        boost += 0.5
    if wants_scope and section_type in {"report", "calculation", "procedure", "precision"}:
        boost -= 0.2
    if wants_proc and _PROC_SECTION_HINT.search(label):
        boost += 0.32
    if wants_proc and section_type in {"procedure", "calculation", "report", "precision"}:
        boost += 0.45
    return boost


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


_DESIGNATION_IN_QUERY = re.compile(
    r"\b(?:astm\s*)?[a-z]\d{2,5}(?:/[a-z]\d{2,5}[a-z]?)?(?:-\d{2,4}[a-z]?)?\b",
    re.IGNORECASE,
)


def _recall_substrings(query: str) -> list[str]:
    """Needles for substring recall when token-bag overlap is empty."""
    needles: list[str] = []
    for m in _DESIGNATION_IN_QUERY.finditer(query):
        raw = m.group(0).lower().replace("astm", "").strip()
        if raw:
            needles.append(raw)
            if "/" in raw:
                needles.append(raw.split("/", 1)[0])
            if "-" in raw:
                needles.append(raw.split("-", 1)[0])
    for t in TOKEN_RE.findall(query):
        tl = t.lower()
        if len(tl) >= 3 and tl not in STOP_WORDS:
            needles.append(_normalize_token(t))
    return list(dict.fromkeys(needles))[:28]


def _document_ids_matching_standard_mention(
    query: str, documents: dict[str, StandardDocument]
) -> set[str]:
    ql = query.lower()
    found: set[str] = set()
    for doc in documents.values():
        sid = doc.standard_id.lower().replace("astm", "").strip()
        if len(sid) >= 3 and sid in ql:
            found.add(doc.document_id)
            continue
        base = sid.split("-", 1)[0] if "-" in sid else sid
        if len(base) >= 3 and base in ql:
            found.add(doc.document_id)
    return found


@dataclass(frozen=True)
class SearchResult:
    chunk: SourceChunk
    document: StandardDocument
    score: float
    matched_terms: tuple[str, ...]


class InMemoryStandardsStore:
    """Small hybrid lexical store used for the MVP and tests.

    Production deployments can replace this class with pgvector/Pinecone while keeping the
    `search()` contract: return chunks, document metadata, scores, and matched terms.
    """

    def __init__(self) -> None:
        self.documents: dict[str, StandardDocument] = {}
        self.chunks: dict[str, SourceChunk] = {}
        self._chunk_tokens: dict[str, Counter[str]] = {}
        self._doc_frequency: Counter[str] = Counter()

    def add_document(self, document: StandardDocument, chunks: Iterable[SourceChunk]) -> None:
        self.documents[document.document_id] = document
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        self._reindex()

    def add_documents(self, items: Iterable[tuple[StandardDocument, Iterable[SourceChunk]]]) -> None:
        for document, chunks in items:
            self.documents[document.document_id] = document
            for chunk in chunks:
                self.chunks[chunk.chunk_id] = chunk
        self._reindex()

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        document_ids: set[str] | None = None,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        query_terms = _tokens(query)
        if not query_terms and query.strip():
            # Avoid returning nothing for short or stop-word-heavy questions when chunks exist.
            query_terms = [
                _normalize_token(t)
                for t in TOKEN_RE.findall(query)
                if len(t) >= 2 and t.lower() not in STOP_WORDS
            ][:16]
        if not query_terms:
            if not self.chunks or not query.strip():
                return []
            mentioned = _document_ids_matching_standard_mention(query, self.documents)
            if mentioned:
                return self._browse_fallback_chunks(mentioned, top_k)
            return []

        expanded_terms = _expand_terms(query_terms)
        query_counter = Counter(expanded_terms)
        results: list[SearchResult] = []

        for chunk_id, chunk_counter in self._chunk_tokens.items():
            chunk = self.chunks[chunk_id]
            if document_ids and chunk.document_id not in document_ids:
                continue
            document = self.documents[chunk.document_id]
            matched = tuple(sorted(set(query_counter) & set(chunk_counter)))
            if not matched:
                continue

            score = self._score(chunk_counter, query_counter, chunk.text)
            score += _metadata_boost(query, document, chunk)
            score += section_intent_boost(query, chunk)
            if score >= min_score:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        document=document,
                        score=round(score, 4),
                        matched_terms=matched,
                    )
                )

        if not results:
            results = self._substring_recall_search(query, document_ids, top_k, min_score)
        if not results:
            mentioned = _document_ids_matching_standard_mention(query, self.documents)
            if mentioned:
                results = self._browse_fallback_chunks(mentioned, top_k)

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def search_with_citation_rerank(
        self,
        query: str,
        *,
        top_k: int = 6,
        document_ids: set[str] | None = None,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        """Broad retrieval followed by page/sentence-level reranking for citation accuracy."""
        broad_k = max(top_k * 4, 16)
        broad = self.search(
            query,
            top_k=broad_k,
            document_ids=document_ids,
            min_score=max(min_score * 0.5, 0.003),
        )
        if not broad:
            return []
        return rerank_results_for_citation(query, broad, top_k=top_k)

    def _substring_recall_search(
        self,
        query: str,
        document_ids: set[str] | None,
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        needles = _recall_substrings(query)
        if not needles:
            return []
        results: list[SearchResult] = []
        for chunk_id, chunk in self.chunks.items():
            if document_ids and chunk.document_id not in document_ids:
                continue
            document = self.documents[chunk.document_id]
            hay = _chunk_index_text(chunk, document).lower()
            hits = [n for n in needles if n in hay]
            if not hits:
                continue
            score = 0.05 * len(hits) + _metadata_boost(query, document, chunk)
            score += section_intent_boost(query, chunk)
            if score < min_score:
                score = float(min_score) + 0.001
            matched = tuple(hits[:6])
            results.append(
                SearchResult(
                    chunk=chunk,
                    document=document,
                    score=round(score, 4),
                    matched_terms=matched,
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def _browse_fallback_chunks(
        self, document_ids: set[str] | None, top_k: int
    ) -> list[SearchResult]:
        """Last resort: return early chunks from cited documents (standard id appears in query)."""
        rows = sorted(self.chunks.values(), key=lambda c: (c.document_id, c.order))
        results: list[SearchResult] = []
        for chunk in rows:
            if document_ids and chunk.document_id not in document_ids:
                continue
            document = self.documents[chunk.document_id]
            mt = tuple(_tokens(document.standard_id + " " + document.title)[:4]) or ("standard",)
            results.append(
                SearchResult(
                    chunk=chunk,
                    document=document,
                    score=0.0001,
                    matched_terms=mt,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def find_documents(self, query: str, *, top_k: int = 5) -> list[StandardDocument]:
        seen: set[str] = set()
        documents: list[StandardDocument] = []
        for result in self.search(query, top_k=max(top_k * 3, 10), min_score=0.01):
            if result.document.document_id in seen:
                continue
            seen.add(result.document.document_id)
            documents.append(result.document)
            if len(documents) == top_k:
                break
        return documents

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [document.to_dict() for document in self.documents.values()],
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "InMemoryStandardsStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        documents = {
            item["document_id"]: StandardDocument.from_dict(item) for item in payload["documents"]
        }
        chunks_by_document: dict[str, list[SourceChunk]] = defaultdict(list)
        for item in payload["chunks"]:
            chunk = SourceChunk.from_dict(item)
            chunks_by_document[chunk.document_id].append(chunk)
        for document_id, document in documents.items():
            store.add_document(document, chunks_by_document.get(document_id, []))
        return store

    def _reindex(self) -> None:
        self._chunk_tokens = {
            chunk_id: Counter(_tokens(_chunk_index_text(chunk, self.documents[chunk.document_id])))
            for chunk_id, chunk in self.chunks.items()
        }
        self._doc_frequency = Counter()
        for counter in self._chunk_tokens.values():
            self._doc_frequency.update(counter.keys())

    def _score(self, chunk_counter: Counter[str], query_counter: Counter[str], text: str) -> float:
        chunk_total = max(sum(chunk_counter.values()), 1)
        chunk_count = max(len(self._chunk_tokens), 1)
        score = 0.0
        for term, query_weight in query_counter.items():
            term_count = chunk_counter.get(term, 0)
            if not term_count:
                continue
            term_frequency = term_count / chunk_total
            inverse_doc_frequency = math.log((chunk_count + 1) / (self._doc_frequency[term] + 0.5)) + 1
            score += query_weight * term_frequency * inverse_doc_frequency

        lower_text = text.lower()
        for phrase in _important_phrases(" ".join(query_counter)):
            if phrase in lower_text:
                score += 0.3
        return score


def _chunk_index_text(chunk: SourceChunk, document: StandardDocument) -> str:
    return " ".join(
        part
        for part in [
            document.standard_id,
            document.title,
            document.issuing_body,
            document.document_type.value,
            chunk.section or "",
            chunk.heading or "",
            chunk.text,
        ]
        if part
    )


def _tokens(value: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 and token.lower() not in STOP_WORDS
    ]


def _expand_terms(terms: list[str]) -> list[str]:
    expansions = {
        "unit": ["units", "measurement", "measurements"],
        "units": ["unit", "measurement", "measurements"],
        "compare": ["difference", "different", "differs"],
        "difference": ["compare", "different", "differs"],
        "stabilization": ["stabilize", "stabilized"],
        "fly": ["fly"],
        "ash": ["ash"],
    }
    expanded = list(terms)
    for term in terms:
        expanded.extend(expansions.get(term, []))
    return expanded


def _normalize_token(token: str) -> str:
    lowered = token.lower()
    if "-" in lowered or "/" in lowered:
        return lowered
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 4 and lowered.endswith("s"):
        return lowered[:-1]
    return lowered


def _metadata_boost(query: str, document: StandardDocument, chunk: SourceChunk) -> float:
    query_lower = query.lower()
    boost = 0.0
    if document.standard_id.lower() in query_lower:
        boost += 0.7
    if document.issuing_body.lower() in query_lower:
        boost += 0.1
    if chunk.section and f"section {chunk.section}".lower() in query_lower:
        boost += 0.4
    return boost


def _important_phrases(value: str) -> list[str]:
    words = [word for word in _tokens(value) if word not in STOP_WORDS]
    return [" ".join(words[index : index + 2]) for index in range(max(len(words) - 1, 0))]


def rerank_results_for_citation(
    query: str,
    results: list[SearchResult],
    *,
    top_k: int,
) -> list[SearchResult]:
    """Boost narrower page spans and sentence-level lexical overlap for citation use."""
    query_terms = set(_tokens(query))
    if not query_terms:
        return results[:top_k]

    reranked: list[tuple[float, SearchResult]] = []
    for result in results:
        score = result.score
        chunk = result.chunk
        if chunk.page_start is not None and chunk.page_end == chunk.page_start:
            score += 0.18
        elif chunk.page_start is not None and chunk.page_end is not None:
            span = chunk.page_end - chunk.page_start + 1
            if span <= 2:
                score += 0.08

        sentences = re.split(r"(?<=[.!?])\s+", chunk.text.replace("\n", " "))
        best_sentence_hits = 0
        for sentence in sentences:
            lowered = sentence.lower()
            hits = sum(1 for term in query_terms if term in lowered)
            best_sentence_hits = max(best_sentence_hits, hits)
        score += min(best_sentence_hits * 0.12, 0.45)

        if chunk.section:
            score += 0.05
        reranked.append((round(score, 4), result))

    reranked.sort(key=lambda item: item[0], reverse=True)
    output: list[SearchResult] = []
    seen_chunks: set[str] = set()
    for score, result in reranked:
        if result.chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(result.chunk.chunk_id)
        output.append(
            SearchResult(
                chunk=result.chunk,
                document=result.document,
                score=score,
                matched_terms=result.matched_terms,
            )
        )
        if len(output) >= top_k:
            break
    return output
