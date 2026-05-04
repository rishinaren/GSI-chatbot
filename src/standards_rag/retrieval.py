"""Dependency-light retrieval store for standards RAG."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from standards_rag.models import SourceChunk, StandardDocument

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)?")
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
        if not query_terms:
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
            if score >= min_score:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        document=document,
                        score=round(score, 4),
                        matched_terms=matched,
                    )
                )

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

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
