"""Pinecone vector retrieval layered on top of the local standards chunk store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from standards_rag.models import SourceChunk, StandardDocument
from standards_rag.retrieval import InMemoryStandardsStore, SearchResult, _tokens

_DEFAULT_EMBED_MODEL = "llama-text-embed-v2"


def normalize_index_host(host: str) -> str:
    cleaned = host.strip()
    if cleaned.startswith("https://"):
        cleaned = cleaned[len("https://") :]
    if cleaned.startswith("http://"):
        cleaned = cleaned[len("http://") :]
    return cleaned.rstrip("/")


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str
    index_name: str
    namespace: str | None
    embed_model: str
    batch_size: int


def load_pinecone_config_from_env() -> PineconeConfig:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX", "").strip()
    namespace_value = os.getenv("PINECONE_NAMESPACE", "").strip()
    embed_model = os.getenv("PINECONE_EMBED_MODEL", _DEFAULT_EMBED_MODEL).strip() or _DEFAULT_EMBED_MODEL
    batch_size_raw = os.getenv("PINECONE_UPSERT_BATCH", "64").strip()
    try:
        batch_size = max(int(batch_size_raw), 1)
    except ValueError:
        batch_size = 64

    namespace = namespace_value or None
    return PineconeConfig(
        api_key=api_key,
        index_name=index_name,
        namespace=namespace,
        embed_model=embed_model,
        batch_size=batch_size,
    )


def pinecone_enabled_from_env() -> bool:
    flag = os.getenv("USE_PINECONE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True

    config = load_pinecone_config_from_env()
    return bool(config.api_key and config.index_name)


def attach_pinecone_index(store: InMemoryStandardsStore) -> PineconeHybridStore:
    """Wrap an existing in-memory store with Pinecone upsert/query behavior."""

    config = load_pinecone_config_from_env()
    if not config.api_key or not config.index_name:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_INDEX to use Pinecone retrieval.")

    hybrid = PineconeHybridStore.__new__(PineconeHybridStore)
    InMemoryStandardsStore.__init__(hybrid)

    hybrid.config = config
    hybrid._pc = _require_pinecone()
    hybrid._client = hybrid._pc(api_key=config.api_key)
    hybrid._index = hybrid._connect_index(hybrid._client, config.index_name)

    hybrid.documents = store.documents
    hybrid.chunks = store.chunks
    hybrid._reindex()

    return hybrid


class PineconeHybridStore(InMemoryStandardsStore):
    """Semantic retrieval via Pinecone, with lexical reranking for citations."""

    def __init__(self, config: PineconeConfig) -> None:
        super().__init__()
        self.config = config
        self._pc = _require_pinecone()
        self._client = self._pc(api_key=config.api_key)
        self._index = self._connect_index(self._client, config.index_name)

    def upsert_embeddings(self, chunks: Iterable[SourceChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return

        embed = self._embed_passages([chunk.text for chunk in chunk_list])
        vectors: list[dict[str, object]] = []
        for chunk, values in zip(chunk_list, embed, strict=True):
            document = self.documents[chunk.document_id]
            metadata = {
                "document_id": document.document_id,
                "standard_id": document.standard_id,
                "chunk_id": chunk.chunk_id,
                "text_preview": chunk.text[:800],
            }
            vectors.append({"id": chunk.chunk_id, "values": values, "metadata": metadata})

        for start in range(0, len(vectors), self.config.batch_size):
            batch = vectors[start : start + self.config.batch_size]
            self._index.upsert(vectors=batch, namespace=self.config.namespace)

    def add_document(self, document: StandardDocument, chunks: Iterable[SourceChunk]) -> None:
        chunk_list = list(chunks)
        super().add_document(document, chunk_list)
        self.upsert_embeddings(chunk_list)

    def add_documents(self, items: Iterable[tuple[StandardDocument, Iterable[SourceChunk]]]) -> None:
        flattened: list[tuple[StandardDocument, list[SourceChunk]]] = []
        for document, chunks in items:
            chunk_list = list(chunks)
            flattened.append((document, chunk_list))

        for document, chunk_list in flattened:
            super().add_document(document, chunk_list)

        for _, chunk_list in flattened:
            self.upsert_embeddings(chunk_list)

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        document_ids: set[str] | None = None,
        min_score: float = 0.01,
    ) -> list[SearchResult]:
        query_terms = _tokens(query)
        if not query_terms:
            return []

        query_vector = self._embed_query(query)
        pinecone_filter = None
        if document_ids:
            if len(document_ids) == 1:
                document_id = next(iter(document_ids))
                pinecone_filter = {"document_id": {"$eq": document_id}}
            else:
                pinecone_filter = {"document_id": {"$in": sorted(document_ids)}}

        query_response = self._index.query(
            vector=query_vector,
            top_k=max(top_k * 8, top_k),
            include_metadata=True,
            namespace=self.config.namespace,
            filter=pinecone_filter,
        )

        matches = getattr(query_response, "matches", None) or []
        fused_scores: list[tuple[float, str]] = []
        for match in matches:
            chunk_id = str(match.id)
            pinecone_score = float(match.score or 0.0)
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            document = self.documents[chunk.document_id]
            lexical_score = _lexical_overlap_score(query_terms, chunk, document)
            fused = 0.65 * pinecone_score + 0.35 * lexical_score
            if fused >= min_score:
                fused_scores.append((fused, chunk_id))

        fused_scores.sort(key=lambda item: item[0], reverse=True)
        if not fused_scores:
            return super().search(
                query,
                top_k=top_k,
                document_ids=document_ids,
                min_score=max(min_score, 0.05),
            )

        results: list[SearchResult] = []
        for score, chunk_id in fused_scores[:top_k]:
            chunk = self.chunks[chunk_id]
            document = self.documents[chunk.document_id]
            matched_terms = tuple(sorted(set(query_terms) & set(_tokens(chunk.text))))
            results.append(
                SearchResult(
                    chunk=chunk,
                    document=document,
                    score=round(score, 4),
                    matched_terms=matched_terms,
                )
            )
        return results

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        response = self._client.inference.embed(
            model=self.config.embed_model,
            inputs=texts,
            parameters=_embed_parameters("passage"),
        )
        return _extract_embedding_vectors(response)

    def _embed_query(self, text: str) -> list[float]:
        response = self._client.inference.embed(
            model=self.config.embed_model,
            inputs=[text],
            parameters=_embed_parameters("query"),
        )
        vectors = _extract_embedding_vectors(response)
        if not vectors:
            raise RuntimeError("Pinecone embed returned no vectors for the query.")
        return vectors[0]

    @staticmethod
    def _connect_index(client: object, index_name: str):
        index_host = os.getenv("PINECONE_INDEX_HOST", "").strip()
        if index_host:
            return client.Index(host=normalize_index_host(index_host))

        describe = client.describe_index(name=index_name)
        host = getattr(describe, "host", None)
        if not host:
            raise RuntimeError("Could not resolve Pinecone index host. Set PINECONE_INDEX_HOST.")
        return client.Index(host=normalize_index_host(str(host)))


def _require_pinecone():
    try:
        from pinecone import Pinecone  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the optional 'pinecone' dependencies to use Pinecone retrieval.") from exc
    return Pinecone


def _extract_embedding_vectors(response: object) -> list[list[float]]:
    if hasattr(response, "embeddings"):
        embeddings = getattr(response, "embeddings")
        if embeddings and hasattr(embeddings[0], "values"):
            return [list(item.values) for item in embeddings]

    data = getattr(response, "data", None)
    if data:
        vectors = []
        for item in data:
            values = getattr(item, "values", None)
            if values is not None:
                vectors.append(list(values))
        if vectors:
            return vectors

    if isinstance(response, dict):
        vectors: list[list[float]] = []
        for item in response.get("data", []):
            values = item.get("values")
            if values:
                vectors.append(list(values))
        if vectors:
            return vectors

    raise RuntimeError("Unexpected Pinecone embed response format.")


def _embed_parameters(input_type: str) -> dict[str, str]:
    truncate = os.getenv("PINECONE_EMBED_TRUNCATE", "END").strip() or "END"
    return {"input_type": input_type, "truncate": truncate}


def _lexical_overlap_score(
    query_terms: list[str], chunk: SourceChunk, document: StandardDocument
) -> float:
    chunk_terms = set(_tokens(chunk.text))
    doc_terms = set(
        _tokens(" ".join([document.standard_id, document.title, document.issuing_body]))
    )
    combined = chunk_terms | doc_terms
    if not query_terms:
        return 0.0
    overlap = len(set(query_terms) & combined)
    return overlap / len(set(query_terms))
