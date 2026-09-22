from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.ingestion import load_document_from_text
from standards_rag.pinecone_hybrid import PineconeConfig, PineconeHybridStore
from standards_rag.retrieval import InMemoryStandardsStore


class _DenseOnlyIndex:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id

    def query(self, **_kwargs):
        return SimpleNamespace(matches=[SimpleNamespace(id=self.chunk_id, score=0.9)])


def test_lexical_candidates_are_fused_with_dense_candidates() -> None:
    irrelevant, irrelevant_chunks = load_document_from_text(
        "ASTM D9999-20\n1. Scope\nThis method concerns unrelated seam aging.",
        metadata_overrides={"title": "Unrelated Seam Aging"},
    )
    expected, expected_chunks = load_document_from_text(
        "ASTM D4751-21\n1. Scope\nThese methods determine apparent opening size of a geotextile.",
        metadata_overrides={"title": "Apparent Opening Size of a Geotextile"},
    )
    local = InMemoryStandardsStore()
    local.add_documents([(irrelevant, irrelevant_chunks), (expected, expected_chunks)])

    hybrid = PineconeHybridStore.__new__(PineconeHybridStore)
    InMemoryStandardsStore.__init__(hybrid)
    hybrid.documents = local.documents
    hybrid.chunks = local.chunks
    hybrid._reindex()
    hybrid.config = PineconeConfig("key", "index", None, "model", 64)
    hybrid._embed_query = lambda _query: [0.0]  # type: ignore[method-assign]
    hybrid._index = _DenseOnlyIndex(irrelevant_chunks[0].chunk_id)

    hits = hybrid.search("apparent opening size of a geotextile", top_k=1)

    assert hits[0].document.document_id == expected.document_id
