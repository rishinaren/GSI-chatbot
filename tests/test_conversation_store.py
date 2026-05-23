from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.auth import auth_public_config, load_auth_config_from_env
from standards_rag.conversation_store import InMemoryConversationStore
from standards_rag.ingestion import chunk_pages, load_document_from_text
from standards_rag.retrieval import InMemoryStandardsStore, rerank_results_for_citation


SAMPLE_TEXT = """
ASTM Designation: D4595-24

Standard Test Method for Wide Width

1. Scope
1.1 This test method covers tensile properties of geotextiles by the wide-width strip method.

2. Referenced Documents
2.1 ASTM D76 defines terminology used in this standard.

10. Procedure
10.1 Prepare specimens at the required width and measure force and elongation.
"""


class ConversationStoreTests(unittest.TestCase):
    def test_create_and_append_turn(self) -> None:
        store = InMemoryConversationStore()
        record = store.create_conversation("user-1", title="New chat")
        updated = store.append_turn(
            "user-1",
            record.conversation_id,
            question="What does D4595 cover?",
            answer="It covers wide-width tensile testing.",
            citations=[{"standard_id": "D4595-24"}],
        )
        self.assertEqual(len(updated.messages), 2)
        listed = store.list_conversations("user-1")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].title, "What does D4595 cover?")


class IngestionMetadataTests(unittest.TestCase):
    def test_chunks_include_page_metadata(self) -> None:
        document, chunks = load_document_from_text(
            SAMPLE_TEXT,
            source_path="D4595-24 Wide Width.txt",
        )
        self.assertGreaterEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.page_start, 1)
        self.assertIn("paragraph_index", chunk.metadata)
        self.assertIn("page_numbers", chunk.metadata)
        self.assertIn("section_type", chunk.metadata)
        self.assertEqual(chunk.metadata.get("section_type"), "scope")

    def test_page_boundary_flush(self) -> None:
        from standards_rag.ingestion import PageText, infer_document_metadata

        document = infer_document_metadata("ASTM D4595-24", source_path="D4595-24.txt")
        pages = [
            PageText(page_number=1, text="1. Scope\n1.1 Short page one paragraph."),
            PageText(page_number=2, text="10. Procedure\n10.1 Another page paragraph with more detail."),
        ]
        chunks = chunk_pages(document, pages, max_chars=80, overlap_chars=20)
        page_starts = {chunk.page_start for chunk in chunks}
        self.assertIn(1, page_starts)
        self.assertIn(2, page_starts)


class RetrievalRerankTests(unittest.TestCase):
    def test_rerank_prefers_single_page_span(self) -> None:
        store = InMemoryStandardsStore()
        document, chunks = load_document_from_text(SAMPLE_TEXT, source_path="D4595-24.txt")
        store.add_document(document, chunks)
        broad = store.search("wide width tensile geotextile procedure", top_k=6, min_score=0.0)
        reranked = rerank_results_for_citation("wide width tensile geotextile procedure", broad, top_k=2)
        self.assertGreaterEqual(len(reranked), 1)
        self.assertIsNotNone(reranked[0].chunk.page_start)


class AuthConfigTests(unittest.TestCase):
    def test_auth_public_config_defaults(self) -> None:
        config = load_auth_config_from_env()
        public = auth_public_config(config)
        self.assertIn("auth_required", public)
        self.assertIn("cognito_user_pool_id", public)


if __name__ == "__main__":
    unittest.main()
