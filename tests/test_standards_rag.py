from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.chat import StandardsRagEngine
from standards_rag.ingestion import load_document_from_text
from standards_rag.models import DocumentType
from standards_rag.retrieval import InMemoryStandardsStore


D7762_TEXT = """
ASTM Designation: D7762-18

Standard Practice for Design of Stabilization of Soil and Soil-Like Materials with
Self-Cementing Fly Ash

1. Scope
1.1 This practice covers procedures for the design of stabilization of soil and
soil-like materials using self-cementing coal fly ash for roadway applications,
treatment of expansive subgrade, or limiting settlement of fills below buildings.
The coal fly ash covered in this method includes self-cementing fly ashes described
in Specification D5239.

5. Mix Design
5.1 The design procedure evaluates fly ash content, moisture content, and compaction
conditions. Laboratory specimens are compacted to at least 95 % of maximum dry density.
The stabilized material is commonly prepared at a nominal maximum particle size of 25 mm.
"""


D698_TEXT = """
ASTM Designation: D698-12

Standard Test Methods for Laboratory Compaction Characteristics of Soil Using Standard Effort

1. Scope
1.1 These test methods cover laboratory compaction methods used to determine the
relationship between water content and dry unit weight of soils.

6. Procedure
6.1 Soil is compacted in a mold using standard effort. The result is a maximum dry
unit weight and optimum water content for the tested soil.
"""


class StandardsRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStandardsStore()
        self.d7762, d7762_chunks = load_document_from_text(
            D7762_TEXT,
            source_path="ASTM_D7762-18.txt",
            metadata_overrides={"title": "Standard Practice for Fly Ash Stabilization"},
        )
        self.d698, d698_chunks = load_document_from_text(
            D698_TEXT,
            source_path="ASTM_D698-12.txt",
            metadata_overrides={
                "title": "Standard Test Methods for Laboratory Compaction Characteristics"
            },
        )
        self.store.add_documents([(self.d7762, d7762_chunks), (self.d698, d698_chunks)])
        self.engine = StandardsRagEngine(self.store)

    def test_metadata_tracks_standard_identity_and_lifecycle(self) -> None:
        self.assertEqual(self.d7762.standard_id, "D7762-18")
        self.assertEqual(self.d7762.issuing_body, "ASTM")
        self.assertEqual(self.d7762.document_type, DocumentType.PRACTICE)
        self.assertEqual(self.d7762.year, 2018)
        self.assertEqual(self.d7762.review_due_year, 2023)
        self.assertIn("D5239", " ".join(self.d7762.references))

    def test_retrieval_finds_relevant_standards_with_citations(self) -> None:
        response = self.engine.ask("Which standards are relevant to fly ash stabilization?")

        self.assertFalse(response.unsupported)
        self.assertGreaterEqual(len(response.citations), 1)
        self.assertIn("D7762-18", response.answer)
        self.assertIn("Sources:", response.answer)
        self.assertTrue(response.citations[0].page_start)

    def test_direct_answer_is_grounded_in_retrieved_evidence(self) -> None:
        response = self.engine.ask("What does D7762-18 cover?")

        self.assertFalse(response.unsupported)
        self.assertIn("self-cementing coal fly ash", response.answer)
        self.assertEqual(response.citations[0].standard_id, "D7762-18")

    def test_comparison_spans_multiple_standards(self) -> None:
        response = self.engine.ask("Compare compaction requirements across the loaded standards")

        self.assertFalse(response.unsupported)
        cited_standards = {citation.standard_id for citation in response.citations}
        self.assertIn("D7762-18", cited_standards)
        self.assertIn("D698-12", cited_standards)
        self.assertIn("comparison", response.answer.lower())

    def test_follow_up_uses_conversation_context_and_converts_units(self) -> None:
        self.engine.ask("What does D7762-18 say about fly ash stabilization?", conversation_id="demo")
        response = self.engine.ask(
            "What units does it use?",
            conversation_id="demo",
            unit_preference="imperial",
        )

        self.assertFalse(response.unsupported)
        self.assertIn("Unit note:", response.answer)
        self.assertIn("approximately", response.answer)
        self.assertEqual(response.citations[0].standard_id, "D7762-18")

    def test_unsupported_question_refuses_without_citation(self) -> None:
        response = self.engine.ask("What tensile strength is required for carbon fiber panels?")

        self.assertTrue(response.unsupported)
        self.assertEqual(response.citations, [])
        self.assertIn("could not find support", response.answer)

    def test_index_round_trip_preserves_documents_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.store.save_json(path)
            loaded = InMemoryStandardsStore.load_json(path)

        self.assertEqual(set(loaded.documents), set(self.store.documents))
        self.assertEqual(set(loaded.chunks), set(self.store.chunks))
        response = StandardsRagEngine(loaded).ask("Find fly ash stabilization standards")
        self.assertFalse(response.unsupported)


if __name__ == "__main__":
    unittest.main()
