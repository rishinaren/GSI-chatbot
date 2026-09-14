from __future__ import annotations

import unittest

from standards_rag.chat import StandardsRagEngine
from standards_rag.design_guidance import (
    DESIGN_GUIDANCE_CORPUS,
    chunk_design_guidance_pages,
    design_book_volume,
    is_design_guidance_question,
)
from standards_rag.ingestion import PageText, infer_document_metadata, load_document_from_text
from standards_rag.models import DocumentType
from standards_rag.retrieval import InMemoryStandardsStore


def _design_document():
    return infer_document_metadata(
        "Designing with Geosynthetics",
        source_path="documents/Design guidance/DWG 6th Edition-Vol.2.pdf",
        overrides={
            "document_id": "designing-with-geosynthetics-6e-volume-2",
            "standard_id": "DWG-6E-V2",
            "title": "Designing with Geosynthetics, 6th Edition, Volume 2",
            "issuing_body": "GSI",
            "document_type": DocumentType.BOOK,
            "year": 2012,
            "metadata": {
                "corpus_kind": DESIGN_GUIDANCE_CORPUS,
                "authority": "secondary_guidance",
                "edition": 6,
                "volume": 2,
            },
        },
    )


class DesignGuidanceIngestionTests(unittest.TestCase):
    def test_volume_is_identified_from_book_file_name(self) -> None:
        self.assertEqual(design_book_volume("DWG 6th Edition-Vol.1.pdf"), 1)
        self.assertEqual(design_book_volume("Designing_with_Geosynthetics_Volume_2.pdf"), 2)

    def test_chunks_keep_printed_page_and_section_metadata(self) -> None:
        document = _design_document()
        pages = [
            PageText(
                page_number=566,
                text=(
                    "5.3 LIQUID CONTAINMENT (POND) LINERS\n\n"
                    "Pond liner design starts with geometry and liquid containment needs.\n\n"
                    + "Material selection and site conditions must be evaluated. " * 80
                ),
            )
        ]
        chunks = chunk_design_guidance_pages(document, pages, volume=2, max_chars=400)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.page_start == 566 for chunk in chunks))
        self.assertTrue(all(chunk.page_end == 566 for chunk in chunks))
        self.assertTrue(all(chunk.section == "5.3" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["corpus_kind"] == DESIGN_GUIDANCE_CORPUS for chunk in chunks))
        self.assertTrue(all(chunk.metadata["volume"] == 2 for chunk in chunks))


class DesignGuidanceRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.standards = InMemoryStandardsStore()
        standard, standard_chunks = load_document_from_text(
            """
            ASTM Designation: D5199-25
            Standard Test Methods for Measuring Nominal Thickness of Geosynthetics
            1. Scope
            1.1 This method measures nominal geomembrane thickness under specified pressure.
            """,
            source_path="D5199-25.txt",
        )
        self.standards.add_document(standard, standard_chunks)

        self.guidance = InMemoryStandardsStore()
        guidance_document = _design_document()
        guidance_chunks = chunk_design_guidance_pages(
            guidance_document,
            [
                PageText(
                    page_number=568,
                    text=(
                        "5.3.1 Geometric Considerations\n\n"
                        "A pond liner design begins by relating liquid volume, available land area, "
                        "depth, and side slope geometry before selecting the geomembrane."
                    ),
                )
            ],
            volume=2,
        )
        self.guidance.add_document(guidance_document, guidance_chunks)
        self.engine = StandardsRagEngine(self.standards, design_store=self.guidance)

    def test_explicit_book_question_routes_to_guidance_in_testing_focus(self) -> None:
        response = self.engine.ask(
            "According to Designing with Geosynthetics, how do I begin a pond liner design?",
            retrieval_focus="testing",
        )

        self.assertTrue(response.citations)
        self.assertTrue(all(citation.source_kind == "design_guidance" for citation in response.citations))
        self.assertIn("Design guidance", response.answer)

    def test_both_focus_can_return_standard_and_design_sources(self) -> None:
        response = self.engine.ask(
            "How should I design a geomembrane pond liner and evaluate thickness?",
            retrieval_focus="both",
        )

        source_kinds = {citation.source_kind for citation in response.citations}
        self.assertIn("library", source_kinds)
        self.assertIn("design_guidance", source_kinds)

    def test_focus_is_a_default_not_a_wall(self) -> None:
        self.assertTrue(is_design_guidance_question("What does Koerner say about GCL slopes?", "testing"))
        self.assertFalse(is_design_guidance_question("What does ASTM D5199 require?", "design"))
        self.assertTrue(is_design_guidance_question("How do I design an MSE wall?", "design"))


if __name__ == "__main__":
    unittest.main()
