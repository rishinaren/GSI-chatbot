from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.answer_prompts import build_rewriter_system_prompt
from standards_rag.chat import StandardsRagEngine
from standards_rag.ingestion import infer_document_metadata, load_document_from_text
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


D5321_TEXT = """
ASTM Designation: D5321/D5321M-20

Standard Test Method for Determining the Shear Strength of Soil-Geosynthetic and
Geosynthetic-Geosynthetic Interfaces by Direct Shear

1. Scope
1.1 This test method covers determining interface shear strength and sliding resistance
between geosynthetics and soils under controlled normal stress.
1.2 Geosynthetic clay liners (GCLs) are excluded from this test method.

5. Significance and Use
5.1 The measured interface shear values are used to assess interface stability for
liner and cover systems and are dependent on laboratory test conditions.
5.2 Shear resistance depends on applied normal stress, material types, adjacent soil properties,
moisture conditions, drainage, displacement rate, and cumulative displacement magnitude, and therefore
these values shall not be taken as interchangeable with field-derived design values without
confirming analogous conditions.
"""


D5887_TEXT = """
ASTM Designation: D5887/D5887M-23

Standard Test Method for Measurement of Index Flux Through Saturated Geosynthetic
Clay Liner Specimens Using a Flexible Wall Permeameter

1. Scope
1.1 This test method measures index flux and hydraulic conductivity through saturated
GCL specimens for evaluating hydraulic barrier behavior.
1.2 This method applies only to geosynthetic clay liner products with geotextile backing(s); GCL types
with an extruded geomembrane, thin geofilm, or polymer-coated facing are excluded.

5. Significance and Use
5.1 This is an index test and the values are not directly representative of installed
field composite liner performance under site-specific conditions.
5.2 Results are an index flux measured under prescribed laboratory boundary conditions and are not intended
to be used directly as design flux without site-specific validation.
"""


D5397_TEXT = """
ASTM Designation: D5397/D5397M-22

Standard Test Method for Evaluation of Stress Crack Resistance of Polyolefin Geomembranes

1. Scope
1.1 This test method covers slow crack growth evaluation of polyolefin geomembranes under
controlled constant tensile loading in a surfactant solution.

5. Significance and Use
5.1 The result is used to rank stress-crack durability of geomembrane materials and is unrelated
to interface shear stability or hydraulic flux through GCL specimens.
"""


D4716_TEXT = """
ASTM Designation: D4716-23

Standard Test Method for Determining the (In-Plane) Flow Rate and Hydraulic
Transmissivity of a Geosynthetic Using a Constant Head

1. Scope
1.1 This test method covers in-plane flow rate and transmissivity measurements for
geosynthetic drainage materials used in liner and cover drainage layers.

5. Significance and Use
5.1 The method is intended for drainage design and does not measure interface shear
stability or GCL hydraulic barrier flux.
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

        self.d5321, d5321_chunks = load_document_from_text(
            D5321_TEXT,
            source_path="ASTM_D5321-20.txt",
        )
        self.d5887, d5887_chunks = load_document_from_text(
            D5887_TEXT,
            source_path="ASTM_D5887-23.txt",
        )
        self.d4716, d4716_chunks = load_document_from_text(
            D4716_TEXT,
            source_path="ASTM_D4716-23.txt",
        )
        self.d5397, d5397_chunks = load_document_from_text(
            D5397_TEXT,
            source_path="ASTM_D5397-22.txt",
        )
        self.store.add_documents(
            [
                (self.d5321, d5321_chunks),
                (self.d5887, d5887_chunks),
                (self.d4716, d4716_chunks),
                (self.d5397, d5397_chunks),
            ]
        )
        self.engine = StandardsRagEngine(self.store)

    def test_standard_id_prefers_filename_over_referenced_designations_in_body(self) -> None:
        """Body text can cite D35-1004 before the cover designation; filename is authoritative."""
        text = (
            "See also ASTM D35-1004 for related procedures.\n\n"
            "Designation: D5887/D5887M – 23\n"
            "Standard Test Method for Measurement of Index Flux Through Saturated "
            "Geosynthetic Clay Liner Specimens Using a Flexible Wall Permeameter\n"
        )
        doc = infer_document_metadata(
            text,
            source_path="/documents/Select ASTM methods/D5887-23 GCL Perm.pdf",
        )
        self.assertEqual(doc.standard_id, "D5887-23")
        self.assertIn("d5887-23", doc.document_id.lower())

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
        self.assertNotIn("Sources:", response.answer)
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

    def test_chunks_include_section_type_metadata(self) -> None:
        chunk = next(iter(self.store.chunks.values()))
        self.assertIn("section_type", chunk.metadata)
        self.assertIn("section_number", chunk.metadata)
        self.assertIn("section_title", chunk.metadata)

    def test_applicability_questions_use_structured_facet_answer(self) -> None:
        response = self.engine.ask(
            "Which methods would apply, what is excluded, and why are values not interchangeable "
            "with field design values?"
        )

        self.assertFalse(response.unsupported)
        self.assertIn("Applicable method(s):", response.answer)
        self.assertIn("Exclusions / non-applicability:", response.answer)
        self.assertIn("Why values may not be directly interchangeable", response.answer)
        self.assertGreaterEqual(len(response.citations), 1)

    def test_method_family_router_prefers_matching_standards_and_flags_mismatch(self) -> None:
        response = self.engine.ask(
            "A landfill liner design team needs interface stability and GCL hydraulic barrier "
            "performance. Which ASTM methods apply and why are values not directly "
            "interchangeable with field design values?"
        )

        self.assertFalse(response.unsupported)
        self.assertIn("D5321", response.answer)
        self.assertIn("D5887", response.answer)
        self.assertIn("Related but not primary", response.answer)
        self.assertIn("D4716", response.answer)
        self.assertIn("Not addressing this question", response.answer)
        self.assertIn("D5397", response.answer)
        self.assertIn("Exclusions / non-applicability (from applicable methods only)", response.answer)
        self.assertRegex(response.answer.lower(), r"gcl|geosynthetic clay")
        self.assertRegex(response.answer.lower(), r"geotextile|geomembrane|geofilm|polymer")
        self.assertRegex(response.answer.lower(), r"normal stress|displacement")
        self.assertRegex(response.answer.lower(), r"index.*flux|flux.*index")
        lower = response.answer.lower()
        excl_start = lower.find("exclusions / non-applicability")
        self.assertGreater(excl_start, -1)
        excl_seg = lower[excl_start : excl_start + 800]
        self.assertNotIn("d5397", excl_seg)
        cited_standards = {citation.standard_id for citation in response.citations}
        self.assertTrue(any(standard.endswith("D5321-20") for standard in cited_standards))
        self.assertTrue(any(standard.endswith("D5887-23") for standard in cited_standards))

    def test_rewriter_prompt_forbids_rewrite_stage_disclosure(self) -> None:
        prompt = build_rewriter_system_prompt(include_comparison_schema=True)
        self.assertIn("Never mention the rewrite process", prompt)
        # The rewriter should explain the evidence rather than quote it verbatim.
        self.assertIn("verbatim", prompt.lower())
        self.assertIn("explain", prompt.lower())

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
        response = self.engine.ask("What curing schedule is required for carbon fiber prepreg panels?")

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
