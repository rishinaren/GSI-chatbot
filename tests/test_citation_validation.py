from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from standards_rag.citation_validation import citation_supports_claim, validate_answer_citations
from standards_rag.models import Citation, SourceChunk


class CitationValidationTests(unittest.TestCase):
    def test_supports_claim_when_overlap_present(self) -> None:
        cit = _fake_citation()
        chunk = SourceChunk(
            chunk_id=cit.chunk_id,
            document_id="astm-demo",
            text="Direct shear determines interface strength along geosynthetic boundaries under controlled shear.",
            page_start=1,
            page_end=1,
        )
        claim = "- D5321-26 determines shear resistance at geosynthetic interfaces [1]"
        self.assertTrue(citation_supports_claim(claim, cit, chunk))

    def test_rejects_claim_without_overlap(self) -> None:
        cit = _fake_citation()
        chunk = SourceChunk(
            chunk_id=cit.chunk_id,
            document_id="astm-demo",
            text="The precision study used ten laboratories measuring index flux variability.",
            page_start=6,
            page_end=7,
        )
        claim = "- D5887 measures wide-width tensile strength of knitted geogrids [1]"
        self.assertFalse(citation_supports_claim(claim, cit, chunk))

    def test_validate_strips_and_renumbers(self) -> None:
        c1 = _fake_citation(chunk_id="c1", standard_id="D5321-26")
        c2 = _fake_citation(chunk_id="c2", standard_id="D5887-23")
        chunks = {
            "c1": SourceChunk(
                chunk_id="c1",
                document_id="d1",
                text="covers direct shear testing interfaces between soils and geosynthetics",
                page_start=1,
                page_end=2,
            ),
            "c2": SourceChunk(
                chunk_id="c2",
                document_id="d2",
                text="mentions only precision statistics for logarithmic variability",
                page_start=12,
                page_end=13,
            ),
        }
        answer = (
            "- Applies to shear testing [1]\n"
            "- Claims cover geomembrane wide-width modulus [2]"
        )
        new_answer, cites = validate_answer_citations(answer, [c1, c2], chunks)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0].chunk_id, "c1")
        self.assertNotIn("[2]", new_answer)
        self.assertRegex(new_answer, r"\[1\]")


def _fake_citation(*, chunk_id: str = "cid", standard_id: str = "D5321-26") -> Citation:
    return Citation(
        document_id="astm-demo",
        standard_id=standard_id,
        title="Demo Test Method Title",
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        section="4",
        quote="Quoted preview",
        pdf_url=None,
    )


if __name__ == "__main__":
    unittest.main()
