from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.ingestion import load_document_from_text
from standards_rag.library import (
    DocumentLibrary,
    LibraryError,
    canonical_issuing_body,
    derive_document_id,
    detect_gri_identity,
    merge_documents_into_index,
)
from standards_rag.models import StandardDocument
from standards_rag.retrieval import InMemoryStandardsStore

GRI_COVER = """
GRI Test Method GM13r

Standard Specification for "Test Methods, Test Properties and Testing Frequency of
High Density Polyethylene (HDPE) Smooth and Textured Geomembranes"

This specification was developed by the Geosynthetic Research Institute (GRI).

1. Scope
1.1 This specification covers HDPE geomembranes.
"""

ASTM_TEXT = """
ASTM Designation: D4595-24

Standard Test Method for Tensile Properties of Geotextiles by the Wide-Width Strip Method

1. Scope
1.1 This test method covers the measurement of tensile properties of geotextiles
using a wide-width strip specimen.
"""


def _pdf_bytes(text: str) -> bytes:
    """Render text to a small real PDF so the ingest path is exercised end to end."""
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((54, 72), text, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


class DocumentIdentityTests(unittest.TestCase):
    def test_document_ids_match_the_existing_corpus_conventions(self) -> None:
        self.assertEqual(derive_document_id("ASTM", "D4595-24"), "astm-d4595-24")
        self.assertEqual(derive_document_id("ISO", "ISO 10319"), "iso-iso-10319")
        # GRI keeps its body out of the slug: GRI-GM13r -> gri-gm13r, not gri-gri-gm13r.
        self.assertEqual(derive_document_id("GRI", "GRI-GM13r"), "gri-gm13r")
        self.assertEqual(derive_document_id("GRI", "GCL7"), "gri-gcl7")

    def test_gri_is_detected_from_the_file_name_or_the_cover_page(self) -> None:
        self.assertEqual(detect_gri_identity("", "gm13r.pdf"), "GM13r")
        self.assertEqual(detect_gri_identity("", "GCL7.pdf"), "GCL7")
        self.assertEqual(detect_gri_identity(GRI_COVER, "scan-from-email.pdf"), "GM13r")
        self.assertIsNone(detect_gri_identity(ASTM_TEXT, "D4595-24 Wide Width.pdf"))

    def test_legacy_unknown_bodies_are_classified_for_retrieval(self) -> None:
        legacy = StandardDocument(
            document_id="unknown-gt1", standard_id="GT1", title="GT1", issuing_body="UNKNOWN"
        )
        self.assertEqual(canonical_issuing_body(legacy), "GRI")


class IndexMergeTests(unittest.TestCase):
    def test_merge_is_additive_and_retires_replaced_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "standards.json"
            first, first_chunks = load_document_from_text(ASTM_TEXT, source_path="a.txt")
            merge_documents_into_index(index_path, [first], first_chunks)

            second, second_chunks = load_document_from_text(GRI_COVER, source_path="b.txt")
            merge_documents_into_index(index_path, [second], second_chunks)

            data = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["documents"]), 2)

            # Re-ingesting the first document must not leave its old chunks behind.
            stale = [chunk.chunk_id for chunk in first_chunks]
            merge_documents_into_index(index_path, [first], [], remove_chunk_ids=stale)
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["documents"]), 2)
            self.assertEqual(
                [chunk["chunk_id"] for chunk in data["chunks"]],
                sorted(chunk.chunk_id for chunk in second_chunks),
            )


class DocumentLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.uploads = Path(self.tmp.name) / "documents" / "uploads"
        # Keep uploads and the index inside the temp dir, and S3 publishing off.
        patcher = mock.patch("standards_rag.library.project_root", return_value=Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict("os.environ", {"STANDARDS_ASSETS_S3_URI": ""}, clear=False)
        env.start()
        self.addCleanup(env.stop)

        self.store = InMemoryStandardsStore()
        self.library = DocumentLibrary(self.store, index_path=Path(self.tmp.name) / "index.json")

    def test_analyze_reads_the_pdf_without_changing_anything(self) -> None:
        preview = self.library.analyze(_pdf_bytes(GRI_COVER), "gm13r.pdf")
        self.assertEqual(preview["standard_id"], "GRI-GM13r")
        self.assertEqual(preview["issuing_body"], "GRI")
        self.assertIn("Geomembranes", preview["title"])
        self.assertGreater(preview["section_count"], 0)
        self.assertIsNone(preview["already_in_library"])
        self.assertEqual(self.store.documents, {})

    def test_add_makes_the_document_searchable_immediately(self) -> None:
        result = self.library.add(
            _pdf_bytes(ASTM_TEXT),
            "D4595-24.pdf",
            standard_id="D4595-24",
            title="Standard Test Method for Tensile Properties of Geotextiles",
            issuing_body="ASTM",
            added_by="admin@gsi.org",
        )
        self.assertEqual(result["document_id"], "astm-d4595-24")
        self.assertFalse(result["replaced"])
        self.assertIn("astm-d4595-24", self.store.documents)

        hits = self.store.search("wide-width strip tensile geotextiles", top_k=3)
        self.assertTrue(any(hit.document.document_id == "astm-d4595-24" for hit in hits))

        rows = self.library.documents()
        self.assertEqual(rows[0]["standard_id"], "D4595-24")
        self.assertEqual(rows[0]["added_by"], "admin@gsi.org")
        self.assertTrue(rows[0]["uploaded"])
        self.assertTrue((self.uploads / "astm-d4595-24.pdf").is_file())

    def test_re_uploading_replaces_rather_than_duplicates(self) -> None:
        pdf = _pdf_bytes(ASTM_TEXT)
        self.library.add(
            pdf, "d4595.pdf", standard_id="D4595-24", title="First title", issuing_body="ASTM"
        )
        first_chunk_ids = set(self.store.chunks)

        result = self.library.add(
            pdf, "d4595.pdf", standard_id="D4595-24", title="Corrected title", issuing_body="ASTM"
        )
        self.assertTrue(result["replaced"])
        self.assertEqual(len(self.store.documents), 1)
        self.assertEqual(self.store.documents["astm-d4595-24"].title, "Corrected title")
        # Same content re-chunks identically, so no orphans are left in the index.
        data = json.loads((Path(self.tmp.name) / "index.json").read_text(encoding="utf-8"))
        self.assertEqual({chunk["chunk_id"] for chunk in data["chunks"]}, first_chunk_ids)

    def test_a_gri_re_upload_is_recognised_as_a_replacement(self) -> None:
        """GRI ids drop the body prefix, so the generic slug rule misses them."""
        pdf = _pdf_bytes(GRI_COVER)
        self.library.add(
            pdf, "gm13r.pdf", standard_id="GRI-GM13r", title="First", issuing_body="GRI"
        )
        preview = self.library.analyze(pdf, "gm13r.pdf")
        self.assertIsNotNone(preview["already_in_library"])
        self.assertEqual(preview["already_in_library"]["standard_id"], "GRI-GM13r")

        result = self.library.add(
            pdf, "gm13r.pdf", standard_id="GRI-GM13r", title="Second", issuing_body="GRI"
        )
        self.assertTrue(result["replaced"])
        self.assertEqual(len(self.store.documents), 1)

    def test_a_standard_filed_under_a_legacy_id_is_replaced_not_duplicated(self) -> None:
        """GT1 lives in the corpus as ``unknown-gt1``; a re-upload must find it."""
        legacy, legacy_chunks = load_document_from_text(
            GRI_COVER, source_path="gt1.txt", metadata_overrides={
                "document_id": "unknown-gt1", "standard_id": "GT1", "issuing_body": "UNKNOWN"
            }
        )
        self.store.add_document(legacy, legacy_chunks)

        result = self.library.add(
            _pdf_bytes(GRI_COVER), "gt1.pdf", standard_id="GT1", title="GT1", issuing_body="GRI"
        )
        self.assertTrue(result["replaced"])
        self.assertEqual(result["document_id"], "unknown-gt1")
        self.assertEqual(len(self.store.documents), 1)

    def test_a_scan_with_no_selectable_text_is_refused_in_plain_language(self) -> None:
        blank = _pdf_bytes(" ")
        with self.assertRaises(LibraryError) as caught:
            self.library.analyze(blank, "scan.pdf")
        self.assertIn("readable text", str(caught.exception))

    def test_a_damaged_pdf_is_refused_instead_of_crashing(self) -> None:
        """Passes the %PDF header check, then fails to parse - must not 500."""
        with self.assertRaises(LibraryError) as caught:
            self.library.analyze(b"%PDF-1.4 truncated garbage" + bytes(300), "damaged.pdf")
        self.assertIn("could not open that PDF", str(caught.exception))

    def test_a_non_pdf_upload_is_refused(self) -> None:
        with self.assertRaises(LibraryError) as caught:
            self.library.analyze(b"PK\x03\x04 this is a zip", "standards.zip")
        self.assertIn("does not look like a PDF", str(caught.exception))

    def test_missing_details_are_refused_before_anything_is_written(self) -> None:
        with self.assertRaises(LibraryError):
            self.library.add(
                _pdf_bytes(ASTM_TEXT), "x.pdf", standard_id="", title="A title", issuing_body="ASTM"
            )
        self.assertEqual(self.store.documents, {})


if __name__ == "__main__":
    unittest.main()
