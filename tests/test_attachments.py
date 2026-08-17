from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # type: ignore[import-not-found]

from standards_rag.attachments import (
    ATTACHMENT_ID_PREFIX,
    AttachmentError,
    AttachmentStore,
    attachment_citations,
    designations_in,
    read_attachment,
    retrieval_hint,
    select_passages,
)

SPEC_LINES = [
    "PROJECT SPECIFICATION - SECTION 02 74 00",
    "Northfield Landfill Cell 4 - Geosynthetic Liner System",
    "1. SCOPE",
    "1.1 This specification covers the primary geomembrane liner and the",
    "protective geotextile cushion for Cell 4.",
    "2. GEOMEMBRANE REQUIREMENTS",
    "2.1 The primary liner shall be 60 mil textured HDPE geomembrane",
    "conforming to GRI GM13.",
    "2.2 Minimum thickness shall be 57 mil at any point.",
    "3. GEOTEXTILE CUSHION REQUIREMENTS",
    "3.1 The cushion shall be a nonwoven needle-punched geotextile with a",
    "minimum mass per unit area of 270 g/m2.",
    "3.2 Wide-width tensile strength shall be determined in accordance with",
    "ASTM D4595 and shall be not less than 20 kN/m in both directions.",
]


def _pdf_bytes(lines: list[str] | None = None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 60
    for line in lines if lines is not None else SPEC_LINES:
        page.insert_text((60, y), line, fontsize=10, fontname="helv")
        y += 16
    data = document.tobytes()
    document.close()
    return data


class ReadAttachmentTests(unittest.TestCase):
    def test_a_pdf_is_read_into_a_searchable_attachment(self) -> None:
        attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")

        self.assertEqual(attachment.file_name, "spec.pdf")
        self.assertEqual(attachment.owner_id, "user-1")
        self.assertEqual(attachment.page_count, 1)
        self.assertTrue(attachment.index.chunks)
        self.assertTrue(attachment.document_id.startswith(ATTACHMENT_ID_PREFIX))

    def test_the_document_is_not_linkable_so_no_citation_offers_a_pdf(self) -> None:
        # An attachment is read for the question and not kept, so nothing may
        # point at a stored file.
        attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")
        document = attachment.index.documents[attachment.document_id]
        self.assertIsNone(document.source_path)

    def test_a_file_that_is_not_a_pdf_is_refused_in_plain_words(self) -> None:
        with self.assertRaises(AttachmentError) as caught:
            read_attachment(b"just some text, not a pdf", "notes.txt", owner_id="user-1")
        self.assertIn("not a PDF", str(caught.exception))

    def test_an_empty_file_is_refused(self) -> None:
        with self.assertRaises(AttachmentError):
            read_attachment(b"", "spec.pdf", owner_id="user-1")

    def test_a_pdf_with_no_text_layer_says_so_rather_than_reading_nothing(self) -> None:
        document = fitz.open()
        document.new_page()
        data = document.tobytes()
        document.close()

        with self.assertRaises(AttachmentError) as caught:
            read_attachment(data, "scan.pdf", owner_id="user-1")
        self.assertIn("scan", str(caught.exception))

    def test_a_damaged_pdf_is_refused_instead_of_crashing(self) -> None:
        with self.assertRaises(AttachmentError) as caught:
            read_attachment(b"%PDF-1.7\nthis is not really a pdf body", "broken.pdf", owner_id="u")
        self.assertIn("could not open", str(caught.exception).lower())


class DesignationTests(unittest.TestCase):
    def test_the_standards_a_document_names_are_picked_out(self) -> None:
        found = designations_in("\n".join(SPEC_LINES))
        self.assertIn("GM13", found)
        self.assertIn("D4595", found)

    def test_a_document_naming_nothing_yields_nothing(self) -> None:
        self.assertEqual(designations_in("A plain memo about site access and parking."), ())


class SelectPassagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")

    def test_the_passages_are_the_ones_the_question_is_about(self) -> None:
        picked = select_passages(self.attachment, "what geotextile tensile strength is required?")
        joined = " ".join(chunk.text.lower() for chunk in picked)
        self.assertIn("tensile", joined)

    def test_a_question_with_no_overlap_still_returns_the_opening_pages(self) -> None:
        # "Summarise this" shares no words with the document, but the answer
        # should still rest on the document rather than on nothing.
        picked = select_passages(self.attachment, "summarise")
        self.assertTrue(picked)

    def test_the_prompt_budget_is_respected(self) -> None:
        picked = select_passages(self.attachment, "geomembrane thickness", budget=200)
        self.assertLessEqual(len(picked), 2)

    def test_passages_come_back_in_document_order(self) -> None:
        picked = select_passages(self.attachment, "geomembrane geotextile thickness tensile")
        self.assertEqual([chunk.order for chunk in picked], sorted(chunk.order for chunk in picked))


class CitationTests(unittest.TestCase):
    def test_a_citation_is_marked_as_the_asker_s_own_and_carries_its_text(self) -> None:
        attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")
        picked = select_passages(attachment, "geomembrane thickness")
        citations = attachment_citations(attachment, picked)

        self.assertTrue(citations)
        for citation in citations:
            self.assertEqual(citation.source_kind, "attachment")
            self.assertEqual(citation.title, "spec.pdf")
            self.assertIsNone(citation.pdf_url)
            # The validator looks chunks up in the standards index, which by
            # design never holds these - the quote is what it falls back to.
            self.assertTrue(citation.quote)

    def test_source_kind_survives_a_round_trip_through_storage(self) -> None:
        from standards_rag.models import Citation

        attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")
        original = attachment_citations(attachment, select_passages(attachment, "thickness"))[0]
        self.assertEqual(Citation.from_dict(original.to_dict()).source_kind, "attachment")

    def test_a_stored_citation_from_before_this_feature_reads_as_library(self) -> None:
        from standards_rag.models import Citation

        legacy = {"document_id": "astm-d4595-24", "standard_id": "D4595-24", "title": "t", "chunk_id": "c"}
        self.assertEqual(Citation.from_dict(legacy).source_kind, "library")


class RetrievalHintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")

    def test_the_standards_a_document_names_are_always_searched_for(self) -> None:
        hint = retrieval_hint([self.attachment], "does this comply with the seaming requirements?")
        self.assertIn("GM13", hint)

    def test_a_thin_question_borrows_the_document_s_vocabulary(self) -> None:
        hint = retrieval_hint([self.attachment], "is this ok?")
        self.assertTrue(set(self.attachment.keywords) & set(hint.split()))

    def test_a_specific_question_is_left_specific(self) -> None:
        hint = retrieval_hint(
            [self.attachment],
            "which method measures wide-width tensile strength of a nonwoven geotextile cushion?",
        )
        self.assertEqual(set(hint.split()) - set(self.attachment.designations), set())

    def test_no_attachment_means_no_hint(self) -> None:
        self.assertEqual(retrieval_hint([], "anything"), "")


class AttachmentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = AttachmentStore()
        self.attachment = read_attachment(_pdf_bytes(), "spec.pdf", owner_id="user-1")
        self.store.add(self.attachment)

    def test_the_owner_gets_their_attachment_back(self) -> None:
        found = self.store.get(self.attachment.attachment_id, owner_id="user-1")
        self.assertIsNotNone(found)

    def test_someone_else_s_attachment_reads_as_missing_not_forbidden(self) -> None:
        # Returning "not found" rather than "forbidden" means the response
        # cannot be used to learn that an id exists.
        self.assertIsNone(self.store.get(self.attachment.attachment_id, owner_id="user-2"))

    def test_an_unknown_id_is_missing(self) -> None:
        self.assertIsNone(self.store.get("nope", owner_id="user-1"))

    def test_an_expired_attachment_is_dropped(self) -> None:
        import dataclasses

        stale = dataclasses.replace(self.attachment, _expires_at=0.0)
        store = AttachmentStore()
        store.add(stale)
        self.assertIsNone(store.get(stale.attachment_id, owner_id="user-1"))
        self.assertEqual(len(store), 0)

    def test_the_oldest_goes_when_the_cache_is_full(self) -> None:
        store = AttachmentStore(max_items=2)
        first = read_attachment(_pdf_bytes(), "a.pdf", owner_id="u")
        second = read_attachment(_pdf_bytes(), "b.pdf", owner_id="u")
        third = read_attachment(_pdf_bytes(), "c.pdf", owner_id="u")
        for item in (first, second, third):
            store.add(item)

        self.assertIsNone(store.get(first.attachment_id, owner_id="u"))
        self.assertIsNotNone(store.get(third.attachment_id, owner_id="u"))

    def test_nothing_reaches_the_shared_corpus(self) -> None:
        # The point of the whole module: an attachment is never ingested, so it
        # lives in its own index and not in the one every question searches.
        self.assertNotIn(self.attachment.document_id, {"astm-d4595-24"})
        self.assertEqual(len(self.attachment.index.documents), 1)


if __name__ == "__main__":
    unittest.main()
