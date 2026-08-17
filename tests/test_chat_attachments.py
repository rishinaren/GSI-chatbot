from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # type: ignore[import-not-found]

from standards_rag.attachments import read_attachment
from standards_rag.chat import StandardsRagEngine
from standards_rag.ingestion import load_document_from_text
from standards_rag.retrieval import InMemoryStandardsStore

D4595_TEXT = """
ASTM Designation: D4595-24

Standard Test Method for Tensile Properties of Geotextiles by the Wide-Width Strip Method

1. Scope
1.1 This test method covers the measurement of tensile properties of geotextiles using a
wide-width strip specimen. It applies to woven and nonwoven geotextile fabrics and is an
index test of tensile strength and elongation.

5. Significance and Use
5.1 The wide-width strip tensile test provides tensile strength values used for comparison
of geotextile products and for design where index tensile strength is required.
"""

SPEC_LINES = [
    "PROJECT SPECIFICATION - SECTION 02 74 00",
    "Northfield Landfill Cell 4 - Geosynthetic Liner System",
    "3. GEOTEXTILE CUSHION REQUIREMENTS",
    "3.1 The cushion shall be a nonwoven needle-punched geotextile with a",
    "minimum mass per unit area of 270 g/m2.",
    "3.2 Wide-width tensile strength shall be determined in accordance with",
    "ASTM D4595 and shall be not less than 20 kN/m in both directions.",
    "3.3 Puncture resistance shall be reported but no minimum value applies.",
]

UNRELATED_LINES = [
    "SITE ACCESS AND PARKING PLAN",
    "Contractor vehicles shall enter by the east gate before 0700.",
    "Visitor parking is limited to the twelve marked bays by the site office.",
    "The haul road speed limit is fifteen miles per hour at all times.",
]


def _pdf_bytes(lines: list[str]) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 60
    for line in lines:
        page.insert_text((60, y), line, fontsize=10, fontname="helv")
        y += 16
    data = document.tobytes()
    document.close()
    return data


class AttachmentAnsweringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStandardsStore()
        document, chunks = load_document_from_text(D4595_TEXT, source_path="ASTM_D4595-24.txt")
        self.store.add_document(document, chunks)
        self.engine = StandardsRagEngine(self.store)
        self.spec = read_attachment(_pdf_bytes(SPEC_LINES), "spec.pdf", owner_id="user-1")

    def test_both_the_library_and_the_attachment_are_cited(self) -> None:
        response = self.engine.ask(
            "Does the geotextile tensile requirement in my spec match the standard?",
            attachments=[self.spec],
        )
        kinds = {citation.source_kind for citation in response.citations}
        self.assertIn("library", kinds)
        self.assertIn("attachment", kinds)

    def test_the_attachment_markers_continue_the_library_numbering(self) -> None:
        # Appended rather than prepended, so the library excerpts keep the
        # markers they were drafted with and nothing needs renumbering.
        response = self.engine.ask(
            "Does the geotextile tensile requirement in my spec match the standard?",
            attachments=[self.spec],
        )
        library = [i for i, c in enumerate(response.citations) if c.source_kind == "library"]
        attached = [i for i, c in enumerate(response.citations) if c.source_kind == "attachment"]
        self.assertTrue(max(library) < min(attached))

    def test_the_attached_document_is_named_in_the_draft(self) -> None:
        response = self.engine.ask(
            "What tensile strength does my spec require?", attachments=[self.spec]
        )
        self.assertIn("spec.pdf", response.answer)

    def test_a_question_the_library_cannot_answer_still_uses_the_attachment(self) -> None:
        # Nothing in this library is about parking. Without an attachment that is
        # a refusal; with one, the document is right there and should be read.
        parking = read_attachment(_pdf_bytes(UNRELATED_LINES), "access.pdf", owner_id="user-1")
        response = self.engine.ask("What is the haul road speed limit?", attachments=[parking])

        self.assertFalse(response.unsupported)
        self.assertTrue(response.citations)
        self.assertEqual({c.source_kind for c in response.citations}, {"attachment"})
        self.assertIn("only on the attached document", response.answer)

    def test_the_same_question_without_an_attachment_still_refuses(self) -> None:
        response = self.engine.ask("What is the haul road speed limit?")
        self.assertTrue(response.unsupported)
        self.assertEqual(response.citations, [])

    def test_a_question_too_thin_to_stand_alone_is_answered_when_a_file_is_attached(self) -> None:
        # "Summarise this" names no material, method or body, so on its own it is
        # turned away as unanchored. With a document on the message it is a
        # perfectly clear request, and the guards are bypassed.
        response = self.engine.ask("Summarise this", attachments=[self.spec])
        self.assertFalse(response.needs_clarification)
        self.assertFalse(response.unsupported)
        self.assertTrue(response.citations)

    def test_the_same_question_without_an_attachment_is_still_turned_away(self) -> None:
        response = self.engine.ask("Summarise this")
        self.assertTrue(response.unsupported or response.needs_clarification)
        self.assertEqual(response.citations, [])

    def test_the_standards_the_document_names_are_looked_up(self) -> None:
        # The question never says D4595 - the attached spec does. Retrieval
        # should still reach the standard.
        response = self.engine.ask(
            "Is the cushion requirement in my document reasonable?", attachments=[self.spec]
        )
        cited = {c.standard_id for c in response.citations if c.source_kind == "library"}
        self.assertIn("D4595-24", cited)


class RewriterHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStandardsStore()
        document, chunks = load_document_from_text(D4595_TEXT, source_path="ASTM_D4595-24.txt")
        self.store.add_document(document, chunks)
        self.spec = read_attachment(_pdf_bytes(SPEC_LINES), "spec.pdf", owner_id="user-1")
        self.seen: dict[str, object] = {}

        def rewriter(draft, question, citations, *, attachments=None):
            self.seen["attachments"] = attachments
            self.seen["citations"] = citations
            return "Rewritten answer. [1]"

        self.engine = StandardsRagEngine(self.store, answer_rewriter=rewriter)

    def test_the_writer_is_told_which_files_were_attached(self) -> None:
        self.engine.ask("Does my spec match the standard?", attachments=[self.spec])
        self.assertEqual(self.seen["attachments"], ["spec.pdf"])

    def test_no_attachment_means_an_empty_list_not_a_surprise(self) -> None:
        self.engine.ask("Which geotextile tensile strength standard applies?")
        self.assertEqual(self.seen["attachments"], [])


class TranscriptTests(unittest.TestCase):
    def test_the_file_name_is_saved_with_the_question_but_not_its_contents(self) -> None:
        from standards_rag.conversation_store import InMemoryConversationStore

        store = InMemoryStandardsStore()
        document, chunks = load_document_from_text(D4595_TEXT, source_path="ASTM_D4595-24.txt")
        store.add_document(document, chunks)
        conversations = InMemoryConversationStore()
        engine = StandardsRagEngine(store, conversation_store=conversations)
        spec = read_attachment(_pdf_bytes(SPEC_LINES), "spec.pdf", owner_id="user-1")

        engine.ask(
            "Does my spec match the standard?",
            conversation_id="c1",
            user_id="user-1",
            attachments=[spec],
        )

        record = conversations.get_conversation("user-1", "c1")
        question = record.messages[0]
        self.assertEqual(question.attachments, [
            {
                "attachment_id": spec.attachment_id,
                "file_name": "spec.pdf",
                "size_bytes": spec.size_bytes,
                "page_count": 1,
            }
        ])
        # The document text is deliberately absent - the transcript records what
        # was asked about, not a copy of the file.
        self.assertNotIn("needle-punched", str(question.attachments))


if __name__ == "__main__":
    unittest.main()
