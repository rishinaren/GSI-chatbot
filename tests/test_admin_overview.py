from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.admin_overview import build_overview, system_status, video_rows
from standards_rag.library import DocumentLibrary
from standards_rag.models import DocumentType, SourceChunk, StandardDocument
from standards_rag.retrieval import InMemoryStandardsStore
from standards_rag.video import VideoTranscript, VideoTranscriptStore, designations_match


def _document(document_id: str, standard_id: str, body: str, **metadata: object) -> StandardDocument:
    return StandardDocument(
        document_id=document_id,
        standard_id=standard_id,
        title=f"{standard_id} title",
        issuing_body=body,
        document_type=DocumentType.TEST_METHOD,
        metadata=dict(metadata),
    )


def _chunk(document_id: str, index: int) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"{document_id}-{index}",
        document_id=document_id,
        text="Tensile properties of geotextiles are measured on a wide-width strip.",
        page_start=index + 1,
        page_end=index + 1,
    )


def _store() -> InMemoryStandardsStore:
    store = InMemoryStandardsStore()
    store.add_documents(
        [
            (_document("astm-d4595-24", "D4595-24", "ASTM"), [_chunk("astm-d4595-24", 0)]),
            (_document("gri-gt12a", "GRI-GT12a", "GRI"), [_chunk("gri-gt12a", 0)]),
            (_document("unknown-gt1", "GT1", "UNKNOWN"), [_chunk("unknown-gt1", 0)]),
        ]
    )
    return store


def _videos(*entries: tuple[str, str]) -> VideoTranscriptStore:
    store = VideoTranscriptStore()
    for video_id, title in entries:
        # from_dict, not the constructor: it is what parses the designations out
        # of the title, which is exactly what the linkage under test relies on.
        store.add(
            VideoTranscript.from_dict(
                {"video_id": video_id, "youtube_id": video_id, "title": title,
                 "transcript": "demo transcript"}
            )
        )
    return store


class DesignationMatchTests(unittest.TestCase):
    def test_a_revision_year_is_ignored(self) -> None:
        self.assertTrue(designations_match("D4595", "D4595-24"))
        self.assertTrue(designations_match("D6574", "D6574-13(2021)"))

    def test_a_bare_designation_covers_its_lettered_variants(self) -> None:
        self.assertTrue(designations_match("GT12", "GRI-GT12a"))
        self.assertTrue(designations_match("GM13", "GRI-GM13r"))

    def test_different_standards_do_not_match(self) -> None:
        # The chat's prefix rule reads GT12 as covering GT1; on a screen that says
        # "this video covers that document" it must not.
        self.assertFalse(designations_match("GT12", "GRI-GT1"))
        self.assertFalse(designations_match("GT12a", "GRI-GT12b"))
        self.assertFalse(designations_match("D4595", "D4596"))

    def test_non_designations_never_match(self) -> None:
        self.assertFalse(designations_match("", "D4595"))
        self.assertFalse(designations_match("geotextiles", "D4595"))


class VideoRowTests(unittest.TestCase):
    def test_videos_are_linked_to_the_documents_they_cover(self) -> None:
        rows = video_rows(
            _store(),
            _videos(
                ("a", "ASTM D4595 Wide Width Tensile"),
                ("b", "GRI GT12 Geotextile Specification"),
            ),
        )
        by_id = {row["video_id"]: row for row in rows}
        self.assertTrue(by_id["a"]["linked"])
        self.assertEqual(by_id["a"]["standards"][0]["document_ids"], ["astm-d4595-24"])
        # GT12 must reach GT12a and NOT the separate GT1 document.
        self.assertEqual(by_id["b"]["standards"][0]["document_ids"], ["gri-gt12a"])

    def test_a_video_with_no_document_in_the_library_reads_as_unlinked(self) -> None:
        rows = video_rows(_store(), _videos(("c", "ASTM D9999 Something Else")))
        self.assertFalse(rows[0]["linked"])
        self.assertFalse(rows[0]["standards"][0]["in_library"])

    def test_unlinked_videos_sort_last(self) -> None:
        rows = video_rows(
            _store(),
            _videos(("c", "AAA ASTM D9999 Unlinked"), ("a", "ZZZ ASTM D4595 Linked")),
        )
        self.assertEqual([row["video_id"] for row in rows], ["a", "c"])


class SystemStatusTests(unittest.TestCase):
    def _row(self, rows: list[dict[str, str]], key: str) -> dict[str, str]:
        return next(row for row in rows if row["key"] == key)

    def test_a_store_without_a_search_service_reports_off_not_broken(self) -> None:
        rows = system_status(
            _store(),
            documents=3,
            sections=3,
            videos=1,
            linked_videos=1,
            answer_writing_active=True,
        )
        self.assertEqual(self._row(rows, "search")["state"], "off")
        self.assertEqual(self._row(rows, "library")["state"], "ok")

    def test_an_unreachable_search_service_warns_instead_of_reporting_zero(self) -> None:
        store = _store()
        store.index_stats = mock.Mock(side_effect=RuntimeError("connection refused"))
        row = self._row(
            system_status(
                store,
                documents=3,
                sections=3,
                videos=1,
                linked_videos=1,
                answer_writing_active=True,
            ),
            "search",
        )
        self.assertEqual(row["state"], "warn")
        self.assertIn("could not reach", row["detail"])

    def test_a_search_service_missing_sections_warns_with_the_shortfall(self) -> None:
        store = _store()
        store.index_stats = mock.Mock(return_value={"standards": 2, "videos": 0, "total": 2})
        row = self._row(
            system_status(
                store,
                documents=3,
                sections=3,
                videos=1,
                linked_videos=1,
                answer_writing_active=True,
            ),
            "search",
        )
        self.assertEqual(row["state"], "warn")
        self.assertIn("2 of the 3", row["detail"])

    def test_an_empty_library_is_flagged(self) -> None:
        rows = system_status(
            InMemoryStandardsStore(),
            documents=0,
            sections=0,
            videos=0,
            linked_videos=0,
            answer_writing_active=False,
        )
        self.assertEqual(self._row(rows, "library")["state"], "warn")
        self.assertEqual(self._row(rows, "videos")["state"], "off")
        self.assertEqual(self._row(rows, "writing")["state"], "off")


class OverviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = _store()
        self.library = DocumentLibrary(self.store, index_path=Path(self.tmp.name) / "index.json")

    def test_totals_match_the_library(self) -> None:
        overview = build_overview(self.library, self.store, _videos(("a", "ASTM D4595 Tensile")))
        self.assertEqual(overview["totals"]["documents"], 3)
        self.assertEqual(overview["totals"]["sections"], 3)
        self.assertEqual(overview["totals"]["videos"], 1)
        self.assertEqual(overview["totals"]["linked_videos"], 1)

    def test_publishers_are_counted_and_ordered_by_size(self) -> None:
        overview = build_overview(self.library, self.store, _videos())
        # GT1 is stored as UNKNOWN but is a GRI standard, so it counts as GRI.
        self.assertEqual(overview["by_publisher"][0], {"name": "GRI", "count": 2})
        self.assertEqual(overview["by_publisher"][1], {"name": "ASTM", "count": 1})

    def test_only_documents_added_through_the_portal_count_as_uploads(self) -> None:
        overview = build_overview(self.library, self.store, _videos())
        self.assertEqual(overview["totals"]["uploads"], 0)
        self.assertEqual(overview["recent_uploads"], [])

        self.store.add_document(
            _document(
                "gri-gm13r",
                "GRI-GM13r",
                "GRI",
                added_at="2026-08-08T10:00:00+00:00",
                added_by="admin@gsi.org",
            ),
            [_chunk("gri-gm13r", 0)],
        )
        overview = build_overview(self.library, self.store, _videos())
        self.assertEqual(overview["totals"]["uploads"], 1)
        self.assertEqual(overview["recent_uploads"][0]["standard_id"], "GRI-GM13r")
        self.assertEqual(overview["recent_uploads"][0]["added_by"], "admin@gsi.org")

    def test_recent_uploads_are_capped(self) -> None:
        for index in range(4):
            self.store.add_document(
                _document(
                    f"gri-gm{index}",
                    f"GRI-GM{index}",
                    "GRI",
                    added_at=f"2026-08-0{index + 1}T10:00:00+00:00",
                ),
                [_chunk(f"gri-gm{index}", 0)],
            )
        overview = build_overview(self.library, self.store, _videos(), recent_limit=2)
        self.assertEqual(len(overview["recent_uploads"]), 2)
        self.assertEqual(overview["totals"]["uploads"], 4)
        # Newest first.
        self.assertEqual(overview["recent_uploads"][0]["standard_id"], "GRI-GM3")


if __name__ == "__main__":
    unittest.main()
