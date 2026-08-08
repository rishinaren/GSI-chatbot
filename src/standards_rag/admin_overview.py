"""The admin dashboard: how big the knowledge base is, and whether it is healthy.

``library`` answers "what is in here" and can change it. This module only reads,
and answers the two questions someone actually has on opening the admin portal:

1. **How much does the assistant know?** - documents, the searchable sections
   they were split into, the video walkthroughs linked to them, what was added
   most recently and by whom.
2. **Is anything broken?** - one line per moving part, in words that tell a
   non-technical administrator whether to act or ignore it.

Every status check is written so that *failing to check* is reported honestly
rather than as good news: an unreachable search service says so, it does not
silently report zero.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from standards_rag.library import DocumentLibrary, assets_s3_target
from standards_rag.retrieval import InMemoryStandardsStore
from standards_rag.video import VideoTranscriptStore, designations_match

logger = logging.getLogger(__name__)

# How many recent additions the dashboard lists before "see all".
RECENT_LIMIT = 6

# The three states a status line can be in. `warn` means look at it; `off` means
# a capability is switched off, which is a fact rather than a fault.
OK, WARN, OFF = "ok", "warn", "off"


# ---------------------------------------------------------------------------
# Videos <-> documents
# ---------------------------------------------------------------------------


def video_rows(store: InMemoryStandardsStore, videos: VideoTranscriptStore) -> list[dict[str, Any]]:
    """Every walkthrough video with the library documents it covers.

    A video is "linked" when at least one designation in its title resolves to a
    document in the library - the same year- and body-agnostic match the chat
    uses to decide whether to surface a video inline, so what the dashboard
    counts is what a reader would actually be shown.
    """
    corpus = [
        (document.standard_id, document.document_id)
        for document in store.documents.values()
    ]

    rows: list[dict[str, Any]] = []
    for video in videos.videos.values():
        standards: list[dict[str, Any]] = []
        for designation in video.standards:
            matches = [doc_id for sid, doc_id in corpus if designations_match(designation, sid)]
            standards.append(
                {
                    "designation": designation,
                    "in_library": bool(matches),
                    "document_ids": matches,
                }
            )
        rows.append(
            {
                "video_id": video.video_id,
                "youtube_id": video.youtube_id,
                "title": video.title,
                "channel": video.channel,
                "youtube_url": video.youtube_url,
                "thumbnail_url": video.thumbnail_url,
                "standards": standards,
                "linked": any(item["in_library"] for item in standards),
                "has_transcript": bool((video.transcript or "").strip()),
            }
        )
    # Linked videos first - an unlinked one is the row worth acting on, so it
    # sits at the bottom where the empty-ish rows are expected, not scattered.
    rows.sort(key=lambda row: (not row["linked"], row["title"].lower()))
    return rows


# ---------------------------------------------------------------------------
# Status lines
# ---------------------------------------------------------------------------


def _row(key: str, label: str, state: str, detail: str) -> dict[str, str]:
    return {"key": key, "label": label, "state": state, "detail": detail}


def _plural(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def _library_status(documents: int, sections: int) -> dict[str, str]:
    if not documents:
        return _row(
            "library",
            "Document library",
            WARN,
            "There are no documents yet, so the assistant has nothing to quote. Add one to start.",
        )
    return _row(
        "library",
        "Document library",
        OK,
        f"{_plural(documents, 'standard', 'standards')} loaded and split into "
        f"{_plural(sections, 'searchable section', 'searchable sections')}.",
    )


def _search_status(store: InMemoryStandardsStore, sections: int) -> dict[str, str]:
    """Whether the semantic search service is reachable and holding what we hold."""
    stats_fn = getattr(store, "index_stats", None)
    if not callable(stats_fn):
        return _row(
            "search",
            "Search service",
            OFF,
            "Running on the built-in keyword search. Answers still work, but they "
            "are less able to match a question that uses different words to the standard.",
        )
    try:
        stats = stats_fn()
    except Exception as exc:  # noqa: BLE001 - an unreachable service is a status, not a crash
        logger.warning("Could not read search index stats: %s", exc)
        return _row(
            "search",
            "Search service",
            WARN,
            "We could not reach the search service just now, so this figure is unknown. "
            "If answers come back thin, this is the thing to check.",
        )
    held = int(stats.get("standards") or 0)
    if sections and held < sections:
        missing = sections - held
        return _row(
            "search",
            "Search service",
            WARN,
            f"Connected, but it holds {held:,} of the {sections:,} sections in the library - "
            f"{_plural(missing, 'section is', 'sections are')} missing. A document added in "
            "the last few minutes may still be settling.",
        )
    return _row(
        "search",
        "Search service",
        OK,
        f"Connected, with {_plural(held, 'section', 'sections')} ready to search.",
    )


def _video_status(total: int, linked: int) -> dict[str, str]:
    if not total:
        return _row(
            "videos",
            "Video walkthroughs",
            OFF,
            "No videos are loaded, so answers will never offer one.",
        )
    if linked < total:
        return _row(
            "videos",
            "Video walkthroughs",
            WARN,
            f"{linked:,} of {total:,} videos are linked to a standard in the library. "
            "The rest can only be found by what is said in them, not by the standard they cover.",
        )
    return _row(
        "videos",
        "Video walkthroughs",
        OK,
        f"All {total:,} videos are linked to a standard in the library.",
    )


def _backup_status() -> dict[str, str]:
    if assets_s3_target() is None:
        return _row(
            "backup",
            "Backup copy",
            OFF,
            "Documents added here are kept on this server only. Ask your developer to "
            "switch on the backup before relying on uploads.",
        )
    return _row(
        "backup",
        "Backup copy",
        OK,
        "Every document added here is copied to long-term storage, so it survives a restart.",
    )


def _writing_status(active: bool) -> dict[str, str]:
    if not active:
        return _row(
            "writing",
            "Answer writing",
            OFF,
            "Answers are returned as the passages found, without being written up. "
            "Citations are unaffected.",
        )
    return _row(
        "writing",
        "Answer writing",
        OK,
        "Answers are written up in full sentences from the passages found.",
    )


def system_status(
    store: InMemoryStandardsStore,
    *,
    documents: int,
    sections: int,
    videos: int,
    linked_videos: int,
    answer_writing_active: bool,
) -> list[dict[str, str]]:
    """One line per moving part, ordered by how much a reader should care."""
    return [
        _library_status(documents, sections),
        _search_status(store, sections),
        _video_status(videos, linked_videos),
        _backup_status(),
        _writing_status(answer_writing_active),
    ]


# ---------------------------------------------------------------------------
# The dashboard payload
# ---------------------------------------------------------------------------


def build_overview(
    library: DocumentLibrary,
    store: InMemoryStandardsStore,
    videos: VideoTranscriptStore,
    *,
    answer_writing_active: bool = False,
    recent_limit: int = RECENT_LIMIT,
) -> dict[str, Any]:
    """Everything the admin landing page shows, in one request."""
    rows = library.documents()
    sections = len(store.chunks)
    by_publisher: dict[str, int] = {}
    pages = 0
    for row in rows:
        by_publisher[row["issuing_body"]] = by_publisher.get(row["issuing_body"], 0) + 1
        pages += int(row.get("page_count") or 0)

    videos_listed = video_rows(store, videos)
    linked = sum(1 for row in videos_listed if row["linked"])

    # `documents()` already puts uploads first, newest first, so the recent list
    # is the head of that - no second sort, and the two views cannot disagree.
    uploads = [row for row in rows if row.get("added_at")]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totals": {
            "documents": len(rows),
            "sections": sections,
            "pages": pages,
            "videos": len(videos_listed),
            "linked_videos": linked,
            "uploads": len(uploads),
        },
        "by_publisher": [
            {"name": name, "count": count}
            for name, count in sorted(by_publisher.items(), key=lambda item: (-item[1], item[0]))
        ],
        "recent_uploads": uploads[:recent_limit],
        "status": system_status(
            store,
            documents=len(rows),
            sections=sections,
            videos=len(videos_listed),
            linked_videos=linked,
            answer_writing_active=answer_writing_active,
        ),
    }
