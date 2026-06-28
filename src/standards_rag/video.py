"""Video transcript storage and retrieval.

This is the backend framework for linking standards questions to relevant
YouTube videos. Transcripts are stored as lightweight documents; at query time
we lexically match a question against transcripts, then cross-reference the
matched transcript to its YouTube video so the frontend can embed it.

The matcher is intentionally dependency-light (lexical overlap) so it works
without Pinecone, but ``VideoMatch``/``VideoTranscriptStore`` are structured so
a vector backend can be layered on later (mirroring ``pinecone_hybrid``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from standards_rag.env_bootstrap import project_root
from standards_rag.retrieval import _tokens

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")
_YT_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_EXPLICIT_VIDEO_TERMS = (
    "video",
    "videos",
    "watch",
    "youtube",
    "clip",
    "footage",
    "demonstration",
    "demo video",
    "show me a video",
    "is there a video",
    "walkthrough",
    "screencast",
    "webinar",
)


def extract_youtube_id(url_or_id: str) -> str | None:
    """Return an 11-char YouTube id from a watch/share/embed URL or bare id."""
    value = (url_or_id or "").strip()
    if not value:
        return None
    if _YT_BARE_ID_RE.match(value):
        return value
    match = _YT_ID_RE.search(value)
    return match.group(1) if match else None


def video_request_explicit(question: str) -> bool:
    """True when the user is explicitly asking to see/watch a video."""
    lowered = (question or "").lower()
    return any(term in lowered for term in _EXPLICIT_VIDEO_TERMS)


# Designations carried in a video title, e.g. "ASTM D4595, D6637 …" or "GRI GM13r".
_ASTM_DESIG_RE = re.compile(r"\b(?:ASTM[-\s]?)?(D\s?-?\s?\d{3,5}[A-Za-z]?)\b", re.IGNORECASE)
_GRI_DESIG_RE = re.compile(
    r"\b(?:GRI[-\s]?)?(G(?:CL|[CGMNST])\s?-?\s?\d+[A-Za-z]?)\b", re.IGNORECASE
)


def parse_standard_ids_from_title(title: str) -> tuple[str, ...]:
    """Extract ASTM/GRI designations from a video title, normalized (``D4491``, ``GT12``)."""
    found: list[str] = []
    for regex in (_ASTM_DESIG_RE, _GRI_DESIG_RE):
        for match in regex.finditer(title or ""):
            norm = re.sub(r"[\s\-]", "", match.group(1)).upper()
            if norm not in found:
                found.append(norm)
    return tuple(found)


def _designation_key(value: str) -> str:
    """Comparable core of a designation: lowercased alnum with the issuing body stripped."""
    compact = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return re.sub(r"^(gri|astm|iso)", "", compact)


def standards_overlap(video_standards: tuple[str, ...], cited: set[str]) -> bool:
    """True if any video designation prefix-matches any cited standard id (year-agnostic)."""
    cited_keys = [_designation_key(c) for c in cited]
    for standard in video_standards:
        key = _designation_key(standard)
        if len(key) < 3:
            continue
        for cited_key in cited_keys:
            if len(cited_key) >= 3 and (key.startswith(cited_key) or cited_key.startswith(key)):
                return True
    return False


# Precision-first gate for the *semantic* (transcript) video signal. A suggestion
# fires only on a strongly sustained match, so a passing topical mention never
# surfaces. Tuned on the 24-video set (see scripts/ingest_video_transcripts.py).
_SEM_BEST_FLOOR = 0.50      # one chunk this strong is enough
_SEM_STRONG_SCORE = 0.42    # "strong chunk" cutoff for the density signal
_SEM_STRONG_COUNT = 3       # this many strong chunks = sustained relevance


def gate_semantic_suggestions(
    hits: list[tuple[str, float]], exclude_ids: set[str], *, top: int = 1
) -> list[str]:
    """From (video_id, chunk_score) hits, return video_ids that clear the precision gate."""
    best: dict[str, float] = {}
    strong: dict[str, int] = {}
    for video_id, score in hits:
        best[video_id] = max(best.get(video_id, 0.0), score)
        if score >= _SEM_STRONG_SCORE:
            strong[video_id] = strong.get(video_id, 0) + 1
    qualified = [
        (vid, best[vid])
        for vid in best
        if vid not in exclude_ids
        and (best[vid] >= _SEM_BEST_FLOOR or strong.get(vid, 0) >= _SEM_STRONG_COUNT)
    ]
    qualified.sort(key=lambda item: item[1], reverse=True)
    return [vid for vid, _ in qualified[:top]]


@dataclass(frozen=True)
class VideoTranscript:
    """A single video's transcript plus enough metadata to embed and cite it."""

    video_id: str
    youtube_id: str
    title: str
    transcript: str
    description: str = ""
    channel: str = ""
    standards: tuple[str, ...] = field(default_factory=tuple)
    url: str | None = None

    @property
    def youtube_url(self) -> str:
        return self.url or f"https://www.youtube.com/watch?v={self.youtube_id}"

    @property
    def embed_url(self) -> str:
        return f"https://www.youtube.com/embed/{self.youtube_id}"

    @property
    def thumbnail_url(self) -> str:
        return f"https://img.youtube.com/vi/{self.youtube_id}/hqdefault.jpg"

    def haystack(self) -> str:
        return " ".join(
            [
                self.title,
                self.description,
                " ".join(self.standards),
                self.transcript,
            ]
        ).lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "youtube_id": self.youtube_id,
            "title": self.title,
            "transcript": self.transcript,
            "description": self.description,
            "channel": self.channel,
            "standards": list(self.standards),
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoTranscript":
        youtube_id = (
            extract_youtube_id(str(data.get("youtube_id") or data.get("url") or ""))
            or str(data.get("youtube_id") or "")
        )
        raw_id = str(data.get("video_id") or youtube_id or data.get("title") or "")
        title = str(data.get("title") or "Untitled video")
        # Merge any hand-tagged standards with designations parsed from the title,
        # so tagging scales to the whole channel without manual upkeep.
        explicit = [str(s) for s in data.get("standards", []) if str(s).strip()]
        standards = tuple(dict.fromkeys(explicit + list(parse_standard_ids_from_title(title))))
        return cls(
            video_id=raw_id,
            youtube_id=youtube_id,
            title=title,
            transcript=str(data.get("transcript") or ""),
            description=str(data.get("description") or ""),
            channel=str(data.get("channel") or ""),
            standards=standards,
            url=data.get("url"),
        )


@dataclass(frozen=True)
class VideoMatch:
    """A transcript matched to a query, with the resolved YouTube embed."""

    video: VideoTranscript
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video.video_id,
            "youtube_id": self.video.youtube_id,
            "title": self.video.title,
            "channel": self.video.channel,
            "standards": list(self.video.standards),
            "embed_url": self.video.embed_url,
            "youtube_url": self.video.youtube_url,
            "thumbnail_url": self.video.thumbnail_url,
            "score": round(self.score, 4),
        }


class VideoTranscriptStore:
    """In-memory transcript store with lexical retrieval (vector-ready)."""

    def __init__(self) -> None:
        self.videos: dict[str, VideoTranscript] = {}

    def __len__(self) -> int:
        return len(self.videos)

    def add(self, video: VideoTranscript) -> None:
        self.videos[video.video_id] = video

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.12,
        cited_standards: set[str] | None = None,
    ) -> list[VideoMatch]:
        cited = cited_standards or set()
        query_terms = set(_tokens(query))
        if not query_terms and not cited:
            return []
        if not self.videos:
            return []

        query_compact = re.sub(r"[^a-z0-9]+", "", query.lower())
        scored: list[VideoMatch] = []
        for video in self.videos.values():
            haystack_terms = set(_tokens(video.haystack()))
            overlap = query_terms & haystack_terms
            score = len(overlap) / len(query_terms) if query_terms else 0.0

            # Strongest signal: the video covers a standard we actually cited.
            if cited and standards_overlap(video.standards, cited):
                score += 1.0
            else:
                # Next best: the question text itself names a standard the video covers.
                for standard in video.standards:
                    std_compact = re.sub(r"[^a-z0-9]+", "", standard.lower())
                    if std_compact and std_compact in query_compact:
                        score += 0.5
                        break

            if score >= min_score:
                scored.append(VideoMatch(video=video, score=score))

        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:top_k]

    @classmethod
    def load_json(cls, path: str | Path) -> "VideoTranscriptStore":
        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.get("videos", data) if isinstance(data, dict) else data
        for item in items:
            store.add(VideoTranscript.from_dict(item))
        return store


def default_video_transcripts_path() -> Path:
    raw = os.getenv("VIDEO_TRANSCRIPTS_PATH", "data/videos/transcripts.json").strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def build_video_store_from_env() -> VideoTranscriptStore:
    path = default_video_transcripts_path()
    if path.is_file():
        try:
            return VideoTranscriptStore.load_json(path)
        except Exception:  # noqa: BLE001 - never block startup on bad transcript data
            return VideoTranscriptStore()
    return VideoTranscriptStore()


def fetch_youtube_metadata(youtube_id: str) -> dict[str, Any] | None:
    """Optional enrichment via the YouTube Data API when ``YOUTUBE_API_KEY`` is set.

    Embedding works without this (we build the iframe URL directly); this is only
    for richer titles/thumbnails if a key is configured.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key or not youtube_id:
        return None
    import urllib.request

    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet&id={youtube_id}&key={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    items = body.get("items") or []
    if not items:
        return None
    snippet = items[0].get("snippet", {})
    return {
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "description": snippet.get("description"),
    }
