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
        return cls(
            video_id=raw_id,
            youtube_id=youtube_id,
            title=str(data.get("title") or "Untitled video"),
            transcript=str(data.get("transcript") or ""),
            description=str(data.get("description") or ""),
            channel=str(data.get("channel") or ""),
            standards=tuple(str(s) for s in data.get("standards", []) if str(s).strip()),
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
    ) -> list[VideoMatch]:
        query_terms = set(_tokens(query))
        if not query_terms or not self.videos:
            return []

        scored: list[VideoMatch] = []
        for video in self.videos.values():
            haystack_terms = set(_tokens(video.haystack()))
            if not haystack_terms:
                continue
            overlap = query_terms & haystack_terms
            score = len(overlap) / len(query_terms)

            # Boost when the question names a standard the video covers.
            query_compact = re.sub(r"[^a-z0-9]+", "", query.lower())
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
