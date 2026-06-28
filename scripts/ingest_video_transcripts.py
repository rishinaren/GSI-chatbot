"""Embed video transcripts into the isolated Pinecone ``videos`` namespace.

This powers the *semantic* video signal: surfacing a video whose spoken content
matches a question even when the answer cites no standard the video covers. The
vectors live in their own namespace so they never appear in standards chunk
retrieval. Transcripts are chunked into small word-windows so a focused video
produces many strong hits while a passing mention produces only one weak hit
(the density gate at query time relies on this).

Re-runnable: clears the namespace first, then re-embeds every video.

Usage:  python scripts/ingest_video_transcripts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.env_bootstrap import load_dotenv_files
from standards_rag.pinecone_hybrid import (
    VIDEO_NAMESPACE,
    PineconeHybridStore,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)
from standards_rag.video import default_video_transcripts_path

WINDOW_WORDS = 80
OVERLAP_WORDS = 15


def chunk_words(text: str) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = WINDOW_WORDS - OVERLAP_WORDS
    for start in range(0, max(len(words), 1), step):
        window = " ".join(words[start : start + WINDOW_WORDS]).strip()
        if len(window) >= 40:
            chunks.append(window)
        if start + WINDOW_WORDS >= len(words):
            break
    return chunks


def main() -> None:
    load_dotenv_files()
    if not pinecone_enabled_from_env():
        raise SystemExit("Pinecone is not configured (set PINECONE_API_KEY / PINECONE_INDEX).")

    data = json.loads(default_video_transcripts_path().read_text(encoding="utf-8"))
    videos = data.get("videos", data) if isinstance(data, dict) else data
    store = PineconeHybridStore(load_pinecone_config_from_env())

    # Start clean so removed/edited videos don't leave orphans.
    try:
        store._index.delete(delete_all=True, namespace=VIDEO_NAMESPACE)
    except Exception:
        pass  # namespace may not exist yet

    vectors: list[dict[str, object]] = []
    total_chunks = 0
    for video in videos:
        vid = video.get("video_id") or video.get("youtube_id")
        transcript = video.get("transcript") or ""
        chunks = chunk_words(transcript)
        if not chunks:
            print(f"  (no transcript) {video.get('title','')[:50]}")
            continue
        embeddings = store.embed_passages(chunks)
        for idx, (chunk, values) in enumerate(zip(chunks, embeddings, strict=True)):
            vectors.append(
                {
                    "id": f"{vid}#c{idx}",
                    "values": values,
                    "metadata": {"video_id": vid, "chunk_idx": idx, "text_preview": chunk[:300]},
                }
            )
        total_chunks += len(chunks)
        print(f"  {len(chunks):3} chunks  {video.get('title','')[:56]}")

    store.upsert_video_chunks(vectors)
    print(f"\nUpserted {total_chunks} transcript chunks for {len(videos)} videos "
          f"into namespace '{VIDEO_NAMESPACE}'.")


if __name__ == "__main__":
    main()
