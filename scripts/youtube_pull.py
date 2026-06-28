"""Pull the GSI YouTube channel's videos + transcripts into the video store.

Owner OAuth (Desktop flow) authorizes once against the channel; the refresh
token is cached in ``youtube_token.json`` (gitignored). For each upload we record
the real 11-char id, title, description, and a transcript sourced (in order) from:

  1. a downloadable caption track via the YouTube Data API (manual captions),
  2. auto/any subtitles via yt-dlp (works for unlisted videos).

Videos with no transcript from either source are listed at the end as candidates
for audio transcription (handled separately).

Usage:  python scripts/youtube_pull.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.video import parse_standard_ids_from_title

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
ROOT = Path(__file__).resolve().parents[1]
CLIENT_SECRET = ROOT / "client_secret.json"
TOKEN = ROOT / "youtube_token.json"
OUT = ROOT / "data" / "videos" / "transcripts.json"


def get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
            print("\n>>> Open this URL in your browser and approve with the GSI account:\n", flush=True)
            creds = flow.run_local_server(port=0, open_browser=False, authorization_prompt_message="{url}")
        TOKEN.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def list_uploads(yt) -> tuple[str, list[dict]]:
    channel = yt.channels().list(part="contentDetails,snippet", mine=True).execute()
    item = channel["items"][0]
    channel_title = item["snippet"]["title"]
    uploads_playlist = item["contentDetails"]["relatedPlaylists"]["uploads"]

    videos: list[dict] = []
    page_token = None
    while True:
        resp = (
            yt.playlistItems()
            .list(part="snippet,contentDetails", playlistId=uploads_playlist, maxResults=50, pageToken=page_token)
            .execute()
        )
        for it in resp["items"]:
            snippet = it["snippet"]
            videos.append(
                {
                    "youtube_id": it["contentDetails"]["videoId"],
                    "title": snippet["title"],
                    "description": snippet.get("description", ""),
                }
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return channel_title, videos


def _subs_to_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        line = re.sub(r"<[^>]+>", "", line)  # strip vtt inline timing tags
        if line:
            lines.append(line)
    deduped: list[str] = []
    for line in lines:  # rolling auto-subs repeat each line; collapse runs
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return " ".join(deduped)


def caption_via_api(yt, video_id: str) -> str:
    try:
        tracks = yt.captions().list(part="snippet", videoId=video_id).execute().get("items", [])
    except Exception:
        return ""
    tracks.sort(key=lambda t: (0 if t["snippet"].get("language", "").startswith("en") else 1,
                               1 if t["snippet"].get("trackKind") == "asr" else 0))
    for track in tracks:
        try:
            data = yt.captions().download(id=track["id"], tfmt="srt").execute()
            text = _subs_to_text(data.decode("utf-8") if isinstance(data, bytes) else str(data))
            if text.strip():
                return text
        except Exception:
            continue
    return ""


def caption_via_ytdlp(video_id: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", "en.*", "--sub-format", "vtt",
            "-o", f"{tmp}/%(id)s.%(ext)s", f"https://www.youtube.com/watch?v={video_id}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        vtts = list(Path(tmp).glob("*.vtt"))
        if not vtts:
            return ""
        return _subs_to_text(vtts[0].read_text(encoding="utf-8", errors="ignore"))


def main() -> None:
    yt = get_service()
    channel_title, videos = list_uploads(yt)
    print(f"Channel: {channel_title} — {len(videos)} videos\n")

    entries, missing = [], []
    for video in videos:
        vid = video["youtube_id"]
        transcript = caption_via_api(yt, vid) or caption_via_ytdlp(vid)
        source = "ok" if transcript else "NO TRANSCRIPT"
        standards = list(parse_standard_ids_from_title(video["title"]))
        entries.append(
            {
                "video_id": vid,
                "youtube_id": vid,
                "title": video["title"],
                "transcript": transcript,
                "description": video["description"],
                "channel": channel_title,
                "standards": standards,
                "url": f"https://www.youtube.com/watch?v={vid}",
            }
        )
        if not transcript:
            missing.append(video["title"])
        print(f"  [{source:13}] {vid}  {video['title'][:52]:52} standards={standards}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "_comment": "Auto-generated by scripts/youtube_pull.py from the GSI YouTube channel.",
                "videos": entries,
            },
            indent=2,
        )
    )
    print(f"\nWrote {len(entries)} videos -> {OUT}")
    if missing:
        print(f"\n{len(missing)} video(s) have NO transcript (need audio transcription):")
        for title in missing:
            print(f"   - {title}")


if __name__ == "__main__":
    main()
