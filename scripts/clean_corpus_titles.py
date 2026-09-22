#!/usr/bin/env python3
"""Normalize stored document titles without re-embedding any chunk vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from standards_rag.env_bootstrap import (  # noqa: E402
    default_standards_index_path,
    load_dotenv_files,
)
from standards_rag.ingestion import clean_document_title, title_from_cover_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_dotenv_files()

    index_path = default_standards_index_path()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    first_chunk_by_document: dict[str, dict] = {}
    for chunk in sorted(payload["chunks"], key=lambda item: (item["document_id"], item["order"])):
        first_chunk_by_document.setdefault(chunk["document_id"], chunk)
    changes: list[tuple[str, str, str]] = []
    for document in payload["documents"]:
        before = str(document.get("title") or "")
        first_chunk = first_chunk_by_document.get(document["document_id"], {})
        recovered = title_from_cover_text(str(first_chunk.get("text") or ""))
        after = recovered or clean_document_title(before)
        if after != before:
            changes.append((document["document_id"], before, after))
            document["title"] = after

    print(f"{len(changes)} title(s) need normalization")
    for document_id, before, after in changes:
        print(f"{document_id}: {before[:72]!r} -> {after[:72]!r}")
    if not args.apply or not changes:
        return 0

    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(index_path)
    print(f"updated {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
