#!/usr/bin/env python3
"""Remove known duplicate/misidentified corpus entries after verifying identical PDFs.

Dry-run is the default. Pass ``--apply`` to rewrite the local JSON index and
``--pinecone`` to delete the same stale chunk IDs from the standards namespace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from standards_rag.env_bootstrap import (  # noqa: E402
    default_standards_index_path,
    load_dotenv_files,
)
from standards_rag.pinecone_hybrid import (  # noqa: E402
    PineconeHybridStore,
    load_pinecone_config_from_env,
)


DUPLICATE_TO_CANONICAL = {
    "unknown-gt1": "gri-gt1",
    "astm-r50-120": "gri-gt7",
    "unknown-gt8": "gri-gt8",
    "iso-iso-10319": "gri-gt9",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="rewrite the local JSON index")
    parser.add_argument(
        "--pinecone",
        action="store_true",
        help="delete duplicate vectors too (requires --apply and Pinecone credentials)",
    )
    args = parser.parse_args()
    if args.pinecone and not args.apply:
        parser.error("--pinecone requires --apply")

    load_dotenv_files()
    index_path = default_standards_index_path()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in payload["documents"]}

    duplicate_ids: set[str] = set()
    for duplicate_id, canonical_id in DUPLICATE_TO_CANONICAL.items():
        duplicate = documents.get(duplicate_id)
        canonical = documents.get(canonical_id)
        if duplicate is None:
            print(f"already absent: {duplicate_id}")
            continue
        if canonical is None:
            raise SystemExit(f"refusing cleanup: canonical document is missing: {canonical_id}")
        if not duplicate.get("checksum") or duplicate.get("checksum") != canonical.get("checksum"):
            raise SystemExit(
                f"refusing cleanup: {duplicate_id} and {canonical_id} do not have identical checksums"
            )
        duplicate_ids.add(duplicate_id)
        print(f"verified duplicate: {duplicate_id} -> {canonical_id}")

    stale_chunk_ids = [
        item["chunk_id"] for item in payload["chunks"] if item["document_id"] in duplicate_ids
    ]
    print(f"would remove {len(duplicate_ids)} documents and {len(stale_chunk_ids)} chunks")
    if not args.apply or not duplicate_ids:
        return 0

    backup_path = index_path.with_name(f"{index_path.stem}.pre-dedupe-20260922.json")
    if not backup_path.exists():
        shutil.copy2(index_path, backup_path)
        print(f"backup: {backup_path}")

    payload["documents"] = [
        item for item in payload["documents"] if item["document_id"] not in duplicate_ids
    ]
    payload["chunks"] = [
        item for item in payload["chunks"] if item["document_id"] not in duplicate_ids
    ]
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(index_path)
    print(
        f"local index now has {len(payload['documents'])} documents / "
        f"{len(payload['chunks'])} chunks"
    )

    if args.pinecone and stale_chunk_ids:
        store = PineconeHybridStore(load_pinecone_config_from_env())
        store.delete_chunks(stale_chunk_ids)
        print(f"deleted {len(stale_chunk_ids)} duplicate vectors from Pinecone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
