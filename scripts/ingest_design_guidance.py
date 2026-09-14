"""Ingest Designing with Geosynthetics into an isolated Pinecone namespace.

The books are secondary engineering guidance, not standards. This script keeps
their vectors in ``design-guidance``, writes a separate local JSON index, and
optionally uploads the JSON and source PDFs to the API's runtime-assets bucket.

Usage:
    python scripts/ingest_design_guidance.py /path/to/books.zip \
      --s3-uri s3://bucket/prefix/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.design_guidance import (  # noqa: E402
    DESIGN_GUIDANCE_NAMESPACE,
    design_book_volume,
    load_design_guidance_pdf,
)
from standards_rag.env_bootstrap import (  # noqa: E402
    default_design_guidance_index_path,
    load_dotenv_files,
    project_root,
)
from standards_rag.pinecone_hybrid import (  # noqa: E402
    PineconeHybridStore,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)

MAX_BOOK_BYTES = 50 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="ZIP, folder, or PDF containing both volumes")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--s3-uri", default=None)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the design-guidance namespace before upserting.",
    )
    return parser


def _pdfs_from_source(source: Path, temporary_dir: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("*.pdf"))
    if source.suffix.lower() == ".pdf":
        return [source]
    if source.suffix.lower() != ".zip":
        return []

    extracted: list[Path] = []
    with zipfile.ZipFile(source) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if any(item.file_size > MAX_BOOK_BYTES for item in members):
            raise ValueError("A ZIP member exceeds the 50 MB safety limit.")
        for member in members:
            file_name = Path(member.filename).name
            if Path(file_name).suffix.lower() != ".pdf":
                continue
            target = temporary_dir / file_name
            with archive.open(member) as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            extracted.append(target)
    return sorted(extracted)


def _stage_books(source: Path) -> list[Path]:
    destination = project_root() / "documents" / "Design guidance"
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gsi-design-guidance-") as temp_name:
        source_pdfs = _pdfs_from_source(source, Path(temp_name))
        by_volume: dict[int, Path] = {}
        for pdf in source_pdfs:
            volume = design_book_volume(pdf)
            if volume in by_volume:
                raise ValueError(f"More than one PDF was identified as Volume {volume}.")
            by_volume[volume] = pdf
        if set(by_volume) != {1, 2}:
            raise ValueError("The source must contain exactly Volume 1 and Volume 2 PDFs.")

        staged: list[Path] = []
        for volume in (1, 2):
            target = destination / f"DWG 6th Edition-Vol.{volume}.pdf"
            shutil.copy2(by_volume[volume], target)
            staged.append(target)
        return staged


def _upload_runtime_assets(s3_uri: str, index_path: Path, pdfs: list[Path]) -> None:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("--s3-uri must look like s3://bucket/prefix/")

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install the optional 'aws' dependencies for S3 upload.") from exc

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)

    for path in [index_path, *pdfs]:
        relative = path.resolve().relative_to(project_root().resolve()).as_posix()
        key = f"{prefix}{relative}"
        client.upload_file(str(path), bucket, key)
        print(f"Uploaded s3://{bucket}/{key}")


def main() -> None:
    args = _parser().parse_args()
    load_dotenv_files()
    if not pinecone_enabled_from_env():
        raise SystemExit("Pinecone is not configured.")

    staged_pdfs = _stage_books(args.source.expanduser().resolve())
    items = [
        load_design_guidance_pdf(path, volume=volume)
        for volume, path in enumerate(staged_pdfs, start=1)
    ]
    expected_chunks = sum(len(chunks) for _, chunks in items)
    if not expected_chunks:
        raise SystemExit("No searchable book text was extracted.")
    if len({chunk.chunk_id for _, chunks in items for chunk in chunks}) != expected_chunks:
        raise SystemExit("Duplicate design-guidance chunk IDs were generated.")

    config = replace(
        load_pinecone_config_from_env(),
        namespace=DESIGN_GUIDANCE_NAMESPACE,
    )
    store = PineconeHybridStore(config)
    if not args.keep_existing:
        from pinecone.exceptions import NotFoundException

        try:
            store._index.delete(delete_all=True, namespace=DESIGN_GUIDANCE_NAMESPACE)
            print(f"Cleared namespace '{DESIGN_GUIDANCE_NAMESPACE}'.")
        except NotFoundException:
            print(f"Namespace '{DESIGN_GUIDANCE_NAMESPACE}' is new; nothing to clear.")

    store.add_documents(items)
    output = (args.out or default_design_guidance_index_path()).expanduser().resolve()
    store.save_json(output)

    stats = store._index.describe_index_stats()
    stats_data = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
    namespace_data = (stats_data.get("namespaces") or {}).get(DESIGN_GUIDANCE_NAMESPACE) or {}
    remote_count = int(namespace_data.get("vector_count") or 0)
    if remote_count != expected_chunks:
        raise SystemExit(
            f"Pinecone verification failed: expected {expected_chunks} vectors, found {remote_count}."
        )

    print(
        f"Verified {len(store.documents)} volumes and {expected_chunks} chunks in "
        f"namespace '{DESIGN_GUIDANCE_NAMESPACE}'."
    )
    print(f"Wrote design-guidance index to {output}")

    s3_uri = args.s3_uri or os.getenv("STANDARDS_ASSETS_S3_URI", "").strip()
    if s3_uri:
        _upload_runtime_assets(s3_uri, output, staged_pdfs)


if __name__ == "__main__":
    main()
