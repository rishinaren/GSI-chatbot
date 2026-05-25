"""Load a project-root ``.env`` into the process environment (stdlib only)."""

from __future__ import annotations

import os
from urllib.parse import urlparse
from pathlib import Path


def project_root() -> Path:
    """``src/standards_rag`` → project root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def load_dotenv_files() -> None:
    """Populate ``os.environ`` from ``.env`` / ``.env.local`` if present.

    Existing environment variables are not overwritten so deployment shells keep precedence.
    """
    root = project_root()
    for name in (".env", ".env.local"):
        path = root / name
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value


def default_standards_index_path() -> Path:
    """Resolve ``STANDARDS_INDEX_PATH`` relative to the project root when not absolute."""
    raw = os.getenv("STANDARDS_INDEX_PATH", "data/index/standards-index.json").strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def sync_runtime_assets_from_s3() -> int:
    """Sync index/PDF artifacts from S3 prefix into project root.

    Enable by setting ``STANDARDS_ASSETS_S3_URI=s3://bucket/prefix``.
    Files are downloaded preserving relative paths under the prefix.
    Returns the number of files downloaded.
    """
    uri = os.getenv("STANDARDS_ASSETS_S3_URI", "").strip()
    if not uri:
        return 0

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("STANDARDS_ASSETS_S3_URI must look like s3://bucket/prefix")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'aws' dependencies for S3 asset sync.") from exc

    target_root = project_root()
    client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    paginator = client.get_paginator("list_objects_v2")

    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key", ""))
            if not key or key.endswith("/"):
                continue
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            if not rel:
                continue
            local_path = target_root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            downloaded += 1
    return downloaded
