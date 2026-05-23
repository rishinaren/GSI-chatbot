"""Load a project-root ``.env`` into the process environment (stdlib only)."""

from __future__ import annotations

import os
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
