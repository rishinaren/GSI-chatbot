"""The document library: what the assistant can read, and how an admin adds to it.

Two callers share this module:

* ``scripts/ingest_astm.py`` / ``scripts/ingest_gri.py`` - bulk, developer-driven
  ingestion of a folder of PDFs from a laptop.
* the ``/admin/documents`` API - one PDF at a time, uploaded through the
  dashboard by a GSI administrator who should never have to know what a vector
  or a chunk is.

Both end in the same place: chunks embedded into Pinecone, document + chunk
records merged into ``STANDARDS_INDEX_PATH``, and the PDF kept on disk so
citations can link to it. The admin path adds three things the scripts do not
need:

1. it mutates the *running* store, so an upload is searchable immediately with
   no redeploy;
2. it pushes the PDF and the refreshed index back to S3, so the addition
   survives a container restart (the container re-syncs from S3 at boot);
3. it replaces a document cleanly when the same standard is uploaded twice -
   the previous version's chunks are removed from Pinecone and the index first,
   instead of being left behind as orphans.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

from standards_rag.env_bootstrap import default_standards_index_path, project_root
from standards_rag.ingestion import (
    PageText,
    chunk_pages,
    compute_sha256,
    extract_pdf_pages,
    infer_document_metadata,
)
from standards_rag.models import SourceChunk, StandardDocument
from standards_rag.retrieval import InMemoryStandardsStore

logger = logging.getLogger(__name__)

# Uploads land here, relative to the project root - which is also the root the
# S3 assets bucket mirrors, so the same relative path is the S3 key.
UPLOAD_DIR = "documents/uploads"
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
# A text PDF of a single standard yields thousands of characters. Anything under
# this is almost certainly a scan, and would be ingested as an empty document.
MIN_EXTRACTED_CHARS = 200
KNOWN_BODIES: tuple[str, ...] = ("ASTM", "GRI", "ISO")


class LibraryError(Exception):
    """A problem worth showing to the person doing the upload, in their words."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# Who may manage the library lives in ``admin_access`` - the root allowlist plus
# the grants admins make at runtime. Kept out of this module so there is exactly
# one answer to "is this person an admin".


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------


def canonical_issuing_body(document: StandardDocument) -> str:
    """Canonical issuing body, inferring GRI/ISO/ASTM for legacy docs stored as UNKNOWN."""
    body = (document.issuing_body or "").upper()
    if body in {"ASTM", "GRI", "ISO"}:
        return body
    sid = document.standard_id.upper().replace("ASTM", "").strip().lstrip("-_ ")
    if sid.startswith("GRI") or re.match(r"^G[A-Z]\d", sid):
        return "GRI"
    if sid.startswith("ISO"):
        return "ISO"
    if re.match(r"^[A-Z]\d", sid):
        return "ASTM"
    return body or "UNKNOWN"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalized_standard_key(standard_id: str) -> str:
    """Collapse a designation to what makes two references the same standard.

    ``GRI-GC1``, ``GC1`` and ``gri gc 1`` all key to ``GC1`` so a re-upload
    replaces the document already in the library instead of adding a twin under
    a slightly different id.
    """
    value = re.sub(r"^(ASTM|GRI|ISO|BS)[-_ ]+", "", (standard_id or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def derive_document_id(issuing_body: str, standard_id: str) -> str:
    """Reproduce the ids already in the corpus: ``astm-d4595-24``, ``gri-gm13r``.

    GRI ids drop the redundant body prefix because the standard id already
    carries it (``GRI-GM13r`` would otherwise slug to ``gri-gri-gm13r``).
    """
    body = (issuing_body or "").strip().upper()
    standard = (standard_id or "").strip()
    if body == "GRI":
        bare = re.sub(r"^GRI[-_ ]*", "", standard, flags=re.IGNORECASE)
        return f"gri-{slugify(bare)}".rstrip("-")
    return slugify(f"{body}-{standard}")


# ---------------------------------------------------------------------------
# GRI cover-page parsing (shared with scripts/ingest_gri.py)
# ---------------------------------------------------------------------------

FAMILY_NAME = {
    "GG": "Geogrid",
    "GM": "Geomembrane",
    "GT": "Geotextile",
    "GN": "Geonet",
    "GC": "Geocomposite",
    "GCL": "Geosynthetic Clay Liner",
    "GS": "Geosynthetic",
}

_TYPE_PHRASE = r"(Standard\s+(?:Test\s+Method|Practice|Specification|Guide))"
_TITLE_RE = re.compile(
    _TYPE_PHRASE + r'\s*(?:®|\*|¹|\d|\s)*\s*for\s*[“"]([^”"]{6,160})[”"]',
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(r'[“"]([^”"]{10,160})[”"]')

# A GRI cover page reads
#   GRI <Type> <code>*  /  Standard <Type> for[:]  /  <title…>  /  This … was developed by the Geosynthetic …
# so the title is the text between the "Standard <Type> [for]" line and that boilerplate
# (or the "Scope" heading). Anchoring on the lead-in skips the header line.
_TYPE_RE = re.compile(
    r"Standard\s+(Test\s+Method|Practice|Specification|Guide|of\s+Practice)\b", re.IGNORECASE
)
_LEADIN_RE = re.compile(r'\s*(for\b|[:“"”])')
_DEVELOPED_RE = re.compile(
    r"This\s+[\w\s]{0,40}?was\s+developed\s+by\s+the\s+Geosynthetic", re.IGNORECASE
)
_SCOPE_RE = re.compile(r"\n\s*1\.?\s*Scope\b|\n\s*Scope\b", re.IGNORECASE)

# GRI PDFs are named for their designation (gm13r.pdf -> GRI-GM13r). GCL before
# GC/GG so the longer family wins the alternation.
_GRI_STEM_RE = re.compile(r"^(?:GRI[-_ ]*)?(GCL|GG|GM|GN|GC|GT|GS)\s*-?\s*(\d+)([a-zA-Z]?)$", re.IGNORECASE)
_GRI_TEXT_CODE_RE = re.compile(
    r"\bGRI\b[^\n]{0,40}?\b(GCL|GG|GM|GN|GC|GT|GS)\s*-?\s*(\d+)([a-zA-Z]?)\b", re.IGNORECASE
)
_GRI_INSTITUTE_RE = re.compile(r"Geosynthetic\s+Research\s+Institute", re.IGNORECASE)


def extract_gri_title(raw_text: str) -> str | None:
    """Pull a clean ``Standard <Type> for "<title>"`` string from a GRI cover page."""
    text = raw_text[:2200]
    intro = next((m for m in _TYPE_RE.finditer(text) if _LEADIN_RE.match(text, m.end())), None)
    if intro is None:
        return None
    grp = intro.group(1)
    phrase = (
        "Standard Practice"
        if grp.lower() == "of practice"
        else "Standard " + " ".join(grp.split()).title()
    )
    start = intro.end()
    consumed = re.match(r"\s*for\s*:?", text[start:], re.IGNORECASE)
    if consumed:
        start += consumed.end()
    ends = [e.start() for e in (_DEVELOPED_RE.search(text, start), _SCOPE_RE.search(text, start)) if e]
    end = min(ends) if ends else start + 300
    inner = re.sub(r"\s+", " ", text[start:end]).strip()
    # strip a trailing footnote / service-mark that trails the closing quote
    #   ...Efficiency"**   ...Method”1   ...Barriers” SM
    inner = re.sub(r'([”“"’\'])\s*(?:SM|TM|[*¹²³™®\d]+)\s*$', r"\1", inner).strip()
    # normalize curly quotes to straight, then peel one wrapping pair
    inner = inner.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").strip()
    if inner[:1] == '"' and inner[-1:] == '"':
        inner = inner[1:-1].strip()
    inner = inner.rstrip(".,;:").strip()
    # any remaining interior double quotes become single quotes for a clean nested label
    inner = inner.replace('"', "'").strip()
    if len(inner) < 8 or inner[:1].isdigit():
        return None
    return f'{phrase} for "{inner[0].upper()}{inner[1:]}"'


def gri_code(stem: str) -> tuple[str, str]:
    """Return (family, code) from a file stem, e.g. 'gm13r' -> ('GM', 'GM13r')."""
    match = re.match(r"^([a-zA-Z]+)(\d+)([a-zA-Z]*)$", stem)
    if match:
        family = match.group(1).upper()
        return family, f"{family}{match.group(2)}{match.group(3).lower()}"
    return stem[:2].upper(), stem.upper()


def clean_title(raw_text: str, inferred: str, family: str, standard_id: str) -> str:
    robust = extract_gri_title(raw_text)
    if robust:
        return robust

    text = raw_text[:3500]
    match = _TITLE_RE.search(text)
    if match:
        phrase = " ".join(match.group(1).split()).title()
        inner = " ".join(match.group(2).split())
        return f'{phrase} for "{inner}"'

    quote = _QUOTE_RE.search(text)
    if quote and not quote.group(1).lstrip()[:1].isdigit():
        return f'GRI {standard_id}: "{" ".join(quote.group(1).split())}"'

    inf = " ".join((inferred or "").split())
    looks_bad = (
        not inf
        or re.match(r"^\d", inf)
        or "is developed by the Geosynthetic Research Institute" in inf
        or "values listed in SI units" in inf
        or len(inf) < 12
    )
    if not looks_bad:
        return inf[:140]
    return f"{standard_id} {FAMILY_NAME.get(family, 'Geosynthetic')} Standard"


def overrides_for(stem: str) -> tuple[str, dict[str, object]]:
    family, code = gri_code(stem)
    return family, {
        "standard_id": f"GRI-{code}",
        "issuing_body": "GRI",
        "document_id": "gri-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-"),
    }


def detect_gri_identity(text: str, filename: str) -> str | None:
    """Return the GRI code (``GM13r``) when this PDF is a GRI standard, else None.

    The file name is the reliable signal - GRI publishes ``gm13r.pdf`` - so it is
    tried first. Uploads arrive with whatever name the sender used, so fall back
    to the cover page, which names the institute and the designation together.
    """
    stem = Path(filename or "").stem.strip()
    match = _GRI_STEM_RE.match(stem)
    if match:
        return _gri_code_from_match(match)
    head = text[:2500]
    if not _GRI_INSTITUTE_RE.search(head):
        return None
    match = _GRI_TEXT_CODE_RE.search(head)
    return _gri_code_from_match(match) if match else None


def _gri_code_from_match(match: re.Match[str]) -> str:
    family, digits, suffix = match.group(1).upper(), match.group(2), match.group(3)
    return f"{family}{digits}{suffix.lower()}"


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------


def merge_documents_into_index(
    index_path: Path,
    documents: Iterable[StandardDocument],
    chunks: Iterable[SourceChunk],
    *,
    remove_chunk_ids: Iterable[str] = (),
) -> tuple[int, int]:
    """Merge documents/chunks into the on-disk index without disturbing the rest.

    Re-runnable: a document already present is overwritten by document id.
    ``remove_chunk_ids`` drops the previous revision's chunks, which otherwise
    linger under ids the new chunking will not reuse.
    """
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        data = {"documents": [], "chunks": []}

    doc_map = {d["document_id"]: d for d in data.get("documents", [])}
    chunk_map = {c["chunk_id"]: c for c in data.get("chunks", [])}
    for chunk_id in remove_chunk_ids:
        chunk_map.pop(chunk_id, None)
    for document in documents:
        doc_map[document.document_id] = document.to_dict()
    for chunk in chunks:
        chunk_map[chunk.chunk_id] = chunk.to_dict()

    data["documents"] = sorted(doc_map.values(), key=lambda x: x["document_id"])
    data["chunks"] = sorted(chunk_map.values(), key=lambda x: x["chunk_id"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return len(data["documents"]), len(data["chunks"])


def merge_into_index(store: InMemoryStandardsStore, index_path: Path) -> tuple[int, int]:
    """Script-facing wrapper: merge everything a freshly-built store holds."""
    return merge_documents_into_index(index_path, store.documents.values(), store.chunks.values())


def assets_s3_target() -> tuple[str, str] | None:
    """Parse ``STANDARDS_ASSETS_S3_URI`` into (bucket, prefix); None when unset."""
    uri = os.getenv("STANDARDS_ASSETS_S3_URI", "").strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("STANDARDS_ASSETS_S3_URI must look like s3://bucket/prefix")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def publish_assets_to_s3(relative_paths: Iterable[str]) -> int:
    """Upload project-root-relative files to the assets bucket. Returns the count.

    The inverse of ``env_bootstrap.sync_runtime_assets_from_s3``: the bucket
    mirrors the project root, so the relative path is the key. Returns 0 when no
    bucket is configured (local dev), which is not an error.
    """
    target = assets_s3_target()
    if target is None:
        return 0
    bucket, prefix = target

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - only when aws extra is missing
        raise RuntimeError("Install the optional 'aws' dependencies for S3 asset publishing.") from exc

    client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    root = project_root()
    uploaded = 0
    for relative in relative_paths:
        local_path = root / relative
        if not local_path.is_file():
            continue
        client.upload_file(str(local_path), bucket, f"{prefix}{relative}")
        uploaded += 1
    return uploaded


# ---------------------------------------------------------------------------
# The library itself
# ---------------------------------------------------------------------------


@contextmanager
def _temporary_pdf(data: bytes) -> Iterator[Path]:
    handle = NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        handle.write(data)
        handle.close()
        yield Path(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _validate_pdf(data: bytes, filename: str) -> None:
    if not data:
        raise LibraryError("That file came through empty. Please try choosing it again.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise LibraryError(
            f"That file is larger than {limit} MB, which is bigger than we can take. "
            "If it is a bundle of several standards, please upload them one at a time."
        )
    if not data.lstrip()[:5].startswith(b"%PDF"):
        name = Path(filename or "").name or "That file"
        raise LibraryError(f"{name} does not look like a PDF. Please upload the PDF version.")


def _read_pages(path: Path) -> list[PageText]:
    """Extract pages, turning an unreadable file into something worth reading.

    A damaged or password-protected PDF raises out of PyMuPDF; left alone that
    surfaces as a bare 500, which the browser reports as "Failed to fetch".
    """
    # Check the dependency up front: PyMuPDF signals a bad *file* with its own
    # RuntimeError subclasses, so the two failures cannot be told apart below.
    try:
        import fitz  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only when the pdf extra is missing
        raise RuntimeError("Install the optional 'pdf' dependencies to ingest PDFs.") from exc

    try:
        return extract_pdf_pages(path)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same story
        raise LibraryError(
            "We could not open that PDF - it looks damaged, or it is protected by a password. "
            "Try opening it on your computer, saving a fresh copy, and uploading that."
        ) from exc


def _page_count(document: StandardDocument, chunks: list[SourceChunk], pages: int) -> int:
    if pages:
        return pages
    return max((chunk.page_end or chunk.page_start or 0) for chunk in chunks) if chunks else 0


def _natural_key(value: str) -> tuple[Any, ...]:
    """Sort designations the way a person reads them: D5321 before D10319."""
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value or "")
    )


def _clean_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


class DocumentLibrary:
    """Read and write the corpus behind a running API instance.

    Holds the *same* store object the chat engine queries, so a document added
    here is answerable on the next question without a restart.
    """

    def __init__(self, store: InMemoryStandardsStore, *, index_path: Path | None = None) -> None:
        self._store = store
        self._index_path = index_path or default_standards_index_path()
        self._lock = threading.Lock()

    # -- reading ----------------------------------------------------------

    def documents(self) -> list[dict[str, Any]]:
        """Every document the assistant can quote, newest additions first."""
        chunk_counts: Counter[str] = Counter()
        last_page: dict[str, int] = {}
        for chunk in self._store.chunks.values():
            chunk_counts[chunk.document_id] += 1
            page = chunk.page_end or chunk.page_start or 0
            if page > last_page.get(chunk.document_id, 0):
                last_page[chunk.document_id] = page

        rows: list[dict[str, Any]] = []
        for document in self._store.documents.values():
            metadata = document.metadata or {}
            rows.append(
                {
                    "document_id": document.document_id,
                    "standard_id": document.standard_id,
                    "title": document.title,
                    "issuing_body": canonical_issuing_body(document),
                    "document_type": document.document_type.value,
                    "year": document.year,
                    "section_count": chunk_counts.get(document.document_id, 0),
                    "page_count": last_page.get(document.document_id, 0),
                    "added_at": metadata.get("added_at"),
                    "added_by": metadata.get("added_by"),
                    "uploaded": bool(metadata.get("added_at")),
                }
            )
        # What someone just added is what they want to see, so those lead (newest
        # first); the standing corpus follows in the order they would look it up.
        added = sorted(
            (row for row in rows if row["added_at"]), key=lambda row: row["added_at"], reverse=True
        )
        standing = sorted(
            (row for row in rows if not row["added_at"]),
            key=lambda row: _natural_key(row["standard_id"]),
        )
        return added + standing

    def summary(self) -> dict[str, Any]:
        rows = self.documents()
        by_body: Counter[str] = Counter(row["issuing_body"] for row in rows)
        return {
            "document_count": len(rows),
            "section_count": len(self._store.chunks),
            "by_publisher": dict(sorted(by_body.items())),
        }

    # -- inspecting an upload before committing ---------------------------

    def analyze(self, data: bytes, filename: str) -> dict[str, Any]:
        """Read a PDF and guess its identity. Writes nothing.

        The dashboard shows this back for confirmation, so a wrong guess costs
        the uploader an edit rather than a bad document in the corpus.
        """
        _validate_pdf(data, filename)
        with _temporary_pdf(data) as path:
            pages = _read_pages(path)
            text = "\n".join(page.text for page in pages)
            if len(text.strip()) < MIN_EXTRACTED_CHARS:
                raise LibraryError(
                    "We could not find any readable text in this PDF - it looks like a scan "
                    "or a photo of the pages. Please upload a version where the words can be "
                    "selected and copied."
                )
            document, chunks = self._build(text, pages, filename)

        existing = self._find_existing(canonical_issuing_body(document), document.standard_id)
        return {
            "file_name": Path(filename or "").name,
            "size_bytes": len(data),
            "standard_id": document.standard_id,
            "title": document.title,
            "issuing_body": canonical_issuing_body(document),
            "document_type": document.document_type.value,
            "year": document.year,
            "page_count": len(pages),
            "section_count": len(chunks),
            "text_preview": re.sub(r"\s+", " ", text[:600]).strip(),
            "already_in_library": (
                {
                    "standard_id": existing.standard_id,
                    "title": existing.title,
                }
                if existing is not None
                else None
            ),
        }

    # -- committing -------------------------------------------------------

    def add(
        self,
        data: bytes,
        filename: str,
        *,
        standard_id: str,
        title: str,
        issuing_body: str,
        year: Any = None,
        added_by: str | None = None,
    ) -> dict[str, Any]:
        """Add (or replace) a document: embed it, index it, and keep the PDF."""
        _validate_pdf(data, filename)
        standard_id = (standard_id or "").strip()
        title = (title or "").strip()
        body = (issuing_body or "").strip().upper() or "UNKNOWN"
        if not standard_id:
            raise LibraryError("Please give the standard a number, like D4595-24 or GM13.")
        if not title:
            raise LibraryError("Please give the standard a title.")

        if not derive_document_id(body, standard_id):
            raise LibraryError("That standard number has no letters or numbers we can file it under.")

        with self._lock:
            # Re-uploading a standard already in the corpus reuses its id, so the
            # newer PDF replaces it rather than sitting beside it as a near-twin.
            existing = self._find_existing(body, standard_id)
            replaced = existing is not None
            document_id = existing.document_id if existing else derive_document_id(body, standard_id)
            pdf_relative = f"{UPLOAD_DIR}/{document_id}.pdf"
            pdf_path = project_root() / pdf_relative
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(data)

            pages = _read_pages(pdf_path)
            text = "\n".join(page.text for page in pages)
            if len(text.strip()) < MIN_EXTRACTED_CHARS:
                pdf_path.unlink(missing_ok=True)
                raise LibraryError(
                    "We could not find any readable text in this PDF - it looks like a scan "
                    "or a photo of the pages. Please upload a version where the words can be "
                    "selected and copied."
                )

            document, chunks = self._build(
                text,
                pages,
                filename,
                source_path=pdf_relative,
                checksum=compute_sha256(pdf_path),
                overrides={
                    "standard_id": standard_id,
                    "title": title,
                    "issuing_body": body,
                    "document_id": document_id,
                    **({"year": _clean_year(year)} if _clean_year(year) is not None else {}),
                    "metadata": {
                        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "added_by": added_by or "",
                        "original_file_name": Path(filename or "").name,
                    },
                },
            )
            if not chunks:
                pdf_path.unlink(missing_ok=True)
                raise LibraryError("We could not split this PDF into readable sections.")

            # Drop the previous revision first: re-chunking mints new chunk ids,
            # so the old ones would otherwise stay in Pinecone forever.
            stale_chunk_ids = self._retire(document_id) if replaced else []

            self._store.add_document(document, chunks)
            merge_documents_into_index(
                self._index_path,
                [document],
                chunks,
                remove_chunk_ids=stale_chunk_ids,
            )
            warning = self._publish(pdf_relative)

        return {
            "document_id": document.document_id,
            "standard_id": document.standard_id,
            "title": document.title,
            "issuing_body": canonical_issuing_body(document),
            "document_type": document.document_type.value,
            "year": document.year,
            "page_count": len(pages),
            "section_count": len(chunks),
            "replaced": replaced,
            "warning": warning,
        }

    # -- internals --------------------------------------------------------

    def _build(
        self,
        text: str,
        pages: list[PageText],
        filename: str,
        *,
        source_path: str | None = None,
        checksum: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[StandardDocument, list[SourceChunk]]:
        """Infer metadata (GRI-aware), apply any confirmed overrides, and chunk."""
        merged: dict[str, Any] = dict(self._auto_metadata(text, filename))
        merged.update(overrides or {})
        document = infer_document_metadata(
            text,
            source_path=source_path or Path(filename or "upload.pdf").name,
            checksum=checksum,
            overrides=merged,
        )
        return document, chunk_pages(document, pages)

    def _auto_metadata(self, text: str, filename: str) -> dict[str, Any]:
        """GRI PDFs need their own identity rules; ASTM/ISO inference handles the rest."""
        code = detect_gri_identity(text, filename)
        if not code:
            return {}
        family = re.match(r"^[A-Z]+", code)
        standard_id = f"GRI-{code}"
        inferred = infer_document_metadata(text, source_path=filename)
        return {
            "standard_id": standard_id,
            "issuing_body": "GRI",
            "title": clean_title(text, inferred.title, family.group(0) if family else "", standard_id),
        }

    def _find_existing(self, issuing_body: str, standard_id: str) -> StandardDocument | None:
        """The document already in the library that this upload is a version of.

        Matches on the derived id first, then on the designation itself - the
        corpus holds a few documents filed under older id conventions
        (``unknown-gt1`` for ``GT1``) that the id rule alone would not find.
        """
        existing = self._store.documents.get(derive_document_id(issuing_body, standard_id))
        if existing is not None:
            return existing
        key = normalized_standard_key(standard_id)
        body = (issuing_body or "").strip().upper()
        if not key:
            return None
        for document in self._store.documents.values():
            if (
                normalized_standard_key(document.standard_id) == key
                and canonical_issuing_body(document) == body
            ):
                return document
        return None

    def _retire(self, document_id: str) -> list[str]:
        """Remove a document's chunks from the live store and from Pinecone."""
        removed = self._store.remove_document(document_id)
        delete_chunks = getattr(self._store, "delete_chunks", None)
        if callable(delete_chunks) and removed:
            try:
                delete_chunks(removed)
            except Exception as exc:  # noqa: BLE001 - the new version still lands
                logger.warning("Could not delete old vectors for %s: %s", document_id, exc)
        return removed

    def _publish(self, pdf_relative: str) -> str | None:
        """Mirror the PDF + refreshed index to S3 so a restart does not lose them."""
        if assets_s3_target() is None:
            return None
        try:
            index_relative = self._index_path.relative_to(project_root()).as_posix()
        except ValueError:
            index_relative = None
        targets = [pdf_relative] + ([index_relative] if index_relative else [])
        try:
            publish_assets_to_s3(targets)
        except Exception as exc:  # noqa: BLE001 - surfaced to the uploader, not raised
            logger.exception("Could not publish library assets to S3")
            return (
                "The document is live and searchable now, but we could not save a backup "
                f"copy ({exc}). Please tell your developer before restarting the service."
            )
        return None
