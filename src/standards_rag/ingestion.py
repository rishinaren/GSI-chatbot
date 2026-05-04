"""PDF/text ingestion utilities for standards documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from standards_rag.models import DocumentType, LifecycleStatus, SourceChunk, StandardDocument

SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,120})\s*$")
ASTM_ID_RE = re.compile(r"\b(?:ASTM\s*)?([A-Z]\d{2,5}(?:/[A-Z]\d{2,5})?-\d{2,4})\b")
ISO_ID_RE = re.compile(r"\b(ISO\s+\d+(?:[-:]\d+)*(?::\d{4})?)\b", re.IGNORECASE)
BS_ID_RE = re.compile(r"\b(BS\s+(?:EN\s+)?\d+(?:[-:]\d+)*(?::\d{4})?)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"\b(?:ASTM\s*)?[A-Z]\d{2,5}(?:/[A-Z]\d{2,5})?(?:-\d{2,4})?\b|"
    r"\bISO\s+\d+(?:[-:]\d+)*(?::\d{4})?\b|"
    r"\bBS\s+(?:EN\s+)?\d+(?:[-:]\d+)*(?::\d{4})?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf_pages(path: str | Path) -> list[PageText]:
    """Extract page text using PyMuPDF when the optional PDF dependency is installed."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the optional 'pdf' dependencies to ingest PDFs.") from exc

    pages: list[PageText] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(PageText(page_number=index, text=page.get_text("text")))
    return pages


def pages_from_text(text: str) -> list[PageText]:
    """Create page records from plain text, honoring form-feed page breaks if present."""

    parts = text.split("\f")
    return [PageText(page_number=index, text=part.strip()) for index, part in enumerate(parts, start=1)]


def infer_document_metadata(
    text: str,
    *,
    source_path: str | None = None,
    checksum: str | None = None,
    overrides: dict[str, object] | None = None,
) -> StandardDocument:
    """Infer useful metadata from a standard's front matter and filename."""

    overrides = overrides or {}
    standard_id = str(overrides.get("standard_id") or _infer_standard_id(text, source_path))
    issuing_body = str(overrides.get("issuing_body") or _infer_issuing_body(standard_id, text))
    title = str(overrides.get("title") or _infer_title(text, source_path, standard_id))
    document_type = overrides.get("document_type") or _infer_document_type(text, title)
    year = overrides.get("year") or _infer_year(standard_id)
    document_id = str(overrides.get("document_id") or _slugify(f"{issuing_body}-{standard_id}"))
    references = tuple(sorted(set(match.group(0).upper() for match in REFERENCE_RE.finditer(text))))

    if standard_id.upper() in references:
        references = tuple(ref for ref in references if ref != standard_id.upper())

    review_due_year = overrides.get("review_due_year")
    if review_due_year is None and isinstance(year, int) and issuing_body.upper() == "ASTM":
        review_due_year = year + 5

    return StandardDocument(
        document_id=document_id,
        standard_id=standard_id,
        title=title,
        issuing_body=issuing_body,
        document_type=DocumentType(document_type),
        year=year if isinstance(year, int) else None,
        revision=str(overrides["revision"]) if "revision" in overrides else None,
        lifecycle_status=LifecycleStatus(
            overrides.get("lifecycle_status", LifecycleStatus.UNKNOWN)
        ),
        review_due_year=review_due_year if isinstance(review_due_year, int) else None,
        source_path=source_path,
        checksum=checksum,
        references=references,
        metadata=dict(overrides.get("metadata", {})),
    )


def load_document_from_pdf(
    path: str | Path, *, metadata_overrides: dict[str, object] | None = None
) -> tuple[StandardDocument, list[SourceChunk]]:
    source_path = str(path)
    pages = extract_pdf_pages(path)
    text = "\n".join(page.text for page in pages)
    document = infer_document_metadata(
        text,
        source_path=source_path,
        checksum=compute_sha256(path),
        overrides=metadata_overrides,
    )
    return document, chunk_pages(document, pages)


def load_document_from_text(
    text: str,
    *,
    source_path: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[StandardDocument, list[SourceChunk]]:
    document = infer_document_metadata(text, source_path=source_path, overrides=metadata_overrides)
    return document, chunk_pages(document, pages_from_text(text))


def chunk_pages(
    document: StandardDocument,
    pages: Iterable[PageText],
    *,
    max_chars: int = 1400,
    overlap_chars: int = 160,
) -> list[SourceChunk]:
    """Split page text into citation-ready chunks while preserving page and section anchors."""

    chunks: list[SourceChunk] = []
    current_parts: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_section: str | None = None
    current_heading: str | None = None
    active_section: str | None = None
    active_heading: str | None = None

    def flush() -> None:
        nonlocal current_parts, current_page_start, current_page_end, current_section, current_heading
        text = "\n\n".join(part.strip() for part in current_parts if part.strip()).strip()
        if not text:
            return
        order = len(chunks)
        chunks.append(
            SourceChunk(
                chunk_id=(
                    f"{document.document_id}:p{current_page_start or 'u'}-"
                    f"{current_page_end or current_page_start or 'u'}:c{order}"
                ),
                document_id=document.document_id,
                text=text,
                page_start=current_page_start,
                page_end=current_page_end,
                section=current_section,
                heading=current_heading,
                order=order,
            )
        )
        overlap = text[-overlap_chars:] if overlap_chars and len(text) > overlap_chars else ""
        current_parts = [overlap] if overlap else []
        current_page_start = current_page_end
        current_section = active_section
        current_heading = active_heading

    for page in pages:
        paragraphs = _paragraphs(page.text)
        for paragraph in paragraphs:
            section, heading = _section_heading(paragraph)
            if section:
                active_section = section
                active_heading = heading
            if current_page_start is None:
                current_page_start = page.page_number
                current_section = active_section
                current_heading = active_heading
            current_page_end = page.page_number
            projected_size = sum(len(part) for part in current_parts) + len(paragraph)
            if current_parts and projected_size > max_chars:
                flush()
                current_page_end = page.page_number
            current_parts.append(paragraph)
            if current_section is None:
                current_section = active_section
                current_heading = active_heading

    flush()
    return chunks


def _paragraphs(text: str) -> list[str]:
    compact_lines = [line.strip() for line in text.splitlines()]
    joined = "\n".join(line for line in compact_lines if line)
    return [part.strip() for part in re.split(r"\n\s*\n|(?<=\.)\s{2,}", joined) if part.strip()]


def _section_heading(paragraph: str) -> tuple[str | None, str | None]:
    first_line = paragraph.splitlines()[0].strip()
    match = SECTION_RE.match(first_line)
    if not match:
        return None, None
    return match.group(1), match.group(2).strip()


def _infer_standard_id(text: str, source_path: str | None) -> str:
    for regex in (ASTM_ID_RE, ISO_ID_RE, BS_ID_RE):
        match = regex.search(text)
        if match:
            value = match.group(1).upper().replace("ASTM ", "")
            return value
    if source_path:
        return Path(source_path).stem.upper().replace("_", "-")
    return "UNKNOWN-STANDARD"


def _infer_issuing_body(standard_id: str, text: str) -> str:
    upper_text = text[:1000].upper()
    if "ASTM" in upper_text or re.match(r"^[A-Z]\d", standard_id):
        return "ASTM"
    if standard_id.upper().startswith("ISO"):
        return "ISO"
    if standard_id.upper().startswith("BS"):
        return "BS"
    return "UNKNOWN"


def _infer_title(text: str, source_path: str | None, standard_id: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:30]):
        if "standard" in line.lower() and len(line) > 20:
            title_lines = [line]
            for next_line in lines[index + 1 : index + 3]:
                if len(next_line) > 12 and not SECTION_RE.match(next_line):
                    title_lines.append(next_line)
            return " ".join(title_lines)
    if source_path:
        return Path(source_path).stem.replace("_", " ").replace("-", " ")
    return standard_id


def _infer_document_type(text: str, title: str) -> str:
    combined = f"{title}\n{text[:1000]}".lower()
    if "standard guide" in combined:
        return DocumentType.GUIDE.value
    if "standard practice" in combined:
        return DocumentType.PRACTICE.value
    if "test method" in combined:
        return DocumentType.TEST_METHOD.value
    if "specification" in combined:
        return DocumentType.SPECIFICATION.value
    if "standard" in combined:
        return DocumentType.STANDARD.value
    return DocumentType.OTHER.value


def _infer_year(standard_id: str) -> int | None:
    if ":" in standard_id:
        suffix = standard_id.rsplit(":", 1)[-1]
    else:
        suffix = standard_id.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    if len(suffix) == 4:
        return int(suffix)
    year = int(suffix)
    return 2000 + year if year < 50 else 1900 + year


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
