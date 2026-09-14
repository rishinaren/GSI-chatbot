"""Ingestion and routing helpers for non-normative design books."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from standards_rag.ingestion import PageText, compute_sha256, infer_document_metadata
from standards_rag.models import DocumentType, SourceChunk, StandardDocument

DESIGN_GUIDANCE_NAMESPACE = "design-guidance"
DESIGN_GUIDANCE_CORPUS = "design_guidance"

_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+([A-Za-z][^\n]{2,140})$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DesignBookSpec:
    volume: int
    physical_start_page: int
    printed_start_page: int
    printed_end_page: int
    outline_pages: frozenset[int]

    @property
    def document_id(self) -> str:
        return f"designing-with-geosynthetics-6e-volume-{self.volume}"

    @property
    def source_id(self) -> str:
        return f"DWG-6E-V{self.volume}"

    @property
    def title(self) -> str:
        return f"Designing with Geosynthetics, 6th Edition, Volume {self.volume}"


BOOK_SPECS = {
    1: DesignBookSpec(
        volume=1,
        physical_start_page=17,
        printed_start_page=1,
        printed_end_page=508,
        outline_pages=frozenset({1, 88, 89, 375, 470}),
    ),
    2: DesignBookSpec(
        volume=2,
        physical_start_page=11,
        printed_start_page=509,
        printed_end_page=914,
        outline_pages=frozenset({509, 510, 511, 750, 806, 835}),
    ),
}


def design_book_volume(path: str | Path) -> int:
    name = Path(path).name.lower()
    match = re.search(r"(?:vol(?:ume)?[.\s_-]*)([12])\b", name)
    if not match:
        raise ValueError(f"Could not determine book volume from file name: {Path(path).name}")
    return int(match.group(1))


def load_design_guidance_pdf(
    path: str | Path,
    *,
    volume: int | None = None,
    max_chars: int = 1200,
) -> tuple[StandardDocument, list[SourceChunk]]:
    """Load one DWG volume with printed-page citations and section-aware chunks."""

    pdf_path = Path(path)
    spec = BOOK_SPECS[volume or design_book_volume(pdf_path)]
    pages = _extract_content_pages(pdf_path, spec)
    full_text = "\n".join(page.text for page in pages)

    try:
        from standards_rag.env_bootstrap import project_root

        source_path = str(pdf_path.relative_to(project_root()))
    except ValueError:
        source_path = str(pdf_path)

    document = infer_document_metadata(
        full_text,
        source_path=source_path,
        checksum=compute_sha256(pdf_path),
        overrides={
            "document_id": spec.document_id,
            "standard_id": spec.source_id,
            "title": spec.title,
            "issuing_body": "GSI",
            "document_type": DocumentType.BOOK,
            "year": 2012,
            "metadata": {
                "corpus_kind": DESIGN_GUIDANCE_CORPUS,
                "authority": "secondary_guidance",
                "author": "Robert M. Koerner",
                "edition": 6,
                "volume": spec.volume,
            },
        },
    )
    chunks = chunk_design_guidance_pages(document, pages, volume=spec.volume, max_chars=max_chars)
    return document, chunks


def _extract_content_pages(path: Path, spec: DesignBookSpec) -> list[PageText]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the optional 'pdf' dependencies to ingest PDFs.") from exc

    pages: list[PageText] = []
    with fitz.open(path) as document:
        expected_pages = spec.printed_end_page - spec.printed_start_page + 1
        available_pages = len(document) - spec.physical_start_page + 1
        if available_pages != expected_pages:
            raise ValueError(
                f"Unexpected page count for Volume {spec.volume}: "
                f"expected {expected_pages} content pages, found {available_pages}"
            )

        for physical_index in range(spec.physical_start_page - 1, len(document)):
            page = document[physical_index]
            blocks: list[str] = []
            for block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, raw = block[:5]
                del x0, x1, y1
                cleaned = _clean_block(str(raw))
                if not cleaned:
                    continue
                # Running headers and isolated footer page numbers add retrieval
                # noise but no design evidence.
                if y0 < 48:
                    continue
                if y0 > page.rect.height - 42 and re.fullmatch(r"\d+", cleaned):
                    continue
                blocks.append(cleaned)

            printed_page = spec.printed_start_page + (physical_index - spec.physical_start_page + 1)
            if printed_page in spec.outline_pages:
                continue
            pages.append(PageText(page_number=printed_page, text="\n\n".join(blocks)))
    return pages


def _clean_block(raw: str) -> str:
    text = raw.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line).strip()


def _split_block(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > max_chars:
                    pieces.append(current)
                    current = word
                else:
                    current = candidate
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _section_type(heading: str | None, text: str) -> str:
    haystack = f"{heading or ''} {text[:240]}".lower()
    if "example" in haystack:
        return "worked_example"
    if "design" in haystack:
        return "design"
    if "propert" in haystack or "test method" in haystack:
        return "test_context"
    if "construction" in haystack or "installation" in haystack:
        return "construction"
    return "other"


def chunk_design_guidance_pages(
    document: StandardDocument,
    pages: list[PageText],
    *,
    volume: int,
    max_chars: int = 1200,
) -> list[SourceChunk]:
    """Build page-bound chunks so citations use the books' printed page numbers."""

    chunks: list[SourceChunk] = []
    active_section: str | None = None
    active_heading: str | None = None
    skip_chapter: str | None = None

    for page in pages:
        page_parts: list[str] = []

        def flush() -> None:
            nonlocal page_parts
            body = "\n\n".join(page_parts).strip()
            if not body:
                return
            heading_prefix = (
                f"{active_section} {active_heading}\n" if active_section and active_heading else ""
            )
            text = f"{heading_prefix}{body}".strip()
            order = len(chunks)
            chunks.append(
                SourceChunk(
                    chunk_id=f"{document.document_id}:p{page.page_number}-{page.page_number}:c{order}",
                    document_id=document.document_id,
                    text=text,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section=active_section,
                    heading=active_heading,
                    order=order,
                    metadata={
                        "section_number": active_section,
                        "section_title": active_heading,
                        "section_type": _section_type(active_heading, body),
                        "paragraph_index": order,
                        "page_numbers": [page.page_number],
                        "printed_page_label": str(page.page_number),
                        "corpus_kind": DESIGN_GUIDANCE_CORPUS,
                        "authority": "secondary_guidance",
                        "volume": volume,
                        "chapter": active_section.split(".", 1)[0] if active_section else None,
                    },
                )
            )
            page_parts = []

        for block in [part.strip() for part in page.text.split("\n\n") if part.strip()]:
            if block.lower() in {"references", "problems"}:
                flush()
                skip_chapter = active_section.split(".", 1)[0] if active_section else None
                continue

            heading_match = _SECTION_RE.fullmatch(block)
            if heading_match:
                next_chapter = heading_match.group(1).split(".", 1)[0]
                if skip_chapter and next_chapter == skip_chapter:
                    continue
                skip_chapter = None
                flush()
                active_section = heading_match.group(1)
                active_heading = heading_match.group(2).strip()
                continue

            if skip_chapter:
                continue

            for piece in _split_block(block, max_chars):
                projected = sum(len(part) for part in page_parts) + len(piece) + 2
                if page_parts and projected > max_chars:
                    flush()
                page_parts.append(piece)
        flush()

    return chunks


def design_chapters_for_question(question: str) -> set[str]:
    lowered = question.lower()
    chapters: set[str] = set()
    mappings = (
        ("2", ("geotextile", "separation", "filtration", "silt fence")),
        ("3", ("geogrid", "mse wall", "reinforced wall")),
        ("4", ("geonet", "planar drainage", "transmissivity")),
        ("5", ("geomembrane", "pond liner", "landfill liner", "landfill cover", "anchor trench")),
        ("6", ("gcl", "geosynthetic clay liner", "clay liner")),
        ("7", ("geofoam", "lightweight fill", "compressible inclusion")),
        ("8", ("geocomposite", "wick drain", "sheet drain", "edge drain")),
    )
    for chapter, terms in mappings:
        if any(term in lowered for term in terms):
            chapters.add(chapter)
    return chapters


def design_query_with_context(question: str) -> tuple[str, set[str]]:
    chapters = design_chapters_for_question(question)
    if not chapters:
        return question, chapters
    chapter_labels = {
        "2": "geotextile design",
        "3": "geogrid design",
        "4": "geonet drainage design",
        "5": "geomembrane design",
        "6": "geosynthetic clay liner GCL design",
        "7": "geofoam design",
        "8": "geocomposite design",
    }
    context = " ".join(chapter_labels[chapter] for chapter in sorted(chapters))
    return f"{question}\nDesigning with Geosynthetics context: {context}", chapters


def design_sections_for_question(question: str) -> set[str]:
    lowered = question.lower()
    sections: set[str] = set()
    mappings = (
        ("2.5", ("geotextile separation", "geotextile separator")),
        ("2.6", ("roadway reinforcement", "unpaved road", "paved road")),
        ("2.8", ("geotextile filtration", "filter behind", "underdrain")),
        ("3.2", ("geogrid", "mse wall", "reinforced wall")),
        ("4.2", ("geonet", "planar drainage")),
        ("5.3", ("pond liner", "liquid containment", "reservoir liner")),
        ("5.6", ("landfill liner", "solid waste liner")),
        ("5.7", ("landfill cover", "landfill closure")),
        ("5.11", ("geomembrane seam", "seaming")),
        ("6.3", ("gcl", "geosynthetic clay liner")),
        ("8.", ("geocomposite", "wick drain", "sheet drain", "edge drain")),
    )
    for section, terms in mappings:
        if any(term in lowered for term in terms):
            sections.add(section)
    if "geofoam" in lowered or "lightweight fill" in lowered or "compressible inclusion" in lowered:
        if any(term in lowered for term in ("settlement", "embankment", "lightweight fill")):
            sections.add("7.2.1")
        elif "compressible inclusion" in lowered:
            sections.add("7.2.2")
        else:
            sections.add("7.2")
    return sections


def is_design_guidance_question(question: str, focus: str) -> bool:
    """Route explicit book/design questions while treating focus as a default."""

    lowered = question.lower()
    explicit_book = any(
        phrase in lowered
        for phrase in (
            "designing with geosynthetics",
            "koerner",
            "design guidance",
            "design book",
            "volume 1",
            "volume 2",
            "dwg-6e",
        )
    )
    if explicit_book:
        return True

    explicit_standard = bool(
        re.search(r"\b(?:astm|iso|gri)\b|\b[a-z]\d{3,5}(?:[-/]\d+)?\b", lowered)
    )
    design_terms = {
        "design",
        "designing",
        "select",
        "selection",
        "size",
        "sizing",
        "factor of safety",
        "reinforcement",
        "separation",
        "filtration",
        "pond liner",
        "landfill liner",
        "landfill cover",
        "mse wall",
        "slope stability",
        "anchor trench",
        "lightweight fill",
        "geofoam",
    }
    has_design_intent = any(term in lowered for term in design_terms)
    if explicit_standard and not has_design_intent:
        return False
    if focus == "design":
        return True
    if focus == "both":
        return has_design_intent
    return False
