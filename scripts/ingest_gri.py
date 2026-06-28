"""Ingest GRI standards PDFs into Pinecone and the local JSON index.

GRI documents are identified from their file name (gg4a -> GRI-GG4a) rather than
the inferred ASTM-style designation, since a GRI PDF lists referenced ASTM/ISO
methods in its body. The script:

  1. embeds + upserts each chunk to Pinecone (same index/namespace as ASTM docs),
  2. merges the new documents/chunks into ``STANDARDS_INDEX_PATH`` without
     disturbing the existing corpus (re-runnable / idempotent).

Citations work identically to the ASTM corpus: each document keeps a
``source_path`` (relative to the project root) so the API can serve the PDF at
``/documents/{document_id}/pdf``.

Usage:
    python scripts/ingest_gri.py ["GRI standards"]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # PyMuPDF

from standards_rag.env_bootstrap import default_standards_index_path, load_dotenv_files
from standards_rag.ingestion import load_document_from_pdf
from standards_rag.pinecone_hybrid import (
    PineconeHybridStore,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)
from standards_rag.retrieval import InMemoryStandardsStore

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

# Robust extractor: a GRI cover page reads
#   GRI <Type> <code>*  /  Standard <Type> for[:]  /  <title…>  /  This … was developed by the Geosynthetic …
# so the title is the text between the "Standard <Type> [for]" line and that boilerplate
# (or the "Scope" heading). Anchoring on the lead-in skips the header line.
_TYPE_RE = re.compile(r"Standard\s+(Test\s+Method|Practice|Specification|Guide|of\s+Practice)\b", re.IGNORECASE)
_LEADIN_RE = re.compile(r'\s*(for\b|[:“"”])')
_DEVELOPED_RE = re.compile(r"This\s+[\w\s]{0,40}?was\s+developed\s+by\s+the\s+Geosynthetic", re.IGNORECASE)
_SCOPE_RE = re.compile(r"\n\s*1\.?\s*Scope\b|\n\s*Scope\b", re.IGNORECASE)
_QUOTE_CHARS = "“”\"‘’'"


def extract_gri_title(raw_text: str) -> str | None:
    """Pull a clean ``Standard <Type> for "<title>"`` string from a GRI cover page."""
    text = raw_text[:2200]
    intro = next((m for m in _TYPE_RE.finditer(text) if _LEADIN_RE.match(text, m.end())), None)
    if intro is None:
        return None
    grp = intro.group(1)
    phrase = "Standard Practice" if grp.lower() == "of practice" else "Standard " + " ".join(grp.split()).title()
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


def merge_into_index(store: InMemoryStandardsStore, index_path: Path) -> tuple[int, int]:
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        data = {"documents": [], "chunks": []}

    docs = {d["document_id"]: d for d in data.get("documents", [])}
    chunks = {c["chunk_id"]: c for c in data.get("chunks", [])}
    for document in store.documents.values():
        docs[document.document_id] = document.to_dict()
    for chunk in store.chunks.values():
        chunks[chunk.chunk_id] = chunk.to_dict()

    data["documents"] = sorted(docs.values(), key=lambda x: x["document_id"])
    data["chunks"] = sorted(chunks.values(), key=lambda x: x["chunk_id"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return len(data["documents"]), len(data["chunks"])


def main() -> None:
    load_dotenv_files()
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "GRI standards")
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {folder}")

    if not pinecone_enabled_from_env():
        raise SystemExit("Pinecone is not configured (set PINECONE_API_KEY / PINECONE_INDEX).")

    store = PineconeHybridStore(load_pinecone_config_from_env())
    print(f"Ingesting {len(pdfs)} GRI PDFs into Pinecone index "
          f"'{store.config.index_name}'…\n")

    for pdf in pdfs:
        family, ov = overrides_for(pdf.stem)
        with fitz.open(pdf) as handle:
            raw = "\n".join(page.get_text("text") for page in handle)
        document, chunks = load_document_from_pdf(pdf, metadata_overrides=ov)
        title = clean_title(raw, document.title, family, ov["standard_id"])
        document, chunks = load_document_from_pdf(
            pdf, metadata_overrides={**ov, "title": title}
        )
        store.add_document(document, chunks)  # embeds + upserts to Pinecone
        print(f"  {document.standard_id:12} {len(chunks):3} chunks  {title[:60]}")

    index_path = default_standards_index_path()
    total_docs, total_chunks = merge_into_index(store, index_path)
    print(f"\nUpserted {len(store.chunks)} GRI chunks to Pinecone.")
    print(f"Index now holds {total_docs} documents / {total_chunks} chunks → {index_path}")


if __name__ == "__main__":
    main()
