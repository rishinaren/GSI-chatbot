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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # PyMuPDF

from standards_rag.env_bootstrap import default_standards_index_path, load_dotenv_files
from standards_rag.ingestion import load_document_from_pdf

# GRI identity/title parsing and index merging live in the package so the admin
# upload dashboard ingests a GRI PDF exactly the way this script does. Re-exported
# here because ingest_astm.py / fix_astm_designations.py import them from this module.
from standards_rag.library import (  # noqa: F401
    FAMILY_NAME,
    clean_title,
    extract_gri_title,
    gri_code,
    merge_into_index,
    overrides_for,
)
from standards_rag.pinecone_hybrid import (
    PineconeHybridStore,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)


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
