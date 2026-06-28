"""Ingest ASTM/ISO standards PDFs into Pinecone and the local JSON index.

Unlike ``ingest_gri.py`` (which forces a ``GRI-`` identity keyed off the file
name), ASTM documents keep their *inferred* designation: the standard id,
issuing body, title and document id are derived from the PDF text / file name by
``load_document_from_pdf`` (e.g. ``D4595-24`` -> ``astm-d4595-24``, body
``ASTM``). The script:

  1. embeds + upserts each chunk to Pinecone (same index/namespace as the rest
     of the corpus),
  2. merges the new documents/chunks into ``STANDARDS_INDEX_PATH`` without
     disturbing the existing corpus (re-runnable / idempotent — re-ingesting a
     PDF overwrites its own entry by document id and leaves every other doc
     untouched).

Citations work identically to the rest of the corpus: each document keeps a
``source_path`` (relative to the project root) so the API can serve the PDF at
``/documents/{document_id}/pdf``.

Usage:
    python scripts/ingest_astm.py "documents/ASTM/batch 3"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.env_bootstrap import default_standards_index_path, load_dotenv_files
from standards_rag.ingestion import load_document_from_pdf
from standards_rag.pinecone_hybrid import (
    PineconeHybridStore,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)

# merge_into_index is identity-agnostic, so reuse the GRI script's implementation.
from ingest_gri import merge_into_index


def main() -> None:
    load_dotenv_files()
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "documents/ASTM/batch 3")
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {folder}")

    if not pinecone_enabled_from_env():
        raise SystemExit("Pinecone is not configured (set PINECONE_API_KEY / PINECONE_INDEX).")

    store = PineconeHybridStore(load_pinecone_config_from_env())
    print(f"Ingesting {len(pdfs)} ASTM PDFs from '{folder}' into Pinecone index "
          f"'{store.config.index_name}'…\n")

    for pdf in pdfs:
        document, chunks = load_document_from_pdf(pdf)
        store.add_document(document, chunks)  # embeds + upserts to Pinecone
        print(f"  {document.standard_id:14} {len(chunks):3} chunks  {document.title[:56]}")

    index_path = default_standards_index_path()
    total_docs, total_chunks = merge_into_index(store, index_path)
    print(f"\nUpserted {len(store.chunks)} ASTM chunks to Pinecone.")
    print(f"Index now holds {total_docs} documents / {total_chunks} chunks → {index_path}")


if __name__ == "__main__":
    main()
