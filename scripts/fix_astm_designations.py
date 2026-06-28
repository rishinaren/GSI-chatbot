"""One-off repair: re-ingest 8 batch-3 ASTM PDFs whose designation was
mis-inferred (the auto-inferrer grabbed ASTM committee codes / an ISO reference
out of the body when the file name carried a parenthetical reapproval year).

For each file we re-ingest with an explicit ``standard_id`` (+ ``issuing_body``)
override, then delete the orphaned bad document/chunks from both the local JSON
index and Pinecone so no duplicate vectors linger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.env_bootstrap import default_standards_index_path, load_dotenv_files
from standards_rag.ingestion import load_document_from_pdf
from standards_rag.pinecone_hybrid import PineconeHybridStore, load_pinecone_config_from_env

from ingest_gri import merge_into_index

BATCH = Path("documents/ASTM/batch 3")

# file name -> corrected standard_id (issuing body forced to ASTM)
CORRECTIONS = {
    "D4533 (2023) - Trap Tear.pdf": "D4533-15(2023)",
    "D4632 (2023) - Grab.pdf": "D4632-15a(2023)",
    "D4833 (2020) - Puncture GM.pdf": "D4833-07(2020)",
    "D5261-10(2024) Mass.pdf": "D5261-10(2024)",
    "D5596-03(2021) Carbon Black Dispersion.pdf": "D5596-03(2021)",
    "D5993-18(2022) GCL Mass.pdf": "D5993-18(2022)",
    "D6637_D6637M-15 Geogrid Single or Multi-rib Tensile.pdf": "D6637-15",
    "D6768-20(2026) GCL Tensile.pdf": "D6768-20(2026)",
}


def main() -> None:
    load_dotenv_files()
    index_path = default_standards_index_path()
    data = json.loads(index_path.read_text(encoding="utf-8"))

    # 1. Identify the orphaned (bad) docs by source file name.
    bad_files = set(CORRECTIONS)
    orphan_doc_ids = {
        d["document_id"]
        for d in data["documents"]
        if Path(d.get("source_path", "")).name in bad_files
    }
    orphan_chunk_ids = [
        c["chunk_id"] for c in data["chunks"] if c["document_id"] in orphan_doc_ids
    ]
    print(f"Orphan docs to remove: {len(orphan_doc_ids)}  "
          f"({len(orphan_chunk_ids)} chunks)")

    # 2. Re-ingest each PDF with the corrected identity (embeds + upserts to Pinecone).
    store = PineconeHybridStore(load_pinecone_config_from_env())
    for fname, std_id in CORRECTIONS.items():
        pdf = BATCH / fname
        document, chunks = load_document_from_pdf(
            pdf, metadata_overrides={"standard_id": std_id, "issuing_body": "ASTM"}
        )
        store.add_document(document, chunks)
        print(f"  {document.standard_id:18} {document.issuing_body:5} "
              f"{len(chunks):3} chunks  {document.title[:48]}")

    # 3. Merge corrected docs into the index.
    merge_into_index(store, index_path)

    # 4. Prune the orphan docs/chunks from the index.
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["documents"] = [d for d in data["documents"] if d["document_id"] not in orphan_doc_ids]
    data["chunks"] = [c for c in data["chunks"] if c["document_id"] not in orphan_doc_ids]
    index_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # 5. Delete the orphan vectors from Pinecone.
    if orphan_chunk_ids:
        store._index.delete(ids=orphan_chunk_ids, namespace=store.config.namespace)
        print(f"Deleted {len(orphan_chunk_ids)} orphan vectors from Pinecone.")

    print(f"Index now holds {len(data['documents'])} documents / "
          f"{len(data['chunks'])} chunks → {index_path}")


if __name__ == "__main__":
    main()
