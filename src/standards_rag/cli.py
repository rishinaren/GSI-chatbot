"""Command line helpers for local ingestion and question answering."""

from __future__ import annotations

import argparse
from pathlib import Path

from standards_rag.chat import StandardsRagEngine
from standards_rag.ingestion import load_document_from_pdf, load_document_from_text
from standards_rag.openai_answer import build_openai_answer_rewriter_from_env
from standards_rag.pinecone_hybrid import (
    PineconeHybridStore,
    attach_pinecone_index,
    load_pinecone_config_from_env,
    pinecone_enabled_from_env,
)
from standards_rag.retrieval import InMemoryStandardsStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="standards-rag")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Build a local JSON index from PDFs or text files.")
    ingest.add_argument("source", type=Path, help="PDF, text file, or folder to ingest.")
    ingest.add_argument("--out", type=Path, default=Path("data/index/standards-index.json"))
    ingest.add_argument(
        "--pinecone",
        action="store_true",
        help="Also embed and upsert chunks to Pinecone (requires pinecone extra and env config).",
    )

    ask = subcommands.add_parser("ask", help="Ask a question against a local JSON index.")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path("data/index/standards-index.json"))
    ask.add_argument("--conversation-id", default="default")
    ask.add_argument("--units", choices=["si", "metric", "imperial", "us", "english"])
    ask.add_argument(
        "--pinecone",
        action="store_true",
        help="Use Pinecone vector retrieval when env vars are configured.",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        _ingest(args.source, args.out, args.pinecone)
    elif args.command == "ask":
        _ask(args.index, args.question, args.conversation_id, args.units, args.pinecone)


def _ingest(source: Path, output: Path, pinecone: bool) -> None:
    use_pinecone = pinecone or pinecone_enabled_from_env()
    if use_pinecone:
        store: InMemoryStandardsStore = PineconeHybridStore(load_pinecone_config_from_env())
    else:
        store = InMemoryStandardsStore()
    paths = _source_paths(source)
    if not paths:
        raise SystemExit(f"No supported source files found under {source}")

    for path in paths:
        if path.suffix.lower() == ".pdf":
            document, chunks = load_document_from_pdf(path)
        else:
            document, chunks = load_document_from_text(
                path.read_text(encoding="utf-8"), source_path=str(path)
            )
        store.add_document(document, chunks)
        print(f"Ingested {document.standard_id} ({len(chunks)} chunks)")

    store.save_json(output)
    print(f"Wrote {len(store.documents)} documents and {len(store.chunks)} chunks to {output}")

    if use_pinecone:
        print("Embedded chunks and upserted vectors to Pinecone during ingestion.")


def _ask(index: Path, question: str, conversation_id: str, units: str | None, pinecone: bool) -> None:
    if not index.exists():
        raise SystemExit(f"Index not found: {index}")
    store = InMemoryStandardsStore.load_json(index)
    if pinecone or pinecone_enabled_from_env():
        store = attach_pinecone_index(store)

    try:
        answer_rewriter = build_openai_answer_rewriter_from_env()
    except Exception:
        answer_rewriter = None

    response = StandardsRagEngine(store, answer_rewriter=answer_rewriter).ask(
        question,
        conversation_id=conversation_id,
        unit_preference=units,
    )
    print(response.answer)


def _source_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in {".pdf", ".txt", ".md"}:
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        )
    return []


if __name__ == "__main__":
    main()
