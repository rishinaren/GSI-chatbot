"""Optional FastAPI app for serving the standards RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from standards_rag.chat import StandardsRagEngine
from standards_rag.openai_answer import build_openai_answer_rewriter_from_env
from standards_rag.pinecone_hybrid import attach_pinecone_index, pinecone_enabled_from_env
from standards_rag.retrieval import InMemoryStandardsStore, resolve_document_pdf_path


def create_app(store: InMemoryStandardsStore | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install the optional 'api' dependencies to serve the API.") from exc

    if store is None:
        index_path = Path(os.getenv("STANDARDS_INDEX_PATH", "data/index/standards-index.json"))
        if index_path.exists():
            store = InMemoryStandardsStore.load_json(index_path)
        else:
            store = InMemoryStandardsStore()
        if pinecone_enabled_from_env():
            store = attach_pinecone_index(store)

    try:
        answer_rewriter = build_openai_answer_rewriter_from_env()
    except Exception:
        answer_rewriter = None

    engine = StandardsRagEngine(store, answer_rewriter=answer_rewriter)
    app = FastAPI(title="Standards RAG Chatbot", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "documents": len(store.documents), "chunks": len(store.chunks)}

    @app.post("/chat")
    def chat(payload: dict[str, Any]) -> dict[str, object]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        response = engine.ask(
            question,
            conversation_id=str(payload.get("conversation_id", "default")),
            unit_preference=payload.get("unit_preference"),
        )
        return response.to_dict()

    @app.get("/documents/{document_id}/pdf")
    def document_pdf(document_id: str) -> FileResponse:
        doc = store.documents.get(unquote(document_id))
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        path = resolve_document_pdf_path(doc)
        if path is None:
            raise HTTPException(status_code=404, detail="PDF not available for this document")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
        )

    return app


app = create_app()
