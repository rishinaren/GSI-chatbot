"""Optional FastAPI app for serving the standards RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from standards_rag.chat import StandardsRagEngine
from standards_rag.pinecone_hybrid import attach_pinecone_index, pinecone_enabled_from_env
from standards_rag.retrieval import InMemoryStandardsStore


def create_app(store: InMemoryStandardsStore | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
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

    engine = StandardsRagEngine(store)
    app = FastAPI(title="Standards RAG Chatbot", version="0.1.0")

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

    return app


app = create_app()
