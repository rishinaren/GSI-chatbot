"""Optional FastAPI app for serving the standards RAG chatbot."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import unquote

from standards_rag.auth import (
    AuthError,
    AuthenticatedUser,
    auth_public_config,
    authenticate_bearer_token,
    confirm_sign_up,
    load_auth_config_from_env,
    login_with_password,
    resend_confirmation_code,
    sign_up_with_password,
)
from standards_rag.chat import StandardsRagEngine
from standards_rag.conversation_store import build_conversation_store_from_env
from standards_rag.env_bootstrap import (
    default_standards_index_path,
    load_dotenv_files,
    sync_runtime_assets_from_s3,
)
from standards_rag.openai_answer import build_openai_answer_rewriter_from_env, openai_rewriter_enabled
from standards_rag.pinecone_hybrid import attach_pinecone_index, pinecone_enabled_from_env
from standards_rag.retrieval import InMemoryStandardsStore, resolve_document_pdf_path

logger = logging.getLogger(__name__)

load_dotenv_files()

try:  # Optional at import time; create_app still validates runtime deps.
    from fastapi import Request as FastAPIRequest
except Exception:  # pragma: no cover - only when api deps missing
    class FastAPIRequest:  # type: ignore[no-redef]
        pass


def _allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    defaults = ["http://localhost:5173", "http://127.0.0.1:5173"]
    if not raw:
        return defaults
    return defaults + [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(store: InMemoryStandardsStore | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install the optional 'api' dependencies to serve the API.") from exc

    if store is None:
        try:
            downloaded = sync_runtime_assets_from_s3()
            if downloaded:
                logger.info("Synced %d runtime asset files from S3", downloaded)
        except Exception as exc:
            logger.warning("Could not sync runtime assets from S3: %s", exc)
        index_path = default_standards_index_path()
        if index_path.exists():
            store = InMemoryStandardsStore.load_json(index_path)
            logger.info("Loaded standards index from %s (%d documents)", index_path, len(store.documents))
        else:
            store = InMemoryStandardsStore()
            logger.warning(
                "No index at %s — API started with an empty collection. "
                "Ingest documents or set STANDARDS_INDEX_PATH in .env",
                index_path,
            )
        if pinecone_enabled_from_env():
            store = attach_pinecone_index(store)

    try:
        answer_rewriter = build_openai_answer_rewriter_from_env()
    except Exception as exc:
        answer_rewriter = None
        if openai_rewriter_enabled():
            logger.warning("OpenAI rewriter requested but not available: %s", exc)
        else:
            logger.debug("OpenAI rewriter disabled (USE_OPENAI_ANSWER_REWRITER not set).")

    auth_config = load_auth_config_from_env()
    conversation_store = build_conversation_store_from_env()
    engine = StandardsRagEngine(
        store,
        answer_rewriter=answer_rewriter,
        conversation_store=conversation_store,
    )
    app = FastAPI(title="Standards RAG Chatbot", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def current_user(request: FastAPIRequest) -> AuthenticatedUser | None:
        try:
            return authenticate_bearer_token(request.headers.get("Authorization"), config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    def effective_user(request: FastAPIRequest) -> AuthenticatedUser:
        user = current_user(request)
        if user is not None:
            return user
        if auth_config.required and auth_config.enabled:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return AuthenticatedUser(user_id="local-dev-user", email="dev@local")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "documents": len(store.documents),
            "chunks": len(store.chunks),
            "answer_rewriter_active": answer_rewriter is not None,
            "openai_rewriter_flag": openai_rewriter_enabled(),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "auth": auth_public_config(auth_config),
        }

    @app.get("/auth/config")
    def auth_config_endpoint() -> dict[str, object]:
        return auth_public_config(auth_config)

    @app.post("/auth/login")
    def auth_login(payload: dict[str, Any]) -> dict[str, str]:
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))
        if not email or not password:
            raise HTTPException(status_code=400, detail="email and password are required")
        try:
            return login_with_password(email, password, config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/signup")
    def auth_signup(payload: dict[str, Any]) -> dict[str, object]:
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))
        if not email or not password:
            raise HTTPException(status_code=400, detail="email and password are required")
        try:
            return sign_up_with_password(email, password, config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/confirm")
    def auth_confirm(payload: dict[str, Any]) -> dict[str, bool]:
        email = str(payload.get("email", "")).strip()
        confirmation_code = str(payload.get("confirmation_code", "")).strip()
        if not email or not confirmation_code:
            raise HTTPException(status_code=400, detail="email and confirmation_code are required")
        try:
            return confirm_sign_up(email, confirmation_code, config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/resend-confirmation")
    def auth_resend_confirmation(payload: dict[str, Any]) -> dict[str, str]:
        email = str(payload.get("email", "")).strip()
        if not email:
            raise HTTPException(status_code=400, detail="email is required")
        try:
            return resend_confirmation_code(email, config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/conversations")
    def list_conversations(request: FastAPIRequest) -> dict[str, object]:
        user = effective_user(request)
        records = conversation_store.list_conversations(user.user_id)
        return {
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "title": record.title,
                    "updated_at": record.updated_at,
                    "created_at": record.created_at,
                    "message_count": len(record.messages),
                }
                for record in records
            ]
        }

    @app.post("/conversations")
    def create_conversation(
        request: FastAPIRequest, payload: dict[str, Any] | None = None
    ) -> dict[str, object]:
        user = effective_user(request)
        payload = payload or {}
        record = conversation_store.create_conversation(
            user.user_id,
            title=str(payload.get("title") or "New chat"),
            organization_id=user.organization_id,
            unit_preference=payload.get("unit_preference"),
        )
        return record.to_dict()

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: FastAPIRequest) -> dict[str, object]:
        user = effective_user(request)
        record = conversation_store.get_conversation(user.user_id, conversation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return record.to_dict()

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, request: FastAPIRequest) -> dict[str, bool]:
        user = effective_user(request)
        deleted = conversation_store.delete_conversation(user.user_id, conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"deleted": True}

    @app.post("/chat")
    def chat(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        user = effective_user(request)
        response = engine.ask(
            question,
            conversation_id=str(payload.get("conversation_id", "default")),
            unit_preference=payload.get("unit_preference"),
            user_id=user.user_id,
        )
        return response.to_dict()

    @app.get("/documents/{document_id}/pdf")
    def document_pdf(document_id: str, request: FastAPIRequest) -> FileResponse:
        effective_user(request)

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
