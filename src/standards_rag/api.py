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
    register_membership_validator,
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
from standards_rag.admin_access import (
    AdminAccessError,
    build_admin_registry_from_env,
)
from standards_rag.admin_overview import build_overview, video_rows
from standards_rag.attachments import (
    MAX_ATTACHMENT_BYTES,
    AttachmentError,
    AttachmentStore,
    read_attachment,
)
from standards_rag.feedback import FeedbackError, build_feedback_log_from_env
from standards_rag.library import (
    KNOWN_BODIES,
    MAX_UPLOAD_BYTES,
    DocumentLibrary,
    LibraryError,
)
from standards_rag.membership import (
    MembershipError,
    build_membership_service_from_env,
)
from standards_rag.openai_answer import (
    build_openai_answer_rewriter_from_env,
    build_title_generator_from_env,
    openai_rewriter_enabled,
)
from standards_rag.pinecone_hybrid import attach_pinecone_index, pinecone_enabled_from_env
from standards_rag.retrieval import InMemoryStandardsStore, resolve_document_pdf_path
from standards_rag.video import build_video_store_from_env

logger = logging.getLogger(__name__)

load_dotenv_files()

try:  # Optional at import time; create_app still validates runtime deps.
    # These two are referenced in route *annotations*. With `from __future__ import
    # annotations` those are strings, so FastAPI resolves them against this module's
    # globals - importing them inside create_app would leave the forward ref undefined.
    from fastapi import Request as FastAPIRequest
    from fastapi import UploadFile
except Exception:  # pragma: no cover - only when api deps missing
    class FastAPIRequest:  # type: ignore[no-redef]
        pass

    class UploadFile:  # type: ignore[no-redef]
        pass


def _allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    defaults = ["http://localhost:5173", "http://127.0.0.1:5173"]
    if not raw:
        return defaults
    return defaults + [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(store: InMemoryStandardsStore | None = None) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException
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

    try:
        title_generator = build_title_generator_from_env()
    except Exception as exc:  # noqa: BLE001 - titles are best-effort
        title_generator = None
        logger.warning("Title generator unavailable: %s", exc)

    video_store = build_video_store_from_env()
    if len(video_store):
        logger.info("Loaded %d video transcripts", len(video_store))

    auth_config = load_auth_config_from_env()

    # Membership layer (GSI-member + subscriber sign-in, subscription checkout).
    # Its session tokens are accepted by the shared bearer-token auth below.
    membership_service = build_membership_service_from_env()

    def _membership_validator(token: str) -> AuthenticatedUser | None:
        principal = membership_service.validate_token(token)
        if principal is None:
            return None
        return AuthenticatedUser(
            user_id=principal.user_id,
            email=principal.email,
            organization_id=principal.organization_id,
        )

    register_membership_validator(_membership_validator)

    conversation_store = build_conversation_store_from_env()
    # A question's attachments live in this process only - never embedded, never
    # written to the corpus. See ``attachments`` for why.
    attachment_store = AttachmentStore()
    feedback_log = build_feedback_log_from_env(conversation_store)
    engine = StandardsRagEngine(
        store,
        answer_rewriter=answer_rewriter,
        conversation_store=conversation_store,
        video_store=video_store,
        title_generator=title_generator,
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
            "videos": len(video_store),
            "answer_rewriter_active": answer_rewriter is not None,
            "title_generator_active": title_generator is not None,
            "openai_rewriter_flag": openai_rewriter_enabled(),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "auth": auth_public_config(auth_config),
            "membership": membership_service.public_config(),
        }

    @app.get("/auth/config")
    def auth_config_endpoint() -> dict[str, object]:
        return auth_public_config(auth_config)

    @app.get("/auth/membership/config")
    def membership_config_endpoint() -> dict[str, object]:
        return membership_service.public_config()

    @app.post("/auth/member/login")
    def member_login(payload: dict[str, Any]) -> dict[str, object]:
        """GSI members: validate against the member directory, no email code."""
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))
        if not email or not password:
            raise HTTPException(status_code=400, detail="email and password are required")
        try:
            return membership_service.member_login(email, password)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/subscriber/login")
    def subscriber_login(payload: dict[str, Any]) -> dict[str, object]:
        """Subscribers: verify password, then email a 2FA code (see /verify)."""
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))
        if not email or not password:
            raise HTTPException(status_code=400, detail="email and password are required")
        try:
            return membership_service.subscriber_login_start(email, password)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/subscriber/verify")
    def subscriber_verify(payload: dict[str, Any]) -> dict[str, object]:
        """Subscribers: exchange the emailed code for a session token."""
        email = str(payload.get("email", "")).strip()
        code = str(payload.get("code", "")).strip()
        if not email or not code:
            raise HTTPException(status_code=400, detail="email and code are required")
        try:
            return membership_service.subscriber_verify(email, code)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/subscriber/resend-code")
    def subscriber_resend_code(payload: dict[str, Any]) -> dict[str, object]:
        email = str(payload.get("email", "")).strip()
        if not email:
            raise HTTPException(status_code=400, detail="email is required")
        try:
            return membership_service.subscriber_resend_code(email)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/subscriber/confirm")
    def subscriber_confirm(payload: dict[str, Any]) -> dict[str, object]:
        """New subscribers: confirm the Cognito signup email code → session token."""
        email = str(payload.get("email", "")).strip()
        code = str(payload.get("code", "")).strip()
        if not email or not code:
            raise HTTPException(status_code=400, detail="email and code are required")
        try:
            return membership_service.subscriber_confirm_signup(email, code)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/auth/subscriber/resend-signup-code")
    def subscriber_resend_signup_code(payload: dict[str, Any]) -> dict[str, object]:
        email = str(payload.get("email", "")).strip()
        if not email:
            raise HTTPException(status_code=400, detail="email is required")
        try:
            return membership_service.subscriber_resend_signup_code(email)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/billing/subscribe")
    def billing_subscribe(payload: dict[str, Any]) -> dict[str, object]:
        """New subscribers: run the (placeholder) checkout, then email a code."""
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))
        if not email or not password:
            raise HTTPException(status_code=400, detail="email and password are required")
        payment = payload.get("payment") if isinstance(payload.get("payment"), dict) else {}
        name = str(payload.get("name", "")).strip() or None
        try:
            return membership_service.subscribe(email, password, payment or {}, name=name)
        except MembershipError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

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
                    "pinned": record.pinned,
                    "project_id": record.project_id,
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

    @app.patch("/conversations/{conversation_id}")
    def update_conversation(
        conversation_id: str, payload: dict[str, Any], request: FastAPIRequest
    ) -> dict[str, object]:
        user = effective_user(request)
        if "pinned" not in payload and "project_id" not in payload:
            raise HTTPException(status_code=400, detail="pinned or project_id is required")

        record = None
        if "pinned" in payload:
            record = conversation_store.set_pinned(
                user.user_id, conversation_id, bool(payload.get("pinned"))
            )
            if record is None:
                raise HTTPException(status_code=404, detail="conversation not found")

        if "project_id" in payload:
            raw_project = payload.get("project_id")
            project_id = str(raw_project).strip() if raw_project else None
            try:
                record = conversation_store.set_project(
                    user.user_id, conversation_id, project_id or None
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="project not found") from exc
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

    @app.get("/projects")
    def list_projects(request: FastAPIRequest) -> dict[str, object]:
        user = effective_user(request)
        projects = conversation_store.list_projects(user.user_id)
        conversations = conversation_store.list_conversations(user.user_id, limit=500)
        counts: dict[str, int] = {}
        for conversation in conversations:
            if conversation.project_id:
                counts[conversation.project_id] = counts.get(conversation.project_id, 0) + 1
        return {
            "projects": [
                {
                    "project_id": project.project_id,
                    "name": project.name,
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                    "conversation_count": counts.get(project.project_id, 0),
                }
                for project in projects
            ]
        }

    @app.post("/projects")
    def create_project(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        user = effective_user(request)
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        project = conversation_store.create_project(user.user_id, name=name)
        return project.to_dict()

    @app.patch("/projects/{project_id}")
    def rename_project(
        project_id: str, payload: dict[str, Any], request: FastAPIRequest
    ) -> dict[str, object]:
        user = effective_user(request)
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        project = conversation_store.rename_project(user.user_id, project_id, name)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project.to_dict()

    @app.delete("/projects/{project_id}")
    def delete_project(project_id: str, request: FastAPIRequest) -> dict[str, bool]:
        user = effective_user(request)
        deleted = conversation_store.delete_project(user.user_id, project_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="project not found")
        return {"deleted": True}

    # --- Admin document library -------------------------------------------
    # A GSI administrator adds a standard here and it is answerable on the next
    # question - no redeploy - because ``library`` holds the same store object
    # the chat engine queries.
    library = DocumentLibrary(store)
    # Root admins come from GSI_ADMIN_EMAILS; further admins are granted at
    # runtime from inside the library itself.
    admins = build_admin_registry_from_env(membership_service.member_directory)

    def require_admin(request: FastAPIRequest) -> AuthenticatedUser:
        user = effective_user(request)
        if not admins.is_admin(user.email):
            raise HTTPException(
                status_code=403,
                detail="Your account does not have access to the document library.",
            )
        return user

    def read_upload(upload: UploadFile) -> bytes:
        # Read one byte past the cap so an oversized file is rejected by size
        # rather than pulled entirely into memory first.
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            limit = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"That file is larger than {limit} MB, which is bigger than we can take. "
                    "If it is a bundle of several standards, please upload them one at a time."
                ),
            )
        return data

    @app.get("/admin/config")
    def admin_config(request: FastAPIRequest) -> dict[str, object]:
        """Whether this signed-in user may manage the library (drives the UI entry point)."""
        user = effective_user(request)
        return {
            "is_admin": admins.is_admin(user.email),
            "email": user.email,
            "publishers": list(KNOWN_BODIES),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    @app.get("/admin/overview")
    def admin_overview(request: FastAPIRequest) -> dict[str, object]:
        """The admin landing page: how big the knowledge base is, and its health."""
        require_admin(request)
        return build_overview(
            library,
            store,
            video_store,
            answer_writing_active=answer_rewriter is not None,
        )

    @app.get("/admin/videos")
    def admin_list_videos(request: FastAPIRequest) -> dict[str, object]:
        """Every walkthrough video and the library documents it is linked to."""
        require_admin(request)
        rows = video_rows(store, video_store)
        return {
            "videos": rows,
            "video_count": len(rows),
            "linked_count": sum(1 for row in rows if row["linked"]),
        }

    @app.get("/admin/documents")
    def admin_list_documents(request: FastAPIRequest) -> dict[str, object]:
        require_admin(request)
        return {"documents": library.documents(), **library.summary()}

    @app.post("/admin/documents/analyze")
    def admin_analyze_document(
        request: FastAPIRequest, file: UploadFile = File(...)
    ) -> dict[str, object]:
        """Read an uploaded PDF and guess its details. Changes nothing."""
        require_admin(request)
        data = read_upload(file)
        try:
            return library.analyze(data, file.filename or "")
        except LibraryError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/admin/documents")
    def admin_add_document(
        request: FastAPIRequest,
        file: UploadFile = File(...),
        standard_id: str = Form(""),
        title: str = Form(""),
        issuing_body: str = Form(""),
        year: str = Form(""),
    ) -> dict[str, object]:
        user = require_admin(request)
        data = read_upload(file)
        try:
            return library.add(
                data,
                file.filename or "",
                standard_id=standard_id,
                title=title,
                issuing_body=issuing_body,
                year=year,
                added_by=user.email,
            )
        except LibraryError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/admin/people")
    def admin_list_people(request: FastAPIRequest) -> dict[str, object]:
        user = require_admin(request)
        return {"people": admins.people(), "you": user.email}

    @app.post("/admin/people")
    def admin_grant_person(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        """Give an existing account access to the library. Does not create accounts."""
        user = require_admin(request)
        email = str(payload.get("email", "")).strip()
        if not email:
            raise HTTPException(status_code=400, detail="email is required")
        try:
            return admins.grant(email, granted_by=user.email)
        except AdminAccessError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.delete("/admin/people/{email}")
    def admin_revoke_person(email: str, request: FastAPIRequest) -> dict[str, bool]:
        user = require_admin(request)
        try:
            admins.revoke(unquote(email), revoked_by=user.email)
        except AdminAccessError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {"removed": True}

    @app.post("/videos/search")
    def videos_search(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        effective_user(request)
        query = str(payload.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        top_k_raw = payload.get("top_k", 3)
        try:
            top_k = max(1, min(int(top_k_raw), 10))
        except (TypeError, ValueError):
            top_k = 3
        matches = video_store.search(query, top_k=top_k, min_score=0.05)
        return {"videos": [match.to_dict() for match in matches]}

    @app.post("/chat/attachments")
    def chat_add_attachment(
        request: FastAPIRequest, file: UploadFile = File(...)
    ) -> dict[str, object]:
        """Read a PDF the asker brought, ready to be quoted alongside the library.

        Nothing here is ingested: no embedding, no Pinecone write, no S3 copy. The
        extracted text stays in this process, owned by this user, until it expires.
        """
        user = effective_user(request)
        data = file.file.read(MAX_ATTACHMENT_BYTES + 1)
        if len(data) > MAX_ATTACHMENT_BYTES:
            limit = MAX_ATTACHMENT_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"That file is larger than {limit} MB. Please attach a smaller PDF, "
                    "or the section of it you want to ask about."
                ),
            )
        try:
            attachment = read_attachment(data, file.filename or "", owner_id=user.user_id)
        except AttachmentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return attachment_store.add(attachment).to_dict()

    @app.post("/chat")
    def chat(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        question = str(payload.get("question", "")).strip()
        user = effective_user(request)

        raw_ids = payload.get("attachment_ids")
        attachments = []
        for attachment_id in raw_ids if isinstance(raw_ids, list) else []:
            attachment = attachment_store.get(str(attachment_id), owner_id=user.user_id)
            if attachment is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "That attachment is no longer available. Please attach the file "
                        "again and resend your question."
                    ),
                )
            attachments.append(attachment)

        if not question and not attachments:
            raise HTTPException(status_code=400, detail="question is required")

        response = engine.ask(
            question,
            conversation_id=str(payload.get("conversation_id", "default")),
            unit_preference=payload.get("unit_preference"),
            user_id=user.user_id,
            attachments=attachments,
        )
        return response.to_dict()

    @app.post("/feedback")
    def submit_feedback(payload: dict[str, Any], request: FastAPIRequest) -> dict[str, object]:
        """Rate an answer, optionally saying what was wrong with it."""
        user = effective_user(request)
        try:
            record = feedback_log.record(
                user_id=user.user_id,
                user_email=user.email or "",
                conversation_id=str(payload.get("conversation_id", "")).strip(),
                rating=str(payload.get("rating", "")).strip(),
                comment=str(payload.get("comment", "")),
                answer=str(payload.get("answer", "")),
                # Passed back when a comment follows the thumbs-down that opened
                # the box, so the two land on one row rather than two.
                feedback_id=str(payload.get("feedback_id", "")),
                created_at=str(payload.get("created_at", "")),
            )
        except FeedbackError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {
            "recorded": True,
            "feedback_id": record.feedback_id,
            "created_at": record.created_at,
        }

    @app.get("/admin/feedback")
    def admin_feedback(request: FastAPIRequest) -> dict[str, object]:
        """Every rated answer, grouped by the conversation it came from."""
        require_admin(request)
        return {"conversations": feedback_log.conversations(), **feedback_log.summary()}

    @app.get("/documents/{document_id}/pdf")
    def document_pdf(document_id: str, request: FastAPIRequest) -> FileResponse:
        # PDFs open via top-level browser navigation (a new tab), so no Authorization
        # header is sent. Accept the access token as a query param for this route.
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            token = request.query_params.get("token") or request.query_params.get("access_token")
            if token:
                auth_header = f"Bearer {token}"
        try:
            user = authenticate_bearer_token(auth_header, config=auth_config)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        if user is None and auth_config.required and auth_config.enabled:
            raise HTTPException(status_code=401, detail="Authentication required.")

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
