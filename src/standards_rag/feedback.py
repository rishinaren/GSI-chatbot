"""What people thought of an answer, and the turn that produced it.

A thumbs-down is only useful with the exchange attached: an administrator needs to
read the question, the answer it got, and what the person said was wrong with it,
in that order. So a feedback row carries all three, resolved **server-side** from
the saved conversation rather than trusted from the browser - the stored answer is
the authoritative one, and the question is whatever preceded it.

Rows live in the shared conversations table under a fixed partition, the same way
``admin_access`` stores granted admins and projects reuse a prefixed sort key. No
new table, no new IAM resource.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Real user ids are "member:<email>" / "subscriber:<email>" / a Cognito sub, so a
# fixed partition of this shape cannot collide with one.
FEEDBACK_PARTITION = "__feedback__"
FEEDBACK_SORT_KEY_PREFIX = "FEEDBACK#"

RATINGS = ("up", "down")
MAX_COMMENT_CHARS = 2000
# The exchange is stored for reading, not for re-answering, so it is capped. A
# long answer still shows the administrator what went wrong.
MAX_EXCERPT_CHARS = 6000


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class FeedbackError(Exception):
    """A problem worth showing to the person leaving the feedback, in their words."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    conversation_id: str
    conversation_title: str
    rating: str
    comment: str = ""
    question: str = ""
    answer: str = ""
    user_email: str = ""
    created_at: str = field(default_factory=_utc_now_iso)

    def to_item(self) -> dict[str, Any]:
        return {
            "user_id": FEEDBACK_PARTITION,
            # Time-ordered sort key, so a single query comes back newest-last with
            # no client-side sort and no secondary index.
            "conversation_id": f"{FEEDBACK_SORT_KEY_PREFIX}{self.created_at}#{self.feedback_id}",
            "record_type": "feedback",
            "feedback_id": self.feedback_id,
            "source_conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "rating": self.rating,
            "comment": self.comment,
            "question": self.question,
            "answer": self.answer,
            "user_email": self.user_email,
            "created_at": self.created_at,
        }

    @classmethod
    def from_item(cls, data: dict[str, Any]) -> "FeedbackRecord":
        return cls(
            feedback_id=str(data.get("feedback_id") or ""),
            conversation_id=str(data.get("source_conversation_id") or ""),
            conversation_title=str(data.get("conversation_title") or ""),
            rating=str(data.get("rating") or "down"),
            comment=str(data.get("comment") or ""),
            question=str(data.get("question") or ""),
            answer=str(data.get("answer") or ""),
            user_email=str(data.get("user_email") or ""),
            created_at=str(data.get("created_at") or _utc_now_iso()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "rating": self.rating,
            "comment": self.comment,
            "question": self.question,
            "answer": self.answer,
            "user_email": self.user_email,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class FeedbackStore(Protocol):
    def add(self, record: FeedbackRecord) -> None:  # pragma: no cover - protocol
        ...

    def list_all(self) -> list[FeedbackRecord]:  # pragma: no cover - protocol
        ...


class InMemoryFeedbackStore:
    """Local/dev backing. Feedback lives only as long as the process."""

    def __init__(self) -> None:
        self._rows: list[FeedbackRecord] = []

    def add(self, record: FeedbackRecord) -> None:
        # Replace in place, matching what put_item does against the same key - a
        # comment added to an existing rating must not become a second row.
        for index, existing in enumerate(self._rows):
            if existing.feedback_id == record.feedback_id:
                self._rows[index] = record
                return
        self._rows.append(record)

    def list_all(self) -> list[FeedbackRecord]:
        return list(self._rows)


class DynamoDBFeedbackStore:
    """Feedback in the shared conversations table under a fixed partition."""

    def __init__(self, *, table_name: str, region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'aws' dependencies for DynamoDB storage.") from exc

        self.table_name = table_name
        self._resource = boto3.resource("dynamodb", region_name=region or os.getenv("AWS_REGION"))
        self._table = self._resource.Table(table_name)

    def add(self, record: FeedbackRecord) -> None:
        self._table.put_item(Item=record.to_item())

    def list_all(self) -> list[FeedbackRecord]:
        from boto3.dynamodb.conditions import Key

        rows: list[FeedbackRecord] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("user_id").eq(FEEDBACK_PARTITION)
        }
        while True:
            response = self._table.query(**kwargs)
            rows.extend(FeedbackRecord.from_item(item) for item in response.get("Items", []))
            token = response.get("LastEvaluatedKey")
            if not token:
                return rows
            kwargs["ExclusiveStartKey"] = token


def build_feedback_store_from_env() -> FeedbackStore:
    table_name = os.getenv("DYNAMODB_CONVERSATIONS_TABLE", "").strip()
    if table_name:
        return DynamoDBFeedbackStore(table_name=table_name, region=os.getenv("AWS_REGION"))
    return InMemoryFeedbackStore()


# ---------------------------------------------------------------------------
# The log the API talks to
# ---------------------------------------------------------------------------


def _clip(value: str, limit: int) -> str:
    text = (value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class FeedbackLog:
    """Records a rating against the turn that earned it, and reads it back grouped."""

    def __init__(self, store: FeedbackStore, conversation_store: Any | None = None) -> None:
        self._store = store
        self._conversations = conversation_store

    # -- writing ----------------------------------------------------------

    def record(
        self,
        *,
        user_id: str,
        user_email: str,
        conversation_id: str,
        rating: str,
        comment: str = "",
        answer: str = "",
        feedback_id: str = "",
        created_at: str = "",
    ) -> FeedbackRecord:
        """Save a rating. Passing back an existing id rewrites that row in place.

        Thumbs-down is two steps for the person - the click, then the comment they
        may or may not write - but it is one opinion, so it must not read as two in
        the admin's list. The click records immediately (the signal is never lost to
        a dismissed box) and the comment lands on the same row.
        """
        if rating not in RATINGS:
            raise FeedbackError("That is not a rating we recognise.")
        if not conversation_id:
            raise FeedbackError("We could not tell which conversation this was about.")

        title, question, stored_answer = self._resolve_turn(user_id, conversation_id, answer)
        record = FeedbackRecord(
            feedback_id=feedback_id.strip() or uuid.uuid4().hex,
            conversation_id=conversation_id,
            conversation_title=title,
            rating=rating,
            comment=_clip(comment, MAX_COMMENT_CHARS),
            question=_clip(question, MAX_EXCERPT_CHARS),
            answer=_clip(stored_answer, MAX_EXCERPT_CHARS),
            user_email=(user_email or "").strip().lower(),
            **({"created_at": created_at.strip()} if created_at.strip() else {}),
        )
        self._store.add(record)
        return record

    def _resolve_turn(
        self, user_id: str, conversation_id: str, answer: str
    ) -> tuple[str, str, str]:
        """Find the saved exchange this rating is about.

        Matched on the answer text rather than an index: a turn that only asked for
        clarification is shown in the browser but never saved, so positions drift.
        Falls back to what the browser sent, so feedback is never lost to a miss.
        """
        fallback = ("Untitled chat", "", answer)
        if self._conversations is None:
            return fallback
        try:
            record = self._conversations.get_conversation(user_id, conversation_id)
        except Exception as exc:  # noqa: BLE001 - a storage blip must not eat the feedback
            logger.warning("Could not load conversation %s for feedback: %s", conversation_id, exc)
            return fallback
        if record is None:
            return fallback

        title = record.title or "Untitled chat"
        needle = (answer or "").strip()[:200]
        for index in range(len(record.messages) - 1, -1, -1):
            message = record.messages[index]
            if message.role != "assistant":
                continue
            if needle and not message.text.strip().startswith(needle):
                continue
            question = ""
            for back in range(index - 1, -1, -1):
                if record.messages[back].role == "user":
                    question = record.messages[back].text
                    break
            return title, question, message.text

        return title, "", answer

    # -- reading ----------------------------------------------------------

    def conversations(self) -> list[dict[str, Any]]:
        """Feedback grouped by the chat it came from, most recently rated first."""
        groups: dict[str, dict[str, Any]] = {}
        for record in self._store.list_all():
            group = groups.setdefault(
                record.conversation_id,
                {
                    "conversation_id": record.conversation_id,
                    "title": record.conversation_title or "Untitled chat",
                    "items": [],
                },
            )
            if record.conversation_title:
                group["title"] = record.conversation_title
            group["items"].append(record.to_dict())

        rows: list[dict[str, Any]] = []
        for group in groups.values():
            items = sorted(group["items"], key=lambda item: item["created_at"], reverse=True)
            rows.append(
                {
                    **group,
                    "items": items,
                    "count": len(items),
                    "down_count": sum(1 for item in items if item["rating"] == "down"),
                    "last_at": items[0]["created_at"] if items else "",
                }
            )
        return sorted(rows, key=lambda row: row["last_at"], reverse=True)

    def summary(self) -> dict[str, Any]:
        rows = self._store.list_all()
        return {
            "total": len(rows),
            "down_count": sum(1 for row in rows if row.rating == "down"),
            "up_count": sum(1 for row in rows if row.rating == "up"),
        }


def build_feedback_log_from_env(conversation_store: Any | None = None) -> FeedbackLog:
    return FeedbackLog(build_feedback_store_from_env(), conversation_store)
