"""Persistent conversation storage for authenticated chat sessions."""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class StoredMessage:
    role: str
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredMessage":
        return cls(
            role=str(data["role"]),
            text=str(data["text"]),
            citations=list(data.get("citations", [])),
            created_at=str(data.get("created_at") or _utc_now_iso()),
        )


@dataclass
class ConversationRecord:
    conversation_id: str
    user_id: str
    title: str
    messages: list[StoredMessage] = field(default_factory=list)
    organization_id: str | None = None
    unit_preference: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [message.to_dict() for message in self.messages],
            "organization_id": self.organization_id,
            "unit_preference": self.unit_preference,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationRecord":
        return cls(
            conversation_id=str(data["conversation_id"]),
            user_id=str(data["user_id"]),
            title=str(data.get("title") or "New chat"),
            messages=[StoredMessage.from_dict(item) for item in data.get("messages", [])],
            organization_id=data.get("organization_id"),
            unit_preference=data.get("unit_preference"),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or _utc_now_iso()),
        )


class ConversationStore(ABC):
    @abstractmethod
    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_conversation(self, user_id: str, conversation_id: str) -> ConversationRecord | None:
        raise NotImplementedError

    @abstractmethod
    def create_conversation(
        self,
        user_id: str,
        *,
        title: str = "New chat",
        organization_id: str | None = None,
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        raise NotImplementedError

    @abstractmethod
    def append_turn(
        self,
        user_id: str,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        raise NotImplementedError

    @abstractmethod
    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ConversationRecord] = {}

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationRecord]:
        rows = [record for (uid, _), record in self._records.items() if uid == user_id]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return rows[:limit]

    def get_conversation(self, user_id: str, conversation_id: str) -> ConversationRecord | None:
        return self._records.get((user_id, conversation_id))

    def create_conversation(
        self,
        user_id: str,
        *,
        title: str = "New chat",
        organization_id: str | None = None,
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        record = ConversationRecord(
            conversation_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            organization_id=organization_id,
            unit_preference=unit_preference,
        )
        self._records[(user_id, record.conversation_id)] = record
        return record

    def append_turn(
        self,
        user_id: str,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            record = ConversationRecord(
                conversation_id=conversation_id,
                user_id=user_id,
                title=_title_from_question(question),
            )
            self._records[(user_id, conversation_id)] = record

        record.messages.append(StoredMessage(role="user", text=question))
        record.messages.append(StoredMessage(role="assistant", text=answer, citations=citations))
        record.updated_at = _utc_now_iso()
        if unit_preference:
            record.unit_preference = unit_preference
        if record.title == "New chat":
            record.title = _title_from_question(question)
        return record

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        return self._records.pop((user_id, conversation_id), None) is not None


class DynamoDBConversationStore(ConversationStore):
    def __init__(self, *, table_name: str, region: str | None = None, ttl_days: int = 90) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'aws' dependencies for DynamoDB storage.") from exc

        self.table_name = table_name
        self.ttl_days = ttl_days
        self._client = boto3.client("dynamodb", region_name=region or os.getenv("AWS_REGION"))
        self._resource = boto3.resource("dynamodb", region_name=region or os.getenv("AWS_REGION"))
        self._table = self._resource.Table(table_name)

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationRecord]:
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [ConversationRecord.from_dict(item) for item in response.get("Items", [])]

    def get_conversation(self, user_id: str, conversation_id: str) -> ConversationRecord | None:
        response = self._table.get_item(Key={"user_id": user_id, "conversation_id": conversation_id})
        item = response.get("Item")
        return ConversationRecord.from_dict(item) if item else None

    def create_conversation(
        self,
        user_id: str,
        *,
        title: str = "New chat",
        organization_id: str | None = None,
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        record = ConversationRecord(
            conversation_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            organization_id=organization_id,
            unit_preference=unit_preference,
        )
        self._table.put_item(Item=record.to_dict())
        return record

    def append_turn(
        self,
        user_id: str,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        unit_preference: str | None = None,
    ) -> ConversationRecord:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            record = ConversationRecord(
                conversation_id=conversation_id,
                user_id=user_id,
                title=_title_from_question(question),
            )
        record.messages.append(StoredMessage(role="user", text=question))
        record.messages.append(StoredMessage(role="assistant", text=answer, citations=citations))
        record.updated_at = _utc_now_iso()
        if unit_preference:
            record.unit_preference = unit_preference
        if record.title == "New chat":
            record.title = _title_from_question(question)
        self._table.put_item(Item=record.to_dict())
        return record

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        response = self._table.delete_item(
            Key={"user_id": user_id, "conversation_id": conversation_id},
            ReturnValues="ALL_OLD",
        )
        return bool(response.get("Attributes"))


def _title_from_question(question: str) -> str:
    cleaned = " ".join(question.split())
    if len(cleaned) <= 72:
        return cleaned or "New chat"
    return cleaned[:69].rstrip() + "..."


def build_conversation_store_from_env() -> ConversationStore:
    table_name = os.getenv("DYNAMODB_CONVERSATIONS_TABLE", "").strip()
    if table_name:
        ttl_days_raw = os.getenv("CONVERSATION_TTL_DAYS", "90").strip()
        try:
            ttl_days = max(int(ttl_days_raw), 1)
        except ValueError:
            ttl_days = 90
        return DynamoDBConversationStore(
            table_name=table_name,
            region=os.getenv("AWS_REGION"),
            ttl_days=ttl_days,
        )
    return InMemoryConversationStore()
