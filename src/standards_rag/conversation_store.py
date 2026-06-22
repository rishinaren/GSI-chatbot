"""Persistent conversation storage for authenticated chat sessions."""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
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
    pinned: bool = False
    project_id: str | None = None
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
            "pinned": self.pinned,
            "project_id": self.project_id,
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
            pinned=bool(data.get("pinned", False)),
            project_id=data.get("project_id"),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or _utc_now_iso()),
        )


# Projects are stored in the same backing table as conversations. To keep a
# single DynamoDB table (PK user_id, SK conversation_id) we store each project
# as an item whose sort key is prefixed so it can be told apart from chats.
PROJECT_SORT_KEY_PREFIX = "PROJECT#"


def _is_project_key(conversation_id: str) -> bool:
    return conversation_id.startswith(PROJECT_SORT_KEY_PREFIX)


@dataclass
class ProjectRecord:
    project_id: str
    user_id: str
    name: str
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_item(self) -> dict[str, Any]:
        """Serialize for the shared conversations table (project as an item)."""
        return {
            "user_id": self.user_id,
            "conversation_id": f"{PROJECT_SORT_KEY_PREFIX}{self.project_id}",
            "record_type": "project",
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_item(cls, data: dict[str, Any]) -> "ProjectRecord":
        return cls(
            project_id=str(data["project_id"]),
            user_id=str(data["user_id"]),
            name=str(data.get("name") or "Untitled project"),
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
        title_generator: Callable[[str, str], str] | None = None,
    ) -> ConversationRecord:
        raise NotImplementedError

    @abstractmethod
    def set_pinned(
        self, user_id: str, conversation_id: str, pinned: bool
    ) -> ConversationRecord | None:
        raise NotImplementedError

    @abstractmethod
    def set_project(
        self, user_id: str, conversation_id: str, project_id: str | None
    ) -> ConversationRecord | None:
        raise NotImplementedError

    @abstractmethod
    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_projects(self, user_id: str) -> list[ProjectRecord]:
        raise NotImplementedError

    @abstractmethod
    def create_project(self, user_id: str, *, name: str) -> ProjectRecord:
        raise NotImplementedError

    @abstractmethod
    def rename_project(self, user_id: str, project_id: str, name: str) -> ProjectRecord | None:
        raise NotImplementedError

    @abstractmethod
    def delete_project(self, user_id: str, project_id: str) -> bool:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ConversationRecord] = {}
        self._projects: dict[tuple[str, str], ProjectRecord] = {}

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationRecord]:
        rows = [record for (uid, _), record in self._records.items() if uid == user_id]
        rows.sort(key=lambda item: (item.pinned, item.updated_at), reverse=True)
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
        title_generator: Callable[[str, str], str] | None = None,
    ) -> ConversationRecord:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            record = ConversationRecord(
                conversation_id=conversation_id,
                user_id=user_id,
                title="New chat",
            )
            self._records[(user_id, conversation_id)] = record

        is_first_turn = not record.messages
        record.messages.append(StoredMessage(role="user", text=question))
        record.messages.append(StoredMessage(role="assistant", text=answer, citations=citations))
        record.updated_at = _utc_now_iso()
        if unit_preference:
            record.unit_preference = unit_preference
        if record.title == "New chat" and is_first_turn:
            record.title = _resolve_title(question, answer, title_generator)
        return record

    def set_pinned(
        self, user_id: str, conversation_id: str, pinned: bool
    ) -> ConversationRecord | None:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            return None
        record.pinned = pinned
        return record

    def set_project(
        self, user_id: str, conversation_id: str, project_id: str | None
    ) -> ConversationRecord | None:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            return None
        if project_id is not None and (user_id, project_id) not in self._projects:
            raise KeyError("project not found")
        record.project_id = project_id
        record.updated_at = _utc_now_iso()
        return record

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        return self._records.pop((user_id, conversation_id), None) is not None

    def list_projects(self, user_id: str) -> list[ProjectRecord]:
        rows = [project for (uid, _), project in self._projects.items() if uid == user_id]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return rows

    def create_project(self, user_id: str, *, name: str) -> ProjectRecord:
        project = ProjectRecord(
            project_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name.strip() or "Untitled project",
        )
        self._projects[(user_id, project.project_id)] = project
        return project

    def rename_project(self, user_id: str, project_id: str, name: str) -> ProjectRecord | None:
        project = self._projects.get((user_id, project_id))
        if project is None:
            return None
        project.name = name.strip() or project.name
        project.updated_at = _utc_now_iso()
        return project

    def delete_project(self, user_id: str, project_id: str) -> bool:
        removed = self._projects.pop((user_id, project_id), None) is not None
        if removed:
            for record in self._records.values():
                if record.user_id == user_id and record.project_id == project_id:
                    record.project_id = None
        return removed


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
        )
        records = [
            ConversationRecord.from_dict(item)
            for item in response.get("Items", [])
            if not _is_project_key(str(item.get("conversation_id", "")))
        ]
        records.sort(key=lambda item: (item.pinned, item.updated_at), reverse=True)
        return records[:limit]

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
        title_generator: Callable[[str, str], str] | None = None,
    ) -> ConversationRecord:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            record = ConversationRecord(
                conversation_id=conversation_id,
                user_id=user_id,
                title="New chat",
            )
        is_first_turn = not record.messages
        record.messages.append(StoredMessage(role="user", text=question))
        record.messages.append(StoredMessage(role="assistant", text=answer, citations=citations))
        record.updated_at = _utc_now_iso()
        if unit_preference:
            record.unit_preference = unit_preference
        if record.title == "New chat" and is_first_turn:
            record.title = _resolve_title(question, answer, title_generator)
        self._table.put_item(Item=record.to_dict())
        return record

    def set_pinned(
        self, user_id: str, conversation_id: str, pinned: bool
    ) -> ConversationRecord | None:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            return None
        record.pinned = pinned
        self._table.put_item(Item=record.to_dict())
        return record

    def set_project(
        self, user_id: str, conversation_id: str, project_id: str | None
    ) -> ConversationRecord | None:
        record = self.get_conversation(user_id, conversation_id)
        if record is None:
            return None
        if project_id is not None and self._get_project(user_id, project_id) is None:
            raise KeyError("project not found")
        record.project_id = project_id
        record.updated_at = _utc_now_iso()
        self._table.put_item(Item=record.to_dict())
        return record

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        response = self._table.delete_item(
            Key={"user_id": user_id, "conversation_id": conversation_id},
            ReturnValues="ALL_OLD",
        )
        return bool(response.get("Attributes"))

    def _project_sort_key(self, project_id: str) -> str:
        return f"{PROJECT_SORT_KEY_PREFIX}{project_id}"

    def _get_project(self, user_id: str, project_id: str) -> ProjectRecord | None:
        response = self._table.get_item(
            Key={"user_id": user_id, "conversation_id": self._project_sort_key(project_id)}
        )
        item = response.get("Item")
        return ProjectRecord.from_item(item) if item else None

    def list_projects(self, user_id: str) -> list[ProjectRecord]:
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("user_id").eq(user_id)
            & Key("conversation_id").begins_with(PROJECT_SORT_KEY_PREFIX),
        )
        projects = [ProjectRecord.from_item(item) for item in response.get("Items", [])]
        projects.sort(key=lambda item: item.updated_at, reverse=True)
        return projects

    def create_project(self, user_id: str, *, name: str) -> ProjectRecord:
        project = ProjectRecord(
            project_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name.strip() or "Untitled project",
        )
        self._table.put_item(Item=project.to_item())
        return project

    def rename_project(self, user_id: str, project_id: str, name: str) -> ProjectRecord | None:
        project = self._get_project(user_id, project_id)
        if project is None:
            return None
        project.name = name.strip() or project.name
        project.updated_at = _utc_now_iso()
        self._table.put_item(Item=project.to_item())
        return project

    def delete_project(self, user_id: str, project_id: str) -> bool:
        if self._get_project(user_id, project_id) is None:
            return False
        # Unassign any conversations that referenced this project.
        for record in self.list_conversations(user_id, limit=500):
            if record.project_id == project_id:
                record.project_id = None
                record.updated_at = _utc_now_iso()
                self._table.put_item(Item=record.to_dict())
        self._table.delete_item(
            Key={"user_id": user_id, "conversation_id": self._project_sort_key(project_id)}
        )
        return True


def _title_from_question(question: str) -> str:
    cleaned = " ".join(question.split())
    if len(cleaned) <= 72:
        return cleaned or "New chat"
    return cleaned[:69].rstrip() + "..."


def _resolve_title(
    question: str,
    answer: str,
    title_generator: Callable[[str, str], str] | None,
) -> str:
    """Use the LLM title generator when available; fall back to the question text."""
    if title_generator is not None:
        try:
            generated = title_generator(question, answer).strip()
        except Exception:  # noqa: BLE001 - never block persistence on title generation
            generated = ""
        if generated:
            cleaned = " ".join(generated.split()).strip("\"'")
            if cleaned:
                return cleaned[:60].rstrip()
    return _title_from_question(question)


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
