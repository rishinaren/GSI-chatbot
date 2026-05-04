"""Domain models for standards, source chunks, and citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class DocumentType(StrEnum):
    """High-level category for the source document."""

    STANDARD = "standard"
    GUIDE = "guide"
    PRACTICE = "practice"
    TEST_METHOD = "test_method"
    SPECIFICATION = "specification"
    OTHER = "other"


class LifecycleStatus(StrEnum):
    """Lifecycle state used to track review/update risk."""

    ACTIVE = "active"
    DUE_FOR_REVIEW = "due_for_review"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StandardDocument:
    """Metadata for a standard or standard-like source document."""

    document_id: str
    standard_id: str
    title: str
    issuing_body: str
    document_type: DocumentType = DocumentType.OTHER
    year: int | None = None
    revision: str | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.UNKNOWN
    review_due_year: int | None = None
    source_path: str | None = None
    checksum: str | None = None
    permission_scope: str = "licensed_internal_source"
    references: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document_type"] = self.document_type.value
        data["lifecycle_status"] = self.lifecycle_status.value
        data["references"] = list(self.references)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardDocument":
        return cls(
            **{
                **data,
                "document_type": DocumentType(data.get("document_type", DocumentType.OTHER)),
                "lifecycle_status": LifecycleStatus(
                    data.get("lifecycle_status", LifecycleStatus.UNKNOWN)
                ),
                "references": tuple(data.get("references", ())),
            }
        )


@dataclass(frozen=True)
class SourceChunk:
    """Searchable text span with enough metadata to cite it."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    heading: str | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceChunk":
        return cls(**data)


@dataclass(frozen=True)
class Citation:
    """User-facing source reference for a claim."""

    document_id: str
    standard_id: str
    title: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    quote: str | None = None

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return "page unknown"
        if self.page_end and self.page_end != self.page_start:
            return f"pages {self.page_start}-{self.page_end}"
        return f"page {self.page_start}"

    def format(self) -> str:
        parts = [self.standard_id, self.title]
        if self.section:
            parts.append(f"Section {self.section}")
        parts.append(self.page_label)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
