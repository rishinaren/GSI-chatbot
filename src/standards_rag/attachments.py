"""Question attachments: a PDF the person brings, read alongside the library.

Someone asking "does my spec meet GM13?" has two documents in play - the one they
uploaded and the ones we hold. This module handles the first, and it deliberately
handles it *differently* from ``library.py``:

* an attachment is **never ingested**. No embedding call, no Pinecone write, no
  index merge, no S3 copy. It belongs to one person's question, not to the corpus
  everybody queries.
* passages are picked **lexically**, by the same scorer the local index already
  uses. Choosing which three paragraphs of a 90-page spec matter costs nothing.
* the chosen passages ride the answer call that was going to happen anyway, so a
  question with an attachment costs one extra PDF text extraction plus a bounded
  number of prompt tokens (``PROMPT_CHAR_BUDGET``) - not a second model call.

The extracted text lives in this process only, under a TTL and an LRU cap, which
matches the single-instance assumption ``library.py`` already makes. If it ages
out the person is asked to attach the file again; nothing is silently answered
from a document we no longer hold.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from standards_rag.ingestion import PageText, chunk_pages, extract_pdf_pages
from standards_rag.models import Citation, DocumentType, SourceChunk, StandardDocument
from standards_rag.retrieval import STOP_WORDS, TOKEN_RE, InMemoryStandardsStore

# Attachments are read in full into memory, so they are capped tighter than a
# library upload (which is written straight to disk).
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
# A one-page memo is a legitimate attachment, so this floor is lower than the
# library's - it is only here to catch a scan with no text layer at all.
MIN_EXTRACTED_CHARS = 120
# Guard rails on what one process will hold. A session's worth of attachments,
# then the oldest goes.
TTL_SECONDS = int(os.getenv("CHAT_ATTACHMENT_TTL_SECONDS", "21600") or 21600)
MAX_CACHED = int(os.getenv("CHAT_ATTACHMENT_CACHE_SIZE", "40") or 40)
# How much attachment text may reach the model per question. This is the whole
# cost story for an attached document, so it is one number, easy to tune.
PROMPT_CHAR_BUDGET = int(os.getenv("CHAT_ATTACHMENT_PROMPT_CHARS", "6000") or 6000)
MAX_PASSAGES = 5
# Designations the assistant should look up in the library because the person's
# own document names them.
MAX_DESIGNATION_HINTS = 6

ATTACHMENT_ID_PREFIX = "attachment:"

_DESIGNATION_RE = re.compile(
    r"\bASTM\s+[A-Z]\s?\d{2,5}(?:/[A-Z]\d{2,5})?(?:-\d{2,4}[a-z]?)?\b"
    r"|\b[DEFG]\s?\d{3,5}(?:/[A-Z]\d{3,5})?(?:-\d{2,4}[a-z]?)?\b"
    r"|\bGRI[-\s]?(?:GM|GT|GCL|GC|GG|GS)\s?\d{1,3}[a-z]?\b"
    r"|\b(?:GM|GT|GCL|GG)\s?\d{1,3}[a-z]?\b"
    r"|\bISO\s?\d{3,5}(?:[-:]\d+)*\b",
    re.IGNORECASE,
)


class AttachmentError(Exception):
    """A problem worth showing to the person who attached the file, in their words."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Attachment:
    """One uploaded PDF, read and held for the length of a working session."""

    attachment_id: str
    owner_id: str
    file_name: str
    size_bytes: int
    page_count: int
    char_count: int
    created_at: str
    designations: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    # A throwaway store holding just this document, so passage selection reuses
    # the tuned lexical scorer instead of a second implementation of it.
    index: InMemoryStandardsStore = field(default_factory=InMemoryStandardsStore, repr=False)
    _expires_at: float = 0.0

    @property
    def document_id(self) -> str:
        return f"{ATTACHMENT_ID_PREFIX}{self.attachment_id}"

    @property
    def expired(self) -> bool:
        return monotonic() >= self._expires_at

    def to_dict(self) -> dict[str, object]:
        """What the browser needs to render the chip. No document text."""
        return {
            "attachment_id": self.attachment_id,
            "file_name": self.file_name,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
        }


def _validate_pdf(data: bytes, filename: str) -> None:
    if not data:
        raise AttachmentError("That file came through empty. Please try choosing it again.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        limit = MAX_ATTACHMENT_BYTES // (1024 * 1024)
        raise AttachmentError(
            f"That file is larger than {limit} MB. Please attach a smaller PDF, "
            "or the section of it you want to ask about."
        )
    if not data.lstrip()[:5].startswith(b"%PDF"):
        name = Path(filename or "").name or "That file"
        raise AttachmentError(f"{name} is not a PDF. Please attach the PDF version.")


def _read_pages(data: bytes) -> list[PageText]:
    """Extract page text, turning an unreadable file into something worth reading.

    PyMuPDF signals a damaged or password-protected file with its own
    ``RuntimeError`` subclasses, so the import is checked separately - otherwise a
    missing dependency and a bad file are indistinguishable here.
    """
    try:
        import fitz  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only when the pdf extra is missing
        raise RuntimeError("Install the optional 'pdf' dependencies to read attachments.") from exc

    from tempfile import NamedTemporaryFile

    handle = NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        handle.write(data)
        handle.close()
        return extract_pdf_pages(handle.name)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same story
        raise AttachmentError(
            "We could not open that PDF - it looks damaged, or it is protected by a "
            "password. Try opening it on your computer, saving a fresh copy, and "
            "attaching that."
        ) from exc
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _normalize_designation(raw: str) -> str:
    value = re.sub(r"\s+", " ", raw.strip().upper())
    value = re.sub(r"^ASTM\s+", "", value)
    value = re.sub(r"^GRI[-\s]", "", value)
    return re.sub(r"(?<=[A-Z])\s+(?=\d)", "", value)


def designations_in(text: str) -> tuple[str, ...]:
    """Standard designations the attached document names, most-mentioned first.

    These become retrieval hints: if someone's spec calls out GM13 and D4595, the
    answer should be reaching for those standards whatever the question says.
    """
    counts: Counter[str] = Counter()
    for match in _DESIGNATION_RE.finditer(text):
        value = _normalize_designation(match.group(0))
        if len(value) >= 3:
            counts[value] += 1
    return tuple(value for value, _ in counts.most_common(MAX_DESIGNATION_HINTS))


def keywords_in(text: str, *, limit: int = 10) -> tuple[str, ...]:
    """The document's own recurring vocabulary, for questions too thin to retrieve on."""
    counts: Counter[str] = Counter()
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) >= 5 and token not in STOP_WORDS and not token.isdigit():
            counts[token] += 1
    return tuple(word for word, count in counts.most_common(limit) if count >= 2)


def read_attachment(data: bytes, filename: str, *, owner_id: str) -> Attachment:
    """Read an uploaded PDF into a searchable, in-process attachment.

    Nothing here touches Pinecone, the standards index, or S3 - by design. See the
    module docstring.
    """
    _validate_pdf(data, filename)
    pages = _read_pages(data)
    text = "\n".join(page.text for page in pages)
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise AttachmentError(
            "We could not find any readable text in that PDF - it looks like a scan or a "
            "photo of the pages. Please attach a version where the words can be selected "
            "and copied."
        )

    attachment_id = uuid.uuid4().hex
    name = Path(filename or "").name or "attachment.pdf"
    document = StandardDocument(
        document_id=f"{ATTACHMENT_ID_PREFIX}{attachment_id}",
        standard_id=name,
        title=name,
        issuing_body="UPLOAD",
        document_type=DocumentType.OTHER,
        # No source_path: the PDF is not kept on disk, so nothing can link to it
        # and no citation will try.
        metadata={"attachment": True},
    )
    chunks = chunk_pages(document, pages)
    index = InMemoryStandardsStore()
    index.add_document(document, chunks)

    return Attachment(
        attachment_id=attachment_id,
        owner_id=owner_id,
        file_name=name,
        size_bytes=len(data),
        page_count=len(pages),
        char_count=len(text),
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        designations=designations_in(text),
        keywords=keywords_in(text),
        index=index,
        _expires_at=monotonic() + TTL_SECONDS,
    )


class AttachmentStore:
    """In-process attachments, oldest evicted first, each expiring on its own.

    Deliberately not persisted. An attachment is scratch context for a working
    session, and keeping it in memory is what makes it free.
    """

    def __init__(self, *, max_items: int = MAX_CACHED) -> None:
        self._items: OrderedDict[str, Attachment] = OrderedDict()
        self._max_items = max(1, max_items)
        self._lock = threading.Lock()

    def add(self, attachment: Attachment) -> Attachment:
        with self._lock:
            self._drop_expired()
            self._items[attachment.attachment_id] = attachment
            self._items.move_to_end(attachment.attachment_id)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return attachment

    def get(self, attachment_id: str, *, owner_id: str) -> Attachment | None:
        """Return the attachment, or None if it is gone, expired, or someone else's.

        Another person's attachment reads as missing rather than forbidden, so the
        response cannot be used to learn that an id exists.
        """
        with self._lock:
            self._drop_expired()
            attachment = self._items.get(attachment_id)
            if attachment is None or attachment.owner_id != owner_id:
                return None
            self._items.move_to_end(attachment_id)
            return attachment

    def _drop_expired(self) -> None:
        for key in [key for key, item in self._items.items() if item.expired]:
            self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            self._drop_expired()
            return len(self._items)


def select_passages(
    attachment: Attachment,
    question: str,
    *,
    budget: int = PROMPT_CHAR_BUDGET,
    max_passages: int = MAX_PASSAGES,
) -> list[SourceChunk]:
    """The parts of the attached document worth putting in front of the model.

    Lexical only - the whole point is that reading someone's document costs nothing
    per question. When the question has no overlap with the text ("summarise this"),
    fall back to the opening pages, which is where a spec states what it is.
    """
    results = attachment.index.search(question, top_k=max_passages * 2, min_score=0.0)
    ordered = [result.chunk for result in results]
    if not ordered:
        ordered = sorted(attachment.index.chunks.values(), key=lambda chunk: chunk.order)

    picked: list[SourceChunk] = []
    used = 0
    for chunk in ordered:
        if len(picked) >= max_passages or used >= budget:
            break
        text = chunk.text.strip()
        if not text:
            continue
        picked.append(chunk)
        used += len(text)

    # Read back in document order: a spec's own sequence is part of its meaning.
    return sorted(picked, key=lambda chunk: chunk.order)


def passage_text(chunk: SourceChunk, *, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", chunk.text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def attachment_citations(attachment: Attachment, chunks: list[SourceChunk]) -> list[Citation]:
    """Citations for the person's own document, marked so the UI never links out.

    ``quote`` carries the excerpt because the citation validator looks the chunk up
    in the standards index, which by design does not hold these.
    """
    citations: list[Citation] = []
    for chunk in chunks:
        citations.append(
            Citation(
                document_id=attachment.document_id,
                standard_id=attachment.file_name,
                title=attachment.file_name,
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                quote=passage_text(chunk),
                source_kind="attachment",
            )
        )
    return citations


def retrieval_hint(attachments: list[Attachment], question: str) -> str:
    """Extra query terms drawn from the attached documents, so the library is searched
    for what the person's document is actually about.

    Designations always go in - a spec naming GM13 should pull GM13 whatever the
    question says. The document's general vocabulary only goes in when the question
    is too thin to retrieve on by itself, so a specific question stays specific.
    """
    if not attachments:
        return ""
    parts: list[str] = []
    for attachment in attachments:
        parts.extend(attachment.designations)

    content_terms = [
        token
        for token in TOKEN_RE.findall(question.lower())
        if len(token) > 2 and token not in STOP_WORDS
    ]
    if len(content_terms) < 5:
        for attachment in attachments:
            parts.extend(attachment.keywords[:8])

    return " ".join(dict.fromkeys(parts))
