"""Validate in-answer citation markers against retrieved chunk text; strip unsupported markers."""

from __future__ import annotations

import re

from standards_rag.models import Citation, SourceChunk

SHORT_DOMAIN_TERMS = frozenset(
    {"gcl", "gsm", "gmb", "gsy", "md", "qm", "cre", "pfa", "pmb", "nctl", "escr"}
)

ANCHOR_STOP = frozenset(
    """
    this that these those with from they them their there than then some such only also into when
    what which where whether while having been being does been have has had were was are is not for
    are but and any all may can use uses used using per one two how its our out the you your we
""".split()
)

# Match citation marker preceded by whitespace (draft format: ... claim [n])
CITE_MARKER = re.compile(r"(\s*)\[(\d{1,2})\](?!\d)")
COMPACT_SID_RE = re.compile(r"[^a-z0-9]+")
NEGATIVE_ASSERTION_RE = re.compile(
    r"\b(?:does|do|is|are|can|may|will|shall)\s+not\b|"
    r"\bno\s+(?:explicit\s+|direct\s+|specific\s+)?(?:requirement|guidance|value|method|support)\b",
    re.IGNORECASE,
)
STANDARD_MENTION_RE = re.compile(
    r"\b(?:ASTM\s*)?[A-Z]\d{2,5}(?:/[A-Z]?\d{2,5}[A-Z]?)?"
    r"(?:[-–]\s*\d{2,4}[A-Z]?(?:\(\d{4}\))?)?\b|"
    r"\bISO\s+\d+(?:[-:]\d+)*(?::\d{4})?\b|"
    r"\bBS\s+(?:EN\s+)?\d+(?:[-:]\d+)*(?::\d{4})?\b|"
    r"\b(?:GRI[-\s]*)?(?:GCL|GG|GM|GN|GC|GT|GS)\s*-?\s*\d+[A-Z]?\b",
    re.IGNORECASE,
)


def _standard_id_compact(value: str) -> str:
    return COMPACT_SID_RE.sub("", value.lower())


def _standard_lookup_keys(value: str) -> set[str]:
    raw = (value or "").upper().strip()
    raw = re.sub(r"^(?:ASTM|GRI)[-\s]*", "", raw)
    compact = re.sub(r"[^A-Z0-9]", "", raw)
    if not compact:
        return set()
    keys = {compact}
    if compact.startswith("ISO"):
        match = re.match(r"ISO(\d+)", compact)
        return keys | ({f"ISO{match.group(1)}"} if match else set())
    astm = re.match(r"([A-Z]\d{2,5})(?=/|[-–\s(]|$)", raw)
    if astm:
        keys.add(astm.group(1))
    gri = re.match(r"((?:GCL|GG|GM|GN|GC|GT|GS)\d+)([A-Z]?)", compact)
    if gri:
        keys.add(gri.group(1))
        if gri.group(2):
            keys.add(gri.group(1) + gri.group(2))
    return keys


def _standard_mentions(value: str) -> set[str]:
    keys: set[str] = set()
    for match in STANDARD_MENTION_RE.finditer(value):
        keys.update(_standard_lookup_keys(match.group(0)))
    return keys


def _claim_line_before_marker(answer: str, match_start: int) -> str:
    line_start = answer.rfind("\n", 0, match_start)
    line_start = 0 if line_start < 0 else line_start + 1
    return answer[line_start:match_start].strip()


def _chunk_haystack(citation: Citation, chunk: SourceChunk | None) -> str:
    parts = [
        (chunk.text if chunk else ""),
        citation.title or "",
        citation.quote or "",
        (chunk.heading if chunk else "") or "",
        citation.standard_id or "",
        " ",
    ]
    # Help substring matches for hyphenated headings in PDF extracts
    return " ".join(parts).lower()


def _meaningful_claim_tokens(claim: str) -> list[str]:
    lowered = claim.lower()
    tokens: set[str] = set(re.findall(r"[a-z]{3,}", lowered))
    tokens |= set(re.findall(r"[a-z]?\d{3,}[a-z]?", lowered))
    for term in SHORT_DOMAIN_TERMS:
        if term in lowered:
            tokens.add(term)
    return sorted(
        t
        for t in tokens
        if t not in ANCHOR_STOP and (len(t) >= 4 or t in SHORT_DOMAIN_TERMS or t.isdigit())
    )


def citation_supports_claim(claim: str, citation: Citation, chunk: SourceChunk | None) -> bool:
    """True if the cited chunk/title/quote contains enough lexical overlap with the claim line."""
    if not citation.chunk_id:
        return False
    if chunk is None and not (citation.quote or "").strip():
        return False

    haystack = _chunk_haystack(citation, chunk)
    if not haystack.strip():
        return False

    # Lexical overlap alone can make a claim about D4632 look supported by a D6768
    # tensile passage. If the claim names a source, the marker must point to that source.
    claim_standards = _standard_mentions(claim)
    citation_standards = _standard_lookup_keys(citation.standard_id)
    if (
        citation.source_kind != "attachment"
        and claim_standards
        and not (claim_standards & citation_standards)
    ):
        return False

    tokens = _meaningful_claim_tokens(claim)
    sid_comp = _standard_id_compact(citation.standard_id)
    claim_comp = _standard_id_compact(claim)
    designation_in_claim = bool(sid_comp) and sid_comp in claim_comp

    if not tokens:
        return designation_in_claim and sid_comp and sid_comp in _standard_id_compact(haystack)

    hits = sum(1 for t in tokens if t in haystack)
    if len(tokens) <= 2:
        return hits >= len(tokens)

    need = max(2, (len(tokens) + 2) // 3)
    negative_assertion = bool(NEGATIVE_ASSERTION_RE.search(claim))
    if negative_assertion:
        # Claims about what a source does *not* say are especially easy to invent from
        # missing context. Demand substantially more direct lexical support.
        need = max(need, (3 * len(tokens) + 4) // 5)
    if designation_in_claim and not negative_assertion:
        need = max(1, need - 1)
    return hits >= min(need, len(tokens))


def unsupported_citation_markers(
    answer: str,
    citations: list[Citation],
    chunks: dict[str, SourceChunk],
) -> list[int]:
    """Citation indices whose marker is missing, out of range, or does not support its claim."""
    invalid: list[int] = []
    for marker in CITE_MARKER.finditer(answer):
        index = int(marker.group(2))
        if index < 1 or index > len(citations):
            invalid.append(index)
            continue
        citation = citations[index - 1]
        claim = _claim_line_before_marker(answer, marker.start())
        if not citation_supports_claim(claim, citation, chunks.get(citation.chunk_id)):
            invalid.append(index)
    return invalid


def uncited_standard_claims(answer: str) -> list[str]:
    """Substantive paragraphs/bullets that name a standard but have no source marker."""
    issues: list[str] = []
    segments: list[str] = []
    for block in re.split(r"\n\s*\n", answer):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if any(line.startswith(("- ", "* ")) for line in lines):
            segments.extend(lines)
        elif lines:
            segments.append(" ".join(lines))

    for raw in segments:
        paragraph = raw.strip().lstrip("#-* ").strip()
        if not paragraph or paragraph.endswith(":") or CITE_MARKER.search(paragraph):
            continue
        if _standard_mentions(paragraph) and len(_meaningful_claim_tokens(paragraph)) >= 3:
            issues.append(paragraph)
    return issues


def validate_answer_citations(
    answer: str,
    citations: list[Citation],
    chunks: dict[str, SourceChunk],
) -> tuple[str, list[Citation]]:
    """Remove markers whose claim is not supported by the cited chunk; renumber + filter citations."""
    if not citations or not answer.strip():
        return answer, citations

    matches = list(CITE_MARKER.finditer(answer))
    if not matches:
        return answer, citations

    dropped = unsupported_citation_markers(answer, citations, chunks)

    if not dropped:
        return answer, citations

    # Remove dropped marker spans (keep supported markers verbatim)
    drop_set = set(dropped)
    parts: list[str] = []
    pos = 0
    for m in matches:
        idx = int(m.group(2))
        parts.append(answer[pos : m.start()])
        if idx in drop_set:
            pos = m.end()
        else:
            parts.append(m.group(0))
            pos = m.end()
    parts.append(answer[pos:])
    stripped = "".join(parts)

    # Remaining citation indices in first-appearance order
    remaining_order: list[int] = []
    seen: set[int] = set()
    for m in CITE_MARKER.finditer(stripped):
        idx = int(m.group(2))
        if idx < 1 or idx > len(citations):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        remaining_order.append(idx)

    if not remaining_order:
        # Every marker failed the lexical support check: strip them silently and keep the
        # citations list (it still reflects the excerpts used for drafting). No user-facing note.
        return stripped.rstrip(), citations

    old_to_new = {old: i + 1 for i, old in enumerate(remaining_order)}

    def renumber(mo: re.Match[str]) -> str:
        ws, old_s = mo.group(1), mo.group(2)
        old = int(old_s)
        if old not in old_to_new:
            return ""
        return f"{ws}[{old_to_new[old]}]"

    renumbered = CITE_MARKER.sub(renumber, stripped)

    new_citations = [citations[i - 1] for i in remaining_order]

    # Marker cleanup is silent — the diagnostic note used to leak into the user-facing answer.
    return renumbered.rstrip(), new_citations
