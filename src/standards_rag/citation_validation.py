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


def _standard_id_compact(value: str) -> str:
    return COMPACT_SID_RE.sub("", value.lower())


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
    if designation_in_claim:
        need = max(1, need - 1)
    return hits >= min(need, len(tokens))


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

    dropped: list[int] = []
    for m in matches:
        idx = int(m.group(2))
        if idx < 1 or idx > len(citations):
            dropped.append(idx)
            continue
        claim = _claim_line_before_marker(answer, m.start())
        cit = citations[idx - 1]
        chunk = chunks.get(cit.chunk_id)
        if not citation_supports_claim(claim, cit, chunk):
            dropped.append(idx)

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
        rem = ", ".join(str(x) for x in sorted(set(dropped)))
        note = (
            "\n\nCitation verification: No inline citation markers passed lexical support checks "
            f"(removed indices: {rem}). The citations list below still reflects "
            "retrieved excerpts used for drafting."
        )
        return stripped.rstrip() + note, citations

    old_to_new = {old: i + 1 for i, old in enumerate(remaining_order)}

    def renumber(mo: re.Match[str]) -> str:
        ws, old_s = mo.group(1), mo.group(2)
        old = int(old_s)
        if old not in old_to_new:
            return ""
        return f"{ws}[{old_to_new[old]}]"

    renumbered = CITE_MARKER.sub(renumber, stripped)

    new_citations = [citations[i - 1] for i in remaining_order]

    note = (
        "\n\nCitation verification: Removed citation markers whose preceding lines were not "
        f"directly supported by the retrieved excerpts (indices: {', '.join(str(x) for x in sorted(set(dropped)))})."
    )
    return renumbered.rstrip() + note, new_citations
