"""Optional OpenAI-backed answer rewriting grounded in retrieved citations."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from standards_rag.models import Citation


def openai_rewriter_enabled() -> bool:
    flag = os.getenv("USE_OPENAI_ANSWER_REWRITER", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return False


def build_openai_answer_rewriter_from_env() -> Callable[[str, str, list[Citation]], str] | None:
    if not openai_rewriter_enabled():
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when OpenAI answer rewriting is enabled.")

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the optional 'llm' dependencies to use OpenAI rewriting.") from exc

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    client = OpenAI(api_key=api_key)

    def rewrite(draft_answer: str, question: str, citations: list[Citation]) -> str:
        citation_lines = []
        for index, citation in enumerate(citations, start=1):
            cite_dict = citation.to_dict()
            citation_lines.append(f"[{index}] {cite_dict}")

        system = (
            "You are a careful technical assistant for standards documents.\n"
            "Rewrite the user's draft answer to be clear and conversational.\n"
            "Rules:\n"
            "- Use ONLY facts supported by the Evidence and Citation objects.\n"
            "- If the evidence is insufficient, say what is missing.\n"
            "- Do not introduce requirements, numbers, units, tests, or clauses not present in Evidence.\n"
            "- Preserve citation markers like [1], [2] when referring to evidence.\n"
            "- End with a 'Sources:' section listing the same citations in the same order.\n"
        )

        user = (
            f"Question:\n{question}\n\n"
            f"Draft answer (must be consistent with evidence):\n{draft_answer}\n\n"
            f"Citations (ground truth metadata):\n" + "\n".join(citation_lines)
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        cleaned = content.strip()
        if not cleaned:
            raise RuntimeError("OpenAI returned an empty answer.")
        return cleaned

    return rewrite
