"""Optional OpenAI-backed answer rewriting grounded in retrieved citations."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from standards_rag.answer_prompts import (
    build_rewriter_system_prompt,
    build_title_system_prompt,
    is_comparison_question,
)
from standards_rag.models import Citation


def openai_rewriter_enabled() -> bool:
    flag = os.getenv("USE_OPENAI_ANSWER_REWRITER", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return False


def _openai_client_and_model() -> "tuple[Any, str] | None":
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        return None
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return OpenAI(api_key=api_key), model


def build_title_generator_from_env() -> "Callable[[str, str], str] | None":
    """Return an LLM title generator, or None when OpenAI is unavailable.

    Independent of the rewriter flag: titles are cheap and improve the sidebar UX.
    """
    bundle = _openai_client_and_model()
    if bundle is None:
        return None
    client, model = bundle
    system = build_title_system_prompt()

    def generate_title(question: str, answer: str) -> str:
        user = f"First question:\n{question}\n\nAssistant answer (for context):\n{answer[:600]}"
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=24,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    return generate_title


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

        system = build_rewriter_system_prompt(
            include_comparison_schema=is_comparison_question(question),
        )

        user = (
            f"Question:\n{question}\n\n"
            f"Draft answer (must be consistent with evidence; each [n] must match the draft):\n{draft_answer}\n\n"
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
