"""System-prompt blocks for claim-level grounding and section-aware answers."""

from __future__ import annotations

import re

CLAIM_LEVEL_GROUNDING = """
You are answering questions using retrieved ASTM standard excerpts.
For every factual claim about applicability, exclusions, scope, limitations, calculations, or interpretation,
cite the most specific retrieved evidence that directly supports that claim (use [1], [2], … markers that match the draft).

Before answering:
1. Identify the user’s requested dimensions (e.g. applicable methods, exclusions, measured property, test conditions, design limitations).
2. For each dimension, extract only statements explicitly supported by the retrieved text in the draft.
3. Do not infer exclusions unless the document explicitly states them.
4. Prefer material from Scope, Summary of Test Method, Significance and Use, and Terminology for broad applicability claims.
5. Prefer Procedure, Calculation, Report, or Precision/Bias only when the claim concerns procedural steps, equations, reporting, or precision.
6. If evidence supports only part of a sentence, split the sentence so each claim can be traced to supporting text.
7. If the retrieved text does not directly support a point, say the retrieved excerpts do not establish that point.

When citing with [n]:
- Place the marker at the end of the sentence or clause the cited excerpt supports.
- Do not cite a chunk merely because it is from the same standard.
- Do not use a calculation/reporting passage to support a scope or applicability claim unless that passage explicitly states the scope.
- If multiple standards are compared, tie each [n] to the claim about that standard.
"""

REWRITER_GROUNDING = """
You are turning retrieved standard excerpts into a clear, explanatory answer for an engineer.
- Explain what the cited evidence means in plain, well-structured prose. Paraphrase and synthesize;
  do NOT copy the excerpt sentences verbatim and do NOT just list raw quotes.
- Lead with a direct, helpful answer to the question, then expand with the supporting detail.
- Every factual claim must stay grounded in the retrieved excerpts. Do not add facts, numbers,
  standards, or applicability statements that are not supported by the cited evidence.
- Attach the inline [n] marker(s) to the claim each excerpt supports, using the same numbering as the
  draft. Keep a marker on every sentence that states a sourced fact so citations stay traceable.
- Prefer short paragraphs or tight bullet points over a wall of text.
- If the evidence only partially answers the question, say plainly what the excerpts do and do not establish.
"""

COMPARISON_ANSWER_SCHEMA = """
When the question compares standards or asks for applicability across several methods, structure the answer as:
- Applicable method(s): designation and what property or behavior each method addresses (only as stated in evidence).
- Exclusions / non-applicability: only exclusions explicitly stated in the retrieved text.
- Test type and conditions: e.g. index vs performance, key stresses, fluids, specimens—only if stated.
- Why values may not match field design: separate lab-condition limits, index-test limits, and material/interface limits—only if stated.
- Missing support: briefly list any part of the question not answered by the retrieved excerpts.
"""

CITATION_AUDIT = """
Quick check before you finish:
- Each [n] should still attach to the same claim as in the draft.
- Adjust wording only if the cited draft meaning stays the same.
"""

MATH_AND_FORMAT_RULES = """
- For mathematical expressions, use LaTeX inside Markdown math delimiters:
  - Inline: $...$ (example: $v$, $A$, $n = \\left(\\frac{t_v}{A}\\right)^2$)
  - Display (standalone equation lines): $$...$$
- Never write \\left$ or \\right$; use \\left( ... \\right) and keep the whole expression inside one $...$ pair.
- Put each COMPLETE equation inside exactly ONE delimiter pair; NEVER place a $ inside another expression. Bad: $$\\frac{A_t}{\\ln$S$ \\cdot $h_1-h_2$}$$  Good: $$k_T = a_T \\cdot \\frac{A_t}{\\ln(S)\\,(h_1 - h_2)}$$
- Do not wrap math in plain parentheses like ( v ) or ( n = \\frac{a}{b} ); use $...$ instead.
- Do not rearrange equations unless the rearrangement is explicitly supported by the cited evidence text.
- Do not add a 'Sources:' or bibliography section; the app shows full citations separately.
"""


_MATH_CMD_RE = re.compile(
    r"\\(?:d?frac|cdot|times|div|sqrt|ln|log|sum|int|left|right|partial|"
    r"leq|geq|le|ge|neq|approx|pm|sigma|alpha|beta|gamma|Delta|theta|mu|rho)\b"
)


def _is_bare_equation_line(line: str) -> bool:
    """True for a standalone-equation line (few prose words) that uses LaTeX math."""
    if not _MATH_CMD_RE.search(line):
        return False
    prose = _MATH_CMD_RE.sub(" ", line)
    prose = re.sub(r"\\[a-zA-Z]+", " ", prose).replace("$", " ")
    prose = re.sub(r"[^A-Za-z ]+", " ", prose)
    long_words = [w for w in prose.split() if len(w) >= 4]
    return len(long_words) <= 2


def sanitize_math_markdown(text: str) -> str:
    """Repair common LLM math-delimiter mistakes so KaTeX renders cleanly.

    Converts ``\\( \\)``/``\\[ \\]`` to ``$``/``$$``, and for a standalone equation
    line that nested or dropped ``$`` delimiters, strips the stray ``$`` and wraps the
    whole expression once in ``$$ ... $$``. Fixes the failure where an answer printed
    raw LaTeX like ``\\frac{A_t}{\\ln$S$ \\cdot $h_1-h_2$}``. Prose lines with valid
    inline ``$...$`` math are left untouched.
    """
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$$ {m.group(1).strip()} $$", text, flags=re.S)
    fixed: list[str] = []
    in_code = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            fixed.append(line)
        elif in_code or not stripped or stripped.startswith("$$"):
            fixed.append(line)
        elif _is_bare_equation_line(stripped):
            body = re.sub(r"\s+", " ", stripped.replace("$", " ")).strip()
            fixed.append(f"$$ {body} $$")
        else:
            fixed.append(line)
    return "\n".join(fixed)


def is_comparison_question(question: str) -> bool:
    """Broad 'compare / contrast' detection; aligned with chat routing."""
    lowered = question.lower()
    if any(
        w in lowered
        for w in ("compare", "comparison", "difference", "differences", "versus", " vs ")
    ):
        return True
    if "each standard" in lowered or ("across the" in lowered and "standard" in lowered):
        return True
    return False


def build_rewriter_system_prompt(*, include_comparison_schema: bool) -> str:
    parts = [
        "You are a knowledgeable technical assistant for geosynthetics standards.\n",
        "The draft below is a list of raw excerpts retrieved from standards documents. "
        "Rewrite it into a clear, conversational explanation that an engineer can read and "
        "understand, as if you are teaching the concept.\n",
        "Explain what the retrieved evidence means in your own words; do NOT just repeat the "
        "excerpts verbatim. Synthesize the points into coherent prose and short paragraphs.\n",
        "Stay strictly grounded: only state facts, numbers, and conclusions that the retrieved "
        "evidence supports. Keep every [1], [2], … marker attached to the claim it supports.\n",
        "\n",
        REWRITER_GROUNDING.strip(),
        "\n\n",
    ]
    if include_comparison_schema:
        parts.append(COMPARISON_ANSWER_SCHEMA.strip())
        parts.append("\n\n")
    parts.append(CITATION_AUDIT.strip())
    parts.append("\n\n")
    parts.append(MATH_AND_FORMAT_RULES.strip())
    parts.append(
        "\n\nAdditional rules:\n"
        "- Lead with a direct, plain-language answer to the question, then explain the supporting "
        "detail from the evidence.\n"
        "- Paraphrase and explain the evidence instead of quoting it word-for-word.\n"
        "- Do not add new facts, numbers, or standards beyond what the evidence supports.\n"
        "- If the evidence is insufficient for part of the question, say what is missing.\n"
        "- Preserve citation markers [1], [2] when referring to evidence.\n"
        "- If the draft says meanings/usages are context-dependent, keep that framing.\n"
        "- Keep per-context bullet points when they are present in the draft.\n"
        "- Never mention the rewrite process (for example, do not say 'Here is a clearer version "
        "of your draft').\n"
    )
    return "".join(parts)


def build_title_system_prompt() -> str:
    return (
        "You generate a very short title for a saved research chat about geosynthetics and "
        "ASTM/ISO standards.\n"
        "Rules:\n"
        "- 2 to 4 words, Title Case. Keep it under 30 characters so it fits a narrow sidebar.\n"
        "- Prefer the standard designation or core test concept alone (e.g. 'D4595 Wide-Width' "
        "or 'GCL Permeability').\n"
        "- No quotation marks, no trailing punctuation, no emojis, no filler words.\n"
        "- Output ONLY the title text, nothing else."
    )
