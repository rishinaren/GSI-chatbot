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
You are turning retrieved source excerpts into a clear, explanatory answer for an engineer.
- Explain what the cited evidence means in plain, well-structured prose. Paraphrase and synthesize;
  do NOT copy the excerpt sentences verbatim and do NOT just list raw quotes.
- Lead with a direct, helpful answer to the question, then expand with the supporting detail.
- Every factual claim must stay grounded in the retrieved excerpts. Do not add facts, numbers,
  standards, or applicability statements that are not supported by the cited evidence.
- Never infer that a standard omits, forbids, or does not require something merely because the
  retrieved excerpts do not mention it. Say that the excerpts do not establish the point.
- Attach the inline [n] marker(s) to the claim each excerpt supports, using the same numbering as the
  draft. Keep a marker on every sentence that states a sourced fact so citations stay traceable.
- Prefer short paragraphs or tight bullet points over a wall of text.
- If the evidence only partially answers the question, say plainly what the excerpts do and do not establish.
"""

DESIGN_GUIDANCE_RULES = """
Some citations have `source_kind: design_guidance`. They come from Robert M. Koerner's
Designing with Geosynthetics, 6th Edition (2012), not from a current standard.
- Call this material "design guidance" or name the book. Never call it a requirement,
  standard, code, or current specification.
- Do not use "shall", "must", or "required" for a claim supported only by design guidance.
- When both source kinds are present, distinguish current standards evidence from design
  guidance. A standard controls if the sources differ.
- Preserve worked equations, factors of safety, assumptions, and application limits only
  when the cited excerpt supports them.
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
- Put each [n] right after the sentence it supports, not at the end of the paragraph.
  Begin every paragraph and every bullet with a sourced sentence that carries its marker,
  including the opening answer sentence. Any sentence that names a standard needs a marker.
- Plain-language explanation may follow a sourced sentence in the same paragraph without a
  marker, as long as it adds no new facts, numbers, requirements, uses or standard names.
- Do not add a closing summary paragraph ("In summary", "Overall"); end on the last
  sourced point.
- Do not add sentences about how important, crucial or essential something is, about
  typical applications, about results being "accurate and reliable", or about how a
  standard is developed or reviewed, unless an excerpt says so.
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
    in_display = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            fixed.append(line)
        elif not in_code and stripped == "$$":
            # A bare "$$" opens or closes a display block; the lines between are
            # already math and must not be wrapped again ("$$\n$$ x $$\n$$").
            in_display = not in_display
            fixed.append(line)
        elif in_display and stripped.startswith("$$") and stripped.endswith("$$"):
            fixed.append(stripped.strip("$").strip())
        elif in_code or in_display or not stripped or stripped.startswith("$$"):
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


ATTACHED_DOCUMENT_RULES = """
The user attached their own document to this question ({names}). Evidence whose citation
metadata says `'source_kind': 'attachment'` comes from that upload; other evidence is a
published standard or clearly labeled design guidance from the library.
- Keep the two apart in every sentence. Say "your document" (or its file name) for the
  upload and name the standard for library evidence. Never present the upload as though it
  were published literature, and never let it stand in for what a standard requires.
- When the question compares the two, answer in that shape: what the attached document
  says, what the standards require, and where they agree, differ, or go unaddressed.
- The upload is not authoritative. If it conflicts with a cited standard, say so and let
  the standard carry the requirement.
- If the retrieved standards do not cover something the attached document raises, say the
  library does not establish it rather than filling the gap from general knowledge.
"""


ANSWER_LEVELS = ("beginner", "intermediate", "advanced")
DEFAULT_ANSWER_LEVEL = "intermediate"


def normalize_answer_level(value: object) -> str | None:
    """The level name if ``value`` is one, else None."""
    level = str(value or "").strip().lower()
    return level if level in ANSWER_LEVELS else None


# How deep to go, by who is reading. The three readers want the same facts and the
# same citations; what changes is how much is explained and how much is assumed.
# Advanced is deliberately "more depth, not harder words" - a specialist wants the
# conditions and limits spelled out, not the prose made denser.
ANSWER_LEVEL_RULES = {
    "beginner": """
Reader level: BEGINNER - someone new to geosynthetics (a student, a contractor or owner's
representative, or an engineer from another field).
- Open with one or two plain sentences that answer the question directly, keeping the key
  words of the excerpt (the property, the material, the test) so the source stays traceable.
- If the question is about a property or test, say in everyday words what that property is
  before describing how it is tested.
- Explain each technical term right after it is used, in a few everyday words, and keep the
  term itself so the reader learns it (for example "permittivity, which is how easily water
  passes through the fabric"). Do not leave any jargon or abbreviation unexplained.
- If the excerpts say what the result is used for, say that plainly. Do not invent uses,
  project examples or consequences the excerpts do not state.
- Leave out equations, specimen dimensions, rates, pressures, unit conversions and
  clause-level detail. If one number is central to the answer, give it and say in words what
  it means.
- When comparing standards, say in one sentence what they share, then what is different
  about each, in plain words.
- About 100 to 160 words, in two or three short paragraphs. No tables, no headings, no
  bullet lists.
""",
    "intermediate": """
Reader level: INTERMEDIATE - a practicing engineer or technician who knows the basics of
geosynthetics. Use standard terminology and briefly explain less common terms. Give the key
detail that answers the question (what is measured, how, and the main conditions or limits
the evidence states) without walking through every clause.
""",
    "advanced": """
Reader level: ADVANCED - a specialist such as a design engineer, lab manager or
specification writer. Advanced means more depth and precision, not harder words: write
plain, direct sentences with no buzzwords, filler or textbook background.
- Lead with a one- or two-sentence precise answer.
- Then give the specifics the excerpts support, one fact per bullet with a short bold label,
  each bullet ending with its marker. Look for test conditions, specimen size and
  conditioning, rates, equations (with each variable defined), values and tolerances,
  units, what must be reported, precision and bias, and stated limitations or exclusions.
- Include only the items the excerpts actually give and skip the rest without comment.
  Never say a detail is "implied", and do not infer what a standard omits.
- Do not define basic terms. Use a short table only to compare values across standards.
- About 200 to 400 words, depending on how much the excerpts support.
""",
}

# Last in the prompt, where a small model weighs it most. These phrases add no facts
# and were the bulk of what made answers read as padded.
PLAIN_WORDS_RULE = """
Never use the words "crucial", "essential", "comprehensive" or "robust", and never say a
method "ensures accurate and reliable results". State the fact and stop.
"""

REPHRASE_RULES = """
This answer was already given to the user at another reader level. The draft holds that
answer followed by the full text of the excerpts it cites. Write a new answer to the same
question for the reader level below. Do not contradict the answer already given, but do not
copy its structure either: restructure freely. A beginner version should be shorter and
simpler and drop detail; an advanced version should add specifics from the excerpt text,
each attached to the matching [n] marker. Copy any equation exactly as it appears in the
answer already given; never rearrange or re-derive one. Do not mention that this is a
rewrite or name the reader level.
"""


def build_rewriter_system_prompt(
    *,
    include_comparison_schema: bool,
    attachment_names: list[str] | None = None,
    include_design_guidance: bool = False,
    level: str = DEFAULT_ANSWER_LEVEL,
    rephrase: bool = False,
) -> str:
    source_description = (
        "standards and non-normative engineering design guidance"
        if include_design_guidance
        else "standards documents"
    )
    level = normalize_answer_level(level) or DEFAULT_ANSWER_LEVEL
    reader = {
        "beginner": "someone new to geosynthetics",
        "intermediate": "an engineer",
        "advanced": "a specialist",
    }[level]
    parts = [
        "You are a knowledgeable technical assistant for geosynthetics.\n",
        f"The draft below is a list of raw excerpts retrieved from {source_description}. "
        f"Rewrite it into a clear, conversational explanation that {reader} can read and "
        "understand, as if you are teaching the concept. Write for the reader level given "
        "at the end of these instructions.\n",
        "Explain what the retrieved evidence means in your own words; do NOT just repeat the "
        "excerpts verbatim. Synthesize the points into coherent prose and short paragraphs.\n",
        "Stay strictly grounded: only state facts, numbers, and conclusions that the retrieved "
        "evidence supports. Keep every [1], [2], … marker attached to the claim it supports.\n",
        "\n",
        REWRITER_GROUNDING.strip(),
        "\n\n",
    ]
    # The five-part comparison layout is too much for a newcomer; the beginner rules
    # carry their own one-line way to compare.
    if include_comparison_schema and level != "beginner":
        parts.append(COMPARISON_ANSWER_SCHEMA.strip())
        parts.append("\n\n")
    if include_design_guidance:
        parts.append(DESIGN_GUIDANCE_RULES.strip())
        parts.append("\n\n")
    if attachment_names:
        parts.append(ATTACHED_DOCUMENT_RULES.format(names=", ".join(attachment_names)).strip())
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
        "- Name the issuing body of each standard you cite (for example 'ASTM D4595', "
        "'GRI GM13', 'ISO 10319'). Label book evidence as design guidance instead. When "
        "the question compares sources, make clear which source each part draws on.\n"
        "- Never mention the rewrite process (for example, do not say 'Here is a clearer version "
        "of your draft').\n"
    )
    if rephrase:
        parts.append("\n" + REPHRASE_RULES.strip() + "\n")
    parts.append("\n" + ANSWER_LEVEL_RULES[level].strip())
    parts.append("\n\n" + PLAIN_WORDS_RULE.strip())
    return "".join(parts)


def build_title_system_prompt() -> str:
    return (
        "You generate a very short title for a saved research chat about geosynthetics and "
        "ASTM/ISO/GRI standards.\n"
        "Rules:\n"
        "- 2 to 3 words. Title Case. HARD LIMIT 26 characters so the whole title is visible "
        "in a narrow sidebar - shorter is better.\n"
        "- Lead with the standard designation or the core test concept (e.g. 'D4595 Wide-Width', "
        "'GCL Permeability', 'GM13 Wrinkles').\n"
        "- Do NOT prefix with the body name ('ASTM'/'GRI') unless it is the only identifier; the "
        "designation already implies it and the prefix wastes width.\n"
        "- No quotation marks, no trailing punctuation, no emojis, no filler words.\n"
        "- Output ONLY the title text, nothing else."
    )
