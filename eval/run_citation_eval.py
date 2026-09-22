#!/usr/bin/env python3
"""Run citation-focused eval cases against the loaded standards index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from standards_rag.chat import StandardsRagEngine  # noqa: E402
from standards_rag.citation_validation import unsupported_citation_markers  # noqa: E402
from standards_rag.env_bootstrap import (  # noqa: E402
    default_standards_index_path,
    load_dotenv_files,
)
from standards_rag.openai_answer import build_openai_answer_rewriter_from_env  # noqa: E402
from standards_rag.pinecone_hybrid import (  # noqa: E402
    attach_pinecone_index,
    pinecone_enabled_from_env,
)
from standards_rag.retrieval import InMemoryStandardsStore  # noqa: E402


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _check_case(engine: StandardsRagEngine, case: dict) -> tuple[bool, str, object]:
    response = engine.ask(case["question"], conversation_id=f"eval-{case['id']}")
    if case.get("must_refuse"):
        if not response.unsupported:
            return False, "expected unsupported refusal", response
        if response.citations:
            return False, "refusal unexpectedly included citations", response
        return True, "refused as expected", response

    if response.unsupported:
        return False, "unexpected unsupported response", response
    if not response.citations:
        return False, "no citations returned", response

    cited = {c.standard_id for c in response.citations}

    expected_standard = case.get("expected_standard_id")
    if expected_standard:
        if not any(expected_standard in sid for sid in cited):
            return False, f"expected citation for {expected_standard}, got {sorted(cited)}", response

    for expected in case.get("expected_standard_ids", []):
        if not any(expected in sid for sid in cited):
            return False, f"expected citation for {expected}, got {sorted(cited)}", response

    for forbidden in case.get("forbidden_standard_ids", []):
        if any(forbidden in sid for sid in cited):
            return False, f"forbidden citation {forbidden}, got {sorted(cited)}", response

    allowed = case.get("allowed_standard_ids")
    if allowed and any(not any(item in sid for item in allowed) for sid in cited):
        return False, f"citations outside allowlist {allowed}: {sorted(cited)}", response

    expected_section = case.get("expected_section")
    if expected_section:
        sections = {c.section for c in response.citations if c.section}
        if sections and expected_section not in sections:
            return False, f"expected section {expected_section}, got {sorted(sections)}", response

    expected_page = case.get("expected_page_start")
    if expected_page is not None:
        pages = [c.page_start for c in response.citations if c.page_start is not None]
        if pages and expected_page not in pages and not any(
            c.page_start is not None
            and c.page_end is not None
            and c.page_start <= expected_page <= c.page_end
            for c in response.citations
        ):
            return False, f"expected page {expected_page}, got pages {pages}", response

    citation_blob = " ".join(
        " ".join(
            [
                citation.standard_id or "",
                citation.title or "",
                citation.section or "",
                citation.quote or "",
            ]
        )
        for citation in response.citations
    )
    for keyword in case.get("expected_keywords", []):
        blob = f"{response.answer} {citation_blob}".lower()
        if keyword.lower() not in blob:
            return False, f"expected keyword '{keyword}' in answer", response

    chunks = dict(engine.store.chunks)
    chunks.update(engine.design_store.chunks)
    invalid = unsupported_citation_markers(response.answer, response.citations, chunks)
    if invalid:
        return False, f"unsupported inline citation markers: {invalid}", response

    return True, "passed", response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="use configured Pinecone retrieval and OpenAI answer rewriting",
    )
    parser.add_argument("--output", type=Path, help="write full case results as JSONL")
    args = parser.parse_args()

    load_dotenv_files()
    index_path = default_standards_index_path()
    if not index_path.exists():
        print(f"No index at {index_path}. Run standards-rag ingest first.")
        return 1

    store = InMemoryStandardsStore.load_json(index_path)
    answer_rewriter = None
    if args.live:
        if not pinecone_enabled_from_env():
            print("Live mode requested, but Pinecone is not configured.")
            return 1
        store = attach_pinecone_index(store)
        answer_rewriter = build_openai_answer_rewriter_from_env()
    engine = StandardsRagEngine(store, answer_rewriter=answer_rewriter)
    cases = _load_cases(Path(__file__).with_name("citation_eval.jsonl"))

    passed = 0
    records: list[str] = []
    for case in cases:
        ok, detail, response = _check_case(engine, case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {detail}")
        records.append(
            json.dumps(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "passed": ok,
                    "detail": detail,
                    "response": response.to_dict(),
                },
                ensure_ascii=False,
            )
        )
        if ok:
            passed += 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(records) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    print(f"\n{passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
