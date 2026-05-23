#!/usr/bin/env python3
"""Run citation-focused eval cases against the loaded standards index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from standards_rag.chat import StandardsRagEngine
from standards_rag.env_bootstrap import default_standards_index_path, load_dotenv_files
from standards_rag.retrieval import InMemoryStandardsStore


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _check_case(engine: StandardsRagEngine, case: dict) -> tuple[bool, str]:
    response = engine.ask(case["question"])
    if case.get("must_refuse"):
        if not response.unsupported:
            return False, "expected unsupported refusal"
        return True, "refused as expected"

    if response.unsupported:
        return False, "unexpected unsupported response"
    if not response.citations:
        return False, "no citations returned"

    expected_standard = case.get("expected_standard_id")
    if expected_standard:
        cited = {c.standard_id for c in response.citations}
        if not any(expected_standard in sid for sid in cited):
            return False, f"expected citation for {expected_standard}, got {sorted(cited)}"

    expected_section = case.get("expected_section")
    if expected_section:
        sections = {c.section for c in response.citations if c.section}
        if sections and expected_section not in sections:
            return False, f"expected section {expected_section}, got {sorted(sections)}"

    expected_page = case.get("expected_page_start")
    if expected_page is not None:
        pages = [c.page_start for c in response.citations if c.page_start is not None]
        if pages and expected_page not in pages and not any(
            c.page_start is not None
            and c.page_end is not None
            and c.page_start <= expected_page <= c.page_end
            for c in response.citations
        ):
            return False, f"expected page {expected_page}, got pages {pages}"

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
            return False, f"expected keyword '{keyword}' in answer"

    return True, "passed"


def main() -> int:
    load_dotenv_files()
    index_path = default_standards_index_path()
    if not index_path.exists():
        print(f"No index at {index_path}. Run standards-rag ingest first.")
        return 1

    store = InMemoryStandardsStore.load_json(index_path)
    engine = StandardsRagEngine(store)
    cases = _load_cases(Path(__file__).with_name("citation_eval.jsonl"))

    passed = 0
    for case in cases:
        ok, detail = _check_case(engine, case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {detail}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
