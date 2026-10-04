"""The 3 few-shot items sit in every domain_qa prompt, so none may give away a scored item."""

import json
import re
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "eval/tasks"

# Scored items from the same passage as a few-shot item (eval/contamination.py, 2026-10-04):
# task versions 1 and 2 split the shots off by item, not by passage. Different facts, so kept in
# the frozen v2 set and tagged fewshot_passage_overlap; --task-version 3 drops them.
SHARED_PASSAGE = {"qa-0003", "qa-0056"}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (TASKS / name).read_text().splitlines()]


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def test_fewshot_disjoint_from_scored_items():
    shots, qa = load("fewshot.jsonl"), load("domain_qa.jsonl")
    assert len({norm(s["question"]) for s in shots}) == len(shots) == 3
    scored = {norm(d["question"]) for d in qa}
    assert not [s["question"] for s in shots if norm(s["question"]) in scored]
    by_chunk = {s["source_chunk"]: s for s in shots}
    shared = [d for d in qa if d["source_chunk"] in by_chunk]
    assert {d["id"] for d in shared} == SHARED_PASSAGE
    assert {d["id"] for d in qa if d.get("fewshot_passage_overlap")} == SHARED_PASSAGE
    for d in shared:
        assert norm(d["answer"]) != norm(by_chunk[d["source_chunk"]]["answer"])
