"""The 3 few-shot items sit in every domain_qa prompt, so none may give away a scored item."""

import json
import re
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "eval/tasks"

# Task versions 1 and 2 split the shots off by item, not by passage, so qa-0003 and qa-0056 shared
# a passage with a few-shot item (different facts; eval/contamination.py, 2026-10-04). The
# committed set is v3 (Stage 3 on), which splits by passage: no scored item shares one.


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (TASKS / name).read_text().splitlines()]


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def test_fewshot_disjoint_from_scored_items():
    shots, qa = load("fewshot.jsonl"), load("domain_qa.jsonl")
    assert len({norm(s["question"]) for s in shots}) == len(shots) == 3
    scored = {norm(d["question"]) for d in qa}
    assert not [s["question"] for s in shots if norm(s["question"]) in scored]
    shot_chunks = {s["source_chunk"] for s in shots}
    assert not [d["id"] for d in qa if d["source_chunk"] in shot_chunks]
    assert not [d["id"] for d in qa if d.get("fewshot_passage_overlap")]
