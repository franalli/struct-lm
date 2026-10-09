"""Stage 5's reward: three rule-based functions of one completion and its task's verifier.

  total = format (0.1) + correctness (0.9, only when format passed) + length (0 to -0.1)

Format (Magistral's gate; a completion failing it scores 0 + 0 + length):
  every kind   terminated: the completion ends on </s> (a truncated one is never correct)
  closed_book  exactly one non-empty line (what the prompts ask for: "A:" few-shot or "Reply with
               just the value"; 0 of 4,336 Stage 4 samples wrote an "Answer:" label)
  grounded     at least one [bracket], every bracket one of the task's four chunk ids
  abstain      the abstain sentence, or a grounded-form answer (which then fails correctness)

Correctness, closed_book: eval/scorers.qa_strict on the answer line (the strict checker, written
  2026-10-09 when Stage 4's same_fact was about to become this reward: it passed any piece of the
  gold and judged multi-number golds on their first number). Values by every gold number in a
  compatible unit with no conflicting value; identifiers and terms by the whole gold, never a piece
  of it; one candidate (an alternative or a range the gold lacks is a format failure here, "two
  values on the answer line"). Rules and fixtures: scorers.py, tests/test_qa_strict.py.
Correctness, grounded: cites the gold passage, at most max_distinct passages (closes "cite all
four"; 371 of 374 teacher answers cite <= 2), no abstain sentence.
Correctness, abstain: the reply is exactly the abstain sentence.

Length: 0 up to MAX_COMPLETION - 64 tokens, linear to -0.1 at MAX_COMPLETION, on every completion,
so a truncated one is strictly the worst.

Pure string handling on top of eval/scorers.py: the probe scorer (data/scripts/grpo_probe.py) and
the trainer (train/grpo.py) import this file, so the probe's window and the training reward agree.
"""

import json
import sys
from pathlib import Path

# eval/scorers.py: locally next to train/, in the Modal training image under /root/eval
for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "scorers.py").exists():
        sys.path.insert(0, str(_d))
        break
from scorers import answer_line, citations, one_candidate, qa_strict

EOS = 2
MAX_COMPLETION = 256
FORMAT_W, CORRECT_W, LENGTH_W = 0.1, 0.9, 0.1
ABSTAIN = "Not in the provided passages."  # sft_common.ABSTAIN_REPLY, the prompts' exact sentence


def verifier_for(fmt: str, rec: dict) -> dict:
    """The task's verifier from its SFT builder record (data/sft/work/judged.jsonl)."""
    if fmt == "closed_book":
        return {"type": "closed_book", "gold": rec["gold"], "kind": rec["kind"], "tolerance": 0.02}
    ids = [p["chunk_id"] for p in rec["passages"]]
    if fmt == "grounded":
        (gold,) = [p["chunk_id"] for p in rec["passages"] if p["label"] == rec["gold_label"]]
        return {"type": "cite", "gold_chunk": gold, "chunk_ids": ids, "max_distinct": 2}
    assert fmt == "abstain", fmt
    return {"type": "abstain", "phrase": ABSTAIN, "chunk_ids": ids}


def is_abstain(text: str) -> bool:
    return text.strip().rstrip(".").strip().lower() == ABSTAIN.rstrip(".").lower()


def _cited_ok(text: str, ids: list[str]) -> tuple[bool, list[str]]:
    cites = citations(text)
    return bool(cites) and all(c in ids for c in cites), cites


def score(text: str, ids: list[int], v: dict) -> dict:
    """One completion against one verifier: {format, correct, terminated, n_tokens, length}."""
    terminated = bool(ids) and ids[-1] == EOS
    t, fmt, correct = v["type"], False, False
    if t == "closed_book":
        line = answer_line(text)
        fmt = (
            len([x for x in text.strip().splitlines() if x.strip()]) == 1
            and bool(line)
            and one_candidate(line, v["gold"], v["kind"])
        )
        correct = qa_strict(line, v["gold"], v["kind"], v["tolerance"])
    elif t == "cite":
        fmt, cites = _cited_ok(text, v["chunk_ids"])
        correct = (
            v["gold_chunk"] in cites
            and len(set(cites)) <= v["max_distinct"]
            and ABSTAIN.rstrip(".").lower() not in text.lower()
        )
    elif t == "abstain":
        fmt = is_abstain(text) or _cited_ok(text, v["chunk_ids"])[0]
        correct = is_abstain(text)
    else:
        raise ValueError(f"unknown verifier type {t!r}")
    fmt = fmt and terminated
    n = len(ids)
    free = MAX_COMPLETION - 64
    return {
        "format": fmt,
        "correct": fmt and correct,
        "terminated": terminated,
        "n_tokens": n,
        "length": -LENGTH_W * min(1.0, max(0, n - free) / (MAX_COMPLETION - free)),
    }


def total(s: dict) -> float:
    return FORMAT_W * s["format"] + CORRECT_W * s["correct"] + s["length"]


# TRL reward functions: f(prompts=, completions=, completion_ids=, **dataset columns) -> [float].
# The dataset's `verifier` column is the verifier as a JSON string (one Arrow type for every kind).
def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


def _scores(completions, completion_ids, verifier) -> list[dict]:
    return [
        score(_text(c), list(ids), json.loads(v) if isinstance(v, str) else v)
        for c, ids, v in zip(completions, completion_ids, verifier, strict=True)
    ]


def format_reward(completions, completion_ids, verifier, **_) -> list[float]:
    return [FORMAT_W * s["format"] for s in _scores(completions, completion_ids, verifier)]


def correctness_reward(completions, completion_ids, verifier, **_) -> list[float]:
    return [CORRECT_W * s["correct"] for s in _scores(completions, completion_ids, verifier)]


def length_penalty(completions, completion_ids, verifier, **_) -> list[float]:
    return [s["length"] for s in _scores(completions, completion_ids, verifier)]
