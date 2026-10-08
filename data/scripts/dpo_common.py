"""Shared by Stage 4's scorers (dpo_probe.py, dpo_judge_bench.py, dpo_score.py).

Scoring one prompt's samples, grounded and definition (user decisions 2026-10-08):
  1. rules first (rule 8; sft_judge.verify, the SFT builder's verifier): grounded citations valid,
     every sentence cited, no false abstain; definition one sentence, no passage reference. A
     sample failing one scores the floor (1.0), with no judge call.
  2. the rest, deduplicated, in one judge call (sft_judge.judge_list), in a hash-shuffled order;
     each sample's position is kept so a position effect can be checked. A sample failing a judge
     hard rule scores the floor too (the A5 keep rule needs both).
A failed call leaves its samples unscored (None), never at the floor.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from scorers import citations
from sft_common import h01
from sft_judge import judge_list, verify

WORK = REPO / "data/sft/work/judged.jsonl"
CHUNK_ID = re.compile(r"[\w.-]+:p\d+:c\d+\b")  # tests/test_sft_data.py's pattern
FLOOR = 1.0
MARGIN = 2.0
RULES = {  # the 2026-10-08 pre-registration (the probe and the benchmark ran on these)
    "grounded": ("citations", "answered", "every_sentence_cited"),
    "definition": ("one_sentence", "no_passage_ref"),
}
# the amendment (2026-10-08, judge out of pair-building): grounded pairs from cite_valid, citing the
# record's gold passage, and no false abstain
AMENDED_GROUNDED = ("citations", "cites_gold", "answered")


def relabel(text: str, passages: list[dict]) -> str:
    """The model cites chunk ids (the prompt shows them); the judge and verifier read [P1]-[P4].
    Every chunk id goes: one of the record's passages becomes its label, any other id (a made-up
    one, the prompt's [doc:p12:c0] example) becomes "unknown", so no chunk id reaches a judge
    prompt or the SFT builder's call cache (test_generator_prompts_clean)."""
    label = {p["chunk_id"]: p["label"] for p in passages}
    return CHUNK_ID.sub(lambda m: label.get(m[0], "unknown"), text)


def records(ids: set[str]) -> dict[str, dict]:
    """The SFT builder's records (data/sft/work/judged.jsonl: gold, passages, term) by eid."""
    out = {}
    for line in WORK.open():
        rec = json.loads(line)
        if rec["eid"] in ids:
            out[rec["eid"]] = rec
    return out


def rule_checks(rec: dict, text: str) -> dict[str, bool]:
    """Every rule's verdict on one (relabelled) answer: the SFT builder's verifier, plus for
    grounded whether it cites the passage the question was built from (gold_label)."""
    out = {r: ok for r, (ok, _) in verify({**rec, "answer": text}).items()}
    if rec["format"] == "grounded" and rec.get("gold_label"):
        out["cites_gold"] = rec["gold_label"] in citations(text)
    return out


def rule_fails(rec: dict, text: str, rules: tuple[str, ...] | None = None) -> list[str]:
    checks = rule_checks(rec, text)
    return [r for r in (rules or RULES[rec["format"]]) if r in checks and not checks[r]]


def order(key: str, n: int) -> list[int]:
    """A fixed shuffle of range(n) for one prompt (the same on every rerun, so the cache hits)."""
    return sorted(range(n), key=lambda i: h01(f"{key}:{i}"))


def judge_samples(rec: dict, texts: list[str], variant: str, key: str) -> list[dict]:
    """Per text: {eff, score, hard_pass, rule_fails, position, n_listed, source, reasoning}.
    texts are already relabelled. eff is the score pairing uses (floor on any failed rule)."""
    out: list[dict] = [{"rule_fails": rule_fails(rec, t)} for t in texts]
    for o in out:
        if o["rule_fails"]:
            o.update(eff=FLOOR, score=None, hard_pass=None, position=None, source="rule")
    distinct = list(dict.fromkeys(t for t, o in zip(texts, out) if not o["rule_fails"]))
    if not distinct:
        return out
    perm = order(key, len(distinct))
    listed = [distinct[i] for i in perm]
    grades = judge_list(rec, listed, variant)
    pos = {t: k for k, t in enumerate(listed, 1)}
    for t, o in zip(texts, out):
        if o["rule_fails"]:
            continue
        if grades is None:
            o.update(eff=None, score=None, hard_pass=None, position=pos[t], source="failed")
            continue
        g = grades[pos[t] - 1]
        hard_pass = all(g["hard"].values())
        o.update(
            eff=g["score"] if hard_pass else FLOOR,
            score=g["score"],
            hard_pass=hard_pass,
            hard=g["hard"],
            position=pos[t],
            source="judge",
            reasoning=g["reasoning"],
            quote=g.get("quote"),
        )
    for o in out:
        o["n_listed"] = len(listed)
    return out
