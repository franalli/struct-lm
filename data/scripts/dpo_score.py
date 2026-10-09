"""Stage 4 step 5: score every pool sample with verifiers and rules; no judge (2026-10-08
amendment, notes/decisions.md: the judge failed its benchmark and is out of pair-building).

  .venv/bin/python data/scripts/dpo_score.py            # -> data/dpo/work/scored.jsonl, results/dpo/score_log.json
  .venv/bin/python data/scripts/dpo_score.py --strict   # -> data/dpo/strict/work/scored.jsonl, results/dpo/score_log_strict.json

Reads results/runs/sft-from-cpt/samples/dpo_pool.jsonl (eval/sample.py's dpo_pool job). A sample
that doesn't end on </s> is dropped. Per format:
  closed_book  the verifier: the answer line states the gold fact (sft_judge.same_fact): 5, else 1.
               Form check for chosen: no repeated line, under 2x the median closed-book length.
  abstain      the SFT builder's rule: cites a passage or runs past 12 words = answered (1), else
               declined (5)
  grounded     rules (dpo_common.rule_checks, on the answer with chunk ids relabelled [P1]-[P4]):
               every rule's verdict is kept, so dpo_pairs.py can count pairs under the
               registered rules and under the amendment's (cite_valid, cites the gold passage, no
               false abstain): 5 if all of a rule set pass, else 1
  definition   not scored: no pairs (no verifier knows a definition's quality)

--strict (2026-10-09, the dpo-strict rerun): closed-book by the strict checker instead
(scorers.qa_strict on the answer line, plus exactly one non-empty line: the GRPO reward's rule);
79 of the as-run set's 458 closed-book chosen answers fail it (results/qa_strict/dpo_pairs.json).
Everything else is unchanged.
"""

import argparse
import json
import statistics
import sys
from collections import Counter

from dpo_common import AMENDED_GROUNDED, FLOOR, REPO, RULES, records, relabel, rule_checks
from sft_common import pmap, read_jsonl, write_jsonl
from sft_judge import same_fact

sys.path.insert(0, str(REPO / "eval"))
from scorers import answer_line, citations, qa_strict

SAMPLES = REPO / "results/runs/sft-from-cpt/samples/dpo_pool.jsonl"
SCORED = REPO / "data/dpo/work/scored.jsonl"
LOG = REPO / "results/dpo/score_log.json"
RULE_SETS = {"registered": RULES["grounded"], "amended": AMENDED_GROUNDED}


def no_repeated_line(t: str) -> bool:
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    return len(lines) == len(set(lines))


def strict_ok(t: str, rec: dict) -> bool:
    one_line = len([x for x in t.splitlines() if x.strip()]) == 1
    return one_line and qa_strict(answer_line(t), rec["gold"], rec["kind"])


def score_prompt(row: dict, rec: dict, cb_median: float, strict: bool = False) -> dict:
    fmt = row["format"]
    kept: list[dict] = [
        {**s, "k": k} for k, s in enumerate(row["samples"]) if s["finish_reason"] == "stop"
    ]
    out = {k: row[k] for k in ("id", "format", "prompt_token_ids")}
    out["dropped_no_eos"] = len(row["samples"]) - len(kept)
    for s in kept:
        t = s["text"].strip()
        if fmt == "closed_book":
            ok = (
                strict_ok(t, rec)
                if strict
                else same_fact(answer_line(t) or t, rec["gold"], rec["kind"])
            )
            s.update(eff=5.0 if ok else FLOOR, source="verifier", correct=ok,
                     form_ok=no_repeated_line(t) and s["n_tokens"] < 2 * cb_median)  # fmt: skip
        elif fmt == "abstain":
            declined = not (citations(t) or len(t.split()) > 12)
            s.update(eff=5.0 if declined else FLOOR, source="verifier", correct=declined)
        elif fmt == "grounded":
            checks = rule_checks(rec, relabel(t, rec.get("passages") or []))
            effs = {name: 5.0 if all(checks.get(r, True) for r in rs) else FLOOR
                    for name, rs in RULE_SETS.items()}  # fmt: skip
            s.update(eff=effs["amended"], eff_by_rules=effs, rules=checks, source="rule")
        else:
            s.update(eff=None, source="unpaired")
    out["samples"] = kept
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="closed-book by scorers.qa_strict")
    args = ap.parse_args()
    scored_path = REPO / "data/dpo/strict/work/scored.jsonl" if args.strict else SCORED
    log_path = REPO / "results/dpo/score_log_strict.json" if args.strict else LOG
    rows = read_jsonl(SAMPLES)
    recs = records({r["id"] for r in rows})
    cb = [s["n_tokens"] for r in rows if r["format"] == "closed_book" for s in r["samples"]
          if s["finish_reason"] == "stop"]  # fmt: skip
    cb_median = statistics.median(cb)
    scored = pmap(lambda r: score_prompt(r, recs[r["id"]], cb_median, args.strict), rows)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored_path, scored)
    log: dict = {"closed_book_median_tokens": cb_median, "rule_sets": RULE_SETS, "by_format": {},
                 "closed_book_verifier": "scorers.qa_strict" if args.strict else "sft_judge.same_fact"}  # fmt: skip
    for f in ("closed_book", "definition", "grounded", "abstain"):
        sel = [r for r in scored if r["format"] == f]
        flat = [s for r in sel for s in r["samples"]]
        d = {
            "prompts": len(sel),
            "samples": len(flat),
            "dropped_no_eos": sum(r["dropped_no_eos"] for r in sel),
            "score_histogram": dict(Counter(s["eff"] for s in flat if s.get("eff") is not None)),
            "mixed_prompts": sum(
                1
                for r in sel
                if len({s["eff"] for s in r["samples"] if s.get("eff") is not None}) == 2
            ),
        }
        if f == "grounded":
            d["rule_fails"] = dict(
                Counter(r for s in flat for r, ok in s["rules"].items() if not ok)
            )
            d["mixed_prompts_registered_rules"] = sum(
                1 for r in sel if len({s["eff_by_rules"]["registered"] for s in r["samples"]}) == 2
            )
        if f == "closed_book":
            d["form_fail_among_correct"] = sum(1 for s in flat if s["correct"] and not s["form_ok"])
        log["by_format"][f] = d
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
