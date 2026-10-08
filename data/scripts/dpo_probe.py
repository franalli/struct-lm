"""Stage 4's pre-pairing probe: how many prompts would yield a DPO pair from 4 samples.

  set -a; . ./.env; set +a
  .venv/bin/python eval/diversity.py collapse <run>      # the stop rule (collapse), first
  .venv/bin/python data/scripts/dpo_probe.py <run>      # -> results/diversity/<run>_pairable.json

Reads results/runs/<run>/samples/dpo_probe.jsonl (eval/sample.py's dpo_probe job: the first 20
prompts of data/dpo/prompts.jsonl x 4 samples at temperature 0.7). A prompt is pairable when its
samples hold at least one chosen and one rejected under the Stage 4 scorer (decided 2026-10-06,
before sampling; notes/decisions.md):
  closed_book   verifier: the answer line states the record's gold fact (sft_judge.same_fact)
  abstain       verifier, as the SFT builder's (sft_judge.verify): a sample answered if it cites a
                passage or runs past 12 words; otherwise it declined (chosen), whatever its wording
  grounded,     rules first, then the A5 rubric with all of a prompt's samples in one call
  definition    (dpo_common.judge_samples; user decision 2026-10-08), in the prompt variant the
                judge benchmark chose for the format (results/dpo/judge_bench.json); a sample
                failing a rule or a judge hard rule counts at the floor (1.0). Pairable if max -
                min >= 2. A format that failed the benchmark is scored by its rules alone (5 pass,
                1 fail). A failed judge call is left out, never scored; a prompt with fewer than
                2 scored samples is not counted
  replay        no scorer yet: counted, not scored
Pairable fraction doesn't gate: below half on a non-abstain format, that format is sampled 8 per
prompt instead of raising temperature further; abstain is sampled 8 regardless (2026-10-08). Writes
the sampling budget to data/dpo/budget.json for eval/sample.py's dpo_pool job. Samples are model output and are
only scored here; nothing is written into training data.
"""

import json
import sys
from collections import Counter

from dpo_common import FLOOR, MARGIN, REPO, judge_samples, records, relabel
from sft_common import pmap, read_jsonl, report_failures, set_rpm
from sft_judge import same_fact

sys.path.insert(0, str(REPO / "eval"))
from scorers import answer_line, citations

BENCH = REPO / "results/dpo/judge_bench.json"
BUDGET = REPO / "data/dpo/budget.json"
TEMPERATURE = 0.7  # the probe's; a stop (collapse) moves it to 1.0 with top_p 0.95, by hand


def verdicts(row: dict, rec: dict, variants: dict) -> dict:
    """Chosen / rejected per sample by the format's scorer, and whether the prompt is pairable."""
    texts = [s["text"].strip() for s in row["samples"]]
    fmt = row["format"]
    if fmt == "closed_book":
        ok = [same_fact(answer_line(t) or t, rec["gold"], rec["kind"]) for t in texts]
        return {"scorer": "verifier", "chosen": ok, "pairable": any(ok) and not all(ok)}
    if fmt == "abstain":
        ok = [not (citations(t) or len(t.split()) > 12) for t in texts]
        return {"scorer": "verifier", "chosen": ok, "pairable": any(ok) and not all(ok)}
    if fmt in ("grounded", "definition"):
        texts = [relabel(t, rec.get("passages") or []) for t in texts]
        variant = variants.get(fmt)
        if variant is None:  # failed the benchmark: rules only
            from dpo_common import rule_fails

            eff = [FLOOR if rule_fails(rec, t) else 5.0 for t in texts]
            scored, failed, scorer = eff, 0, "rules"
        else:
            got = judge_samples(rec, texts, variant, f"probe:{row['id']}")
            scored = [g["eff"] for g in got if g["eff"] is not None]
            failed, scorer = sum(g["eff"] is None for g in got), f"judge:{variant}"
            eff = scored
        return {
            "scorer": scorer,
            "effective": eff,
            "failed_samples": failed,
            "margin": round(max(scored) - min(scored), 2) if len(scored) >= 2 else None,
            "pairable": max(scored) - min(scored) >= MARGIN if len(scored) >= 2 else None,
        }
    return {"scorer": None, "pairable": None}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = sys.argv[1]
    set_rpm(30)
    bench = json.loads(BENCH.read_text())  # the benchmark comes first (2026-10-08)
    variants = {f: v["variant"] if v["passed"] else None for f, v in bench["chosen"].items()}
    rows = read_jsonl(REPO / f"results/runs/{run}/samples/dpo_probe.jsonl")
    work = records({r["id"] for r in rows})

    def one(r: dict) -> dict:
        if r["id"] not in work:
            return {"id": r["id"], "format": r["format"], "scorer": None, "pairable": None}
        return {"id": r["id"], "format": r["format"], **verdicts(r, work[r["id"]], variants)}

    per = pmap(one, rows)
    by_format = {}
    for f in sorted({x["format"] for x in per}):
        sel = [x for x in per if x["format"] == f and x["pairable"] is not None]
        n = len([x for x in per if x["format"] == f])
        by_format[f] = {
            "prompts": n,
            "scored": len(sel),
            "pairable": sum(x["pairable"] for x in sel),
            "pairable_fraction": round(sum(x["pairable"] for x in sel) / len(sel), 4)
            if sel
            else None,
        }
    budget = {
        f: 8
        for f, v in by_format.items()
        if f == "abstain" or (v["pairable_fraction"] is not None and v["pairable_fraction"] < 0.5)
    }
    res = {
        "run": run,
        "rule": "pairable = >= 1 chosen and >= 1 rejected of 4 (verifier: closed_book, abstain; "
        f"rules + listwise judge margin >= {MARGIN}: grounded, definition); doesn't gate; < 0.5 on "
        "a non-abstain format -> 8 samples per prompt for it; abstain 8 regardless",
        "variants": variants,
        "by_format": by_format,
        "samples_per_prompt_8": sorted(budget),
        "scorers": dict(Counter(x["scorer"] for x in per)),
        "per_prompt": per,
    }
    res["failed_judge_samples"] = sum(x.get("failed_samples", 0) for x in per)
    out = REPO / "results/diversity" / f"{run}_pairable.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    formats = ("closed_book", "definition", "grounded", "abstain")
    BUDGET.write_text(
        json.dumps(
            {
                "temperature": TEMPERATURE,
                "top_p": None,
                "n": {f: 8 if f in budget else 4 for f in formats},
                "from": str(out.relative_to(REPO)),
            },
            indent=2,
        )
        + "\n"
    )
    report_failures()
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
