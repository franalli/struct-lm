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
  grounded,     the A5 judge (sft_judge.judge: Mistral Large 3, the format's ch. 12 rubric, the
  definition    gold fact as a hard rule) scores each sample; a sample failing a hard rule counts
                at the floor (1.0), as the judge's own keep rule requires both. Pairable if max -
                min >= 2. A failed judge call is left out, never scored; a prompt with fewer than
                2 scored samples is not counted
  replay        no scorer yet: counted, not scored
Pairable fraction doesn't gate: below half on a non-abstain format, that format is sampled 8 per
prompt instead of raising temperature further. The judge is the SFT builder's, not benchmarked for
preference labels (the retrospective's rule: Stage 4's judge gets its benchmark before any pair is
labelled), so this is a sampling-budget estimate, not labels. Samples are model output and are
only scored here; nothing is written into training data.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from sft_common import pmap, read_jsonl, report_failures, set_rpm
from sft_judge import judge, same_fact

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from scorers import answer_line, citations

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "data/sft/work/judged.jsonl"
MARGIN = 2.0
CHUNK_ID = re.compile(r"[\w.-]+:p\d+:c\d+\b")  # tests/test_sft_data.py's pattern
FLOOR = 1.0


def relabel(text: str, passages: list[dict]) -> str:
    """The model cites chunk ids (the prompt shows them); the judge and verifier read [P1]-[P4].
    Every chunk id goes: one of the record's passages becomes its label, any other id (a made-up
    one, the prompt's [doc:p12:c0] example) becomes "unknown", so no chunk id reaches a judge
    prompt or the SFT builder's call cache (test_generator_prompts_clean)."""
    label = {p["chunk_id"]: p["label"] for p in passages}
    return CHUNK_ID.sub(lambda m: label.get(m[0], "unknown"), text)


def verdicts(row: dict, rec: dict) -> dict:
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
        items = [{**rec, "answer": relabel(t, rec.get("passages") or [])} for t in texts]
        assert not any(CHUNK_ID.search(i["answer"]) for i in items)
        judged = pmap(judge, items)
        scored = [j for j in judged if not j["failed_call"]]
        eff = [j["score"] if all(j["hard"].values()) else FLOOR for j in scored]
        return {
            "scorer": "judge",
            "scores": [j["score"] for j in scored],
            "hard_pass": [all(j["hard"].values()) for j in scored],
            "effective": eff,
            "failed_calls": len(judged) - len(scored),
            "margin": round(max(eff) - min(eff), 2) if len(eff) >= 2 else None,
            "pairable": max(eff) - min(eff) >= MARGIN if len(eff) >= 2 else None,
        }
    return {"scorer": None, "pairable": None}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = sys.argv[1]
    set_rpm(30)
    rows = read_jsonl(REPO / f"results/runs/{run}/samples/dpo_probe.jsonl")
    work = {}
    ids = {r["id"] for r in rows}
    for line in WORK.open():
        rec = json.loads(line)
        if rec["eid"] in ids:
            work[rec["eid"]] = rec
    per = []
    for r in rows:
        v = verdicts(r, work[r["id"]]) if r["id"] in work else {"scorer": None, "pairable": None}
        per.append({"id": r["id"], "format": r["format"], **v})
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
        if f != "abstain" and v["pairable_fraction"] is not None and v["pairable_fraction"] < 0.5
    }
    res = {
        "run": run,
        "rule": "pairable = >= 1 chosen and >= 1 rejected of 4 (verifier: closed_book, abstain; "
        f"judge margin >= {MARGIN}: grounded, definition); doesn't gate; < 0.5 on a non-abstain "
        "format -> 8 samples per prompt for it",
        "by_format": by_format,
        "samples_per_prompt_8": sorted(budget),
        "scorers": dict(Counter(x["scorer"] for x in per)),
        "per_prompt": per,
    }
    res["failed_judge_calls"] = sum(x.get("failed_calls", 0) for x in per)
    out = REPO / "results/diversity" / f"{run}_pairable.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    report_failures()
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
