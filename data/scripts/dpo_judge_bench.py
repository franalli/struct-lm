"""Stage 4's judge benchmark, before any pair is labelled (the 2026-10-05 rule; listwise form and
pass lines fixed 2026-10-08, notes/decisions.md).

  .venv/bin/python data/scripts/dpo_judge_bench.py prep   # -> data/dpo/bench_prompts.jsonl
  # GPU: eval/sample.py job dpo_bench (3 student samples per benchmark prompt)
  set -a; . ./.env; set +a
  .venv/bin/python data/scripts/dpo_judge_bench.py run    # -> results/dpo/judge_bench.json

Labels: the full-passage reads (data/sft/read_filter.jsonl) of the SFT builder's grounded and
definition records, for the content they read (fingerprint). defect = a reader found it wrong
(answer-level categories only: question_not_answerable and not_a_term judge the question, not the
answer); ok = no flaw. Every one of these records had passed the A5 judge, one at a time.
Per format, up to 60 defects by a fixed hash, and as many ok records.

Each benchmark call is the scoring call as it will run on the pool: the labelled teacher answer
listed with the student's 3 samples of the same prompt (those passing the rules), in a fixed
shuffle, graded in one call (sft_judge.judge_list). Three prompt variants: "a5" (the rubric as
is), "quote" (quote the supporting sentence, then the verdict), "page" (the full page instead of
the chunk). Measured on the labelled answer:
  recall   defects caught: a failed hard rule or a score <= 3
  ok_pass  ok records passed: every hard rule and a score >= 4
  auroc    effective score (floor on a failed hard rule), ok above defect
  gap      mean effective score of its student siblings minus its own (defects)
Rule, per format: the variant with the highest recall among those with ok_pass >= 0.8 (ties:
higher AUROC, then the variant order); the format passes at recall >= 0.5. A failed grounded is
paired on rule failures only, a failed definition not at all.
"""

import json
import sys
from collections import defaultdict

from dpo_common import REPO, judge_samples, records, relabel
from sft_assemble import prompt_text
from sft_common import fingerprint, h01, pmap, read_jsonl, report_failures, set_rpm, write_jsonl
from sft_guard import Guard
from sft_judge import VARIANTS

PROMPTS = REPO / "data/dpo/bench_prompts.jsonl"
SAMPLES = REPO / "results/runs/sft-from-cpt/samples/dpo_bench.jsonl"
OUT = REPO / "results/dpo/judge_bench.json"
PER_FORMAT = 60
QUESTION_LEVEL = {"question_not_answerable", "not_a_term"}
OK_PASS, RECALL = 0.8, 0.5
RULE = (
    f"per format, the variant with the highest recall among those with ok_pass >= {OK_PASS} (ties: "
    f"AUROC, then variant order); the format passes at recall >= {RECALL}"
)


def prep() -> None:
    reads = [r for r in read_jsonl(REPO / "data/sft/read_filter.jsonl")
             if r["format"] in ("grounded", "definition")]  # fmt: skip
    recs = records({r["eid"] for r in reads})
    guard = Guard()
    rows = []
    for f in ("grounded", "definition"):
        cur = [r for r in reads if r["format"] == f and r["eid"] in recs
               and fingerprint(recs[r["eid"]]) == r["fp"]]  # fmt: skip
        defects = [
            r for r in cur if r["verdict"] == "defect" and set(r["categories"]) - QUESTION_LEVEL
        ]
        oks = [r for r in cur if r["verdict"] == "ok"]
        defects = sorted(defects, key=lambda r: h01(f"dpo-bench:{r['eid']}"))[:PER_FORMAT]
        oks = sorted(oks, key=lambda r: h01(f"dpo-bench:{r['eid']}"))[: len(defects)]
        for r in defects + oks:
            e = recs[r["eid"]]
            rows.append({
                "id": r["eid"], "format": f, "label": r["verdict"], "categories": r["categories"],
                "prompt": [{"role": "user", "content": prompt_text(e, guard)}],
            })  # fmt: skip
    write_jsonl(PROMPTS, rows)


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(an ok record scores above a defect), ties half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def run() -> None:
    set_rpm(30)
    items = read_jsonl(PROMPTS)
    samples = {r["id"]: r for r in read_jsonl(SAMPLES)}
    recs = records({i["id"] for i in items})

    def one(job: tuple[dict, str]) -> dict:
        item, variant = job
        rec = recs[item["id"]]
        sibs = [
            s["text"].strip()
            for s in samples[item["id"]]["samples"]
            if s["finish_reason"] == "stop"
        ]
        sibs = [relabel(t, rec.get("passages") or []) for t in sibs]
        texts = [rec["answer"]] + [t for t in sibs if t != rec["answer"]]
        got = judge_samples(rec, texts, variant, f"bench:{item['id']}")
        me, rest = got[0], got[1:]
        return {
            "id": item["id"], "format": item["format"], "label": item["label"], "variant": variant,
            "eff": me["eff"], "score": me["score"], "hard_pass": me["hard_pass"],
            "rule_fails": me["rule_fails"], "position": me["position"], "n_listed": me["n_listed"],
            "siblings": [g["eff"] for g in rest], "sibling_positions": [g["position"] for g in rest],
            "reasoning": me.get("reasoning"), "quote": me.get("quote"),
        }  # fmt: skip

    jobs = [(i, v) for v in VARIANTS for i in items]
    per = pmap(one, jobs)
    res = {"rule": RULE, "metrics": defaultdict(dict), "chosen": {}}
    for f in ("grounded", "definition"):
        for v in VARIANTS:
            sel = [
                p for p in per if p["format"] == f and p["variant"] == v and p["eff"] is not None
            ]
            d = [p for p in sel if p["label"] == "defect"]
            o = [p for p in sel if p["label"] == "ok"]
            caught = [p for p in d if not p["hard_pass"] or p["score"] <= 3]
            passed = [p for p in o if p["hard_pass"] and p["score"] >= 4]
            gaps = [
                sum(s for s in p["siblings"] if s is not None) / len([s for s in p["siblings"] if s is not None]) - p["eff"]
                for p in d if any(s is not None for s in p["siblings"])
            ]  # fmt: skip
            by_pos = defaultdict(list)
            for p in sel:
                for e, pos in [(p["eff"], p["position"])] + list(
                    zip(p["siblings"], p["sibling_positions"])
                ):
                    if e is not None and pos is not None:
                        by_pos[pos].append(e)
            res["metrics"][f][v] = {
                "defects": len(d), "ok": len(o),
                "failed": sum(1 for p in per if p["format"] == f and p["variant"] == v and p["eff"] is None),
                "recall": round(len(caught) / len(d), 4) if d else None,
                "ok_pass": round(len(passed) / len(o), 4) if o else None,
                "auroc": auroc([p["eff"] for p in o], [p["eff"] for p in d]),
                "defect_gap": round(sum(gaps) / len(gaps), 3) if gaps else None,
                "mean_eff_by_position": {k: round(sum(x) / len(x), 3) for k, x in sorted(by_pos.items())},
            }  # fmt: skip
        ok_vs = [v for v in VARIANTS if (res["metrics"][f][v]["ok_pass"] or 0) >= OK_PASS]
        best = max(ok_vs, key=lambda v: (res["metrics"][f][v]["recall"] or 0,
                                         res["metrics"][f][v]["auroc"] or 0, -VARIANTS.index(v)),
                   default=None)  # fmt: skip
        recall = res["metrics"][f][best]["recall"] if best else None
        res["chosen"][f] = {
            "variant": best,
            "recall": recall,
            "passed": bool(best) and (recall or 0) >= RECALL,
        }
    res["per_item"] = per
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    report_failures()
    print(json.dumps({k: v for k, v in res.items() if k != "per_item"}, indent=2))


if __name__ == "__main__":
    {"prep": prep, "run": run}.get(
        sys.argv[1] if len(sys.argv) == 2 else "", lambda: sys.exit(__doc__)
    )()
