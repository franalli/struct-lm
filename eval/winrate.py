"""Stage 4's win rate: a run's greedy answers against another's on the pool's 100 judge-split
prompts (eval/sample.py job dpo_judge), judged pairwise by Mistral Large 3 at temperature 0 in both
orders. A win needs both orders to agree; a split is a tie (0.5).

  set -a; . ./.env; set +a
  .venv/bin/python eval/winrate.py dpo sft-from-cpt     # -> results/winrate/dpo_vs_sft-from-cpt.json

Reported, not read (2026-10-08 amendment, notes/decisions.md): the same judge caught 28% of known
grounded defects in the Stage 4 benchmark, so a pairwise verdict from it isn't trusted as a
primary line. The reference is the SFT builder's gold (closed-book fact, definition reference,
grounded gist); abstain prompts' reference is the decline itself.
"""

import json
import math
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from judge import Judge, pairwise_rubric
from prompts import ABSTAIN_PHRASE

POOL = REPO / "data/dpo/prompts.jsonl"
WORK = REPO / "data/sft/work/judged.jsonl"


def answers(run: str) -> dict[str, str]:
    rows = [json.loads(x) for x in (REPO / f"results/runs/{run}/samples/dpo_judge.jsonl").open()]
    return {r["id"]: r["samples"][0]["text"].strip() for r in rows}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = sys.argv[1:]
    pool = {r["id"]: r for r in map(json.loads, POOL.open()) if r["dpo_split"] == "judge"}
    gold = {}
    for line in WORK.open():
        e = json.loads(line)
        if e["eid"] in pool:
            gold[e["eid"]] = e
    ans_a, ans_b = answers(a), answers(b)
    judge = Judge(REPO / "results/judge_cache.jsonl")
    per, by = [], defaultdict(list)
    for pid, row in pool.items():
        fmt, prompt = row["format"], row["prompt"][0]["content"]
        ref = (f'Decline with exactly "{ABSTAIN_PHRASE}": the passages do not contain the answer.'
               if fmt == "abstain" else gold[pid]["gold"] or "")  # fmt: skip
        x, y = ans_a[pid], ans_b[pid]
        if x == y:
            v = {"id": pid, "format": fmt, "outcome": 0.5, "identical": True}
        else:
            v1 = judge("pairwise", pairwise_rubric(prompt, ref, x, y))
            v2 = judge("pairwise", pairwise_rubric(prompt, ref, y, x))
            if v1 is None or v2 is None:
                v = {"id": pid, "format": fmt, "outcome": None}
            else:
                a_first, a_second = v1["score"] == 1, v2["score"] == 0
                outcome = 1.0 if a_first and a_second else 0.0 if not (a_first or a_second) else 0.5
                v = {"id": pid, "format": fmt, "outcome": outcome, "consistent": a_first == a_second,
                     "reasons": [v1["reason"], v2["reason"]]}  # fmt: skip
        per.append(v)
        by[fmt].append(v)

    def rate(vs: list[dict]) -> dict:
        done = [v["outcome"] for v in vs if v["outcome"] is not None]
        n = len(done)
        w = sum(done) / n if n else None
        return {
            "n": n,
            "win_rate": round(w, 4) if w is not None else None,
            "se": round(math.sqrt(w * (1 - w) / n), 4) if n and w is not None else None,
            "ties": sum(o == 0.5 for o in done),
            "identical": sum(1 for v in vs if v.get("identical")),
            "failed": len(vs) - n,
        }

    judged = [v for v in per if "consistent" in v]
    res = {
        "a": a,
        "b": b,
        "judge_calls": judge.calls,
        "overall": rate(per),
        "position_consistency": round(sum(v["consistent"] for v in judged) / len(judged), 4)
        if judged
        else None,
        "by_format": {f: rate(vs) for f, vs in sorted(by.items())},
        "per_prompt": per,
    }
    out = REPO / "results/winrate" / f"{a}_vs_{b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
