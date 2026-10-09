"""pass@1, pass@8 and maj@8 on the KPI's closed-book items, from sampled answers (Stage 5's read).

  .venv/bin/python eval/passk.py <run> [<run> ...]   # -> results/passk/<run>.json

Reads results/runs/<run>/samples/passk.jsonl (eval/sample.py's passk job: the 322 domain_qa
prompts of the KPI eval, 8 samples each at temperature 0.7, the KPI's token limits). Each sample's
answer line is scored twice: by the strict checker (scorers.qa_strict, the GRPO reward's rule; the
primary read) and by the eval's original scorer (scorers.qa_correct, as qa_acc). Per item, with c
of n samples correct: pass@1 = c / n (the unbiased estimator at k = 1), pass@n = c > 0, maj@n = the
most frequent normalised answer line is correct (ties to the earliest). Reported over all items
and by the seen / unseen half (eval/tasks/sft_seen_chunks.txt), with the SE over items.

DeepSeekMath §5.2's read: RL that raises pass@1 and maj@k while pass@k stays flat sharpens the
distribution the model already had; a pass@k gain would be new reach.
"""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from scorers import answer_line, normalize, qa_correct, qa_strict

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "eval/tasks/domain_qa.jsonl"
SEEN = REPO / "eval/tasks/sft_seen_chunks.txt"
OUT = REPO / "results/passk"


def item_scores(ref: dict, texts: list[str]) -> dict:
    lines = [answer_line(t) for t in texts]
    tol = ref.get("tolerance", 0.02)
    out = {}
    for name, ok in (
        ("strict", lambda a: qa_strict(a, ref["answer"], ref.get("answer_kind", "other"), tol)),
        ("original", lambda a: qa_correct(a, ref["answer"], ref["answer_type"], tol)),
    ):
        hits = [ok(a) for a in lines]
        votes = Counter(normalize(a) for a in lines if a)
        top = (
            max(votes, key=lambda k: (votes[k], -[normalize(a) for a in lines].index(k)))
            if votes
            else None
        )
        maj = next((h for a, h in zip(lines, hits) if normalize(a) == top), False) if top else False
        out[name] = {"pass@1": sum(hits) / len(hits), "pass@n": any(hits), "maj@n": maj}
    return out


def summarise(rows: list[dict]) -> dict:
    out: dict = {"items": len(rows)}
    for scorer in ("strict", "original"):
        for metric in ("pass@1", "pass@n", "maj@n"):
            xs = [float(r[scorer][metric]) for r in rows]
            m = statistics.fmean(xs)
            out[f"{scorer}_{metric}"] = round(m, 4)
            out[f"{scorer}_{metric}_se"] = (
                round(statistics.stdev(xs) / math.sqrt(len(xs)), 4) if len(xs) > 1 else None
            )
    return out


def score(run: str) -> dict:
    items = {r["id"]: r for r in map(json.loads, TASKS.open())}
    seen = set(SEEN.read_text().split())
    rows = [json.loads(x) for x in (REPO / "results/runs" / run / "samples/passk.jsonl").open()]
    if {r["id"] for r in rows} != set(items):
        raise SystemExit(f"{run}: passk samples don't cover the {len(items)} domain_qa items")
    (n,) = {len(r["samples"]) for r in rows}
    per = []
    for r in rows:
        ref = items[r["id"]]
        per.append({"id": r["id"], "half": "seen" if ref["source_chunk"] in seen else "unseen",
                    **item_scores(ref, [s["text"] for s in r["samples"]])})  # fmt: skip
    res = {
        "run": run,
        "samples_per_item": n,
        "all": summarise(per),
        "seen": summarise([p for p in per if p["half"] == "seen"]),
        "unseen": summarise([p for p in per if p["half"] == "unseen"]),
        "per_item": per,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    for run in sys.argv[1:]:
        r = score(run)
        print(
            run,
            {
                h: {k: v for k, v in r[h].items() if not k.endswith("_se")}
                for h in ("seen", "unseen")
            },
        )


if __name__ == "__main__":
    main()
