"""Stage 5 step 2: the calibration probe's window -> data/grpo/{train,val}.jsonl, SHA256SUMS

  .venv/bin/python data/scripts/grpo_probe.py [--run dpo-strict]

Reads results/runs/<run>/samples/grpo_probe.jsonl (eval/sample.py's grpo_probe job on the GRPO
start, stage4-final: 8 samples per task at temperature 1.0, up to 256 tokens, the rollout's
settings) and scores every sample with the training reward itself (train/grpo_rewards.score), so the
window and the reward agree. A task is in the window when 1 to n-1 of its n samples are correct
(Magistral's and the RLHF Book ch. 7's difficulty filter: a group whose samples all score the same
has zero advantage and teaches nothing). Fewer than MIN_TASKS in the window: stop and report (the
plan's fallback is a second probe at n 16, keeping 1 to 15).

grpo_val: VAL_TASKS in-window tasks held out whole by fact_id (every paraphrase of a held-out fact
leaves train with it), stratified by format and closed-book answer kind (largest remainder), facts
taken in a fixed hash order and only while they fit the stratum's quota. train.jsonl is every
other in-window task. Each row keeps its probe count (probe_correct of probe_n).
Writes data/grpo/{train,val}.jsonl, SHA256SUMS (with tasks.jsonl) and tasks_meta.json's probe
section (pass-rate histogram per format, window counts).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

from dpo_common import REPO
from dpo_prompts import shares
from sft_common import h01, read_jsonl, write_jsonl

sys.path.append(str(REPO / "train"))  # appended: data/scripts/common.py stays first for `common`
from grpo_rewards import score

OUT = REPO / "data/grpo"
VAL_TASKS = 50
MIN_TASKS = 300


def stratum(t: dict) -> str:
    return f"{t['format']}:{t['answer_kind']}" if t["answer_kind"] else t["format"]


def hold_out(window: list[dict]) -> set[str]:
    """fact_ids of the val tasks: per stratum, whole facts in hash order while they fit."""
    quota = shares(Counter(stratum(t) for t in window), VAL_TASKS)
    facts: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in window:
        facts[stratum(t)][t["fact_id"]].append(t)
    out = set()
    for s, groups in facts.items():
        n = 0
        for fid in sorted(groups, key=lambda f: h01(f"grpo-val:{f}")):
            if n + len(groups[fid]) <= quota[s]:
                out.add(fid)
                n += len(groups[fid])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dpo-strict", help="the GRPO start the probe sampled")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in read_jsonl(OUT / "tasks.jsonl")}
    rows = read_jsonl(REPO / "results/runs" / args.run / "samples/grpo_probe.jsonl")
    assert {r["id"] for r in rows} == set(tasks), "probe and task file disagree"
    hist: dict[str, Counter] = defaultdict(Counter)
    window, n_samples = [], set()
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        t = tasks[r["id"]]
        k = sum(score(s["text"], s["token_ids"], t["verifier"])["correct"] for s in r["samples"])
        n = len(r["samples"])
        n_samples.add(n)
        hist[stratum(t)][k] += 1
        acc[t["format"]][0] += k
        acc[t["format"]][1] += n
        if 0 < k < n:
            window.append({**t, "probe_correct": k, "probe_n": n})
    (n,) = n_samples
    val_facts = hold_out(window)
    val = [t for t in window if t["fact_id"] in val_facts]
    train = [t for t in window if t["fact_id"] not in val_facts]
    meta = json.loads((OUT / "tasks_meta.json").read_text())
    meta["probe"] = {
        "run": args.run,
        "samples_per_task": n,
        "window": f"1 to {n - 1} of {n} correct",
        "sample_accuracy": {f: round(a / b, 4) for f, (a, b) in acc.items()},
        "histogram": {s: dict(sorted(c.items())) for s, c in sorted(hist.items())},
        "in_window": dict(Counter(stratum(t) for t in window)),
        "in_window_total": len(window),
        "train": dict(Counter(stratum(t) for t in train)),
        "val": dict(Counter(stratum(t) for t in val)),
        "train_total": len(train),
        "val_total": len(val),
    }
    if len(window) < MIN_TASKS:
        (OUT / "tasks_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta["probe"], indent=2))
        sys.exit(
            f"{len(window)} tasks in the window < {MIN_TASKS}: stop and report (probe at n 16)"
        )
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "val.jsonl", val)
    sums = "".join(
        f"{hashlib.sha256((OUT / f).read_bytes()).hexdigest()}  {f}\n"
        for f in ("tasks.jsonl", "train.jsonl", "val.jsonl")
    )
    (OUT / "SHA256SUMS").write_text(sums)
    meta["probe"]["dataset_hash"] = hashlib.sha256(sums.encode()).hexdigest()
    (OUT / "tasks_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["probe"], indent=2))


if __name__ == "__main__":
    main()
