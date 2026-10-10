"""The README's computed numbers that no generated table holds, recomputed from committed files.

  .venv/bin/python eval/summary_stats.py     # -> results/summary_stats.json (make reproduce-score runs it)

- **SFT against stock Instruct, strict closed-book, by half:** paired per item and
  item-bootstrapped (10,000 draws, seed 0, the 2.5 / 97.5 percentiles), for `sft-from-cpt`, its
  seed twin and the untrained `base-8b-hf`, each against `instruct-8b`.
- **CPT's arm gap as a per-token gain:**
  - the log-perplexity gap on the train slice between the arms' means (two SFT runs each);
  - the mean answer tokens of the unseen gold answers;
  - their product, against the arms' measured unseen answer-token gap;
  - the arms' strict unseen accuracy.
- **The demo's candidate pools:** grounded items `dpo-strict` answered correctly with valid,
  judge-supported citations; adversarial items it declined.
- **The replay licence audit:** SFT records in all and by format, replay records by subset, the
  non-commercial (No Robots) and withdrawn (withdrawn math set) counts, and the held-out
  Tülu probe prompts.
- **Training cost per stage:** the GPU-hours and dollars of `results/train_runs.md`'s
  training-run tables, summed.
"""

import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from scorers import qa_strict_reason

OUT = REPO / "results/summary_stats.json"
TASKS = REPO / "eval/tasks/domain_qa.jsonl"
SEEN = set((REPO / "eval/tasks/sft_seen_chunks.txt").read_text().split())
ARMS = {
    "cpt": ("sft-from-cpt", "sft-from-cpt-seed1"),
    "base": ("sft-from-base", "sft-from-base-seed1"),
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


ITEMS = {r["id"]: r for r in jsonl(TASKS)}


def half(item_id: str) -> str:
    return "seen" if ITEMS[item_id]["source_chunk"] in SEEN else "unseen"


def strict(run: str) -> dict[str, bool]:
    out = {}
    for g in jsonl(REPO / "results/runs" / run / "generations.jsonl"):
        if g["task"] == "domain_qa" and g["id"] in ITEMS:
            ref = ITEMS[g["id"]]
            kind, tol = ref.get("answer_kind", "other"), ref.get("tolerance", 0.02)
            out[g["id"]] = qa_strict_reason(g["output"], ref["answer"], kind, tol) is None
    return out


def paired(a: dict[str, bool], b: dict[str, bool], ids: list[str]) -> dict:
    d = [int(b[i]) - int(a[i]) for i in ids]
    rng, n = random.Random(0), len(d)
    boot = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000))
    return {
        "n": n,
        "correct": [sum(b[i] for i in ids), sum(a[i] for i in ids)],
        "diff_pt": 100 * sum(d) / n,
        "ci_pt": [100 * boot[249], 100 * boot[9749]],
    }


def ppl_train(run: str) -> float:
    return json.loads((REPO / "results/ppl" / f"{run}.json").read_text())["ppl_train"]


def answer_tokens(run: str) -> list[int]:
    return [
        len(g["gold_lp_tokens"]) - 1  # the last element is the end token
        for g in jsonl(REPO / "results/runs" / run / "generations.jsonl")
        if g["task"] == "domain_qa"
        and g.get("gold_lp_tokens")
        and g["id"] in ITEMS
        and half(g["id"]) == "unseen"
    ]


def answer_lp(run: str) -> dict[str, float]:
    return {
        g["id"]: g["gold_lp"] - g["gold_lp_end"]
        for g in jsonl(REPO / "results/runs" / run / "generations.jsonl")
        if g["task"] == "domain_qa"
        and g.get("gold_lp") is not None
        and g["id"] in ITEMS
        and half(g["id"]) == "unseen"
    }


def stage_costs() -> dict:
    out = {}
    for section in re.split(r"(?m)^# ", (REPO / "results/train_runs.md").read_text())[1:]:
        name = section.splitlines()[0].strip()
        if "## Training runs" not in section:
            continue
        table = section.split("## Training runs", 1)[1].split("\n\n", 2)[1]
        lines = [ln for ln in table.splitlines() if ln.startswith("|")]
        head = [c.strip() for c in lines[0].strip("|").split("|")]
        ih, idl = head.index("GPU-h"), head.index("$")
        rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
        out[name] = {
            "runs": len(rows),
            "gpu_h": round(sum(float(r[ih]) for r in rows), 2),
            "usd": round(sum(float(r[idl]) for r in rows), 2),
        }
    out["all training"] = {
        "runs": sum(v["runs"] for v in out.values()),
        "gpu_h": round(sum(v["gpu_h"] for v in out.values()), 2),
        "usd": round(sum(v["usd"] for v in out.values()), 2),
    }
    return out


def main() -> None:
    res: dict = {}
    instruct = strict("instruct-8b")
    halves = {h: [i for i in ITEMS if half(i) == h] for h in ("seen", "unseen")}
    res["against_instruct"] = {
        run: {h: paired(instruct, strict(run), ids) for h, ids in halves.items()}
        for run in ("sft-from-cpt", "sft-from-cpt-seed1", "base-8b-hf")
    }

    log_ppl = {
        arm: statistics.fmean(math.log(ppl_train(r)) for r in runs) for arm, runs in ARMS.items()
    }
    gap_per_token = log_ppl["base"] - log_ppl["cpt"]
    tokens = statistics.fmean(answer_tokens("sft-from-cpt"))
    lp = {arm: [answer_lp(r) for r in runs] for arm, runs in ARMS.items()}
    ids = sorted(set.intersection(*(set(x) for v in lp.values() for x in v)))
    measured = statistics.fmean(
        statistics.fmean(x[i] for x in lp["cpt"]) - statistics.fmean(x[i] for x in lp["base"])
        for i in ids
    )
    acc = {
        arm: statistics.fmean(
            statistics.fmean(strict(r)[i] for i in halves["unseen"]) for r in runs
        )
        for arm, runs in ARMS.items()
    }
    res["cpt_arm_gap"] = {
        "train_slice_log_ppl_gap_per_token": gap_per_token,
        "unseen_answer_tokens_mean": tokens,
        "predicted_nats": gap_per_token * tokens,
        "measured_unseen_answer_nats": measured,
        "unseen_strict_pct": {arm: 100 * v for arm, v in acc.items()},
        "unseen_strict_diff_pt": 100 * (acc["cpt"] - acc["base"]),
    }

    scored = jsonl(REPO / "results/runs/dpo-strict/scored.jsonl")
    res["demo_pools"] = {
        "grounded_correct_cited_supported": sum(
            s["task"] == "grounded"
            and s["correct"] == 1
            and s["cite_valid"]
            and s["supported"] == 1
            for s in scored
        ),
        "adversarial_declined": sum(
            s["task"] == "adversarial" and s["abstained"] == 1 for s in scored
        ),
    }

    sft = [
        r
        for name in ("train.jsonl", "sft_val.jsonl")
        for r in jsonl(REPO / "data/sft/hosted" / name)
    ]
    subsets = Counter(r["replay_source"].split("#")[0] for r in sft if r["format"] == "replay")
    res["licence_audit"] = {
        "sft_records": len(sft),
        "replay_records": sum(subsets.values()),
        "replay_by_subset": dict(subsets.most_common()),
        "non_commercial": subsets["ai2-adapt-dev/no_robots_converted"],
        "withdrawn": subsets["withdrawn"],
    }
    train = jsonl(REPO / "data/sft/hosted/train.jsonl")
    res["licence_audit"]["sft_train_records"] = len(train)
    res["licence_audit"]["sft_train_by_format"] = dict(Counter(r["format"] for r in train))
    res["licence_audit"]["sft_train_teacher_written"] = sum(
        r["format"] in ("closed_book", "grounded", "definition") for r in train
    )
    probes = jsonl(REPO / "eval/hosted/diversity_prompts.jsonl")
    res["licence_audit"]["probe_prompts"] = sum(r["format"] == "general" for r in probes)
    res["licence_audit"]["blocking"] = (
        res["licence_audit"]["non_commercial"] + res["licence_audit"]["withdrawn"]
    )
    res["licence_audit"]["sft_train_teacher"] = dict(
        Counter(
            r["teacher"] for r in train if r["format"] in ("closed_book", "grounded", "definition")
        )
    )
    stats = json.loads((REPO / "data/processed/stats.json").read_text())
    flags = next(
        (v for k, v in stats.items() if isinstance(v, dict) and "copyright_flags" in v), {}
    ).get("copyright_flags")
    res["licence_audit"]["copyright_flagged_docs_kept"] = (
        len(flags) if isinstance(flags, list | dict) else flags
    )
    rows = (REPO / "data/sources.csv").read_text().splitlines()
    res["licence_audit"]["source_rows"] = len(rows) - 1  # the header
    res["training_cost"] = stage_costs()

    # the entropy stop that fired on one batch (grpo-seed1, step 34), against its old baseline
    log = jsonl(REPO / "results/runs/grpo-seed1/train_log.jsonl")
    ent = {r["step"]: r["entropy"] for r in log if r.get("entropy") is not None and r.get("step")}
    base15 = statistics.fmean(ent[s] for s in range(1, 6))
    res["grpo_seed1_entropy"] = {
        "step34": ent[34],
        "steps1_5_mean": base15,
        "steps25_34_mean": statistics.fmean(ent[s] for s in range(25, 35)),
        "steps25_34_over_steps1_5_pct": 100
        * statistics.fmean(ent[s] for s in range(25, 35))
        / base15,
    }

    # dpo-2ep against dpo: answer-token gold_lp by half, strict seen accuracy
    def ans(run: str, h: str) -> float:
        rows = jsonl(REPO / "results/runs" / run / "generations.jsonl")
        return statistics.fmean(
            g["gold_lp"] - g["gold_lp_end"]
            for g in rows
            if g["task"] == "domain_qa"
            and g.get("gold_lp") is not None
            and g["id"] in ITEMS
            and half(g["id"]) == h
        )

    seen_ids = halves["seen"]
    res["dpo_2ep_vs_dpo"] = {
        "answer_nats": {h: ans("dpo-2ep", h) - ans("dpo", h) for h in ("seen", "unseen")},
        "strict_seen_pt": 100
        * (
            statistics.fmean(strict("dpo-2ep")[i] for i in seen_ids)
            - statistics.fmean(strict("dpo")[i] for i in seen_ids)
        ),
    }

    # the demo's pools in the FP8 run it serves
    fp8 = jsonl(REPO / "results/runs/dpo-strict-fp8/scored.jsonl")
    res["demo_pools"]["fp8_grounded_correct_cited_supported"] = sum(
        s["task"] == "grounded" and s["correct"] == 1 and s["cite_valid"] and s["supported"] == 1
        for s in fp8
    )
    res["demo_pools"]["fp8_adversarial_declined"] = sum(
        s["task"] == "adversarial" and s["abstained"] == 1 for s in fp8
    )

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
