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

    # the corpus and set sizes the stage diagrams state (docs/diagrams/src), from the data files
    s = stats
    res["corpus"] = {
        "sources": s["download"]["sources"],
        "pdf_gb": s["download"]["bytes"] / 1e9,
        "tokens_extracted_m": s["filter"]["tokens_in"] / 1e6,
        "docs_after_filter": s["filter"]["docs_out"],
        "docs": s["dedup"]["docs_out"],
        "tokens_m": s["dedup"]["tokens_out"] / 1e6,
        "train_tokens_m": s["split"]["train"]["tokens"] / 1e6,
        "val_token_pct": 100 * s["split"]["val_token_frac"],
        "replay_tokens_m": s["replay"]["tokens"] / 1e6,
    }
    count = lambda f: len((REPO / f).read_text().splitlines())
    res["set_sizes"] = {
        "sft_val_records": count("data/sft/hosted/sft_val.jsonl"),
        "dpo_prompt_pool": count("data/dpo/prompts.jsonl"),
        "dpo_as_run_pairs": count("data/dpo/train.jsonl") + count("data/dpo/val.jsonl"),
        "dpo_strict_pairs": count("data/dpo/strict/train.jsonl")
        + count("data/dpo/strict/val.jsonl"),
        "dpo_strict_train": count("data/dpo/strict/train.jsonl"),
        "dpo_strict_val": count("data/dpo/strict/val.jsonl"),
        "grpo_train_tasks": count("data/grpo/train.jsonl"),
        "grpo_val_tasks": count("data/grpo/val.jsonl"),
    }

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

    # the eval's size, and the rest of the README's computed lines
    tasks = {
        f: count(f"eval/tasks/{f}.jsonl") for f in ("domain_qa", "grounded", "vocab", "adversarial")
    }
    res["eval"] = {"docs": count("eval/tasks/eval_docs.txt"), "items": sum(tasks.values()), **tasks}
    for halves_ in res["against_instruct"].values():
        for v in halves_.values():
            v["pct"] = [100 * c / v["n"] for c in v["correct"]]
    ev = json.loads((REPO / "results/qa_strict/evals.json").read_text())
    shift = {r: 100 * (v["qa_acc"] - v["qa_strict"]) for r, v in ev.items() if not v.get("partial")}
    res["strict_rescore_shift_pt"] = {
        "max_8b": max(v for r, v in shift.items() if r != "mistral-large-3"),
        "mistral_large_3": shift["mistral-large-3"],
    }
    dp = json.loads((REPO / "results/qa_strict/dpo_pairs.json").read_text())
    res["licence_audit"]["dpo_labels_failing_strict_pct"] = (
        100 * dp["chosen_wrong_total"] / dp["pairs"]
    )

    # calibration on the gold answer's tokens, seen half: DPO from its start, GRPO's two seeds from theirs
    res["calibration_seen_nats"] = {
        "dpo_strict_vs_sft_from_cpt": ans("dpo-strict", "seen") - ans("sft-from-cpt", "seen"),
        "grpo_pair_vs_dpo_strict": statistics.fmean(ans(r, "seen") for r in ("grpo", "grpo-seed1"))
        - ans("dpo-strict", "seen"),
    }

    # ablation A's MMLU reading as the rule applied it (three-decimal accuracies), and MMLU's SE
    def mmlu(run: str) -> tuple[float, float]:
        f = max((REPO / "results/lm_eval" / run).glob("**/results*.json"))
        m = json.loads(f.read_text())["results"]["mmlu"]
        return m["acc,none"], m["acc_stderr,none"]

    acc = {r: round(mmlu(r)[0], 3) for r in ("cpt-8b", "cpt-8b-seed1", "cpt-8b-replay10")}
    res["replay_rule_mmlu_pt"] = {
        "replay_minus_main": 100 * (acc["cpt-8b-replay10"] - acc["cpt-8b"]),
        "seed_floor": 100 * abs(acc["cpt-8b"] - acc["cpt-8b-seed1"]),
        "mmlu_se": 100 * mmlu("cpt-8b")[1],
        "replay_tokens_pct_of_train": 100 * s["replay"]["tokens"] / s["split"]["train"]["tokens"],
    }

    # GSM8K with one BOS (the Stage 6 gate) against the frozen no-BOS row, as correct counts
    def gsm8k(path: Path) -> tuple[int, int]:
        r = json.loads(path.read_text())
        n = r["n-samples"]["gsm8k"]["effective"] if "n-samples" in r else 1319
        return round(r["results"]["gsm8k"]["exact_match,strict-match"] * n), n

    gate = max((REPO / "results/serve/gate/lm_eval/dpo-strict").glob("results_*.json"))
    frozen = max((REPO / "results/lm_eval/dpo-strict").glob("**/results*.json"))
    res["gsm8k_bos"] = {"one_bos": gsm8k(gate), "no_bos": gsm8k(frozen)}

    # Stage 6 bench: FP8 against bf16 at 64 concurrent; the prefix cache; FP8's cacheable share
    def bench(variant: str, name: str, key: str) -> float:
        reps = sorted((REPO / "results/serve/bench" / variant).glob(f"{name}-rep*.json"))
        return statistics.fmean(json.loads(f.read_text())[key] for f in reps)

    res["serving"] = {
        "fp8_vs_bf16_c64_req_per_s_pct": 100
        * (
            bench("fp8", "c64", "request_throughput") / bench("bf16", "c64", "request_throughput")
            - 1
        ),
        "prefix_cache_ttft_cut_pct": 100
        * (
            1
            - bench("fp8", "grounded-rag-c8", "median_ttft_ms")
            / bench("fp8", "grounded-unique-c8", "median_ttft_ms")
        ),
        "fp8_prompt_length_share_of_ttft_pct": 100
        * (
            1
            - bench("fp8", "closedbook-c1", "median_ttft_ms")
            / bench("fp8", "grounded-unique-c1", "median_ttft_ms")
        ),
    }

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
