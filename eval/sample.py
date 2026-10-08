"""Sampled generations from a checkpoint through the KPI eval's vLLM engine (run_eval.make_llm), for
checks that need more than the greedy KPI outputs. One engine load serves every job asked for:

  template   the five SFT template prompts (eval/sft_template_prompts.jsonl, one per format),
             1 token each: vLLM's prompt_token_ids, which tests/test_template.py compares with the
             trainer's (B1's vLLM half)
  eos        20 prompts of the Stage 4 pool (data/dpo/prompts.jsonl, 4 per format by a fixed hash)
             x 4 samples at temperature 0.8, up to 1,024 tokens: does </s> end generation (B5)
  diversity  eval/diversity_prompts.jsonl (100 prompts) x 1 sample at temperature 0.7, up to 1,024
             tokens: distinct-4, token entropy and length, scored by eval/diversity.py (B6)
  dpo_probe  the first 20 prompts of the Stage 4 pool x 4 samples at temperature 0.7, up to 1,024
             tokens: whether samples collapse to one answer before DPO pairs are built
             (eval/diversity.py collapse; stop rule in notes/decisions.md, 2026-10-06)
  dpo_bench  the judge benchmark's prompts (data/dpo/bench_prompts.jsonl, from
             data/scripts/dpo_judge_bench.py prep) x 3 samples at temperature 0.7, up to 512 tokens:
             the student answers each labelled teacher answer is judged next to
  dpo_pool   the Stage 4 pool minus its judge split, the samples DPO pairs are built from:
             data/dpo/budget.json sets the temperature (top_p 0.95 only at 1.0) and the samples per
             prompt by format; up to 512 tokens; written per format as each finishes
  dpo_judge  the pool's 100 judge-split prompts, greedy, up to 1,024 tokens: the win-rate answers
             (eval/winrate.py), from the SFT start and from each DPO run

  python eval/sample.py --model /vol/checkpoints/sft-from-cpt --run-name sft-from-cpt --chat \
      --jobs eos,diversity
  On Modal: eval/modal_app.py::sample, or modal_train.py --steps ...,sample

Writes <results-dir>/runs/<run>/samples/<job>.jsonl: per prompt its id, format, prompt_token_ids
and every sample's text, token ids and finish_reason ("stop" is </s>: no stop strings are set;
"length" is the cap). Seeded (seed 0), so a rerun samples the same outputs.
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_eval import make_llm, write_jsonl

HERE = pathlib.Path(__file__).parent
JOBS = {
    "template": {"prompts": HERE / "sft_template_prompts.jsonl", "n": 1, "temperature": 0.0, "max_tokens": 1},
    "eos": {"prompts": pathlib.Path("data/dpo/prompts.jsonl"), "n": 4, "temperature": 0.8, "max_tokens": 1024, "per_format": 4},
    "diversity": {"prompts": HERE / "diversity_prompts.jsonl", "n": 1, "temperature": 0.7, "max_tokens": 1024},
    "dpo_probe": {"prompts": pathlib.Path("data/dpo/prompts.jsonl"), "n": 4, "temperature": 0.7, "max_tokens": 1024, "first": 20},
    "dpo_bench": {"prompts": pathlib.Path("data/dpo/bench_prompts.jsonl"), "n": 3, "temperature": 0.7, "max_tokens": 512},
    "dpo_pool": {"prompts": pathlib.Path("data/dpo/prompts.jsonl"), "budget": pathlib.Path("data/dpo/budget.json"), "max_tokens": 512, "exclude_split": "judge"},
    "dpo_judge": {"prompts": pathlib.Path("data/dpo/prompts.jsonl"), "n": 1, "temperature": 0.0, "max_tokens": 1024, "only_split": "judge"},
}  # fmt: skip
# the pool-sized jobs: vLLM's batch width and prefix cache (n samples share one prompt) set explicitly
POOL_ENGINE = {"max_num_seqs": 256, "enable_prefix_caching": True}


def h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def prompts_for(job: str) -> list[dict]:
    """[{id, format, prompt}] for a job. The Stage 4 pool stores the prompt as a message list."""
    spec = JOBS[job]
    rows = [json.loads(line) for line in spec["prompts"].read_text().splitlines()]
    out = []
    for r in rows:
        if "only_split" in spec and r.get("dpo_split") != spec["only_split"]:
            continue
        if "exclude_split" in spec and r.get("dpo_split") == spec["exclude_split"]:
            continue
        p = r["prompt"]
        if not isinstance(p, str) and len(p) != 1:  # system-prompt variants: not supported yet
            raise SystemExit(f"{r['id']}: {len(p)} prompt messages; sample.py sends one user turn")
        text = p if isinstance(p, str) else p[0]["content"]
        out.append({"id": r["id"], "format": r.get("format"), "prompt": text})
    if "first" in spec:  # the head of the file, in its fixed order
        return out[: spec["first"]]
    if "per_format" in spec:  # a fixed-hash pick per format
        picked = []
        for f in sorted({r["format"] for r in out}):
            rows_f = sorted(
                (r for r in out if r["format"] == f), key=lambda r: h(f"{job}:{r['id']}")
            )
            picked += rows_f[: spec["per_format"]]
        out = picked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--jobs", default="eos,diversity", help=f"comma-separated, from {list(JOBS)}")
    ap.add_argument("--chat", action="store_true", help="the model's chat template (llm.chat)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tokenizer-mode", default="mistral")
    ap.add_argument("--config-format", default="hf", choices=("hf", "auto"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    jobs = [j for j in args.jobs.split(",") if j]
    if unknown := set(jobs) - set(JOBS):
        ap.error(f"unknown jobs {sorted(unknown)}")
    from vllm import SamplingParams

    pool_sized = {"dpo_pool", "dpo_bench", "dpo_judge"} & set(jobs)
    llm = make_llm(
        args.model,
        1,
        args.max_model_len,
        args.tokenizer_mode,
        args.config_format,
        **(POOL_ENGINE if pool_sized else {}),
    )
    out_dir = pathlib.Path(args.results_dir) / "runs" / args.run_name / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        spec, items = JOBS[job], prompts_for(job)
        if "budget" in spec:  # per format: its own n, and a file per format as each finishes
            budget = json.loads(spec["budget"].read_text())
            groups = [
                (f, [it for it in items if it["format"] == f], budget["n"][f])
                for f in sorted({it["format"] for it in items})
            ]
            temperature, top_p = budget["temperature"], budget.get("top_p") or 1.0
        else:
            groups = [(None, items, spec["n"])]
            temperature, top_p = spec["temperature"], 1.0
        rows = []
        for fmt, group, n in groups:
            params = SamplingParams(
                n=n, temperature=temperature, top_p=top_p, max_tokens=spec["max_tokens"], seed=0
            )
            part = generate(llm, group, params, args.chat)
            if fmt is not None:
                write_jsonl(out_dir / f"{job}.{fmt}.jsonl", part)
            rows += part
        write_jsonl(out_dir / f"{job}.jsonl", rows)
        flat = [s for r in rows for s in r["samples"]]
        stopped = sum(s["finish_reason"] == "stop" for s in flat)
        print(
            f"{job}: {len(rows)} prompts, {len(flat)} samples, stopped on </s> {stopped}/{len(flat)}"
        )


def generate(llm, items: list[dict], params, chat: bool) -> list[dict]:
    if chat:
        outs = llm.chat([[{"role": "user", "content": it["prompt"]}] for it in items], params)
    else:
        outs = llm.generate([it["prompt"] for it in items], params)
    rows = []
    for it, o in zip(items, outs):
        samples = [
            {
                "text": c.text,
                "token_ids": list(c.token_ids),
                "n_tokens": len(c.token_ids),
                "finish_reason": c.finish_reason,
            }
            for c in o.outputs
        ]
        rows.append(
            {
                "id": it["id"],
                "format": it["format"],
                "prompt_token_ids": list(o.prompt_token_ids),
                "samples": samples,
            }
        )
    return rows


if __name__ == "__main__":
    main()
