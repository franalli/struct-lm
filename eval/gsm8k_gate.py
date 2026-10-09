"""Stage 6's GSM8K gate line: lm-eval GSM8K, 5-shot, all 1,319 items, with add_bos_token=True, in
lm-eval's vLLM backend with exactly run_lm_eval.sh's model arguments otherwise.

  python eval/gsm8k_gate.py --model /vol/checkpoints/dpo-strict-fp8 \
      --out /vol/results/serve/gate/lm_eval/dpo-strict-fp8
  (on Modal: serve/modal_serve.py::gate, step gsm8k)

Why not run_lm_eval.sh with add_bos_token in --model_args: lm-eval 0.4's vLLM backend forwards it to
vLLM's tokenizer loader, and under tokenizer_mode=mistral transformers' MistralCommonBackend refuses
the keyword. So the model is built without it and the attribute set afterwards, which is what
generate_until reads (eval/bos_probe.py showed the encoding: one leading BOS with it, none without).
The first batch of prompt ids sent to vLLM is checked here as well: every prompt must start with
exactly one BOS, or the run stops.

Writes <out>/results_<timestamp>.json with lm-eval's "results" (report.s6_metrics reads
results.gsm8k["exact_match,strict-match"]) and the arguments used.
"""

import argparse
import datetime
import json
from pathlib import Path

BOS = 1  # Tekken's <s>
# run_lm_eval.sh's MODEL_ARGS, verbatim
MODEL_ARGS = {
    "tokenizer_mode": "mistral",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "seed": 0,
    "limit_mm_per_prompt": {"image": 0},
    "config_format": "hf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="smoke tests only; the gate runs all 1,319")
    ap.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=("auto", "fp8"),
        help="fp8 for the FP8-KV variant: its GSM8K line must run with the KV cache it serves with",
    )
    args = ap.parse_args()
    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    model_args = {**MODEL_ARGS, "kv_cache_dtype": args.kv_cache_dtype}  # passed through to vLLM
    lm = VLLM(pretrained=args.model, **model_args)
    lm.add_bos_token = True
    checked = {"prompts": 0}
    generate = lm._model_generate

    def checked_generate(*a, **kw):
        requests = kw.get("requests", a[0] if a else None)
        if requests and not checked["prompts"]:
            bad = [r[:4] for r in requests if r[:1] != [BOS] or r.count(BOS) != 1]
            if bad:
                raise SystemExit(f"prompts without exactly one leading BOS, e.g. {bad[:2]}")
            checked["prompts"] = len(requests)
        return generate(*a, **kw)

    lm._model_generate = checked_generate
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=5,
        batch_size="auto",
        limit=args.limit,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    if not checked["prompts"]:
        raise SystemExit("no generation batch was checked for its BOS")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    record = {
        "results": res["results"],
        "n-samples": res.get("n-samples"),
        "model": args.model,
        "model_args": {**model_args, "add_bos_token": True},
        "num_fewshot": 5,
        "limit": args.limit,
        "bos_checked_prompts": checked["prompts"],
        "lm_eval": lm_eval.__version__,
    }
    (out / f"results_{stamp}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(res["results"]["gsm8k"], default=str))


if __name__ == "__main__":
    main()
