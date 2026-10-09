"""Which token ids lm-eval's vLLM backend sends for a GSM8K prompt, with and without add_bos_token
(Stage 6's GSM8K gate line runs with add_bos_token=True, as vLLM's quantization docs evaluate: a
quantized model can be sensitive to a missing BOS). Passes only if add_bos_token=True gives exactly
one BOS at position 0: Tekken through vLLM's MistralTokenizer must not add a second one.

  python eval/bos_probe.py --model /vol/checkpoints/dpo-strict --out results/serve/gate/bos_probe.json
  (on Modal: serve/modal_serve.py::gate, before the bf16 GSM8K run)

The table's lm-eval rows keep their frozen flags (run_lm_eval.sh); this only documents what the
gate's flag changes.
"""

import argparse
import json
from pathlib import Path

TEXT = "Question: Natalia sold clips to 48 of her friends in April. How many clips did she sell?\nAnswer:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from lm_eval.models.vllm_causallms import VLLM

    lm = VLLM(
        pretrained=args.model,
        tokenizer_mode="mistral",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        seed=0,
        limit_mm_per_prompt={"image": 0},
        config_format="hf",
    )
    bos = 1  # Tekken's <s>
    out = {"model": args.model, "text": TEXT, "bos_id": bos}
    for flag in (False, True):
        # generate_until (GSM8K) encodes with add_special_tokens=self.add_bos_token
        lm.add_bos_token = flag
        ids = lm.tok_encode(TEXT, add_special_tokens=flag)
        out[f"add_bos_token={flag}"] = {"head": ids[:6], "n_bos": ids.count(bos), "len": len(ids)}
    with_bos = out["add_bos_token=True"]
    out["one_bos"] = with_bos["head"][:1] == [bos] and with_bos["n_bos"] == 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    if not out["one_bos"]:
        raise SystemExit(
            "add_bos_token=True doesn't give exactly one leading BOS: gate GSM8K stops"
        )


if __name__ == "__main__":
    main()
