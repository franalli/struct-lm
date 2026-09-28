"""Check that vLLM runs a checkpoint the way transformers does: perplexity from vLLM's prompt
logprobs on the trainer's val slice, to compare with eval/perplexity.py's ppl_val_slice (same
windows, same tokens; transformers is the reference).

  python eval/vllm_ppl.py --model /vol/checkpoints/base-8b-hf --config-format hf
  python eval/vllm_ppl.py --model /vol/checkpoints/base-8b-hf --config-format hf --no-yarn-scale
  python eval/vllm_ppl.py --model mistralai/Ministral-3-8B-Base-2512 --config-format mistral
  (on Modal: modal run eval/modal_app.py::vllm_ppl --model ... --config-format ...)

Why it exists: vLLM 0.29 builds Ministral 3's YaRN rotary embedding from the HF config without its
mscale / mscale_all_dim keys, so it applies YaRN's default attention scaling
(0.1 ln(16) + 1 = 1.277 on cos and sin), which neither transformers nor Mistral's native params.json
(`apply_scale: false`) does. --no-yarn-scale runs a copy of the checkpoint whose config adds
`"apply_yarn_scaling": false` (the key vLLM reads) to text_config.rope_parameters.
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))  # repo or /root/train
from packing import SLICE_WINDOWS, pack, spread


def without_yarn_scale(model: str) -> str:
    """A copy of a local checkpoint dir (weights symlinked) whose config tells vLLM not to scale."""
    src, dst = Path(model), Path(tempfile.mkdtemp()) / Path(model).name
    dst.mkdir()
    for f in src.iterdir():
        if f.name != "config.json":
            (dst / f.name).symlink_to(f.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg["text_config"]["rope_parameters"]["apply_yarn_scaling"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config-format", default="hf", choices=("auto", "hf", "mistral"))
    ap.add_argument("--no-yarn-scale", action="store_true")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    val = pack([str(Path(args.data_dir) / "val.jsonl")], args.model)
    windows = val.ids[spread(len(val.ids), SLICE_WINDOWS)]
    model = without_yarn_scale(args.model) if args.no_yarn_scale else args.model
    llm = LLM(
        model=model,
        tokenizer_mode="mistral",
        config_format=args.config_format,
        load_format="mistral" if args.config_format == "mistral" else "auto",
        limit_mm_per_prompt={"image": 0},
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=w.tolist()) for w in windows],
        SamplingParams(max_tokens=1, prompt_logprobs=0),
    )
    nll, n = 0.0, 0
    for w, out in zip(windows, outs):
        for pos in range(1, len(w)):  # position 0 has no logprob (nothing predicts it)
            nll -= out.prompt_logprobs[pos][int(w[pos])].logprob
            n += 1
    result = {
        "model": args.model,
        "config_format": args.config_format,
        "no_yarn_scale": args.no_yarn_scale,
        "windows": len(windows),
        "tokens": n,
        "nats": round(nll / n, 5),
        "ppl_val_slice": round(math.exp(nll / n), 4),
    }
    print(json.dumps(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    if args.no_yarn_scale:
        shutil.rmtree(Path(model).parent, ignore_errors=True)


if __name__ == "__main__":
    main()
