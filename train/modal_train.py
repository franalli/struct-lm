"""Stage 2 on Modal: CPT training, merge, perplexity and the eval harness, chained remotely.

  M=.venv/bin/modal   # always from the repo root (the images copy train/ and eval/ from there)
  $M run train/modal_train.py --config train/configs/cpt.yaml --run-name smoke-cpt --smoke --steps train,merge,ppl
  $M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b --steps train
  $M run --detach train/modal_train.py --run-name cpt-8b --steps merge,ppl,eval
  $M run --detach train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name cpt-8b-full --gpus 2
  $M run --detach train/modal_train.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b --steps ppl
  $M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-fsdp2 --gpus 2 \
      --steps train --overrides "training.gradient_accumulation_steps=4 training.gradient_checkpointing=false training.eval_strategy=no run.stop_at_step=100"

Steps, in order: train (cpt.py; 2 GPUs: accelerate + configs/fsdp2.yaml) -> merge (merge.py into
/vol/checkpoints/<run>) -> ppl, eval and latency in parallel (eval/perplexity.py; eval/modal_app.py's
lm_eval and kpi_eval --generate-only; latency only when asked for, since it measures the
architecture and serving setup, not the weights), each in its own container. Without --model, the
checkpoint evaluated is /vol/checkpoints/<run>. The KPI eval runs without --chat: every Stage 2
checkpoint is a base model (rule 2), and main() refuses an Instruct one.

main() makes exactly one remote call, pipeline.remote(), and the chain runs inside that CPU
container: with --detach, only the call in flight survives the client disconnecting, so a chain
of .remote() calls made here would stop after the first.

Every function runs with cwd=/vol, whose layout mirrors the repo's (data/processed uploaded with
`modal volume put`, checkpoints/, results/), so the configs' relative paths work in both places.
Volume writes are committed on every trainer save and eval and at the end, so a relaunch of the
same run resumes from the newest checkpoint.
"""

import subprocess
import sys
from pathlib import Path

import modal

# eval/modal_app.py: locally next to train/, in the containers under /root/eval
for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "modal_app.py").exists():
        sys.path.insert(0, str(_d))
        break
from modal_app import app as eval_app
from modal_app import kpi_eval, latency, lm_eval

app = modal.App("struct-lm-train").include(eval_app)
vol = modal.Volume.from_name("struct-lm", create_if_missing=True)

# Pinned to uv.lock, so a run on Modal uses what the Mac resolved. No flash-attn: every window
# is a full 4,096-token causal sequence, where SDPA already runs its FlashAttention-2 kernel (the
# package matters for padding-free packing, which we don't use), and it would need a CUDA build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "trl==0.29.1",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "datasets==5.0.1",
        "mistral-common==1.12.0",
        "torchao==0.18.0",  # AdamW8bit (optim: adamw_torch_8bit) for the full fine-tune under FSDP2
        "numpy",
        "pyyaml",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",  # the 8B base is already cached there by the eval runs
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",  # less fragmentation near 80 GB
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("train", remote_path="/root/train")
    .add_local_dir("eval", remote_path="/root/eval")  # perplexity.py, and modal_app for include
)
COMMON = {
    "image": image,
    "volumes": {"/vol": vol},
    "secrets": [modal.Secret.from_name("huggingface")],
}
# eval = lm-eval + KPI generation; latency is its own step: it depends on the architecture and the
# serving setup, not on these weights, so it is measured per deployed checkpoint, not per ablation
STEPS = ("train", "merge", "ppl", "eval", "latency")


def cpt_args(config: str, run_name: str, overrides: list[str], smoke: bool) -> list[str]:
    """cpt.py arguments: the repo config, with the run name and output dir set from run_name."""
    return [
        "--config",
        f"/root/{config}",
        "--override",
        f"training.run_name={run_name}",
        f"training.output_dir=checkpoints/_train/{run_name}",
        *overrides,
        *(["run.smoke=true"] if smoke else []),
    ]


@app.function(**COMMON, gpu="H100", timeout=6 * 3600)
def train(config: str, run_name: str, overrides: list[str], smoke: bool) -> dict:
    """cpt.py in-process on one H100, committing the volume on every save and eval."""
    import os

    os.chdir("/vol")
    sys.path.insert(0, "/root/train")
    import cpt
    from common import parse_config
    from transformers import TrainerCallback

    class Commit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            vol.commit()

        def on_evaluate(self, args, state, control, **kwargs):
            vol.commit()

    summary = cpt.train(parse_config(cpt_args(config, run_name, overrides, smoke)), [Commit()])
    vol.commit()
    return summary


@app.function(**COMMON, gpu="H100:2", memory=131072, timeout=6 * 3600)
def train_fsdp(config: str, run_name: str, overrides: list[str], smoke: bool) -> None:
    """cpt.py on two H100s with FSDP2: ablation B (8B full-parameter) and C (the LoRA config's
    scaling run). accelerate starts one process per GPU, and each loads the whole model on CPU
    before sharding it (fsdp2.yaml keeps cpu_ram_efficient_loading off): ~36 GB per rank in
    fp32 for B, hence 128 GiB. The ranks can't reach this container's volume handle, so this
    process commits every 5 minutes and at the end."""
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "/root/train/configs/fsdp2.yaml",
        "--num_processes",
        "2",
        "/root/train/cpt.py",
        *cpt_args(config, run_name, overrides, smoke),
    ]
    proc = subprocess.Popen(cmd, cwd="/vol")
    while True:
        try:
            proc.wait(timeout=300)
            break
        except subprocess.TimeoutExpired:
            vol.commit()
    vol.commit()
    if proc.returncode:
        raise RuntimeError(f"accelerate launch exited with {proc.returncode}")


@app.function(**COMMON, cpu=8, memory=98304, timeout=2 * 3600)
def merge(run_name: str, source: str = "") -> str:
    """merge.py on CPU: the adapter (or full weights) of checkpoints/_train/<run> into
    checkpoints/<run>. LoRA or full, and the base, come from the run's train_summary.json.

    With `source` (a hub id), re-save that model through the same path instead: a no-op "merge"
    that gives the base exactly the files a merged checkpoint has (HF only, no Mistral-native
    params.json / consolidated.safetensors, config written by the same transformers), so vLLM
    evaluates base and fine-tuned checkpoints identically (notes/decisions.md, base-8b-hf)."""
    import json

    vol.reload()
    out = f"/vol/checkpoints/{run_name}"
    if source:
        how = ["--full", source, "--base", source]
    else:
        src = Path(f"/vol/checkpoints/_train/{run_name}")
        summary = json.loads((src / "train_summary.json").read_text())
        how = (
            ["--adapter", str(src)]
            if summary["lora"]
            else ["--full", str(src), "--base", summary["base"]]
        )
    subprocess.run(
        [sys.executable, "/root/train/merge.py", *how, "--out", out], check=True, cwd="/vol"
    )
    vol.commit()
    return out


@app.function(**COMMON, gpu="H100", timeout=2 * 3600)
def perplexity(model: str, run_name: str) -> None:
    """eval/perplexity.py -> /vol/results/ppl/<run>.json."""
    vol.reload()
    subprocess.run(
        [sys.executable, "/root/eval/perplexity.py", "--model", model, "--run-name", run_name],
        check=True,
        cwd="/vol",
    )
    vol.commit()


def wait_for_weights(ckpt: Path, timeout_s: int = 900) -> None:
    """A checkpoint written by another app (a merge launched separately) shows up only once that
    app's volume commit lands; eval containers started before then find no weights. Reload until
    config.json and the weights are there, so the parallel evals all see the same checkpoint."""
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        vol.reload()
        if (ckpt / "config.json").exists() and any(ckpt.glob("model*.safetensors")):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"{ckpt}: no config.json + model*.safetensors after {timeout_s} s")
        time.sleep(20)


@app.function(**COMMON, timeout=12 * 3600)
def pipeline(
    config: str,
    run_name: str,
    model: str,
    steps: list[str],
    gpus: int,
    overrides: list[str],
    smoke: bool,
) -> None:
    """The chain, on a CPU container that mostly waits. Each step's function commits its writes
    and each later one reloads the volume, so a step sees what the previous one wrote."""
    if "train" in steps:
        fn = train_fsdp if gpus == 2 else train
        print(fn.remote(config, run_name, overrides, smoke))
    if "merge" in steps:
        model = merge.remote(run_name, "" if "train" in steps else model)
    model = model or f"/vol/checkpoints/{run_name}"
    if model.startswith("/vol/"):
        wait_for_weights(Path(model))
    calls = {}
    if "ppl" in steps:
        calls["ppl"] = perplexity.spawn(model, run_name)
    if "eval" in steps:
        # lm_eval and kpi_eval write separate dirs; with --generate-only nothing merges them on
        # Modal (scoring and the table row happen locally, rule 4), so they can run in parallel.
        calls["lm_eval"] = lm_eval.spawn(model, run_name, "mistral")
        calls["kpi_eval"] = kpi_eval.spawn(model, run_name, False, None, False, "mistral", True)
    if "latency" in steps:
        calls["latency"] = latency.spawn(model, run_name, "mistral")
    # Wait for every call before failing: raising on the first error (FunctionCall.gather) ends
    # this function and with it the app, which cancels the calls still running.
    failed = []
    for name, call in calls.items():
        try:
            call.get()
        except Exception as e:  # noqa: BLE001  collect, report all at the end
            failed.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
    if failed:
        raise RuntimeError(f"{run_name}: " + " | ".join(failed))
    print(f"{run_name}: {', '.join(steps)} done ({model})")


@app.local_entrypoint()
def main(
    run_name: str,
    config: str = "",
    model: str = "",
    steps: str = "train,merge,ppl,eval",
    gpus: int = 1,
    overrides: str = "",
    smoke: bool = False,
) -> None:
    """Checks the arguments locally, before any container starts, then starts pipeline."""
    todo = [s.strip() for s in steps.split(",") if s.strip()]
    if not todo or any(s not in STEPS for s in todo):
        raise SystemExit(f"--steps: comma-separated, from {', '.join(STEPS)}; got {steps!r}")
    if "train" in todo and not config:
        raise SystemExit("--steps train needs --config")
    if config and not (Path(config).exists() and config.startswith("train/configs/")):
        raise SystemExit(f"--config must be a file under train/configs/, got {config!r}")
    if gpus not in (1, 2):
        raise SystemExit("--gpus is 1 or 2")
    if model and "train" in todo:
        raise SystemExit(
            "--model is for an existing model: --steps ppl,eval, or merge to re-save it"
        )
    if "instruct" in (model + config).lower():
        raise SystemExit("Stage 2 checkpoints are base models; evaluate Instruct with modal_app.py")
    pipeline.remote(config, run_name, model, todo, gpus, overrides.split(), smoke)
