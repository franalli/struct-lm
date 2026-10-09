"""Stages 2-4 on Modal: CPT, SFT or DPO training, merge, perplexity and the eval harness, chained
remotely.

  M=.venv/bin/modal   # always from the repo root (the images copy train/ and eval/ from there)
  $M run train/modal_train.py --config train/configs/cpt.yaml --run-name smoke-cpt --smoke --steps train,merge,ppl
  $M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b --steps train
  $M run --detach train/modal_train.py --run-name cpt-8b --steps merge,ppl,eval
  $M run --detach train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name cpt-8b-full --gpus 2
  $M run --detach train/modal_train.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b --steps ppl
  $M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-fsdp2 --gpus 2 \
      --steps train --overrides "training.gradient_accumulation_steps=4 training.gradient_checkpointing=false training.eval_strategy=no run.stop_at_step=100"

Stage 3 (SFT, a config with `stage: sft`, train/sft.py; reasons in notes/decisions.md, Stage 3b):
  $M run train/modal_train.py --config train/configs/sft.yaml --run-name smoke-sft --smoke \
      --steps noop,train,merge,mergecheck,sample --sample-jobs template
  $M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-cpt --steps train
  $M run --detach train/modal_train.py --run-name sft-from-cpt --merge-from checkpoint-154 --chat \
      --steps merge,mergecheck,ppl,eval,latency,sample

Stage 4 (DPO, a config with `stage: dpo`, train/dpo.py; rules in notes/decisions.md, Stage 4):
  $M run train/modal_train.py --config train/configs/dpo.yaml --run-name smoke-dpo --smoke --chat \
      --steps noop,train,merge,mergecheck
  $M run --detach train/modal_train.py --config train/configs/dpo.yaml --run-name dpo --chat \
      --merge-from rule --steps noop,train,merge,mergecheck,ppl,eval,latency,sample
  (--merge-from rule: the pre-registered checkpoint rule on the run's own dpo_val curve, written to
  results/runs/<run>/b4.json before the merge; the merge gate runs on data/dpo/val.jsonl; the
  sample step adds the dpo_judge job, the win-rate prompts.)

Steps, in order: noop (merge_check.py noop: an untrained adapter on the run's start checkpoint,
merged, must equal it; CPU, alongside the rest) -> train (cpt.py, sft.py or dpo.py by the config's
stage; 2 GPUs: accelerate + configs/fsdp2.yaml, CPT only) -> merge (merge.py into
/vol/checkpoints/<run>; --merge-from picks a checkpoint-N instead of the final adapter) ->
mergecheck (B5's gate, on sft_val or a DPO run's dpo_val: the evals don't start if it fails) -> ppl, eval, latency and sample in parallel (eval/perplexity.py;
eval/modal_app.py's lm_eval and kpi_eval --generate-only; latency only when asked for, since it
measures the serving setup more than the weights; sample: eval/sample.py's --sample-jobs), each in
its own container. Without --model, the checkpoint evaluated is /vol/checkpoints/<run>. The KPI
eval runs with --chat for chat checkpoints (SFT on) and without for base models (rule 2): main()
requires --chat for a run named sft/dpo/grpo and refuses an Instruct one.

main() makes exactly one remote call, pipeline.remote(), and the chain runs inside that CPU
container: with --detach, only the call in flight survives the client disconnecting, so a chain
of .remote() calls made here would stop after the first.

Every function runs with cwd=/vol, whose layout mirrors the repo's (data/processed uploaded with
`modal volume put`, checkpoints/, results/), so the configs' relative paths work in both places.
Volume writes are committed on every trainer save and eval and at the end, so a relaunch of the
same run resumes from the newest checkpoint.
"""

import re
import subprocess
import sys
from pathlib import Path

import modal
import yaml

# eval/modal_app.py: locally next to train/, in the containers under /root/eval
for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "modal_app.py").exists():
        sys.path.insert(0, str(_d))
        break
from modal_app import app as eval_app
from modal_app import kpi_eval, latency, lm_eval, sample

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
# Stage 5 (GRPO): the training pins plus vLLM, which TRL colocates on the training GPU for the
# rollouts. vllm 0.30.0 is uv.lock's pair with torch 2.13.0, so the trainer's torch is SFT/DPO's;
# the SFT/DPO image above is untouched (its runs stay reproducible). The engine runs in-process
# (TRL syncs the LoRA-merged weights through llm_engine.model_executor.driver_worker), and
# FlashInfer's sampler JIT-compiles with nvcc, which debian_slim lacks (as eval/modal_app.py).
grpo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "vllm==0.30.0",
        "transformers==5.16.1",
        "trl==0.29.1",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "datasets==5.0.1",
        "mistral-common==1.12.0",
        "numpy",
        "pyyaml",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir("train", remote_path="/root/train")
    .add_local_dir("eval", remote_path="/root/eval")
)
COMMON = {
    "image": image,
    "volumes": {"/vol": vol},
    "secrets": [modal.Secret.from_name("huggingface")],
}
# eval = lm-eval + KPI generation; latency is its own step: it depends on the architecture and the
# serving setup, not on these weights, so it is measured per deployed checkpoint, not per ablation
STEPS = ("noop", "train", "merge", "mergecheck", "ppl", "eval", "latency", "sample")
SFT_VAL = (
    "data/sft/sft_val.jsonl"  # the merge gate's val set, DPO runs included (2026-10-08 freeze)
)
DPO_NAME = re.compile(r"(^|[/_-])dpo([/_-]|$)")  # run_eval.needs_chat's convention
GRPO_NAME = re.compile(r"(^|[/_-])grpo([/_-]|$)")


def is_stage(stage: str, config: str, run_name: str) -> bool:
    """A run of this stage: its config says `stage: <stage>` (read from /root/<config> in a
    container, the repo locally), or, for a merge or eval without a config, its run name has the
    stage's token."""
    if config:
        path = Path(f"/root/{config}") if Path(f"/root/{config}").exists() else Path(config)
        return yaml.safe_load(path.read_text()).get("stage") == stage
    return bool({"dpo": DPO_NAME, "grpo": GRPO_NAME}[stage].search(run_name))


def is_dpo(config: str, run_name: str) -> bool:
    return is_stage("dpo", config, run_name)


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
    """cpt.py, sft.py or dpo.py (the config's `stage`) in-process on one H100, committing the
    volume on every save and eval."""
    return run_stage(config, run_name, overrides, smoke)


@app.function(**{**COMMON, "image": grpo_image}, gpu="H100", timeout=6 * 3600)
def train_grpo(config: str, run_name: str, overrides: list[str], smoke: bool) -> dict:
    """grpo.py on one H100 with vLLM colocated (grpo_image), committing on every save and eval."""
    return run_stage(config, run_name, overrides, smoke)


def run_stage(config: str, run_name: str, overrides: list[str], smoke: bool) -> dict:
    import os

    os.chdir("/vol")
    sys.path.insert(0, "/root/train")
    from common import parse_config
    from transformers import TrainerCallback

    cfg = parse_config(cpt_args(config, run_name, overrides, smoke))
    if cfg.get("stage") == "sft":
        import sft as stage
    elif cfg.get("stage") == "dpo":
        import dpo as stage
    elif cfg.get("stage") == "grpo":
        import grpo as stage
    else:
        import cpt as stage

    class Commit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            vol.commit()

        def on_evaluate(self, args, state, control, **kwargs):
            vol.commit()

    summary = stage.train(cfg, [Commit()])
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
def merge(run_name: str, source: str = "", adapter: str = "") -> str:
    """merge.py on CPU: the adapter (or full weights) of checkpoints/_train/<run> into
    checkpoints/<run>. LoRA or full, and the base, come from the run's train_summary.json.

    With `source` (a hub id), re-save that model through the same path instead: a no-op "merge"
    that gives the base exactly the files a merged checkpoint has (HF only, no Mistral-native
    params.json / consolidated.safetensors, config written by the same transformers), so vLLM
    evaluates base and fine-tuned checkpoints identically (notes/decisions.md, base-8b-hf).

    With `adapter` (e.g. "checkpoint-77"), merge that trainer checkpoint's adapter instead of the
    final one: SFT keeps one per epoch, and B4's rule may pick epoch 1."""
    import json

    vol.reload()
    out = f"/vol/checkpoints/{run_name}"
    if source:
        how = ["--full", source, "--base", source]
    else:
        src = Path(f"/vol/checkpoints/_train/{run_name}")
        summary = json.loads((src / "train_summary.json").read_text())
        if adapter and not (src / adapter / "adapter_config.json").exists():
            raise RuntimeError(f"{src / adapter}: no adapter_config.json")
        how = (
            ["--adapter", str(src / adapter) if adapter else str(src)]
            if summary["lora"]
            else ["--full", str(src), "--base", summary["base"]]
        )
    subprocess.run(
        [sys.executable, "/root/train/merge.py", *how, "--out", out], check=True, cwd="/vol"
    )
    vol.commit()
    return out


@app.function(**COMMON, cpu=8, memory=98304, timeout=2 * 3600)
def noop_control(config: str, run_name: str, overrides: list[str], smoke: bool) -> dict:
    """merge_check.py noop on the run's start checkpoint (model.init_from after overrides): an
    untrained adapter, merged by merge.py, must give the start back tensor for tensor (the
    retrospective's no-op control, before a stage's training). /vol/results/noop/<start>.json."""
    import json

    sys.path.insert(0, "/root/train")
    from common import parse_config

    vol.reload()
    start = parse_config(cpt_args(config, run_name, overrides, smoke))["model"]["init_from"]
    name = Path(start).name
    cmd = [
        sys.executable,
        "/root/train/merge_check.py",
        "noop",
        "--start",
        start,
        "--config",
        f"/root/{config}",
        "--work",
        f"/vol/scratch/noop-{name}",
        "--out",
        f"/vol/results/noop/{name}.json",
    ]
    proc = subprocess.run(cmd, cwd="/vol", check=False)  # commit the json, then raise
    vol.commit()
    if proc.returncode:
        raise RuntimeError(f"no-op control failed for {start}: /vol/results/noop/{name}.json")
    return json.loads(Path(f"/vol/results/noop/{name}.json").read_text())


@app.function(**COMMON, gpu="H100", timeout=2 * 3600)
def merge_check(run_name: str, adapter: str = "", val: str = SFT_VAL) -> None:
    """merge_check.py check: start + adapter (unmerged) against /vol/checkpoints/<run> on `val`
    (sft_val; a DPO run's dpo_val pairs) and 3 probe records; writes
    /vol/results/runs/<run>/merge_check.json and raises on failure."""
    vol.reload()
    src = Path(f"/vol/checkpoints/_train/{run_name}") / adapter
    cmd = [
        sys.executable,
        "/root/train/merge_check.py",
        "check",
        "--adapter",
        str(src),
        "--merged",
        f"/vol/checkpoints/{run_name}",
        "--val",
        val,
        "--out",
        f"/vol/results/runs/{run_name}/merge_check.json",
    ]
    proc = subprocess.run(cmd, cwd="/vol", check=False)  # commit the json, then raise
    vol.commit()
    if proc.returncode:
        raise RuntimeError(f"{run_name}: merge check failed (merge_check.json); never serve it")


@app.function(**COMMON, cpu=4, memory=16384, timeout=3600)
def digest(run_name: str) -> dict:
    """merge_check.py digest: sha256 of each file of /vol/checkpoints/<run> (and of the list), so
    a gate result and the evals can be shown to be on the same file.
      modal run train/modal_train.py::digest --run-name sft-from-cpt"""
    import json

    vol.reload()
    out = f"/vol/results/runs/{run_name}/checkpoint_sha256.json"
    cmd = [sys.executable, "/root/train/merge_check.py", "digest"]
    subprocess.run(cmd + ["--merged", f"/vol/checkpoints/{run_name}", "--out", out], check=True)
    vol.commit()
    return json.loads(Path(out).read_text())


@app.function(**COMMON, gpu="H100", timeout=2 * 3600)
def merge_diagnose(run_name: str, adapter: str = "", val: str = SFT_VAL) -> None:
    """merge_check.py diagnose after a failed merge check: argmax flips and log-prob error of the
    merged and the unmerged bf16 model against an fp32 reference, over every completion position
    of `val` (sft_val; --val data/dpo/val.jsonl for a DPO run). .../runs/<run>/merge_diagnose.json.
      modal run train/modal_train.py::merge_diagnose --run-name sft-from-cpt --adapter checkpoint-77"""
    vol.reload()
    cmd = [
        sys.executable,
        "/root/train/merge_check.py",
        "diagnose",
        "--adapter",
        str(Path(f"/vol/checkpoints/_train/{run_name}") / adapter),
        "--merged",
        f"/vol/checkpoints/{run_name}",
        "--val",
        val,
        "--out",
        f"/vol/results/runs/{run_name}/merge_diagnose.json",
    ]
    subprocess.run(cmd, cwd="/vol", check=True)
    vol.commit()


@app.function(**COMMON, gpu="H100", timeout=2 * 3600)
def perplexity(model: str, run_name: str, only: str = "") -> None:
    """eval/perplexity.py -> /vol/results/ppl/<run>.json. only="postcutoff" adds the post-cutoff
    set to an existing json (needs /vol/data/exposure/postcutoff.jsonl):
      modal run train/modal_train.py::perplexity --model ... --run-name ... --only postcutoff"""
    vol.reload()
    cmd = [sys.executable, "/root/eval/perplexity.py", "--model", model, "--run-name", run_name]
    subprocess.run(cmd + (["--only", only] if only else []), check=True, cwd="/vol")
    vol.commit()


def b4_checkpoint(run_name: str) -> str:
    """B4 (amended 2026-10-06) applied to the run's own train_summary.json, for an unattended
    chain: epoch 2 unless the closed-book or the definition sft_val loss rose from the end of
    epoch 1 to the end of epoch 2, then epoch 1. Decided from the loss curve only, written to
    /vol/results/runs/<run>/b4.json before the merge, and returned as that epoch's checkpoint-N."""
    import json

    vol.reload()
    summary = json.loads(Path(f"/vol/checkpoints/_train/{run_name}/train_summary.json").read_text())
    v = summary["val_loss_by_format_epoch_end"]
    rose = {f: v["2"][f] > v["1"][f] for f in ("closed_book", "definition")}
    epoch = 1 if any(rose.values()) else 2
    ckpt = f"checkpoint-{summary['steps_per_epoch'] * epoch}"
    if not Path(f"/vol/checkpoints/_train/{run_name}/{ckpt}/adapter_config.json").exists():
        raise RuntimeError(f"{run_name}: B4 picked {ckpt}, which has no adapter")
    decision = {
        "run": run_name,
        "epoch": epoch,
        "checkpoint": ckpt,
        "rose": rose,
        "val_loss_by_format_epoch_end": v,
    }
    out = Path(f"/vol/results/runs/{run_name}/b4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n")
    vol.commit()
    print(f"B4: {decision}")
    return ckpt


def rule_checkpoint(run_name: str, stage: str) -> str:
    """The stage's pre-registered checkpoint rule applied to the run's own train_summary.json for
    an unattended chain. Stage 4 (dpo.checkpoint_rule): the final step unless dpo_val loss at the
    end is above its value at step 50 (runs under 100 steps: the save nearest the midpoint).
    Stage 5 (grpo.checkpoint_rule): the best grpo_val pass@1 among the saves at or before the
    stop, ties within one SE to the earliest. Written to /vol/results/runs/<run>/b4.json before
    the merge; returns "" (the final adapter) or checkpoint-N."""
    import json

    sys.path.insert(0, "/root/train")
    checkpoint_rule = __import__(stage).checkpoint_rule

    vol.reload()
    src = Path(f"/vol/checkpoints/_train/{run_name}")
    decision = {
        "run": run_name,
        **checkpoint_rule(json.loads((src / "train_summary.json").read_text())),
    }
    ckpt = decision["checkpoint"]
    if not (src / ckpt / "adapter_config.json").exists():
        raise RuntimeError(
            f"{run_name}: the rule picked {ckpt or 'the final adapter'}, which has none"
        )
    out = Path(f"/vol/results/runs/{run_name}/b4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n")
    vol.commit()
    print(f"checkpoint rule: {decision}")
    return ckpt


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
    chat: bool = False,
    merge_from: str = "",
    sample_jobs: str = "eos,diversity",
) -> None:
    """The chain, on a CPU container that mostly waits. Each step's function commits its writes
    and each later one reloads the volume, so a step sees what the previous one wrote."""
    calls = {}
    if "noop" in steps:  # independent of training: runs alongside it
        calls["noop"] = noop_control.spawn(config, run_name, overrides, smoke)
    grpo = is_stage("grpo", config, run_name)
    if "train" in steps:
        fn = train_fsdp if gpus == 2 else train_grpo if grpo else train
        print(fn.remote(config, run_name, overrides, smoke))
    if merge_from == "b4":  # the pre-registered rule picks the epoch, before anything is merged
        merge_from = b4_checkpoint(run_name)
    elif merge_from == "rule":  # Stage 4's (dpo_val curve) or Stage 5's (grpo_val pass@1)
        merge_from = rule_checkpoint(run_name, "grpo" if grpo else "dpo")
    if "merge" in steps:
        model = merge.remote(run_name, "" if "train" in steps else model, merge_from)
    if "mergecheck" in steps:
        # raises on failure: nothing below runs
        # sft_val for DPO runs too: dpo_val is 22 pairs, 665 positions (0 flips allowed at 0.1%), the
        # underpowered-gate failure of 2026-10-06; sft_val is the calibrated 11,351 (2026-10-08 freeze)
        merge_check.remote(run_name, merge_from, SFT_VAL)
    model = model or f"/vol/checkpoints/{run_name}"
    if model.startswith("/vol/") and set(steps) & {"ppl", "eval", "latency", "sample"}:
        wait_for_weights(Path(model))
    if "ppl" in steps:
        calls["ppl"] = perplexity.spawn(model, run_name)
    if "eval" in steps:
        # lm_eval and kpi_eval write separate dirs; with --generate-only nothing merges them on
        # Modal (scoring and the table row happen locally, rule 4), so they can run in parallel.
        calls["lm_eval"] = lm_eval.spawn(model, run_name, "mistral")
        calls["kpi_eval"] = kpi_eval.spawn(model, run_name, chat, None, False, "mistral", True)
    if "latency" in steps:
        calls["latency"] = latency.spawn(model, run_name, "mistral")
    if "sample" in steps:
        calls["sample"] = sample.spawn(model, run_name, sample_jobs, chat, "hf")
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
    chat: bool = False,  # KPI eval and sampling in the chat template: every SFT/DPO/GRPO checkpoint
    # checkpoint-77: merge that adapter; "b4": apply B4 after training (SFT); "rule": Stage 4's
    # or Stage 5's rule
    merge_from: str = "",
    # eval/sample.py jobs for the sample step; default eos,diversity (DPO runs: + dpo_judge;
    # GRPO runs: + passk, dpo_judge)
    sample_jobs: str = "",
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
        raise SystemExit("Instruct isn't trained here; evaluate it with eval/modal_app.py")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from run_eval import needs_chat

    dpo = is_dpo(config, run_name)
    grpo = is_stage("grpo", config, run_name)
    chat_ckpt = needs_chat(model, run_name) or "sft" in Path(config).stem or dpo or grpo
    if set(todo) & {"eval", "sample"} and chat_ckpt and not chat:
        raise SystemExit(f"{run_name} is a chat checkpoint: its KPI eval and samples need --chat")
    if chat and not chat_ckpt:
        raise SystemExit(f"--chat on {run_name or model}, which isn't a chat checkpoint (rule 2)")
    if gpus == 2 and ("sft" in Path(config).stem or dpo or grpo):
        raise SystemExit("SFT, DPO and GRPO run on one GPU (no FSDP path; vLLM is colocated)")
    if merge_from and not {"merge", "mergecheck"} & set(todo):
        raise SystemExit("--merge-from names the adapter for the merge and mergecheck steps")
    if merge_from == "rule" and not (dpo or grpo):
        raise SystemExit("--merge-from rule is Stage 4's or Stage 5's checkpoint rule")
    if merge_from == "b4" and (dpo or grpo):
        raise SystemExit("--merge-from b4 is Stage 3's epoch rule; DPO/GRPO use --merge-from rule")
    default_jobs = (
        "eos,diversity,passk,dpo_judge"
        if grpo
        else "eos,diversity,dpo_judge"
        if dpo
        else "eos,diversity"
    )
    sample_jobs = sample_jobs or default_jobs
    pipeline.remote(
        config,
        run_name,
        model,
        todo,
        gpus,
        overrides.split(),
        smoke,
        chat,
        merge_from,
        sample_jobs,
    )
