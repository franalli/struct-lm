"""Continued pre-training (Stage 2): one epoch of next-token loss on the domain corpus.

  python train/cpt.py --config train/configs/cpt.yaml [--override training.x=y run.smoke=true]
  On Modal: train/modal_train.py (one GPU in-process; two GPUs via accelerate + fsdp2.yaml).

Data: train/packing.py windows (BOS + document + EOS, concatenated, 4,096-token windows) from
data.train, handed to SFTTrainer as input_ids with TRL's packing and dataset preparation off, so
the loss is plain next-token on every token. Trainer eval: the SLICE_WINDOWS val windows spread
across data.val (the same ones eval/perplexity.py reports as ppl_val_slice).

Guards, before any GPU time is spent on a wrong run:
  - windows == sum(n_tokens + 2) // 4096 from the files themselves, the count split.py reports:
    a mismatch means the tokenizer changed or something truncated.
  - one epoch is 120-190 optimizer steps (the 150-step rule, notes/decisions.md); otherwise it
    exits with the accumulation factor the rule implies. run.smoke skips this and runs 20 steps.

Writes: training.output_dir (adapter or full model, checkpoints every save_steps; a rerun
resumes from the newest), <run.results_dir>/runs/<run_name>/train_log.jsonl (every log line as it
happens, with per-step time and tokens/s) and train_summary.json (also copied to output_dir, where
merge.py reads the base model and whether it was LoRA).
"""

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from common import freeze_non_text, load_model_and_tokenizer, lora_config, parse_config
from datasets import Dataset
from packing import SEQ_LEN, SLICE_WINDOWS, pack, spread
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

STEP_RANGE = (120, 190)  # the 150-step rule, with slack for corpus changes within a power of two
SMOKE = {"max_steps": 20, "gradient_accumulation_steps": 2, "eval_steps": 10, "save_steps": 10}


class JsonlLog(TrainerCallback):
    """Appends every log dict to train_log.jsonl when it is logged, so a crash keeps the curve.
    Step time is measured from on_step_begin to on_step_end, which brackets one optimizer step
    (all its micro-batches) and excludes evaluation and checkpointing."""

    def __init__(self, path: Path, tokens_per_step: int):
        self.path, self.tokens_per_step = path, tokens_per_step
        self.t0, self.step_times = 0.0, []
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_times.append(time.perf_counter() - self.t0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        rec = {"step": state.global_step, **(logs or {})}
        if "loss" in rec and self.step_times:
            rec["step_time"] = round(self.step_times[-1], 3)
            rec["tokens_per_s"] = round(self.tokens_per_step / self.step_times[-1], 1)
            rec["peak_mem_gb"] = peak_mem_gb()
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class StopAtStep(TrainerCallback):
    """Stops (and saves) at a step without touching max_steps, so the LR schedule still spans the
    whole epoch: the 2-GPU run's first 100 steps then see the same learning rates as the 1-GPU
    run's, and their loss curves are comparable step for step."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
            control.should_save = True


def train(cfg: dict, callbacks: list | None = None) -> dict:
    started = time.time()
    t, run = cfg["training"], cfg.get("run", {})
    smoke = run.get("smoke", False)
    if smoke:
        t.update(SMOKE)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    run_name = t["run_name"]
    results = Path(run.get("results_dir", "results")) / "runs" / run_name

    # seed before the model: TRL wraps the LoRA adapters (random A init) before Trainer seeds
    set_seed(t.get("seed", 42))
    model_name = cfg["model"]["init_from"]
    train_p = pack(cfg["data"]["train"], model_name)
    val_p = pack([cfg["data"]["val"]], model_name)
    for name, p in (("train", train_p), ("val", val_p)):
        if len(p.ids) != p.expected:
            raise SystemExit(
                f"{name}: {len(p.ids)} windows, but the files' n_tokens imply {p.expected}: the "
                "tokenizer differs from data prep's, or documents were truncated"
            )
    seqs = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * world
    steps = math.ceil(len(train_p.ids) / seqs)
    if not smoke and not STEP_RANGE[0] <= steps <= STEP_RANGE[1]:
        rule = 2 ** round(math.log2(len(train_p.ids) / 150))
        per = t["per_device_train_batch_size"] * world
        raise SystemExit(
            f"{len(train_p.ids)} windows at {seqs} seqs/step is {steps} steps/epoch, outside "
            f"{STEP_RANGE}; the 150-step rule wants {rule} seqs/step: "
            f"gradient_accumulation_steps={max(1, rule // per)}"
        )

    rng = np.random.default_rng(t.get("seed", 42))
    train_ids = train_p.ids
    if smoke:  # just the windows 20 steps consume, sampled across the corpus
        n = SMOKE["max_steps"] * seqs
        train_ids = train_ids[np.sort(rng.choice(len(train_ids), n, replace=False))]
    val_ids = val_p.ids[spread(len(val_p.ids), SLICE_WINDOWS)]  # = ppl_val_slice, smoke too
    train_ds = Dataset.from_dict({"input_ids": train_ids.tolist()})
    eval_ds = Dataset.from_dict({"input_ids": val_ids.tolist()})
    print(
        f"train {len(train_p.ids)} windows ({train_p.docs} docs, {train_p.sources}); "
        f"{seqs} seqs/step x {world} GPU(s) -> {steps} steps/epoch; eval {len(val_ids)} windows"
    )

    model, tok = load_model_and_tokenizer(cfg["model"])
    peft_cfg = lora_config(cfg)
    if peft_cfg is None:
        freeze_non_text(model)

    tokens_per_step = seqs * SEQ_LEN
    log = JsonlLog(results / "train_log.jsonl", tokens_per_step)
    cbs = [log, *(callbacks or [])]
    if run.get("stop_at_step"):
        cbs.append(StopAtStep(run["stop_at_step"]))
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            **t,
            max_length=SEQ_LEN,  # TRL's default is 1024, which would cut every window
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},  # already token ids; no EOS strings
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,  # a tokenizer, not a processor: TRL then treats this as text-only
        peft_config=peft_cfg,
        callbacks=cbs,
    )
    if trainer.is_world_process_zero():
        n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        n_lora = sum(1 for n, _ in trainer.model.named_modules() if n.endswith(".lora_A"))
        print(f"trainable params {n_train:,}; LoRA modules {n_lora}")
    out_dir = t["output_dir"]
    last = get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None
    if last:
        print(f"resuming from {last}")
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model()

    if not trainer.is_world_process_zero():
        return {}
    tok.save_pretrained(out_dir)
    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
    steady = log.step_times[3:] or log.step_times  # the first steps include warm-up and compile
    wall = time.time() - started
    summary = {
        "run": run_name,
        "base": model_name,
        "lora": peft_cfg is not None,
        "gpus": world,
        "windows": len(train_p.ids),
        "tokens": len(train_p.ids) * SEQ_LEN,
        "seqs_per_step": seqs,
        "steps_per_epoch": steps,
        "steps": trainer.state.global_step,
        "tokens_per_s": round(tokens_per_step / statistics.median(steady), 1) if steady else None,
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "wall_s": round(wall, 1),  # includes tokenising and model load: what the GPU is billed
        "gpu_hours": round(wall * world / 3600, 3),
        "final_train_loss": round(statistics.fmean(losses[-10:]), 4) if losses else None,
        "final_eval_loss": round(evals[-1], 4) if evals else None,
        "peak_mem_gb": peak_mem_gb(),
        "smoke": smoke,
        "data": {f: sha256(f) for f in [*cfg["data"]["train"], cfg["data"]["val"]]},
        "config": cfg,
    }
    for d in (results, Path(out_dir)):
        (d / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "data")}))
    return summary


def peak_mem_gb() -> float | None:
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
