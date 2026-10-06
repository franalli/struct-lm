"""Supervised fine-tuning (Stage 3): LoRA on the frozen SFT set, in Mistral's chat format.

  python train/sft.py --config train/configs/sft.yaml [--override model.init_from=... run.smoke=true]
  On Modal: train/modal_train.py --config train/configs/sft.yaml (the config's `stage: sft`).

Data: data.train / data.val, refused unless every file matches <data.dir>/SHA256SUMS. The dataset
hash (sha256 of SHA256SUMS) is the first log line and goes into train_summary.json: training runs
cite it. Records are pre-tokenised by sft_data.encode (mistral-common, the KPI eval's --chat
rendering, no system prompt), so TRL's dataset preparation is off and the loss mask is each
record's completion_mask (completion_only_loss): prompt tokens and padding are -100, the answer and
its </s> are trained. tests/test_template.py checks those tensors (B1).

Checks written into train_log.jsonl, before and during the run:
  - trainable parameters: r * (in + out) over the seven projections of every language-model layer,
    computed from the model config, and nothing else trains;
  - at step 1, num_items_in_batch: it equals the completion tokens of the step's micro-batches, so
    the logged loss is their token mean; the mean of per-micro-batch means is logged beside it
    (what gradient accumulation without the fix would train on: the Tulu 3 bug).
Every evaluation (eval_steps, and each epoch end, which B4's checkpoint rule reads):
  - val_loss: the token mean of the NLL over every sft_val completion token, and per format;
  - 10 fixed greedy generations, two per format (256 new tokens), each with whether it ended on
    </s>, appended to <results>/runs/<run>/eval_generations.jsonl.
run.smoke: one optimizer step on 32 records (the 8 longest as the first micro-batch, so peak
memory is the worst case), evaluated at step 1.

Writes: training.output_dir (the adapter, plus one checkpoint per epoch; a rerun resumes from the
newest), <run.results_dir>/runs/<run>/train_log.jsonl, eval_generations.jsonl and
train_summary.json (also in output_dir, where merge.py reads the base and that it was LoRA).
"""

import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from common import load_model_and_tokenizer, lora_config, parse_config
from cpt import JsonlLog, peak_mem_gb
from datasets import Dataset
from sft_data import FORMATS, dataset_hash, encode, load, sha256, tokenizer
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# warmup 0: with 0.03 of one step, that step would run at LR 0 and leave LoRA B at zero
SMOKE = {
    "max_steps": 1,
    "eval_steps": 1,
    "logging_steps": 1,
    "save_strategy": "no",
    "warmup_steps": 0,
}
SMOKE_RECORDS = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GEN_TOKENS = 256


class Log(JsonlLog):
    """cpt.JsonlLog with measured throughput: SFT batches are padded to their longest record, so
    tokens per step vary. TRL logs num_tokens (cumulative non-padding tokens); tokens/s is its
    increase over the step's time."""

    def __init__(self, path: Path):
        super().__init__(path, tokens_per_step=0)
        self.prev_tokens, self.tps = 0, []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        rec = {"step": state.global_step, **(logs or {})}
        if "loss" in rec and "num_tokens" in rec and self.step_times:
            step_tokens = rec["num_tokens"] - self.prev_tokens
            self.prev_tokens = rec["num_tokens"]
            self.tps.append(step_tokens / self.step_times[-1])
            rec["step_tokens"] = int(step_tokens)
            rec["step_time"] = round(self.step_times[-1], 3)
            rec["tokens_per_s"] = round(self.tps[-1], 1)
            rec["peak_mem_gb"] = peak_mem_gb()
        self.write(rec)

    def write(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class Trainer(SFTTrainer):
    """SFTTrainer that records, for the first optimizer step, each micro-batch's completion-token
    count and loss, so the num_items_in_batch check can be logged; and, in smoke mode, feeds the
    records in file order (the 8 longest first)."""

    def __init__(self, *a, sequential: bool = False, **kw):
        super().__init__(*a, **kw)
        self.sequential, self.step1 = sequential, []

    def _get_train_sampler(self, *a, **kw):
        return (
            SequentialSampler(self.train_dataset)
            if self.sequential
            else super()._get_train_sampler(*a, **kw)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        n_tokens = int((inputs["labels"][..., 1:] != -100).sum()) if model.training else 0
        out = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if model.training and self.state.global_step == 0:
            loss = out[0] if return_outputs else out
            self.step1.append(
                {
                    "n": n_tokens,
                    "loss": float(loss.detach()),
                    "num_items": int(num_items_in_batch or 0),
                }
            )
        return out


class Checks(TrainerCallback):
    """Step 1's num_items_in_batch check, written next to the logged step-1 loss."""

    def __init__(self, log: Log, trainer: Trainer):
        self.log, self.trainer = log, trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        mb = self.trainer.step1
        if state.global_step != 1 or not logs or "loss" not in logs or not mb:
            return
        n = sum(m["n"] for m in mb)
        num_items = mb[0]["num_items"]
        # each micro-batch returned sum(NLL_i) / num_items, so their sum is the token mean;
        # NLL_i / n_i is that micro-batch's own mean
        token_mean = sum(m["loss"] for m in mb)
        mean_of_means = statistics.fmean(m["loss"] * num_items / m["n"] for m in mb)
        rec = {
            "step": 1,
            "check": "num_items_in_batch",
            "micro_batches": len(mb),
            "completion_tokens": [m["n"] for m in mb],
            "num_items_in_batch": num_items,
            "passed": num_items == n,
            "logged_loss": logs["loss"],
            "token_mean": round(token_mean, 6),
            "mean_of_means": round(mean_of_means, 6),
            "logged_is_token_mean": abs(logs["loss"] - token_mean) <= 1e-3 * max(1.0, token_mean),
        }
        self.log.write(rec)
        print(json.dumps(rec))
        if not (rec["passed"] and rec["logged_is_token_mean"]):
            raise SystemExit(f"num_items_in_batch check failed: {rec}")


class EvalExtras(TrainerCallback):
    """At every evaluation: the token-mean val loss (overall and per format) and the fixed
    generations. Also forces an evaluation at each epoch end, which eval_steps doesn't hit."""

    def __init__(self, log: Log, val: list[dict], gens: list[dict], gen_path: Path):
        self.log, self.val, self.gens, self.gen_path = log, val, gens, gen_path
        self.last_step = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.global_step != self.last_step:  # unless eval_steps just evaluated this step
            control.should_evaluate = True

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        self.last_step = state.global_step
        was_training = model.training
        model.eval()
        sums: dict[str, list[float]] = {}
        for i in range(0, len(self.val), 2):  # 2 x 2,875 x 131k fp32 logits is ~3 GB
            chunk = self.val[i : i + 2]
            ids, att, lab = pad(
                [r["input_ids"] for r in chunk], [r["completion_mask"] for r in chunk]
            )
            dev = model.device
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), lab[:, 1:].to(dev), ignore_index=-100, reduction="none"
            )
            for r, row, lab_row in zip(chunk, nll, lab[:, 1:]):
                s = sums.setdefault(r["format"], [0.0, 0])
                s[0] += float(row[lab_row.to(dev) != -100].sum())
                s[1] += int((lab_row != -100).sum())
        total = [sum(v[0] for v in sums.values()), sum(v[1] for v in sums.values())]
        rec = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0, 4),
            "val_loss": round(total[0] / total[1], 6),
            "val_loss_by_format": {f: round(v[0] / v[1], 6) for f, v in sorted(sums.items())},
            "val_tokens": total[1],
            "epoch_end": is_epoch_end(state),
        }
        self.log.write(rec)
        self.generate(model, state)
        model.train(was_training)

    def generate(self, model, state) -> None:
        prompts = [g["prompt_ids"] for g in self.gens]
        width = max(map(len, prompts))
        pad_id = 11
        ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device=model.device)
        att = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=model.device
        )
        out = model.generate(
            input_ids=ids,
            attention_mask=att,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            eos_token_id=2,
            pad_token_id=pad_id,
            use_cache=True,
        )[:, width:].tolist()

        tek = tokenizer(self.gens[0]["tokenizer"]).instruct_tokenizer.tokenizer
        with self.gen_path.open("a") as f:
            for g, toks in zip(self.gens, out):
                ended = 2 in toks
                toks = toks[: toks.index(2)] if ended else toks
                toks = [t for t in toks if t != pad_id]
                row = {
                    "step": state.global_step,
                    "id": g["id"],
                    "format": g["format"],
                    "split": g["split"],
                    "ended_on_eos": ended,
                    "n_tokens": len(toks),
                    "output": tek.decode(toks),
                    "reference": g["reference"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_eos = sum(2 in t for t in out)
        self.log.write(
            {"step": state.global_step, "generations_ended_on_eos": n_eos, "of": len(out)}
        )
        if n_eos == 0:
            print(f"WARNING step {state.global_step}: no generation ended on </s> (failure mode 4)")


def epoch_val_losses(path: Path) -> dict[str, float]:
    """{epoch: val_loss} at each epoch end, from the log (so a resumed run keeps earlier epochs)."""
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("epoch_end"):
            out[str(round(r["epoch"]))] = r["val_loss"]
    return out


def is_epoch_end(state) -> bool:
    return (
        state.epoch is not None
        and abs(state.epoch - round(state.epoch)) < 1e-6
        and state.epoch >= 1
    )


def pad(seqs: list[list[int]], masks: list[list[int]]):
    width = max(map(len, seqs))
    ids = torch.tensor([s + [11] * (width - len(s)) for s in seqs])
    att = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    lab = torch.tensor(
        [
            [t if m else -100 for t, m in zip(s, k)] + [-100] * (width - len(s))
            for s, k in zip(seqs, masks)
        ]
    )
    return ids, att, lab


def expected_trainable(model, r: int) -> int:
    """r * (in + out) for the seven projections of every language-model layer, from the config."""
    c = model.config.get_text_config()
    h, i, hd = (
        c.hidden_size,
        c.intermediate_size,
        c.head_dim or c.hidden_size // c.num_attention_heads,
    )
    q, kv = c.num_attention_heads * hd, c.num_key_value_heads * hd
    per_layer = (h + q) + 2 * (h + kv) + (q + h) + 2 * (h + i) + (i + h)
    return c.num_hidden_layers * r * per_layer


def fixed_generations(train: list[dict], val: list[dict], model: str) -> list[dict]:
    """Two records per format from sft_val (file order is a fixed hash order); sft_val holds one
    definition, so the second is the first train definition, marked split=train."""
    picks = []
    for f in FORMATS:
        chosen = [(r, "val") for r in val if r["format"] == f][:2]
        chosen += [(r, "train") for r in train if r["format"] == f][: 2 - len(chosen)]
        picks += chosen
    out = []
    for r, split in picks:
        enc = encode(r, model)
        n_prompt = enc["completion_mask"].index(1)
        out.append(
            {
                "id": r["eid"],
                "format": r["format"],
                "split": split,
                "prompt_ids": enc["input_ids"][:n_prompt],
                "reference": r["completion"][0]["content"],
                "tokenizer": model,
            }
        )
    return out


def train(cfg: dict, callbacks: list | None = None) -> dict:
    started = time.time()
    t, run, d = cfg["training"], cfg.get("run", {}), cfg["data"]
    smoke = run.get("smoke", False)
    if smoke:
        t.update(SMOKE)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    run_name = t["run_name"]
    results = Path(run.get("results_dir", "results")) / "runs" / run_name
    results.mkdir(parents=True, exist_ok=True)
    log = Log(results / "train_log.jsonl")

    data_hash = dataset_hash(d["dir"])
    model_name = cfg["model"]["init_from"]
    train_recs, val_recs = load(d["train"]), load(d["val"])
    log.write(
        {
            "step": 0,
            "dataset_hash": data_hash,
            "train": len(train_recs),
            "val": len(val_recs),
            "init_from": model_name,
        }
    )
    print(f"dataset {data_hash}: train {len(train_recs)}, val {len(val_recs)}; from {model_name}")

    # seed before the model: TRL wraps the LoRA adapters (random A init) before Trainer seeds
    set_seed(t.get("seed", 0))
    enc_train = [{**encode(r, model_name), "format": r["format"]} for r in train_recs]
    enc_val = [{**encode(r, model_name), "format": r["format"]} for r in val_recs]
    if smoke:  # the 8 longest records as micro-batch 1, then 24 in file order
        longest = sorted(range(len(enc_train)), key=lambda k: -len(enc_train[k]["input_ids"]))[:8]
        rest = [k for k in range(len(enc_train)) if k not in set(longest)][: SMOKE_RECORDS - 8]
        enc_train = [enc_train[k] for k in longest + rest]
    cols = ("input_ids", "completion_mask")
    train_ds = Dataset.from_list([{c: e[c] for c in cols} for e in enc_train])
    eval_ds = Dataset.from_list([{c: e[c] for c in cols} for e in enc_val])
    seqs = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * world
    micro = math.ceil(len(enc_train) / (t["per_device_train_batch_size"] * world))
    steps_per_epoch = micro // t["gradient_accumulation_steps"] + int(
        micro % t["gradient_accumulation_steps"] > 0
    )

    model, tok = load_model_and_tokenizer(cfg["model"])
    peft_cfg = lora_config(cfg)
    gens = fixed_generations(train_recs, val_recs, model_name)
    extras = EvalExtras(log, enc_val, gens, results / "eval_generations.jsonl")
    trainer = Trainer(
        model=model,
        args=SFTConfig(**t, dataset_kwargs={"skip_prepare_dataset": True}),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,  # a tokenizer, not a processor: TRL treats the model as text-only
        peft_config=peft_cfg,
        callbacks=[log, extras, *(callbacks or [])],
        sequential=smoke,
    )
    trainer.add_callback(Checks(log, trainer))

    trainable = {n: p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad}
    want = expected_trainable(model, cfg["lora"]["r"])
    stray = sorted(n for n in trainable if "lora_" not in n or "language_model" not in n)
    n_lora = sum(1 for n, _ in trainer.model.named_modules() if n.endswith(".lora_A"))
    check = {
        "step": 0,
        "check": "trainable_params",
        "trainable": sum(trainable.values()),
        "expected": want,
        "lora_modules": n_lora,
        "stray": stray[:5],
        "passed": sum(trainable.values()) == want and not stray,
    }
    log.write(check)
    print(json.dumps(check))
    if not check["passed"]:
        raise SystemExit(f"trainable parameter check failed: {check}")

    out_dir = t["output_dir"]
    last = get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None
    if last:
        print(f"resuming from {last}")
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model()
    tok.save_pretrained(out_dir)

    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
    steady = log.tps[3:] or log.tps
    wall = time.time() - started
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    summary = {
        "stage": "sft",
        "run": run_name,
        "base": model_name,
        "lora": peft_cfg is not None,
        "gpus": world,
        "dataset_hash": data_hash,
        "records": len(enc_train),
        "tokens": sum(len(e["input_ids"]) for e in enc_train),
        "completion_tokens": sum(sum(e["completion_mask"]) for e in enc_train),
        "seqs_per_step": seqs,
        "steps_per_epoch": steps_per_epoch,
        "steps": trainer.state.global_step,
        "epochs": t.get("num_train_epochs"),
        "optim": getattr(trainer.args.optim, "value", str(trainer.args.optim)),
        "tokens_per_s": round(statistics.median(steady), 1) if steady else None,
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "wall_s": round(wall, 1),  # includes tokenising and model load: what the GPU is billed
        "gpu_hours": round(wall * world / 3600, 3),
        "final_train_loss": round(statistics.fmean(losses[-10:]), 4) if losses else None,
        "final_eval_loss": round(evals[-1], 4) if evals else None,
        "val_loss_epoch_end": epoch_val_losses(log.path),  # B4's quantity, per epoch
        "checkpoints": [p.name for p in ckpts],
        "peak_mem_gb": peak_mem_gb(),
        "smoke": smoke,
        "data": {f: sha256(f) for f in (d["train"], d["val"])},
        "config": cfg,
    }
    for dd in (results, Path(out_dir)):
        (dd / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "data")}))
    return summary


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
