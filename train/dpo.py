"""Direct preference optimisation (Stage 4): LoRA on the frozen on-policy pair set, one epoch.

  python train/dpo.py --config train/configs/dpo.yaml [--override training.seed=1 run.smoke=true]
  On Modal: train/modal_train.py --config train/configs/dpo.yaml (the config's `stage: dpo`).

Data: data.train / data.val, refused unless every file matches <data.dir>/SHA256SUMS; the dataset
hash (sha256 of SHA256SUMS) is the first log line and goes into train_summary.json. Each record
carries the exact token ids the pair builder took from vLLM: prompt_ids (the mistral-common chat
render, [<s>, [INST], ..., [/INST]], tests/test_dpo_data.py checks it against the SFT render) and
chosen_ids / rejected_ids, the sampled completions ending in exactly one </s>. On-policy means
these ids, not a re-encoding of their text. TRL's dataset preparation is skipped (it would
re-tokenise strings through MistralCommonBackend, which refuses prompt + completion conversations)
and its preference collator concatenates prompt + completion as they are, adding no BOS or EOS.
No truncation: a record over 4,096 tokens (prompt + the longer completion) is refused.

Reference: ref_model=None with LoRA, so the reference is the init_from checkpoint with the adapter
disabled (rule 11), its log-probs precomputed once before step 1 (precompute_ref_log_probs).

Two changes to TRL 0.29.1's DPOTrainer:
  - fp32 log-probs: TRL takes the log-softmax of bf16 logits in bf16 and sums the per-token
    log-probs in bf16 (dpo_trainer.py: selective_log_softmax, then .sum(dim=1)), which rounds a
    sequence log-prob of -100 to a step of 0.5 nat, larger than the policy/reference gaps early in
    training. dpo_trainer's selective_log_softmax is replaced by one that upcasts the logits, so
    the per-token values, their sum, the precomputed reference and eval are all fp32.
  - length_norm (config `dpo.length_norm`, the dpo-lnorm ablation): each sequence's summed
    log-prob is divided by its completion length (Tulu 3's length-normalised DPO), for the policy
    and the reference alike. Done by scaling each row's per-token log-probs by 1 / its completion
    length inside the same replacement, so TRL's masking and sum give the mean.

Every evaluation (eval_steps, every save, the last step): TRL's eval_loss and eval_rewards/*,
eval_logps/* on dpo_val. At every save: 10 fixed greedy generations from dpo_val prompts (round
robin over formats), each with distinct-4, the most repeated 4-gram, repeated lines, an English
check and its length against the start's greedy answer to the same prompt (measured with the
adapter disabled, so a resumed run measures the same start), appended to
<results>/runs/<run>/eval_generations.jsonl. The last step is always evaluated and saved (with
optimizer state): the checkpoint rule (modal_train.py --merge-from rule) reads dpo_val loss there
and at step 50, and a second epoch could resume from it.
run.smoke: one optimizer step on 32 pairs (the 2 longest as the first micro-batch, so peak memory
is the worst case), 16 val pairs, evaluated at step 1.

Writes: training.output_dir (the adapter, checkpoint-N every save_steps and at the last step; a
rerun resumes from the newest), <run.results_dir>/runs/<run>/train_log.jsonl,
eval_generations.jsonl and train_summary.json (also in output_dir, where merge.py reads the base
and that it was LoRA).
"""

import contextlib
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from common import load_model_and_tokenizer, lora_config, parse_config
from cpt import peak_mem_gb
from datasets import Dataset
from sft import Log, expected_trainable
from sft_data import MAX_TOKENS, dataset_hash, load, sha256, special_ids, tokenizer
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer
from trl.trainer import dpo_trainer

SMOKE = {
    "max_steps": 1,
    "eval_steps": 1,
    "logging_steps": 1,
    "save_strategy": "no",
    "warmup_steps": 0,
}
SMOKE_PAIRS, SMOKE_VAL = 32, 16
COLUMNS = ("prompt_ids", "chosen_ids", "rejected_ids")
N_GENERATIONS, GEN_TOKENS = 10, 512  # 512: the pool's sampling cap
FUNCTION_WORDS = frozenset(
    "the a an of to in is are be by for on with as at or and that this it from not which".split()  # noqa: SIM905
)

# --- TRL patch: fp32 log-probs, optional per-row length normalisation ------------------------

# the original even if this module is imported twice (the replacement records it)
_trl_selective_log_softmax = getattr(
    dpo_trainer.selective_log_softmax, "trl_original", dpo_trainer.selective_log_softmax
)
_row_scale: torch.Tensor | None = None


def selective_log_softmax_fp32(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """TRL's selective_log_softmax on fp32 logits (its exact fp32 path: gather minus logsumexp),
    times 1 / completion length per row while length_normalised() is active."""
    out = _trl_selective_log_softmax(logits.float(), index)
    if _row_scale is not None:
        out = out * _row_scale.to(out.device)[:, None]
    return out


selective_log_softmax_fp32.trl_original = _trl_selective_log_softmax  # type: ignore[attr-defined]
# every call in dpo_trainer (policy, reference, precomputed reference, eval) resolves the name here
dpo_trainer.selective_log_softmax = selective_log_softmax_fp32


@contextlib.contextmanager
def length_normalised(completion_mask: torch.Tensor):
    """Rows of the batch (chosen then rejected) scaled by 1 / their completion length: TRL zeroes
    the prompt positions of the shifted mask and sums, so the sum becomes the mean."""
    global _row_scale
    _row_scale = 1.0 / completion_mask[:, 1:].sum(1).clamp_min(1).float()
    try:
        yield
    finally:
        _row_scale = None


class Trainer(DPOTrainer):
    """DPOTrainer on pre-tokenised pairs: no dataset preparation, the length-normalised variant,
    and in smoke mode the pairs in file order (the 2 longest first)."""

    def __init__(self, *a, sequential: bool = False, length_norm: bool = False, **kw):
        # set before super().__init__: the reference log-probs are precomputed in there
        self.sequential, self.length_norm = sequential, length_norm
        super().__init__(*a, **kw)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset  # prompt_ids / chosen_ids / rejected_ids, exactly as sampled

    def _get_train_sampler(self, *a, **kw):
        return (
            SequentialSampler(self.train_dataset)
            if self.sequential
            else super()._get_train_sampler(*a, **kw)
        )

    def _norm(self, inputs):
        if not self.length_norm:
            return contextlib.nullcontext()
        return length_normalised(inputs["completion_mask"])

    def compute_ref_log_probs(self, inputs):
        with self._norm(inputs):
            return super().compute_ref_log_probs(inputs)

    def _compute_loss(self, model, inputs, return_outputs):
        with self._norm(inputs):
            return super()._compute_loss(model, inputs, return_outputs)


# --- data ---------------------------------------------------------------------------------------


def load_pairs(path: str, sp: dict[str, int]) -> list[dict]:
    """The pair records, refused if any isn't shaped as the trainer needs (the full check is
    tests/test_dpo_data.py): prompt [<s>, [INST], ..., [/INST]], completions ending in exactly one
    </s>, prompt + the longer completion <= 4,096."""
    bos, inst, end_inst, eos = sp["<s>"], sp["[INST]"], sp["[/INST]"], sp["</s>"]
    recs = load(path)
    for r in recs:
        p, c, j = r["prompt_ids"], r["chosen_ids"], r["rejected_ids"]
        n = len(p) + max(len(c), len(j))
        if n > MAX_TOKENS:
            raise SystemExit(f"{r['id']}: {n} tokens > {MAX_TOKENS} (pairs are dropped, not cut)")
        if p[:2] != [bos, inst] or p[-1] != end_inst or p.count(bos) != 1:
            raise SystemExit(f"{r['id']}: prompt_ids are not <s> [INST] ... [/INST]")
        for side in (c, j):
            if len(side) < 2 or side[-1] != eos or side.count(eos) != 1:
                raise SystemExit(f"{r['id']}: a completion doesn't end in exactly one </s>")
    return recs


def fixed_generations(val: list[dict]) -> list[dict]:
    """Up to 10 distinct dpo_val prompts, round robin over formats, each format in a fixed hash
    order: the same prompts at every save and in every run on this set."""
    by_format: dict[str, list[dict]] = {}
    seen = set()

    def key(r: dict) -> bytes:
        return hashlib.sha256(f"dpo-gen:{r['prompt_id']}".encode()).digest()

    for r in sorted(val, key=key):
        if r["prompt_id"] not in seen:
            seen.add(r["prompt_id"])
            by_format.setdefault(r["format"], []).append(r)
    out, k = [], 0
    while len(out) < N_GENERATIONS and any(k < len(v) for v in by_format.values()):
        out += [v[k] for _, v in sorted(by_format.items()) if k < len(v)]
        k += 1
    return [
        {
            "id": r["prompt_id"],
            "format": r["format"],
            "prompt_ids": r["prompt_ids"],
            "chosen": r["chosen"][0]["content"],
        }
        for r in out[:N_GENERATIONS]
    ]


def text_checks(toks: list[int], text: str) -> dict:
    """Repetition and language checks on one generation (the degradation trigger reads them)."""
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chars = [ch for ch in text if not ch.isspace()]
    words = re.findall(r"[a-z']+", text.lower())
    ascii_share = sum(ch.isascii() for ch in chars) / len(chars) if chars else 1.0
    fw_rate = sum(w in FUNCTION_WORDS for w in words) / len(words) if words else 0.0
    max4 = max(Counter(grams).values()) if grams else 0
    dup_lines = len(lines) - len(set(lines))
    return {
        "distinct_4": round(len(set(grams)) / len(grams), 4) if grams else 1.0,
        "max_4gram_count": max4,
        "repeated_lines": dup_lines,
        "ascii_share": round(ascii_share, 4),
        "function_word_rate": round(fw_rate, 4),
        # short answers (a value, a term) have no function words to count
        "english": ascii_share >= 0.9 and (len(words) < 8 or fw_rate >= 0.1),
        "repetitive": max4 >= 4 or dup_lines > 0,
    }


class DPOExtras(TrainerCallback):
    """Forces an evaluation at every save and at the last step, a save at the last step, and the
    fixed generations at every save, against the start's greedy lengths (taken at train begin
    with the adapter disabled)."""

    def __init__(self, log: Log, gens: list[dict], gen_path: Path, model_name: str, every: int):
        self.log, self.gens, self.gen_path, self.every = log, gens, gen_path, every
        self.tek = tokenizer(model_name).instruct_tokenizer.tokenizer
        self.start_len: dict[str, int] = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.gens and not self.start_len:
            with model.disable_adapter():
                rows = self.generate(model, state, "start")
            self.start_len = {r["id"]: r["n_tokens"] for r in rows}

    def on_step_end(self, args, state, control, **kwargs):
        last = state.global_step >= state.max_steps
        if last or (self.every and state.global_step % self.every == 0):
            control.should_evaluate = True
        if last and args.save_strategy != "no":
            control.should_save = True

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.gens:
            self.generate(model, state, "policy")

    @torch.no_grad()
    def generate(self, model, state, which: str) -> list[dict]:
        was_training = model.training
        model.eval()
        prompts = [g["prompt_ids"] for g in self.gens]
        width, pad_id = max(map(len, prompts)), 11
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
        model.train(was_training)
        rows = []
        for g, toks in zip(self.gens, out):
            ended = 2 in toks
            toks = toks[: toks.index(2)] if ended else [t for t in toks if t != pad_id]
            text = self.tek.decode(toks)
            start = self.start_len.get(g["id"])
            rows.append(
                {
                    "step": state.global_step,
                    "model": which,  # "start": adapter disabled; "policy": the trained adapter
                    "id": g["id"],
                    "format": g["format"],
                    "ended_on_eos": ended,
                    "n_tokens": len(toks),
                    "length_ratio": round(len(toks) / start, 3) if start else None,
                    **text_checks(toks, text),
                    "output": text,
                    "chosen": g["chosen"],
                }
            )
        with self.gen_path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if which == "policy":
            n_start = sum(self.start_len.get(r["id"], 0) for r in rows)
            rec = {
                "step": state.global_step,
                "generations": len(rows),
                "ended_on_eos": sum(r["ended_on_eos"] for r in rows),
                "length_ratio": round(sum(r["n_tokens"] for r in rows) / n_start, 3)
                if n_start
                else None,
                "repetitive": sum(r["repetitive"] for r in rows),
                "not_english": sum(not r["english"] for r in rows),
                "min_distinct_4": min(r["distinct_4"] for r in rows),
            }
            self.log.write(rec)
            if rec["repetitive"] or rec["not_english"] or (rec["length_ratio"] or 0) > 1.3:
                print(f"WARNING step {state.global_step}: generations degraded? {rec}")
        return rows


class Step1(TrainerCallback):
    """At step 1 the adapter is still zero, so policy = reference and the rewards should be ~0:
    logged next to the step-1 metrics (a check on the reference path, not a gate)."""

    def __init__(self, log: Log):
        self.log = log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step != 1 or not logs or "loss" not in logs:
            return
        r = [logs.get("rewards/chosen", 0.0), logs.get("rewards/rejected", 0.0)]
        rec = {
            "step": 1,
            "check": "step1_rewards",
            "rewards/chosen": r[0],
            "rewards/rejected": r[1],
            "logged_loss": logs["loss"],
            "passed": max(map(abs, r)) <= 0.05,  # beta x a 0.5-nat log-prob gap
        }
        self.log.write(rec)
        print(json.dumps(rec))


RULE_STEP, RULE_MIN_STEPS = 50, 100


def checkpoint_rule(summary: dict) -> dict:
    """Stage 4's checkpoint rule (pre-registered), from the run's own dpo_val loss curve: the final
    step, unless the dpo_val loss at the end is above its value at step 50, then step 50's save.
    A run of fewer than 100 steps compares against the save nearest its midpoint instead (ties to
    the earlier one); a run with no save before its end keeps the final step. `checkpoint` is ""
    for the final adapter (output_dir itself) or checkpoint-N."""
    steps = summary["steps"]
    curve = {e["step"]: e["eval_loss"] for e in summary["eval_curve"]}
    saves = sorted(int(c.split("-")[1]) for c in summary["checkpoints"])
    earlier = [s for s in saves if s < steps]
    rule = (
        f"final step unless dpo_val loss at the end > at step {RULE_STEP} "
        f"(runs under {RULE_MIN_STEPS} steps: the save nearest the midpoint)"
    )
    if steps >= RULE_MIN_STEPS:
        ref = RULE_STEP
    elif earlier:
        ref = min(earlier, key=lambda s: (abs(s - steps / 2), s))
    else:
        return {
            "rule": rule,
            "steps": steps,
            "ref_step": None,
            "picked_step": steps,
            "checkpoint": "",
        }
    missing = [k for k in (ref, steps) if k not in curve] + ([] if ref in saves else ["save"])
    if missing:
        raise ValueError(f"checkpoint rule: no dpo_val loss or save at {missing} (steps {steps})")
    above = curve[steps] > curve[ref]
    return {
        "rule": rule,
        "steps": steps,
        "ref_step": ref,
        "eval_loss_ref": curve[ref],
        "eval_loss_end": curve[steps],
        "end_above_ref": above,
        "picked_step": ref if above else steps,
        "checkpoint": f"checkpoint-{ref}" if above else "",
    }


def eval_curve(history: list[dict]) -> list[dict]:
    """Every evaluation's dpo_val metrics by step, the last value if a step repeats."""
    by_step = {}
    for h in history:
        if "eval_loss" in h:
            by_step[h["step"]] = {k: v for k, v in h.items() if k.startswith("eval_")}
    return [{"step": s, **by_step[s]} for s in sorted(by_step)]


def train(cfg: dict, callbacks: list | None = None) -> dict:
    started = time.time()
    t, run, d = cfg["training"], cfg.get("run", {}), cfg["data"]
    length_norm = bool((cfg.get("dpo") or {}).get("length_norm", False))
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
    sp = special_ids(model_name)
    train_recs, val_recs = load_pairs(d["train"], sp), load_pairs(d["val"], sp)
    by_format = dict(sorted(Counter(r["format"] for r in train_recs).items()))
    log.write(
        {
            "step": 0,
            "dataset_hash": data_hash,
            "train_pairs": len(train_recs),
            "val_pairs": len(val_recs),
            "by_format": by_format,
            "init_from": model_name,
            "length_norm": length_norm,
            "fp32_logps": dpo_trainer.selective_log_softmax is selective_log_softmax_fp32,
        }
    )
    print(f"dataset {data_hash}: {len(train_recs)} pairs, val {len(val_recs)}; from {model_name}")

    set_seed(t.get("seed", 0))  # before the model: TRL wraps the LoRA adapters (random A init)
    if smoke:  # the 2 longest pairs as micro-batch 1, then the rest in file order
        size = [
            len(r["prompt_ids"]) + max(len(r["chosen_ids"]), len(r["rejected_ids"]))
            for r in train_recs
        ]
        longest = sorted(range(len(train_recs)), key=lambda k: -size[k])[:2]
        rest = [k for k in range(len(train_recs)) if k not in set(longest)][: SMOKE_PAIRS - 2]
        train_recs = [train_recs[k] for k in longest + rest]
        val_recs = val_recs[:SMOKE_VAL]
    train_ds = Dataset.from_list([{c: r[c] for c in COLUMNS} for r in train_recs])
    eval_ds = Dataset.from_list([{c: r[c] for c in COLUMNS} for r in val_recs])
    pairs_per_step = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * world
    micro = math.ceil(len(train_recs) / (t["per_device_train_batch_size"] * world))
    steps_per_epoch = math.ceil(micro / t["gradient_accumulation_steps"])

    model, tok = load_model_and_tokenizer(cfg["model"])
    peft_cfg = lora_config(cfg)
    gens = fixed_generations(val_recs)
    save_every = t.get("save_steps", 0) if t.get("save_strategy") == "steps" else 0
    extras = DPOExtras(log, gens, results / "eval_generations.jsonl", model_name, save_every)
    trainer = Trainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**t),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,  # a tokenizer, not a processor: TRL treats the model as text-only
        peft_config=peft_cfg,
        callbacks=[log, extras, Step1(log), *(callbacks or [])],
        sequential=smoke,
        length_norm=length_norm,
    )

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
    curve = eval_curve(hist)
    steady = log.tps[3:] or log.tps
    wall = time.time() - started
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    summary = {
        "stage": "dpo",
        "run": run_name,
        "base": model_name,
        "lora": peft_cfg is not None,
        "gpus": world,
        "dataset_hash": data_hash,
        "pairs": len(train_recs),
        "val_pairs": len(val_recs),
        "by_format": by_format,
        "tokens": sum(
            2 * len(r["prompt_ids"]) + len(r["chosen_ids"]) + len(r["rejected_ids"])
            for r in train_recs
        ),
        "pairs_per_step": pairs_per_step,
        "steps_per_epoch": steps_per_epoch,
        "steps": trainer.state.global_step,
        "epochs": t.get("num_train_epochs"),
        "beta": t.get("beta"),
        "loss_type": trainer.args.loss_type,
        "length_norm": length_norm,
        "tokens_per_s": round(statistics.median(steady), 1) if steady else None,
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "wall_s": round(wall, 1),  # includes model load and the reference pass: what is billed
        "gpu_hours": round(wall * world / 3600, 3),
        "final_train_loss": round(statistics.fmean(losses[-10:]), 4) if losses else None,
        "final_eval": curve[-1] if curve else None,
        "eval_curve": curve,  # the checkpoint rule's input (modal_train.py --merge-from rule)
        "checkpoints": [p.name for p in ckpts],
        "peak_mem_gb": peak_mem_gb(),
        "smoke": smoke,
        "data": {f: sha256(f) for f in (d["train"], d["val"])},
        "config": cfg,
    }
    for dd in (results, Path(out_dir)):
        (dd / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("config", "data", "eval_curve")})
    )
    return summary


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
