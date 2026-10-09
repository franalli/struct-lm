"""Group Relative Policy Optimization with verifiable rewards (Stage 5): LoRA on stage4-final.

  python train/grpo.py --config train/configs/grpo.yaml [--override training.seed=1 run.smoke=true]
  On Modal: train/modal_train.py --config train/configs/grpo.yaml (the config's `stage: grpo`)

Data: data.train / data.val (data/grpo/, from data/scripts/grpo_probe.py), refused unless every file
matches <data.dir>/SHA256SUMS. Each task is one user message (the Stage 4 pool's prompt) plus its
verifier; the policy writes every completion, the rule-based reward scores it (train/grpo_rewards.py:
format 0.1 gate, correctness 0.9 by eval/scorers.qa_strict / the citation and abstain rules, length
0 to -0.1). Prompts go to TRL as conversations: transformers' MistralCommonBackend renders them to
exactly mistral-common's [<s>, [INST], ..., [/INST]] ids (tests/test_grpo_train.py), and vLLM
samples from those ids. No reference model and no KL term (beta 0: Magistral, DAPO; rule 11's
reference would be the adapter-disabled start).

Three changes around TRL 0.29.1's GRPOTrainer:
  - fp32 log-probs: dpo.py's selective_log_softmax_fp32 (log-softmax of the bf16 logits in fp32)
    replaces grpo_trainer's, so the policy's per-token log-probs, the ratio and the importance
    weights against vLLM's (fp32) sampling log-probs are not bf16-rounded.
  - vLLM colocate: TRL builds LLM(model=name_or_path, ...) with none of the repo's engine settings
    (rule 3: tokenizer_mode="mistral", config_format="hf", limit_mm_per_prompt={"image": 0}, without
    which a merged checkpoint fails vLLM's dummy-image profiling) and seeds every run's sampler with
    0; the LLM name in trl.generation.vllm_generation is wrapped to add them and the run's seed.
  - no sleep mode: TRL's generate() reloads the weights from disk after waking a sleeping engine
    (the vLLM #29341 workaround), which would overwrite the LoRA-merged weights it synced just
    before, so the engine would keep sampling from the start. vLLM keeps its share of the GPU
    instead (vllm_gpu_memory_utilization) and the micro-batch is sized for what is left.

Every step: TRL's metrics (reward, rewards/<fn>/mean, reward_std, frac_reward_zero_std,
completions/*, entropy, clip_ratio/*, sampling/*). Every rollout's text and reward parts:
<results>/runs/<run>/rollouts.jsonl (the hack audit and per-task curves). Every evaluation (step 0,
every eval_steps, the last step): grpo_val pass@1 and pass@8 from num_generations_eval samples per
task, with the per-task SE (the checkpoint rule's input), and at every save 10 fixed greedy
generations against the start's (dpo.DPOExtras: repetition, length, language). Stop rules
(pre-registered, notes/decisions.md): frac_reward_zero_std > 0.8 for 10 consecutive steps; entropy
under a third of its steps 1-5 mean; no new best val pass@1 in 2 evaluations while the train reward
rose. A stop saves that step. run.smoke: 3 steps of 2 tasks x 8 at lr 1e-4 (a broken weight sync
shows as a jump in sampling/sampling_logp_difference), evaluated at steps 0 and 2 on 5 tasks.

Writes: training.output_dir (adapter; checkpoint-N at every save), <run.results_dir>/runs/<run>/
train_log.jsonl, rollouts.jsonl, eval_generations.jsonl and train_summary.json (also in output_dir).
"""

import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from common import load_model_and_tokenizer, lora_config, parse_config
from cpt import peak_mem_gb
from datasets import Dataset
from dpo import DPOExtras, selective_log_softmax_fp32
from grpo_rewards import CORRECT_W, format_reward, length_penalty, score, total
from sft import Log, expected_trainable
from sft_data import dataset_hash, load, sha256, tokenizer
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from trl.generation import vllm_generation
from trl.trainer import grpo_trainer

SMOKE = {
    "max_steps": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1.0e-4,
    "warmup_steps": 0,
    "eval_steps": 2,
    "save_steps": 2,
    "per_device_eval_batch_size": 40,
    "logging_steps": 1,
}
SMOKE_TRAIN, SMOKE_VAL = 6, 5
VLLM_ARGS = {
    "tokenizer_mode": "mistral",
    "config_format": "hf",
    "limit_mm_per_prompt": {"image": 0},
    "dtype": "bfloat16",
}
N_GENERATIONS = 10
ZERO_STD_MAX, ZERO_STD_RUN = 0.8, 10
ENTROPY_FLOOR = 1 / 3
FLAT_EVALS, REWARD_WINDOW = 2, 25
DRIFT_PARAMS = ("layers.0.self_attn.q_proj", "layers.33.mlp.down_proj")

grpo_trainer.selective_log_softmax = selective_log_softmax_fp32


def patch_vllm(seed: int) -> None:
    """TRL's colocated LLM(...) with the repo's engine settings (rule 3) and the run's seed."""
    orig = getattr(vllm_generation, "LLM", None)  # absent without vLLM (the Mac's tests)
    if orig is None:
        return
    orig = getattr(orig, "trl_original", orig)

    def llm(**kw):
        return orig(**{**kw, **VLLM_ARGS, "seed": seed})

    llm.trl_original = orig  # type: ignore[attr-defined]
    vllm_generation.LLM = llm


def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


class Rollouts:
    """correctness_reward as a callable that also writes every rollout (one row per completion)
    and, in evaluation, hands each task's verdicts to the val tracker. In a row, `format` is the
    format gate's verdict (score() is spread over the task's format, as the 2026-10-09 runs wrote
    it); the task's format is in data/grpo/tasks.jsonl by task_id."""

    __name__ = "correctness_reward"

    def __init__(self, path: Path, watch: "Watch"):
        self.path, self.watch, self.trainer = path, watch, None

    def __call__(self, completions, completion_ids, verifier, task_id, format, trainer_state, **_):
        mode = "eval" if self.trainer is not None and not self.trainer.model.training else "train"
        scores = [
            score(_text(c), list(ids), json.loads(v))
            for c, ids, v in zip(completions, completion_ids, verifier, strict=True)
        ]
        if mode == "eval":
            for t, s in zip(task_id, scores):
                self.watch.val[t].append(s["correct"])
        with self.path.open("a") as f:
            for c, t, fmt, s in zip(completions, task_id, format, scores):
                row = {"step": trainer_state.global_step, "mode": mode, "task_id": t, "format": fmt,
                       **s, "total": round(total(s), 4), "text": _text(c)}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [CORRECT_W * s["correct"] for s in scores]


class Watch(TrainerCallback):
    """grpo_val pass@k per evaluation, the step-1 importance-sampling check, base-weight drift and
    the pre-registered stop rules."""

    def __init__(self, log: Log):
        self.log = log
        self.val: dict[str, list[bool]] = defaultdict(list)
        self.curve: list[dict] = []
        self.rewards: list[float] = []
        self.entropy: list[float] = []
        self.zero_std_run = 0
        self.stop: dict | None = None
        self.base: dict[str, torch.Tensor] = {}

    def _base_weights(self, model) -> dict[str, torch.Tensor]:
        out = {}
        for name, p in model.named_parameters():
            if name.endswith("base_layer.weight") and any(d in name for d in DRIFT_PARAMS):
                out[name] = p.detach()
        return out

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.base = {n: p.float().cpu().clone() for n, p in self._base_weights(model).items()}

    def on_train_end(self, args, state, control, model=None, **kwargs):
        now = self._base_weights(model)
        drift = max(((now[n].float().cpu() - w).abs().max().item() for n, w in self.base.items()),
                    default=None)  # fmt: skip
        self.log.write({"step": state.global_step, "check": "base_drift_max", "value": drift})
        self.base_drift = drift

    def _halt(self, state, control, reason: str) -> None:
        if self.stop is None:
            self.stop = {"step": state.global_step, "reason": reason}
            self.log.write({"step": state.global_step, "check": "stop", "reason": reason})
            print(f"STOP at step {state.global_step}: {reason}")
            control.should_training_stop = True
            control.should_save = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "reward" not in logs or any(k.startswith("eval_") for k in logs):
            return
        self.rewards.append(logs["reward"])
        if "entropy" in logs:
            self.entropy.append(logs["entropy"])
        if state.global_step == 1:
            r = logs.get("sampling/importance_sampling_ratio/mean")
            rec = {"step": 1, "check": "step1_is_ratio", "value": r,
                   "passed": r is not None and abs(r - 1) <= 0.02}  # fmt: skip
            self.log.write(rec)
            print(json.dumps(rec))
        z = logs.get("frac_reward_zero_std", 0.0)
        self.zero_std_run = self.zero_std_run + 1 if z > ZERO_STD_MAX else 0
        if self.zero_std_run >= ZERO_STD_RUN:
            self._halt(
                state, control, f"frac_reward_zero_std > {ZERO_STD_MAX} for {ZERO_STD_RUN} steps"
            )
        if len(self.entropy) > 5 and self.entropy[-1] < ENTROPY_FLOOR * statistics.fmean(
            self.entropy[:5]
        ):
            self._halt(state, control, "entropy under a third of its steps 1-5 mean")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.val:
            return
        rates = [sum(v) / len(v) for v in self.val.values()]
        n = len(rates)
        rec = {
            "step": state.global_step,
            "val_pass@1": round(statistics.fmean(rates), 4),
            "val_pass@8": round(sum(r > 0 for r in rates) / n, 4),
            "val_se": round(statistics.stdev(rates) / math.sqrt(n), 4) if n > 1 else None,
            "val_tasks": n,
            "val_samples": sum(len(v) for v in self.val.values()),
        }
        self.val = defaultdict(list)
        self.curve.append(rec)
        self.log.write(rec)
        print(json.dumps(rec))
        best = max(self.curve, key=lambda e: e["val_pass@1"])
        since = [e for e in self.curve if e["step"] > best["step"]]
        w = REWARD_WINDOW
        rising = len(self.rewards) >= 2 * w and statistics.fmean(
            self.rewards[-w:]
        ) > statistics.fmean(self.rewards[-2 * w : -w])
        if len(since) >= FLAT_EVALS and rising:
            self._halt(
                state,
                control,
                f"no new best val pass@1 in {FLAT_EVALS} evaluations while the train reward rose",
            )


def checkpoint_rule(summary: dict) -> dict:
    """Stage 5's checkpoint rule (pre-registered): the best grpo_val pass@1 among the saves at or
    before the stop; within one SE of the best counts as a tie, and ties go to the earliest."""
    curve = {e["step"]: e for e in summary["val_curve"]}
    stop = (summary.get("stop") or {}).get("step") or summary["steps"]
    saves = sorted(int(c.split("-")[1]) for c in summary["checkpoints"])
    cands = [s for s in saves if s <= stop and s in curve]
    rule = "best grpo_val pass@1 among saves <= the stop; within 1 SE of the best = tie -> earliest"
    if not cands:
        raise ValueError(f"checkpoint rule: no save with a grpo_val evaluation (saves {saves})")
    top = max(cands, key=lambda s: (curve[s]["val_pass@1"], -s))
    se = curve[top]["val_se"] or 0.0
    pick = min(s for s in cands if curve[s]["val_pass@1"] >= curve[top]["val_pass@1"] - se)
    return {
        "rule": rule,
        "steps": summary["steps"],
        "stop": summary.get("stop"),
        "best_step": top,
        "best_val_pass@1": curve[top]["val_pass@1"],
        "best_se": se,
        "picked_step": pick,
        "picked_val_pass@1": curve[pick]["val_pass@1"],
        "checkpoint": f"checkpoint-{pick}",
    }


def rows(path: str) -> list[dict]:
    keep = lambda t: {
        "prompt": t["prompt"],
        "verifier": json.dumps(t["verifier"]),  # one Arrow type for every verifier kind
        "task_id": t["id"],
        "format": t["format"],
    }
    return [keep(t) for t in load(path)]


def round_robin(recs: list[dict], n: int) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by[r["format"]].append(r)
    out, k = [], 0
    while len(out) < n and any(k < len(v) for v in by.values()):
        out += [v[k] for _, v in sorted(by.items()) if k < len(v)]
        k += 1
    return out[:n]


def fixed_generations(val: list[dict], model_name: str) -> list[dict]:
    """10 grpo_val tasks, round robin over formats: rendered with mistral-common, and their gold
    (or the gold passage, or the abstain sentence) as the reference shown next to each output."""
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    tok = tokenizer(model_name)
    out = []
    for t in round_robin(val, N_GENERATIONS):
        v = json.loads(t["verifier"])
        ids = tok.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=t["prompt"][0]["content"])])
        ).tokens
        ref = v.get("gold") or v.get("gold_chunk") or v.get("phrase")
        out.append({"id": t["task_id"], "format": t["format"], "prompt_ids": ids, "chosen": ref})
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
    train_recs, val_recs = rows(d["train"]), rows(d["val"])
    if smoke:
        train_recs, val_recs = (
            round_robin(train_recs, SMOKE_TRAIN),
            round_robin(val_recs[::-1], SMOKE_VAL),
        )
    by_format = dict(sorted(Counter(r["format"] for r in train_recs).items()))
    seed = t.get("seed", 0)
    patch_vllm(seed)
    log.write(
        {
            "step": 0,
            "dataset_hash": data_hash,
            "train_tasks": len(train_recs),
            "val_tasks": len(val_recs),
            "by_format": by_format,
            "init_from": model_name,
            "fp32_logps": grpo_trainer.selective_log_softmax is selective_log_softmax_fp32,
            "vllm_args": VLLM_ARGS,
        }
    )
    print(f"dataset {data_hash}: {len(train_recs)} tasks, val {len(val_recs)}; from {model_name}")

    set_seed(seed)  # before the model: TRL wraps the LoRA adapters (random A init)
    model, tok = load_model_and_tokenizer(cfg["model"])
    peft_cfg = lora_config(cfg)
    watch = Watch(log)
    correctness = Rollouts(results / "rollouts.jsonl", watch)
    gens = fixed_generations(val_recs, model_name)
    every = t.get("save_steps", 0) if t.get("save_strategy", "steps") == "steps" else 0
    extras = DPOExtras(log, gens, results / "eval_generations.jsonl", model_name, every)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness, length_penalty],
        args=GRPOConfig(**t),
        train_dataset=Dataset.from_list(train_recs),
        eval_dataset=Dataset.from_list(val_recs),
        processing_class=tok,  # MistralCommonBackend: renders the prompts as mistral-common does
        peft_config=peft_cfg,
        callbacks=[log, watch, extras, *(callbacks or [])],
    )
    correctness.trainer = trainer

    trainable = {n: p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad}
    want = expected_trainable(model, cfg["lora"]["r"])
    stray = sorted(n for n in trainable if "lora_" not in n or "language_model" not in n)
    check = {
        "step": 0,
        "check": "trainable_params",
        "trainable": sum(trainable.values()),
        "expected": want,
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
    train_logs = [h for h in hist if "reward" in h and not any(k.startswith("eval_") for k in h)]
    wall = time.time() - started
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last_n = lambda key: (
        round(statistics.fmean(h[key] for h in train_logs[-10:] if key in h), 4)
        if any(key in h for h in train_logs[-10:])
        else None
    )
    summary = {
        "stage": "grpo",
        "run": run_name,
        "base": model_name,
        "lora": peft_cfg is not None,
        "gpus": world,
        "dataset_hash": data_hash,
        "tasks": len(train_recs),
        "val_tasks": len(val_recs),
        "by_format": by_format,
        "completions_per_step": t["per_device_train_batch_size"]
        * t["gradient_accumulation_steps"]
        * world,
        "num_generations": trainer.args.num_generations,
        "steps": trainer.state.global_step,
        "loss_type": trainer.args.loss_type,
        "scale_rewards": trainer.args.scale_rewards,
        "beta": trainer.args.beta,
        "epsilon": [trainer.args.epsilon, trainer.args.epsilon_high],
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "wall_s": round(wall, 1),
        "gpu_hours": round(wall * world / 3600, 3),
        "final_reward": last_n("reward"),
        "final_entropy": last_n("entropy"),
        "final_mean_length": last_n("completions/mean_length"),
        "val_curve": watch.curve,  # the checkpoint rule's input (modal_train.py --merge-from rule)
        "stop": watch.stop,
        "base_drift_max": getattr(watch, "base_drift", None),
        "checkpoints": [p.name for p in ckpts],
        "peak_mem_gb": peak_mem_gb(),
        "smoke": smoke,
        "data": {f: sha256(f) for f in (d["train"], d["val"])},
        "config": cfg,
    }
    for dd in (results, Path(out_dir)):
        (dd / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("config", "data", "val_curve")})
    )
    return summary


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
