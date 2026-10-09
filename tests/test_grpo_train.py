"""Stage 5 trainer (train/grpo.py) on CPU, no model weights and no vLLM.

- prompts: TRL's GRPO path renders a conversational prompt through transformers'
  MistralCommonBackend (its _tokenize_prompts call, verbatim) to exactly mistral-common's
  encode_chat_completion ids, the KPI eval's --chat rendering, on real task prompts.
- the TRL changes: grpo_trainer's selective_log_softmax is the fp32 replacement, and the vLLM
  wrapper adds the repo's engine settings and the run's seed.
- grpo.train() end to end on a tiny random Mistral3 (HF generation instead of vLLM): data hash,
  trainable check, rollouts written with their reward parts, grpo_val pass@1/pass@8 at step 0
  and every evaluation, saves, summary, and the checkpoint rule.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = "mistralai/Ministral-3-8B-Base-2512"
TASKS = REPO / "data/grpo/tasks.jsonl"


@pytest.fixture(scope="module")
def grpo():
    # train/ first and no cached data/scripts `common` (as tests/test_dpo_train.py)
    path = str(REPO / "train")
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    m = sys.modules.get("common")
    if m is not None and "data/scripts" in (getattr(m, "__file__", "") or ""):
        del sys.modules["common"]
    import grpo

    return grpo


def tasks(n_per_format: int) -> list[dict]:
    rows = [json.loads(x) for x in TASKS.open()]
    out = []
    for f in ("closed_book", "grounded", "abstain"):
        out += [r for r in rows if r["format"] == f][:n_per_format]
    return out


@pytest.mark.skipif(not TASKS.exists(), reason="no data/grpo/tasks.jsonl")
def test_trl_prompt_ids_are_mistral_commons():
    from huggingface_hub import try_to_load_from_cache
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers import AutoTokenizer

    cached = try_to_load_from_cache(BASE, "tekken.json")
    if not isinstance(cached, str):
        pytest.skip(f"no cached {BASE} tokenizer")
    tok = AutoTokenizer.from_pretrained(Path(cached).parent)
    mc = MistralTokenizer.from_file(cached)
    sample = tasks(7)[:20]
    # grpo_trainer._tokenize_prompts, conversational branch
    enc = tok.apply_chat_template(
        conversation=[t["prompt"] for t in sample],
        tools=None,
        chat_template=None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
    )
    for t, ids, mask in zip(sample, enc["input_ids"], enc["attention_mask"], strict=True):
        got = [i for i, m in zip(ids, mask) if m]
        want = mc.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=t["prompt"][0]["content"])])
        ).tokens
        assert got == want, t["id"]
        assert got[:2] == [1, 3] and got[-1] == 4


def test_fp32_logps_bound(grpo):
    from trl.trainer import grpo_trainer

    assert grpo_trainer.selective_log_softmax is grpo.selective_log_softmax_fp32


def test_vllm_wrapper(grpo, monkeypatch):
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(grpo.vllm_generation, "LLM", FakeLLM, raising=False)
    grpo.patch_vllm(7)
    grpo.patch_vllm(7)  # idempotent: wraps the original, not the wrapper
    grpo.vllm_generation.LLM(model="checkpoints/dpo-strict", seed=0, gpu_memory_utilization=0.35)
    assert seen["seed"] == 7 and seen["gpu_memory_utilization"] == 0.35
    assert seen["model"] == "checkpoints/dpo-strict"
    for k, v in grpo.VLLM_ARGS.items():
        assert seen[k] == v


def test_checkpoint_rule(grpo):
    def summary(curve, saves, stop=None, steps=150):
        return {
            "val_curve": [{"step": s, "val_pass@1": p, "val_se": se} for s, p, se in curve],
            "checkpoints": [f"checkpoint-{s}" for s in saves],
            "stop": stop,
            "steps": steps,
        }

    curve = [
        (0, 0.40, 0.04),
        (25, 0.48, 0.04),
        (50, 0.55, 0.04),
        (75, 0.53, 0.04),
        (100, 0.60, 0.04),
    ]
    saves = [25, 50, 75, 100]
    # best 0.60 at 100; 0.55 at 50 is not within 1 SE (0.04): 100
    assert grpo.checkpoint_rule(summary(curve, saves))["checkpoint"] == "checkpoint-100"
    # 0.58 at 50 is within 1 SE of 0.60: the earliest tie, 50
    curve[2] = (50, 0.58, 0.04)
    assert grpo.checkpoint_rule(summary(curve, saves))["checkpoint"] == "checkpoint-50"
    # a stop at 80: only saves at or before it count; step 0 (no save) never does
    r = grpo.checkpoint_rule(summary(curve, saves, stop={"step": 80, "reason": "x"}))
    assert r["best_step"] == 50 and r["checkpoint"] == "checkpoint-50"


@pytest.fixture(scope="module")
def tiny_mistral3(tmp_path_factory):
    """The tiny random Mistral3ForConditionalGeneration of tests/test_dpo_train.py."""
    import shutil

    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig, AutoModelForImageTextToText

    cached = try_to_load_from_cache(BASE, "config.json")
    if not isinstance(cached, str):
        pytest.skip(f"no cached {BASE} config")
    src = Path(cached).parent
    cfg = AutoConfig.from_pretrained(src)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 32, 64, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 16
    v = cfg.vision_config
    v.hidden_size, v.intermediate_size, v.num_hidden_layers = 16, 32, 1
    v.num_attention_heads, v.head_dim = 1, 16
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg)
    out = tmp_path_factory.mktemp("tiny-mistral3")
    model.save_pretrained(out)
    for f in ("tekken.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    return out


@pytest.mark.skipif(not TASKS.exists(), reason="no data/grpo/tasks.jsonl")
def test_train_function_end_to_end(grpo, tiny_mistral3, tmp_path, monkeypatch):
    import dpo
    import yaml

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/grpo"
    data.mkdir(parents=True)
    rows = tasks(4)
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:8]))
    (data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[8:12]))
    sums = "".join(
        f"{hashlib.sha256((data / n).read_bytes()).hexdigest()}  {n}\n"
        for n in ("train.jsonl", "val.jsonl")
    )
    (data / "SHA256SUMS").write_text(sums)
    cfg = yaml.safe_load((REPO / "train/configs/grpo.yaml").read_text())
    cfg["model"].update(init_from=str(tiny_mistral3), dtype="float32")
    cfg["lora"]["r"] = 4
    cfg["training"].update(
        output_dir="checkpoints/_train/tiny",
        run_name="tiny",
        use_vllm=False,
        num_generations=2,
        per_device_train_batch_size=4,  # 2 tasks x 2 completions per micro-batch
        gradient_accumulation_steps=2,  # 4 tasks per step
        max_steps=2,
        max_completion_length=8,
        num_generations_eval=2,
        per_device_eval_batch_size=4,  # 4 val tasks: 2 eval batches
        eval_steps=1,
        save_steps=1,
        warmup_steps=0,
        bf16=False,
        tf32=False,
        gradient_checkpointing=False,
        use_cpu=True,
        log_completions=False,
    )
    monkeypatch.setattr(dpo, "GEN_TOKENS", 8)
    s = grpo.train(cfg)
    assert s["stage"] == "grpo" and s["steps"] == 2 and s["tasks"] == 8
    assert [e["step"] for e in s["val_curve"]] == [0, 1, 2]
    for e in s["val_curve"]:
        assert e["val_tasks"] == 4 and e["val_samples"] == 8
        assert 0 <= e["val_pass@1"] <= e["val_pass@8"] <= 1
    assert s["checkpoints"] == ["checkpoint-1", "checkpoint-2"]
    log = [json.loads(x) for x in (tmp_path / "results/runs/tiny/train_log.jsonl").open()]
    assert next(r for r in log if r.get("check") == "trainable_params")["passed"]
    assert next(r for r in log if "fp32_logps" in r)["fp32_logps"]
    train_logs = [r for r in log if "reward" in r]
    assert {"rewards/format_reward/mean", "rewards/correctness_reward/mean",
            "rewards/length_penalty/mean", "frac_reward_zero_std", "entropy"} <= set(train_logs[0])  # fmt: skip
    roll = [json.loads(x) for x in (tmp_path / "results/runs/tiny/rollouts.jsonl").open()]
    assert {r["mode"] for r in roll} == {"train", "eval"}
    assert sum(r["mode"] == "train" for r in roll) == 2 * 4 * 2  # steps x tasks x generations
    for r in roll:
        assert r["correct"] <= r["format"] <= r["terminated"]  # the gate: no credit unterminated
        assert r["total"] == pytest.approx(0.1 * r["format"] + 0.9 * r["correct"] + r["length"])
    assert s["base_drift_max"] == 0.0  # no vLLM sync: the base is never merged into
    rule = grpo.checkpoint_rule(s)
    assert rule["checkpoint"] in ("checkpoint-1", "checkpoint-2")


def test_report_stage5(tmp_path, monkeypatch):
    """report.py's Stage 5 branch on two synthetic runs: the run table with the rule's pick from
    b4.json, and grpo.png (written to a scratch dir; the README is not touched)."""
    sys.path.insert(0, str(REPO / "train"))
    import report

    runs_dir = tmp_path / "runs"
    runs = {}
    for name, shift in (("grpo", 0.0), ("grpo-seed1", 0.02)):
        log = []
        for step in range(1, 151):
            log.append(
                {
                    "step": step,
                    "loss": 0.01,
                    "reward": 0.45 + 0.002 * step + shift,
                    "entropy": 0.4 - 0.001 * step,
                    "completions/mean_length": 20 + 0.01 * step,
                    "frac_reward_zero_std": 0.2 + 0.002 * step,
                    "sampling/sampling_logp_difference/mean": 0.006,
                }
            )
            if step % 25 == 0:
                log.append({"step": step, "val_pass@1": 0.4 + step / 1000, "val_pass@8": 0.8,
                            "val_se": 0.03, "val_tasks": 50})  # fmt: skip
        curve = [{"step": s, "val_pass@1": 0.4 + s / 1000, "val_pass@8": 0.8, "val_se": 0.03}
                 for s in range(0, 151, 25)]  # fmt: skip
        summary = {
            "stage": "grpo", "base": "checkpoints/dpo-strict", "tasks": 622, "steps": 150,
            "stop": None, "val_curve": curve, "final_reward": 0.75, "final_entropy": 0.25,
            "final_mean_length": 21.5, "base_drift_max": 2.4e-4, "wall_s": 5400,
            "gpu_hours": 1.5, "peak_mem_gb": 61.0,
        }  # fmt: skip
        d = runs_dir / name
        d.mkdir(parents=True)
        (d / "train_summary.json").write_text(json.dumps(summary))
        (d / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        runs[name] = (summary, log)
    (runs_dir / "grpo" / "b4.json").write_text(json.dumps({"picked_step": 125, "best_step": 150}))
    monkeypatch.setattr(report, "RUNS", runs_dir)
    md = report.grpo_table(runs, 3.95)
    assert "| grpo | dpo-strict | 622 | 150 | none | step 125 (best 150) | 0.525 / 0.800 |" in md
    report.plot_grpo(runs, tmp_path / "grpo.png")
    assert (tmp_path / "grpo.png").stat().st_size > 10_000
