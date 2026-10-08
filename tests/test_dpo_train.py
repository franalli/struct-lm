"""Stage 4 trainer (train/dpo.py) on CPU, no model weights: the two changes to TRL 0.29.1's
DPOTrainer and the pre-tokenised path, run end to end on a tiny random model.

  - fp32 log-probs: dpo_trainer's selective_log_softmax is replaced, every call site resolves the
    replacement, and on bf16 logits it matches an fp64 reference where TRL's bf16 path doesn't.
  - length_norm: the masked sum of the scaled per-token log-probs is the per-token mean, for the
    policy and for the precomputed reference alike.
  - the trainer takes prompt_ids / chosen_ids / rejected_ids as they are (no re-tokenising, no
    BOS/EOS added), precomputes the reference with the adapter disabled, and at step 1 (LoRA B = 0)
    policy and reference agree, so the rewards are 0.
  - merge_check reads pair files: two sequences per pair, the mask over the completion.

The tiny model has the Ministral tokenizer's vocabulary; the tokenizer comes from the local HF
cache (as tests/test_template.py)."""

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
BASE = "mistralai/Ministral-3-8B-Base-2512"


@pytest.fixture(scope="module")
def dpo():
    # tests/test_sft_data.py puts data/scripts (its own common.py) on sys.path at collection:
    # train/ goes first again, and a cached data/scripts `common` is dropped before the import
    sys.path.remove(str(REPO / "train"))
    sys.path.insert(0, str(REPO / "train"))
    m = sys.modules.get("common")
    if m is not None and "data/scripts" in (getattr(m, "__file__", "") or ""):
        del sys.modules["common"]
    import dpo

    return dpo


def test_every_call_site_uses_the_fp32_replacement(dpo):
    import inspect

    from trl.trainer import dpo_trainer

    assert dpo_trainer.selective_log_softmax is dpo.selective_log_softmax_fp32
    src = inspect.getsource(dpo_trainer)
    # the import binds the name in dpo_trainer's namespace; the calls look it up there
    assert "from .utils import" in src and "selective_log_softmax(" in src
    assert dpo.selective_log_softmax_fp32.trl_original is not dpo.selective_log_softmax_fp32


def test_fp32_logps_match_fp64(dpo):
    torch.manual_seed(0)
    logits = (torch.randn(2, 300, 1000) * 4).to(torch.bfloat16)
    index = torch.randint(0, 1000, (2, 300))
    ref = torch.log_softmax(logits.double(), -1).gather(-1, index[..., None])[..., 0].sum(1)
    ours = dpo.selective_log_softmax_fp32(logits, index)
    trl_bf16 = dpo.selective_log_softmax_fp32.trl_original(logits, index)
    assert ours.dtype == torch.float32
    assert torch.allclose(ours.double().sum(1), ref, atol=1e-3)
    # the bug being fixed: bf16 per-token values summed in bf16 are off by a visible amount
    assert (trl_bf16.sum(1).double() - ref).abs().max() > 1e-2


def test_length_norm_gives_the_per_token_mean(dpo):
    torch.manual_seed(1)
    logits = torch.randn(4, 12, 50)
    index = torch.randint(0, 50, (4, 12))
    mask = torch.zeros(4, 13, dtype=torch.long)  # shift: mask[:, 1:] lines up with index
    for row, n in enumerate((3, 5, 7, 11)):
        mask[row, 13 - n :] = 1
    plain = dpo.selective_log_softmax_fp32(logits, index)
    with dpo.length_normalised(mask):
        scaled = dpo.selective_log_softmax_fp32(logits, index)
    shift = mask[:, 1:]
    plain[shift == 0], scaled[shift == 0] = 0.0, 0.0
    assert torch.allclose(scaled.sum(1), plain.sum(1) / shift.sum(1))
    assert dpo._row_scale is None  # reset on exit


def pair(k: int) -> dict:
    p = [1, 3, 1500 + k, 1501, 1502, 4]
    return {
        "id": f"p{k}#0",
        "prompt_id": f"p{k}",
        "format": ("closed_book", "grounded", "definition", "abstain")[k % 4],
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}],
        "prompt_ids": p,
        "chosen_ids": [2000 + k, 2001, 2002, 2],
        "rejected_ids": [3000 + k, 2],
    }


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 2-layer Llama with the Ministral vocabulary, saved, plus 8 train / 4 val pairs."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained(BASE)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no cached {BASE} tokenizer: {e}")
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=131072,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        pad_token_id=11,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg).float()
    return model, tok, [pair(k) for k in range(8)], [pair(k) for k in range(8, 12)]


def make_trainer(dpo, tiny, tmp_path, length_norm: bool):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig

    model, tok, train, val = tiny
    cols = dpo.COLUMNS
    args = DPOConfig(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        max_steps=2,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        max_length=None,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=2,
        bf16=False,
        use_cpu=True,
        gradient_checkpointing=False,
        seed=0,
    )
    import copy

    return dpo.Trainer(
        model=copy.deepcopy(model),
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list([{c: r[c] for c in cols} for r in train]),
        eval_dataset=Dataset.from_list([{c: r[c] for c in cols} for r in val]),
        processing_class=tok,
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
        sequential=True,
        length_norm=length_norm,
    )


@pytest.mark.parametrize("length_norm", [False, True])
def test_tiny_end_to_end(dpo, tiny, tmp_path, length_norm):
    trainer = make_trainer(dpo, tiny, tmp_path, length_norm)
    _, _, train, _ = tiny
    # the dataset is the ids as given, plus the precomputed reference columns
    row = trainer.train_dataset[0]
    assert row["prompt_ids"] == train[0]["prompt_ids"]
    assert row["chosen_ids"] == train[0]["chosen_ids"]
    assert {"ref_chosen_logps", "ref_rejected_logps"} <= set(row)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    ids, cm = batch["input_ids"][0].tolist(), batch["completion_mask"][0].tolist()
    assert ids[: len(train[0]["prompt_ids"])] == train[0]["prompt_ids"]
    assert ids.count(1) == 1 and sum(cm) == len(train[0]["chosen_ids"])

    # the reference equals the adapter-disabled model's own log-probs, normalised or not
    model = trainer.model
    seq = torch.tensor([train[0]["prompt_ids"] + train[0]["chosen_ids"]])
    with torch.no_grad(), model.disable_adapter():
        lp = torch.log_softmax(model(input_ids=seq).logits[0, :-1].double(), -1)
    n_p, n_c = len(train[0]["prompt_ids"]), len(train[0]["chosen_ids"])
    tgt = seq[0, 1:]
    per_tok = lp.gather(-1, tgt[:, None])[:, 0][n_p - 1 :]
    want = per_tok.sum() / (n_c if length_norm else 1)
    assert abs(row["ref_chosen_logps"] - float(want)) < 1e-4

    trainer.train()
    step1 = next(h for h in trainer.state.log_history if h.get("step") == 1 and "loss" in h)
    # LoRA B = 0 at step 1: policy = reference, so the log-ratios and rewards are 0
    assert abs(step1["rewards/chosen"]) < 1e-5 and abs(step1["rewards/rejected"]) < 1e-5
    assert abs(step1["loss"] - 0.693147) < 1e-4
    step2 = next(h for h in trainer.state.log_history if h.get("step") == 2 and "loss" in h)
    assert step2["rewards/margins"] != 0.0


def test_text_checks(dpo):
    ok = dpo.text_checks(list(range(40)), "The load is applied to the beam at midspan.")
    assert ok["english"] and not ok["repetitive"] and ok["distinct_4"] == 1.0
    loop = dpo.text_checks([5, 6, 7, 8] * 6, "a\na\n")
    assert loop["repetitive"] and loop["max_4gram_count"] >= 4


def test_merge_check_reads_pair_files(tmp_path):
    import merge_check

    path = tmp_path / "val.jsonl"
    path.write_text("".join(json.dumps(pair(k)) + "\n" for k in range(3)))
    recs = merge_check.val_records(str(path), start="unused")
    assert len(recs) == 6
    first = recs[0]
    assert first["input_ids"] == pair(0)["prompt_ids"] + pair(0)["chosen_ids"]
    assert first["completion_mask"] == [0] * 6 + [1] * 4
    assert merge_check.probe_formats(recs) == ["closed_book", "grounded", "definition"]


@pytest.fixture(scope="module")
def tiny_mistral3(tmp_path_factory):
    """A tiny random Mistral3ForConditionalGeneration (the real config, shrunk) saved with the
    base's tokenizer files, so dpo.train() loads it exactly as it loads checkpoints/sft-from-cpt."""
    import shutil

    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig, AutoModelForImageTextToText

    cached = try_to_load_from_cache(BASE, "config.json")  # config + tokenizer files, no weights
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


def test_train_function_end_to_end(dpo, tiny_mistral3, tmp_path, monkeypatch):
    """dpo.train() as Modal runs it, on CPU: data hash, trainable check, precomputed reference,
    evals at every save and the last step, generations, summary, and the checkpoint rule."""
    import hashlib

    import yaml

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/dpo"
    data.mkdir(parents=True)
    for name, ks in (("train.jsonl", range(12)), ("val.jsonl", range(12, 16))):
        (data / name).write_text("".join(json.dumps(pair(k)) + "\n" for k in ks))
    sums = "".join(
        f"{hashlib.sha256((data / n).read_bytes()).hexdigest()}  {n}\n"
        for n in ("train.jsonl", "val.jsonl")
    )
    (data / "SHA256SUMS").write_text(sums)
    cfg = yaml.safe_load((REPO / "train/configs/dpo.yaml").read_text())
    cfg["model"].update(init_from=str(tiny_mistral3), dtype="float32")
    cfg["lora"]["r"] = 4
    cfg["training"].update(
        output_dir="checkpoints/_train/tiny",
        run_name="tiny",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        precompute_ref_batch_size=2,
        gradient_accumulation_steps=2,  # 12 pairs / 4 = 3 steps
        save_steps=2,
        eval_steps=10,
        bf16=False,
        tf32=False,
        use_cpu=True,
    )
    monkeypatch.setattr(dpo, "GEN_TOKENS", 8)
    s = dpo.train(cfg)
    assert s["stage"] == "dpo" and s["steps"] == 3 and s["pairs"] == 12
    # evaluated at the save (2) and the last step (3), though eval_steps is 10
    assert [e["step"] for e in s["eval_curve"]] == [2, 3]
    assert s["checkpoints"] == ["checkpoint-2", "checkpoint-3"]
    log = [json.loads(x) for x in (tmp_path / "results/runs/tiny/train_log.jsonl").open()]
    assert next(r for r in log if r.get("check") == "trainable_params")["passed"]
    assert next(r for r in log if r.get("check") == "step1_rewards")["passed"]
    assert next(r for r in log if "fp32_logps" in r)["fp32_logps"]
    gens = [json.loads(x) for x in (tmp_path / "results/runs/tiny/eval_generations.jsonl").open()]
    assert {g["model"] for g in gens} == {"start", "policy"}
    assert {g["step"] for g in gens if g["model"] == "policy"} == {2, 3}
    rule = dpo.checkpoint_rule(s)  # under 100 steps: the save nearest the midpoint (2 of 3)
    assert rule["ref_step"] == 2 and rule["checkpoint"] in ("", "checkpoint-2")


def test_checkpoint_rule(dpo):
    def summary(steps, losses, saves):
        return {
            "steps": steps,
            "eval_curve": [{"step": k, "eval_loss": v} for k, v in losses.items()],
            "checkpoints": [f"checkpoint-{k}" for k in saves],
        }

    long = {50: 0.50, 100: 0.45, 104: 0.44}
    assert dpo.checkpoint_rule(summary(104, long, [25, 50, 75, 100, 104]))["checkpoint"] == ""
    long[104] = 0.55
    r = dpo.checkpoint_rule(summary(104, long, [25, 50, 75, 100, 104]))
    assert r["checkpoint"] == "checkpoint-50" and r["end_above_ref"]
    short = {25: 0.6, 50: 0.5, 60: 0.55}  # 60 steps: midpoint 30 -> save 25
    r = dpo.checkpoint_rule(summary(60, short, [25, 50, 60]))
    assert r["ref_step"] == 25 and r["checkpoint"] == ""
    assert dpo.checkpoint_rule(summary(20, {20: 0.6}, [20]))["checkpoint"] == ""


def test_report_stage4(tmp_path, monkeypatch):
    """report.py's Stage 4 branch on two synthetic runs: the table, the rule column from b4.json,
    and dpo.png (written to a scratch dir; the README is not touched)."""
    import report

    runs_dir = tmp_path / "runs"
    runs = {}
    for name, shift in (("dpo", 0.0), ("dpo-seed1", 0.01)):
        log = []
        for step in range(1, 61):
            m = 0.05 * step
            log.append(
                {
                    "step": step,
                    "loss": 0.69 - 0.004 * step + shift,
                    "rewards/margins": m,
                    "rewards/accuracies": min(0.9, 0.5 + 0.01 * step),
                    "logps/chosen": -50 - 0.01 * step,
                    "logps/rejected": -60 - 0.1 * step,
                }
            )
            if step % 10 == 0 or step in (25, 50, 60):
                log.append(
                    {
                        "step": step,
                        "eval_loss": 0.69 - 0.003 * step,
                        "eval_rewards/margins": m,
                        "eval_rewards/accuracies": 0.7,
                    }
                )
        summary = {
            "stage": "dpo",
            "base": "checkpoints/sft-from-cpt",
            "pairs": 960,
            "steps": 60,
            "tokens_per_s": 1000.0,
            "wall_s": 3600,
            "gpu_hours": 1.0,
            "peak_mem_gb": 60.0,
            "final_train_loss": 0.45,
            "final_eval": {"eval_loss": 0.51, "eval_rewards/accuracies": 0.8},
            "length_norm": False,
        }
        d = runs_dir / name
        d.mkdir(parents=True)
        (d / "train_summary.json").write_text(json.dumps(summary))
        (d / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        runs[name] = (summary, log)
    (runs_dir / "dpo" / "b4.json").write_text(
        json.dumps(
            {
                "steps": 60,
                "ref_step": 25,
                "eval_loss_ref": 0.615,
                "eval_loss_end": 0.51,
                "picked_step": 60,
                "checkpoint": "",
            }
        )
    )
    monkeypatch.setattr(report, "RUNS", runs_dir)
    md = report.stage4_md(runs, 3.95)
    assert "| dpo | sft-from-cpt | 960 | 60 |" in md
    assert "step 60 (final): 0.5100 vs 0.6150 at 25" in md
    report.plot_dpo(runs, tmp_path / "dpo.png")
    assert (tmp_path / "dpo.png").stat().st_size > 10_000
