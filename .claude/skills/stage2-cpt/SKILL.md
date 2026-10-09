---
name: stage2-cpt
description: Stage 2 CPT runbook - smoke tests, main run, ablations (replay, 8B full-parameter, 2-GPU LoRA), seed and LR-up runs, memorisation probe, and the pull/score/report commands. Use when relaunching or reproducing a Stage 2 run.
---

# Stage 2: CPT (Modal)

Every `modal run` below is a GPU launch: ask the user first (CLAUDE.md, Environment). The rules
that apply to these runs (YAML overrides, seed floor, steps and volume layout) are in CLAUDE.md,
Stage 2.

```bash
M=.venv/bin/modal   # from the repo root; one pipeline.remote() per launch, so --detach is safe
# smoke tests (20 steps): main, 8B full-parameter (2 GPUs), 2-GPU LoRA scaling
$M run train/modal_train.py --config train/configs/cpt.yaml --run-name smoke-cpt --smoke --steps train,merge,ppl
$M run eval/modal_app.py --which kpi --model /vol/checkpoints/smoke-cpt --run-name smoke-cpt --limit 5 --no-judge --generate-only
$M run train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name smoke-8b-full --gpus 2 --smoke --steps train,merge
$M run train/modal_train.py --config train/configs/cpt.yaml --run-name smoke-fsdp --gpus 2 --smoke --steps train \
  --overrides "training.gradient_checkpointing=false training.eval_strategy=no"
# reference row (base-8b already has lm-eval, KPI generation and latency from Stage 0)
$M run --detach train/modal_train.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b --steps ppl
# main run: train, read the curve, then merge + perplexity + eval
$M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b --steps train
$M run --detach train/modal_train.py --run-name cpt-8b --steps merge,ppl,eval
# ablations: A replay, B 8B full-parameter (2 GPUs), C 2-GPU LoRA scaling (throughput only)
$M run --detach train/modal_train.py --config train/configs/cpt_replay10.yaml --run-name cpt-8b-replay10
$M run --detach train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name cpt-8b-full --gpus 2
$M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-fsdp2 --gpus 2 --steps train \
  --overrides "training.gradient_accumulation_steps=4 training.gradient_checkpointing=false training.eval_strategy=no run.stop_at_step=100"
# pull per run, then score locally (no --chat: every Stage 2 checkpoint is a base model)
$M volume get --force struct-lm results/runs/<run> results/runs/
$M volume get --force struct-lm results/ppl/<run>.json results/ppl/
.venv/bin/python eval/run_eval.py --run-name cpt-8b --rescore --lm-eval-dir results/lm_eval \
  --ppl-dir results/ppl --model /vol/checkpoints/cpt-8b
.venv/bin/python train/report.py   # -> results/train_runs.md, results/curves/cpt.png, cpt_ppl.png
.venv/bin/python eval/ppl_compare.py base-8b cpt-8b   # paired bootstrap CIs, nats and % ppl
.venv/bin/python eval/contamination.py   # 13-gram overlap: val/2026 vs train, benchmarks vs train+replay,
                                         # few-shot vs domain_qa -> results/contamination.md (rerun after data changes)
# noise floor and LR-up (decisions.md, ablation rules): same config, one change each
$M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-seed1 --overrides "training.seed=1"
$M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-lr2x --overrides "training.learning_rate=2.0e-4"
# memorisation probe: corpus vs post-cutoff documents (eval/exposure_sources.csv, PDFs curl'd into
# data/exposure/, gitignored), base plus the CPT checkpoint as positive control
.venv/bin/python eval/memorization.py build            # -> data/exposure/docs.jsonl
$M volume put --force struct-lm data/exposure/docs.jsonl data/exposure/docs.jsonl
$M run --detach eval/modal_app.py::memorization --model /vol/checkpoints/base-8b-hf --run-name base-8b-hf
mkdir -p results/exposure && $M volume get --force struct-lm results/exposure/base-8b-hf.json results/exposure/base-8b-hf.json
.venv/bin/python eval/memorization.py report base-8b-hf cpt-8b-replay10
```
