# End-to-end pipeline. Each target is resumable; outputs land in data/, checkpoints/, results/.
PY ?= .venv/bin/python
MODAL ?= .venv/bin/modal

.PHONY: data sft-data sft-replay dpo-data dpo-pairs dpo-pairs-strict grpo-data cpt sft dpo grpo train serve \
	bench-data bench report audit reproduce-score reproduce-stage0 reproduce-stage1 reproduce-stage2 \
	reproduce-stage3 reproduce-stage4 reproduce-stage5 reproduce-stage6

# --- data -------------------------------------------------------------------
# Stage 1 corpus: data/sources.csv -> data/processed/{docs_raw,docs,train,val,replay,general_val}.jsonl
# + stats.json.
# Each step reads the previous step's output; run from the repo root with .env loaded (HF_TOKEN).
# New sources: data/scripts/crawl_index.py <index> --pattern ... --publisher ..., then `make data`.
DATA_STEPS = download extract filter dedup pii split replay tokenizer_coverage stats
data:
	for s in $(DATA_STEPS); do $(PY) data/scripts/$$s.py || exit 1; done

# --- SFT data ---------------------------------------------------------------
# Stage 3 SFT set: data/processed/chunks.jsonl + eval/tasks (seen half) -> data/sft/{train,sft_val}.jsonl.
# Mistral API (MISTRAL_API_KEY, ~6,000 calls at 30 a minute) and HF_TOKEN (Tulu 3 replay); every
# call is cached in data/sft/.cache, so a rerun only pays for what changed. Steps: data/scripts/sft_*.py.
SFT_STEPS = sft_pool sft_questions sft_filter sft_answers sft_judge sft_replay sft_assemble
sft-data:
	for s in $(SFT_STEPS); do $(PY) data/scripts/$$s.py || exit 1; done
	$(PY) eval/contamination.py --only sft
	$(PY) data/scripts/sft_replay_fetch.py --strip   # -> data/sft/hosted/ (what git keeps)
# a fresh clone: rebuild data/sft/{train,sft_val}.jsonl from data/sft/hosted/ and the Tülu 3 mixture
sft-replay:
	$(PY) data/scripts/sft_replay_fetch.py

# --- DPO data (Stage 4) -----------------------------------------------------
# Prompt pool + judge-benchmark prompts (Mac), then GPU sampling and API scoring in between
# (commands: CLAUDE.md, Stage 4), then pairs. Mistral API for the judge (.env loaded).
dpo-data:
	$(PY) data/scripts/dpo_prompts.py
	$(PY) data/scripts/dpo_judge_bench.py prep
	$(PY) eval/contamination.py --only dpo
dpo-pairs:
	$(PY) data/scripts/dpo_score.py
	$(PY) data/scripts/dpo_pairs.py
	$(PY) eval/contamination.py --only dpo
	$(PY) -m pytest tests/test_dpo_data.py -q
# dpo-strict (2026-10-09): closed-book labels by eval/scorers.qa_strict -> data/dpo/strict/.
# dpo_pairs.py exits 1 at 463 pairs (< the 500 floor, overridden for this rerun: notes/decisions.md).
dpo-pairs-strict:
	$(PY) data/scripts/dpo_score.py --strict
	-$(PY) data/scripts/dpo_pairs.py --strict
	$(PY) eval/contamination.py --only dpo
	$(PY) -m pytest tests/test_dpo_data.py tests/test_qa_strict.py -q
# Stage 5: candidate tasks with verifiers; the window (train/val) comes from the probe, grpo_probe.py
grpo-data:
	$(PY) data/scripts/grpo_tasks.py
	$(PY) eval/contamination.py --only grpo
	$(PY) -m pytest tests/test_qa_strict.py tests/test_grpo_rewards.py tests/test_grpo_data.py -q

# --- train ------------------------------------------------------------------
# Stage 2 runs on Modal (commands and ablations: .claude/skills/stage2-cpt/SKILL.md); this starts the main run.
cpt:  ; $(MODAL) run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b --steps train
sft:  ; $(PY) train/sft.py  --config train/configs/sft.yaml
dpo:  ; $(PY) train/dpo.py  --config train/configs/dpo.yaml
grpo: ; $(PY) train/grpo.py --config train/configs/grpo.yaml
train: cpt sft dpo grpo

# --- serve ------------------------------------------------------------------
# Stage 6: quantized checkpoints are built on Modal (serve/modal_serve.py --action quantize; DEPLOY.md)
serve:
	bash serve/serve_vllm.sh $(MODEL)

bench-data:
	$(PY) serve/bench_data.py

bench:
	$(PY) serve/bench_latency.py --model struct-lm --concurrency 1 8 32

# --- report and reproduce (docs/reproduce.md) --------------------------------
report:
	$(PY) train/report.py
audit:
	$(PY) train/readme_audit.py
	$(PY) -m pytest tests/test_docs.py tests/test_readme_audit.py -q

# No GPU, no API key: every row of results/table.md rescored from its committed generations and
# compared, the strict checker and pass@k re-run, every table and figure regenerated; a clean diff
# afterwards means the committed numbers reproduce.
PASSK_RUNS = instruct-8b sft-from-cpt dpo dpo-seed1 dpo-strict grpo grpo-seed1
reproduce-score:
	$(PY) eval/rescore_all.py
	$(PY) eval/qa_strict.py
	$(PY) eval/summary_stats.py
	$(PY) eval/passk.py $(PASSK_RUNS)
	$(PY) train/report.py
	git diff --stat --exit-code -- results docs README.md DEPLOY.md

# Each stage on Modal, from the committed data (CLAUDE.md has every step and the pull/score
# commands). These launch GPU jobs: run one stage after the previous stage's checkpoints exist,
# with .env loaded. Ablations and probes beyond the chain: docs/reproduce.md.
RUN_GPU = $(MODAL) run --detach
reproduce-stage0:  # row 0: the instruct bar (KPI + lm-eval); the base comes with Stage 2's base-8b-hf
	$(RUN_GPU) eval/modal_app.py --which kpi --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
		--run-name instruct-8b --chat --generate-only
	$(RUN_GPU) eval/modal_app.py --which lm --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
		--run-name instruct-8b
reproduce-stage1: data
reproduce-stage2:  # the base re-saved through merge.py, the chain's CPT, the main run and its seed, full-parameter
	$(RUN_GPU) train/modal_train.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b-hf \
		--steps merge,eval
	$(RUN_GPU) train/modal_train.py --config train/configs/cpt_replay10.yaml --run-name cpt-8b-replay10
	$(RUN_GPU) train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b
	$(RUN_GPU) train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b-seed1 \
		--overrides "training.seed=1"
	$(RUN_GPU) train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name cpt-8b-full --gpus 2
SFT_CHAIN = --config train/configs/sft.yaml --merge-from b4 --chat \
	--steps train,merge,mergecheck,ppl,eval,latency,sample
reproduce-stage3:  # both arms, two seeds each (the two-arm floor)
	$(PY) data/scripts/sft_replay_fetch.py   # the SFT set, checked against data/sft/SHA256SUMS
	for f in train.jsonl sft_val.jsonl SHA256SUMS; do \
		$(MODAL) volume put --force struct-lm data/sft/$$f data/sft/$$f; done
	$(RUN_GPU) train/modal_train.py $(SFT_CHAIN) --run-name sft-from-cpt
	$(RUN_GPU) train/modal_train.py $(SFT_CHAIN) --run-name sft-from-cpt-seed1 \
		--overrides "training.seed=1 training.data_seed=1"
	$(RUN_GPU) train/modal_train.py $(SFT_CHAIN) --run-name sft-from-base \
		--overrides "model.init_from=checkpoints/base-8b-hf"
	$(RUN_GPU) train/modal_train.py $(SFT_CHAIN) --run-name sft-from-base-seed1 \
		--overrides "model.init_from=checkpoints/base-8b-hf training.seed=1 training.data_seed=1"
reproduce-stage4:  # dpo-strict, the chain's DPO, on the committed strict pairs
	for f in train.jsonl val.jsonl SHA256SUMS; do \
		$(MODAL) volume put --force struct-lm data/dpo/strict/$$f data/dpo/strict/$$f; done
	$(RUN_GPU) train/modal_train.py --config train/configs/dpo.yaml --run-name dpo-strict --chat \
		--merge-from rule --steps noop,train,merge,mergecheck,ppl,eval,sample \
		--overrides "data.dir=data/dpo/strict data.train=data/dpo/strict/train.jsonl data.val=data/dpo/strict/val.jsonl"
reproduce-stage5:  # GRPO from dpo-strict, two seeds, on the committed tasks
	for f in tasks.jsonl train.jsonl val.jsonl SHA256SUMS; do \
		$(MODAL) volume put --force struct-lm data/grpo/$$f data/grpo/$$f; done
	$(RUN_GPU) train/modal_train.py --config train/configs/grpo.yaml --run-name grpo --chat \
		--merge-from rule --steps noop,train,merge,mergecheck,ppl,eval,latency,sample
	$(RUN_GPU) train/modal_train.py --config train/configs/grpo.yaml --run-name grpo-seed1 --chat \
		--merge-from rule --steps noop,train,merge,mergecheck,ppl,eval,sample \
		--overrides "training.seed=1 training.data_seed=1"
reproduce-stage6:  # FP8 and INT4 from dpo-strict, the quality gate, the bench on one pinned H100
	$(MODAL) run serve/modal_serve.py --action quantize --scheme fp8
	$(RUN_GPU) serve/modal_serve.py --action quantize --scheme w4a16
	$(MODAL) run serve/modal_serve.py --action gate --variant bf16
	for v in fp8 fp8kv w4a16; do $(RUN_GPU) serve/modal_serve.py --action gate --variant $$v; done
	$(RUN_GPU) serve/modal_serve.py --action bench --variants bf16,fp8
