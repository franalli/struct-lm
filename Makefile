# End-to-end pipeline. Each target is resumable; outputs land in data/, checkpoints/, results/.
PY ?= .venv/bin/python
MODAL ?= .venv/bin/modal

.PHONY: data sft-data train eval serve all

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

# --- train ------------------------------------------------------------------
# Stage 2 runs on Modal (commands and ablations: CLAUDE.md, Stage 2); this starts the main run.
cpt:  ; $(MODAL) run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b --steps train
sft:  ; $(PY) train/sft.py  --config train/configs/sft.yaml
dpo:  ; $(PY) train/dpo.py  --config train/configs/dpo.yaml
grpo: ; $(PY) train/grpo.py --config train/configs/grpo.yaml
train: cpt sft dpo grpo

# --- eval (expects a server from `make serve` on :8000) ---------------------
eval:
	$(PY) eval/run_eval.py --tasks eval/tasks --model struct-lm --run-name $(RUN)
	bash eval/run_lm_eval.sh $(MODEL) $(RUN)

# --- serve ------------------------------------------------------------------
serve:
	bash serve/quantize.sh $(MODEL) checkpoints/awq
	bash serve/serve_vllm.sh checkpoints/awq

bench:
	$(PY) serve/bench_latency.py --model struct-lm --concurrency 1 8 32
