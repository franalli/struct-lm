# End-to-end pipeline. Each target is resumable; outputs land in data/, checkpoints/, results/.
PY ?= python
BASE ?= mistralai/Ministral-3-8B-Base-2512
SEQ_LEN ?= 4096

.PHONY: data train eval serve all

# --- data -------------------------------------------------------------------
data: data/packed/cpt

data/raw/manifest.json: data/scripts/sources.csv
	$(PY) data/scripts/download.py --sources $< --out data/raw

data/processed/chunks.jsonl: data/raw/manifest.json
	$(PY) data/scripts/extract.py --raw data/raw --out $@

data/interim/filtered.jsonl: data/processed/chunks.jsonl
	$(PY) data/scripts/filter.py --in $< --out $@ --lang en

data/processed/dedup.jsonl: data/interim/filtered.jsonl
	$(PY) data/scripts/dedup.py --in $< --out $@ --threshold 0.8

coverage: data/processed/dedup.jsonl
	$(PY) data/scripts/tokenizer_coverage.py --in $< --tokenizers $(BASE)

data/packed/cpt: data/processed/dedup.jsonl
	$(PY) data/scripts/pack.py --in $< --out $@ --tokenizer $(BASE) --seq-len $(SEQ_LEN)

# --- train ------------------------------------------------------------------
cpt:  ; $(PY) train/cpt.py  --config train/configs/cpt.yaml
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
