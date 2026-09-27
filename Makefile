# End-to-end pipeline. Each target is resumable; outputs land in data/, checkpoints/, results/.
PY ?= .venv/bin/python

.PHONY: data train eval serve all

# --- data -------------------------------------------------------------------
# Stage 1 corpus: data/sources.csv -> data/processed/{docs_raw,docs,train,val,replay}.jsonl + stats.json.
# Each step reads the previous step's output; run from the repo root with .env loaded (HF_TOKEN).
# New sources: data/scripts/crawl_index.py <index> --pattern ... --publisher ..., then `make data`.
DATA_STEPS = download extract filter dedup pii split replay tokenizer_coverage stats
data:
	for s in $(DATA_STEPS); do $(PY) data/scripts/$$s.py || exit 1; done

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
