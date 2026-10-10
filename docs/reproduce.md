# Reproduce

## At a glance

**Check the numbers, no GPU and no API key** (about two minutes on a laptop):

```bash
uv sync --extra data --extra dev
make reproduce-score   # every row of results/table.md rescored from its committed generations
                       # and compared (eval/rescore_all.py), then the strict checker, pass@k and
                       # every table and figure regenerated; ends with `git diff --exit-code`
make audit             # each number in the README's prose against a generated table or a named file
```

The judge's verdicts are in `results/judge_cache.jsonl`, so a rescore makes no API call; a miss
would need `MISTRAL_API_KEY`, and `rescore_all.py` fails if the judge is called at all.

**Rebuild a stage** (`make reproduce-stage0` … `reproduce-stage6`, Modal GPU jobs, run in order
with `.env` loaded): each target puts the committed data set on the volume and launches the chain's
runs with the commands below. Pull, score and report as in [Evaluate](#evaluate-repeat-per-stage-including-the-base-and-instruct-baselines).
Ablations and probes beyond the chain (LR-up, the 2-GPU LoRA run, the memorisation probe) are in
`.claude/skills/stage2-cpt/SKILL.md` and `CLAUDE.md`.

**Data sets** (sha256 of each `SHA256SUMS`, which lists every file's own hash):

| set | `SHA256SUMS` hash | files |
|---|---|---|
| CPT corpus, `data/processed` | 966d1e0c | train, val, replay, general_val (gitignored; rebuilt by `make data`; the hashes match the Modal volume's copies, checked 2026-10-10) |
| SFT, `data/sft` | 70f47740 | train (2,436 records), sft_val (80); git keeps `data/sft/hosted/` (the Tülu 3 replay text blanked, row ids kept), and `make sft-replay` rebuilds both files and checks them |
| DPO as run, `data/dpo` | a899f7d2 | train, val (506 pairs) |
| DPO strict, `data/dpo/strict` | 5e3effaf | train (445 pairs), val (18) |
| GRPO, `data/grpo` | 4db8f7a6 | tasks, train (622), val (50) |
| Stage 6 bench sets | `serve/bench_manifest.json` | unique, grounded_unique, grounded_rag, closedbook |

The 246 source PDFs are re-fetched from `data/sources.csv` (URL and sha256 per document).

**Pinned versions, per Modal image:**

| image | used for | pins |
|---|---|---|
| `train/modal_train.py` `image` | CPT, SFT, DPO, merge | torch 2.13.0, transformers 5.16.1, TRL 0.29.1, PEFT 0.21.0, accelerate 1.15.0, mistral-common 1.12.0 |
| `train/modal_train.py` `grpo_image` | GRPO (colocated rollouts) | the above plus vLLM 0.30.0 |
| `eval/modal_app.py` `image` | KPI generations, lm-eval, latency, the Stage 6 gate and bench | vLLM 0.29.0 (recorded with torch 2.13.0+cu130), lm-eval ≥ 0.4.13 (0.4.13 recorded); transformers < 5.17 and mistral-common ≥ 1.8.6 are ranges, not pins |
| `serve/modal_serve.py` `quant_image` | FP8 and INT4 quantization | llmcompressor 0.14.0 (writes compressed-tensors 0.19.0; vLLM 0.29 reads it with 0.17.0) |

- **Every merged config carries `"apply_yarn_scaling": false`** (`train/merge.py`). Without it,
  vLLM 0.29 applies YaRN attention scaling the model doesn't use: perplexity 7.23 against 6.89
  (`notes/contributions.md`).
- **Every eval loads with** `tokenizer_mode=mistral`, `config_format=hf` and
  `limit_mm_per_prompt={"image": 0}`.
- **The Stage 6 bench runs on `gpu="H100!"`** and aborts on any other device; each result records
  the GPU, driver, CUDA, vLLM and torch.

**Where things live:**

| what | where |
|---|---|
| code, eval tasks, SFT / DPO / GRPO data, every generation, lm-eval output, judge verdict, perplexity, bench and training log | git (this repo) |
| the CPT corpus files, LoRA adapters (`checkpoints/_train/<run>`), merged and quantized checkpoints | the Modal volume `struct-lm` (merged checkpoints rebuild from the adapters with `--steps merge`) |
| the base and instruct models | Hugging Face (`mistralai/Ministral-3-8B-Base-2512`, `-Instruct-2512-BF16`) |

**The Tülu 3 replay records** (500: 475 train, 25 val; git holds their row ids only) by subset,
with the licence the mixture's card gives each:

| subset | records | licence |
|---|---|---|
| Evol CodeAlpaca | 97 | Apache 2.0 |
| FLAN v2 | 74 | not given on the card |
| NuminaMath-TIR | 58 | Apache 2.0 |
| withdrawn math set (GSM8K, 50k) | 46 | withdrawn after training; not in the card's list |
| WildJailbreak | 46 | ODC-BY-1.0 |
| WildGuardMix | 46 | Apache 2.0 |
| Persona GSM | 46 | ODC-BY-1.0 |
| WildChat (GPT-4) | 36 | ODC-BY-1.0 |
| Persona Algebra | 18 | ODC-BY-1.0 |
| CoCoNot | 10 | ODC-BY-1.0 |
| No Robots | 8 | CC-BY-NC-4.0 (non-commercial) |
| SciRIFF | 7 | ODC-BY-1.0 |
| TableGPT | 4 | MIT |
| OASST | 4 | Apache 2.0 |

The mixture as a whole is ODC-BY-1.0, and its card notes that "different licenses apply to subsets
of the data" and that some outputs come "from third party models that are subject to separate terms".
The three Claude-written Persona subsets are excluded (`data/scripts/sft_replay.py`, `EXCLUDED`).
The diversity probe's 50 held-out prompts and the replay template prompt come from the same
mixture and are handled the same way (`eval/hosted/`, `eval/hosted/SHA256SUMS`).

## Pipeline

![The lifecycle](diagrams/lifecycle.svg)

Each stage's own diagram opens its doc ([`docs/stage0.md`](stage0.md) to [`stage6.md`](stage6.md)).
The corpus steps are in [`stage1.md`](stage1.md).

| Stage | Script | Input data | Signal |
|-------|--------|-----------|--------|
| CPT  | `train/cpt.py`  | `data/processed/{train,val}.jsonl` (+ `replay.jsonl` for the replay ablation) | next-token on domain text |
| SFT  | `train/sft.py`  | `data/sft/{train,sft_val}.jsonl` `{"prompt": [...], "completion": [...]}` | completion-only loss |
| DPO  | `train/dpo.py`  | `data/dpo/{train,val}.jsonl` `{"prompt","chosen","rejected"}` + token ids | verifiable preference pairs (sigmoid DPO) |
| GRPO | `train/grpo.py` | `data/grpo/{train,val}.jsonl` `{"prompt", "verifier", ...}` (Stage 4 pool prompts in the probe's window) | rule-based reward: format gate, strict correctness, length (`train/grpo_rewards.py`) |

**Who wrote the training data.** The SFT questions and completions were written by Mistral Large 3
(`mistral-large-2512`) and Mistral Medium 3.5 (`mistral-medium-2604`, 25% of completions), apart from
the abstain records' fixed refusal sentence, and the 500 general replay records come from the Tülu 3
SFT mixture without its Claude-written subsets. Claude, through Claude Code, built the tooling and
reviewed the generated records against their source passages with keep/drop verdicts only: no
training record contains text Claude wrote. The DPO pairs' chosen and rejected answers are
sft-from-cpt's own samples, labelled by verifiers and rules (no judge, no Claude verdicts);
`dpo-strict`'s are the same samples, relabelled by the strict checker. The GRPO completions are the
policy's own rollouts, scored by rules alone; its prompts are the Stage 4 pool's. The hand-written
completions in `tests/test_grpo_rewards.py` and `tests/test_qa_strict.py` are test fixtures and are
never trained on.

Each stage trains a LoRA adapter; `train/merge.py` folds it into the weights, and the next
stage's `model.init_from` points at the merged directory.

## Quickstart

### Setup

```bash
uv sync --extra data --extra dev            # Mac: data prep, task generation, scoring
uv sync --extra data --extra train --extra eval --extra serve   # Linux GPU box
pre-commit install                          # ruff, uv-lock, shellcheck, file checks
export MISTRAL_API_KEY=...                  # task generation + judge (scripts don't read .env)
```

`quantize` (llm-compressor) conflicts with `serve`/`eval` (compressed-tensors pins), so it gets
its own env: `uv sync --extra quantize` (Stage 6 runs it in its own Modal image). Modal needs two
secrets: `huggingface` (`HF_TOKEN`) and `mistral` (`MISTRAL_API_KEY`).

### Data and eval tasks

```bash
set -a; . ./.env; set +a                    # HF_TOKEN (tokenizers, FineWeb-Edu), MISTRAL_API_KEY
make data                                   # download → extract → filter → dedup → pii → split → replay → tokenizer_coverage → stats
python data/scripts/extract.py --chunks     # eval input: chunks.jsonl from the pool in eval/tasks/eval_docs.txt (frozen)
python eval/make_tasks.py                   # eval/tasks/*.jsonl + eval_chunk_ids.txt (Mistral Large 3)
```

`make data` writes the Stage 1 corpus to `data/processed/` on the Mac (CPU and network only).
Stage 2 uploads `train.jsonl`, `val.jsonl` and `replay.jsonl` to the Modal volume.
USACE, fema.gov and ROSA P block scripted downloads (Akamai 403). Those PDFs were fetched in a
browser (Chrome DevTools MCP): save them as `data/raw/<slug>.pdf` (or the URL's filename), then
re-run `download.py`, which records each file's sha256 and page count in `data/sources.csv`
(`stats.py` adds each document's final token count).
`chunks.jsonl` is built only from the documents pinned in `eval/tasks/eval_docs.txt`, so adding
sources never resamples the eval; never build SFT data from chunks in `eval/tasks/eval_chunk_ids.txt`,
except the seen half listed in `eval/tasks/sft_seen_chunks.txt` (rule 10, `data/scripts/sft_guard.py`).

### Train

Training runs on Modal (`train/modal_train.py`): one command trains, merges the adapter into the
base (fp32, with the vLLM YaRN fix in the config), and evaluates the merged checkpoint.

```bash
M=.venv/bin/modal
modal volume put --force struct-lm data/processed data/processed   # after any data change
# Stage 2, CPT: train -> merge -> perplexity + lm-eval + KPI generations (latency is opt-in: ,latency)
$M run --detach train/modal_train.py --config train/configs/cpt.yaml --run-name cpt-8b
$M run --detach train/modal_train.py --config train/configs/cpt_replay10.yaml --run-name cpt-8b-replay10
$M run --detach train/modal_train.py --config train/configs/cpt_8b_full.yaml --run-name cpt-8b-full --gpus 2
# the base, re-saved through merge.py so vLLM evaluates it exactly like a fine-tuned checkpoint
$M run --detach train/modal_train.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b-hf --steps merge,eval
# then pull and score locally (next section), and refresh the tables and figures:
.venv/bin/python train/report.py   # results/train_runs.md, results/curves/*.png, README blocks
```

SFT is wired (a config with `stage: sft`; `train/sft.py`, pre-tokenised by `train/sft_data.py`).
DPO is wired (`stage: dpo`, `train/dpo.py`); GRPO (`grpo.py`) is not yet.

```bash
# Stage 3: SFT (the set must match data/sft/SHA256SUMS on the volume), B4 and the merge gate run in
# the chain; chat checkpoints are evaluated with --chat
for f in train.jsonl sft_val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/sft/$f data/sft/$f; done
$M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-cpt \
  --merge-from b4 --chat --steps train,merge,mergecheck,ppl,eval,latency,sample
# Stage 4: DPO on verifiable preferences. Pairs from the SFT model's own samples (make dpo-data,
# then the probe/benchmark/pool sampling jobs, then make dpo-pairs: CLAUDE.md, Stage 4); the set
# must match data/dpo/SHA256SUMS on the volume; the checkpoint rule (--merge-from rule) and the
# merge gate (on sft_val) run in the chain
for f in train.jsonl val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/dpo/$f data/dpo/$f; done
$M run --detach train/modal_train.py --config train/configs/dpo.yaml --run-name dpo \
  --merge-from rule --chat --steps noop,train,merge,mergecheck,ppl,eval,latency,sample
.venv/bin/python eval/winrate.py dpo sft-from-cpt   # reported only: the judge failed its benchmark
```

### Evaluate (repeat per stage, including the base and instruct baselines)

Generation runs on a Modal H100; scoring (rules + Mistral judge) runs locally, off the GPU clock.

```bash
# 1. Modal: lm-eval (MMLU / GSM8K / HellaSwag, 5-shot) + KPI generations
modal run --detach eval/modal_app.py --model mistralai/Ministral-3-8B-Base-2512 \
  --run-name base-8b --generate-only
modal run --detach eval/modal_app.py --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --run-name instruct-8b --chat --generate-only

# 2. Pull the run's outputs off the Modal volume (not results/table.md: it's written locally)
modal volume get struct-lm results/runs/base-8b results/runs/
modal volume get struct-lm results/lm_eval/base-8b results/lm_eval/

# 3. Score locally; appends a row to results/table.md
python eval/run_eval.py --run-name base-8b --rescore --lm-eval-dir results/lm_eval \
  --model mistralai/Ministral-3-8B-Base-2512
```

`--chat` wraps the KPI prompts in the checkpoint's chat template; it is required for Instruct and
later chat checkpoints (SFT/DPO/GRPO), and base models run without it. lm-eval never uses a chat
template, for any checkpoint, so its columns compare across every row (see notes/decisions.md).
`--which lm|kpi` runs one half; `--limit 5 --no-judge --which kpi` is a smoke test.
On a GPU box without Modal, drop `--generate-only` and run `eval/run_eval.py` and
`eval/run_lm_eval.sh` directly (usage at the top of each file).

What each KPI task measures:

| Task | Context | Scoring | Metric |
|------|---------|---------|--------|
| `domain_qa` | closed-book, 3-shot | exact match / numeric ±2% (lenient); the whole gold, one candidate, units compared (strict, 2026-10-09) | `qa_acc`; strict via `eval/qa_strict.py` |
| `vocab` | closed-book, 3-shot definitions | judge vs reference definition | `vocab_recall` |
| `grounded` | 4 passages (gold, neighbours, off-doc distractor) | citations are provided ids; judge checks support | `cite_valid`, `cite_supported` |
| `adversarial` | 3 related passages without the answer | exact abstain phrase, else judge | `halluc_rate` |

### Quantize, gate, serve, benchmark (Stage 6)

```bash
python serve/bench_data.py      # the bench request sets from the eval prompts (pinned: serve/bench_manifest.json)
modal run serve/modal_serve.py --action quantize --scheme fp8          # -> checkpoints/dpo-strict-fp8
modal run --detach serve/modal_serve.py --action quantize --scheme w4a16   # GPTQ -> checkpoints/dpo-strict-w4a16
modal run --detach serve/modal_serve.py --action gate --variant fp8    # KPI, eos, perplexity, GSM8K
modal run --detach serve/modal_serve.py --action bench --variants bf16,fp8,fp8kv,w4a16 --spec   # one H100
bash serve/serve_vllm.sh checkpoints/dpo-strict-fp8   # serving it yourself: DEPLOY.md
```

The full sequence (pulls, scoring, report) is in `CLAUDE.md`, Stage 6. The earlier per-checkpoint
latency step (`--which latency`, `serve/bench_latency.py`) still exists; its rows are the
"GPU not recorded" table in [Serving](stage6.md).

Override any config value from the CLI:
`python train/sft.py --config train/configs/sft.yaml --override training.learning_rate=1e-4`

## Layout

```
data/     sources.csv (every document: slug, publisher, title, url, sha256, pages, tokens),
          raw/ (PDFs, gitignored), scripts/ (one per step + common.py), processed/ (stats.json
          and tokenizer_coverage.md committed; docs_raw, docs, train/val/replay, dropped_samples,
          duplicates and the eval's chunks.jsonl gitignored)
eval/     make_tasks.py (task generation), tasks/*.jsonl (KPI tasks + rejects.jsonl hand-review
          list + eval_chunk_ids.txt), prompts.py, scorers.py, judge.py, run_eval.py,
          run_lm_eval.sh, modal_app.py (Modal H100 runner)
train/    one script per stage + common.py, merge.py, configs/*.yaml
serve/    serve_vllm.sh (the served config), quantize.py (FP8 / GPTQ W4A16), modal_serve.py
          (quantize, quality gate, H100 bench), bench_data.py + bench_manifest.json (request sets),
          served_check.py (template ids, served-vs-eval smoke), bench_latency.py (Stages 2-5)
results/  all committed: table.md (one row per run), runs/<run>/{generations,scored}.jsonl +
          metrics.json, judge_cache.jsonl, lm_eval/<run>/**/results_*.json, bench/, curves/.
          Anyone can re-score with `run_eval.py --rescore` without a GPU or API key.
notes/    decisions.md: dated log of every choice
checkpoints/  adapters and merged weights (gitignored)
```
