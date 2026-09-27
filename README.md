# struct-lm

> Independent project, using only public documents, open weights and open-source tools. Forge is
> described from Mistral AI's public announcement.

Domain adaptation of an open base LLM
([`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512))
to structural and civil engineering through **CPT → SFT → DPO → GRPO**, with every stage
measured on the same KPI tasks and general-capability regression suite, then quantized
(AWQ) and served with vLLM.

> This README is the write-up. Numbers live in [`results/table.md`](results/table.md);
> the reasoning behind every choice lives in [`notes/decisions.md`](notes/decisions.md).

## The problem

Engineering organisations sit on decades of internal documents: design manuals, inspection
procedures, worked calculation examples, test reports. A general-purpose LLM has seen little of
it. Asked about those documents, it misses the specific values, clauses and vocabulary, invents
plausible answers, and can't point to the page it relied on. Retrieval helps with lookup, but it
doesn't teach the model the domain's language, and it doesn't change what the model does when the
answer isn't in front of it.

Mistral AI's Forge platform, as publicly announced, addresses this by training open-weight models
further on a client's own data across the whole lifecycle: continued pre-training on raw
proprietary text, synthetic data, SFT and preference optimisation, reinforcement learning, and
LoRA where lighter adaptation is enough, all measured against evals tied to the client's KPIs.
The default path starts from an existing checkpoint, not from scratch.

**This repo runs that lifecycle once, end to end, at roughly 1% scale.** Public-domain US federal
structural-engineering documents stand in for a client's private corpus: 242 manuals, reports and
design examples from USACE, FEMA, FHWA, NIST and NASA (20.0M Tekken tokens after cleaning; the KPI
eval tasks come from a 52-document seed subset, 19,337 page-anchored chunks).
US federal works carry no licensing risk; copyrighted standards such as ASCE 7 and the AISC manual
are deliberately excluded. The starting checkpoint is Ministral 3 8B Base, from the same model
generation such engagements start from.

The two kinds of training do different jobs:

- **Continued pre-training (CPT)** on raw domain text (next-token loss) teaches knowledge and
  vocabulary. Its risk is forgetting general capability, which is managed by replaying general
  text and keeping the learning rate low.
- **Fine-tuning (SFT → DPO → GRPO)** on a few thousand curated examples teaches behaviour: answer
  from the passages given, cite them, decline when the answer isn't there, and get verifiable
  calculations right.

The goal is not a great model. It is a clean experiment with honest numbers and decisions that can
be defended, measured before and after every stage.

### How success is measured

The eval harness was built before any training, so it couldn't be fitted to a model. The same six
measurements run after every stage, with the base model and the off-the-shelf instruct model as
the first two rows.

| # | What it measures | Task | Metrics |
|---|---|---|---|
| 1 | Closed-book domain knowledge | 140 questions with exact or numeric answers from the corpus | `qa_acc` |
| 2 | Answering from given passages, with citations | 131 questions, 4 passages each (gold plus distractors) | `grounded_acc`, `cite_valid`, `cite_supported` |
| 3 | Domain vocabulary | 303 terms to define in one sentence | `vocab_recall` |
| 4 | Declining when the answer isn't there | 87 questions whose 3 passages don't contain the answer | `halluc_rate` (lower is better) |
| 5 | General capability, to catch forgetting | MMLU, GSM8K, HellaSwag, 5-shot | `mmlu`, `gsm8k`, `hellaswag` |
| 6 | Serving cost | vLLM at 1, 8 and 32 concurrent requests | time to first token, inter-token latency, throughput |

Every task item was hand-reviewed. Tasks 2–4 are graded by a pinned judge (Mistral Large 3,
temperature 0, cached verdicts), with anything a rule can decide (missing citations, empty answers,
the exact refusal phrase) decided by rule first.

### Where it starts

Row zero, from [`results/table.md`](results/table.md) (27 September 2026):

- **The base model** finds the right answer in the passages 76% of the time but cites correctly only
  6% of the time, and answers 92% of unanswerable questions with something invented.
- **The instruct model** has the behaviour (69% of answers correct and backed by their citations,
  99% of unanswerable questions declined), but it knows no more of the domain closed-book: 12%
  against the base model's 17%.
- **Closed-book domain accuracy is low for both.** That is the knowledge gap CPT is meant to close,
  while SFT and DPO bring citation and refusal behaviour up to the instruct model's level or beyond,
  and the general-capability columns stay flat.

### Scope

This is the same pipeline shape as a production engagement at roughly 1% scale. It leaves out the
two parts that make the real thing hard: distributed full-parameter training on billions of
tokens, and enterprise data readiness and governance.

| | This repo | Production engagement |
|---|---|---|
| Base model | Ministral 3 8B, open weights | tens to hundreds of billions of parameters, dense or MoE |
| Corpus | 20M tokens of cleaned public PDFs | billions of tokens: documents, code, databases, images; messy |
| Continued pre-training | LoRA on one GPU, hours | full-parameter, multi-node, days to weeks, replay mix, annealing |
| Post-training | a few thousand SFT examples, ~1k DPO pairs, a small GRPO run | 10k–100k+ examples reviewed with domain experts, RL with distillation |
| Evaluation | six-measurement harness plus regression suite | the same idea, built with domain experts, with audit lineage |
| Infrastructure | rented GPUs (Modal), open-source stack | isolated environments, data residency, versioned datasets and runs |

What transfers: the stages and their order, the failure modes, the eval design, and the recurring
decisions (how much general text to replay, LoRA versus full-parameter training, how far to trust a
judge). What doesn't: the engineering difficulty at scale.

## Pipeline

```
sources.csv ─ download ─ extract ─ filter ─ dedup ─ pii ─ split ──► CPT ─merge─► SFT ─merge─► DPO ─merge─► GRPO ─merge─► AWQ ─► vLLM
                            │                               │  replay (FineWeb-Edu)                                              │
                            │                               └─ tokenizer_coverage, stats    run_eval + run_lm_eval + bench  ◄────┘
                            └─ extract --chunks ─ make_tasks (eval/tasks/, frozen eval docs only)
```

| Stage | Script | Input data | Signal |
|-------|--------|-----------|--------|
| CPT  | `train/cpt.py`  | `data/processed/{train,val}.jsonl` (+ `replay.jsonl` for the replay ablation) | next-token on domain text |
| SFT  | `train/sft.py`  | `data/sft/{train,val}.jsonl` `{"messages": [...]}` | assistant-only loss |
| DPO  | `train/dpo.py`  | `data/dpo/{train,val}.jsonl` `{"prompt","chosen","rejected"}` | preference pairs |
| GRPO | `train/grpo.py` | `data/grpo/train.jsonl` `{"prompt","answer"}` | verifiable reward functions |

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
its own env: `uv sync --extra quantize`. Modal needs two secrets: `huggingface` (`HF_TOKEN`) and
`mistral` (`MISTRAL_API_KEY`).

### Data and eval tasks

```bash
make data                                   # download → extract → filter → dedup → pii → split → replay → tokenizer_coverage → stats
python eval/make_tasks.py                   # eval/tasks/*.jsonl + eval_chunk_ids.txt (Mistral Large 3)
```

USACE, ROSA P and FEMA block scripted downloads: save those PDFs by hand as
`data/raw/<slug>.pdf`, then re-run `download.py`, which records each file's sha256 and page
count in `data/sources.csv`. `chunks.jsonl` (the eval's input, `extract.py --chunks`) is built only
from the eval documents, so adding sources never moves chunk ids; never build SFT data from
chunks in `eval/tasks/eval_chunk_ids.txt`.

### Train

```bash
make cpt  && python train/merge.py --adapter checkpoints/cpt  --out checkpoints/cpt-merged
make sft  && python train/merge.py --adapter checkpoints/sft  --out checkpoints/sft-merged
make dpo  && python train/merge.py --adapter checkpoints/dpo  --out checkpoints/dpo-merged
make grpo && python train/merge.py --adapter checkpoints/grpo --out checkpoints/grpo-merged
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
| `domain_qa` | closed-book, 3-shot | exact match / numeric ±2% | `qa_acc` |
| `vocab` | closed-book, 3-shot definitions | judge vs reference definition | `vocab_recall` |
| `grounded` | 4 passages (gold, neighbours, off-doc distractor) | citations are provided ids; judge checks support | `cite_valid`, `cite_supported` |
| `adversarial` | 3 related passages without the answer | exact abstain phrase, else judge | `halluc_rate` |

### Quantize, serve, benchmark

```bash
uv sync --extra quantize && bash serve/quantize.sh checkpoints/grpo-merged checkpoints/awq
uv sync --extra serve && bash serve/serve_vllm.sh checkpoints/awq &
python serve/bench_latency.py --label awq   # → results/bench/awq.json
```

On Modal, `--which latency` starts `serve_vllm.sh` in the container, waits for it, and runs the
benchmark (64 KPI prompts, seeded sample across all four tasks; concurrency 1 / 8 / 32):

```bash
modal run --detach eval/modal_app.py --which latency --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b
# → results/bench/base-8b.json on the volume
```

Override any config value from the CLI:
`python train/sft.py --config train/configs/sft.yaml --override training.learning_rate=1e-4`

## Layout

```
data/     sources.csv (every document: slug, publisher, title, url, sha256, pages), raw/ (PDFs,
          gitignored), scripts/ (one per step), processed/stats.json (every corpus count,
          committed); interim/ and the rest of processed/ gitignored
eval/     make_tasks.py (task generation), tasks/*.jsonl (KPI tasks + rejects.jsonl hand-review
          list + eval_chunk_ids.txt), prompts.py, scorers.py, judge.py, run_eval.py,
          run_lm_eval.sh, modal_app.py (Modal H100 runner)
train/    one script per stage + common.py, merge.py, configs/*.yaml
serve/    quantize.sh (AWQ), serve_vllm.sh, bench_latency.py
results/  all committed: table.md (one row per run), runs/<run>/{generations,scored}.jsonl +
          metrics.json, judge_cache.jsonl, lm_eval/<run>/**/results_*.json, bench/, curves/.
          Anyone can re-score with `run_eval.py --rescore` without a GPU or API key.
notes/    decisions.md: dated log of every choice
checkpoints/  adapters and merged weights (gitignored)
```

## Write-up

### 1. Goal and KPIs
See [The problem](#the-problem) and [How success is measured](#how-success-is-measured) above.

### 2. Data
The corpus card in [`notes/decisions.md`](notes/decisions.md) ("Stage 1 corpus card") has every
number, generated from [`data/processed/stats.json`](data/processed/stats.json) by `stats.py`:
documents and pages per publisher, pages dropped, paragraphs dropped per filter rule, exact and
near-duplicate removal, PII replacements, tokenizer fertility, and the train/val split.

#### Corpus sources

All US federal works (public domain, 17 USC 105); documents carrying a third-party copyright notice
are excluded (`extract.py` flags them for a hand check). The full list is
[`data/sources.csv`](data/sources.csv): slug, publisher, title, URL, and the sha256 and page count
of the exact file processed. `download.py` reads it; each file lands at `data/raw/<slug>.pdf`.
USACE, ROSA P and FEMA block scripted downloads (Akamai bot filter): get those in a browser, save
them under their slug or URL filename, and re-run `download.py` to hash them.

### 3. Training
_Per stage: data size, key hyperparameters, curves (`results/curves/`), what changed._

### 4. Results
_`results/table.md`: KPI gain per stage vs regression-suite cost. Which stages earned
their keep, and which didn't._

### 5. Serving
_AWQ quality delta vs bf16; TTFT / ITL / throughput at 1 and 32 concurrent requests._

### 6. What I'd do next
_Honest limitations: judge bias, eval set size, reward hacking observed in GRPO, etc._
