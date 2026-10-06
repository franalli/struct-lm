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
structural-engineering documents stand in for a client's private corpus: 246 manuals, reports and
design examples from USACE, FEMA, FHWA, NIST and NASA (20.6M Tekken tokens after cleaning; the KPI
eval tasks are sampled from the 234 training documents, at most 6 source passages per document per
task, and draw on 152 of them).
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
| 1 | Closed-book domain knowledge | 325 questions with exact or numeric answers from the corpus (130 until 2026-10-04) | `qa_acc`, by answer kind (`qa_num` / `qa_ident` / `qa_term`) and by SFT half (`qa_seen` / `qa_unseen`), plus the gold answer's log-probability (`gold_lp`) |
| 2 | Answering from given passages, with citations | 108 questions, 4 passages each (gold plus distractors) | `grounded_acc`, `cite_valid`, `cite_supported` |
| 3 | Domain vocabulary | 210 terms to define in one sentence | `vocab_recall` |
| 4 | Declining when the answer isn't there | 76 questions whose 3 passages don't contain the answer | `halluc_rate` (lower is better) |
| 5 | General capability, to catch forgetting | MMLU, GSM8K, HellaSwag, 5-shot | `mmlu`, `gsm8k`, `hellaswag` |
| 6 | Serving cost | vLLM at 1, 8 and 32 concurrent requests | time to first token, inter-token latency, throughput |

Every task item was reviewed against its source passages before any model was run on it: 524 of
1,176 generated items survived the first review (2026-09-27), and 219 of 1,174 the second (195 in
the set after the identifier cap), which grew the closed-book task to cut its noise (2026-10-04,
[`notes/eval_review_rubric.md`](notes/eval_review_rubric.md)). Layout locators (page, table and
figure numbers) are removed by rule, and identifiers are capped at 20% of the task. Items were rejected only for defects: a wrong or unsupported
gold answer, a correct answer the scorer would mark wrong, or a question that tests general
knowledge or trivia instead of the corpus. None was rejected for being hard. Tasks 2–4 are graded
by a pinned judge (Mistral Large 3,
temperature 0, cached verdicts), with anything a rule can decide (missing citations, empty answers,
the exact refusal phrase) decided by rule first.

### Where it starts

Row zero, from [`results/table.md`](results/table.md) (eval v2, 4 October 2026; the original
130-item scores are in [`results/table_v1.md`](results/table_v1.md)):

- **The base model** finds the right answer in the passages 85% of the time but cites correctly only
  10% of the time, and answers 88% of unanswerable questions with something invented.
- **The instruct model** has the behaviour (79% of answers correct and backed by their citations,
  99% of unanswerable questions declined), but it knows no more of the domain closed-book: 9.9%
  against the base model's 12.0% on the 325 questions (12% against 14% on the original 130). For
  scale, Mistral Large 3 answers 28% of them closed-book.
- **Closed-book domain accuracy is low for both.** That is the knowledge gap CPT is meant to close,
  while SFT and DPO bring citation and refusal behaviour up to the instruct model's level or beyond,
  and the general-capability columns stay flat.

**Why the instruct model is the bar.** `Ministral-3-8B-Instruct-2512` (the BF16 HF checkpoint) is
Mistral's own instruct post-trained version of the base trained on here (model card).
- **Same model:** the same 8B dense weights and architecture, the same Tekken tokenizer and the same
  vision tower. Every difference between it and the SFT runs comes from post-training data and
  method, not model size.
- **The business claim:** it is what a client would deploy off the shelf. So beating it is "an 8B
  tuned on your corpus beats the stock 8B on your questions".

**How it is scored: without its default system prompt.** Its HF chat template
(`chat_template.jinja`) inserts a default system prompt ("You are Ministral-3-8B-Instruct-2512, a
Large Language Model (LLM) created by Mistral AI...") when none is given. The eval doesn't use that
template.
- **The KPI eval** renders every chat model through the same `--chat` path: vLLM with
  `tokenizer_mode=mistral`, that is mistral-common's `encode_chat_completion` of one user turn.
  It renders `<s>[INST] prompt [/INST]` with no system prompt.
- **The serving benchmark** goes through the same rendering, and lm-eval uses no chat template at
  all (rule 2).
- **So the comparison is on the template the eval renders, the one the SFT models were trained in,
  not the template Mistral ships for the instruct model.** Its default system prompt might change
  its refusal and citation behaviour; that was not measured here.

### Scope

This is the same pipeline shape as a production engagement at roughly 1% scale. It leaves out the
two parts that make the real thing hard: distributed full-parameter training on billions of
tokens, and enterprise data readiness and governance.

| | This repo | Production engagement |
|---|---|---|
| Base model | Ministral 3 8B, open weights | tens to hundreds of billions of parameters, dense or MoE |
| Corpus | 20M tokens of cleaned public PDFs | billions of tokens: documents, code, databases, images; messy |
| Continued pre-training | LoRA on one GPU, hours | full-parameter, multi-node, days to weeks, replay mix, annealing |
| CPT data | documents concatenated and read once, 20M tokens | the same next-token objective with data engineering around it: rewrites of the facts that matter, per-source mixture and repetition, section-level boundaries |
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
| SFT  | `train/sft.py`  | `data/sft/{train,sft_val}.jsonl` `{"prompt": [...], "completion": [...]}` | completion-only loss |
| DPO  | `train/dpo.py`  | `data/dpo/{train,val}.jsonl` `{"prompt","chosen","rejected"}` | preference pairs |
| GRPO | `train/grpo.py` | `data/grpo/train.jsonl` `{"prompt","answer"}` | verifiable reward functions |

**Who wrote the training data.** The SFT questions and completions were written by Mistral Large 3
(`mistral-large-2512`) and Mistral Medium 3.5 (`mistral-medium-2604`, 25% of completions), apart from
the abstain records' fixed refusal sentence, and the 500 general replay records come from the Tülu 3
SFT mixture without its Claude-written subsets. Claude, through Claude Code, built the tooling and
reviewed the generated records against their source passages with keep/drop verdicts only: no
training record contains text Claude wrote.

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

SFT, DPO and GRPO (`train/sft.py`, `dpo.py`, `grpo.py`) are not wired into the Modal app yet.

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
data/     sources.csv (every document: slug, publisher, title, url, sha256, pages, tokens),
          raw/ (PDFs, gitignored), scripts/ (one per step + common.py), processed/ (stats.json
          and tokenizer_coverage.md committed; docs_raw, docs, train/val/replay, dropped_samples,
          duplicates and the eval's chunks.jsonl gitignored)
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

#### Tokenizer coverage

Does Tekken (131k vocabulary) fit this text, or does CPT need new tokens? Measured by
`data/scripts/tokenizer_coverage.py` on the clean corpus against ~1M tokens of FineWeb-Edu as
general English. The full report is
[`data/processed/tokenizer_coverage.md`](data/processed/tokenizer_coverage.md).

| text | tokens per word |
|---|---|
| domain corpus (20.6M tokens) | 1.44 |
| FineWeb-Edu | 1.34 |
| top-500 TF-IDF domain terms | 1.12 |
| the 303 terms in the vocab eval | 1.43 |
| 18 designations and units (`ASCE 7-22`, `kip-ft`, `A709 Grade 50W`, ...) | 3.21 |

- **No vocabulary extension.** The corpus costs only 1.08x the tokens per word of general
  English, and the words that mark the domain (shear, girder, flange, diaphragm) are mostly single
  tokens.
- **Designations fragment, not vocabulary.** Only 34 of 843 term words take 4+ tokens:
  - document numbers and digit strings (`EM 1110-2-2104` is 13 tokens, because Tekken gives every
    digit its own token by design);
  - parenthesised acronyms;
  - hyphenated compounds.

  New embeddings trained on 20M tokens would cost more than those save.
- **Base and Instruct encode identically** (a corpus sample and every term word), so data and
  evals tokenised once serve both checkpoints.

#### Replay slice

To limit forgetting, CPT can mix general text back in. `data/processed/replay.jsonl` is 1.9M Tekken
tokens (10% of train) from FineWeb-Edu, English web text filtered for educational quality, taken as
raw text from the head of its `sample-10BT` stream. Stage 2 runs CPT with and without it and
compares domain val perplexity with the MMLU, GSM8K and HellaSwag regression columns. It is ODC-By
web text, not public domain, so it is rebuilt from Hugging Face rather than committed, and nothing
in the eval comes from it. It is not a random sample (caveats in the corpus card). Its overlap with
the regression benchmarks was measured, below: none beyond stock phrases.

#### Contamination checks

`eval/contamination.py` (CPU, about a minute) measures token 13-gram overlap between what CPT trained
on and everything evaluated against it. The method is GPT-3's (appendix C), counted on Tekken tokens
as in Llama 2. Full tables are in [`results/contamination.md`](results/contamination.md).
- **What dedup already covers:** dedup keeps near-duplicate paragraphs off both sides of the
  document-level split, so this looks for sharing below the paragraph.
- **The matcher works on short items:** text cut from a training document and tokenised on its
  own, as a benchmark item is, comes back 99% matched.

| check | result |
|---|---|
| domain val vs train | 1.5% of val's 13-grams occur in train. A train document shares more with the other 233 (median 2.0%). Three val documents are above 5%, up to 20%, each next to a sibling in train (FHWA HIF-18-044 and -043, FEMA P-1100-2A and -2B, HIF-17-020 and -019). Without them the val gain is -2.33%, unchanged. |
| 2026 reports vs train | at most 1.35% per report, so none is a revision of a corpus document |
| MMLU / GSM8K / HellaSwag vs train and replay | GSM8K and HellaSwag: no 13-gram in either. MMLU: 21 of 14,042 items share one, all stock phrases ("in the 1960s and 1970s"), none half covered. |
| general val vs replay (both FineWeb-Edu) | 0.017% of tokens; disjoint |
| domain_qa few-shot vs the 325 scored items | no shot is a scored question, and the shots share no 13-gram with each other. Answer values recur only by coincidence ("4" in three unrelated items). Two shots come from the same passage as a scored item, qa-0003 and qa-0056, and ask a different fact. |

- **QA items from training documents are by design, not a leak.** The closed-book QA items come
  from training documents on purpose: they measure recall of what CPT read.
- **Why two scored items share a passage with the shots.** `make_tasks.py` takes the shots out of
  the item list, not the passage list. Neither shot gives away the answer:
  - 0.95 d_b in the shot, 75% scored;
  - 4 fiber elements in the shot, L/500 scored.

  The prompt is identical for every checkpoint, so no delta moves, and the frozen set keeps them.
  - **Tagged:** both carry `fewshot_passage_overlap: true` in `domain_qa.jsonl`, for per-item
    analyses to exclude.
  - **Fixed for the next rebuild:** `make_tasks.py --task-version 3` splits the shots off by
    passage. It keeps the v2 ids and only removes items: 322, the two plus one identifier under the
    20% cap.
  - **Guarded:** `tests/test_fewshot.py` fails if a new shared passage appears.

**Where the held-out gain sits.** The val and 2026 perplexity windows, binned by the share of their
tokens inside a 13-gram that also occurs in train (`cpt-8b-replay10` vs `base-8b-hf`; seed 1
agrees):

| tokens shared with train | val windows | val change | 2026 windows | 2026 change |
|---|---|---|---|---|
| under 1% | 157 | -1.70% | 56 | -0.11% |
| 1-5% | 86 | -2.54% | 21 | -0.90% |
| 5-20% | 44 | -3.56% | 5 | -2.00% |
| over 20% | 9 | -5.21% | 0 | |
| all | 296 | -2.33% | 82 | -0.43% |

- **Shared text is learned most, and the gradient is the shape of learning, not of a leak.**
  - The gain rises smoothly with overlap, from -1.70% on windows sharing under 1% of their tokens
    to -5.21% over 20% (Spearman -0.57 over the 296 val windows).
  - A leak would sit in a few copied windows instead.
  - Against that -1.70% floor, verbatim sharing carries about a quarter of the val gain.
  - This is an association: windows with more shared text are also more formulaic (base perplexity
    5.6 against 7.2).
- **It doesn't explain the gap to new documents.** On windows that share almost nothing, val still
  gains -1.70% [-2.25, -1.24] and the 2026 reports -0.11% [-0.46, +0.22]. That is 1.6 of the
  1.9-point gap.
  - The same holds at 8-grams, bin by bin.
  - So what val shares with train lies below copied strings: terms, notation, layout, a series'
    house style.

**Genre is not the explanation either.** The val documents are mostly manuals and the 2026 set is
mostly research reports, so "new documents" could have meant "a different genre". Each document was
labelled from the purpose stated in its front matter (`notes/decisions.md`), then the per-document
perplexity sums were pooled (`eval/ppl_compare.py --docs`; `cpt-8b-replay10`, `cpt-8b` and seed 1
agree within 0.15 points):

| | guidance: manuals, standards, design examples | research reports |
|---|---|---|
| val: the document's series is in train | -2.23% (10 documents) | -3.64% (2) |
| 2026: its series is not | -0.89% (2) | -0.32% (11) |

- **Neither genre explains it.** Val's research reports gain more than its manuals. The two new
  guidance documents (Army bridge load-rating guidance, an FHWA procurement guide) gain less than
  half of what val's manuals gain.
- **What separates the rows is the series.**
  - Every val document's series is in train: USACE EM, FEMA P, NIST GCR 917, FHWA HIF, NASA-STD.
  - None of the 2026 reports' series is: no NIST TN, ERDC or FHWA-HRT document is in train.
  - The two 2026 NIST GCRs are workshop reports from another programme; they gain -0.7% and -0.3%.

  So CPT transfers within a document series, across genre, and barely to a new series.
- **Small cells:** the val reports are two documents, and NIST GCR 12-917-21 is 92% of their tokens;
  HIF-18-047 alone gains -2.54%.

#### Corpus sources

All US federal works (public domain, 17 USC 105); documents carrying a third-party copyright notice
are excluded (`extract.py` flags them for a hand check). The full list is
[`data/sources.csv`](data/sources.csv): slug, publisher, title, URL, and the sha256 and page count
of the exact file processed. `download.py` reads it; each file lands at `data/raw/<slug>.pdf`.
USACE, fema.gov and ROSA P block scripted downloads (Akamai bot filter): get those in a browser,
save them under their slug or URL filename, and re-run `download.py` to hash them. Four FEMA
publications come from other hosts, recorded in the url column:
- P-695 from NIST's NEHRP clearinghouse (FEMA's copy is gone);
- P-751 from WBDG;
- P-58-4 and P-58-5 from ATC, their preparer (fema.gov serves them truncated).

### 3. Training

#### Stage 2: continued pre-training (CPT)

One epoch of plain next-token training on the 19.4M-token train split, starting from
`Ministral-3-8B-Base-2512`: LoRA r=64 (alpha 128) on all seven projections of the language model,
LR 1e-4 with a cosine schedule, 32 windows of 4,096 tokens per optimizer step, so 149 steps (the
"150-step rule"), on one H100. Every number and its reason, the pre-registered success rule and the
ablation rules are in [`notes/decisions.md`](notes/decisions.md); the rough edges hit on the way
(TRL packing that would have kept 5% of the corpus, an EOS token encoded as text, a tokenizer that
broke vLLM's processor) are in [`notes/contributions.md`](notes/contributions.md).

The runs vary one thing each against the main run (`cpt-8b`):

| run | what changes | question |
|---|---|---|
| `cpt-8b-lr2x` | LR 2e-4 | the pre-registered fix when domain perplexity "barely moves" |
| `cpt-8b-seed1` | seed 1 | the run-to-run noise floor every other delta is read against |
| A, `cpt-8b-replay10` | +10% FineWeb-Edu replay | does replay protect general ability, at what domain cost |
| B, `cpt-8b-full` | full-parameter on 2 x H100 (FSDP2) | LoRA vs full fine-tuning on the same model and tokens |
| C, `cpt-8b-fsdp2` | 2 x H100 (FSDP2), 100 steps | throughput scaling and loss agreement |

![Stage 2 loss curves: train loss (10-step moving average) and val loss per run](results/curves/cpt.png)

![Stage 2 perplexity change vs base-8b on the train slice, domain val and general val](results/curves/cpt_ppl.png)

Generated by `train/report.py` from `results/runs/` and `results/ppl/` (regenerated whenever a
run lands):

<!-- stage2-tables:start -->
##### Training runs

| run | GPUs | steps | tokens | tokens/s | tokens/s/GPU | wall (h) | GPU-h | $ | final train loss | final eval loss |
|---|---|---|---|---|---|---|---|---|---|---|
| cpt-8b | 1 | 149 | 19.4M | 5,939 | 5,939 | 0.95 | 0.95 | 3.74 | 1.769 | 1.906 |
| cpt-8b-replay10 | 1 | 164 | 21.4M | 5,980 | 5,980 | 1.04 | 1.03 | 4.09 | 1.770 | 1.906 |
| cpt-8b-full | 2 | 149 | 19.4M | 13,226 | 6,613 | 0.45 | 0.91 | 3.59 | 1.725 | 1.902 |
| cpt-8b-fsdp2 | 2 | 100 | 13.1M | 13,221 | 6,610 | 0.30 | 0.59 | 2.34 | 1.741 |  |
| cpt-8b-lr2x | 1 | 149 | 19.4M | 5,963 | 5,963 | 0.95 | 0.95 | 3.74 | 1.756 | 1.904 |
| cpt-8b-seed1 | 1 | 149 | 19.4M | 5,914 | 5,914 | 0.97 | 0.97 | 3.82 | 1.753 | 1.906 |

- **cpt-8b-fsdp2 vs cpt-8b:** 2 GPUs give 2.23x the tokens/s (13,221 vs 5,939), so per GPU +11% at the same micro-batch and per-layer checkpointing (peak memory 43 vs 51 GB): the FSDP2 code path, not scaling (finding 6 below); per-step losses differ by 0.038% (median) / 0.171% (max) over 100 steps, the 10-step moving averages by 0.01% on average.

$ at 3.95 per GPU-hour (Modal's H100 list price as assumed, not checked against modal.com/pricing); wall time includes tokenising and model load.

##### Perplexity vs base-8b

| run | domain val | change | nats [95% CI, docs] | general val | change | nats [95% CI, docs] | train slice | change |
|---|---|---|---|---|---|---|---|---|
| base-8b | 6.880 |  |  | 8.151 |  |  | 6.185 |  |
| cpt-8b | 6.720 | -2.33% | -0.0236 [-0.0302, -0.0180] | 8.184 | +0.40% | +0.0040 [+0.0025, +0.0056] | 5.673 | -8.28% |
| cpt-8b-replay10 | 6.720 | -2.33% | -0.0236 [-0.0303, -0.0180] | 7.969 | -2.24% | -0.0226 [-0.0257, -0.0198] | 5.646 | -8.71% |
| cpt-8b-full | 6.732 | -2.15% | -0.0217 [-0.0357, -0.0104] | 8.248 | +1.19% | +0.0118 [+0.0091, +0.0143] | 4.815 | -22.15% |
| cpt-8b-lr2x | 6.713 | -2.42% | -0.0245 [-0.0335, -0.0172] | 8.220 | +0.85% | +0.0085 [+0.0068, +0.0102] | 5.442 | -12.01% |
| cpt-8b-seed1 | 6.718 | -2.35% | -0.0238 [-0.0307, -0.0180] | 8.165 | +0.17% | +0.0017 [+0.0002, +0.0030] | 5.687 | -8.05% |
| sft-from-base | 7.008 | +1.86% | +0.0184 [+0.0166, +0.0211] | 8.234 | +1.01% | +0.0100 [+0.0091, +0.0110] | 6.311 | +2.05% |
| sft-from-base-seed1 | 7.010 | +1.89% | +0.0187 [+0.0169, +0.0214] | 8.251 | +1.22% | +0.0121 [+0.0110, +0.0133] | 6.316 | +2.13% |
| sft-from-cpt | 6.872 | -0.11% | -0.0011 [-0.0071, +0.0051] | 8.061 | -1.11% | -0.0111 [-0.0144, -0.0081] | 5.800 | -6.23% |
| sft-from-cpt-seed1 | 6.867 | -0.20% | -0.0020 [-0.0078, +0.0044] | 8.066 | -1.04% | -0.0105 [-0.0138, -0.0074] | 5.789 | -6.39% |

##### Change vs base-8b-hf, next to the noise

| metric | base-8b-hf | cpt-8b | cpt-8b-seed1 | cpt-8b-replay10 | cpt-8b-full | noise |
|---|---|---|---|---|---|---|
| domain val perplexity | 6.88 | -2.33% | -2.35% | -2.33% | -2.15% | 0.02% |
| 2026-report perplexity | 6.21 |  | -0.51% | -0.43% |  |  |
| general val perplexity | 8.15 | +0.40% | +0.17% | -2.24% | +1.19% | 0.23% |
| train slice perplexity | 6.18 | -8.28% | -8.05% | -8.71% | -22.15% | 0.23% |
| MMLU | 0.767 | -0.4 | -0.2 | -0.1 | -0.5 | 0.3 |
| GSM8K | 0.793 | -0.8 | -0.7 | -0.2 | -3.0 | 1.1 |
| HellaSwag | 0.801 | +0.1 | -0.1 | -0.0 | -0.2 | 0.4 |
| closed-book gold-answer log-prob (nats) | -6.77 | +0.59 | +0.58 | +0.51 |  | 0.08 |
| closed-book qa_acc | 0.121 | +2.5 | +0.9 | +0.9 |  | 1.8 |
| grounded_acc (with passages) | 0.843 | -0.9 | -4.6 | -6.5 | +4.6 | 3.7 |
| vocab_recall | 0.705 | +0.5 | -0.5 | -0.5 | +3.8 | 3.1 |
| halluc_rate | 0.895 | +1.3 | +3.9 | +3.9 | +2.6 | 3.5 |

Perplexity in %, the gold-answer log-probability in nats per answer, the rest in points. noise = max(the seed gap cpt-8b vs cpt-8b-seed1, the metric's standard error: for base-8b-hf, or for the log-probability the paired per-item difference): a change smaller than it is not a result. QA rows are on the 322-item domain_qa (eval v3; Stage 2 was first read on v2's 325, results/table_v2.md), so cpt-8b-full, whose weights were deleted, has none.
<!-- stage2-tables:end -->

**Bottom line.** At 20M tokens and one epoch, CPT learns the documents it reads (-8% perplexity) and
almost nothing that transfers to documents it hasn't seen (-0.4% on reports published in 2026).
It does make the facts in the documents it read more likely: closed-book gold answers become about
1.8x more probable (+0.59 nats per answer, both seeds), a gain pass/fail accuracy is too coarse to
resolve at this size.

- **Pushing harder doesn't help.** A higher learning rate and full-parameter training learn the
  documents harder, with 2-4x the forgetting and no held-out gain.
- **What goes forward.** LoRA with 10% replay is the carried-forward checkpoint. A pre-registered
  rule chose it, the rule fired at the edge of the noise, and that is recorded.
- **Why the eval harness matters.** It found a serving bug larger than any training effect.

Every delta below is against `base-8b-hf`, the base evaluated through the same vLLM path as the
fine-tuned checkpoints. Each is read against its noise: the larger of the seed gap (`cpt-8b` vs
`cpt-8b-seed1`) and the metric's standard error.

**Against the pre-registered rules** (`notes/decisions.md`, set before the main run):

| rule | cpt-8b (seed 1) | cpt-8b-lr2x | cpt-8b-full | verdict |
|---|---|---|---|---|
| domain val perplexity down at least 20% | -2.33% (-2.35%) | -2.42% | -2.15% | failed everywhere |
| train/val gap grows under ~10 points over base's 11.2% (over ~25 = memorising) | +7.2 (+6.9) | +12.1 | +28.6 | LoRA passes; lr2x fails; full is memorising |
| general val perplexity up under 3% | +0.40% (+0.17%) | +0.85% | +1.19% | passed everywhere |

**The target was set for the wrong data scale.** Held-out perplexity measures domain transfer, which
continued pre-training produces at the scale of billions of tokens. With a 20M-token client corpus
the realistic goal is knowledge of those documents. That is measured by closed-book QA after SFT,
and it needs repetition or augmentation to stick.

**What Stage 2 established:**
1. **The learnable signal is the documents and their series, not the domain.** A probe
   (`eval/memorization.py`) compared the 246 corpus documents with 13 federal reports published in
   2026, after the base model's release:

   | Documents | Spans the base continues for 32+ tokens verbatim | Base perplexity | Change after CPT (`cpt-8b-replay10`) |
   |---|---|---|---|
   | Train (read once in CPT) | 6 of 3,744, all in-document patterns | 6.00 | -8.3% [-8.6, -7.9] |
   | Val (held out, same series) | 0 of 192 | 5.47 | -4.2% [-5.5, -3.0] |
   | 2026 (never seen) | 0 of 208 | 6.27 | -0.4% [-0.8, 0.0] |

   **How the probe's perplexity is measured.** It is the median over documents of each document's
   first two 4,096-token windows, and the change is per document on the same windows. That is not
   the measurement in the tables above (pooled over all val windows, and a 49-window train slice),
   so only ratios within this table are comparable. On the probe's footing val comes out easier
   than train, the reverse of the main table.

   **What it shows:**
   - **Gain stays within the series.** Every held-out document improves, but the 2026 reports barely
     move, including the closest ones (bridge load rating -0.5%, corroded steel beams -0.4%). That
     roughly 10x ratio between within-series gain and new-document gain is the finding.
   - **What val measures.** The val documents are siblings of train documents (USACE EMs, FEMA
     P-series, NIST 917 briefs, FHWA HIF reports) and share their structure, boilerplate and
     phrasing. So domain val perplexity measures learning within those series; the 2026 set
     measures transfer.
   - **Measured, it isn't copied text or genre** ([contamination checks](#contamination-checks)).
     - Val shares 1.5% of its 13-grams with train, less than train documents share with each
       other.
     - Verbatim sharing carries about a quarter of the val gain.
     - On windows with almost no shared text, val still gains -1.70% against -0.11% for the 2026
       reports.
     - Val's research reports gain -3.6% and its manuals -2.2%, against -0.9% for new guidance
       documents.
     - So CPT transfers within a document series, across genre, and barely to a new series.
   - **Little domain-general structure left.** The base already models this register well (domain
     val 6.9 against 8.2 on web text), so there is little general structure left to learn at this
     scale.
   - **Confirmed over whole documents.** Measured like domain val (`ppl_postcutoff`, all 82
     windows of the 13 reports), CPT moves the 2026 set -0.43% (replay) and -0.51% (seed 1),
     against -2.3% on held-out val.
   - **Caveats:** 13 documents, mostly research reports rather than manuals. A genre split rules
     genre out as the explanation (the table under [contamination checks](#contamination-checks)).
2. **Pushing harder moves the wrong way.** The train slice fell -8% (`cpt-8b`), -12% (LR 2e-4) and
   -22% (full-parameter), while held-out stayed flat each time, between -2.15% and -2.42%.
   - **lr2x:** general-text perplexity rose about 2x as much as the main run's.
   - **Full-parameter:** about 3x the main run's general-text rise, and about 4x its GSM8K loss
     (-3.0 points against -0.7 to -0.8). That GSM8K drop is the one benchmark delta that clears the
     noise (1.1) by a wide margin.
   - **Full-parameter's task gains** on the 130-item eval v1 (`results/table_v1.md`): +3.1 qa_acc
     (noise 3.2) and +3.8 vocab (noise 3.1), at the noise edge. Every one of those items comes from
     a training document, so they are knowledge of what it read. Its weights were deleted before
     the QA task grew, so it has no 325-item QA scores.

   LoRA stays the default: "LoRA learns less and forgets less" (Biderman et al., 2024).
3. **The base hadn't memorised the corpus.** The six "recalled" spans continue patterns set up in
   the prompt, such as `Table D-11` followed by `D-12`. Corpus documents are no easier for the base
   than 2026 documents it cannot have seen: geometric-mean perplexity +3.0%, interval [-8.5, +17.0].
   So pretraining exposure doesn't explain the small gain. A single pass in pretraining can't be
   ruled out: one epoch of our own CPT moves verbatim recall by 0.13 tokens, less than this
   comparison resolves.
4. **Pass/fail scores can't see Stage 2; the gold answer's probability can.** On the 325-item
   closed-book task (eval v2), qa_acc moves +2.5, +0.9 and +0.9 points for the three LoRA runs,
   against a noise of 1.8. The log-probability of the gold answer (`gold_lp`, per item, paired
   against `base-8b-hf`) moves clearly:

   | Run | Change in `gold_lp`, nats per answer [95% CI] | Items more likely |
   |---|---|---|
   | `cpt-8b` | +0.59 [+0.45, +0.75] | 70% |
   | `cpt-8b-seed1` | +0.58 [+0.44, +0.74] | 71% |
   | `cpt-8b-replay10` | +0.52 [+0.38, +0.66] | 68% |

   - **The seeds agree:** the two seeds differ by 0.007 nats.
   - **Numbers move least.** Per answer token, CPT moves values +0.06 nats, identifiers +0.13 and
     terms +0.15.
   - **What it is:** every eval item comes from a document CPT read, so this is the closed-book
     counterpart of the -8% train-slice perplexity: knowledge of what it read, not transfer.
   - **The end-marker fix:** the answer is scored with the end token the prompt uses after every
     answer ("\n\n"). The first measurement used a lone "\n", which never follows an answer in
     the prompt and made CPT look worse (`notes/decisions.md`).
   - **The other task scores** (grounded, citations, hallucination) need instruction following,
     which a base model lacks, and swing 4-5 points between seeds; they are Stage 3 metrics.
5. **Replay was adopted on a rule that fired at one noise unit; it's kept.** Mixing in 10%
   FineWeb-Edu left domain perplexity identical and brought the MMLU dip to -0.1 from -0.4. That
   margin is about one noise unit (0.3), so the pre-registered rule fired on noise; the flaw is
   recorded in `notes/decisions.md`. Its general-text perplexity fell 2.2%, but that slice is also
   FineWeb-Edu, so it measures in-distribution training, not protection. The 6.5 grounded and 6.4
   cite_valid points it "cost" are probably not a real cost:
   - they measure a base model's citation formatting, which SFT overwrites completely;
   - one seed alone moved grounded accuracy by 4.6 points.

   _Re-read after Stage 3:_ the grounded cost was real and the citation one formatting.
   - **Grounded:** replay10's grounded_acc (0.843 → 0.778) is 1.8x the 3.7-point noise. Raw-text CPT
     eroding few-shot instruction behaviour is a known cost, and this is a clean instance of it.
   - **Both are repairable:** SFT took both arms to grounded 0.91-0.94 and cite_valid 0.98-1.00.
6. **C: two GPUs reproduce one-GPU training step for step, at 2.23x the tokens/s.** Per-step loss
   is within 0.04% (median), and both runs see the same batches. Per GPU that is 6,610 against
   5,939 tokens/s (+11%).
   - **Matched:** micro-batch 4, and every decoder layer checkpointed (non-reentrant).
   - **Not memory-bound:** the one-GPU run peaked at 51 GB of 80.
   - **Different, so the +11% is not isolated by a run:** the code path. FSDP2 casts parameters to
     bf16 and uses its own checkpoint wrapper; the one-GPU run uses the Trainer's.
   - **Full-parameter vs LoRA cost:** on two GPUs, full-parameter matched LoRA's GPU-hours (0.91 vs
     0.95) despite about 1.33x the FLOPs per token. At roughly 8N against 6N FLOPs per token with
     checkpointing (N ≈ 8B, attention ignored), that is about 43% of H100 bf16 peak for full against
     about 30% for LoRA. LoRA's FLOP saving doesn't turn into speed; it is plausibly lost to the
     adapters' many small kernels, which nobody has profiled.
7. **The eval harness found a bug larger than any training effect.** vLLM 0.29 runs any HF-format
   Ministral 3 checkpoint, which includes every fine-tuned one, at the wrong attention temperature.
   - **Mechanism:** its YaRN code drops the config's `mscale` / `mscale_all_dim` keys and scales
     attention by 1.28.
   - **Cost:** on untouched base weights, 4.8% perplexity, 3.1 MMLU points, 6.1 GSM8K points and
     19.5 grounded points. It first made CPT look like it had wrecked MMLU and grounding.
   - **Culprit:** vLLM's reading of the config, not transformers' re-save or `merge.py`. Mistral's
     own `config.json` ships those keys.
   - **Fix:** `merge.py` writes `apply_yarn_scaling: false`, which matches transformers and
     Mistral's native path to five decimals. The repro and the upstream fix are in
     [`notes/contributions.md`](notes/contributions.md); the issue is not filed yet.

#### What I would do differently

Written after Stage 2 and before Stage 3's training, ordered by how much each would have changed the
result. Most of it is knowledge Stage 2 produced. The evidence for each item, and the rule it became,
is in [`notes/decisions.md`](notes/decisions.md) (2026-10-05 retrospective).

1. **Build the sensitive metric before the intervention.** Stage 2 was first read on a 130-item
   pass/fail task with a 3-point standard error. Its real effect (+0.59 nats of gold-answer
   log-probability, 70% of items up) showed only once `gold_lp`, the seen/unseen halves and the
   325-item set existed. All three belonged in Stage 0, a day's work moved a week earlier, and the
   right pre-registered target was `gold_lp` on facts from the documents read, not a 20% drop in
   held-out perplexity.
2. **Run a no-op control through the eval path before any training.** The base re-saved through
   `merge.py` and evaluated like a fine-tuned checkpoint was built only after `cpt-8b` seemed to lose
   3.7 MMLU points, and it exposed the YaRN bug that outweighed every training effect. On day one it
   would have cost one eval pass (about half an H100-hour), because an instrument has to read zero
   before it measures.
3. **Augment the facts that matter instead of reading them once.** One pass gives each value about
   one exposure, and values moved least (+0.06 nats per token against +0.13-0.15). QA and paraphrase
   rewrites of the 197 eval-seen chunks mixed into the CPT stream were affordable (about 1,000 API
   calls) and would have made Stage 2 a knowledge-injection test; Stage 3 now does the
   instruction-data version.
4. **Ablate along the axis the goal lives on.** The goal was knowledge, but the ablations asked about
   forgetting (replay, predicted null and null) and parameterisation (full vs LoRA). Epochs (1 vs 3)
   and augmentation (0 vs K rewrites), at about one H100-hour per LoRA epoch, were the informative
   runs; full-parameter and the 2-GPU run taught the engineering, and replay answered a question the
   pre-registration had already answered.
5. **Judge CPT by the downstream number.** Whether CPT was worth doing is SFT-from-CPT against
   SFT-from-base, not perplexity. That comparison should have been written into Stage 2's decision,
   which had no branch for leaving CPT out and picked the checkpoint on perplexity and MMLU; it is
   now Stage 3's control.
6. **Put boundaries where the text has them.** One EOS per manual (234 in 19.4M tokens) is the
   likely reason `cpt-8b` runs to the 256-token cap where the base stops after ~28 tokens.
   Section-level units with their own EOS would have kept ends common at no GPU cost, and packing
   whole units would also stop windows from starting mid-sentence; it surfaced only in the latency
   table.
7. **Choose the "new documents" set before training, matched to the corpus.** The 2026 reports were
   sourced after Stage 2, and 11 of 13 are research reports against a corpus of manuals. A genre
   split made afterwards rules genre out, but rests on two documents. Post-cutoff manuals from the
   same publishers, with series both in and out of train, chosen before training, would have made
   the -0.4% unarguable.
8. **Hygiene rules from day one.** Adapters kept, no checkpoint with a table row deleted, a paced
   API client instead of retry-on-429, a judge-cache guard. Each was written after a loss: the
   full-parameter run's weights (and with them its 325-item QA scores), worker time asleep or
   retrying, and 50 re-judged items.
9. **Calibrate every judge rubric before it labels anything.** The grading judge was hand-checked at
   Stage 0, but the SFT judge, a new rubric labelling training data, went straight to 4,020
   examples. A full-passage audit then found 18% of what it kept defective, every defect a framing
   error it had passed; 40 items with known defects would have shown that first.

#### Stage 3: supervised fine-tuning (SFT)

Supervised fine-tuning on a synthetic, fully read instruction set: the CPT checkpoint
(`cpt-8b-replay10`) and, as the control for what CPT bought, the base (`base-8b-hf`), each with
the same set, config and data order, and each run twice (seeds 0 and 1) so the noise floor is
measured on both arms.
Everything below was fixed before training, including the checkpoint rule, the merge gate and how
the results are read. It is in [`notes/decisions.md`](notes/decisions.md) (2026-10-06, Stage 3b),
with the two amendments made after the smoke run and the merge gate's, all before the evals they
govern.

**The data:** SFT set v1, 2,516 records (train 2,436, sft_val 80), dataset hash `70f47740`
(`data/sft/SHA256SUMS`).

| format | records | from eval-seen chunks | completion written by | read in full against the passages |
|---|---|---|---|---|
| closed-book | 1,144 | 892 | Mistral Large 3 869, Mistral Medium 3.5 275 | every record |
| definition | 276 | 266 | Large 213, Medium 63 | every record |
| grounded (4 passages, cited) | 401 | | Large 307, Medium 94 | every record |
| abstain (4 passages, no answer) | 195 | | the fixed sentence "Not in the provided passages." | the 53 hard negatives; 0 defects in 40 of the rest |
| replay (Tulu 3 SFT mixture) | 500 | | Tulu 3's sources: GPT-4o, GPT-3.5/4, Mixtral, people | sampled, not read |

- **Questions:** Mistral Large 3 wrote every domain question; half use the eval's exact
  instruction text and half are paraphrased.
- **Claude's part:** Claude built the tooling and read records against their passages,
  returning keep-or-drop verdicts only. No training record holds text Claude wrote, and the three
  Tulu 3 Persona subsets with Claude-written answers are excluded (rule 13).
- **sft_val** is the loss curve only, not a metric.

**The training run** (`train/configs/sft.yaml`; reasons in the pre-registration):

| item | value |
|---|---|
| starts | `cpt-8b-replay10` (sft-from-cpt) and `base-8b-hf` (sft-from-base, the control for what CPT bought) |
| format | Mistral chat, `<s>[INST] prompt [/INST] answer </s>`, no system prompt, pre-tokenised by mistral-common exactly as the eval's `--chat` path renders it; loss on the answer and its `</s>` only |
| LoRA | r 64, alpha 128, dropout 0.05, all seven projections of the language model (178M trainable) |
| optimiser and schedule | AdamW, LR 1e-4 linear to 0, 3% warmup, 32 sequences per step, 2 epochs (154 steps), one H100 |
| checkpoint rule | epoch 2 unless the closed-book or definition sft_val loss rose from epoch 1 to epoch 2 |
| gradient weight by format | token-weighted loss: replay 75%, grounded 16%, closed-book 4.6%, definition 4.3%, abstain 0.7% of the 179k completion tokens |

The training loss is token-weighted, so the 500 replay answers (general Tulu 3 instructions, long
completions) carry 75% of the gradient weight. Grounded answers carry 16%, and the recall formats
(closed-book and definition) 9%. Per-format loss weighting goes in next steps, not now.

![Stage 3 loss curves: train loss (10-step moving average) and the token-mean sft_val loss per run, epoch boundary marked](results/curves/sft.png)

![Stage 3 closed-book knowledge by half: gold-answer log-probability (chat-format runs) and qa_acc with binomial SE](results/curves/sft_kpi.png)

Generated by `train/report.py` from `results/runs/`, `results/noop/` and `results/diversity/`:

<!-- stage3-tables:start -->
##### Training runs

| run | start | steps | tokens trained | tokens/s | wall (h) | GPU-h | $ | peak GB | final train loss | val_loss epoch 1 / 2 | closed-book 1 / 2 | definition 1 / 2 | B4 picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft-from-cpt | cpt-8b-replay10 | 154 | 2.70M | 1,790 | 0.50 | 0.50 | 1.96 | 55 | 0.427 | 0.5592 / 0.5771 | 1.1655 / 1.3040 | 1.9832 / 2.0341 | epoch 1 |
| sft-from-base | base-8b-hf | 154 | 2.70M | 1,834 | 0.48 | 0.48 | 1.89 | 55 | 0.435 | 0.5592 / 0.5790 | 1.2134 / 1.3765 | 2.0178 / 2.0723 | epoch 1 |
| sft-from-cpt-seed1 | cpt-8b-replay10 | 154 | 2.70M | 1,737 | 0.54 | 0.54 | 2.12 | 55 | 0.362 | 0.5586 / 0.5804 | 1.1696 / 1.3126 | 1.8940 / 2.1049 | epoch 1 |
| sft-from-base-seed1 | base-8b-hf | 154 | 2.70M | 2,022 | 0.45 | 0.45 | 1.79 | 55 | 0.364 | 0.5580 / 0.5782 | 1.2347 / 1.3453 | 1.9116 / 2.0432 | epoch 1 |

B4 (pre-registered, amended before training): epoch 2 unless the closed-book or the definition sft_val loss (token mean) rose from epoch 1 to epoch 2. The overall val_loss is 83% replay tokens, so it is shown, not used. $ at 3.95 per GPU-hour (assumed).

##### Results next to the noise

| metric | instruct-8b | base-8b-hf | sft-from-base | sft-from-base-seed1 | cpt-8b-replay10 | sft-from-cpt | sft-from-cpt-seed1 | noise (Stage 3) | noise (Stage 2) |
|---|---|---|---|---|---|---|---|---|---|
| unseen gold-answer log-prob (nats) | -8.581 | -6.822 | -6.780 | -6.741 | -6.157 | -6.426 | -6.168 | 0.258 | 0.033 |
| seen gold-answer log-prob (nats) | -8.008 | -6.722 | -5.409 | -5.249 | -6.356 | -5.120 | -4.873 | 0.247 | 0.032 |
| qa_unseen | 0.084 | 0.116 | 0.110 | 0.097 | 0.110 | 0.136 | 0.129 | 2.7 | 2.7 |
| qa_seen | 0.114 | 0.126 | 0.245 | 0.234 | 0.150 | 0.281 | 0.270 | 3.5 | 2.8 |
| qa_ident (identifiers) | 0.031 | 0.078 | 0.125 | 0.141 | 0.125 | 0.203 | 0.234 | 5.0 | 4.1 |
| grounded_acc | 0.898 | 0.843 | 0.926 | 0.898 | 0.778 | 0.907 | 0.935 | 2.8 | 3.7 |
| cite_supported | 0.787 | 0.083 | 0.870 | 0.880 | 0.037 | 0.861 | 0.880 | 3.3 | 3.7 |
| halluc_rate (lower is better) | 0.013 | 0.895 | 0.066 | 0.013 | 0.934 | 0.053 | 0.013 | 3.9 | 3.3 |
| false_abstain (lower is better) | 0.074 | 0.009 | 0.000 | 0.009 | 0.000 | 0.000 | 0.009 | 0.9 | 0.0 |
| vocab_seen | 0.802 | 0.713 | 0.911 | 0.911 | 0.713 | 0.891 | 0.901 | 3.1 | 4.6 |
| vocab_unseen | 0.771 | 0.697 | 0.761 | 0.752 | 0.688 | 0.780 | 0.817 | 4.0 | 4.3 |
| MMLU | 0.761 | 0.767 | 0.767 | 0.768 | 0.766 | 0.766 | 0.770 | 0.4 | 0.3 |
| GSM8K | 0.855 | 0.793 | 0.792 | 0.790 | 0.791 | 0.814 | 0.792 | 2.2 | 1.1 |
| HellaSwag | 0.801 | 0.801 | 0.795 | 0.799 | 0.800 | 0.794 | 0.799 | 0.6 | 0.4 |

Values as fractions (gold_lp in nats per answer); noise in points (gold_lp in nats). Noise = max(the seed gap, the SE): Stage 3 for sft-from-cpt vs sft-from-cpt-seed1, Stage 2 for cpt-8b vs cpt-8b-seed1. The SE is binomial on the metric's items, lm-eval's stderr, or for gold_lp the paired per-item SE of the twins on that half. Starts: sft-from-cpt from cpt-8b-replay10, sft-from-base from base-8b-hf; instruct-8b is the bar.

##### B7's first line: what CPT bought, measured after SFT

- **unseen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.464 nats per answer [95% CI over items +0.276, +0.660; 155 items, 68% up]; noise 0.130 (run-variance SD 0.130 from seed gaps 0.258 (CPT arm) and 0.039 (base arm), paired SE 0.099; the difference is 3.6 run SD, indicative only: each arm's SD rests on one seed pair (1 df). Single-run pairs +0.355, +0.315, +0.612, +0.573; every CPT run on one side of every base run, an ordering with exact permutation probability 1 in 6): beyond the noise: CPT bought something that survives SFT. The item CI conditions on these training runs; run variance enters only through the noise.
- **seen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.332 nats per answer [95% CI over items +0.191, +0.482; 167 items, 66% up]; noise 0.147 (run-variance SD 0.147 from seed gaps 0.247 (CPT arm) and 0.160 (base arm), paired SE 0.074; the difference is 2.3 run SD, indicative only: each arm's SD rests on one seed pair (1 df). Single-run pairs +0.289, +0.129, +0.536, +0.376; every CPT run on one side of every base run, an ordering with exact permutation probability 1 in 6): beyond the noise: CPT bought something that survives SFT. The item CI conditions on these training runs; run variance enters only through the noise.

##### Checks

No-op control (an untrained adapter, merged, against its start):

| start | tensors | differ | max abs diff | passed |
|---|---|---|---|---|
| base-8b-hf | 531 | 0 | 0.0 | yes |
| cpt-8b-replay10 | 531 | 0 | 0.0 | yes |

Merge gate (B5, amended): over all 11,351 sft_val completion positions against an fp32 reference, the argmax flips the merge adds over the unmerged bf16 model's own, and its mean |delta log-prob| relative to the unmerged model's (max 1.5); val loss within 0.5%. Merged-vs-unmerged agreement and the 3-probe mean merge-error ratio are reported, not gated; sha256 of the merged checkpoint's file list:

| run | flips added | \|dlp\| ratio | val loss diff | merged vs unmerged top-1 | probe error ratio | checkpoint sha256 | passed |
|---|---|---|---|---|---|---|---|
| sft-from-cpt | 5 of 11 | 1.005 | 0.03% | 99.60% | 0.0516 | `db8ddabfc9a8` | yes |
| sft-from-base | 1 of 11 | 0.965 | 0.02% | 99.67% | 0.046 | `7c6458a78404` | yes |
| sft-from-cpt-seed1 | -8 of 11 | 0.978 | 0.02% | 99.54% | 0.0427 | `5f805f85ed59` | yes |
| sft-from-base-seed1 | 11 of 11 | 0.984 | 0.02% | 99.53% | 0.0466 | `272cbcb28872` | yes |

Diversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over output tokens) and </s> on sampled answers (20 Stage 4 prompts x 4 at T 0.8):

| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | distinct-4 domain | stopped (T 0.7) | stopped (eos job) |
|---|---|---|---|---|---|---|---|
| instruct-8b | 0.8104 | 9.7998 | 343.2 | 0.8099 | 0.817 | 0.82 |  |
| sft-from-cpt | 0.7724 | 9.0197 | 163.4 | 0.7567 | 0.8646 | 0.98 | 98.8% of 80 |
| sft-from-base | 0.79 | 9.3114 | 185.3 | 0.7765 | 0.8847 | 0.95 | 98.8% of 80 |
| sft-from-cpt-seed1 | 0.7799 | 9.1733 | 196.4 | 0.7719 | 0.837 | 0.97 | 98.8% of 80 |
| sft-from-base-seed1 | 0.8026 | 9.3672 | 190.8 | 0.7955 | 0.8533 | 1.0 | 98.8% of 80 |
<!-- stage3-tables:end -->

**What holds.**
1. **At 2,436 records, the recall formats overfit in the second epoch, in all four runs.**
   - Closed-book and definition val loss bottomed at the end of epoch 1 and rose through epoch 2,
     while train loss kept falling: the set was being memorised.
   - The pre-registered rule (amended before training to read those formats, not the mixture mean)
     caught it on every run and took epoch 1.
   - For a domain SFT set this size, one epoch is the budget, and the recall formats are where to
     watch for overfitting.
2. **It stops.**
   - Every KPI answer, 98.75% of sampled answers and 100% of bench requests end on `</s>`.
   - End-to-end latency at one request is 159 ms, against the CPT checkpoint's 1,753 ms, at the same
     per-token speed (Serving).
3. **It cites and it doesn't over-refuse.**
   - Cited answers backed by the cited passages: 86.1% against Instruct's 78.7%.
   - False refusals on answerable grounded questions: 0% against Instruct's 7.4%.
   - Grounded accuracy matches Instruct (90.7% vs 89.8%).
4. **General ability is intact.** MMLU, GSM8K and HellaSwag are within noise of each start.
5. **SFT repaired the passage reading that CPT eroded.**
   - Raw-text CPT cost few-shot passage reading: grounded_acc went from 0.843 for the base to 0.778
     for `cpt-8b-replay10`, 1.8x the 3.7-point noise. Eroded instruction behaviour is a known cost
     of continued pre-training.
   - SFT took both arms to 0.91-0.94, so the cost didn't carry into the chain.
6. **CPT's contribution survives SFT.**
   - **The rule:** it passes the pre-registered rule.
   - **Consistency:** it holds in all four pairings of a CPT-start run with a base-start run, and
     on both halves.
   - **The seed-matched pairs are the cleanest comparisons.** Each seed used the same data order and
     LoRA init in both arms. At seed 0 the difference is +0.355; at seed 1 it is +0.573. The
     difference of the arm means is +0.464.
   - **The per-item CI:** [+0.276, +0.660] nats per answer on the unseen half.
   - **The run-variance estimate rests on one seed pair per arm,** so the SD multiple is indicative.
     A third seed per arm is what would turn it into a test.

   | unseen-half `gold_lp`, CPT arm − base arm (2 seeds each) | value |
   |---|---|
   | difference of arm means (nats per answer) | +0.464 |
   | per-item 95% CI | [+0.276, +0.660], 68% of items up |
   | the four single-run pairs | +0.355, +0.315, +0.612, +0.573 |
   | run-variance floor from both seed gaps (0.258, 0.039) | 0.130, so 3.6 SD |
   | the same with the CPT arm's gap on both arms | 0.182, so 2.5 SD |
   | seen half, difference of arm means | +0.332 (2.3 SD) |

   - **The SD multiples aren't sigmas.** They rest on 1 degree of freedom per arm, so read them as
     indicative, not as a p-value.
   - **The robust statement is the rank one.** Every CPT run beats every base run, and the smallest
     of the four pair differences (+0.32) exceeds the CPT arm's own seed gap (0.258). With 2 runs
     per arm, that ordering has an exact permutation probability of 1 in 6.
   - **Pass/fail:** unseen qa_acc is 13.2% against 10.3% between the arm means, inside the binomial
     noise: reported, not argued. The effect lives in the probabilities, as Stage 2's did.
   - **Identifiers are the line that holds across stages.**
     - Stage 2's CPT moved them most per answer. Per token (+0.127 nats) they were second to terms
       (+0.147, a wide interval on 38 items); per answer they lead because they are the longest
       answers.
     - After SFT, the CPT arm leads the base arm on qa_ident by +8.6 points (0.219 vs 0.133). That
       is on 64 items, with a standard error of about 5 points per run.
   - **The design is blocked by seed, and the seed shows.**
     - Seed 1 beats seed 0 in both arms: on seen `gold_lp` (−5.25 vs −5.41 base, −4.87 vs −5.12
       CPT), on unseen `gold_lp` (−6.74 vs −6.78, −6.17 vs −6.43), on qa_ident, and on final train
       loss (0.36 vs 0.43 in both arms).
     - Hallucinations follow the same line: 5 and 4 at seed 0, 1 and 1 at seed 1.
     - In a one-epoch run, data order is a hyperparameter, and the "seed gap" here is LoRA init
       plus data order, inseparable. This is observed, on two blocks; nothing more is claimed.
   - **What can't be computed here: how much of CPT's own gain survives SFT.** CPT's `gold_lp`
     (−6.16 unseen) is scored in base format and the SFT rows' in chat format, so the two columns
     can't be subtracted. The comparison that has a number is the one above: SFT from CPT against
     SFT from the base, both in chat format.

**What is weaker than a summary would make it.**
1. **The gain over Instruct on closed-book facts is retention, not capability.** The seen half's
   facts were in the training set by design; the unseen half's were not.

   | closed-book | seen half: retention of trained facts | unseen half: transfer |
   |---|---|---|
   | instruct-8b | 19/167 (11.4%) | 13/155 (8.4%) |
   | cpt-8b-replay10 (start) | 25/167 (15.0%) | 17/155 (11.0%) |
   | sft-from-base | 41/167 (24.5%) | 17/155 (11.0%) |
   | **sft-from-cpt** | **47/167 (28.1%)** | **21/155 (13.5%)** |
   | identifiers, instruct-8b | 2/30 | 0/34 |
   | identifiers, sft-from-cpt | 10/30 | 3/34 (its CPT start: 3/34) |

   - **Terms are 38 items, so read them as counts.** Correct answers are 2 for instruct-8b, base and
     CPT, 0 and 2 for sft-from-base s0/s1, 1 and 4 for sft-from-cpt s0/s1, and 7 for Large 3. As a
     rate (0.000-0.105) the column invites a reading it can't support: one item is 2.6 points.

   - **Seen half:** a client wants the model to know their documents, and SFT delivers that here,
     about 2.5x Instruct. At 28.1% it matches Mistral Large 3's 27.5% on the same items. These are
     facts that were in the training set, and Large 3 never saw them.
   - **Unseen half:** 13.5% against Instruct's 8.4% sits inside the noise floor. Every unseen
     identifier it gets right, its CPT start already had.
2. **Replay carried the gradient.**
   - 500 general Tulu 3 answers are 20% of the records, but their long completions are 75% of the
     token-weighted loss. The closed-book and definition records the seen half measures are 9%.
   - The run was mostly general instruction tuning with a domain component.
3. **Hallucination on unanswerable questions is 4 of 76 against Instruct's 1 of 76** (5.3% vs 1.3%).
   The seed twin also has 1 of 76, so the difference is a few items, at the edge of the run-to-run
   spread. Both are inside the pre-registered target (< 20%).
4. **Diversity passes its line on length-confounded numbers.**
   - **The numbers:** distinct-4 −4.7% and entropy −8.0% against Instruct, inside the
     pre-registered 10%.
   - **The confound:** answers are half Instruct's length, a style the teacher's short completions
     taught. Pooled entropy and distinct-4 move with length.
   - **Why it matters for Stage 4:** if 4 samples per prompt at T 0.7 come out near-identical, DPO
     pairs have no margin. It is checked on the first 20 prompts before sampling the pool.
5. **Two gates were amended before the evals they govern** (both recorded in `decisions.md`).
   - **The step-1 loss band** assumed recall answers. It is now read per format.
   - **The merge gate's 3 probes** (295 positions) put a 99% line 2 flips from failing. The CPT
     run's merge failed on 4 flips of near-tie noise. The amended gate compares the merge against an
     fp32 reference over all 11,351 positions: it adds 5 flips to bf16's own 74. Every gated
     checkpoint's sha256 matches the one evaluated.
   - **sft-from-base-seed1's merge** passed the amended gate at its line: exactly 11 added flips of 11
     allowed, with log-prob error 0.98x and val loss within 0.02%.

**Scored against its pre-registered targets:** one miss, by a few items.

| target (B7, fixed before training) | result | verdict |
|---|---|---|
| CPT's advantage survives SFT (unseen `gold_lp` beyond the floor) | +0.46 nats, all four pairings positive | pass; the SD multiple rests on one seed pair per arm |
| beat Instruct on identifiers | 0.203 vs 0.031 | pass |
| beat Instruct on vocab | 0.833 vs 0.786 | pass, narrowly (1.5x the noise) |
| match Instruct on grounded and citation | grounded 0.907 vs 0.898; cite_supported 0.861 vs 0.787 | pass; beats on citation |
| guards: no half below its start; MMLU and GSM8K within noise | all four runs | pass |
| abstain: > 80% on unanswerable, < 5% false refusals | 94.7%; 0% | pass |
| diversity within 10% of Instruct | distinct-4 −4.7%, entropy −8.0% | pass, length-confounded |
| stops before the cap; latency | 100%; 159 ms vs 1,753 ms | pass |
| hallucination at or under Instruct | 4 of 76 vs 1 of 76 | miss, inside the run-to-run spread (the seed twin has 1 of 76) |

**What SFT did, and what it didn't.** SFT did its job, behaviour and the facts it was shown, and
it did not erase CPT's knowledge. It did not generalise to unseen facts, which was never a
target.
- **Behaviour is SFT's own, and the deliverable.** Answer form, citations that are valid and
  supported, abstaining without over-refusing, and stopping. This is where it beats Instruct
  outright.
- **The facts it was shown are SFT's own too.** The seen half doubled in both arms: 0.150 → 0.281
  from CPT and 0.126 → 0.245 from the base. Most of that gain is SFT teaching the facts in its
  data, not CPT's knowledge surfacing.
- **What is CPT-specific is the arm difference.** SFT preserved CPT's knowledge and made it usable
  in chat form. Starting from CPT rather than the base is worth +0.46 nats on unseen `gold_lp`,
  +3.6 points on the seen half and +8.6 on identifiers (about 1.7 SE). It is real by the rule and
  smaller than the seen-half gain.
  - Unseen accuracy differs by +2.9 points between the arms (0.133 vs 0.104). That is inside the
    binomial noise, so it isn't the evidence; `gold_lp` is.
  - Whether this is "the same" knowledge CPT added (+0.66 nats over the base, in base format)
    can't be computed across the format split. The arm difference is consistent with it.
- **SFT added no unseen knowledge in either arm.**
  - Unseen accuracy moved 0.116 → 0.104 (base arm) and 0.110 → 0.133 (CPT arm), both inside the
    noise.
  - The only unseen effect in the whole table is CPT's. The unseen half is bounded by what the
    weights knew before SFT: pretraining plus 19M tokens of CPT.
  - This is the textbook division of labour. Continued pre-training puts knowledge in;
    fine-tuning teaches behaviour and makes stored facts extractable.
    - QA fine-tuning extracts facts only when pre-training exposed them with enough variety
      (Allen-Zhu & Li 2023, "Physics of Language Models, Part 3.1").
    - Fine-tuning on facts a model doesn't know is learned slowly and raises hallucination rather
      than knowledge (Gekhman et al. 2024).
- **The lever for the unseen half is upstream.**
  - Only more, and more varied, CPT exposure acts on unseen facts here: paraphrased and
    restructured corpus text, not one pass over concatenated documents (Stage 2's "what I would
    do differently", item 3). DPO and GRPO shape behaviour on prompts, as SFT does, and won't move
    it.
  - Mistral Large 3 answers 28.4% of the same unseen questions, so the questions are answerable.
    The ceiling here is how much the weights know.

#### What I would do differently (Stage 3)

1. **Look at the gradient shares before freezing the loss.**
   - Replay's long answers carry 75% of the token-weighted loss, and the recall formats the seen
     half measures 9%, so the run was mostly Tulu training.
   - It surfaced only after the smoke run, from a number misread on the smoke batch.
   - Per-record or per-format weighting, or a replay token cap, decided with the data, would have
     aimed the gradient at the facts.
2. **Give both arms of the control a seed twin from the start.** "What CPT bought" was first read
   against a noise floor from one arm, at 1.4 SD. The base arm's twin (0.5 GPU-h) put it at 3.6 SD;
   it should have been in the plan, not added after the read.
3. **Size a gate's sample for its threshold, and reference it to the exact function.**
   - A 99% agreement line on 295 positions is two flips from failing.
   - Comparing two bf16 models measures bf16's own near-tie noise.
   - The fp32-referenced, full-set comparison costs a few GPU minutes and should have been the gate
     from the start.
4. **Pre-register sanity bands per format.** A mixture's token mean hides the formats that matter,
   and the step-1 band failed for that reason, not for a bug.
5. **Find the minimum, don't bracket it.** All three runs overfit in epoch 2. An eval every ~15
   steps on the recall formats would locate the best step, rather than choosing between two epoch
   ends.

### 4. Results

Every scored checkpoint, from [`results/table.md`](results/table.md) (copied here by
`train/report.py`), on eval v3: domain_qa has 322 items (v2's 325 minus the three that shared a
few-shot item's passage), the other tasks are unchanged. Every row was rescored on v3 from its
saved generations. The earlier tables are frozen:
- [`results/table_v2.md`](results/table_v2.md): the 325-item set, as Stage 2 was first read;
- [`results/table_v1.md`](results/table_v1.md) and [`results/table_v1.1.md`](results/table_v1.1.md):
  the 130-item set.

Stage 2 rows are base models, scored without `--chat`; Instruct and the SFT rows are chat models,
scored with it. `qa_term` covers 38 items (2.6 points each), so read it as counts (Stage 3's
section gives them).

**Reading the closed-book numbers.** The `qa_*` and `gold_lp` columns ask for facts from specific
pages of the manuals (a value, a document or article number, a term) with no retrieval and no
passage in the prompt.
- **The gap is the point.** With passages from the same documents in the prompt (the grounded
  task), the base model answers 84% correctly; closed-book, 12%. That gap is the knowledge the
  documents hold and the model doesn't, and these columns measure what training does to it.
- **For scale:** Mistral Large 3 answers 28% of the same 325 questions closed-book (row
  `mistral-large-3`, through the API, closed-book only): 45% of identifiers, 25% of values and 18%
  of terms. On SimpleQA, whose facts are far more common, frontier models score 30-40%.
- **The scores are knowledge, not scoring:** a hand audit of 169 wrong answers found 2 scoring
  errors, both fixed.
- **The column to watch is Stage 3's seen half.** SFT synthesis supplies exposures to those facts;
  the unseen half shows whether anything transfers.

- **Task scores** come from the frozen eval, with judge columns scored locally:
  - `qa_num`, `qa_ident` and `qa_term` split `qa_acc` by answer kind (`eval/qa_rules.py`):
    values, identifiers (document ids and article numbers) and terms, which include everything
    else. A hand audit of 169 misses found 2 scoring errors, both fixed, so the low scores are
    genuine.
  - The `_seen` / `_unseen` columns split `qa_acc` and `vocab_recall` by whether Stage 3's SFT
    synthesis may use the item's source chunk (`eval/sft_split.py`). Before Stage 3 nothing is
    seen, so the two halves (167 and 155 QA items) are a null check.
  - `false_abstain` is the share of grounded answers that use the abstain phrase although the
    passages hold the answer: the cost side of a low `halluc_rate`.
  - `cpt-8b-full` and the Stage 0 `base-8b` have no QA scores on v2. Full's weights were deleted
    before the task grew, and `base-8b-hf` supersedes the Stage 0 row.
- **Gold-answer log-probability** (`gold_lp`, nats per answer, higher is better,
  `eval/gold_lp.py`) is the continuous companion to `qa_acc` on the same items. Instruct and the
  SFT rows are scored in their chat format, so their values compare with each other, not with the
  base-format rows.
- **Benchmarks** are from lm-eval: 5-shot, never with the chat template.
- **Perplexity** is from `eval/perplexity.py` (lower is better). `ppl_postcutoff` covers the 13
  federal reports published after the base model.

<!-- results-table:start -->
Items per task: domain_qa 322, grounded 108, vocab 210, adversarial 76, qa_number 220, qa_identifier 64, qa_term 38.

**Closed-book knowledge: no retrieval, no passage in the prompt; questions about facts on specific pages of the manuals (gold_lp: nats per answer, higher is better)**

| run | gold_lp | gold_lp_seen | gold_lp_unseen | qa_acc | qa_num | qa_ident | qa_term | qa_seen | qa_unseen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-8b-hf | -6.770 | -6.722 | -6.822 | 0.121 | 0.145 | 0.078 | 0.053 | 0.126 | 0.116 |
| instruct-8b | -8.284 | -8.008 | -8.581 | 0.099 | 0.127 | 0.031 | 0.053 | 0.114 | 0.084 |
| cpt-8b | -6.184 | -6.237 | -6.128 | 0.146 | 0.168 | 0.125 | 0.053 | 0.156 | 0.136 |
| cpt-8b-seed1 | -6.192 | -6.236 | -6.145 | 0.130 | 0.145 | 0.125 | 0.053 | 0.132 | 0.129 |
| cpt-8b-replay10 | -6.260 | -6.356 | -6.157 | 0.130 | 0.145 | 0.125 | 0.053 | 0.150 | 0.110 |
| mistral-large-3 |  |  |  | 0.280 | 0.245 | 0.453 | 0.184 | 0.275 | 0.284 |
| sft-from-base | -6.069 | -5.409 | -6.780 | 0.180 | 0.227 | 0.125 | 0.000 | 0.245 | 0.110 |
| sft-from-cpt-seed1 | -5.497 | -4.873 | -6.168 | 0.202 | 0.209 | 0.234 | 0.105 | 0.270 | 0.129 |
| sft-from-cpt | -5.749 | -5.120 | -6.426 | 0.211 | 0.245 | 0.203 | 0.026 | 0.281 | 0.136 |
| sft-from-base-seed1 | -5.967 | -5.249 | -6.741 | 0.168 | 0.196 | 0.141 | 0.053 | 0.234 | 0.097 |

**With the passages: grounded answers and citations (4 passages given), abstention when the passages lack the answer (halluc_rate, lower is better), and definitions**

| run | grounded_acc | cite_valid | cite_supported | halluc_rate | false_abstain | vocab_recall | vocab_seen | vocab_unseen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-8b-hf | 0.843 | 0.120 | 0.083 | 0.895 | 0.009 | 0.705 | 0.713 | 0.697 |
| instruct-8b | 0.898 | 0.833 | 0.787 | 0.013 | 0.074 | 0.786 | 0.802 | 0.771 |
| cpt-8b | 0.833 | 0.148 | 0.074 | 0.908 | 0.000 | 0.710 | 0.693 | 0.725 |
| cpt-8b-seed1 | 0.796 | 0.102 | 0.037 | 0.934 | 0.000 | 0.700 | 0.713 | 0.688 |
| cpt-8b-replay10 | 0.778 | 0.056 | 0.037 | 0.934 | 0.000 | 0.700 | 0.713 | 0.688 |
| cpt-8b-full | 0.889 | 0.148 | 0.102 | 0.921 | 0.000 | 0.743 | 0.733 | 0.752 |
| base-8b | 0.852 | 0.130 | 0.102 | 0.882 | 0.009 | 0.719 | 0.723 | 0.716 |
| sft-from-base | 0.926 | 1.000 | 0.870 | 0.066 | 0.000 | 0.833 | 0.911 | 0.761 |
| sft-from-cpt-seed1 | 0.935 | 0.982 | 0.880 | 0.013 | 0.009 | 0.857 | 0.901 | 0.817 |
| sft-from-cpt | 0.907 | 1.000 | 0.861 | 0.053 | 0.000 | 0.833 | 0.891 | 0.780 |
| sft-from-base-seed1 | 0.898 | 0.991 | 0.880 | 0.013 | 0.009 | 0.829 | 0.911 | 0.752 |

**General benchmarks (5-shot, no chat template) and perplexity (lower is better)**

| run | mmlu | mmlu_stem | mmlu_hum | mmlu_soc | mmlu_other | gsm8k | hellaswag | ppl_train | ppl_domain_val | ppl_general_val | ppl_postcutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-8b-hf | 0.767 | 0.733 | 0.704 | 0.862 | 0.803 | 0.793 | 0.801 | 6.18 | 6.88 | 8.15 | 6.21 |
| instruct-8b | 0.761 | 0.735 | 0.691 | 0.854 | 0.802 | 0.855 | 0.801 |  |  |  |  |
| cpt-8b | 0.764 | 0.729 | 0.699 | 0.857 | 0.804 | 0.785 | 0.801 | 5.67 | 6.72 | 8.18 |  |
| cpt-8b-seed1 | 0.765 | 0.727 | 0.699 | 0.861 | 0.809 | 0.786 | 0.800 | 5.69 | 6.72 | 8.17 | 6.17 |
| cpt-8b-replay10 | 0.766 | 0.730 | 0.707 | 0.856 | 0.806 | 0.791 | 0.800 | 5.65 | 6.72 | 7.97 | 6.18 |
| cpt-8b-full | 0.762 | 0.729 | 0.692 | 0.860 | 0.806 | 0.763 | 0.798 | 4.82 | 6.73 | 8.25 |  |
| base-8b | 0.768 | 0.732 | 0.707 | 0.862 | 0.805 | 0.794 | 0.801 | 6.18 | 6.88 | 8.15 |  |
| sft-from-base | 0.767 | 0.729 | 0.700 | 0.862 | 0.811 | 0.792 | 0.795 | 6.31 | 7.01 | 8.23 | 6.31 |
| sft-from-cpt-seed1 | 0.770 | 0.727 | 0.710 | 0.861 | 0.814 | 0.792 | 0.799 | 5.79 | 6.87 | 8.07 | 6.27 |
| sft-from-cpt | 0.766 | 0.730 | 0.701 | 0.859 | 0.812 | 0.814 | 0.794 | 5.80 | 6.87 | 8.06 | 6.29 |
| sft-from-base-seed1 | 0.768 | 0.732 | 0.706 | 0.857 | 0.813 | 0.790 | 0.799 | 6.32 | 7.01 | 8.25 | 6.31 |
<!-- results-table:end -->

**Stage 2 (CPT) earned little.**
- **Perplexity:** held-out perplexity fell 2.3%, and documents published in 2026 gained 0.4%.
- **Task scores:** no pass/fail score moved outside the noise.
- **Gold answers:** closed-book gold answers to facts from the documents it read became about 1.8x
  more probable (`gold_lp` +0.59 nats per answer), which SFT can build on.
- **Why `cpt-8b-replay10` goes forward:** LoRA with replay costs almost nothing in forgetting.

**Stage 3 (SFT):**
- **Holds:**
  - The model stops: 159 ms against 1,753 ms end to end.
  - It cites better than Instruct and never refuses an answerable question.
  - The recall formats overfit in epoch 2, which the pre-registered rule caught on every run.
- **Retention, not capability:** the seen half's closed-book score (28% from 15%; facts that were
  in the training set) is retention. The unseen half moved inside the noise.
- **Attribution:** SFT did its job, behaviour and the facts it was shown, and did not erase CPT's
  knowledge. It did not generalise to unseen facts, which was never a target; only more varied CPT
  exposure acts on those.
- **CPT's contribution survives SFT:** unseen-half `gold_lp` is +0.46 nats over the SFT-only
  control (arm means of two seeds each), 3.6 SD of run variance measured on both arms. Pass/fail
  closed-book accuracy doesn't resolve it.
- **Going forward:** `sft-from-cpt` (epoch 1) is the Stage 3 checkpoint; `train/configs/dpo.yaml`
  starts from it.

### 5. Serving

vLLM on one H100, `serve/bench_latency.py` (64 streamed requests per concurrency level, 256 output
tokens max; generated by `train/report.py` from `results/bench/`).
- **Paths:** `base-8b` and `instruct-8b` run Mistral's native vLLM path. The rest run the HF path
  with the YaRN fix.
- **New columns from Stage 3:** stop before cap and mean output tokens, at one request.
- **Prompts:** the SFT rows' prompt sample differs slightly from earlier rows'. Prompts are drawn
  from the eval tasks, which lost three items in v3.

<!-- serving-table:start -->
| run | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | stop before cap | mean output tokens | tok/s @1 | tok/s @8 | tok/s @32 |
|---|---|---|---|---|---|---|---|---|
| base-8b | 17.7 | 6.6 | 187 |  |  | 143 | 875 | 1,922 |
| instruct-8b | 17.7 | 6.6 | 320 |  |  | 139 | 677 | 1,144 |
| base-8b-hf | 15.3 | 6.6 | 180 |  |  | 145 | 896 | 1,967 |
| cpt-8b | 16.5 | 6.9 | 1,753 |  |  | 141 | 922 | 2,166 |
| sft-from-cpt | 17.8 | 6.8 | 159 | 100% | 23 | 122 | 297 | 783 |
| sft-from-base | 18.4 | 6.8 | 154 | 100% | 24 | 127 | 509 | 829 |
| sft-from-cpt-seed1 | 16.2 | 6.9 | 153 | 100% | 24 | 127 | 563 | 788 |
| sft-from-base-seed1 | 17.6 | 6.8 | 147 | 100% | 23 | 126 | 558 | 816 |
<!-- serving-table:end -->

- **The two paths serve the base at the same speed** once the YaRN fix is in: per-token latency
  6.6 ms either way, end to end 180 vs 187 ms.
- **`cpt-8b`'s end-to-end time is not a latency result.** It decodes at the same 6.6-6.9 ms per
  token as the base. It runs to the 256-token cap, though, where the base stops after ~28 tokens,
  hence the ~10x end-to-end time and the higher throughput (more tokens per request).
  - **Cause:** the corpus holds one EOS per whole manual, 234 in 19.4M tokens. Section-level units
    would likely have avoided it (item 6 of
    [What I would do differently](#what-i-would-do-differently)).
  - **Fix, confirmed by Stage 3:** after SFT every request stops before the cap (mean 23 output
    tokens), and end to end at one request is 159 ms against 1,753 ms, at the same per-token speed
    (6.8 ms).
  - **Use:** compare the row only with this note.
- **The SFT rows' lower throughput is short answers, not slower decoding.** At 8 and 32 concurrent
  requests, tokens/s counts output tokens. With ~23 tokens per answer, prefill and scheduling take
  a larger share of each request, so the SFT rows produce fewer output tokens per second.
  - **One cell isn't explained by that:** `sft-from-cpt` at 8 concurrent requests (297 tok/s,
    against 509-563 for its siblings at the same mean length). Each cell is a single 64-request
    sample, so read this one as such until it is rerun.
- _Stage 6 adds the AWQ checkpoint of the SFT'd model: its quality delta against bf16 and its
  latency at 1 and 32 concurrent requests._

### 6. What I'd do next

**From Stage 3:**
- **Check sample diversity before building DPO pairs:** 4 samples per prompt at T 0.7 on the first
  20 prompts. Near-identical samples give pairs with no margin.
- **Weight the loss per format.** SFT's token-weighted loss gives the 500 replay answers 75% of the
  gradient and the closed-book and definition records, the ones the seen half measures, 9%.
  Normalising per record or per format, or capping replay's share, would aim the gradient at the
  domain facts. It wasn't changed for Stage 3, whose set and loss were frozen before training.
- **Padding-free batches with FlashAttention-2.** SFT trained at about 1.8k tokens/s against
  Stage 2's 5.9k. That is the cost of no packing: each micro-batch is padded to its longest record,
  and SDPA computes on the padding. `padding_free` with FA2 (a flash-attn build in the image)
  recovers it. It belongs with the per-format loss weighting, since both change the batch.

**From Stage 2:**
- **Split Stage 3's eval into seen and unseen halves.** Report `domain_qa` and `vocab` in two
  halves: items whose source chunk fed an SFT example (seen) and items whose chunk did not (unseen).
  Hold a deliberate share of the eval's source chunks out of SFT synthesis so the unseen half
  exists. Seen measures knowledge injection, unseen measures transfer. One pooled number would be
  as uninterpretable as Stage 2's single held-out perplexity.
- **The bar is `instruct-8b`:** grounded 0.898, cite_valid 0.833, cite_supported 0.787, halluc
  0.013, vocab 0.786, qa 0.099 (eval v2). Match it on grounding and beat it on both qa halves. Read
  halluc_rate next to grounded_acc, so a low hallucination rate means abstaining on unanswerable
  questions and not refusing across the board.
- **Knowledge of a 20M-token corpus needs repetition or augmentation,** such as paraphrased
  restatements or QA rewrites of each document, rather than one pass of next-token training.
- **Keep the 2026 set as the standing transfer measure.** It is `ppl_postcutoff`, now measured
  over whole documents. Grow it; FHWA's 2026 reports need a browser download.
- **Use fused or chunked cross-entropy for single-GPU training.** The 131k-vocab logits are the
  largest activation, and smaller logits memory allows larger micro-batches.
- **File the vLLM YaRN issue** with the repro in `notes/contributions.md`.

_Limitations from later stages (judge bias, eval set size, reward hacking in GRPO) follow as those
stages land._
