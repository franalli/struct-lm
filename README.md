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
| 1 | Closed-book domain knowledge | 130 questions with exact or numeric answers from the corpus | `qa_acc` |
| 2 | Answering from given passages, with citations | 108 questions, 4 passages each (gold plus distractors) | `grounded_acc`, `cite_valid`, `cite_supported` |
| 3 | Domain vocabulary | 210 terms to define in one sentence | `vocab_recall` |
| 4 | Declining when the answer isn't there | 76 questions whose 3 passages don't contain the answer | `halluc_rate` (lower is better) |
| 5 | General capability, to catch forgetting | MMLU, GSM8K, HellaSwag, 5-shot | `mmlu`, `gsm8k`, `hellaswag` |
| 6 | Serving cost | vLLM at 1, 8 and 32 concurrent requests | time to first token, inter-token latency, throughput |

Every task item was reviewed against its source passages before any model was run on it, and 524
of 1,176 generated items survived. Items were rejected only for defects: a wrong or unsupported
gold answer, a correct answer the scorer would mark wrong, or a question that tests general
knowledge or trivia instead of the corpus. None was rejected for being hard. Tasks 2–4 are graded
by a pinned judge (Mistral Large 3,
temperature 0, cached verdicts), with anything a rule can decide (missing citations, empty answers,
the exact refusal phrase) decided by rule first.

### Where it starts

Row zero, from [`results/table.md`](results/table.md) (27 September 2026, on the eval set frozen
that day):

- **The base model** finds the right answer in the passages 85% of the time but cites correctly only
  10% of the time, and answers 88% of unanswerable questions with something invented.
- **The instruct model** has the behaviour (79% of answers correct and backed by their citations,
  99% of unanswerable questions declined), but it knows no more of the domain closed-book: 12%
  against the base model's 14%.
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
sources never resamples the eval; never build SFT data from chunks in `eval/tasks/eval_chunk_ids.txt`.

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

#### Replay slice

To limit forgetting, CPT can mix general text back in. `data/processed/replay.jsonl` is 1.9M Tekken
tokens (10% of train) from FineWeb-Edu, English web text filtered for educational quality, taken as
raw text from the head of its `sample-10BT` stream. Stage 2 runs CPT with and without it and
compares domain val perplexity with the MMLU, GSM8K and HellaSwag regression columns. It is ODC-By
web text, not public domain, so it is rebuilt from Hugging Face rather than committed, and nothing
in the eval comes from it. Caveats (not a random sample, not decontaminated against the regression
benchmarks) are in the corpus card.

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

- **cpt-8b-fsdp2 vs cpt-8b:** 2 GPUs give 2.23x the tokens/s (13,221 vs 5,939); per-step losses differ by 0.038% (median) / 0.171% (max) over 100 steps, the 10-step moving averages by 0.01% on average.

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
<!-- stage2-tables:end -->

**Findings** (all Stage 2 runs). Every delta is against
`base-8b-hf`, the base evaluated through the same vLLM path as the fine-tuned checkpoints, and is
read against the seed floor (`cpt-8b` vs `cpt-8b-seed1`):
- **Domain perplexity moved, but far less than planned.** Held-out documents: -2.3% (95% interval
  over documents -3.0% to -1.8%; the second seed -2.35%), with all 12 val documents improving. The
  pre-registered target was -20%, so this is the rule's "barely moved" branch.
- **The model learns what it reads; a quarter of it transfers.** Perplexity on documents it trained
  on fell 8.3%, on unseen ones 2.3%. By publisher: FEMA -4.3%, NIST -3.8%, FHWA -2.3%, USACE -2.0%,
  NASA -1.3%; USACE is 76% of val tokens, so the pooled figure sits near its rate.
- **A higher LR doesn't help.** At 2e-4 (the rule's prescribed fix), domain val moves another -0.1%
  (interval covers 0), general-text perplexity rises +0.44% more, and the train slice drops 4.1%
  more: more memorising and more forgetting, no transfer. 1e-4 stays.
- **Almost no forgetting.** General-text perplexity +0.2% to +0.4% across the two seeds (bound +3%);
  MMLU -0.2 to -0.3 points, GSM8K -0.7 to -0.8, HellaSwag unchanged.
- **No detectable KPI change.** qa_acc +0.0 to +0.8 points and every other KPI move inside the seed
  floor, which is several points at this eval size (76-210 items per task). 20M tokens of LoRA CPT
  lowers domain perplexity without a knowledge gain large enough to show up in closed-book QA.
- **B: full-parameter learns what it reads, and forgets more.** On the same model and tokens,
  full-parameter CPT (2 x H100, FSDP2) cut perplexity on the documents it trained on by 22% (LoRA:
  8%) and lifted closed-book QA +3.1 and vocab +3.8 points (every KPI item comes from a training
  document), but gained nothing extra on unseen documents (domain val -2.15% vs LoRA's -2.33%) and
  forgot about 3x as much: general-text perplexity +1.2% (LoRA +0.4%), GSM8K -3.0 points (LoRA
  -0.8), MMLU -0.5 (LoRA -0.2/-0.3). Same GPU-hours as LoRA (0.91 vs 0.95). LoRA stays the default:
  "LoRA learns less and forgets less".
- **A: replay is free on the domain and erases the small forgetting, on thin evidence.** Mixing in
  10% FineWeb-Edu left domain perplexity exactly where it was (0.00%) and turned the main run's small
  dip into almost none (MMLU -0.1 instead of -0.3, GSM8K -0.2 instead of -0.8), all inside benchmark
  noise; its general-text perplexity fell 2.2%, but that slice is FineWeb-Edu too, so it measures
  in-distribution training, not protection. It cost 6 points of grounded and citation accuracy.
  Adopted for later stages by the pre-registered rule, because it is cheap; revisited after SFT.
- **Why so small, the working hypothesis:** these are public US federal documents on the open web,
  very likely in the base model's pretraining data already. A first check finds no overall trend of
  base perplexity with document date (51 dated documents, Spearman +0.07, p = 0.60,
  `eval/exposure_check.py`); the four 2024-2026 documents are harder than their publishers' median
  (+6.6%) and the newest two most of all (+10.5%, +15.1%), the direction exposure predicts but on
  n = 4. Unresolved: the corpus has almost no post-cutoff documents to compare against.
- **C: FSDP2 on two GPUs reproduces single-GPU training step for step** (per-step loss within 0.04%,
  median; both runs see the same batches) at 2.23x the tokens/s. More than 2x means the FSDP code
  path is also faster per GPU than the single-GPU one, not a scaling effect (not isolated by a run).
- **An evaluation bug worth more than the training effect.** vLLM 0.29 runs any HF-format Ministral 3
  checkpoint (every fine-tuned one) at the wrong attention temperature: it drops the YaRN config's
  `mscale` keys and scales attention by 1.28. On the base weights that costs 4.8% perplexity, 3.1 MMLU
  points, 6.1 GSM8K points and 19.5 points of grounded accuracy, and it first made CPT look like
  it had wrecked MMLU and grounding. `merge.py` now writes `apply_yarn_scaling: false`, which
  matches transformers and Mistral's native path to five decimals
  ([`notes/contributions.md`](notes/contributions.md) has the repro).

### 4. Results

Every scored checkpoint, from [`results/table.md`](results/table.md) (copied here by
`train/report.py`). The KPI columns come from the frozen eval (judge columns scored locally), the
next seven from lm-eval (5-shot, never the chat template), the last three from
`eval/perplexity.py` (lower is better). Stage 2 rows are base models, scored without `--chat`.

<!-- results-table:start -->
| run | qa_acc | grounded_acc | cite_valid | cite_supported | vocab_recall | halluc_rate | mmlu | mmlu_stem | mmlu_hum | mmlu_soc | mmlu_other | gsm8k | hellaswag | ppl_train | ppl_domain_val | ppl_general_val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-8b | 0.139 | 0.852 | 0.130 | 0.102 | 0.719 | 0.882 | 0.768 | 0.732 | 0.707 | 0.862 | 0.805 | 0.794 | 0.801 | 6.18 | 6.88 | 8.15 |
| instruct-8b | 0.115 | 0.898 | 0.833 | 0.787 | 0.786 | 0.013 | 0.761 | 0.735 | 0.691 | 0.854 | 0.802 | 0.855 | 0.801 |  |  |  |
| cpt-8b | 0.154 | 0.833 | 0.148 | 0.074 | 0.710 | 0.908 | 0.764 | 0.729 | 0.699 | 0.857 | 0.804 | 0.785 | 0.801 | 5.67 | 6.72 | 8.18 |
| cpt-8b-seed1 | 0.146 | 0.796 | 0.102 | 0.037 | 0.700 | 0.934 | 0.765 | 0.727 | 0.699 | 0.861 | 0.809 | 0.786 | 0.800 | 5.69 | 6.72 | 8.17 |
| base-8b-hf | 0.146 | 0.843 | 0.120 | 0.083 | 0.705 | 0.895 | 0.767 | 0.733 | 0.704 | 0.862 | 0.803 | 0.793 | 0.801 | 6.18 | 6.88 | 8.15 |
| cpt-8b-full | 0.177 | 0.889 | 0.148 | 0.102 | 0.743 | 0.921 | 0.762 | 0.729 | 0.692 | 0.860 | 0.806 | 0.763 | 0.798 | 4.82 | 6.73 | 8.25 |
| cpt-8b-replay10 | 0.146 | 0.778 | 0.056 | 0.037 | 0.700 | 0.934 | 0.766 | 0.730 | 0.707 | 0.856 | 0.806 | 0.791 | 0.800 | 5.65 | 6.72 | 7.97 |
<!-- results-table:end -->

_Which stages earned their keep, and which didn't: after Stage 2's ablations and Stage 3._

### 5. Serving
_AWQ quality delta vs bf16; TTFT / ITL / throughput at 1 and 32 concurrent requests._

### 6. What I'd do next
_Honest limitations: judge bias, eval set size, reward hacking observed in GRPO, etc._
