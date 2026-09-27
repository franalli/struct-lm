# struct-lm

Domain adaptation of Ministral 3 8B on public-domain US federal structural-engineering documents
(USACE, FHWA, NIST, FEMA, NASA): CPT -> SFT -> DPO -> GRPO, measured by KPI evals, an lm-eval
regression suite and serving latency. Decision log with the reasoning: `notes/decisions.md`.
Current baselines: `results/table.md`.

## Environment

- `uv sync --extra data --extra dev` on the Mac (data prep, task generation, scoring; add
  `--extra train` to keep the local training deps, or sync removes them). The GPU work
  runs on Modal. `quantize` is a separate extra (llmcompressor conflicts with vLLM 0.29).
- Claude Code's shell has no `python` on PATH: use `.venv/bin/python` (and `.venv/bin/modal`).
- Keys live in `.env` (gitignored): `MISTRAL_API_KEY`, `HF_TOKEN`. Scripts don't read `.env`, so
  load it first: `set -a; . ./.env; set +a`. Never print key values.
- Modal: volume `struct-lm` (mounted at `/vol`, results under `/vol/results`), secrets
  `huggingface` and `mistral`.

| Role | Model | Run name |
|---|---|---|
| Base | `mistralai/Ministral-3-8B-Base-2512` | `base-8b` |
| Instruct | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `instruct-8b` (chat) |
| Task generator and judge | `mistral-large-2512` (pinned: `make_tasks.GEN_MODEL`, `judge.JUDGE_MODEL`) | |

## Commands

### CPT corpus (Mac; Stage 1)

```bash
set -a; . ./.env; set +a                        # HF_TOKEN for the tokenizers and FineWeb-Edu
make data                                       # all steps below, in order
.venv/bin/python data/scripts/download.py       # data/sources.csv -> data/raw/<slug>.pdf; writes sha256 + pages back
.venv/bin/python data/scripts/extract.py        # -> docs_raw.jsonl (header/footer strip, <200-char pages dropped, page offsets)
.venv/bin/python data/scripts/filter.py         # -> docs.jsonl + dropped_samples.jsonl
.venv/bin/python data/scripts/dedup.py          # docs.jsonl in place (exact + paragraph MinHash) + duplicates.jsonl
.venv/bin/python data/scripts/pii.py            # docs.jsonl in place
.venv/bin/python data/scripts/split.py          # -> train.jsonl / val.jsonl (document-level; eval docs stay in train)
.venv/bin/python data/scripts/replay.py         # -> replay.jsonl (FineWeb-Edu, 10% of train tokens)
.venv/bin/python data/scripts/tokenizer_coverage.py  # -> tokenizer_coverage.md (committed)
.venv/bin/python data/scripts/stats.py           # tokens column in sources.csv + corpus card tables for notes/decisions.md
```

- All outputs land in `data/processed/`; each step writes its own section of `stats.json`.
  Committed: `stats.json`, `tokenizer_coverage.md`. `dedup.py` and `pii.py` rewrite `docs.jsonl`
  in place, so after a change to any step rerun from `filter.py` (or `make data`).
- Stage 2 reads `train.jsonl`, `val.jsonl` and `replay.jsonl` from the Modal volume:
  `modal volume put struct-lm data/processed data/processed` (not done in Stage 1).
- New sources: `crawl_index.py <index url or saved .html> --pattern REGEX --publisher P` appends
  rows to `data/sources.csv`, or add rows by hand. USACE and FEMA block scripts (Akamai 403): save
  the index page / PDFs from a browser (Chrome DevTools MCP works) into `data/raw/` (as
  `<slug>.pdf` or the URL's filename) and rerun `download.py`. Hand-check every `copyright_flags` entry `extract.py` prints (rule 1).
- Read `dropped_samples.jsonl` after any filter change; the tuning rationale is in `filter.py`.

### Eval tasks (Mac)

```bash
.venv/bin/python data/scripts/extract.py --chunks   # -> data/processed/chunks.jsonl (eval docs only; frozen)
set -a; . ./.env; set +a
.venv/bin/python eval/make_tasks.py           # -> eval/tasks/*.jsonl (Mistral API, disk-cached)
```

`chunks.jsonl` is built only from the documents in `eval/tasks/eval_chunk_ids.txt`, so corpus
expansion can't resample the reviewed tasks; `--chunks` must stay byte-identical (`cmp`).
After any edit to `make_tasks.py`, rerun it and diff `eval/tasks/` against the reviewed version.
Byte-identical output needs no re-review; any changed item must be hand-reviewed.

### Generation, lm-eval, latency (Modal)

```bash
M=.venv/bin/modal
# KPI generations: always --generate-only (scoring is local); Instruct always --chat
$M run --detach eval/modal_app.py --which kpi --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b --generate-only
$M run --detach eval/modal_app.py --which kpi --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --run-name instruct-8b --chat --generate-only
# lm-eval regression (MMLU / GSM8K / HellaSwag, 5-shot): never --chat
$M run --detach eval/modal_app.py --which lm --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b
$M run --detach eval/modal_app.py --which lm --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --run-name instruct-8b
# Serving latency (concurrency 1 / 8 / 32)
$M run --detach eval/modal_app.py --which latency --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b
$M run --detach eval/modal_app.py --which latency --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --run-name instruct-8b
```

### Pull results, then score locally

Pull per run, never all of `results/`: `results/table.md` and `results/judge_cache.jsonl` are kept
locally (a Modal run without `--generate-only` would start its own `table.md` on the volume).

```bash
$M volume get --force struct-lm results/runs/<run> results/runs/
$M volume get --force struct-lm results/lm_eval/<run> results/lm_eval/
$M volume get --force struct-lm results/bench results/

set -a; . ./.env; set +a
.venv/bin/python eval/run_eval.py --run-name base-8b --rescore --lm-eval-dir results/lm_eval \
  --model mistralai/Ministral-3-8B-Base-2512
.venv/bin/python eval/run_eval.py --run-name instruct-8b --rescore --lm-eval-dir results/lm_eval \
  --chat --model mistralai/Ministral-3-8B-Instruct-2512-BF16
```

- Always pass `--model` (and `--chat` for chat checkpoints) when scoring: generate-only runs write
  no `metrics.json`, so otherwise the row records `model: null`.
- `table.md` is append-only: delete the superseded row for that run by hand.
- If the output reports `judge_failed` (Mistral 429s), rerun the same command: failed verdicts
  aren't cached, and cached ones are free.
- Never delete `results/judge_cache.jsonl`. Editing a rubric re-judges everything it grades.

Smoke test: `--which kpi --limit 5 --no-judge`. Don't use `make eval`: that target is stale (it
passes flags `run_eval.py` doesn't have).

## Decisions (rules to keep)

1. **Copyright:** only public-domain US federal documents. ASCE 7, the AISC manual and the 2025
   NSBA handbook are excluded. Check any new source before adding it to `sources.csv`.
2. **Chat format:** lm-eval never uses the chat template, for any checkpoint; the KPI eval always
   runs chat checkpoints (Instruct, SFT/DPO/GRPO) with `--chat`, base models without. Enforced:
   `run_lm_eval.sh` exits on `CHAT=1`, `modal_app.py` and `run_eval.py` refuse an Instruct model
   without `--chat`, `merge_lm_eval` skips chat-template results files. (lm-eval renders the
   template to text and re-encodes it, so Mistral control tokens arrive as ordinary text.)
3. **Tokenizer:** Tekken via mistral-common, `tokenizer_mode=mistral` everywhere (vLLM, lm-eval,
   `extract.py`), and `limit_mm_per_prompt={"image": 0}`.
4. **Generation on Modal, judging local:** Modal KPI runs always use `--generate-only`.
5. **One grounded prompt for all models** (`prompts.GROUNDED_FORMAT`, a one-line format example).
   A worked example with its own passages made Instruct refuse 130/131.
6. **Chat models on the one-line tasks** (`domain_qa`, `vocab`): `prompts.CHAT_GEN` drops the `"\n"`
   stop and `scorers.answer_line` keeps the first line with content (Instruct opens with a
   `**Term: x**` header). Base models keep `GEN`.
7. **Metrics:** `qa_acc` (rules), `grounded_acc` (judge, correct per gold, citations ignored),
   `cite_valid` (rules), `cite_supported` (judge, correct and backed by cited passages),
   `vocab_recall` (judge), `halluc_rate` (answered an unanswerable question; lower is better),
   plus `mmlu` / `gsm8k` (strict-match) / `hellaswag` (acc_norm).
8. **Judge:** decide by rule anything a rule can decide, before the judge sees it: empty or
   citation-only answers, answers citing no provided passage, the exact abstain phrase, and vocab
   outputs with no definition line. Give the judge only what the verdict depends on (the
   adversarial rubric sees no passages). After any rubric change, hand-check verdicts against the
   gold text before trusting the numbers.
9. **Eval tasks** (664 items, all hand-reviewed): filters run post-cap and only remove items;
   per-task RNG streams; supplementary grounded/adversarial items have ids >= 501; human rejects
   live in `eval/tasks/rejects.jsonl` (task + match + reason).
10. **Training:** LoRA targets the language model only, with the regex single-quoted in YAML;
    `train/merge.py` copies `tekken.json` and `processor_config.json`. SFT/DPO/GRPO data and chat
    template work are deferred until the user asks.

## Known gaps

- `train/cpt.py` still reads a pre-packed `data/packed/cpt`; Stage 2 switches it to
  `data/processed/{train,val,replay}.jsonl` with TRL packing (`SFTConfig(packing=True, max_length=4096)`).
- Pyright errors about `prompts` / `scorers` / `judge` / `vllm` imports in `eval/` are false
  positives (`sys.path` imports; vLLM and lm-eval are only installed in the Modal image).
- `results/lm_eval/_invalid/` holds an excluded chat-template lm-eval run (see its README).
