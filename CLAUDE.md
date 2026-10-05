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
- The Mistral key allows 30 requests a minute (tokens are no constraint), shared by `make_tasks.py`
  (paced: `--rpm`), the judge and any synthesis, so ~1,800 calls an hour in total; don't run two
  API-heavy jobs at once.
- Modal: volume `struct-lm` (mounted at `/vol`, results under `/vol/results`), secrets
  `huggingface` and `mistral`.
- GPU work: always ask the user before launching any Modal GPU job (`modal run` of
  `train/modal_train.py` or `eval/modal_app.py`, smoke tests included), even when a plan or the
  next step calls for it. Say which command, how many GPUs, and the rough duration. An approval
  covers that launch only, not later ones. `modal volume get/put` needs no confirmation.
- Git: never create or check out a new branch unless the user says to. Commit on the current
  branch (`main`) when asked.

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
.venv/bin/python data/scripts/split.py          # -> train.jsonl / val.jsonl (document-level; eval pool stays in train)
.venv/bin/python data/scripts/replay.py         # -> replay.jsonl (FineWeb-Edu, 10% of train tokens) + general_val.jsonl (500k, last shard)
.venv/bin/python data/scripts/tokenizer_coverage.py  # -> tokenizer_coverage.md (committed)
.venv/bin/python data/scripts/stats.py           # tokens column in sources.csv + corpus card tables for notes/decisions.md
```

- All outputs land in `data/processed/`; each step writes its own section of `stats.json`.
  Committed: `stats.json`, `tokenizer_coverage.md`. `dedup.py` and `pii.py` rewrite `docs.jsonl`
  in place, so after a change to any step rerun from `filter.py` (or `make data`).
- Stage 2 reads `train.jsonl`, `val.jsonl`, `replay.jsonl` and `general_val.jsonl` from the Modal
  volume: `modal volume put --force struct-lm data/processed data/processed` after any data change.
- `split.py` reports the 150-step rule's batch (`stats.json` split.seqs_per_step: 32 at ~20M
  tokens, 64 at ~40M); set `gradient_accumulation_steps` in `train/configs/cpt*.yaml` to match
  (`cpt.py` refuses an epoch outside 120-190 steps).
- New sources: `crawl_index.py <index url or saved .html> --pattern REGEX --publisher P` appends
  rows to `data/sources.csv`, or add rows by hand. USACE and FEMA block scripts (Akamai 403): save
  the index page / PDFs from a browser (Chrome DevTools MCP works) into `data/raw/` (as
  `<slug>.pdf` or the URL's filename) and rerun `download.py`. Hand-check every `copyright_flags` entry `extract.py` prints (rule 1).
- Read `dropped_samples.jsonl` after any filter change; the tuning rationale is in `filter.py`.

### Eval tasks (Mac)

```bash
.venv/bin/python data/scripts/extract.py --chunks   # -> data/processed/chunks.jsonl (eval pool only; frozen)
set -a; . ./.env; set +a
.venv/bin/python eval/make_tasks.py           # -> eval/tasks/*.jsonl (Mistral API, disk-cached)
.venv/bin/python eval/sft_split.py            # -> eval/tasks/sft_seen_chunks.txt (seen/unseen halves)
.venv/bin/python -m pytest tests/             # qa_rules, few-shot guard; with .env loaded, v1/v2/v3 rebuild from the LLM cache
```

- `--task-version 1` rebuilds the 2026-09-27 eval (130-item domain_qa, `results/table_v1.md`) byte
  for byte. `2` (default) is the grown set: after rejects it tags `answer_kind`, removes layout
  locators (`qa_rules.is_locator` -> `locators.jsonl`) and holds identifiers to 20% of the set
  (`held_back.jsonl`). Reviewer tag corrections live in `eval/tasks/answer_kinds.jsonl`. v2 tags
  `fewshot_passage_overlap` on qa-0003 / qa-0056 (same passage as a few-shot item, other facts);
  `3` is v2 with the few-shot split off by passage (322 items, v2 ids kept), for the next rebuild.

`chunks.jsonl` is built only from the documents pinned in `eval/tasks/eval_docs.txt` (the 234
train documents when the eval was frozen), so corpus expansion can't resample the reviewed tasks;
`--chunks` must stay byte-identical (`cmp`). `split.py` keeps every pinned document in train.
Sampling caps source chunks at `--per-doc 6` per document per task.
After any edit to `make_tasks.py`, rerun it and diff `eval/tasks/` against the reviewed version.
Byte-identical output needs no re-review; any changed item must be reviewed (rule 9) against
`notes/eval_review_rubric.md`. A task file that grows makes every earlier run stale for that task:
`run_eval.py --rescore` refuses generations that miss items (`--allow-partial` scores the subset
and must not go into `table.md`), so regenerate the affected task for every compared checkpoint.

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

Frontier closed-book reference (Mistral API, no GPU, ~15 min at 30 requests a minute; domain_qa only):
`.venv/bin/python eval/api_eval.py --model mistral-large-2512 --run-name mistral-large-3`, then
`run_eval.py --run-name mistral-large-3 --rescore --chat --allow-partial --model "mistral-large-2512 (Mistral API)"`.

### Pull results, then score locally

**Always pull every Modal run's results into the repo as soon as it finishes** (training logs,
perplexity, lm-eval, KPI generations, latency), score it locally, and write its row to
`results/table.md`. A result that exists only on the volume doesn't count as collected.

Pull per run, never all of `results/`: `results/table.md` and `results/judge_cache.jsonl` are kept
locally (a Modal run without `--generate-only` would start its own `table.md` on the volume).

```bash
$M volume get --force struct-lm results/runs/<run> results/runs/          # train_log, train_summary, generations
$M volume get --force struct-lm results/ppl/<run>.json results/ppl/       # Stage 2+: perplexity
$M volume get --force struct-lm results/lm_eval/<run> results/lm_eval/
$M volume get --force struct-lm results/bench/<run>.json results/bench/

set -a; . ./.env; set +a
.venv/bin/python eval/run_eval.py --run-name base-8b --rescore --lm-eval-dir results/lm_eval \
  --model mistralai/Ministral-3-8B-Base-2512
.venv/bin/python eval/run_eval.py --run-name instruct-8b --rescore --lm-eval-dir results/lm_eval \
  --chat --model mistralai/Ministral-3-8B-Instruct-2512-BF16
```

- Always pass `--model` (and `--chat` for chat checkpoints) when scoring: generate-only runs write
  no `metrics.json`, so otherwise the row records `model: null`.
- `table.md` is append-only: delete the superseded row for that run by hand. Its first line
  records the task sizes it was started on; `run_eval.py` refuses rows scored on other sizes or on
  part of a task. When a task file changes, freeze the table as `results/table_vN.md` (v1 = the
  Stage 2 write-up's 130-item table) and start a new one.
- Regenerate one task with `--tasks domain_qa` (Modal `--tasks domain_qa`): the run's other
  generations are kept, keyed by (task, id), and `generations_meta.json` records which run produced
  each task. Push the local `generations.jsonl` to the volume first and diff the pulled file's
  other tasks against it. A run that can't be regenerated gets that task blank with
  `--allow-partial`.
- A `--results-dir` other than `results/` needs `--judge-cache` (normally
  `results/judge_cache.jsonl`), or every judged item is paid for again.
- If the output reports `judge_failed` (Mistral 429s), rerun the same command: failed verdicts
  aren't cached, and cached ones are free.
- Never delete `results/judge_cache.jsonl`. Editing a rubric re-judges everything it grades.
- **Then update the README's figures and tables every time new results land:** rerun the stage's
  report (Stage 2: `train/report.py` -> `results/curves/cpt.png`, `cpt_ppl.png`,
  `results/train_runs.md`), which overwrites the plots in place and rewrites the README's generated
  blocks (`<!-- stage2-tables:... -->` in Training, `<!-- results-table:... -->` in Results,
  `<!-- serving-table:... -->` in Serving: never edit inside them by hand). Embed any new figure in the README's write-up and update the prose
  findings next to it. Every stage shows its train/val loss curves and its headline metric
  (perplexity for CPT) as plots, not only tables.

Smoke test: `--which kpi --limit 5 --no-judge`. Don't use `make eval`: that target is stale (it
passes flags `run_eval.py` doesn't have).

### Stage 2: CPT (Modal)

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

- Override values go through YAML: write floats with a dot (`2.0e-4`; PyYAML reads `2e-4` as a
  string), and `no`/`yes`/`on`/`off` stay strings (`common.parse_config`).
- A delta between runs counts only beyond the seed floor (`cpt-8b` vs `cpt-8b-seed1`) and, for
  perplexities, with a 95% document-bootstrap interval excluding 0 (`eval/ppl_compare.py`).
- `run_lm_eval.sh` passes model_args as JSON with `limit_mm_per_prompt={"image": 0}` (from Stage 2;
  a merged checkpoint otherwise fails vLLM's dummy-image profiling).

- Steps: `train` (`cpt.py`) -> `merge` (`merge.py` into `/vol/checkpoints/<run>`) -> `ppl`
  (`eval/perplexity.py`), `eval` (`modal_app.py`'s lm_eval and kpi_eval `--generate-only`) and
  `latency` in parallel. `latency` is opt-in (`--steps ...,latency`): it measures the architecture
  and serving setup, not the weights, so it runs per deployed checkpoint, not per ablation (cost). The volume mirrors the repo (cwd `/vol`): `checkpoints/_train/<run>` holds the
  adapter and trainer checkpoints, and relaunching a run resumes from the newest.
- Per run on the volume: `results/runs/<run>/train_log.jsonl` (every step) and `train_summary.json`,
  `results/ppl/<run>.json`, plus the usual lm_eval / runs / bench outputs for evaluated runs.
- Base rows get perplexity by rescoring them with `--ppl-dir results/ppl` too.

### Stage 3: SFT data (Mac)

```bash
set -a; . ./.env; set +a                    # MISTRAL_API_KEY (~6,000 calls at 30 a minute), HF_TOKEN
make sft-data                               # all steps below, in order
.venv/bin/python data/scripts/sft_pool.py        # A1 -> data/sft/work/pool.jsonl (seen half + ~900 allowed chunks)
.venv/bin/python data/scripts/sft_questions.py   # A2 questions (Large 3, 0.7; title + passage only)
.venv/bin/python data/scripts/sft_filter.py      # A3 rules, dedup, rule 1/2, caps, passage sets -> questions_kept.jsonl
.venv/bin/python data/scripts/sft_answers.py     # A4 completions (Large 3 / Medium 3.5 25%), paraphrases, Evol-Instruct
.venv/bin/python data/scripts/sft_judge.py       # A5 verifier + rubric judge + one revise round
.venv/bin/python data/scripts/sft_replay.py      # 500 Tulu 3 SFT examples (downloads 1.4 GB once)
.venv/bin/python data/scripts/sft_assemble.py    # A6/A7 -> data/sft/{train,sft_val}.jsonl, review.md, SHA256SUMS
.venv/bin/python eval/contamination.py --only sft # section 6 of results/contamination.md
# quality: full-passage reads of the generated records (packets in data/sft/work/)
.venv/bin/python data/scripts/sft_audit.py filter-packets   # pass 1: every closed-book record, 10 packets
.venv/bin/python data/scripts/sft_audit.py filter-packets 2 # pass 2: pass-1 "minor" records, stricter defect line
.venv/bin/python data/scripts/sft_audit.py filter-merge     # -> data/sft/closed_book_filter.jsonl (assembly drops defects)
.venv/bin/python data/scripts/sft_audit.py sample [N]       # audit round N sample -> report -> data/sft/audit.md
```

- Every call is cached in `data/sft/.cache/llm_cache.jsonl` (gitignored, with each prompt), so a
  rerun pays only for prompts that changed. Each step writes its section of `data/sft/stats.json`.
- `sft_pool.py --pilot N` builds the first N chunks of each kind (the head of the full pool, same
  assignments): run it through A2-A5 and read the outputs before a full run.
- After any change to a step, rerun from that step, then `sft_assemble.py`, the contamination
  section and `pytest tests/test_sft_data.py`; a changed `train.jsonl` changes `SHA256SUMS` and
  needs the hand-read in `review.md` redone.
- The Mistral judge passes defects a full-passage read catches (audit 2026-10-05: 18% of judged-kept
  records). A rebuild that adds or changes closed-book records needs `filter-packets`, a
  reader per packet with the audit's closed-book rubric, then pass 2 over what pass 1 kept as
  "minor" (the line that matters: a wrong framing, overclaim, scope or example value is a defect
  even when the answer value is right), then `filter-merge`: assembly refuses a closed-book record
  without a verdict. Then re-audit a fresh sample (`sample N` names the round).

## Decisions (rules to keep)

1. **Copyright:** only public-domain US federal documents. ASCE 7, the AISC manual and the 2025
   NSBA handbook are excluded. Check any new source before adding it to `sources.csv`.
2. **Chat format:** lm-eval never uses the chat template, for any checkpoint; the KPI eval always
   runs chat checkpoints (Instruct, SFT/DPO/GRPO) with `--chat`, base models without. Enforced:
   `run_lm_eval.sh` exits on `CHAT=1`, `modal_app.py` and `run_eval.py` refuse an Instruct model
   without `--chat`, `merge_lm_eval` skips chat-template results files. (lm-eval renders the
   template to text and re-encodes it, so Mistral control tokens arrive as ordinary text.)
3. **Tokenizer and vLLM loading:** Tekken via mistral-common, `tokenizer_mode=mistral` everywhere
   (vLLM, lm-eval, `extract.py`), `limit_mm_per_prompt={"image": 0}`, and `config_format=hf`
   (`run_eval.py`, `run_lm_eval.sh`, `serve_vllm.sh`): without it the hub base loads through vLLM's
   Mistral-native implementation and merged checkpoints through the HF one, which confounds every
   eval delta between them. Compare Stage 2+ rows against `base-8b-hf` (the base re-saved through
   `merge.py`: `modal_train.py --model <hub id> --run-name base-8b-hf --steps merge,eval`), not the
   Stage 0 `base-8b`; `config_format=hf` can't load the hub repo itself (its
   `consolidated.safetensors`). Every merged config carries `"apply_yarn_scaling": false`
   (`merge.py`): vLLM 0.29 otherwise applies YaRN attention scaling the model doesn't use (HF path
   perplexity 7.23 vs 6.89). Check any new serving path with `eval/vllm_ppl.py` against
   `perplexity.py`'s `ppl_val_slice` before trusting its evals.
4. **Generation on Modal, judging local:** Modal KPI runs always use `--generate-only`.
5. **One grounded prompt for all models** (`prompts.GROUNDED_FORMAT`, a one-line format example).
   A worked example with its own passages made Instruct refuse 130/131.
6. **Chat models on the one-line tasks** (`domain_qa`, `vocab`): `prompts.CHAT_GEN` drops the `"\n"`
   stop and `scorers.answer_line` keeps the first line with content (Instruct opens with a
   `**Term: x**` header). Base models keep `GEN`.
7. **Metrics:** `qa_acc` (rules; also `qa_num` / `qa_text` by answer type, and `qa_seen` /
   `qa_unseen`, `vocab_seen` / `vocab_unseen` by `sft_seen_chunks.txt`: report Stage 3+ knowledge
   gains per half, never pooled), `grounded_acc` (judge, correct per gold, citations ignored),
   `cite_valid` (rules), `cite_supported` (judge, correct and backed by cited passages),
   `vocab_recall` (judge), `halluc_rate` (answered an unanswerable question; lower is better),
   plus `mmlu` and its four groups / `gsm8k` (strict-match) / `hellaswag` (acc_norm), and from
   Stage 2 `ppl_train` / `ppl_domain_val` / `ppl_general_val` (`eval/perplexity.py`, lower is
   better; `ppl_train` is measured on a train slice, for base too) and `ppl_postcutoff` (the 13
   2026 reports, `--only postcutoff`). `gold_lp` / `gold_lp_seen` / `gold_lp_unseen`: mean
   log-probability of the gold QA answer in nats per item (`eval/gold_lp.py`, computed in the
   generation engine; Instruct in chat format, not comparable to base rows). It scores the answer
   plus the end token the prompt uses after answers ("\n\n"; a lone "\n" penalised CPT, fixed
   2026-10-04); `run_eval.py --gold-lp-only` recomputes it in saved generations without touching
   outputs. Read it paired per item against the base, with the seed gap (0.007 nats on v2).
8. **Judge:** decide by rule anything a rule can decide, before the judge sees it: empty or
   citation-only answers, answers citing no provided passage, the exact abstain phrase, and vocab
   outputs with no definition line. Give the judge only what the verdict depends on (the
   adversarial rubric sees no passages). After any rubric change, hand-check verdicts against the
   gold text before trusting the numbers.
9. **Eval tasks** (v2, 2026-10-04: domain_qa 325, grounded 108, vocab 210, adversarial 76 = 719 items +
   3 few-shot, every one reviewed; v1 of 2026-09-27 had domain_qa 130, frozen before Stage 2,
   sampled from the 234 train documents pinned in `eval/tasks/eval_docs.txt`): filters run post-cap
   and only remove items; per-task RNG streams; supplementary grounded/adversarial/domain_qa items
   have ids >= 501 and are sampled after every other task, and the second domain_qa supplement
   (2026-10-04) ids >= 1001 after that; rejects live in
   `eval/tasks/rejects.jsonl` (task + match + reason). Review rejects only for defects (wrong gold,
   correct answer scored wrong, wrong answer scored right, general knowledge/trivia/one example's
   value), never for difficulty, and finishes before any model generation.
10. **Training:** LoRA targets the language model only, with the regex single-quoted in YAML (full
    fine-tuning freezes the vision tower and projector). CPT windows come from `train/packing.py`
    (BOS + document + EOS as ids, 4,096-token windows), never TRL's packing, which truncates each
    document and appends EOS as text. Effective batch: the 150-step rule. `train/merge.py` copies
    the base's non-weight files (never `params.json` / `consolidated.safetensors`) and checks the
    weight names match the base. SFT data may draw on eval chunks only from
    `eval/tasks/sft_seen_chunks.txt` (`eval/sft_split.py`), never on any other chunk in
    `eval_chunk_ids.txt`, and never reuses an eval question. Enforced by `data/scripts/sft_guard.py`
    (the SFT builder's only reader of `eval/tasks/`, never imported by a step that calls the LLM),
    which also keeps out every chunk on or next to an unseen eval chunk's page, and checked by
    `tests/test_sft_data.py` and `contamination.py --only sft`. DPO/GRPO data and chat template work
    are deferred until the user asks.
11. **Reference model = the previous stage, not the base:** with LoRA and `ref_model=None`, TRL's
    reference is the adapter-disabled `init_from` checkpoint, so DPO's is the SFT checkpoint and
    GRPO's the DPO checkpoint. Each stage's KL term (DPO's `beta`, GRPO's logged `kl`) measures drift
    from the previous stage, not from the base. Drift from the base shows only in the eval rows.
    `grpo.yaml` has `beta: 0.0` (no KL term, no reference) until raised.
12. **Checkpoints:** no checkpoint with a row in a results table is deleted until that stage's
    write-up is frozen, and adapters (`checkpoints/_train/<run>`) are never deleted; merged LoRA
    checkpoints are reproducible from them, full-parameter ones are not.

## Known gaps

- The `gen_qa` prompt still shows the chunk id (`slug:p12:c0`), so the generator sees the page.
  The locator rule removes page/table/figure items at assembly, and none came from the page so
  far. Drop the id from the prompt only in a from-scratch rebuild, since it changes every cached
  generation, the frozen items included.
- Pyright errors about `prompts` / `scorers` / `judge` / `vllm` imports in `eval/`, and about
  `common` / `packing` / `modal_app` imports in `train/`, are false positives (`sys.path` imports;
  vLLM and lm-eval are only installed in the Modal image).
- SFT/DPO/GRPO are not wired into `train/modal_train.py` yet, and their configs still say
  `report_to: wandb` (no W&B secret exists; Stage 2 logs to `results/runs/<run>/train_log.jsonl`).
  `sft.yaml` starts from `checkpoints/cpt-8b-replay10`; `tf32: true` is on from Stage 3.
- `train/sft.py` still expects `{"messages": [...]}` with `assistant_only_loss`, which needs a chat
  template with generation markers that no checkpoint here has. The SFT set is conversational
  `prompt` / `completion` (TRL masks the prompt itself), and training must render it as
  mistral-common does for the eval's `--chat` path (`[INST]...[/INST]answer</s>`, no system
  prompt): Part B of Stage 3.
- Modal's H100 price in `train/report.py` (`--usd-per-gpu-hour`, default 3.95) is unverified: check
  modal.com/pricing before quoting dollars.
- `results/lm_eval/_invalid/` holds an excluded chat-template lm-eval run (see its README).
