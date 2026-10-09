# data/: CPT corpus (Stage 1) and SFT data (Stage 3)

`extract.py --chunks` and `split.py` feed the frozen eval (pinned documents in
`eval/tasks/eval_docs.txt`): read `eval/CLAUDE.md` before changing either.

## CPT corpus (Mac; Stage 1)

```bash
set -a; . ./.env; set +a                        # HF_TOKEN for the tokenizers and FineWeb-Edu
make data                                       # Makefile DATA_STEPS, in order; each script's docstring says what it writes
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

## Stage 3: SFT data (Mac)

```bash
set -a; . ./.env; set +a                    # MISTRAL_API_KEY (~6,000 calls at 30 a minute), HF_TOKEN
make sft-data                               # Makefile SFT_STEPS + contamination --only sft. A1 sft_pool, A2 sft_questions,
                                            # A3 sft_filter, A4 sft_answers, A5 sft_judge, A6/A7 sft_assemble
# quality: full-passage reads of the generated records (packets in data/sft/work/)
.venv/bin/python data/scripts/sft_audit.py read-packets LABEL # records assembly left unread -> packets
#   each packet is read with data/sft/read_rubrics.md -> work/read_verdicts_<packet>.jsonl
.venv/bin/python data/scripts/sft_audit.py read-merge         # -> data/sft/read_filter.jsonl; rerun sft_assemble
.venv/bin/python data/scripts/sft_audit.py sample [N]         # audit round N sample -> report -> data/sft/audit.md
```

- Every call is cached in `data/sft/.cache/llm_cache.jsonl` (gitignored, with each prompt), so a
  rerun pays only for prompts that changed. Each step writes its section of `data/sft/stats.json`.
- `sft_pool.py --pilot N` builds the first N chunks of each kind (the head of the full pool, same
  assignments): run it through A2-A5 and read the outputs before a full run.
- After any change to a step, rerun from that step, then `sft_assemble.py`, the contamination
  section and `pytest tests/test_sft_data.py`; a changed `train.jsonl` changes `SHA256SUMS` and
  needs the `review.md` read redone.
- The Mistral judge passes defects a full-passage read catches (audit 2026-10-05: 18% of judged-kept
  records), so every closed-book record, definition and grounded answer, every hard-negative abstain
  set and every record of a document in `sft_common.REGENERATED_DOCS` must be read. A verdict holds for the
  content it read (`fingerprint`); `sft_assemble.py` leaves the rest out and lists them in
  `work/unread.jsonl`. Loop: assemble -> `read-packets` -> read -> `read-merge` -> assemble, until
  `stats.json` assemble.unread is 0 (`tests/test_sft_data.py` checks it). The line that matters in
  the rubric: a wrong framing, overclaim, scope or example value is a defect even when the answer
  value is right. Then re-audit a fresh sample (`sample N` names the round).
