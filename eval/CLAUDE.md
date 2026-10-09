# Eval tasks (Mac)

```bash
.venv/bin/python data/scripts/extract.py --chunks   # -> data/processed/chunks.jsonl (eval pool only; frozen)
set -a; . ./.env; set +a
.venv/bin/python eval/make_tasks.py           # -> eval/tasks/*.jsonl (Mistral API, disk-cached)
.venv/bin/python eval/sft_split.py            # -> eval/tasks/sft_seen_chunks.txt (seen/unseen halves)
.venv/bin/python -m pytest tests/             # qa_rules, few-shot guard; with .env loaded, v1/v2/v3 rebuild from the LLM cache
```

- `--task-version 1` rebuilds the 2026-09-27 eval (130-item domain_qa, `results/table_v1.md`) byte
  for byte. `2` is the grown set: after rejects it tags `answer_kind`, removes layout
  locators (`qa_rules.is_locator` -> `locators.jsonl`) and holds identifiers to 20% of the set
  (`held_back.jsonl`). Reviewer tag corrections live in `eval/tasks/answer_kinds.jsonl`. v2 tags
  `fewshot_passage_overlap` on qa-0003 / qa-0056 (same passage as a few-shot item, other facts).
  `3` (default) is v2 with the few-shot split off by passage (322 items, v2 ids kept; qa-1143 moves
  to `held_back.jsonl`): the committed set from Stage 3 on (`results/table_v2.md` froze v2's table).

`chunks.jsonl` is built only from the documents pinned in `eval/tasks/eval_docs.txt` (the 234
train documents when the eval was frozen), so corpus expansion can't resample the reviewed tasks;
`--chunks` must stay byte-identical (`cmp`). `split.py` keeps every pinned document in train.
Sampling caps source chunks at `--per-doc 6` per document per task.
After any edit to `make_tasks.py`, rerun it and diff `eval/tasks/` against the reviewed version.
Byte-identical output needs no re-review; any changed item must be reviewed (rule 9) against
`notes/eval_review_rubric.md`. A task file that grows makes every earlier run stale for that task:
`run_eval.py --rescore` refuses generations that miss items (`--allow-partial` scores the subset
and must not go into `table.md`), so regenerate the affected task for every compared checkpoint.
