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

Corpus build (`make data`), its per-step rules and adding sources: `data/CLAUDE.md` (loads with
files under `data/`).

### Eval tasks (Mac)

Task generation (`make_tasks.py`, `sft_split.py`, `extract.py --chunks`), task versions and the
frozen-eval rules (rule 9): `eval/CLAUDE.md`.

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
`results/table.md`. A result that exists only on the volume doesn't count as collected. Then run
`data/scripts/sft_replay_fetch.py --results`: Tülu 3 rows' prompt ids and references are committed
as sha256 only (third-party text stays out of git; `make sft-replay` rebuilds the inputs).

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
- **Then update the write-up's figures and tables every time new results land:** rerun
  `train/report.py` (`make report`). It overwrites the plots in `results/curves/` and
  `results/train_runs.md`, and rewrites every generated block where `report.BLOCK_FILES` puts it:
  the README's `headline-*` tables, each stage's `<!-- stageN-tables -->` in `docs/stageN.md`,
  `results-table` in `docs/results.md`, `serving-table` in `docs/stage6.md`, and DEPLOY.md's
  `deploy-*`. Never edit inside a block by hand; a missing marker stops the report. Embed any new
  figure in its stage doc and update the prose findings next to it. Every stage shows its
  train/val loss curves and its headline metric (perplexity for CPT) as plots, not only tables.
- **The README** (Stage 7) holds seven parts in under 3,000 prose words; the stage write-ups live
  in `docs/`. Every number in its prose must trace: `make audit` (`train/readme_audit.py`) passes
  a number with 3+ significant digits found in a generated block, anything else only through a row
  of `notes/readme_numbers.tsv` naming a file that contains it. That file must be data (`results/`,
  the data sets, `eval/tasks/`, often via `eval/summary_stats.py`), code or a config, or the
  decision log for a registered rule or a reading ruling; never the docs, this file or DEPLOY.md. The stage diagrams' labels
  (`docs/diagrams/src/*.json`) are checked the same way, with rows whose `in` column names the
  diagram. Add the row when you add the number.
- **`make reproduce-score`** checks every committed number with no GPU and no API key:
  `eval/rescore_all.py` rescores each `results/table.md` row in `results/_rescore/` and compares,
  then the strict checker, pass@k and the report rerun, and `git diff` must be clean.

Smoke test: `--which kpi --limit 5 --no-judge`.

### Stage 2: CPT (Modal)

Launch commands for every Stage 2 run (smoke tests, main run, ablations, seed/LR-up runs,
memorisation probe): the `stage2-cpt` skill (`.claude/skills/stage2-cpt/SKILL.md`).

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

Build (`make sft-data`), the full-passage read loop and its rules: `data/CLAUDE.md`.

### Stage 3: SFT training and eval (Modal)

```bash
M=.venv/bin/modal
# data on the volume after any SFT data change (the run refuses files that don't match SHA256SUMS);
# in a fresh clone, rebuild the files first: make sft-replay (git keeps data/sft/hosted/ only)
for f in train.jsonl sft_val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/sft/$f data/sft/$f; done
$M volume put --force struct-lm data/dpo/prompts.jsonl data/dpo/prompts.jsonl   # the eos and dpo_probe sample jobs
# smoke: no-op control, 1 step on 32 records, merge, merge check, vLLM template prompt ids
$M run train/modal_train.py --config train/configs/sft.yaml --run-name smoke-sft --smoke --chat \
  --steps noop,train,merge,mergecheck,sample --sample-jobs template
$M run train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-base --steps noop \
  --overrides "model.init_from=checkpoints/base-8b-hf"            # CPU: no-op control of the other start
# runs (train only), then B4's rule on the epoch-end val_loss, then merge the chosen epoch + eval
$M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-cpt --steps train
$M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-base --steps train \
  --overrides "model.init_from=checkpoints/base-8b-hf"
$M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name sft-from-cpt-seed1 --steps train \
  --overrides "training.seed=1 training.data_seed=1"
$M run --detach train/modal_train.py --run-name sft-from-cpt --merge-from checkpoint-77 --chat \
  --steps merge,mergecheck,ppl,eval,latency,sample      # checkpoint-N: the epoch B4 picked
# or one unattended chain: --merge-from b4 applies B4 to the run's own loss curve after training
# (results/runs/<run>/b4.json, written before the merge)
$M run --detach train/modal_train.py --config train/configs/sft.yaml --run-name <run> \
  --merge-from b4 --chat --steps train,merge,mergecheck,ppl,eval,latency,sample
$M run train/modal_train.py::digest --run-name sft-from-cpt   # CPU: checkpoint sha256, after the evals too
$M run --detach eval/modal_app.py --which sample --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --run-name instruct-8b --chat --sample-jobs diversity --config-format auto
# pull (runs/<run> includes samples/ and merge_check.json; results/noop/), then score with --chat
.venv/bin/python eval/run_eval.py --run-name sft-from-cpt --rescore --chat --lm-eval-dir results/lm_eval \
  --ppl-dir results/ppl --model /vol/checkpoints/sft-from-cpt
.venv/bin/python eval/diversity.py score sft-from-cpt     # -> results/diversity/<run>.json
```

### Stage 4: DPO on verifiable preferences (data on the Mac, sampling and training on Modal)

Rules, the judge benchmark's failure and the amendment: `notes/decisions.md` (2026-10-08).
Stage4-final is `checkpoints/dpo-strict` (seed 0, strict closed-book labels, 2026-10-09);
`grpo.yaml` starts from it. `dpo` / `dpo-seed1` are the as-run rows on `same_fact` labels (79 of
458 closed-book chosen wrong under the strict checker).

```bash
make dpo-data                      # pool (data/dpo/prompts.jsonl, dpo_split), benchmark prompts, contamination --only dpo
for f in prompts.jsonl bench_prompts.jsonl; do $M volume put --force struct-lm data/dpo/$f data/dpo/$f; done
$M run --detach eval/modal_app.py --which sample --model /vol/checkpoints/sft-from-cpt \
  --run-name sft-from-cpt --chat --sample-jobs dpo_probe,dpo_bench
set -a; . ./.env; set +a
.venv/bin/python eval/diversity.py collapse sft-from-cpt          # the probe's stop rule
.venv/bin/python data/scripts/dpo_judge_bench.py run              # listwise judge vs the full-passage reads
.venv/bin/python data/scripts/dpo_probe.py sft-from-cpt           # pairable fraction -> data/dpo/budget.json
$M volume put --force struct-lm data/dpo/budget.json data/dpo/budget.json
$M run --detach eval/modal_app.py --which sample --model /vol/checkpoints/sft-from-cpt \
  --run-name sft-from-cpt --chat --sample-jobs dpo_pool,dpo_judge
make dpo-pairs                     # verifier/rule scores -> pairs (+ the as-registered count), contamination, tests
for f in train.jsonl val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/dpo/$f data/dpo/$f; done
$M run train/modal_train.py --config train/configs/dpo.yaml --run-name smoke-dpo --smoke --chat \
  --steps noop,train,merge,mergecheck
$M run --detach train/modal_train.py --config train/configs/dpo.yaml --run-name dpo --chat \
  --merge-from rule --steps noop,train,merge,mergecheck,ppl,eval,latency,sample
# dpo-seed1: --overrides "training.seed=1 training.data_seed=1"; dpo-2ep (ablation, epoch-2 adapter only):
#   --overrides "training.num_train_epochs=2 training.save_strategy=epoch", no --merge-from
# pull runs/<run>, ppl, lm_eval, bench; score with --chat --ppl-dir results/ppl; then
.venv/bin/python eval/diversity.py score dpo
.venv/bin/python eval/winrate.py dpo sft-from-cpt                 # reported only (the judge failed its benchmark)
.venv/bin/python train/report.py                                  # dpo.png, the Stage 4 read table, the docs' blocks
# dpo-strict (2026-10-09): closed-book labels by scorers.qa_strict -> data/dpo/strict/ (463 pairs:
# dpo_pairs.py exits 1 under the 500 floor, overridden for this rerun only)
make dpo-pairs-strict
for f in train.jsonl val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/dpo/strict/$f data/dpo/strict/$f; done
$M run --detach train/modal_train.py --config train/configs/dpo.yaml --run-name dpo-strict --chat --merge-from rule \
  --steps noop,train,merge,mergecheck,ppl,eval,sample \
  --overrides "data.dir=data/dpo/strict data.train=data/dpo/strict/train.jsonl data.val=data/dpo/strict/val.jsonl"
```

- **No judge in pair-building:** closed-book by the verifier, abstain by the decline rule, grounded
  by rules (`cite_valid`, cites the gold passage, no false abstain), no definition pairs. The
  listwise judge (`sft_judge.judge_list`) caught 28% of grounded and 9% of definition defects.
- **The trainer gets vLLM's sampled ids** (`prompt_ids`, `chosen_ids`, `rejected_ids`):
  `train/dpo.py` passes them through TRL's `_prepare_dataset` and computes log-probs in fp32
  (TRL 0.29.1 sums them in bf16). `tests/test_dpo_data.py` gates launch.
- **The merge gate runs on `sft_val`** for DPO runs too, since dpo_val (665 positions) is too
  small for the 0.1% line.
- **The judge's prompts go to the SFT builder's cache** (`data/sft/.cache`), so `relabel` strips
  every chunk id (`test_generator_prompts_clean`).

- `train/sft.py` trains on pre-tokenised records (`train/sft_data.py`: mistral-common, the eval's
  `--chat` rendering, `completion_mask`); `tests/test_template.py` is the gate on those tensors.
  Every reason and the pre-registered rules: `notes/decisions.md`, Stage 3b pre-registration.
- SFT checkpoints are chat models: `--chat` for their KPI eval and samples, never for lm-eval
  (`run_eval.needs_chat` refuses otherwise for run names with sft/dpo/grpo).
- The merge gate (`mergecheck`, `merge_check.py check`) compares merged and unmerged bf16 against
  an fp32 reference over all 11,351 sft_val positions (flips added <= 11, |dlp| ratio <= 1.5, val
  loss within 0.5%) and records the checkpoint's sha256. A failed gate stops the pipeline.
- Judge a run's generations into the shared cache before its lm-eval lands (`--results-dir
  <scratch> --judge-cache results/judge_cache.jsonl`), then score into `results/` for free.

### Stage 5: GRPO with verifiable rewards (tasks on the Mac, probe and training on Modal)

Pre-registration, corrections and the read: `notes/decisions.md` (2026-10-09). Stage5-final is
`checkpoints/dpo-strict` (nothing cleared the floor on the primary line; `grpo` / `grpo-seed1`,
both checkpoint-25, are the evaluated rows). Both runs stopped on entropy collapse (steps 52, 65).

```bash
make grpo-data                     # data/grpo/tasks.jsonl (Stage 4 pool minus judge split, + verifiers), contamination --only grpo, tests
$M volume put --force struct-lm data/grpo/tasks.jsonl data/grpo/tasks.jsonl
$M run --detach eval/modal_app.py --which sample --model /vol/checkpoints/dpo-strict --run-name dpo-strict \
  --chat --sample-jobs grpo_probe                                 # 8 x T 1.0 x 256 tokens per task
.venv/bin/python data/scripts/grpo_probe.py --run dpo-strict      # 1-7 of 8 window -> train/val (50 by fact), SHA256SUMS
for f in train.jsonl val.jsonl SHA256SUMS; do $M volume put --force struct-lm data/grpo/$f data/grpo/$f; done
$M run train/modal_train.py --config train/configs/grpo.yaml --run-name smoke-grpo --smoke --chat \
  --steps noop,train,merge,mergecheck --merge-from rule --overrides "data.train=data/grpo/tasks.jsonl data.val=data/grpo/tasks.jsonl"
$M run --detach train/modal_train.py --config train/configs/grpo.yaml --run-name grpo --chat --merge-from rule \
  --steps noop,train,merge,mergecheck,ppl,eval,latency,sample    # sample: eos,diversity,passk,dpo_judge
# grpo-seed1: --overrides "training.seed=1 training.data_seed=1", no latency
$M run --detach eval/modal_app.py --which sample --model /vol/checkpoints/<row> --run-name <row> --chat --sample-jobs passk
# pull runs/<run> (rollouts.jsonl too), ppl, lm_eval, bench; score with --chat --ppl-dir results/ppl; then
.venv/bin/python eval/qa_strict.py                                # strict closed-book column for every table row
.venv/bin/python eval/passk.py grpo grpo-seed1                    # pass@1 / maj@8 / pass@8 -> results/passk/
.venv/bin/python data/scripts/grpo_audit.py grpo                  # the 50 top-reward rollouts, rule flags; read, verdicts only
.venv/bin/python train/report.py                                  # grpo.png, passk.png, the Stage 5 tables, the docs' blocks
```

- **The reward** (`train/grpo_rewards.py`): format 0.1 (a gate), correctness 0.9 (closed-book by
  `scorers.qa_strict`, grounded cites the gold passage in at most 2, abstain the exact sentence),
  length 0 to -0.1. The probe scorer, the trainer and the DPO strict labels share it.
- **The rollout engine:** `train/grpo.py` wraps TRL's colocated `LLM(...)` with rule 3's settings
  and the run's seed, and keeps sleep mode off (TRL reloads weights from disk on wake, which would
  drop the LoRA sync). `grpo_image` in `modal_train.py` is the training pins plus vllm 0.30.0.
- **Rollouts:** `results/runs/<run>/rollouts.jsonl`, one row per completion; its `format` field is
  the gate's verdict, the task's format is in `data/grpo/tasks.jsonl`.

### Stage 6: serving (quantize, gate and bench on Modal; bench sets, scoring and report on the Mac)

Pre-registration (gate lines and floors, served-path checks, bench rules) and the corrections to the
pasted plan: `notes/decisions.md` (2026-10-09, Stage 6). Deployment reference: `DEPLOY.md`. Served:
`checkpoints/dpo-strict` (stage5-final) as bf16, `-fp8` (FP8_DYNAMIC), `-fp8` + `--kv-cache-dtype fp8`
(fp8kv) and `-w4a16` (GPTQ, group 128). Stage6-final is `dpo-strict` served by load: bf16 up to the
measured 16 req/s, FP8 near saturation or when the KV cache binds; fp8kv and w4a16 failed the gate.
Not run (Modal spend limit, 2026-10-09): the fp8kv / w4a16 / n-gram bench rows and fp8kv's GSM8K.

```bash
.venv/bin/python serve/bench_data.py      # serve/bench/*.jsonl (gitignored) + serve/bench_manifest.json (committed)
$M run serve/modal_serve.py --action quantize --scheme fp8                 # -> /vol/checkpoints/dpo-strict-fp8, ~10 min
$M run --detach serve/modal_serve.py --action quantize --scheme w4a16      # GPTQ on 512 SFT records, ~45 min
$M run serve/modal_serve.py --action gate --variant bf16                   # vllm_ppl, BOS probe, GSM8K (BOS) rerun
$M run --detach serve/modal_serve.py --action gate --variant fp8           # kpi, eos, ppl, gsm8k; also fp8kv, w4a16
$M run serve/modal_serve.py --action bench --variants bf16 --check-only    # template ids, 20-prompt smoke, 24-request bench
$M run --detach serve/modal_serve.py --action bench --variants bf16,fp8,fp8kv,w4a16 --spec   # one H100!, ~3 h
# pull: bench, gate GSM8K, served checks; the variants' generations; perplexities; quantized configs
$M volume get --force struct-lm results/serve results/
for r in dpo-strict-fp8 dpo-strict-fp8kv dpo-strict-w4a16; do $M volume get --force struct-lm results/runs/$r results/runs/; done
$M volume get --force struct-lm results/vllm_ppl results/
for v in fp8 w4a16; do for f in config.json quantize_meta.json; do
  $M volume get --force struct-lm checkpoints/dpo-strict-$v/$f results/serve/quantize/$v/; done; done
set -a; . ./.env; set +a
.venv/bin/python eval/run_eval.py --run-name dpo-strict-fp8 --rescore --chat --model /vol/checkpoints/dpo-strict-fp8
.venv/bin/python eval/qa_strict.py && .venv/bin/python -m pytest tests/test_serve.py tests/test_template.py
.venv/bin/python train/report.py          # the Stage 6 tables, serve_latency.png, serve_load.png, the docs' blocks
```

- **The README demo:** `DEMO_API_KEY=... modal serve serve/modal_demo.py` (a GPU launch: ask), then
  `serve/demo.py` with `DEMO_URL` / `DEMO_API_KEY` in the environment, captured and rendered to
  `docs/demo.gif` by `serve/demo_render.py`; stop the ephemeral app by id (`modal app stop -y ap-...`).
- **The bench runs on `gpu="H100!"`** (a plain "H100" may run on an H200) and aborts on any other
  device; every result records GPU, driver, CUDA, vLLM and torch. All variants share one container.
- **Gate GSM8K goes to `results/serve/gate/lm_eval/`**, never `results/lm_eval/`: it runs with
  `add_bos_token=True` through `eval/gsm8k_gate.py` (lm-eval's CLI can't pass it under
  `tokenizer_mode=mistral`), and the table's rows keep the frozen flags, which send no BOS.
- **`VLLM_SERVER_DEV_MODE=1` is the bench container's only** (it exposes `/reset_prefix_cache`);
  `DEPLOY.md`'s commands never set it. `vllm bench serve` always runs with `--skip-chat-template`
  (the server templates; client-side templating would send `[INST]` as text) and `--no-oversample`.
- **Quantized checkpoints come from `serve/quantize.py` only:** the full Mistral3 class, `lm_head` /
  tower / projector ignored, merge.py's post-save steps (YaRN key, tokenizer files, untied lm_head).
  Its image pins llmcompressor 0.14.0 (compressed-tensors 0.19); vLLM 0.29 reads the result with 0.17.

## Decisions (rules to keep)

1. **Copyright:** only public-domain US federal documents. ASCE 7, the AISC manual and the 2025
   NSBA handbook are excluded. Check any new source before adding it to `sources.csv`.
2. **Chat format:** lm-eval never uses the chat template, for any checkpoint; the KPI eval always
   runs chat checkpoints (Instruct, SFT/DPO/GRPO) with `--chat`, base models without. Enforced:
   `run_lm_eval.sh` exits on `CHAT=1`, `modal_app.py`, `modal_train.py` and `run_eval.py` refuse a
   chat checkpoint (Instruct, or a run name / path with sft, dpo or grpo) without `--chat`, `merge_lm_eval` skips chat-template results files. (lm-eval renders the
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
   `false_abstain` (grounded answers with the abstain phrase, by rule; from Stage 3),
   plus `mmlu` and its four groups / `gsm8k` (strict-match) / `hellaswag` (acc_norm), and from
   Stage 2 `ppl_train` / `ppl_domain_val` / `ppl_general_val` (`eval/perplexity.py`, lower is
   better; `ppl_train` is measured on a train slice, for base too) and `ppl_postcutoff` (the 13
   2026 reports, `--only postcutoff`). `qa_strict` (`eval/qa_strict.py` -> `results/qa_strict/`,
   not a table column): the strict checker on the stored answers; Stage 5's primary reads it.
   `gold_lp` / `gold_lp_seen` / `gold_lp_unseen`: mean
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
9. **Eval tasks** (v3 from Stage 3: domain_qa 322, grounded 108, vocab 210, adversarial 76 = 716
   items + 3 few-shot, every one reviewed; v2 of 2026-10-04 had domain_qa 325; v1 of 2026-09-27
   had domain_qa 130, frozen before Stage 2,
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
    `tests/test_sft_data.py` and `contamination.py --only sft`. The chat template work is done
    (Stage 3: `train/sft_data.py`, `tests/test_template.py`). Stage 4 is DPO on verifiable
    preferences: the preference judge failed its benchmark (2026-10-08) and labels nothing. Stage
    5's tasks are the Stage 4 pool's prompts in the probe's window; compute tasks are not built.
11. **Reference model = the previous stage, not the base:** with LoRA and `ref_model=None`, TRL's
    reference is the adapter-disabled `init_from` checkpoint, so DPO's is the SFT checkpoint and
    GRPO's the DPO checkpoint (`dpo-strict`). Each stage's KL term (DPO's `beta`, GRPO's logged `kl`) measures drift
    from the previous stage, not from the base. Drift from the base shows only in the eval rows.
    `grpo.yaml` has `beta: 0.0` (no KL term, no reference) until raised.
12. **Checkpoints:** no checkpoint with a row in a results table is deleted until that stage's
    write-up is frozen, and adapters (`checkpoints/_train/<run>`) are never deleted; merged LoRA
    checkpoints are reproducible from them, full-parameter ones are not.
13. **No Claude-written text in training data** (SFT, DPO chosen and rejected, GRPO): every
    completion comes from the Mistral teachers, a fixed string, or a replay source whose card and
    paper name a generator that isn't Claude (`sft_replay.EXCLUDED` drops three Tulu 3 Persona
    subsets). Claude builds the tooling and reviews records with verdicts only, never by rewriting
    one. The README's "Who wrote the training data" line states the roles.

## Known gaps

- The `gen_qa` prompt still shows the chunk id (`slug:p12:c0`), so the generator sees the page.
  The locator rule removes page/table/figure items at assembly, and none came from the page so
  far. Drop the id from the prompt only in a from-scratch rebuild, since it changes every cached
  generation, the frozen items included.
- Pyright errors about `prompts` / `scorers` / `judge` / `vllm` imports in `eval/`, and about
  `common` / `packing` / `modal_app` imports in `train/`, are false positives (`sys.path` imports;
  vLLM and lm-eval are only installed in the Modal image).
- No W&B secret exists; every stage logs to `results/runs/<run>/train_log.jsonl`. CPT, SFT, DPO
  and GRPO are wired into `train/modal_train.py` (`stage:` in the config); `tf32: true` is on from
  Stage 3.
- PEFT's merge/unmerge for each GRPO weight sync edits the bf16 base in place (one ulp,
  `base_drift_max` in the summary); the merged checkpoint uses the untouched base from disk.
- Modal's H100 price in `train/report.py` (`--usd-per-gpu-hour`, default 3.95) is the H100 SXM5 list
  price ($0.001097/s), checked 2026-10-09: recheck modal.com/pricing before quoting new dollars.
- `results/lm_eval/_invalid/` holds an excluded chat-template lm-eval run (see its README).
