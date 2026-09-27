# struct-lm

Domain adaptation of an open base LLM
([`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512))
to structural and civil engineering through **CPT → SFT → DPO → GRPO**, with every stage
measured on the same KPI tasks and general-capability regression suite, then quantized
(AWQ) and served with vLLM.

> This README is the write-up. Numbers live in [`results/table.md`](results/table.md);
> the reasoning behind every choice lives in [`notes/decisions.md`](notes/decisions.md).

## Pipeline

```
sources.csv ─ download ─ extract ─ filter ─ dedup ─ pack ──► CPT ─merge─► SFT ─merge─► DPO ─merge─► GRPO ─merge─► AWQ ─► vLLM
                                                     │                                                                    │
                                    tokenizer_coverage                              run_eval + run_lm_eval + bench  ◄────┘
                                       make_tasks (eval/tasks/, from chunks.jsonl)
```

| Stage | Script | Input data | Signal |
|-------|--------|-----------|--------|
| CPT  | `train/cpt.py`  | `data/packed/cpt` (from `pack.py`) | next-token on domain text |
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
make data                                   # download → extract → filter → dedup → pack
make coverage                               # tokenizer fit on the domain corpus
python eval/make_tasks.py                   # eval/tasks/*.jsonl + eval_chunk_ids.txt (Mistral Large 3)
```

USACE, ROSA P and FEMA block scripted downloads: save those PDFs by hand as
`data/raw/<slug>.pdf`, then re-run `download.py` to hash them into `data/raw/manifest.json`.
Regenerate `eval/tasks/` whenever `chunks.jsonl` is re-extracted (chunk ids move), and never
build SFT data from chunks in `eval/tasks/eval_chunk_ids.txt`.

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
data/     raw/ (PDFs gitignored; manifest.json with sha256s committed), scripts/ for each
          preprocessing step + sources.csv (corpus table); interim/, processed/, packed/ gitignored
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
_What the model must do, and the task-level metrics in `eval/tasks/` that define success._

### 2. Data
_Sources and licenses, extraction backend, filter reject rates by rule (`filter.py` prints
them), dedup rate, final token count, tokenizer fertility._

#### Corpus sources

All US federal works (public domain, 17 USC 105). Generated from
[`data/scripts/sources.csv`](data/scripts/sources.csv), which `download.py` reads; each file lands at
`data/raw/<slug>.pdf`. USACE and ROSA P (Akamai bot filter) and FEMA block scripted downloads: get those
in a browser, save them under their slug, and re-run `download.py` to hash them into the manifest.

| # | Title | Publisher | Pages | Link |
|---|-------|-----------|------:|------|
| 1 | Strength Design for Reinforced Concrete Hydraulic Structures (2016) | USACE | 120 | [PDF](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2104.pdf) |
| 2 | Earthquake Design and Evaluation of Concrete Hydraulic Structures (2007) | USACE | 200 | [PDF](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-6053.pdf) |
| 3 | Stability Analysis of Concrete Structures (2005) | USACE | 200 | [PDF](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2100.pdf) |
| 4 | Design and Construction of Levees (2000) | USACE | 150 | [PDF](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1913.pdf) |
| 5 | NEHRP Seismic Design Technical Brief 1: RC Special Moment Frames (2nd ed.) | NIST | 42 | [PDF](https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-40.pdf) |
| 6 | NEHRP Seismic Design Technical Brief 2: Steel Special Moment Frames (2nd ed.) | NIST | 36 | [PDF](https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-41.pdf) |
| 7 | NEHRP Seismic Design Technical Brief 3: Cast-in-Place Concrete Diaphragms, Chords and Collectors (2nd ed.) | NIST | 39 | [PDF](https://nvlpubs.nist.gov/nistpubs/gcr/2016/NIST.GCR.16-917-42.pdf) |
| 8 | NEHRP Seismic Design Technical Brief 4: Nonlinear Structural Analysis for Seismic Design | NIST | 32 | [PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=915469) |
| 9 | NEHRP Seismic Design Technical Brief 13: Precast Concrete Diaphragms | NIST | 41 | [PDF](https://nvlpubs.nist.gov/nistpubs/gcr/2017/NIST.GCR.17-917-47.pdf) |
| 10 | Bridge Inspector's Reference Manual (NHI-23-024; 2023) | FHWA | 1000 | [PDF](https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi23024.pdf) |
| 11 | LRFD for Highway Bridge Superstructures Reference Manual (NHI-15-047) | FHWA | 1700 | [PDF](https://www.fhwa.dot.gov/bridge/pubs/nhi15047.pdf) |
| 12 | Engineering for Structural Stability in Bridge Construction (NHI-15-044) | FHWA | 300 | [PDF](https://www.fhwa.dot.gov/bridge/pubs/nhi15044.pdf) |
| 13 | Design and Evaluation of Steel Bridges for Fatigue and Fracture (NHI-16-016) | FHWA | 310 | [PDF](https://www.fhwa.dot.gov/bridge/steel/pubs/nhi16016.pdf) |
| 14 | LRFD Design Example: Steel Girder Superstructure Bridge with Commentary (NHI-04-041) | FHWA | 640 | [PDF](https://www.fhwa.dot.gov/bridge/lrfd/fhwanhi04041_steel.pdf) |
| 15 | Manual for Refined Analysis in Bridge Design and Evaluation (HIF-18-046) | FHWA | — | [PDF](https://www.fhwa.dot.gov/bridge/pubs/hif18046.pdf) |
| 16 | NASA-STD-5001B w/Change 3: Structural Design and Test Factors of Safety | NASA | 36 | [PDF](https://standards.nasa.gov/sites/default/files/standards/NASA/B-w/CHANGE-3/3/2022-10-24-NASA-STD-5001B-w-Change-3-Approved.pdf) |
| 17 | NASA-STD-5020B: Threaded Fastening Systems in Spaceflight Hardware | NASA | 114 | [PDF](https://standards.nasa.gov/sites/default/files/standards/NASA/B/0/2021-08-06-nasa-std-5020b_final.pdf) |
| 18 | 2020 NEHRP Design Examples Vol. 1 | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/documents/fema_nehrp_design-examples-training-materials_volume-1.pdf) |
| 19 | FEMA P-2012: Assessing Seismic Performance of Buildings with Configuration Irregularities | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/2020-08/fema_assessing-seismic-performance-irregularities_p-2012.pdf) |
| 20 | FEMA P-155: Rapid Visual Screening of Buildings for Potential Seismic Hazards: Supporting Documentation (3rd ed.) | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/2020-07/fema_earthquakes_rapid-visual-screening-of-buildings-for-potential-seismic-hazards-supporting-documentation-third-edition-fema-p-155.pdf) |
| 21 | FEMA P-1050-1: 2015 NEHRP Recommended Seismic Provisions for New Buildings and Other Structures | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/2020-07/fema_nehrp-seismic-provisions-new-buildings_p-1050-1_2015.pdf) |
| 22 | FEMA P-1051: 2015 NEHRP Recommended Seismic Provisions: Design Examples | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/2020-07/fema_nehrp-seismic-provisions-examples_p-1051_7-2016.pdf) |
| 23 | FEMA P-2082-1: 2020 NEHRP Recommended Seismic Provisions (Parts 1 and 2) | FEMA | — | [PDF](https://www.fema.gov/sites/default/files/2020-10/fema_2020-nehrp-provisions_part-1-and-part-2.pdf) |
| 24 | LRFD for Highway Bridge Superstructures: Design Examples (NHI-15-058) | FHWA | — | [PDF](https://www.fhwa.dot.gov/bridge/pubs/nhi15058.pdf) |
| 25 | Bridge Welding Reference Manual (HIF-19-088) | FHWA | — | [PDF](https://www.fhwa.dot.gov/bridge/steel/pubs/hif19088.pdf) |
| 26 | Manual for Repair and Retrofit of Fatigue Cracks in Steel Bridges (HIF-13-020) | FHWA | — | [PDF](https://www.fhwa.dot.gov/bridge/steel/pubs/hif13020/hif13020.pdf) |
| 27 | Manual for Orthotropic Deck Bridges (IF-12-027) | FHWA | — | [PDF](https://www.fhwa.dot.gov/bridge/pubs/if12027/if12027.pdf) |
| 28 | Steel Bridge Design Handbook (HIF-16-002; 2015): Bridge Steels and Their Mechanical Properties | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42712/dot_42712_DS1.pdf) |
| 29 | Steel Bridge Design Handbook (HIF-16-002; 2015): Steel Bridge Fabrication | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42901/dot_42901_DS1.pdf) |
| 30 | Steel Bridge Design Handbook (HIF-16-002; 2015): Structural Steel Bridge Shop Drawings | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42894/dot_42894_DS1.pdf) |
| 31 | Steel Bridge Design Handbook (HIF-16-002; 2015): Structural Behavior of Steel | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42892/dot_42892_DS1.pdf) |
| 32 | Steel Bridge Design Handbook (HIF-16-002; 2015): Selecting the Right Bridge Type | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42909/dot_42909_DS1.pdf) |
| 33 | Steel Bridge Design Handbook (HIF-16-002; 2015): Stringer Bridges and Making the Right Choices | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42713/dot_42713_DS1.pdf) |
| 34 | Steel Bridge Design Handbook (HIF-16-002; 2015): Loads and Load Combinations | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42714/dot_42714_DS1.pdf) |
| 35 | Steel Bridge Design Handbook (HIF-16-002; 2015): Structural Analysis | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42906/dot_42906_DS1.pdf) |
| 36 | Steel Bridge Design Handbook (HIF-16-002; 2015): Redundancy | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42893/dot_42893_DS1.pdf) |
| 37 | Steel Bridge Design Handbook (HIF-16-002; 2015): Limit States | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42895/dot_42895_DS1.pdf) |
| 38 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design for Constructibility | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42905/dot_42905_DS1.pdf) |
| 39 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design for Fatigue | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42898/dot_42898_DS1.pdf) |
| 40 | Steel Bridge Design Handbook (HIF-16-002; 2015): Bracing System Design | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42710/dot_42710_DS1.pdf) |
| 41 | Steel Bridge Design Handbook (HIF-16-002; 2015): Splice Design | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42902/dot_42902_DS1.pdf) |
| 42 | Steel Bridge Design Handbook (HIF-16-002; 2015): Bearing Design | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42711/dot_42711_DS1.pdf) |
| 43 | Steel Bridge Design Handbook (HIF-16-002; 2015): Substructure Design | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42897/dot_42897_DS1.pdf) |
| 44 | Steel Bridge Design Handbook (HIF-16-002; 2015): Bridge Deck Design | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42908/dot_42908_DS1.pdf) |
| 45 | Steel Bridge Design Handbook (HIF-16-002; 2015): Load Rating of Steel Bridges | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42900/dot_42900_DS1.pdf) |
| 46 | Steel Bridge Design Handbook (HIF-16-002; 2015): Corrosion Protection of Steel Bridges | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42715/dot_42715_DS1.pdf) |
| 47 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 1: Three-Span Continuous Straight Composite Steel I-Girder Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42904/dot_42904_DS1.pdf) |
| 48 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 2A: Two-Span Continuous Straight Composite Steel I-Girder Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42891/dot_42891_DS1.pdf) |
| 49 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 2B: Two-Span Continuous Straight Composite Steel Wide-Flange Beam Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42907/dot_42907_DS1.pdf) |
| 50 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 3: Three-Span Continuous Horizontally Curved Composite Steel I-Girder Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42903/dot_42903_DS1.pdf) |
| 51 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 4: Three-Span Continuous Straight Composite Steel Tub Girder Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42899/dot_42899_DS1.pdf) |
| 52 | Steel Bridge Design Handbook (HIF-16-002; 2015): Design Example 5: Three-Span Continuous Horizontally Curved Composite Steel Tub-Girder Bridge | FHWA | — | [PDF](https://rosap.ntl.bts.gov/view/dot/42896/dot_42896_DS1.pdf) |

### 3. Training
_Per stage: data size, key hyperparameters, curves (`results/curves/`), what changed._

### 4. Results
_`results/table.md`: KPI gain per stage vs regression-suite cost. Which stages earned
their keep, and which didn't._

### 5. Serving
_AWQ quality delta vs bf16; TTFT / ITL / throughput at 1 and 32 concurrent requests._

### 6. What I'd do next
_Honest limitations: judge bias, eval set size, reward hacking observed in GRPO, etc._
