# struct-lm

> Independent project, using only public documents, open weights and open-source tools. Forge is
> described from Mistral AI's public announcement.

Domain adaptation of an open base LLM
([`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512))
to structural and civil engineering through **CPT → SFT → DPO → GRPO**, with every stage
measured on the same KPI tasks and general-capability regression suite, then quantized
(FP8, INT4) behind a pre-registered quality gate and served with vLLM on one H100
([`DEPLOY.md`](DEPLOY.md)).

> This README is the write-up. Numbers live in [`results/table.md`](results/table.md);
> the reasoning behind every choice lives in [`notes/decisions.md`](notes/decisions.md).

## Contents

- [Stage 0: the problem, the eval and the baselines](docs/stage0.md)
- [Stage 1: the corpus](docs/stage1.md)
- [Stage 2: continued pre-training (CPT)](docs/stage2.md)
- [Stage 3: supervised fine-tuning (SFT)](docs/stage3.md)
- [Stage 4: DPO on verifiable preferences](docs/stage4.md)
- [Stage 5: GRPO with verifiable rewards](docs/stage5.md)
- [Stage 6: serving](docs/stage6.md)
- [Results: every scored checkpoint](docs/results.md)
- [Reproduce: pipeline, quickstart, layout](docs/reproduce.md)

<!-- headline-summary:start -->
| run | what it is | closed-book seen (strict) | closed-book unseen (strict) | gold_lp answer, unseen | grounded_acc (judge) | halluc_rate ↓ | GSM8K (no BOS) |
|---|---|---|---|---|---|---|---|
| `base-8b-hf` | base [base format] | 12.6 | 11.6 | -6.36 | 84.3 | 89.5 | 79.3 |
| `instruct-8b` | stock Instruct | 11.4 | 7.7 | -7.71 | 89.8 | 1.3 | 85.5 |
| `sft-from-base` | SFT on the base (control) | 23.9 | 8.4 | -6.20 | 92.6 | 6.6 | 79.2 |
| `sft-from-cpt` | CPT → SFT | 28.7 | 11.6 | -5.89 | 90.7 | 5.3 | 81.4 |
| `dpo-strict` | → DPO: final, served | 30.5 | 11.0 | -6.27 | 92.6 | 4.0 | 80.9 |
| `grpo` | → GRPO (rejected) | 29.9 | 9.7 | -7.54 | 93.5 | 1.3 | 82.0 |
|  | *floor, chat rows: sft-from-cpt vs sft-from-cpt-seed1, or SE* | 3.5 | 2.6 | 0.21 | 2.8 | 3.9 | 2.2 |
<!-- headline-summary:end -->

<!-- headline-knowledge:start -->
| run | what it is | closed-book seen (strict) | closed-book unseen (strict) | identifiers (strict) | gold_lp answer, seen | gold_lp answer, unseen | gold_lp end, seen | gold_lp end, unseen |
|---|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 12.6 | 11.6 | 7.8 | -6.28 | -6.36 | -0.44 | -0.47 |
| `instruct-8b` | stock Instruct: the bar | 11.4 | 7.7 | 3.1 | -7.31 | -7.71 | -0.70 | -0.87 |
| `mistral-large-3` | frontier reference (API, closed-book only) | 26.4 | 25.8 | 40.6 |  |  |  |  |
| `cpt-8b` | CPT [base format] | 15.6 | 13.6 | 12.5 | -5.84 | -5.68 | -0.40 | -0.45 |
| `cpt-8b-seed1` | CPT, seed 1 [base format] | 13.2 | 12.9 | 12.5 | -5.84 | -5.70 | -0.40 | -0.45 |
| `cpt-8b-replay10` | CPT + 10% replay: the chain's [base format] | 15.0 | 11.0 | 12.5 | -5.93 | -5.70 | -0.43 | -0.46 |
| `sft-from-base` | SFT on the base: control arm | 23.9 | 8.4 | 10.9 | -4.97 | -6.20 | -0.44 | -0.58 |
| `sft-from-base-seed1` | control arm, seed 1 | 23.4 | 7.7 | 14.1 | -4.87 | -6.20 | -0.38 | -0.54 |
| `sft-from-cpt` | SFT on CPT: the chain's | 28.7 | 11.6 | 20.3 | -4.70 | -5.89 | -0.42 | -0.53 |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 25.8 | 11.0 | 23.4 | -4.52 | -5.68 | -0.35 | -0.48 |
| `dpo` | DPO, as-run labels | 30.5 | 11.6 | 20.3 | -4.70 | -6.16 | -0.30 | -0.38 |
| `dpo-seed1` | DPO, seed 1 | 28.7 | 11.0 | 17.2 | -4.85 | -6.28 | -0.35 | -0.45 |
| `dpo-2ep` | DPO 2 epochs (ablation, failed merge gate) | 25.8 | 12.9 | 14.1 | -7.82 | -10.57 | -0.20 | -0.28 |
| `dpo-strict` | DPO, strict labels: final, served bf16 | 30.5 | 11.0 | 18.8 | -4.81 | -6.27 | -0.32 | -0.42 |
| `grpo` | GRPO checkpoint-25 (rejected) | 29.9 | 9.7 | 20.3 | -5.52 | -7.54 | -0.27 | -0.41 |
| `grpo-seed1` | GRPO, seed 1 | 28.7 | 10.3 | 18.8 | -5.52 | -7.65 | -0.30 | -0.45 |
| `dpo-strict-fp8` | served FP8 (passed the gate) | 29.3 | 9.7 | 17.2 | -4.82 | -6.27 | -0.32 | -0.44 |
| `dpo-strict-fp8kv` | FP8 + FP8 KV cache (failed the gate) | 26.4 | 11.6 | 17.2 | -4.87 | -6.33 | -0.33 | -0.45 |
| `dpo-strict-w4a16` | INT4 W4A16 (failed the gate) | 27.5 | 8.4 | 10.9 | -5.12 | -6.70 | -0.30 | -0.36 |
|  | *floor, base-format rows: cpt-8b vs cpt-8b-seed1, or SE* | 2.8 | 2.7 | 4.1 | 0.03 | 0.03 | 0.01 | 0.01 |
|  | *floor, chat rows: sft-from-cpt vs sft-from-cpt-seed1, or SE* | 3.5 | 2.6 | 5.0 | 0.17 | 0.21 | 0.07 | 0.05 |

No numbers in this table: `cpt-8b-full`.
<!-- headline-knowledge:end -->

<!-- headline-behaviour:start -->
| run | what it is | grounded_acc (judge) | cite_valid | cite_supported (judge, reported) | halluc_rate ↓ | false_abstain ↓ | vocab_recall (judge) |
|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 84.3 | 12.0 | 8.3 | 89.5 | 0.9 | 70.5 |
| `instruct-8b` | stock Instruct: the bar | 89.8 | 83.3 | 78.7 | 1.3 | 7.4 | 78.6 |
| `cpt-8b` | CPT [base format] | 83.3 | 14.8 | 7.4 | 90.8 | 0.0 | 71.0 |
| `cpt-8b-seed1` | CPT, seed 1 [base format] | 79.6 | 10.2 | 3.7 | 93.4 | 0.0 | 70.0 |
| `cpt-8b-replay10` | CPT + 10% replay: the chain's [base format] | 77.8 | 5.6 | 3.7 | 93.4 | 0.0 | 70.0 |
| `cpt-8b-full` | CPT full-parameter (ablation) [base format] | 88.9 | 14.8 | 10.2 | 92.1 | 0.0 | 74.3 |
| `sft-from-base` | SFT on the base: control arm | 92.6 | 100.0 | 87.0 | 6.6 | 0.0 | 83.3 |
| `sft-from-base-seed1` | control arm, seed 1 | 89.8 | 99.1 | 88.0 | 1.3 | 0.9 | 82.9 |
| `sft-from-cpt` | SFT on CPT: the chain's | 90.7 | 100.0 | 86.1 | 5.3 | 0.0 | 83.3 |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 93.5 | 98.2 | 88.0 | 1.3 | 0.9 | 85.7 |
| `dpo` | DPO, as-run labels | 91.7 | 100.0 | 87.0 | 1.3 | 0.0 | 83.8 |
| `dpo-seed1` | DPO, seed 1 | 90.7 | 100.0 | 88.0 | 2.6 | 0.0 | 84.8 |
| `dpo-2ep` | DPO 2 epochs (ablation, failed merge gate) | 90.7 | 100.0 | 85.2 | 1.3 | 0.0 | 85.2 |
| `dpo-strict` | DPO, strict labels: final, served bf16 | 92.6 | 100.0 | 86.1 | 4.0 | 0.0 | 82.4 |
| `grpo` | GRPO checkpoint-25 (rejected) | 93.5 | 100.0 | 88.0 | 1.3 | 0.0 | 84.3 |
| `grpo-seed1` | GRPO, seed 1 | 91.7 | 98.2 | 88.9 | 1.3 | 1.8 | 85.2 |
| `dpo-strict-fp8` | served FP8 (passed the gate) | 90.7 | 100.0 | 85.2 | 5.3 | 0.0 | 82.9 |
| `dpo-strict-fp8kv` | FP8 + FP8 KV cache (failed the gate) | 93.5 | 100.0 | 84.3 | 6.6 | 0.0 | 83.3 |
| `dpo-strict-w4a16` | INT4 W4A16 (failed the gate) | 89.8 | 100.0 | 84.3 | 4.0 | 0.0 | 77.6 |
|  | *floor, base-format rows: cpt-8b vs cpt-8b-seed1, or SE* | 3.7 | 4.6 | 3.7 | 3.3 | 0.0 | 3.1 |
|  | *floor, chat rows: sft-from-cpt vs sft-from-cpt-seed1, or SE* | 2.8 | 1.8 | 3.3 | 3.9 | 0.9 | 2.6 |

No numbers in this table: `mistral-large-3`.
<!-- headline-behaviour:end -->

<!-- headline-general:start -->
| run | what it is | MMLU (no BOS) | GSM8K (no BOS) | HellaSwag (no BOS) | ppl domain val ↓ | ppl general val ↓ | TTFT p50 @1 (ms) | ITL p50 @1 (ms) | latency source |
|---|---|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 76.7 | 79.3 | 80.1 | 6.88 | 8.15 | 15.3 | 6.60 | 1 sample, GPU not recorded |
| `instruct-8b` | stock Instruct: the bar | 76.1 | 85.5 | 80.1 |  |  | 17.7 | 6.60 | 1 sample, GPU not recorded |
| `cpt-8b` | CPT [base format] | 76.4 | 78.5 | 80.1 | 6.72 | 8.18 | 16.5 | 6.90 | 1 sample, GPU not recorded |
| `cpt-8b-seed1` | CPT, seed 1 [base format] | 76.5 | 78.6 | 80.0 | 6.72 | 8.17 |  |  |  |
| `cpt-8b-replay10` | CPT + 10% replay: the chain's [base format] | 76.6 | 79.1 | 80.0 | 6.72 | 7.97 |  |  |  |
| `cpt-8b-full` | CPT full-parameter (ablation) [base format] | 76.2 | 76.3 | 79.8 | 6.73 | 8.25 |  |  |  |
| `sft-from-base` | SFT on the base: control arm | 76.7 | 79.2 | 79.5 | 7.01 | 8.23 | 18.4 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-base-seed1` | control arm, seed 1 | 76.8 | 79.0 | 79.9 | 7.01 | 8.25 | 17.6 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-cpt` | SFT on CPT: the chain's | 76.6 | 81.4 | 79.4 | 6.87 | 8.06 | 17.8 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 77.0 | 79.2 | 80.0 | 6.87 | 8.07 | 16.2 | 6.90 | 1 sample, GPU not recorded |
| `dpo` | DPO, as-run labels | 76.7 | 81.0 | 79.6 | 6.88 | 8.07 | 17.3 | 6.80 | 1 sample, GPU not recorded |
| `dpo-seed1` | DPO, seed 1 | 76.5 | 80.7 | 79.5 | 6.89 | 8.07 |  |  |  |
| `dpo-2ep` | DPO 2 epochs (ablation, failed merge gate) | 76.8 | 80.6 | 80.2 | 6.97 | 8.11 |  |  |  |
| `dpo-strict` | DPO, strict labels: final, served bf16 | 76.7 | 80.9 | 79.6 | 6.88 | 8.06 | 17.9 | 6.90 | H100, Stage 6 bench |
| `grpo` | GRPO checkpoint-25 (rejected) | 76.7 | 82.0 | 79.8 | 6.93 | 8.09 | 25.5 | 6.70 | 1 sample, GPU not recorded |
| `grpo-seed1` | GRPO, seed 1 | 76.7 | 80.2 | 79.8 | 6.93 | 8.09 |  |  |  |
| `dpo-strict-fp8` | served FP8 (passed the gate) |  |  |  |  |  | 35.2 | 4.70 | H100, Stage 6 bench |
|  | *floor, base-format rows: cpt-8b vs cpt-8b-seed1, or SE* | 0.3 | 1.1 | 0.4 | 0.02% | 0.23% |  |  |  |
|  | *floor, chat rows: sft-from-cpt vs sft-from-cpt-seed1, or SE* | 0.4 | 2.2 | 0.6 | 0.08% | 0.07% |  |  |  |

No numbers in this table: `mistral-large-3`, `dpo-strict-fp8kv`, `dpo-strict-w4a16`.
<!-- headline-general:end -->
