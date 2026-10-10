# struct-lm

> Independent project, using only public documents, open weights and open-source tools. Forge is
> described from Mistral AI's public announcement.

The post-training lifecycle a Forge engagement runs, at roughly 1% scale: Ministral 3 8B Base
adapted to US federal structural-engineering documents by continued pre-training, SFT, DPO and
GRPO, then quantized behind a pre-registered quality gate and served with vLLM on one H100.

## 1. Summary

**What was trained.** [`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512)
on 246 public-domain manuals, reports and design examples from USACE, FEMA, FHWA, NIST and NASA
(20.6M tokens; CPT read the 19.4M-token train split once).
- **Training:** each stage of the shipped chain is a LoRA adapter (r64) on one H100, merged before
  the next. Two Stage 2 ablations used 2 GPUs.
- **The eval:** Mistral Large 3 generated its 716 items from the training documents, and each was
  reviewed against its passage before any training. The closed-book task later grew from 130 to
  322 items, and every row was rescored.
- **The rules:** reads follow rules written before their data. Those amended after their read are
  listed at the top of [`notes/decisions.md`](notes/decisions.md).
- **The floor:** a change counts only beyond its noise floor, the larger of a seed-pair gap and the
  metric's standard error. That is about one standard deviation: a screen, not a significance
  test.

**In one sentence:** CPT made the corpus's text, gold answers included, more probable but not more
recallable. SFT taught the facts it showed and the behaviours: answer from passages, cite them,
decline. DPO and GRPO sharpened sampling at a cost in calibration, without clearing their bars.

- **CPT's gain survives SFT as probability, not accuracy.**
  - On the 155 facts SFT never showed, gold-answer tokens are 0.41 nats more probable in the CPT arm
    than in a control arm without CPT (95% CI +0.23 to +0.60; 1 df per arm).
  - That is about what CPT's general rise in corpus-text probability predicts: 0.086 nats per
    token, over 4.69 answer tokens.
  - The arms' strict unseen accuracy differs by 3.2 points, which doesn't resolve at this size.
- **SFT beats stock Instruct on the facts it trained on, not on the rest.**
  - On the 167 seen facts, strict closed-book accuracy is 28.7% against 11.4% (paired 95% CI +10.8
    to +24.6).
  - On the unseen facts it is 11.6% against 7.7%. That is past the 2.6-point floor, but the paired
    CI is −1.3 to +9.0, and the untrained base also scores 11.6%.
- **DPO and GRPO didn't clear their primary lines,** the metric each stage's pre-registered verdict
  reads.
  - Each raised seen pass@1 (8 samples, T 0.7) by about 3 points, a gain greedy serving doesn't
    use: DPO by 3.5 [+1.2, +6.0], in a line added after the fact, and GRPO by 3.1 [+1.3, +5.0].
  - Each paid in calibration on the gold answer's tokens. DPO lost 0.38 nats on unseen facts,
    beyond its 0.21 floor (0.11 on seen, inside it). GRPO lost 0.71 on seen and 1.32 on unseen.

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

**On unseen facts, probability went up once, at CPT:** +0.66 nats on the answer tokens for
`cpt-8b-replay10` (95% CI +0.45 to +0.90). Every later stage that can be measured lowered it,
DPO by 0.38 and GRPO by 1.32. On the facts SFT showed, SFT added accuracy in both arms: from 12.6%
to 23.9% without CPT, and from 15.0% to 28.7% with it.

**Serving:** FP8 (W8A8) passed the pre-registered quality gate.
- **Which variant:** bf16 serves up to the measured 16 req/s, with a faster first token; FP8
  serves near saturation.
- **Cost:** at peak goodput one H100 costs $0.035 per 1,000 requests with FP8 ($0.039 with bf16).
  The same tokens cost $0.127 through Mistral Small 4's API. That compares cost only, since
  Small 4 is a 119B MoE. The GPU is cheaper above 8.7 sustained req/s.
- **Bench numbers** are means of two runs ([`DEPLOY.md`](DEPLOY.md)).

**Final checkpoint: `dpo-strict`.**
- **How it was picked:** Stage 4's pre-registration made the DPO run's checkpoint final, with no
  fall-back to SFT. The move to strict labels was a user decision, and Stage 5's rule kept it
  when GRPO missed.
- **What it is:** the SFT model within noise on every primary line, with one cost beyond its
  floor: unseen answer tokens −0.38 nats. So `sft-from-cpt` would serve as well.
- **[Demo](#demo).**

## 2. Lifecycle

Row names are the tables' `run` column. Each arrow is labelled with the rule that picked the
checkpoint; dashed arrows are ablations and controls, which inform the chain but don't feed it.
The rule names are the pre-registration's:
- **ablation A** is the replay ablation;
- **B4** is the SFT rule that picks an epoch on the closed-book and definition losses;
- **stage5-final** is the checkpoint Stage 5 hands on.

```mermaid
flowchart LR
  base["base-8b-hf<br/>Ministral 3 8B Base"]
  cpt["cpt-8b-replay10<br/>CPT, LoRA r64, + 10% general replay"]
  sft["sft-from-cpt<br/>SFT, 2,436 records, 1,778 teacher-written"]
  dpo["dpo-strict<br/>DPO, 445 verifier-labelled pairs"]
  served["served: dpo-strict bf16 to 16 req/s,<br/>dpo-strict-fp8 near saturation"]
  sftb["sft-from-base<br/>control arm: the same SFT, no CPT"]
  grpo["grpo<br/>GRPO, 622 tasks: rejected"]
  full["cpt-8b-full<br/>full-parameter ablation"]
  dpo2["dpo-2ep<br/>2-epoch ablation"]
  base -->|"ablation A's rule: replay adopted"| cpt
  cpt -->|"B4: epoch 1, before the recall formats overfit"| sft
  sft -->|"checkpoint rule: final step; strict relabel by user decision"| dpo
  dpo -->|"quality gate: FP8 passes; served by load"| served
  base -.->|"control"| sftb
  base -.-> full
  sft -.-> dpo2
  dpo -.->|"best val pass@1: checkpoint-25; Stage 5 read sets it aside"| grpo
```

The stage write-ups, each with its curves, tables and what I'd do differently:
[0, the eval and baselines](docs/stage0.md) · [1, the corpus](docs/stage1.md) ·
[2, CPT](docs/stage2.md) · [3, SFT](docs/stage3.md) · [4, DPO](docs/stage4.md) ·
[5, GRPO](docs/stage5.md) · [6, serving](docs/stage6.md) · [every scored row](docs/results.md) ·
[verifiers and judges](docs/verifiers.md).

## 3. Results

Every stage and ablation, base first. Rates are in %, `gold_lp` is nats per gold answer (higher
is better), and ↓ marks lower is better.
- **Floors:** a difference counts only beyond the floor row of its format group. Base-format and
  chat-format `gold_lp` don't compare. Single-seed rows are read against the same floor.
- **Scorers:**
  - Closed-book accuracy uses the strict checker.
  - Judge columns are Mistral Large 3 at temperature 0, after rules. That judge was hand-checked,
    not benchmarked, and `cite_supported` is reported, not read ([why](docs/verifiers.md)).
  - Large 3 also wrote the eval and the SFT data, which could favour SFT on the seen half.
- **Benchmarks:** lm-eval, 5-shot, with no chat template and, since Stage 0, no BOS token. Rows
  compare with each other, not with published scores. Stage 6's GSM8K gate ran with one BOS
  ([docs/stage6.md](docs/stage6.md)).
- **Latency:** compare rows within one source only, either Stage 6's pinned-H100 bench or the
  earlier single samples.
- **Seen and unseen** are fixed per fact before SFT:
  - about half the eval's source passages may feed SFT synthesis, and half may not;
  - every eval fact comes from a document CPT read, so unseen means unseen by SFT;
  - this follows Tülu 3's development/unseen split (§7), here by fact within one eval.
- **`cpt-8b-full`** has no closed-book numbers: its weights were deleted before the task grew.

**Knowledge (closed-book: no retrieval, no passage in the prompt)**

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

**Behaviour (with passages: grounded answers, citations, abstention, definitions)**

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

**General capability and serving**

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

The lenient scorer's columns, MMLU's four groups, the train-slice and 2026-report perplexities and
the Stage 0 `base-8b` row are in [`docs/results.md`](docs/results.md).

## 4. Decisions and trade-offs

Each decision is listed with the alternative it beat, and the row that shows the alternative's
number:
- **10% general replay, kept on thin evidence** (against no replay, `cpt-8b`).
  - The rule fired on MMLU +0.2 against a 0.1 seed floor, inside MMLU's own 0.34 SE.
  - Its −2.24% general perplexity is in-distribution: FineWeb-Edu is both the replay and the
    general val.
  - Grounded accuracy fell 6.5 points against a 3.7 floor; the runs without replay fell 0.9 and
    4.6.
  - Kept as cheap (+10% tokens), and SFT restored the citations. Row: `cpt-8b-replay10`.
- **LoRA, not full-parameter CPT,** which memorised more for the same domain gain.
  - Train-slice perplexity fell 22%, against LoRA's 8%.
  - General perplexity rose 1.19%, against LoRA's 0.40% and 0.17% over two seeds.
  - GSM8K fell 3.0 points against a 1.1 floor, though grounded and vocab rose (one run).
  - Row: `cpt-8b-full`.
- **SFT stopped at epoch 1** by the pre-registered rule B4, which takes epoch 2 unless the
  closed-book or definition validation loss rises in it.
  - On `sft-from-cpt` they rose from 1.17 to 1.30 and from 1.98 to 2.03.
  - All four runs did the same.
  - Row: `sft-from-cpt`.
- **DPO labels from verifiers, not the judge.** The judge caught 28% of grounded and 9% of
  definition defects, against a 0.5 line. Relabelling with the strict checker (79 of 458
  closed-book labels wrong) made `dpo-strict`. Row: `dpo-strict`.
- **GRPO rejected.**
  - Its gain, seen pass@1 +3.1, is one greedy serving doesn't use. Greedy accuracy didn't move.
  - Calibration cost 0.71 nats seen and 1.32 unseen, beyond its floors.
  - On Magistral's recipe (DAPO loss, ε_high 0.28, no KL term) with one update per batch,
    clip-higher never bound (`clip_ratio/high` 0).
  - Row: `grpo`.
- **FP8 (W8A8), not INT4 or an FP8 KV cache.** FP8 moved no gated line past its Stage 3 floor.
  - FP8-KV failed strict seen accuracy (−4.2 against 3.5).
  - INT4 failed three gated lines: GSM8K −6.1 against 2.2, and answer-token log-probability at
    about twice the floor on both halves.
  - bf16 stays the default up to 16 req/s (first token 21-40 against 39-76 ms p50); FP8 serves 22%
    more requests per second at 64 concurrent.
  - Row: `dpo-strict-fp8`.

## 5. What changes at Forge scale

Forge's announcement names the same stages in its own words: pre-training on internal data
(continued pre-training, CPT, here), post-training, reinforcement learning, and evaluation
frameworks.

| | This repo, measured | [Forge](https://mistral.ai/news/forge/), as announced (17 March 2026) |
|---|---|---|
| Data | 246 public PDFs, 20.6M tokens after cleaning and deduplication | "large volumes of internal documentation, codebases, structured data, and operational records" |
| Data licences | audited after training: 54 of the 2,516 SFT records under non-commercial or restrictive terms | not stated |
| Pre-training | LoRA r64 on the 19.4M-token train split, read once, + 10% general replay | "build domain-aware models by learning from large internal datasets" |
| Post-training | SFT on 2,436 records (1,778 teacher-written); DPO on 445 verifier-labelled pairs | "post-training methods allow teams to refine model behavior" |
| RL | 622 verifiable tasks, synchronous GRPO, stopped by step 65 | "align models and agents with internal policies, evaluation criteria" |
| Evaluation | a 716-item KPI eval (fixed from Stage 3), a regression suite, seed floors | "test models against internal benchmarks, compliance rules, and domain-specific tasks" |
| Compute | H100s, one per run except two 2-GPU ablations: 8.8 GPU-hours, $35 of training | not stated |

What the small version surfaced that gets harder at full scale:
1. **The judge comes back.** Verifiers replaced a failed judge here because the tasks had exact
   answers. Open-ended client tasks don't, so calibrating a judge against subject-matter experts'
   labels becomes the main work.
2. **Verifiers are adversarial objects.** Three holes in one day, and one was learned. Every reward
   needs a fixture suite and an audit of what it rewards before training.
3. **Synchronous GRPO had no working brake.** Clip-higher acts on the policy gap that asynchronous
   generators open; here the gap was zero, and both runs collapsed. Whether a brake would have
   stopped that is untested.
4. **Replay needs lineage.** Here an exclusion list kept Claude-written subsets out of a public
   mix. With client data, which records went into which run has to be tracked.
5. **The seen/unseen split needs an owner.** Which facts training may see is a governance decision;
   here one script enforced it, and a guard test checked it.
6. **Licences get checked at intake.** Here the audit ran after training and found records that
   block publishing the weights. At scale, lineage is checked before any GPU time.
7. **Multi-node full-parameter CPT wasn't done.** There was one 2-GPU full-parameter run, and one
   2-GPU LoRA run, which was 11% faster per GPU with the cause not isolated.

## 6. What went wrong, and what next

**Failures:**
1. **The judge failed its benchmark** (recall 0.28 grounded, 0.09 definition), so Stage 4's
   registered design couldn't run.
2. **The substring labeller passed wrong answers:** 79 of 458 closed-book chosen answers (17%).
   - The strict checker relabelled DPO and re-scored every row. The 8B rows moved 0–1.6 points
     and Large 3 moved 1.9.
   - Five pairs of rows less than a point apart swapped order. None of them is a comparison any
     read makes.
3. **Four rules sat on point readings where a window was meant:** the merge gate, the step-1 loss
   band, the DPO checkpoint rule, and entropy on one batch. Three were amended, two of them after
   their read (index in `notes/decisions.md`).
4. **Three verifier holes:**
   - fragments, found by auditing passes;
   - comma lists, found by reading the code;
   - years within 2%, found by the hack audit. It was the only one learned: 10 of 16 late
     rollouts on one task.
5. **Two DPO epochs displaced the chosen answers:** seen accuracy fell 4.8 points, and gold
   log-probability fell 3.0 and 4.3 nats (`dpo-2ep`, one run).
6. **GRPO collapsed on entropy in both runs,** with the clip inert.
7. **INT4 failed three gated lines.** Its 512-record, domain-only calibration is the hypothesis,
   untested.
8. **FP8 doubles first-token time at low load** (17.9 to 35.2 ms p50), still unexplained. The two
   slow GEMM paths vLLM's docs name are ruled out.
9. **Prefix caching cut grounded first-token time by 29%,** not the half predicted. Only the
   prompt-length part is cacheable, and that is 40% on FP8.
10. **lm-eval has sent no BOS since Stage 0.** Rows still compare with each other: GSM8K with one
    BOS scored the same 1,067 of 1,319 on bf16 `dpo-strict`.
11. **The licence audit ran after training.** 54 of the 2,516 SFT records (8 non-commercial, 46
    restrictive) block publishing the weights ([Weights](#weights)).

The verifier story, with each hole's fixture and the rule the repo ended with, is in
[`docs/verifiers.md`](docs/verifiers.md). A verifier gets an adversarial fixture suite before it
becomes a reward, and its passes are read before its scores are believed.

**Method lessons:**
- **Stop rules on windows, not points.** One batch's entropy stopped `grpo-seed1` at step 34 with
  its 10-step mean at 57% of the start. The rule was rewritten after it fired, and the run resumed
  to step 65.
- **Decompose a metric before comparing it.** `gold_lp`'s end token hid DPO's shift toward
  stopping.
- **The start's own seed gap belongs in a two-arm floor.** Adding it, after the read, moved Stage
  4's hallucination line and the as-run unseen `gold_lp` inside the noise.
- **Say "1 df per arm" out loud.**
- **Fix the eval's primary metric before the first training run.** Stage 2's −20% perplexity
  target was set for the wrong data scale, and the seen/unseen split arrived after Stage 2.

**Next, ranked:**
1. **A licence-clean SFT set:** OASST1 for No Robots, and a unrestricted math source with no GSM8K
   test overlap. Then SFT, DPO and FP8 again, the gate rows, and the weights.
2. **Compute tasks for GRPO** (a formula from a passage, sampled inputs, a checked answer): a
   skill RL could sharpen, untested here.
3. **Two updates per batch (μ = 2),** so clip-higher can bind; then a KL term if needed.
4. **An NLL anchor on DPO's chosen answers (RPO).**
5. **A per-claim support check as the grounded judge,** benchmarked first.
6. **Diagnose FP8's first token, and run Stage 6's unrun rows:** FP8-KV's GSM8K and second seen
   line, and n-gram speculative decoding.
7. **INT4 with mixed calibration, or AWQ.**
8. **A third seed per arm.**
9. **Varied CPT exposure** (paraphrased restatements). One pass over 19.4M tokens raised gold
   answers' probability without making them answerable.

## 7. Reproduce

[`docs/reproduce.md`](docs/reproduce.md) has one command chain per stage and pins the versions
and dataset hashes. It also says what lives in git, on the Modal volume and on Hugging Face.

<!-- reproduce-table:start -->
| stage | command | training GPU-h | training $ | data (sha256 of its `SHA256SUMS`) |
|---|---|---|---|---|
| 0: the eval and baselines | `make reproduce-stage0` |  |  | `eval/tasks/` (committed, reviewed) |
| 1: the corpus | `make reproduce-stage1` |  |  | `data/processed`: 966d1e0c |
| 2: CPT | `make reproduce-stage2` | 5.40 | 21.32 | the corpus |
| 3: SFT | `make reproduce-stage3` | 1.97 | 7.76 | `data/sft`: 70f47740 |
| 4: DPO | `make reproduce-stage4` | 0.37 | 1.49 | `data/dpo/strict`: 5e3effaf |
| 5: GRPO | `make reproduce-stage5` | 1.08 | 4.28 | `data/grpo`: 4db8f7a6 |
| 6: serving | `make reproduce-stage6` | not totalled |  | `serve/bench_manifest.json` |
| all training | | 8.82 | 34.85 | |
<!-- reproduce-table:end -->

Dollars are at $3.95 per H100-hour (Modal's list price, checked 2026-10-09). Evals, sampling, the
Stage 6 bench and the Mistral API calls aren't totalled anywhere. Stages 2-6 launch Modal GPU
jobs. The bench pins `gpu="H100!"`, since a plain "H100" request can land on an H200.

**Checking the numbers needs no GPU and no API key.** Every row's generations, lm-eval outputs and
judge verdicts are committed. `make reproduce-score` rescores them, re-runs the strict checker and
pass@k, and regenerates every table and figure. `make audit` checks that each number in this
README's prose appears in a generated table or a named file.

## Who wrote the data, licences, weights

**Who wrote the data:**
- **The eval items:** generated by Mistral Large 3. Each was reviewed against its source passage,
  with keep or reject verdicts ([rubric](notes/eval_review_rubric.md)).
- **SFT:** Large 3 (`mistral-large-2512`) wrote the domain questions and, with Medium 3.5
  (`mistral-medium-2604`, 25%), the completions. The abstain records use one fixed sentence.
  - 500 general records come from the Tülu 3 SFT mixture, minus its Claude-written subsets.
  - Those records' own generators are GPT-4o, GPT-3.5/4, Mixtral and people.
- **DPO pairs:** `sft-from-cpt`'s own samples, labelled by verifiers and rules.
- **GRPO:** the policy's own rollouts, scored by rules.
- **Claude**, through Claude Code, built the tooling and reviewed generated records with keep or
  drop verdicts only. No training set contains text Claude wrote. The answers in `tests/` are
  fixtures and are never trained on.

**Licences:**

| what | licence | in this repo |
|---|---|---|
| code | Apache-2.0 ([`LICENSE`](LICENSE)) | all of it |
| the corpus: 246 US federal documents | public domain (17 U.S.C. § 105); ASCE 7, the AISC manual and other copyrighted standards excluded | URLs and sha256 in [`data/sources.csv`](data/sources.csv); the PDFs aren't redistributed |
| FineWeb-Edu (CPT replay, general val) | ODC-By | not redistributed |
| Mistral Large 3 / Medium 3.5 output (eval items, SFT records) | assigned to the customer by Mistral's Commercial Terms (§3.1), labelled as model-written (§3.2) | committed |
| Tülu 3 SFT mixture: 500 replay records, 50 held-out probe prompts | ODC-BY-1.0 as a collection; subsets below | row ids only (`data/sft/hosted/`, `eval/hosted/`; results keep sha256s of their prompt ids); `make sft-replay` rebuilds the files and checks their sha256. Commits before `653f932` still hold the text, since the repo was public before the change. |
| Ministral 3 8B Base | Apache 2.0 (model card) | not redistributed |

The replay records by subset, with the licence the mixture's card gives each:

<!-- replay-licences:start -->
| subset | records | licence |
|---|---|---|
| Evol CodeAlpaca | 97 | Apache 2.0 |
| NuminaMath-TIR | 58 | Apache 2.0 |
| WildGuardMix | 46 | Apache 2.0 |
| OASST | 4 | Apache 2.0 |
| WildJailbreak | 46 | ODC-BY-1.0 |
| Persona GSM | 46 | ODC-BY-1.0 |
| WildChat (GPT-4) | 36 | ODC-BY-1.0 |
| Persona Algebra | 18 | ODC-BY-1.0 |
| CoCoNot | 10 | ODC-BY-1.0 |
| SciRIFF | 7 | ODC-BY-1.0 |
| TableGPT | 4 | MIT |
| FLAN v2 | 74 | not given on the card |
| withdrawn math set (GSM8K) | 46 | withdrawn after training |
| No Robots | 8 | CC-BY-NC-4.0 (non-commercial) |
<!-- replay-licences:end -->

### Weights

**Not published at v1.0.**
- **Why:** they were trained on the last two subset rows above. No Robots is non-commercial, so
  an Apache-2.0 release would misstate the licence. Its licence restricts how models trained on its outputs may be named, so a non-commercial licence alone
  doesn't cover it.
- **The fix:** swap those 54 records (OASST1 for No Robots; a unrestricted math source with no GSM8K
  test overlap). Then retrain SFT, DPO and FP8 and rerun the gate rows.
- **Until then:** the checkpoints stay on the Modal volume, rebuildable from the adapters.

### Demo

`serve/modal_demo.py` serves `dpo-strict-fp8` on one H100, and `serve/demo.py` sends it two eval
prompts, built as the eval builds them. Both items are hand-picked, so they show the behaviour;
the rates are in the tables.
- **`gr-0038`:** of the 93 grounded items `dpo-strict` answered correctly with valid,
  judge-supported citations, the three with the shortest passages were read. This one was kept for
  its concrete answer.
- **`adv-0050`:** the third-shortest of the 73 unanswerable items it declined, kept as the clearest
  question out of context.

In both the bf16 and the FP8 run's saved answers:
- `gr-0038` passes every rule: its one citation is the gold passage, the strict grounded check
  GRPO's reward uses, and the judge marked it correct and supported.
- `adv-0050` gives the exact abstain sentence.
- The closed-book strict checker doesn't apply to either.
