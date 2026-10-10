# Results


Every scored checkpoint, from [`results/table.md`](../results/table.md) (copied here by
`train/report.py`), on eval v3: domain_qa has 322 items (v2's 325 minus the three that shared a
few-shot item's passage), the other tasks are unchanged. Every row was rescored on v3 from its
saved generations. The earlier tables are frozen:
- [`results/table_v2.md`](../results/table_v2.md): the 325-item set, as Stage 2 was first read;
- [`results/table_v1.md`](../results/table_v1.md) and [`results/table_v1.1.md`](../results/table_v1.1.md):
  the 130-item set.

Stage 2 rows are base models, scored without `--chat`; Instruct and the SFT rows are chat models,
scored with it. `qa_term` covers 38 items (2.6 points each), so read it as counts
([Stage 3](stage3.md) gives them).

**Reading the closed-book numbers.** The `qa_*` and `gold_lp` columns ask for facts from specific
pages of the manuals (a value, a document or article number, a term) with no retrieval and no
passage in the prompt.
- **The gap is the point.** With passages from the same documents in the prompt (the grounded
  task), the base model answers 84% correctly; closed-book, 12%. That gap is the knowledge the
  documents hold and the model doesn't, and these columns measure what training does to it.
- **For scale:** Mistral Large 3 answers 28% of the same 322 questions closed-book (row
  `mistral-large-3`, through the API, closed-book only): 45% of identifiers, 25% of values and 18%
  of terms. On SimpleQA, whose facts are far more common, frontier models score 30-40%.
- **The scores are knowledge, not scoring:** a hand audit of 169 wrong answers found 2 scoring
  errors, both fixed.
- **The strict re-score (2026-10-09):** `qa_acc`'s scorer passes a few wrong answers: a range for a
  point value, a fraction read as its first number, a child section by containment.
  - Re-scored with the strict checker written for Stage 5's reward, each 8B row drops 0 to 1.6
    points and Large 3 1.9 ([`results/qa_strict/evals.md`](../results/qa_strict/evals.md)).
  - No ordering changes, so the table keeps `qa_acc`; Stage 5's read uses the strict column.
- **The column to watch is Stage 3's seen half.** SFT synthesis supplies exposures to those facts;
  the unseen half shows whether anything transfers.

- **Task scores** come from the frozen eval, with judge columns scored locally:
  - **Two scorers:** every `qa_*` column is shown lenient (`scorers.qa_correct`, what
    `results/table.md` stores) and strict (`scorers.qa_strict`, 2026-10-09, re-scored from the saved
    generations by `eval/qa_strict.py`). The strict column is the read from Stage 5 on.
  - `qa_num`, `qa_ident` and `qa_term` split `qa_acc` by answer kind (`eval/qa_rules.py`):
    values, identifiers (document ids and article numbers) and terms, which include everything
    else. A hand audit of 169 misses found 2 scoring errors, both fixed, so the low scores are
    genuine.
  - The `_seen` / `_unseen` columns split `qa_acc` and `vocab_recall` by whether Stage 3's SFT
    synthesis may use the item's source chunk (`eval/sft_split.py`). Before Stage 3 nothing is
    seen, so the two halves (167 and 155 QA items) are a null check.
  - `false_abstain` is the share of grounded answers that use the abstain phrase although the
    passages hold the answer: the cost side of a low `halluc_rate`.
  - `cpt-8b-full` and the Stage 0 `base-8b` have no QA scores on v2 or v3. Full's weights were deleted
    before the task grew, and `base-8b-hf` supersedes the Stage 0 row.
- **Gold-answer log-probability** (`gold_lp`, nats per answer, higher is better,
  `eval/gold_lp.py`) is the continuous companion to `qa_acc` on the same items. Instruct and the
  SFT rows are scored in their chat format, so their values compare with each other, not with the
  base-format rows.
  - It is the sum of two parts: the answer tokens, and the one end token the prompt expects after an
    answer.
  - From 2026-10-09 every `gold_lp` row shows both parts, because a stage can change the answer's
    format (whether the model stops after the gold) without changing the fact.
  - The composite was the right single number in Stage 3, where stopping was the failure measured.
- **Benchmarks** are from lm-eval: 5-shot, never with the chat template, and without a BOS token.
  The frozen flags send none, as Stage 6 found (`eval/bos_probe.py`). Every row since Stage 0 ran
  the same way, so every comparison between rows stands, but no absolute MMLU, GSM8K or HellaSwag
  number here is comparable with a published one.
- **The Stage 6 rows** (`dpo-strict-fp8`, `-fp8kv`, `-w4a16`) are the quantized serving variants of
  `dpo-strict`. They have no lm-eval or perplexity columns; their GSM8K (with one BOS) and vLLM
  perplexity lines are in the Stage 6 gate under [Serving](stage6.md).
- **Perplexity** is from `eval/perplexity.py` (lower is better). `ppl_postcutoff` covers the 13
  federal reports published after the base model.

<!-- results-table:start -->
Items per task: domain_qa 322, grounded 108, vocab 210, adversarial 76, qa_number 220, qa_identifier 64, qa_term 38.

**Closed-book knowledge: no retrieval, no passage in the prompt; questions about facts on specific pages of the manuals (gold_lp: nats per answer, higher is better)**

| run | gold_lp | gold_lp_seen | gold_lp_unseen | qa_acc (lenient) | qa_num (lenient) | qa_ident (lenient) | qa_term (lenient) | qa_seen (lenient) | qa_unseen (lenient) | qa_acc (strict) | qa_num (strict) | qa_ident (strict) | qa_term (strict) | qa_seen (strict) | qa_unseen (strict) | gold_lp answer | gold_lp answer seen | gold_lp answer unseen | gold_lp end | gold_lp end seen | gold_lp end unseen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-8b-hf | -6.770 | -6.722 | -6.822 | 0.121 | 0.145 | 0.078 | 0.053 | 0.126 | 0.116 | 0.121 | 0.145 | 0.078 | 0.054 | 0.126 | 0.116 | -6.324 | -6.278 | -6.356 | -0.453 | -0.444 | -0.466 |
| instruct-8b | -8.284 | -8.008 | -8.581 | 0.099 | 0.127 | 0.031 | 0.053 | 0.114 | 0.084 | 0.096 | 0.123 | 0.031 | 0.054 | 0.114 | 0.077 | -7.496 | -7.313 | -7.709 | -0.775 | -0.695 | -0.872 |
| cpt-8b | -6.184 | -6.237 | -6.128 | 0.146 | 0.168 | 0.125 | 0.053 | 0.156 | 0.136 | 0.146 | 0.168 | 0.125 | 0.054 | 0.156 | 0.136 | -5.764 | -5.838 | -5.677 | -0.422 | -0.399 | -0.450 |
| cpt-8b-seed1 | -6.192 | -6.236 | -6.145 | 0.130 | 0.145 | 0.125 | 0.053 | 0.132 | 0.129 | 0.130 | 0.145 | 0.125 | 0.054 | 0.132 | 0.129 | -5.773 | -5.838 | -5.697 | -0.420 | -0.397 | -0.447 |
| cpt-8b-replay10 | -6.260 | -6.356 | -6.157 | 0.130 | 0.145 | 0.125 | 0.053 | 0.150 | 0.110 | 0.130 | 0.145 | 0.125 | 0.054 | 0.150 | 0.110 | -5.821 | -5.931 | -5.697 | -0.440 | -0.425 | -0.460 |
| mistral-large-3 |  |  |  | 0.280 | 0.245 | 0.453 | 0.184 | 0.275 | 0.284 | 0.261 | 0.232 | 0.406 | 0.189 | 0.264 | 0.258 |  |  |  |  |  |  |
| sft-from-base | -6.069 | -5.409 | -6.780 | 0.180 | 0.227 | 0.125 | 0.000 | 0.245 | 0.110 | 0.165 | 0.209 | 0.109 | 0.000 | 0.239 | 0.084 | -5.562 | -4.973 | -6.197 | -0.507 | -0.436 | -0.583 |
| sft-from-cpt-seed1 | -5.497 | -4.873 | -6.168 | 0.202 | 0.209 | 0.234 | 0.105 | 0.270 | 0.129 | 0.186 | 0.186 | 0.234 | 0.108 | 0.258 | 0.110 | -5.083 | -4.524 | -5.684 | -0.414 | -0.349 | -0.484 |
| sft-from-cpt | -5.749 | -5.120 | -6.426 | 0.211 | 0.245 | 0.203 | 0.026 | 0.281 | 0.136 | 0.205 | 0.236 | 0.203 | 0.027 | 0.287 | 0.116 | -5.273 | -4.697 | -5.895 | -0.475 | -0.423 | -0.531 |
| sft-from-base-seed1 | -5.967 | -5.249 | -6.741 | 0.168 | 0.196 | 0.141 | 0.053 | 0.234 | 0.097 | 0.158 | 0.182 | 0.141 | 0.054 | 0.234 | 0.077 | -5.509 | -4.869 | -6.199 | -0.458 | -0.380 | -0.542 |
| dpo-seed1 | -5.935 | -5.196 | -6.731 | 0.211 | 0.250 | 0.172 | 0.053 | 0.287 | 0.129 | 0.202 | 0.236 | 0.172 | 0.054 | 0.287 | 0.110 | -5.538 | -4.848 | -6.282 | -0.397 | -0.348 | -0.449 |
| dpo | -5.745 | -5.003 | -6.544 | 0.227 | 0.264 | 0.203 | 0.053 | 0.311 | 0.136 | 0.214 | 0.245 | 0.203 | 0.054 | 0.305 | 0.116 | -5.406 | -4.705 | -6.161 | -0.339 | -0.298 | -0.383 |
| dpo-2ep | -9.376 | -8.012 | -10.845 | 0.199 | 0.236 | 0.141 | 0.079 | 0.264 | 0.129 | 0.196 | 0.232 | 0.141 | 0.081 | 0.258 | 0.129 | -9.142 | -7.816 | -10.570 | -0.234 | -0.196 | -0.276 |
| dpo-strict | -5.883 | -5.130 | -6.695 | 0.217 | 0.255 | 0.188 | 0.053 | 0.299 | 0.129 | 0.211 | 0.245 | 0.188 | 0.054 | 0.305 | 0.110 | -5.516 | -4.811 | -6.274 | -0.368 | -0.319 | -0.421 |
| grpo | -6.829 | -5.784 | -7.956 | 0.205 | 0.227 | 0.203 | 0.079 | 0.293 | 0.110 | 0.202 | 0.223 | 0.203 | 0.081 | 0.299 | 0.097 | -6.491 | -5.517 | -7.541 | -0.338 | -0.267 | -0.415 |
| grpo-seed1 | -6.917 | -5.819 | -8.101 | 0.199 | 0.223 | 0.188 | 0.079 | 0.281 | 0.110 | 0.199 | 0.223 | 0.188 | 0.081 | 0.287 | 0.103 | -6.543 | -5.519 | -7.647 | -0.374 | -0.299 | -0.454 |
| dpo-strict-fp8 | -5.898 | -5.141 | -6.715 | 0.208 | 0.241 | 0.172 | 0.079 | 0.293 | 0.116 | 0.199 | 0.227 | 0.172 | 0.081 | 0.293 | 0.097 | -5.519 | -4.818 | -6.274 | -0.380 | -0.323 | -0.441 |
| dpo-strict-fp8kv | -5.957 | -5.194 | -6.780 | 0.202 | 0.236 | 0.172 | 0.053 | 0.264 | 0.136 | 0.193 | 0.223 | 0.172 | 0.054 | 0.264 | 0.116 | -5.571 | -4.867 | -6.330 | -0.386 | -0.327 | -0.450 |
| dpo-strict-w4a16 | -6.211 | -5.421 | -7.061 | 0.183 | 0.223 | 0.109 | 0.079 | 0.270 | 0.090 | 0.183 | 0.223 | 0.109 | 0.081 | 0.275 | 0.084 | -5.879 | -5.118 | -6.699 | -0.332 | -0.304 | -0.362 |

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
| dpo-seed1 | 0.907 | 1.000 | 0.880 | 0.026 | 0.000 | 0.848 | 0.891 | 0.807 |
| dpo | 0.917 | 1.000 | 0.870 | 0.013 | 0.000 | 0.838 | 0.891 | 0.789 |
| dpo-2ep | 0.907 | 1.000 | 0.852 | 0.013 | 0.000 | 0.852 | 0.921 | 0.789 |
| dpo-strict | 0.926 | 1.000 | 0.861 | 0.040 | 0.000 | 0.824 | 0.861 | 0.789 |
| grpo | 0.935 | 1.000 | 0.880 | 0.013 | 0.000 | 0.843 | 0.881 | 0.807 |
| grpo-seed1 | 0.917 | 0.982 | 0.889 | 0.013 | 0.018 | 0.852 | 0.881 | 0.826 |
| dpo-strict-fp8 | 0.907 | 1.000 | 0.852 | 0.053 | 0.000 | 0.829 | 0.881 | 0.780 |
| dpo-strict-fp8kv | 0.935 | 1.000 | 0.843 | 0.066 | 0.000 | 0.833 | 0.881 | 0.789 |
| dpo-strict-w4a16 | 0.898 | 1.000 | 0.843 | 0.040 | 0.000 | 0.776 | 0.881 | 0.679 |

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
| dpo-seed1 | 0.765 | 0.726 | 0.701 | 0.858 | 0.811 | 0.807 | 0.795 | 5.81 | 6.89 | 8.07 | 6.30 |
| dpo | 0.767 | 0.729 | 0.703 | 0.859 | 0.814 | 0.810 | 0.796 | 5.80 | 6.88 | 8.07 | 6.30 |
| dpo-2ep | 0.767 | 0.730 | 0.701 | 0.862 | 0.812 | 0.806 | 0.802 | 5.89 | 6.97 | 8.11 | 6.38 |
| dpo-strict | 0.767 | 0.730 | 0.701 | 0.861 | 0.813 | 0.809 | 0.796 | 5.81 | 6.88 | 8.06 | 6.30 |
| grpo | 0.767 | 0.728 | 0.703 | 0.859 | 0.814 | 0.820 | 0.798 | 5.84 | 6.93 | 8.09 | 6.34 |
| grpo-seed1 | 0.767 | 0.728 | 0.702 | 0.859 | 0.813 | 0.802 | 0.798 | 5.84 | 6.93 | 8.09 | 6.34 |

qa_* (lenient) is `scorers.qa_correct`, the column `results/table.md` stores; qa_* (strict) is `scorers.qa_strict` (2026-10-09, the GRPO reward's rule: the whole gold, one candidate, units compared), re-scored from every row's saved generations by `eval/qa_strict.py`. No ordering changes between the two.
<!-- results-table:end -->

**Stage 2 (CPT) earned little.**
- **Perplexity:** held-out perplexity fell 2.3%, and documents published in 2026 gained 0.4%.
- **Task scores:** no pass/fail score moved outside the noise.
- **Gold answers:** closed-book gold answers to facts from the documents it read became about 1.8x
  more probable (`gold_lp` +0.59 nats per answer, +0.56 of it in the answer tokens), which SFT can
  build on.
- **Why `cpt-8b-replay10` goes forward:** ablation A's rule adopted replay on MMLU +0.2 against a
  0.1 seed floor, which is noise (MMLU's own SE is 0.34). Its general-perplexity gain (−2.24%) is
  in-distribution, since the general val is FineWeb-Edu, the replay's source, and it cost
  grounded_acc −6.5 against a 3.7 floor. Kept: cheap, and matching the main run elsewhere.
  *(Corrected 2026-10-10: this read "LoRA with replay costs almost nothing in forgetting".)*

**Stage 3 (SFT):**
- **Holds:**
  - The model stops: 159 ms against 1,753 ms end to end.
  - It cites better than Instruct and never refuses an answerable question.
  - The recall formats overfit in epoch 2, which the pre-registered rule caught on every run.
- **Retention, not capability:**
  - The seen half's closed-book score is retention: 28.7% from 15.0% on the strict checker, on
    facts that were in the training set.
  - The unseen half moved inside the noise.

  | closed-book, strict checker | seen half (167) | unseen half (155) | identifiers (64) |
  |---|---|---|---|
  | instruct-8b | 11.4% | 7.7% | 3.1% |
  | sft-from-cpt | 28.7% | 11.6% | 20.3% |

  (Lenient, as first reported: 11.4% / 8.4% / 3.1% and 28.1% / 13.5% / 20.3%; identifiers don't
  change under the strict rule. No ordering changes.)
- **Attribution:** SFT did its job, behaviour and the facts it was shown, and did not erase CPT's
  knowledge. It did not generalise to unseen facts, which was never a target; only more varied CPT
  exposure acts on those.
- **CPT's contribution survives SFT:**
  - unseen-half `gold_lp` is +0.46 nats over the SFT-only control (arm means of two seeds each),
    +0.41 of it in the answer tokens: knowledge, not format;
  - every CPT-arm run is above every base-arm run (1 in 6 by permutation);
  - the 3.6 SD multiple is indicative: each arm's run variance rests on one seed pair (1 df per
    arm);
  - pass/fail closed-book accuracy doesn't resolve it, strict or lenient.
- **Going forward:** `sft-from-cpt` (epoch 1) is the Stage 3 checkpoint; `train/configs/dpo.yaml`
  starts from it.

**Stage 4 (DPO on verifiable preferences):**
- **One epoch is indistinguishable from the SFT start** on every pre-registered line, with the
  start's own seed gap in the floor.
  - Sampled accuracy, measured after the fact (a reported line): seen pass@1 +3.1 points, pass@8
    flat.
  - Split after the fact: on the answer tokens the unseen cost is −0.33 nats, beyond the floor; the
    end token rose +0.12 (a format shift the composite netted against it).
  - Read as written, hallucination (4 → 1-2 of 76) and unseen gold_lp (−0.21 nats) cleared the
    noise at 1.3× and 1.1×.
  - 58 of 100 greedy answers are byte-identical to SFT's.
- **Two epochs (`dpo-2ep`, ablation) fit the pairs and displaced the chosen answers.**
  - seen qa_acc −4.8 points against one epoch (strict and lenient alike);
  - gold-answer log-probability −3.0 nats (seen) and −4.3 nats (unseen);
  - dpo_val `rewards/chosen` −0.23.
  - That is the RLHF Book's ch. 8 preference displacement, measured on verifier-labelled pairs.
- **The judge failed its benchmark:** recall 0.28 (grounded) and 0.09 (definition), position and
  self-preference biases, pairwise order agreement at chance. Every pair is verifier- or
  rule-labelled: 506, against 114 as registered, 458 of them closed-book.
- **The label audit (2026-10-09):**
  - 79 of the 458 closed-book chosen labels were wrong under the strict checker written before
    Stage 5's reward.
  - Retrained on strict labels, `dpo-strict` is still SFT within noise on seen accuracy and
    hallucination. Label noise was not why DPO didn't move.
  - Its unseen gold-answer log-probability fell 0.27 nats, at the floor. Split after the fact, that
    is −0.38 on the answer tokens (beyond the floor) behind an end-token gain of +0.11: DPO also
    shifted the format toward stopping.
- **Going forward:** `dpo-strict` (seed 0, strict labels) is the Stage 4 checkpoint and
  `train/configs/grpo.yaml` starts from it. Stage 5 compares offline pairs against on-policy groups
  on the same strict verifier.

**Stage 5 (GRPO with verifiable rewards):**
- **Training:** 622 in-window tasks with a rule-based reward. Both seeds collapsed (stopped at 52
  and 65) and kept step 25.
- **Sharpening, not knowledge** (against `dpo-strict`):
  - greedy accuracy unchanged;
  - sampled pass@1 +3.1 points on seen items, with pass@8 −3.6 (inside the noise);
  - gold-answer log-probability −0.67 nats seen and −1.33 unseen, all in the answer tokens;
  - hallucination 3 → 1 of 76 is inside what reseeding does.
- **Why it collapsed:** the colocated loop is synchronous with one update per batch, so
  Magistral's clip-higher had no ratio to bind, and β was 0.
- **Three verifier holes caught in one day:**
  - fragments, by auditing the checker's passes;
  - comma lists, by reading its code;
  - years within 2%, by the hack audit.
  - Only the last was learned, on one task.
- **Going forward:** `stage5-final` stays `dpo-strict`.
  - CPT and SFT delivered; DPO and GRPO at this scale did not clear the floor on the primary lines.
    GRPO's checkpoint was set aside; DPO's carried forward as the SFT model within noise.
    *(Corrected 2026-10-10: this read "the pre-registered rules rejected both"; no rule set
    DPO's checkpoint aside.)*
  - Both sharpened sampled accuracy by about 3 points from their own starts; DPO did so at
    a small calibration cost (0.11 / 0.38 nats on the answer tokens), GRPO at a large one (0.71 /
    1.32).
  - Stage 6 serves `dpo-strict`, the SFT model within noise, in bf16 or FP8 (W8A8) by load: FP8
    passed the pre-registered quality gate on every line; INT4 and an FP8 KV cache did not
    ([Serving](stage6.md)).

**What the whole chain shows: on unseen facts, probability went up once, at CPT.** Every number in
this paragraph is
on the gold answer's tokens alone (the end token excluded), unseen half, with item-bootstrap 95%
CIs.
- **CPT is the only stage that raised the probability of unseen gold answers:**
  - +0.66 nats in Stage 2 (`cpt-8b-replay10`, the chain's CPT, [+0.45, +0.90]; +0.56 over all
    items for `cpt-8b`);
  - +0.41 [+0.23, +0.60] across the Stage 3 arms, CPT's gain surviving SFT.
- **That gain is text familiarity, not recall:**
  - It is about what CPT's rise on all corpus text predicts: 0.086 nats per token over 4.69
    answer tokens (`results/summary_stats.json`).
  - The arms' strict unseen accuracy differs by 3.2 points, which doesn't resolve.
- **Every stage after it that can be measured lowered them:**
  - DPO by 0.38 [0.27, 0.50];
  - GRPO by 1.32 [1.08, 1.58].
  - SFT's own change can't be read, since it moves the model from base format to chat format.
- **The trade:** each later stage bought behaviour with some of that probability. The on-policy
  optimiser cost more than the offline one, on one run pair each.

*(Corrected 2026-10-10: this paragraph read "knowledge went in once, at CPT" before the per-token
check, notes/decisions.md, Stage 7.)*
