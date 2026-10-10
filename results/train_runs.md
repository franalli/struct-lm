# Stage 2: CPT

## Training runs

| run | GPUs | steps | tokens | tokens/s | tokens/s/GPU | wall (h) | GPU-h | $ | final train loss | final eval loss |
|---|---|---|---|---|---|---|---|---|---|---|
| cpt-8b | 1 | 149 | 19.4M | 5,939 | 5,939 | 0.95 | 0.95 | 3.74 | 1.769 | 1.906 |
| cpt-8b-replay10 | 1 | 164 | 21.4M | 5,980 | 5,980 | 1.04 | 1.03 | 4.09 | 1.770 | 1.906 |
| cpt-8b-full | 2 | 149 | 19.4M | 13,226 | 6,613 | 0.45 | 0.91 | 3.59 | 1.725 | 1.902 |
| cpt-8b-fsdp2 | 2 | 100 | 13.1M | 13,221 | 6,610 | 0.30 | 0.59 | 2.34 | 1.741 |  |
| cpt-8b-lr2x | 1 | 149 | 19.4M | 5,963 | 5,963 | 0.95 | 0.95 | 3.74 | 1.756 | 1.904 |
| cpt-8b-seed1 | 1 | 149 | 19.4M | 5,914 | 5,914 | 0.97 | 0.97 | 3.82 | 1.753 | 1.906 |

- **cpt-8b-fsdp2 vs cpt-8b:** 2 GPUs give 2.23x the tokens/s (13,221 vs 5,939), so per GPU +11% at the same micro-batch and per-layer checkpointing (peak memory 43 vs 51 GB): the FSDP2 code path, not scaling (finding 6 below); per-step losses differ by 0.038% (median) / 0.171% (max) over 100 steps, the 10-step moving averages by 0.01% on average.

$ at 3.95 per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09); wall time includes tokenising and model load.

## Perplexity vs base-8b

| run | domain val | change | nats [95% CI, docs] | general val | change | nats [95% CI, docs] | train slice | change |
|---|---|---|---|---|---|---|---|---|
| base-8b | 6.880 |  |  | 8.151 |  |  | 6.185 |  |
| cpt-8b | 6.720 | -2.33% | -0.0236 [-0.0302, -0.0180] | 8.184 | +0.40% | +0.0040 [+0.0025, +0.0056] | 5.673 | -8.28% |
| cpt-8b-replay10 | 6.720 | -2.33% | -0.0236 [-0.0303, -0.0180] | 7.969 | -2.24% | -0.0226 [-0.0257, -0.0198] | 5.646 | -8.71% |
| cpt-8b-full | 6.732 | -2.15% | -0.0217 [-0.0357, -0.0104] | 8.248 | +1.19% | +0.0118 [+0.0091, +0.0143] | 4.815 | -22.15% |
| cpt-8b-lr2x | 6.713 | -2.42% | -0.0245 [-0.0335, -0.0172] | 8.220 | +0.85% | +0.0085 [+0.0068, +0.0102] | 5.442 | -12.01% |
| cpt-8b-seed1 | 6.718 | -2.35% | -0.0238 [-0.0307, -0.0180] | 8.165 | +0.17% | +0.0017 [+0.0002, +0.0030] | 5.687 | -8.05% |
| dpo | 6.882 | +0.03% | +0.0003 [-0.0053, +0.0064] | 8.066 | -1.05% | -0.0105 [-0.0138, -0.0075] | 5.805 | -6.14% |
| dpo-2ep | 6.971 | +1.32% | +0.0131 [+0.0085, +0.0188] | 8.109 | -0.52% | -0.0053 [-0.0085, -0.0022] | 5.894 | -4.70% |
| dpo-seed1 | 6.885 | +0.08% | +0.0008 [-0.0049, +0.0069] | 8.067 | -1.03% | -0.0104 [-0.0137, -0.0073] | 5.808 | -6.10% |
| dpo-strict | 6.883 | +0.05% | +0.0005 [-0.0052, +0.0066] | 8.064 | -1.07% | -0.0107 [-0.0140, -0.0077] | 5.805 | -6.14% |
| grpo | 6.927 | +0.69% | +0.0068 [+0.0019, +0.0127] | 8.086 | -0.80% | -0.0080 [-0.0113, -0.0049] | 5.843 | -5.53% |
| grpo-seed1 | 6.928 | +0.70% | +0.0070 [+0.0020, +0.0129] | 8.088 | -0.77% | -0.0077 [-0.0111, -0.0046] | 5.843 | -5.52% |
| sft-from-base | 7.008 | +1.86% | +0.0184 [+0.0166, +0.0211] | 8.234 | +1.01% | +0.0100 [+0.0091, +0.0110] | 6.311 | +2.05% |
| sft-from-base-seed1 | 7.010 | +1.89% | +0.0187 [+0.0169, +0.0214] | 8.251 | +1.22% | +0.0121 [+0.0110, +0.0133] | 6.316 | +2.13% |
| sft-from-cpt | 6.872 | -0.11% | -0.0011 [-0.0071, +0.0051] | 8.061 | -1.11% | -0.0111 [-0.0144, -0.0081] | 5.800 | -6.23% |
| sft-from-cpt-seed1 | 6.867 | -0.20% | -0.0020 [-0.0078, +0.0044] | 8.066 | -1.04% | -0.0105 [-0.0138, -0.0074] | 5.789 | -6.39% |

## Change vs base-8b-hf, next to the noise

| metric | base-8b-hf | cpt-8b | cpt-8b-seed1 | cpt-8b-replay10 | cpt-8b-full | noise |
|---|---|---|---|---|---|---|
| domain val perplexity | 6.88 | -2.33% | -2.35% | -2.33% | -2.15% | 0.02% |
| 2026-report perplexity | 6.21 |  | -0.51% | -0.43% |  |  |
| general val perplexity | 8.15 | +0.40% | +0.17% | -2.24% | +1.19% | 0.23% |
| train slice perplexity | 6.18 | -8.28% | -8.05% | -8.71% | -22.15% | 0.23% |
| MMLU | 0.767 | -0.4 | -0.2 | -0.1 | -0.5 | 0.3 |
| GSM8K | 0.793 | -0.8 | -0.7 | -0.2 | -3.0 | 1.1 |
| HellaSwag | 0.801 | +0.1 | -0.1 | -0.0 | -0.2 | 0.4 |
| closed-book gold-answer log-prob (nats) | -6.77 | +0.59 | +0.58 | +0.51 |  | 0.08 |
|   of it, the answer tokens | -6.32 | +0.56 | +0.55 | +0.50 |  | 0.07 |
|   of it, the end token | -0.45 | +0.03 | +0.03 | +0.01 |  | 0.01 |
| closed-book qa_strict | 0.121 | +2.5 | +0.9 | +0.9 |  | 1.8 |
| closed-book qa_acc (lenient) | 0.121 | +2.5 | +0.9 | +0.9 |  | 1.8 |
| grounded_acc (with passages) | 0.843 | -0.9 | -4.6 | -6.5 | +4.6 | 3.7 |
| vocab_recall | 0.705 | +0.5 | -0.5 | -0.5 | +3.8 | 3.1 |
| halluc_rate | 0.895 | +1.3 | +3.9 | +3.9 | +2.6 | 3.5 |

lm-eval rows run 5-shot without a BOS token (the frozen flags send none, found in Stage 6): the changes stand, absolute values aren't comparable with published scores. Perplexity in %, the gold-answer log-probability in nats per answer, the rest in points. noise = max(the seed gap cpt-8b vs cpt-8b-seed1, the metric's standard error: for base-8b-hf, or for the log-probability the paired per-item difference): a change smaller than it is not a result. QA rows are on the 322-item domain_qa (eval v3; Stage 2 was first read on v2's 325, results/table_v2.md), so cpt-8b-full, whose weights were deleted, has none.

# Stage 3: SFT

## Training runs

| run | start | steps | tokens trained | tokens/s | wall (h) | GPU-h | $ | peak GB | final train loss | val_loss epoch 1 / 2 | closed-book 1 / 2 | definition 1 / 2 | B4 picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft-from-cpt | cpt-8b-replay10 | 154 | 2.70M | 1,790 | 0.50 | 0.50 | 1.96 | 55 | 0.427 | 0.5592 / 0.5771 | 1.1655 / 1.3040 | 1.9832 / 2.0341 | epoch 1 |
| sft-from-base | base-8b-hf | 154 | 2.70M | 1,834 | 0.48 | 0.48 | 1.89 | 55 | 0.435 | 0.5592 / 0.5790 | 1.2134 / 1.3765 | 2.0178 / 2.0723 | epoch 1 |
| sft-from-cpt-seed1 | cpt-8b-replay10 | 154 | 2.70M | 1,737 | 0.54 | 0.54 | 2.12 | 55 | 0.362 | 0.5586 / 0.5804 | 1.1696 / 1.3126 | 1.8940 / 2.1049 | epoch 1 |
| sft-from-base-seed1 | base-8b-hf | 154 | 2.70M | 2,022 | 0.45 | 0.45 | 1.79 | 55 | 0.364 | 0.5580 / 0.5782 | 1.2347 / 1.3453 | 1.9116 / 2.0432 | epoch 1 |

B4 (pre-registered, amended before training): epoch 2 unless the closed-book or the definition sft_val loss (token mean) rose from epoch 1 to epoch 2. The overall val_loss is 83% replay tokens, so it is shown, not used. $ at 3.95 per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).

## Results next to the noise

| metric | instruct-8b | base-8b-hf | sft-from-base | sft-from-base-seed1 | cpt-8b-replay10 | sft-from-cpt | sft-from-cpt-seed1 | noise (Stage 3) | noise (Stage 2) |
|---|---|---|---|---|---|---|---|---|---|
| unseen gold-answer log-prob (nats) | -8.581 | -6.822 | -6.780 | -6.741 | -6.157 | -6.426 | -6.168 | 0.258 | 0.033 |
|   of it, the answer tokens | -7.709 | -6.356 | -6.197 | -6.199 | -5.697 | -5.895 | -5.684 | 0.210 | 0.033 |
|   of it, the end token | -0.872 | -0.466 | -0.583 | -0.542 | -0.460 | -0.531 | -0.484 | 0.048 | 0.005 |
| seen gold-answer log-prob (nats) | -8.008 | -6.722 | -5.409 | -5.249 | -6.356 | -5.120 | -4.873 | 0.247 | 0.032 |
|   of it, the answer tokens | -7.313 | -6.278 | -4.973 | -4.869 | -5.931 | -4.697 | -4.524 | 0.173 | 0.032 |
|   of it, the end token | -0.695 | -0.444 | -0.436 | -0.380 | -0.425 | -0.423 | -0.349 | 0.074 | 0.005 |
| qa_strict unseen | 0.077 | 0.116 | 0.084 | 0.077 | 0.110 | 0.116 | 0.110 | 2.6 | 2.7 |
| qa_unseen (lenient) | 0.084 | 0.116 | 0.110 | 0.097 | 0.110 | 0.136 | 0.129 | 2.7 | 2.7 |
| qa_strict seen | 0.114 | 0.126 | 0.239 | 0.234 | 0.150 | 0.287 | 0.258 | 3.5 | 2.8 |
| qa_seen (lenient) | 0.114 | 0.126 | 0.245 | 0.234 | 0.150 | 0.281 | 0.270 | 3.5 | 2.8 |
| qa_strict identifiers | 0.031 | 0.078 | 0.109 | 0.141 | 0.125 | 0.203 | 0.234 | 5.0 | 4.1 |
| qa_ident (lenient) | 0.031 | 0.078 | 0.125 | 0.141 | 0.125 | 0.203 | 0.234 | 5.0 | 4.1 |
| grounded_acc | 0.898 | 0.843 | 0.926 | 0.898 | 0.778 | 0.907 | 0.935 | 2.8 | 3.7 |
| cite_supported | 0.787 | 0.083 | 0.870 | 0.880 | 0.037 | 0.861 | 0.880 | 3.3 | 3.7 |
| halluc_rate (lower is better) | 0.013 | 0.895 | 0.066 | 0.013 | 0.934 | 0.053 | 0.013 | 3.9 | 3.3 |
| false_abstain (lower is better) | 0.074 | 0.009 | 0.000 | 0.009 | 0.000 | 0.000 | 0.009 | 0.9 | 0.0 |
| vocab_seen | 0.802 | 0.713 | 0.911 | 0.911 | 0.713 | 0.891 | 0.901 | 3.1 | 4.6 |
| vocab_unseen | 0.771 | 0.697 | 0.761 | 0.752 | 0.688 | 0.780 | 0.817 | 4.0 | 4.3 |
| MMLU | 0.761 | 0.767 | 0.767 | 0.768 | 0.766 | 0.766 | 0.770 | 0.4 | 0.3 |
| GSM8K | 0.855 | 0.793 | 0.792 | 0.790 | 0.791 | 0.814 | 0.792 | 2.2 | 1.1 |
| HellaSwag | 0.801 | 0.801 | 0.795 | 0.799 | 0.800 | 0.794 | 0.799 | 0.6 | 0.4 |

Values as fractions (gold_lp in nats per answer); noise in points (gold_lp in nats). Noise = max(the seed gap, the SE): Stage 3 for sft-from-cpt vs sft-from-cpt-seed1, Stage 2 for cpt-8b vs cpt-8b-seed1. The SE is binomial on the metric's items, lm-eval's stderr, or for gold_lp the paired per-item SE of the twins on that half. Starts: sft-from-cpt from cpt-8b-replay10, sft-from-base from base-8b-hf; instruct-8b is the bar.

## B7's first line: what CPT bought, measured after SFT

- **unseen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.464 nats per answer [95% CI over items +0.276, +0.660; 155 items, 68% up]; noise 0.130 (run-variance SD 0.130 from seed gaps 0.258 (CPT arm) and 0.039 (base arm), paired SE 0.099; the difference is 3.6 run SD, indicative only: each arm's SD rests on one seed pair (1 df). Single-run pairs +0.355, +0.315, +0.612, +0.573; every CPT-arm run above every base-arm run, an ordering with exact one-sided permutation probability 1 in 6): beyond the noise: CPT bought something that survives SFT. Of it, the answer tokens +0.409 [+0.227, +0.602] and the end token +0.055 [+0.021, +0.099]. The item CI conditions on these training runs; run variance enters only through the noise.
- **seen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.332 nats per answer [95% CI over items +0.191, +0.482; 167 items, 66% up]; noise 0.147 (run-variance SD 0.147 from seed gaps 0.247 (CPT arm) and 0.160 (base arm), paired SE 0.074; the difference is 2.3 run SD, indicative only: each arm's SD rests on one seed pair (1 df). Single-run pairs +0.289, +0.129, +0.536, +0.376; every CPT-arm run above every base-arm run, an ordering with exact one-sided permutation probability 1 in 6): beyond the noise: CPT bought something that survives SFT. Of it, the answer tokens +0.311 [+0.173, +0.456] and the end token +0.022 [+0.003, +0.039]. The item CI conditions on these training runs; run variance enters only through the noise.

## Checks

No-op control (an untrained adapter, merged, against its start):

| start | tensors | differ | max abs diff | passed |
|---|---|---|---|---|
| base-8b-hf | 531 | 0 | 0.0 | yes |
| cpt-8b-replay10 | 531 | 0 | 0.0 | yes |

Merge gate (B5, amended): over all 11,351 sft_val completion positions against an fp32 reference, the argmax flips the merge adds over the unmerged bf16 model's own, and its mean |delta log-prob| relative to the unmerged model's (max 1.5); val loss within 0.5%. Merged-vs-unmerged agreement and the 3-probe mean merge-error ratio are reported, not gated; sha256 of the merged checkpoint's file list:

| run | flips added | \|dlp\| ratio | val loss diff | merged vs unmerged top-1 | probe error ratio | checkpoint sha256 | passed |
|---|---|---|---|---|---|---|---|
| sft-from-cpt | 5 of 11 | 1.005 | 0.03% | 99.60% | 0.0516 | `db8ddabfc9a8` | yes |
| sft-from-base | 1 of 11 | 0.965 | 0.02% | 99.67% | 0.046 | `7c6458a78404` | yes |
| sft-from-cpt-seed1 | -8 of 11 | 0.978 | 0.02% | 99.54% | 0.0427 | `5f805f85ed59` | yes |
| sft-from-base-seed1 | 11 of 11 | 0.984 | 0.02% | 99.53% | 0.0466 | `272cbcb28872` | yes |

Diversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over output tokens) and </s> on sampled answers (the eos job: 4 Stage 4 pool prompts per format x 4 at T 0.8; 20 prompts while the pool held replay, 16 after, 2026-10-08):

| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | distinct-4 domain | stopped (T 0.7) | stopped (eos job) |
|---|---|---|---|---|---|---|---|
| instruct-8b | 0.8104 | 9.7998 | 343.2 | 0.8099 | 0.817 | 0.82 |  |
| sft-from-cpt | 0.7724 | 9.0197 | 163.4 | 0.7567 | 0.8646 | 0.98 | 98.8% of 80 |
| sft-from-base | 0.79 | 9.3114 | 185.3 | 0.7765 | 0.8847 | 0.95 | 98.8% of 80 |
| sft-from-cpt-seed1 | 0.7799 | 9.1733 | 196.4 | 0.7719 | 0.837 | 0.97 | 98.8% of 80 |
| sft-from-base-seed1 | 0.8026 | 9.3672 | 190.8 | 0.7955 | 0.8533 | 1.0 | 98.8% of 80 |

# Stage 4: DPO

## Training runs

| run | start | pairs | steps | tokens/s | wall (h) | GPU-h | $ | peak GB | final train loss | dpo_val loss | dpo_val reward accuracy | dpo_val margin | rule picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dpo | sft-from-cpt | 484 | 31 |  | 0.04 | 0.04 | 0.18 | 29 | 0.654 | 0.665 | 0.682 | 0.062 | step 31 (final): 0.6654 vs 0.6701 at 20 |
| dpo-seed1 | sft-from-cpt | 484 | 31 | 1,867 | 0.12 | 0.12 | 0.47 | 37 | 0.658 | 0.668 | 0.682 | 0.055 | step 31 (final): 0.6684 vs 0.6738 at 20 |
| dpo-2ep | sft-from-cpt | 484 | 62 | 2,297 | 0.13 | 0.13 | 0.52 | 37 | 0.382 | 0.571 | 0.682 | 0.378 | step 62 (final): 0.5711 vs 0.6471 at 30 |
| dpo-strict | sft-from-cpt | 445 | 28 | 2,589 | 0.08 | 0.08 | 0.32 | 37 | 0.662 | 0.670 | 0.556 | 0.051 | step 28 (final): 0.6696 vs 0.6839 at 10 |

The checkpoint rule (pre-registered): the final step unless the dpo_val loss at the end is above its value at step 50 (runs under 100 steps: the save nearest the midpoint). dpo_val values at the last evaluation. $ at 3.95 per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).

## The read: change against sft-from-cpt, next to the noise

| metric | read | sft-from-cpt | dpo | dpo-seed1 | change (mean of 2) | noise as written | beyond (as written) | floor with the start's seed gap | beyond (Stage 3 way) |
|---|---|---|---|---|---|---|---|---|---|
| halluc_rate (lower is better) | primary | 0.053 | 0.013 | 0.026 | -3.3 pt | 2.6 pt | yes | 3.9 pt | no |
| false_abstain (lower is better) | primary | 0.000 | 0.000 | 0.000 | +0.0 pt | 0.0 pt | no | 0.9 pt | no |
| cite_valid | primary | 1.000 | 1.000 | 1.000 | +0.0 pt | 0.0 pt | no | 1.8 pt | no |
| qa_seen (lenient, as registered) | primary | 0.281 | 0.311 | 0.287 | +1.8 pt | 3.5 pt | no | 3.5 pt | no |
| qa_unseen (lenient, as registered) | primary | 0.136 | 0.136 | 0.129 | -0.3 pt | 2.7 pt | no | 2.7 pt | no |
| qa_strict seen | reported | 0.287 | 0.305 | 0.287 | +0.9 pt | 3.5 pt | no | 3.5 pt | no |
| qa_strict unseen | reported | 0.116 | 0.116 | 0.110 | -0.3 pt | 2.6 pt | no | 2.6 pt | no |
| seen gold-answer log-prob (nats) | primary | -5.120 | -5.003 | -5.196 | +0.021 [-0.072, +0.113] | 0.193 | no | 0.247 | no |
|   of it, the answer tokens | reported | -4.697 | -4.705 | -4.848 | -0.079 [-0.168, +0.008] | 0.143 | no | 0.173 | no |
|   of it, the end token | reported | -0.423 | -0.298 | -0.348 | +0.100 [+0.079, +0.123] | 0.050 | yes | 0.074 | yes |
| unseen gold-answer log-prob (nats) | primary | -6.426 | -6.544 | -6.731 | -0.212 [-0.335, -0.097] | 0.187 | yes | 0.258 | no |
|   of it, the answer tokens | reported | -5.895 | -6.161 | -6.282 | -0.327 [-0.445, -0.218] | 0.121 | yes | 0.210 | yes |
|   of it, the end token | reported | -0.531 | -0.383 | -0.449 | +0.115 [+0.090, +0.143] | 0.066 | yes | 0.066 | yes |
| MMLU | guard | 0.766 | 0.767 | 0.765 | -0.0 pt | 0.3 pt | no | 0.4 pt | no |
| GSM8K | guard | 0.814 | 0.810 | 0.807 | -0.6 pt | 1.1 pt | no | 2.2 pt | no |
| grounded_acc (judge) | reported | 0.907 | 0.917 | 0.907 | +0.5 pt | 2.8 pt | no | 2.8 pt | no |
| cite_supported (judge) | reported | 0.861 | 0.870 | 0.880 | +1.4 pt | 3.3 pt | no | 3.3 pt | no |
| vocab_recall (judge) | reported | 0.833 | 0.838 | 0.848 | +1.0 pt | 2.6 pt | no | 2.6 pt | no |

Rows marked primary are the amended read (2026-10-08, fixed before launch); guards must stay within the noise; judge-scored rows are reported, not read. As written: max(the DPO seed gap, the start's SE). The Stage 3 way also takes the start's own seed gap (sft-from-cpt vs sft-from-cpt-seed1): the start is one run. One seed pair each (1 df). The verdict uses the Stage 3 way.

## More training: dpo-2ep against dpo (ablation, not a candidate)

| metric | read | dpo (1 epoch) | dpo-2ep (2 epochs) | change | floor | beyond floor |
|---|---|---|---|---|---|---|
| halluc_rate (lower is better) | primary | 0.013 | 0.013 | +0.0 pt | 1.3 pt | no |
| false_abstain (lower is better) | primary | 0.000 | 0.000 | +0.0 pt | 0.0 pt | no |
| cite_valid | primary | 1.000 | 1.000 | +0.0 pt | 0.0 pt | no |
| qa_seen (lenient, as registered) | primary | 0.311 | 0.264 | -4.8 pt | 3.6 pt | yes |
| qa_unseen (lenient, as registered) | primary | 0.136 | 0.129 | -0.7 pt | 2.7 pt | no |
| qa_strict seen | reported | 0.305 | 0.258 | -4.8 pt | 3.6 pt | yes |
| qa_strict unseen | reported | 0.116 | 0.129 | +1.3 pt | 2.6 pt | no |
| seen gold-answer log-prob (nats) | primary | -5.003 | -8.012 | -3.010 [-3.522, -2.514] | 0.193 | yes |
|   of it, the answer tokens | reported | -4.705 | -7.816 | -3.112 [-3.618, -2.622] | 0.143 | yes |
|   of it, the end token | reported | -0.298 | -0.196 | +0.102 [+0.067, +0.140] | 0.050 | yes |
| unseen gold-answer log-prob (nats) | primary | -6.544 | -10.845 | -4.301 [-4.999, -3.653] | 0.187 | yes |
|   of it, the answer tokens | reported | -6.161 | -10.570 | -4.408 [-5.101, -3.759] | 0.121 | yes |
|   of it, the end token | reported | -0.383 | -0.276 | +0.107 [+0.062, +0.154] | 0.066 | yes |
| MMLU | guard | 0.767 | 0.767 | +0.0 pt | 0.3 pt | no |
| GSM8K | guard | 0.810 | 0.806 | -0.4 pt | 1.1 pt | no |
| grounded_acc (judge) | reported | 0.917 | 0.907 | -0.9 pt | 2.7 pt | no |
| cite_supported (judge) | reported | 0.870 | 0.852 | -1.8 pt | 3.2 pt | no |
| vocab_recall (judge) | reported | 0.838 | 0.852 | +1.4 pt | 2.5 pt | no |

Same data, seed and recipe; two epochs, epoch 2 evaluated. Floor: max(the dpo / dpo-seed1 seed gap, dpo's SE), one seed pair.

## Win rate against sft-from-cpt (reported, not read)

| run | win rate | SE | ties | identical greedy answers | n | position consistency |
|---|---|---|---|---|---|---|
| dpo | 0.520 | 0.050 | 82 | 58 | 100 | 0.43 |
| dpo-seed1 | 0.515 | 0.050 | 79 | 60 | 100 | 0.53 |
| dpo-2ep | 0.520 | 0.050 | 64 | 33 | 100 | 0.54 |
| dpo-strict | 0.540 | 0.050 | 84 | 60 | 100 | 0.40 |

## Checks

No-op control (an untrained adapter, merged, against its start):

| start | tensors | differ | max abs diff | passed |
|---|---|---|---|---|
| sft-from-cpt | 531 | 0 | 0.0 | yes |

Merge gate (B5, amended): over every sft_val completion position (11,351; dpo_val's 665 are too few for the 0.1% line, 2026-10-08 freeze) against an fp32 reference, the argmax flips the merge adds over the unmerged bf16 model's own (at most 0.1% of positions), and its mean |delta log-prob| relative to the unmerged model's (max 1.5); val loss within 0.5%. Merged-vs-unmerged agreement and the 3-probe mean merge-error ratio are reported, not gated; sha256 of the merged checkpoint's file list:

| run | flips added | \|dlp\| ratio | val loss diff | merged vs unmerged top-1 | probe error ratio | checkpoint sha256 | passed |
|---|---|---|---|---|---|---|---|
| dpo | 4 of 11 | 1.084 | 0.06% | 99.44% | 0.3635 | `bf9a01c704d3` | yes |
| dpo-seed1 | 1 of 11 | 1.089 | 0.04% | 99.56% | 0.3944 | `24405b930436` | yes |
| dpo-2ep | 5 of 11 | 1.427 | 0.58% | 99.46% | 0.2455 | `b7606d2350fd` | NO |
| dpo-strict | -5 of 11 | 1.083 | 0.08% | 99.50% | 0.418 | `e686ca7bf11e` | yes |

Diversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over output tokens) and </s> on sampled answers (the eos job: 4 Stage 4 pool prompts per format x 4 at T 0.8; 20 prompts while the pool held replay, 16 after, 2026-10-08):

| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | distinct-4 domain | stopped (T 0.7) | stopped (eos job) |
|---|---|---|---|---|---|---|---|
| sft-from-cpt | 0.7724 | 9.0197 | 163.4 | 0.7567 | 0.8646 | 0.98 | 98.8% of 80 |
| dpo | 0.7963 | 9.2183 | 167.6 | 0.7846 | 0.8729 | 1.0 | 100.0% of 64 |
| dpo-seed1 | 0.7797 | 9.1358 | 171.5 | 0.7643 | 0.8791 | 0.97 | 100.0% of 64 |
| dpo-2ep | 0.7992 | 9.2158 | 164.8 | 0.8022 | 0.7855 | 0.99 | 100.0% of 64 |
| dpo-strict | 0.7575 | 9.0924 | 172.4 | 0.7394 | 0.8671 | 0.96 | 100.0% of 64 |

- **dpo-2ep: merge gate flagged.** Val-loss criterion 0.58% against 0.5% (merged 0.5663, unmerged 0.5696 nats on sft_val). The merge-isolating criteria pass in `merge_diagnose`: 5 added flips of 11, |dlp| ratio 1.43 against 1.5. Evaluated under the rule fixed before the diagnosis (notes/decisions.md, 2026-10-09).

# Stage 5: GRPO

## Training runs

| run | start | tasks | steps | stop | rule picks | grpo_val pass@1 / pass@8 at the pick | final train reward | final entropy | mean length | base drift | wall (h) | GPU-h | $ | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grpo | dpo-strict | 622 | 52 | step 52: entropy's 10-step mean under a third of its steps 1-10 mean | step 25 (best 25) | 0.550 / 0.820 | 0.8179 | 0.158 | 19.857 | 2.4e-04 | 0.50 | 0.50 | 1.98 | 77 |
| grpo-seed1 | dpo-strict | 622 | 65 | step 65: entropy's 10-step mean under a third of its steps 1-10 mean | step 25 (best 25) | 0.562 / 0.800 | 0.8162 | 0.1499 | 18.2883 | 1.2e-04 | 0.58 | 0.58 | 2.30 | 76 |

The checkpoint rule (pre-registered): the best grpo_val pass@1 among the saves at or before any stop, ties within one SE to the earliest. grpo_val: 50 held-out tasks x 8 samples at T 1.0. $ at 3.95 per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).

## The read: change against dpo-strict, next to the noise

| metric | read | dpo-strict | grpo | grpo-seed1 | change (mean of 2) | floor | beyond |
|---|---|---|---|---|---|---|---|
| qa_strict seen (the reward's rule) | primary | 0.305 | 0.299 | 0.287 | -1.2 pt | 3.6 pt | no |
| seen gold-answer log-prob (nats) | primary | -5.130 | -5.784 | -5.819 | -0.671 [-0.858, -0.492] | 0.193 | yes |
|   of it, the answer tokens | reported | -4.811 | -5.517 | -5.519 | -0.707 [-0.888, -0.531] | 0.143 | yes |
|   of it, the end token | reported | -0.319 | -0.267 | -0.299 | +0.036 [+0.012, +0.059] | 0.050 | no |
| qa_acc seen (lenient) | reported | 0.299 | 0.293 | 0.281 | -1.2 pt | 3.6 pt | no |
| qa_strict unseen | reported | 0.110 | 0.097 | 0.103 | -1.0 pt | 2.6 pt | no |
| qa_acc unseen (lenient) | reported | 0.129 | 0.110 | 0.110 | -1.9 pt | 2.7 pt | no |
| unseen gold-answer log-prob (nats) | reported | -6.695 | -7.956 | -8.101 | -1.333 [-1.608, -1.076] | 0.187 | yes |
|   of it, the answer tokens | reported | -6.274 | -7.541 | -7.647 | -1.319 [-1.583, -1.078] | 0.121 | yes |
|   of it, the end token | reported | -0.421 | -0.415 | -0.454 | -0.013 [-0.068, +0.031] | 0.066 | no |
| halluc_rate (lower is better) | guard | 0.040 | 0.013 | 0.013 | -2.6 pt | 2.2 pt | yes |
| false_abstain (lower is better) | guard | 0.000 | 0.000 | 0.018 | +0.9 pt | 1.8 pt | no |
| cite_valid | guard | 1.000 | 1.000 | 0.982 | -0.9 pt | 1.8 pt | no |
| MMLU | guard | 0.767 | 0.767 | 0.767 | -0.0 pt | 0.3 pt | no |
| GSM8K | guard | 0.809 | 0.820 | 0.802 | +0.2 pt | 1.7 pt | no |
| grounded_acc (judge) | reported | 0.926 | 0.935 | 0.917 | +0.0 pt | 2.7 pt | no |
| cite_supported (judge) | reported | 0.861 | 0.880 | 0.889 | +2.3 pt | 3.3 pt | no |
| pass@1 seen (sampled) | reported | 0.249 | 0.278 | 0.281 | +3.1 pt [+1.3, +5.0] | 0.4 pt | yes |
| pass@8 seen | reported | 0.449 | 0.413 | 0.413 | -3.6 pt [-8.1, +0.9] | 2.4 pt | no |
| pass@1 unseen (sampled) | reported | 0.105 | 0.119 | 0.112 | +1.0 pt [-0.1, +2.2] | 0.6 pt | no |
| pass@8 unseen | reported | 0.297 | 0.258 | 0.271 | -3.2 pt [-7.7, +1.0] | 2.6 pt | no |

Rows marked primary are the pre-registered read (2026-10-09); guards must stay within the noise and their absolute lines; reported rows are not argued. Floor: max(the GRPO seed gap, the start's SE, the start's own seed gap: dpo-strict has no twin, so the lenient dpo / dpo-seed1 gap). One seed pair each (1 df).

## Same verifier, two algorithms: each from its own start, and the chain's cumulative change from sft-from-cpt

| metric | sft-from-cpt | DPO from its own start (sft-from-cpt) | beyond (floor) | GRPO from its own start (dpo-strict) | beyond (floor) | cumulative from sft-from-cpt (grpo) | beyond (floor) |
|---|---|---|---|---|---|---|---|
| qa_strict seen (the reward's rule) | 0.287 | +1.8 pt | no (3.6 pt) | -1.2 pt | no (3.6 pt) | +0.6 pt | no (3.5 pt) |
| seen gold-answer log-prob (nats) | -5.120 | -0.010 [-0.102, +0.080] | no (0.247) | -0.671 [-0.858, -0.492] | yes (0.193) | -0.681 [-0.938, -0.436] | yes (0.247) |
|   of it, the answer tokens | -4.697 | -0.114 [-0.203, -0.028] | no (0.173) | -0.707 [-0.888, -0.531] | yes (0.143) | -0.821 [-1.072, -0.579] | yes (0.173) |
|   of it, the end token | -0.423 | +0.104 [+0.084, +0.127] | yes (0.074) | +0.036 [+0.012, +0.059] | no (0.050) | +0.140 [+0.110, +0.171] | yes (0.074) |
| qa_strict unseen | 0.116 | -0.6 pt | no (2.6 pt) | -1.0 pt | no (2.6 pt) | -1.6 pt | no (2.6 pt) |
| unseen gold-answer log-prob (nats) | -6.426 | -0.270 [-0.396, -0.157] | yes (0.258) | -1.333 [-1.608, -1.076] | yes (0.187) | -1.603 [-1.975, -1.256] | yes (0.258) |
|   of it, the answer tokens | -5.895 | -0.380 [-0.504, -0.267] | yes (0.210) | -1.319 [-1.583, -1.078] | yes (0.121) | -1.699 [-2.063, -1.363] | yes (0.210) |
|   of it, the end token | -0.531 | +0.110 [+0.084, +0.138] | yes (0.066) | -0.013 [-0.068, +0.031] | no (0.066) | +0.097 [+0.052, +0.136] | yes (0.048) |
| halluc_rate (lower is better) | 0.053 | -1.3 pt | no (3.9 pt) | -2.6 pt | yes (2.2 pt) | -3.9 pt | no (3.9 pt) |
| MMLU | 0.766 | +0.1 pt | no (0.4 pt) | -0.0 pt | no (0.3 pt) | +0.1 pt | no (0.4 pt) |
| GSM8K | 0.814 | -0.5 pt | no (2.2 pt) | +0.2 pt | no (1.7 pt) | -0.3 pt | no (2.2 pt) |
| pass@1 seen (sampled) | 0.213 | +3.5 pt [+1.2, +6.0] | yes (0.4 pt) | +3.1 pt [+1.3, +5.0] | yes (0.4 pt) | +6.6 pt [+3.6, +9.8] | yes (0.2 pt) |
| pass@8 seen | 0.479 | -3.0 pt [-7.8, +1.2] | no (2.4 pt) | -3.6 pt [-8.1, +0.9] | no (2.4 pt) | -6.6 pt [-12.3, -0.9] | yes (0.0 pt) |
| pass@1 unseen (sampled) | 0.094 | +1.1 pt [+0.0, +2.4] | no (0.6 pt) | +1.0 pt [-0.1, +2.2] | no (0.6 pt) | +2.2 pt [+0.4, +4.0] | yes (0.6 pt) |
| pass@8 unseen | 0.310 | -1.3 pt [-5.2, +2.6] | no (2.6 pt) | -3.2 pt [-7.7, +1.0] | no (2.6 pt) | -4.5 pt [-9.7, +0.3] | no (1.3 pt) |

dpo-strict: 445 offline pairs from the SFT model's samples, labelled by the strict checker, one run. grpo: on-policy groups scored by the same checker, two seeds. Each floor also takes sft-from-cpt's own seed gap. pass@k lines: paired per item (8 samples at T 0.7, the strict scorer), floor = the seed gaps sampled (grpo's pair; the lenient dpo pair for dpo-strict); sft-from-cpt's twin was not sampled. Win rate: not run for these rows: the judge failed its benchmark and its two orders agreed at coin-flip.

## pass@k on the closed-book eval (strict scorer)

| run | seen pass@1 | seen maj@8 | seen pass@8 | unseen pass@1 | unseen maj@8 | unseen pass@8 |
|---|---|---|---|---|---|---|
| sft-from-cpt | 0.213 ± 0.024 | 0.264 ± 0.034 | 0.479 ± 0.039 | 0.093 ± 0.016 | 0.110 ± 0.025 | 0.310 ± 0.037 |
| dpo | 0.247 ± 0.027 | 0.270 ± 0.034 | 0.461 ± 0.039 | 0.104 ± 0.019 | 0.103 ± 0.025 | 0.284 ± 0.036 |
| dpo-seed1 | 0.242 ± 0.027 | 0.258 ± 0.034 | 0.437 ± 0.038 | 0.111 ± 0.019 | 0.103 ± 0.025 | 0.310 ± 0.037 |
| dpo-strict | 0.248 ± 0.028 | 0.264 ± 0.034 | 0.449 ± 0.039 | 0.105 ± 0.018 | 0.116 ± 0.026 | 0.297 ± 0.037 |
| grpo | 0.278 ± 0.031 | 0.275 ± 0.035 | 0.413 ± 0.038 | 0.118 ± 0.021 | 0.136 ± 0.028 | 0.258 ± 0.035 |
| grpo-seed1 | 0.281 ± 0.031 | 0.264 ± 0.034 | 0.413 ± 0.038 | 0.112 ± 0.020 | 0.103 ± 0.025 | 0.271 ± 0.036 |
| instruct-8b | 0.079 ± 0.013 | 0.096 ± 0.023 | 0.264 ± 0.034 | 0.066 ± 0.011 | 0.077 ± 0.021 | 0.265 ± 0.035 |

8 samples per item at T 0.7 (eval/passk.py); ± is the SE over items.

## Checks


Merge gate (B5, amended): over every sft_val completion position (11,351) against an fp32 reference, the argmax flips the merge adds over the unmerged bf16 model's own (at most 0.1% of positions), and its mean |delta log-prob| relative to the unmerged model's (max 1.5); val loss within 0.5%. Merged-vs-unmerged agreement and the 3-probe mean merge-error ratio are reported, not gated; sha256 of the merged checkpoint's file list:

| run | flips added | \|dlp\| ratio | val loss diff | merged vs unmerged top-1 | probe error ratio | checkpoint sha256 | passed |
|---|---|---|---|---|---|---|---|
| grpo | 3 of 11 | 1.079 | 0.08% | 99.53% | 0.2662 | `9f09e4aa569d` | yes |
| grpo-seed1 | -1 of 11 | 1.185 | 0.15% | 99.52% | 0.4027 | `66c2092e0783` | yes |

Diversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over output tokens) and </s> on sampled answers (the eos job: 4 Stage 4 pool prompts per format x 4 at T 0.8; 20 prompts while the pool held replay, 16 after, 2026-10-08):

| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | distinct-4 domain | stopped (T 0.7) | stopped (eos job) |
|---|---|---|---|---|---|---|---|
| dpo-strict | 0.7575 | 9.0924 | 172.4 | 0.7394 | 0.8671 | 0.96 | 100.0% of 64 |
| grpo | 0.7805 | 9.1367 | 167.0 | 0.7654 | 0.8756 | 0.98 | 100.0% of 64 |
| grpo-seed1 | 0.7741 | 9.0967 | 171.9 | 0.7694 | 0.8038 | 0.98 | 100.0% of 64 |

# Stage 6: serving

## Quality gate (pre-registered): change against bf16, next to the Stage 3 floor

| line | bf16 | FP8 − bf16 | FP8 + FP8 KV − bf16 | INT4 W4A16 − bf16 | floor |
|---|---|---|---|---|---|
| qa_strict unseen | 0.110 | -1.3 | +0.6 | -2.6 | 2.6 pt |
| qa_strict seen | 0.305 | -1.2 | **-4.2** | -3.0 | 3.5 pt |
| gold_lp answer tokens, unseen (nats) | -6.274 | +0.000 | -0.055 | **-0.424** | 0.210 |
| gold_lp answer tokens, seen (nats) | -4.811 | -0.007 | -0.056 | **-0.307** | 0.173 |
| gold_lp end token, unseen (nats) | -0.421 | -0.020 | -0.029 | **+0.059** | 0.048 |
| gold_lp end token, seen (nats) | -0.319 | -0.004 | -0.008 | +0.015 | 0.074 |
| grounded_acc | 0.926 | -1.8 | +0.9 | -2.8 | 2.8 pt |
| cite_valid | 1.000 | +0.0 | +0.0 | +0.0 | 1.85 pt |
| halluc_rate | 0.040 | +1.3 | +2.6 | +0.0 | 3.9 pt |
| false_abstain | 0.000 | +0.0 | +0.0 | +0.0 | 0.9 pt |
| GSM8K (all 1,319, add_bos_token) | 0.809 | -0.2 |  | **-6.1** | 2.2 pt |
| qa_strict identifiers (reported) | 0.188 | -1.6 | -1.6 | **-7.8** | 5 pt |
| answers ending on </s> (eos job) | 1.00 | 1.00 | 1.00 | 1.00 | >= 0.95 |
| vLLM val-slice perplexity (reported) | 6.894 | +0.33% | +0.58% | +3.53% |  |
| **verdict** |  | **ships** | **fails** | **fails** |  |

A variant ships if every gated line is within its floor (bold: beyond it) and the eos job ends >= 95% of answers on </s> (notes/decisions.md, 2026-10-09). The floor is the sft-from-cpt seed gap or the SE, whichever is larger: a cost under it is invisible to every other comparison here, which is what ships means, not that it costs nothing. GSM8K is the gate's own run (all 1,319, 5-shot, add_bos_token=True, bf16 rerun under the same flags), not the table's frozen-flag row.

## Serving memory (vLLM's own accounting at start-up)

| variant | GPU | weights (vLLM, GiB) | KV cache (GiB) | KV cache tokens | 8,192-token sequences: vLLM / Step 0 estimate |
|---|---|---|---|---|---|
| bf16 | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 15.94 | 52.75 | 406,688 | 49.6 / 44 |
| FP8 | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 9.01 | 60.61 | 467,296 | 57.0 / 50 |
| FP8 + FP8 KV | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 9.01 | 60.61 | 934,608 | 114.1 / 100 |

## Latency and throughput vs concurrency (`unique`: 360 eval requests, 60 / 30 / 10)

| variant | concurrency | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) | E2EL p50 / p99 (ms) | req/s | output tok/s | goodput share | run pair spread |
|---|---|---|---|---|---|---|---|---|
| bf16 | 1 | 17.9 / 54.0 | 6.90 / 7.80 | 71 / 964 | 4.78 | 130 | 100% | 0% |
| bf16 | 8 | 44.9 / 224.1 | 7.83 / 36.74 | 111 / 1,512 | 23.88 | 655 | 100% | 1% |
| bf16 | 32 | 87.9 / 638.4 | 10.90 / 88.83 | 286 / 3,058 | 40.28 | 1,096 | 70% | 0% |
| bf16 | 64 | 213.4 / 952.5 | 17.54 / 197.57 | 635 / 5,411 | 43.67 | 1,189 | 24% | 0% |
| FP8 | 1 | 35.2 / 59.9 | 4.70 / 5.63 | 71 / 804 | 5.94 | 164 | 100% | 0% |
| FP8 | 8 | 79.8 / 214.6 | 5.30 / 49.55 | 149 / 1,688 | 21.20 | 580 | 99% | 2% |
| FP8 | 32 | 128.3 / 478.6 | 7.66 / 59.36 | 290 / 3,172 | 40.75 | 1,117 | 77% | 4% |
| FP8 | 64 | 222.5 / 717.2 | 10.53 / 127.30 | 575 / 4,434 | 53.08 | 1,461 | 30% | 1% |

Means of two runs per config, prefix cache reset before each; goodput share = requests with TTFT <= 500 ms and TPOT <= 25 ms. ITL at concurrency 1 under the decode floor (weight bytes / 3.35 TB/s: bf16 4.75, FP8 2.5, INT4 1.5 ms) would mean the run wasn't on an H100.

## Latency under load (Poisson arrivals, `unique`)

| variant | offered req/s | achieved req/s | TTFT p50 / p99 (ms) | E2EL p99 (ms) | goodput share |
|---|---|---|---|---|---|
| bf16 | 1 | 1.00 | 21.1 / 57.3 | 1,028 | 100% |
| bf16 | 4 | 4.00 | 25.7 / 67.8 | 1,031 | 100% |
| bf16 | 16 | 15.42 | 40.1 / 116.4 | 1,298 | 99% |
| FP8 | 1 | 1.00 | 39.0 / 92.9 | 857 | 100% |
| FP8 | 4 | 4.00 | 55.7 / 107.9 | 795 | 100% |
| FP8 | 16 | 15.65 | 75.8 / 143.3 | 1,399 | 96% |

## Prefix caching: shared retrieval context (FP8, concurrency 8)

| run | TTFT p50 (ms) | TTFT p99 (ms) | prefix-cache hit rate | E2EL p50 (ms) |
|---|---|---|---|---|
| grounded, passages as evaluated (none shared) | 62.4 | 257.4 | 6% | 633 |
| grounded, 4 questions per shared context (rag) | 44.6 | 256.4 | 75% | 523 |

Same 108 questions in the same order; rag prompts are ~9% longer (gold passages run long), which counts against rag. TTFT is the comparison; output lengths differ.

## Cost

| variant | gate | best concurrency | goodput (req/s) | $ / 1k requests (1 H100) | API $ / 1k requests (Small 4 prices) | break-even sustained req/s |
|---|---|---|---|---|---|---|
| bf16 | reference | 32 | 28.1 | 0.0390 | 0.1266 | 8.7 |
| FP8 | ships | 32 | 31.5 | 0.0348 | 0.1267 | 8.7 |

One H100 at $3.95/h (Modal's H100 SXM5 list price, $0.001097/s, checked 2026-10-09); Small 4's API at $0.15 / $0.6 per million input / output tokens and its self-hosting minimum of 4 H100s ($15.80/h before any request), both from Mistral's Small 4 announcement. Tokens per request are the bench mix's measured means. The break-even is the sustained load above which the GPU is cheaper than the API; below it, an idle GPU costs the same per hour.
