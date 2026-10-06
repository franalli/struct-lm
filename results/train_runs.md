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

$ at 3.95 per GPU-hour (Modal's H100 list price as assumed, not checked against modal.com/pricing); wall time includes tokenising and model load.

## Perplexity vs base-8b

| run | domain val | change | nats [95% CI, docs] | general val | change | nats [95% CI, docs] | train slice | change |
|---|---|---|---|---|---|---|---|---|
| base-8b | 6.880 |  |  | 8.151 |  |  | 6.185 |  |
| cpt-8b | 6.720 | -2.33% | -0.0236 [-0.0302, -0.0180] | 8.184 | +0.40% | +0.0040 [+0.0025, +0.0056] | 5.673 | -8.28% |
| cpt-8b-replay10 | 6.720 | -2.33% | -0.0236 [-0.0303, -0.0180] | 7.969 | -2.24% | -0.0226 [-0.0257, -0.0198] | 5.646 | -8.71% |
| cpt-8b-full | 6.732 | -2.15% | -0.0217 [-0.0357, -0.0104] | 8.248 | +1.19% | +0.0118 [+0.0091, +0.0143] | 4.815 | -22.15% |
| cpt-8b-lr2x | 6.713 | -2.42% | -0.0245 [-0.0335, -0.0172] | 8.220 | +0.85% | +0.0085 [+0.0068, +0.0102] | 5.442 | -12.01% |
| cpt-8b-seed1 | 6.718 | -2.35% | -0.0238 [-0.0307, -0.0180] | 8.165 | +0.17% | +0.0017 [+0.0002, +0.0030] | 5.687 | -8.05% |
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
| closed-book qa_acc | 0.121 | +2.5 | +0.9 | +0.9 |  | 1.8 |
| grounded_acc (with passages) | 0.843 | -0.9 | -4.6 | -6.5 | +4.6 | 3.7 |
| vocab_recall | 0.705 | +0.5 | -0.5 | -0.5 | +3.8 | 3.1 |
| halluc_rate | 0.895 | +1.3 | +3.9 | +3.9 | +2.6 | 3.5 |

Perplexity in %, the gold-answer log-probability in nats per answer, the rest in points. noise = max(the seed gap cpt-8b vs cpt-8b-seed1, the metric's standard error: for base-8b-hf, or for the log-probability the paired per-item difference): a change smaller than it is not a result. QA rows are on the 322-item domain_qa (eval v3; Stage 2 was first read on v2's 325, results/table_v2.md), so cpt-8b-full, whose weights were deleted, has none.

# Stage 3: SFT

## Training runs

| run | start | steps | tokens trained | tokens/s | wall (h) | GPU-h | $ | peak GB | final train loss | val_loss epoch 1 / 2 | closed-book 1 / 2 | definition 1 / 2 | B4 picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft-from-cpt | cpt-8b-replay10 | 154 | 2.70M | 1,790 | 0.50 | 0.50 | 1.96 | 55 | 0.427 | 0.5592 / 0.5771 | 1.1655 / 1.3040 | 1.9832 / 2.0341 | epoch 1 |
| sft-from-base | base-8b-hf | 154 | 2.70M | 1,834 | 0.48 | 0.48 | 1.89 | 55 | 0.435 | 0.5592 / 0.5790 | 1.2134 / 1.3765 | 2.0178 / 2.0723 | epoch 1 |
| sft-from-cpt-seed1 | cpt-8b-replay10 | 154 | 2.70M | 1,737 | 0.54 | 0.54 | 2.12 | 55 | 0.362 | 0.5586 / 0.5804 | 1.1696 / 1.3126 | 1.8940 / 2.1049 | epoch 1 |
| sft-from-base-seed1 | base-8b-hf | 154 | 2.70M | 2,022 | 0.45 | 0.45 | 1.79 | 55 | 0.364 | 0.5580 / 0.5782 | 1.2347 / 1.3453 | 1.9116 / 2.0432 | epoch 1 |

B4 (pre-registered, amended before training): epoch 2 unless the closed-book or the definition sft_val loss (token mean) rose from epoch 1 to epoch 2. The overall val_loss is 83% replay tokens, so it is shown, not used. $ at 3.95 per GPU-hour (assumed).

## Results next to the noise

| metric | instruct-8b | base-8b-hf | sft-from-base | sft-from-base-seed1 | cpt-8b-replay10 | sft-from-cpt | sft-from-cpt-seed1 | noise (Stage 3) | noise (Stage 2) |
|---|---|---|---|---|---|---|---|---|---|
| unseen gold-answer log-prob (nats) | -8.581 | -6.822 | -6.780 | -6.741 | -6.157 | -6.426 | -6.168 | 0.258 | 0.033 |
| seen gold-answer log-prob (nats) | -8.008 | -6.722 | -5.409 | -5.249 | -6.356 | -5.120 | -4.873 | 0.247 | 0.032 |
| qa_unseen | 0.084 | 0.116 | 0.110 | 0.097 | 0.110 | 0.136 | 0.129 | 2.7 | 2.7 |
| qa_seen | 0.114 | 0.126 | 0.245 | 0.234 | 0.150 | 0.281 | 0.270 | 3.5 | 2.8 |
| qa_ident (identifiers) | 0.031 | 0.078 | 0.125 | 0.141 | 0.125 | 0.203 | 0.234 | 5.0 | 4.1 |
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

- **unseen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.464 nats per answer [95% CI over items +0.276, +0.660; 155 items, 68% up]; noise 0.130 (run-variance SD 0.130 from seed gaps 0.258 (CPT arm) and 0.039 (base arm), paired SE 0.099; the difference is 3.6 run SD): beyond the noise: CPT bought something that survives SFT. The item CI conditions on these training runs; run variance enters only through the noise.
- **seen gold_lp, mean of 2 CPT-arm runs - mean of 2 base-arm runs:** +0.332 nats per answer [95% CI over items +0.191, +0.482; 167 items, 66% up]; noise 0.147 (run-variance SD 0.147 from seed gaps 0.247 (CPT arm) and 0.160 (base arm), paired SE 0.074; the difference is 2.3 run SD): beyond the noise: CPT bought something that survives SFT. The item CI conditions on these training runs; run variance enters only through the noise.

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

Diversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over output tokens) and </s> on sampled answers (20 Stage 4 prompts x 4 at T 0.8):

| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | distinct-4 domain | stopped (T 0.7) | stopped (eos job) |
|---|---|---|---|---|---|---|---|
| instruct-8b | 0.8104 | 9.7998 | 343.2 | 0.8099 | 0.817 | 0.82 |  |
| sft-from-cpt | 0.7724 | 9.0197 | 163.4 | 0.7567 | 0.8646 | 0.98 | 98.8% of 80 |
| sft-from-base | 0.79 | 9.3114 | 185.3 | 0.7765 | 0.8847 | 0.95 | 98.8% of 80 |
| sft-from-cpt-seed1 | 0.7799 | 9.1733 | 196.4 | 0.7719 | 0.837 | 0.97 | 98.8% of 80 |
| sft-from-base-seed1 | 0.8026 | 9.3672 | 190.8 | 0.7955 | 0.8533 | 1.0 | 98.8% of 80 |
