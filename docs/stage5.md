# Stage 5: GRPO with verifiable rewards

**Result: GRPO sharpened what the model already answered and added no knowledge.**
- **Both seeds collapsed and stopped early.** They stopped on entropy collapse (steps 52 and 65:
  entropy under a third of its start while the train reward climbed from 0.5 to 0.83), and the
  pre-registered rule kept step 25 of each.
- **Against `dpo-strict`:**
  - greedy closed-book accuracy is unchanged (strict, seen −1.2 points, floor 3.6);
  - sampled pass@1 rose 3.1 points on seen items (paired CI [+1.3, +5.0]) while pass@8 fell 3.6,
    inside the noise. That is DeepSeekMath's sharpening signature;
  - the gold answer's log-probability fell 0.67 nats on seen facts and 1.33 on unseen (−0.71 and
    −1.32 on the answer tokens; the end token barely moved), beyond the
    floor: the cost of that sharpening;
  - hallucination 3 → 1 of 76 is noise: the SFT twins sit at 4 and 1, the DPO twins at 1 and 2.
- **The expected +3 to +8 points on seen accuracy (strict) did not happen,** so `stage5-final` stays
  `dpo-strict`, as the rule requires.
- **Serving is greedy, so GRPO's only gain is one serving doesn't use.** That gain is sampled
  accuracy at T 0.7, and it cost calibration on every fact.
- **The chain's verdict:** CPT and SFT delivered; DPO and GRPO at this scale did not clear the
  floor on the primary lines. GRPO's checkpoint was set aside; DPO's carried forward as the SFT
  model within noise. *(Corrected 2026-10-10: this read "the pre-registered rules rejected
  both"; no rule set DPO's checkpoint aside.)*
  - Both sharpened sampled accuracy by about 3 points. DPO did so at a small calibration cost (0.11
    / 0.38 nats on the answer tokens), GRPO at a large one (0.71 / 1.32).
  - The chain ends on `dpo-strict`, which is the SFT model within noise.

Group Relative Policy Optimization from `dpo-strict`, with a fresh LoRA, on tasks the start
sometimes solves. The policy writes every completion, rules score it, and no judge or reference
model is involved. The plan, its corrections and the read were fixed before training
([`notes/decisions.md`](../notes/decisions.md), 2026-10-09).

## The tasks

- **Candidates** (`data/scripts/grpo_tasks.py`): 1,619 prompts, every closed-book, grounded and
  abstain prompt of the Stage 4 pool outside its win-rate split (1,084 / 359 / 176), each with its
  verifier.
  - No alias lists exist, so closed-book answers are checked against the one gold string with the
    strict checker.
  - Compute tasks (formula + sampled inputs) were not built.
- **Contamination** (`eval/contamination.py --only grpo`):
  - 0 leaked eval chunks, 0 copied eval questions;
  - positive control 40 of 40;
  - the remaining overlaps are the shared document designations Stage 4 listed.
- **The calibration probe:** 8 samples per task from `dpo-strict` at T 1.0 and 256 tokens (the
  rollout's settings), scored by the reward itself.
  - Sample accuracy: closed-book 29.2%, grounded 93.2%, abstain 96.9%.
  - A task enters the window when 1 to 7 of its 8 samples are correct (a group whose samples all
    score the same has zero advantage).

| | closed-book value | identifier | term | grounded | abstain | total |
|---|---|---|---|---|---|---|
| in the window | 346 | 151 | 31 | 121 | 23 | **672** |
| `grpo_val` (held out by fact) | 26 | 11 | 2 | 9 | 2 | **50** |

- **Grounded:** 38 of the 121 grounded tasks are in the window only through the "at most two
  passages" rule. About a third of the grounded signal is therefore "cite at most two", which is
  the teacher's norm (371 of 374).
- **Train:** 622 tasks, dataset hash `4db8f7a6` (`data/grpo/SHA256SUMS`).

## The reward

`train/grpo_rewards.py`, one function per part, so TRL logs each one:

- **Format (0.1, a gate: fail it and correctness is not read):**
  - every kind must end on `</s>`;
  - closed-book: exactly one line holding one candidate. The prompts ask for a bare value, so the
    gate is not an "Answer:" line;
  - grounded: every bracket is one of the four chunk ids;
  - abstain: the sentence, or a grounded-form answer.
- **Correctness (0.9):**
  - closed-book by the strict checker;
  - grounded: cites the gold passage, at most two passages, and doesn't abstain;
  - abstain: exactly the sentence.
- **Length (0 to −0.1):** zero to 192 tokens, then linear to −0.1 at the 256 cap.
- **Tested before any sampling:** 41 reward fixtures, every hack case among them (two answers, an
  answer then a contradiction, a range, a wrong unit, a piece of the gold, citing all four
  passages, a stray bracket).
- **Pinned on the Stage 4 pool:** the reward passes 1,305 of its 4,336 closed-book samples,
  exactly the `dpo-strict` labels.

## The training run

`train/configs/grpo.yaml`, TRL 0.29.1 `GRPOTrainer`:

| Item | Value | Why |
|---|---|---|
| Group | 8 completions per task, 16 tasks (128 completions) per step, used once | on-policy (RLHF Book ch. 6) |
| Loss | `dapo` token-level, advantages scaled by the batch std, clip 0.2 / 0.28, no KL | Magistral, DAPO, Dr. GRPO |
| Sampling | T 1.0, top-p 1.0, 256 tokens; truncated completions out of the loss | Magistral |
| Optimiser | LoRA r 64 / α 128, LR 1e-5 constant after 10 warmup steps, 150 steps | RL runs are not decayed |
| Rollouts | vLLM 0.30 colocated, 35% of the GPU, the repo's engine settings and the run's seed | rule 3 |
| Precision | bf16, fp32 log-probs; truncated importance sampling against vLLM's log-probs | train/inference mismatch (ch. 6) |
| Held out | `grpo_val` pass@1 and pass@8 at step 0 and every 25 steps, 8 samples per task | the checkpoint rule's input |

Three corrections the code forced on the plan:
- **TRL builds the rollout engine without the repo's tokenizer, config and image settings.** It is
  wrapped to add them, along with the run's seed (TRL seeds every run's sampler with 0).
- **No sleep mode.** TRL reloads the weights from disk when it wakes a sleeping engine, which would
  have discarded each step's LoRA sync and sampled from the start every step.
- **The engine takes 35% of the GPU, not 25%.** Its own bf16 weight copy alone is 17.8 GB.

The smoke run checked the sync:
- step 1's importance-sampling ratio was 0.9997;
- after updates at 10× the learning rate, the ratio stayed at 1.00 and the vLLM-vs-policy log-prob
  gap didn't grow.

**Checkpoint rule:** the best `grpo_val` pass@1 among the saves at or before any stop. A save
within one SE of the best counts as a tie, and ties go to the earliest.

**Stop rules:**
- groups with zero reward spread above 80% for 10 steps;
- entropy below a third of its start;
- no new best `grpo_val` in two evaluations while the train reward rises.

![Stage 5 GRPO curves: train reward, grpo_val pass@1 and pass@8, length, entropy, zero-spread groups, and the vLLM-vs-policy log-prob gap](../results/curves/grpo.png)

## The read: sharpening, not knowledge

Pre-registered in `notes/decisions.md` (2026-10-09). The tables are below the figures. The floor is
max(the GRPO seed gap, the start's SE, the start's own seed gap); `dpo-strict` has no twin, so the
start's gap is the lenient `dpo` pair's. One seed pair each (1 df).

1. **Primary:**
   - Seen strict accuracy: 0.305 → 0.299 / 0.287, −1.2 points against a 3.6-point floor. No
     change.
   - Seen gold-answer log-probability: −0.67 nats [−0.86, −0.49]; −0.71 [−0.89, −0.53] of it on the
     answer tokens and +0.04 on the end token, so a calibration cost, not a format change. Beyond the floor in the wrong
     direction.
   - The expected +3 to +8 points did not come.
2. **pass@1 against pass@8** (322 closed-book items, 8 samples at T 0.7, the strict scorer, paired
   per item):
   - Seen: pass@1 +3.1 points [+1.3, +5.0], pass@8 −3.6 [−8.1, +0.9], maj@8 +0.6.
   - Unseen: pass@1 +1.0 [−0.1, +2.2], pass@8 −3.2 [−7.7, +1.3].
   - Sampled accuracy rose while coverage stayed within the noise and leaned down.
   - **The mechanism** (DeepSeekMath §5.2.2: RL sharpens the output distribution rather than
     adding capability):
     - sharpening moves mass onto the answer the model already ranks first;
     - greedy accuracy cannot move, because the argmax doesn't change;
     - sampled pass@1 rises, because sampling lands on the argmax more often;
     - pass@8 trends down, because fewer distinct answers get drawn;
     - when the top answer is wrong, the gold loses mass. On unseen items `dpo-strict`'s top
       answer is wrong 87% of the time (`qa_unseen` 0.129 lenient; 89% strict), so the unseen gold log-probability
       falls hardest.
3. **Unseen:** strict accuracy −1.0 point (inside the noise). Gold-answer log-probability −1.33
   nats [−1.61, −1.08]: −1.32 on the answer tokens, −0.01 on the end token. A cost, named as one,
   through the mechanism above.
4. **Guards:**
   - MMLU (0.767 / 0.767) and GSM8K are within the noise.
   - Hallucination is 1 of 76 in both seeds. It carries no headline: the start sat at 3, the SFT
     twins at 4 and 1 and the DPO twins at 1 and 2. A 2-item move is inside what reseeding does.
   - Length is −3% and 0%.
   - The hack audit read 1 and 0 of 50: the one is the year hole above.
   - `grpo-seed1` misses two absolute lines by one item each: false abstain 2 of 108 (line ≤ 1) and
     `cite_valid` 0.982 (line 1.000). Both are inside the noise floor and equal to
     `sft-from-cpt-seed1`'s own values.
   - `grpo` meets every line.
5. **Same verifier, two algorithms**, each measured from its own start:

   | | seen pass@1 | seen pass@8 | seen gold_lp, answer tokens | unseen gold_lp, answer tokens | end token (seen / unseen) |
   |---|---|---|---|---|---|
   | DPO (from SFT) | +3.5 [+1.2, +6.0] | −3.0, inside the noise | −0.11 [−0.20, −0.03] | −0.38 [−0.50, −0.27] | +0.10 / +0.11 |
   | GRPO (from DPO) | +3.1 [+1.3, +5.0] | −3.6 [−8.1, +0.9] | −0.71 [−0.89, −0.53] | −1.32 [−1.58, −1.08] | +0.04 / −0.01 |

   The full table, with the unseen pass@k lines and the chain's cumulative change from SFT, is
   below the figures.
   - **Same gain, different price.** From their own starts, the two algorithms sharpened sampled
     accuracy by the same amount, about 3 points. What separates them is the price on the gold
     answer's tokens: DPO paid 0.11 nats on seen facts and 0.38 on unseen; GRPO paid 0.71 and 1.32.
   - **DPO's composite looked near zero on seen facts** because DPO also made the end token more
     likely (+0.10). It shifted the format toward stopping, and the composite netted that against
     the cost. GRPO's end token barely moved, so its composite and its answer-token cost agree.
   - **At this scale offline DPO was the gentler optimiser.** GRPO's extra cost is what running
     without a brake in the objective looks like (item 6).
   - **Cumulative from SFT:** the chain reaches +6.6 points on seen pass@1 [+3.6, +9.8] with pass@8
     at −6.6 [−12.3, −0.9]. Greedy accuracy moved in neither stage, and neither added a fact.
   - **Win rate:** not run for these rows. The judge failed its benchmark and its two orders agreed
     at coin-flip, so the column stays empty.
6. **Why it collapsed so fast: the repo imported Magistral's hyperparameters without the system
   that makes them act.**
   - **Where the ratio comes from:** Magistral's brake is a clip on the ratio between the
     trainer's policy and the generator's. That ratio departs from 1 because its generators run
     asynchronously, on stale weights.
   - **Here the ratio is always 1:** the colocated loop is synchronous, with one update per
     generation batch (μ = 1). The generator is the policy, so the ratio is identically 1 up to
     vLLM numerics.
     - `clip_ratio/high_mean` read 0 at every step.
     - ε_high 0.28 was inert by construction and gets no credit.
   - **Nothing bounded each step's change:** β was also 0, so only the learning rate and the stop
     rule limited it, and both runs collapsed inside 65 steps.
   - **It fit the set without transfer:** the train reward rose on tasks it had already seen (about
     1.3 visits each by step 52) while `grpo_val` stayed flat.
   - **The recipe transfers only with its off-policy degree.** Any one of three fixes gives the
     objective a brake:
     - μ = 2, so the clip has a ratio to bind;
     - β = 0.05 (Tülu 3's choice under PPO);
     - an off-policy lag.

![pass@1 against pass@8 on the closed-book eval, seen and unseen, strict scorer](../results/curves/passk.png)

## What I would do differently (Stage 5)

1. **Write every stop rule on a moving window from the start.**
   - The entropy rule read one batch, and one batch tracks its task mix.
   - It stopped `grpo-seed1` at step 34 (0.14 against a 0.159 line) while the 10-step mean sat at
     57% of the start.
   - It is the fourth pre-registered threshold set on a point reading where a window was meant.
2. **Import the system with the hyperparameters.**
   - Magistral's ε_high acts on the gap its asynchronous generators open. In a synchronous loop
     with one update per batch it is inert, and with β 0 nothing in the objective bounded a step.
   - μ = 2 (the clip then has a ratio to bind), β = 0.05, or an off-policy lag would each have
     braked the collapse that ended both runs within 65 steps.
3. **Give RL something to improve, not only something to recall.**
   - Closed-book facts are either in the model or not, and on-policy sampling can only reweight
     what it already produces.
   - Compute tasks (a formula from a passage, sampled inputs, a Python-checked answer) would have
     offered a skill GRPO can sharpen and that transfers to held-out items.
4. **Make chain retries exclusive.** A Modal retry re-ran `grpo-seed1`'s training on a warm GPU
   still holding the first run's engine, which failed for lack of memory. A pipeline that never
   re-enters the train step after it returns would have saved a relaunch.

<!-- stage5-tables:start -->
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
<!-- stage5-tables:end -->


## What I'd do next

**From Stage 5:**
1. **μ = 2 with ε_high 0.28, first.** Two optimisation passes per generation batch give the
   policy ratio room to leave 1, so the imported clip-higher does what it is for. Then β = 0.05,
   or an off-policy lag, if that isn't enough.
2. **Compute tasks:** a formula from a passage, sampled inputs and a Python-checked answer. This is
   a skill on-policy RL can sharpen and that transfers to held-out items; closed-book recall can
   only be reweighted.
3. **Refilter the task set on the new policy and run a second round:** Magistral's curriculum in
   miniature, once the objective has a brake.
