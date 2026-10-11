# struct-lm

> Independent project, using only public documents, open weights and open-source tools.

This project adapts a small open-weight language model, Mistral's **Ministral 3 8B**, to one
technical field: US federal structural-engineering documents. These are the design manuals,
standards and reports published by the Army Corps of Engineers (USACE), FEMA, the Federal Highway
Administration (FHWA), NIST and NASA. The model went through the four usual training steps, then
was compressed and served:

1. **Continued pre-training (CPT):** the model reads the documents.
2. **Supervised fine-tuning (SFT):** it learns from worked examples to answer questions in fixed
   formats.
3. **Preference training (DPO):** it is shown pairs of its own answers, one right and one wrong.
4. **Reinforcement learning (GRPO):** it is rewarded for answers that a rule checks as correct.
5. **Serving:** the model is compressed to fewer bits per weight and benchmarked for quality and
   speed.

Each step is measured against the untouched model (**Base**) and against Mistral's own chat
version of the same model (**Instruct**). Each training step below shows what it did, one real
example of its training data, its full scores and its training curves.

## In brief

- **Given the right passages, stock Instruct already answers about as well as the tuned model.**
  Judged against the source, the served model answers 92.6% of these questions correctly and
  Instruct 89.8%.
- **Tuning changed behaviour.** The tuned model cites only passages it was given (100%, against
  Instruct's 83.3%) and never refuses a question the passages answer (Instruct refuses 7.4%).
- **It learned the facts it was fine-tuned on.** With no passages, it answers 28.7% of questions
  on those facts after SFT, against Instruct's 11.4%.
- **It did not learn facts it only read.** On facts that appear only in the documents, it scores
  11.6% after SFT, the same as the untrained Base. Reading the documents made the right answers
  more probable to the model, but not probable enough to be the answers it gives.
- **Preference training and reinforcement learning added nothing measurable.** The
  preference-trained model is the one served; the reinforcement-learning model was not served.
- **Compressing the weights to 8 bits (FP8) cost no measurable quality; 4 bits (INT4) did.**

The main models side by side. "Taught" facts are those the fine-tuning data covers and
"document-only" facts the rest; the log-probability measures how likely the model finds the right
answer (closer to zero is better); ↓ marks lower is better; the last row is the noise floor. The
next section explains each.

<!-- headline-summary:start -->
| run | what it is | closed-book, taught (%) | closed-book, document-only (%) | gold-answer log-prob, document-only | grounded correct (%, judge) | unanswerable answered (%) ↓ | GSM8K (%) |
|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 12.6 | 11.6 | -6.36 | 84.3 | 89.5 | 79.3 |
| `instruct-8b` | Ministral 3 8B Instruct | 11.4 | 7.7 | -7.71 | 89.8 | 1.3 | 85.5 |
| `sft-from-base` | SFT on Base (control) | 23.9 | 8.4 | -6.20 | 92.6 | 6.6 | 79.2 |
| `sft-from-cpt` | CPT → SFT | 28.7 | 11.6 | -5.89 | 90.7 | 5.3 | 81.4 |
| `dpo-strict` | CPT → SFT → DPO (served) | 30.5 | 11.0 | -6.27 | 92.6 | 4.0 | 80.9 |
| `grpo` | CPT → SFT → DPO → GRPO (not served) | 29.9 | 9.7 | -7.54 | 93.5 | 1.3 | 82.0 |
|  | *floor, chat rows: two SFT seeds' gap, or SE* | 3.5 | 2.6 | 0.21 | 2.8 | 3.9 | 2.2 |
<!-- headline-summary:end -->

![The pipeline: the documents and the Base model feed CPT, then SFT, then DPO, which is served; GRPO, a control run and variants branch off](docs/diagrams/lifecycle.svg)

*Blue: the line that was served. Grey: controls and variants run for comparison. The names in
brackets are the run names in the tables.*

## How the model was tested

**The domain test:** 716 questions written by Mistral Large 3 (Mistral's large model) from
passages of the documents, each checked against its source passage before any model was scored.
There are four kinds:

| kind | questions | the model gets | it is right if |
|---|---|---|---|
| closed-book | 322 | the question alone | its answer matches the source's, checked by rule |
| grounded | 108 | the question and four passages, one of which holds the answer | the answer is correct (judged); its citations are scored separately |
| definitions | 210 | a domain term | its definition matches the source's (judged) |
| unanswerable | 76 | a question the four passages don't answer | it declines, rather than making an answer up |

An example closed-book question, from the passage quoted in the CPT section below:

```text
Q: What document provides dimensional data, weights, and truck reactions for cranes in
   Corps of Engineers powerhouses?
gold answer: EM 1110-2-4203
```

**How answers are scored:**
- **By rule** where a rule can decide: a closed-book answer must match the whole gold answer
  after normalisation (numbers within 2%, units compared), citations must name a given passage,
  and the fixed refusal sentence "Not in the provided passages." counts as declining. The closed-book rule is the strict checker;
  [`docs/verifiers.md`](docs/verifiers.md) has its rules and the holes found in earlier versions.
- **By a judge** where no rule can: Mistral Large 3, at temperature 0, decides whether a grounded
  answer or a definition is correct, and whether any other reply to an unanswerable question
  answers it anyway. Its verdicts were checked by hand, but the judge itself was
  never benchmarked.

**Taught and document-only facts.** Before any training, the closed-book questions were split
in two by source passage. For the **taught** half (167 questions), the fine-tuning data may draw
on the same passages, and for 121 of the 167 it asks the same fact in other words; it never
contains a test question itself. The **document-only** half (155 questions) comes from passages
kept out of all later training, so the model could learn those facts only by reading the
documents in CPT. The closed-book set grew from 130 to 322 items after continued pre-training, and
every model was rescored on the larger set.

**General benchmarks,** to catch forgetting: MMLU (multiple-choice general knowledge), GSM8K
(grade-school maths word problems) and HellaSwag (common-sense sentence completion), 5-shot,
with the standard lm-eval harness. They ran without the chat format and without a start-of-text
token, so the scores compare with each other but not with published ones.

**Two more measures:**
- **Perplexity:** how surprised the model is by text it hasn't trained on (lower is better), on
  held-out document text and on general web text (FineWeb-Edu).
- **Gold answer's log-probability:** how likely the model finds the correct answer, in nats (the
  natural-log unit) per answer; closer to zero is better. It moves before accuracy does, so it
  shows learning too small to change an answer.

**The noise floor.** Each main run was trained twice, with different random seeds. A difference
counts only if it is larger than the gap between those two runs, or than the measure's standard
error if that is larger. That is about one standard deviation: a screen, not a significance test.
With 155 to 167 questions per half, the floor is 2.6 to 3.5 points, so smaller gains can't be
seen.

## The documents

251 candidate PDFs from the five agencies, all US-government works and so in the public domain.
Copyrighted standards that the documents cite (ASCE 7, the AISC steel manual) were excluded. The
text was extracted, filtered (table debris, contents pages, reference lists, garbled scans),
deduplicated and scrubbed of personal data, leaving 246 documents and 20.6M tokens; the training split is 19.4M.

**Contamination checks:** a 13-gram overlap search (GPT-3's method) finds no GSM8K or HellaSwag
question in the training text, and 21 of 14,042 MMLU questions that share a 13-gram with it, all
stock phrases. No test
question is copied into the SFT, DPO or GRPO data
([checks](docs/stage1.md#contamination-checks)). Full write-up: [`docs/stage1.md`](docs/stage1.md).

## Step 1: continued pre-training (CPT)

**What it is:** the model keeps doing what it did in pre-training, predicting the next token, but
on the documents. It read the training split once, mixed with 10% general web text (FineWeb-Edu)
so it wouldn't forget general skills. It was trained as a LoRA adapter: small extra matrices
(rank 64) trained on top of the frozen weights, on one H100 GPU.

**Training data sample:** plain document text. This excerpt is from EM 1110-2-3001, *Planning and
Design of Hydroelectric Power Plant Structures* (USACE):

```text
Dimensional data, weights, and truck reactions for cranes installed in Corps of Engineers
powerhouses are given in EM 1110-2-4203. An impact allowance of 10 percent for cranes over
80-ton capacity, and from 12 percent to 18 percent for smaller cranes, should be added to the
static loads. Side thrust at the top of the rail should be taken as 10 percent of the summation
of the trolley weight and rated capacity, with three-fourths of this amount distributed equally
among the wheels at either side of the runway.
```

**Scores:**

<!-- step-cpt:start -->
| measure | Base | CPT + 10% general text (used next) | CPT, documents only | same, seed 1 | CPT, all weights | noise floor |
|---|---|---|---|---|---|---|
| closed-book, taught facts (%) | 12.6 | 15.0 | 15.6 | 13.2 |  | 2.8 |
| closed-book, document-only facts (%) | 11.6 | 11.0 | 13.6 | 12.9 |  | 2.7 |
| closed-book, document and section numbers (%) | 7.8 | 12.5 | 12.5 | 12.5 |  | 4.1 |
| gold answer's log-probability, taught facts (nats) | -6.28 | -5.93 | -5.84 | -5.84 |  | 0.03 |
| gold answer's log-probability, document-only facts (nats) | -6.36 | -5.70 | -5.68 | -5.70 |  | 0.03 |
| grounded: answer correct (%, judge) | 84.3 | 77.8 | 83.3 | 79.6 | 88.9 | 3.7 |
| grounded: citations name only given passages (%) | 12.0 | 5.6 | 14.8 | 10.2 | 14.8 | 4.6 |
| grounded: cited passages support the answer (%, judge) | 8.3 | 3.7 | 7.4 | 3.7 | 10.2 | 3.7 |
| grounded: answerable question declined (%) ↓ | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| unanswerable: answered anyway (%) ↓ | 89.5 | 93.4 | 90.8 | 93.4 | 92.1 | 3.3 |
| definitions correct (%, judge) | 70.5 | 70.0 | 71.0 | 70.0 | 74.3 | 3.1 |
| MMLU (%) | 76.7 | 76.6 | 76.4 | 76.5 | 76.2 | 0.3 |
| GSM8K (%) | 79.3 | 79.1 | 78.5 | 78.6 | 76.3 | 1.1 |
| HellaSwag (%) | 80.1 | 80.0 | 80.1 | 80.0 | 79.8 | 0.4 |
| perplexity, held-out document text ↓ | 6.88 | 6.72 | 6.72 | 6.72 | 6.73 | 0.02% |
| perplexity, general web text ↓ | 8.15 | 7.97 | 8.18 | 8.17 | 8.25 | 0.23% |

- The floor is the gap between the two documents-only seeds, or the standard error if larger; perplexity's floor is that gap as a % change.
- CPT, all weights has no closed-book numbers: its weights were deleted before the closed-book set grew, and a full-weight run can't be rebuilt from an adapter.
<!-- step-cpt:end -->

**What it found:**
- **Perplexity on held-out document text fell 2.33%. The aim, set before training, was 20%, so
  CPT missed its aim.** General-text perplexity stayed inside the 3% rise allowed (+0.40%
  without the general text), and MMLU, GSM8K and HellaSwag stayed within the floor.
- **The right answers became more probable to the model** (the log-probability rows), **but
  closed-book accuracy didn't rise beyond the floor.** CPT made the text familiar; it didn't make
  the model able to recall it.

**Choices:**
- **10% general text, kept on thin evidence.** The rule set before training adopted it if it beat
  the documents-only run on MMLU or general-text perplexity by more than the seed gap. MMLU rose
  0.2 points against a 0.1-point gap, inside MMLU's own 0.34 standard error, and the general-text
  perplexity is measured on the same kind of web text it trained on. It cost 6.5 points of grounded
  accuracy (floor 3.7). It was kept because it adds only 10% more tokens and SFT restored the
  citations.
- **A LoRA adapter, not training all the weights.** The all-weights run (two GPUs, one run)
  memorised more for the same gain on held-out text: perplexity on its own training text fell 22%,
  against LoRA's 8%. Its general-text perplexity rose 1.19%, against LoRA's +0.40% (itself past the
  0.23% floor, so LoRA isn't free), and it lost 3.0 points of GSM8K (floor 1.1).

![CPT loss curves](results/curves/cpt.png)

*Training loss (moving average) and validation loss: validation loss flattens by the end of the
one pass.*

Full write-up: [`docs/stage2.md`](docs/stage2.md).

## Step 2: supervised fine-tuning (SFT)

**What it is:** training on example conversations, a prompt and the answer the model should give,
so it learns the answer formats. The training set is 2,436 examples:
- **1,778 domain examples** in the test's closed-book, grounded and definition formats, written by
  Mistral Large 3 and Mistral Medium 3.5 from the documents' passages (Medium 3.5 wrote 415), and
  each checked by rule, by the judge and by a full read against its passage;
- **183 refusals** for unanswerable questions, all the one fixed sentence;
- **475 general records** from AI2's Tülu 3 instruction mixture, so general chat skill isn't lost.

SFT ran twice from different starts: on the CPT model (the line that was served), and on the
Base as a control, to see what CPT added.

**Training data sample:** one domain example, on the passage above (from
`data/sft/hosted/train.jsonl`):

```text
user:      Short answer only, please: When assessing crane runways in hydroelectric power
           plants, what percentage of the total trolley weight plus rated capacity must be
           factored in as side thrust at the rail top?
assistant: 10 percent
```

**Scores:**

<!-- step-sft:start -->
| measure | Base | Instruct | SFT on Base | SFT on CPT (used next) | same, seed 1 | noise floor |
|---|---|---|---|---|---|---|
| closed-book, taught facts (%) | 12.6 | 11.4 | 23.9 | 28.7 | 25.8 | 3.5 |
| closed-book, document-only facts (%) | 11.6 | 7.7 | 8.4 | 11.6 | 11.0 | 2.6 |
| closed-book, document and section numbers (%) | 7.8 | 3.1 | 10.9 | 20.3 | 23.4 | 5.0 |
| closed-book, taught facts, sampled (%, mean of 8 answers) |  | 7.9 |  | 21.3 |  |  |
| gold answer's log-probability, taught facts (nats) | -6.28 | -7.31 | -4.97 | -4.70 | -4.52 | 0.17 |
| gold answer's log-probability, document-only facts (nats) | -6.36 | -7.71 | -6.20 | -5.89 | -5.68 | 0.21 |
| grounded: answer correct (%, judge) | 84.3 | 89.8 | 92.6 | 90.7 | 93.5 | 2.8 |
| grounded: citations name only given passages (%) | 12.0 | 83.3 | 100.0 | 100.0 | 98.2 | 1.8 |
| grounded: cited passages support the answer (%, judge) | 8.3 | 78.7 | 87.0 | 86.1 | 88.0 | 3.3 |
| grounded: answerable question declined (%) ↓ | 0.9 | 7.4 | 0.0 | 0.0 | 0.9 | 0.9 |
| unanswerable: answered anyway (%) ↓ | 89.5 | 1.3 | 6.6 | 5.3 | 1.3 | 3.9 |
| definitions correct (%, judge) | 70.5 | 78.6 | 83.3 | 83.3 | 85.7 | 2.6 |
| MMLU (%) | 76.7 | 76.1 | 76.7 | 76.6 | 77.0 | 0.4 |
| GSM8K (%) | 79.3 | 85.5 | 79.2 | 81.4 | 79.2 | 2.2 |
| HellaSwag (%) | 80.1 | 80.1 | 79.5 | 79.4 | 80.0 | 0.6 |
| perplexity, held-out document text ↓ | 6.88 |  | 7.01 | 6.87 | 6.87 | 0.08% |
| perplexity, general web text ↓ | 8.15 |  | 8.23 | 8.06 | 8.07 | 0.07% |

- The floor is the gap between the two SFT-on-CPT seeds, or the standard error if larger.
- Base is scored as a base model (a plain-text prompt, no chat format), so its log-probabilities don't compare with the other columns'.
- Sampled scores were measured for Instruct and SFT on CPT only.
<!-- step-sft:end -->

**What it found** (SFT on CPT against Instruct, unless stated):
- **Taught facts, closed-book:** 28.7% against 11.4% (paired 95% CI +10.8 to +24.6 points).
- **Document-only facts:** 11.6% against 7.7%. The gap clears the 2.6-point floor, but its paired
  95% CI (−1.3 to +9.0) includes 0, and the untrained Base also scores 11.6%.
- **With passages:** citations name only given passages 100% of the time against 83.3% (a format
  check, on a format SFT trained on); no answerable question is refused (Instruct refuses 7.4%);
  grounded accuracy is level with Instruct's (judge: 90.7% against 89.8%).
- **Unanswerable questions:** its hallucination is worse than Instruct's, 4 of 76 answered
  anyway against 1, but that gap equals the gap between SFT's own two seeds, so it is not beyond
  the floor.
- **What CPT added:** against SFT on the Base, the right answers to document-only facts are 0.46
  nats more probable (95% CI +0.28 to +0.66), from only two runs of each. SFT on the Base lost
  some of the Base's document-only accuracy; SFT on CPT kept it.
- **One pass over the data:** a rule set before training picked the epoch. In the second epoch,
  validation loss on closed-book examples rose from 1.17 to 1.30, and all four runs overfit the
  same way.

![SFT loss curves](results/curves/sft.png)

*Training and validation loss: validation loss is lowest at the end of the first epoch in all four
runs.*

Full write-up: [`docs/stage3.md`](docs/stage3.md).

## Step 3: preference training (DPO)

**What it is:** DPO (Direct Preference Optimisation) shows the model pairs of answers to one
prompt, a preferred ("chosen") and a dispreferred ("rejected") one. It raises the probability of
the chosen answer relative to the rejected one, while staying close to the starting model.

The pairs came from the SFT model's own answers: it answered each training prompt several times,
and rule checks labelled each answer right or wrong. A closed-book answer must match the gold
answer, an unanswerable question must be declined, and a grounded answer must cite the passage
that holds the answer. That gave 445 training pairs.

A judge was planned to label pairs too, but it failed its own accuracy test first. Against
defects a careful reader had found, it caught 28% in grounded answers and 9% in definitions,
against a 50% bar, so no pair was labelled by a judge.

**Training data sample:** one pair, both answers sampled from the SFT model (from
`data/dpo/strict/train.jsonl`):

```text
prompt:   Short answer only, please: When assessing crane runways in hydroelectric power
          plants, what percentage of the total trolley weight plus rated capacity must be
          factored in as side thrust at the rail top?
chosen:   10 percent     (matches the gold answer)
rejected: 5 percent      (wrong)
```

**Scores:**

<!-- step-dpo:start -->
| measure | SFT (start) | DPO, first labels | same, seed 1 | DPO, corrected labels (served) | DPO, 2 epochs | noise floor |
|---|---|---|---|---|---|---|
| closed-book, taught facts (%) | 28.7 | 30.5 | 28.7 | 30.5 | 25.8 | 3.5 |
| closed-book, document-only facts (%) | 11.6 | 11.6 | 11.0 | 11.0 | 12.9 | 2.6 |
| closed-book, document and section numbers (%) | 20.3 | 20.3 | 17.2 | 18.8 | 14.1 | 5.0 |
| closed-book, taught facts, sampled (%, mean of 8 answers) | 21.3 | 24.7 | 24.2 | 24.9 |  |  |
| gold answer's log-probability, taught facts (nats) | -4.70 | -4.70 | -4.85 | -4.81 | -7.82 | 0.17 |
| gold answer's log-probability, document-only facts (nats) | -5.89 | -6.16 | -6.28 | -6.27 | -10.57 | 0.21 |
| grounded: answer correct (%, judge) | 90.7 | 91.7 | 90.7 | 92.6 | 90.7 | 2.8 |
| grounded: citations name only given passages (%) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 1.8 |
| grounded: cited passages support the answer (%, judge) | 86.1 | 87.0 | 88.0 | 86.1 | 85.2 | 3.3 |
| grounded: answerable question declined (%) ↓ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 |
| unanswerable: answered anyway (%) ↓ | 5.3 | 1.3 | 2.6 | 4.0 | 1.3 | 3.9 |
| definitions correct (%, judge) | 83.3 | 83.8 | 84.8 | 82.4 | 85.2 | 2.6 |
| judge prefers its answer to SFT's (%, 50 = even) |  | 52 | 52 | 54 | 52 |  |
| MMLU (%) | 76.6 | 76.7 | 76.5 | 76.7 | 76.8 | 0.4 |
| GSM8K (%) | 81.4 | 81.0 | 80.7 | 80.9 | 80.6 | 2.2 |
| HellaSwag (%) | 79.4 | 79.6 | 79.5 | 79.6 | 80.2 | 0.6 |
| perplexity, held-out document text ↓ | 6.87 | 6.88 | 6.89 | 6.88 | 6.97 | 0.08% |
| perplexity, general web text ↓ | 8.06 | 8.07 | 8.07 | 8.06 | 8.11 | 0.07% |

- The floor is SFT's (the gap between its two seeds, or the standard error if larger).
- The judge's preference is reported only: that judge failed its own accuracy check.
- DPO, 2 epochs has no sampled score: it was an ablation.
<!-- step-dpo:end -->

**What it found:**
- **None of its five main measures improved beyond the floor:** closed-book accuracy,
  gold-answer probability, hallucination, refused answerable questions and citation format.
- **The first-label runs' drop in hallucination** cleared the floor as first set, but not the
  stricter floor adopted after the results were in, which also counts the starting model's own
  seed gap.
- **Sampled accuracy on taught facts rose 3.5 points** (95% CI +1.2 to +6.0), a measure added
  after the fact. The served model doesn't benefit: it answers with its single most likely answer,
  not a sample.
- **The cost:** the right answers to document-only facts became 0.38 nats less probable, beyond
  the 0.21 floor (0.11 on taught facts, inside it).
- **The first labels were wrong in places.** The first labeller accepted any fragment of the gold
  answer, and an answer matching only the first number of a multi-number one, so 80 of 458
  closed-book chosen answers (17%) fail the strict checker, and 79 of those are wrong on reading. The pairs were relabelled with the strict checker, which trained
  "DPO, corrected labels", the served model.
- **Two epochs instead of one** (one run) pushed the right answers down: the gold answers'
  log-probability fell 3.1 nats on taught facts and 4.4 on document-only ones, and taught accuracy
  fell 4.8 points.
- **The judge's preference,** reported only, is about even with SFT.

![DPO training curves](results/curves/dpo.png)

*Training loss, validation loss and how often the chosen answer is preferred: two epochs fit the
training pairs while validation loss stalls.*

Full write-up: [`docs/stage4.md`](docs/stage4.md).

## Step 4: reinforcement learning (GRPO)

**What it is:** GRPO (Group Relative Policy Optimisation) has the model answer each question 8
times, scores every answer, and moves the model toward the answers that scored above their
group's average. No right answer is shown to the model; the reward does the teaching. The reward is
0.1 for a well-formed answer plus 0.9 if the strict checker accepts it, minus up to 0.1 for
length. There were 622 training questions, chosen so the DPO model got between 1 and 7 of 8 tries
right: a question it always or never gets right teaches nothing.

**Training data sample:** a training question and the 8 answers the model gave it at training
step 20, with their rewards (from `data/grpo/train.jsonl` and
`results/runs/grpo/rollouts.jsonl`):

```text
question:    Short answer only, please: When assessing crane runways in hydroelectric power
             plants, what percentage of the total trolley weight plus rated capacity must be
             factored in as side thrust at the rail top?
gold answer: 10 percent

answer           reward
15 percent       0.1
10 percent       1.0    (correct)
2 percent        0.1
5%               0.1
fifteen percent  0.1
15%              0.1
15 percent       0.1
15%              0.1
```

**Scores:**

<!-- step-grpo:start -->
| measure | DPO (start) | GRPO | GRPO, seed 1 | noise floor |
|---|---|---|---|---|
| closed-book, taught facts (%) | 30.5 | 29.9 | 28.7 | 3.5 |
| closed-book, document-only facts (%) | 11.0 | 9.7 | 10.3 | 2.6 |
| closed-book, document and section numbers (%) | 18.8 | 20.3 | 18.8 | 5.0 |
| closed-book, taught facts, sampled (%, mean of 8 answers) | 24.9 | 27.8 | 28.1 |  |
| gold answer's log-probability, taught facts (nats) | -4.81 | -5.52 | -5.52 | 0.17 |
| gold answer's log-probability, document-only facts (nats) | -6.27 | -7.54 | -7.65 | 0.21 |
| grounded: answer correct (%, judge) | 92.6 | 93.5 | 91.7 | 2.8 |
| grounded: citations name only given passages (%) | 100.0 | 100.0 | 98.2 | 1.8 |
| grounded: cited passages support the answer (%, judge) | 86.1 | 88.0 | 88.9 | 3.3 |
| grounded: answerable question declined (%) ↓ | 0.0 | 0.0 | 1.8 | 0.9 |
| unanswerable: answered anyway (%) ↓ | 4.0 | 1.3 | 1.3 | 3.9 |
| definitions correct (%, judge) | 82.4 | 84.3 | 85.2 | 2.6 |
| MMLU (%) | 76.7 | 76.7 | 76.7 | 0.4 |
| GSM8K (%) | 80.9 | 82.0 | 80.2 | 2.2 |
| HellaSwag (%) | 79.6 | 79.8 | 79.8 | 0.6 |
| perplexity, held-out document text ↓ | 6.88 | 6.93 | 6.93 | 0.08% |
| perplexity, general web text ↓ | 8.06 | 8.09 | 8.09 | 0.07% |

- The floor is SFT's (the gap between its two seeds, or the standard error if larger).
<!-- step-grpo:end -->

**What it found:**
- **The aim, set before training, was to raise taught-fact accuracy by 3 to 8 points. It fell
  1.2.**
- **Sampled accuracy rose 3.1 points** (95% CI +1.3 to +5.0), the mean of its two seeds. It cost
  1.32 nats on the right answers to document-only facts and 0.71 on taught ones.
- **Both runs stopped early when the model's output entropy collapsed:** its 8 answers to a
  question became nearly identical, which leaves the group nothing to compare. The setting meant to
  limit each update never engaged, because each batch was used for only one update. Seed 1 was
  stopped at step 34 by a rule reading one batch; the rule was rewritten after it fired, and the
  run resumed to step 65.
- **Not served.**

![GRPO training curves](results/curves/grpo.png)

*Reward, output entropy and validation accuracy over training: reward climbs as entropy collapses
and validation pass@8 (any of 8 answers right) falls.*

Full write-up: [`docs/stage5.md`](docs/stage5.md).

## Step 5: serving

**What it is:** the served model (DPO, corrected labels) was compressed three ways and run with
vLLM on one H100:
- **FP8:** weights and activations in 8 bits;
- **FP8 + FP8 KV cache:** the same, plus the attention cache in 8 bits;
- **INT4:** weights in 4 bits (GPTQ, calibrated on 512 domain records).

A variant passes the quality check if no measure moves beyond the floor.

**Scores:**

<!-- step-serving:start -->
| measure | bf16 (served) | FP8 (served) | FP8 + FP8 KV cache | INT4 | noise floor |
|---|---|---|---|---|---|
| closed-book, taught facts (%) | 30.5 | 29.3 | 26.4 | 27.5 | 3.5 |
| closed-book, document-only facts (%) | 11.0 | 9.7 | 11.6 | 8.4 | 2.6 |
| closed-book, document and section numbers (%) | 18.8 | 17.2 | 17.2 | 10.9 | 5.0 |
| gold answer's log-probability, taught facts (nats) | -4.81 | -4.82 | -4.87 | -5.12 | 0.17 |
| gold answer's log-probability, document-only facts (nats) | -6.27 | -6.27 | -6.33 | -6.70 | 0.21 |
| grounded: answer correct (%, judge) | 92.6 | 90.7 | 93.5 | 89.8 | 2.8 |
| grounded: citations name only given passages (%) | 100.0 | 100.0 | 100.0 | 100.0 | 1.8 |
| grounded: cited passages support the answer (%, judge) | 86.1 | 85.2 | 84.3 | 84.3 | 3.3 |
| grounded: answerable question declined (%) ↓ | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 |
| unanswerable: answered anyway (%) ↓ | 4.0 | 5.3 | 6.6 | 4.0 | 3.9 |
| definitions correct (%, judge) | 82.4 | 82.9 | 83.3 | 77.6 | 2.6 |
| MMLU (%) | 76.7 |  |  |  | 0.4 |
| GSM8K (%) | 80.9 |  |  |  | 2.2 |
| GSM8K, the quality check's run (%) | 80.9 | 80.7 |  | 74.8 | 2.2 |
| HellaSwag (%) | 79.6 |  |  |  | 0.6 |
| perplexity, held-out document text ↓ | 6.88 |  |  |  | 0.08% |
| perplexity, general web text ↓ | 8.06 |  |  |  | 0.07% |
| first-token latency, one request (ms) ↓ | 17.9 | 35.2 |  |  |  |
| time per output token, one request (ms) ↓ | 6.90 | 4.70 |  |  |  |

- The floor is SFT's (the gap between its two seeds, or the standard error if larger).
- The quality check reran GSM8K with one start-of-text token for every variant; FP8 + FP8 KV cache's run, and its and INT4's latency runs, weren't done (Modal spend limit).
- Latency is the median on one H100, one request at a time, mean of two runs.
<!-- step-serving:end -->

**What it found:**
- **FP8 passed every quality check.** FP8 + FP8 KV cache failed (taught-fact accuracy −4.2,
  against a 3.5 floor). INT4 failed three checks, including GSM8K (−6.1, against 2.2); its
  calibration used domain records only, which may be the cause.
- **Speed:** FP8 generates each token faster, but takes twice as long to the first token at low
  load, which is still unexplained. At 64 concurrent requests FP8 serves 22% more requests per
  second (raw throughput, not requests within a latency target). So bf16 serves up to 16
  requests per second, the highest rate measured, and FP8 near saturation.
- [`DEPLOY.md`](DEPLOY.md) has the serving commands, the load curves and the cost per request.

Full write-up: [`docs/stage6.md`](docs/stage6.md).

## The served model and its limits

**Served: "DPO, corrected labels" (`dpo-strict`).** It is within noise of the SFT model on every
measure. It carried forward because DPO's plan had no fall-back to the SFT model, not on merit.
SFT on CPT is the better choice, since it keeps the 0.38 nats DPO lost; switching needs its own
FP8 quality check.

**Use and limits:**
- **Use it grounded:** answering from retrieved passages, with citations, and have an engineer
  check every value against its source. The field is safety-critical.
- **Not closed-book:** with no passages, it answers 30.5% of taught facts and 11.0% of
  document-only ones.
- **Retrieval is untested:** the grounded test supplies each question's source passage among four,
  so it measures reading, not finding.
- **Chat behaviour beyond the test is untested:** the general benchmarks ran without the chat
  format, so instruction-following, chat quality and safety weren't evaluated.
- **No domain experts:** Mistral Large 3 wrote the test and most of the SFT data, which could favour
  SFT on taught facts, and Claude reviewed the items. No structural engineer saw items or outputs.
- **The test is too small for 3-point gains:** with 155 to 167 questions per half and two seeds,
  the floors are 2.6 to 3.5 points and paired CIs about ±5 to ±7 points. No minimum detectable
  effect was set before the runs.

## What went wrong, and what next

**What went wrong:**
1. **The DPO labeller passed wrong answers** (Step 3). The strict checker then rescored every
   model: the 8B rows moved 0–1.6 points and Mistral Large 3 moved 1.9. Five pairs of rows less
   than a point apart swapped order; no planned comparison involves any of them.
2. **The strict checker had three holes, each found and fixed:** answer fragments (found by
   auditing the labeller's passes), comma-separated lists (found by reading the code), and years
   within 2% of the gold (found by auditing GRPO's highest-reward answers; only this one was
   exploited, on one question). Each fix has a test, listed in
   [`docs/verifiers.md`](docs/verifiers.md).
3. **The preference judge failed its accuracy test** (Step 3), so no judge labels anything.
4. **GRPO's entropy collapsed** (Step 4).
5. **FP8's slower first token** is unexplained (Step 5).
6. **The general benchmarks have sent no start-of-text token since the first baseline.** Rows still
   compare with each other: GSM8K with the token scored the same 1,067 of 1,319 on the served
   model.
7. **The licence audit ran after training,** so the weights can't be published
   ([Weights](#weights)).

**Lessons on method:**
- **Stop rules should read a window, not one point.** Four rules read single points: the check
  that merging the adapter changes nothing, the first-step loss band, DPO's checkpoint rule and
  GRPO's entropy on one batch. Three were changed, two after the results they govern; GRPO's was
  rewritten after it fired.
- **Split a measure into its parts before comparing it.** The gold answer's log-probability
  includes the end-of-answer token, which hid DPO's shift toward stopping sooner.
- **Count the starting model's own seed gap in the floor** when comparing two trained runs. The
  repo added this only after DPO's results were in, and it moved two DPO results inside the noise:
  the drop in hallucination and the loss on document-only answers.
- **Fix the test before the first training run.** CPT's perplexity aim was set for the wrong data
  scale, and the closed-book set grew after CPT.

**Next, most important first:**
1. A licence-clean SFT set, then retraining and publishing the weights ([Weights](#weights)).
2. Compress and check SFT on CPT, and serve it if it passes.
3. Before any more training: a larger test sized for 3-point gains, a third seed per run, items
   reviewed by structural engineers, a real retriever in the grounded test, and chat and safety
   tests.
4. Calculation tasks for GRPO: a formula from a passage, sampled inputs, a checked result.
5. Two updates per GRPO batch, so the update limit can engage; then a penalty for drifting from
   the starting model if needed.
6. An extra loss on DPO's chosen answers that keeps their probability from falling (RPO).
7. A per-claim support check as the grounded judge, benchmarked first.
8. Diagnose FP8's first-token time, and run the serving tests not run: FP8 KV cache's GSM8K and
   speed, INT4's speed, and n-gram speculative decoding.
9. INT4 calibrated on mixed data, or AWQ.
10. More varied CPT exposure, such as restatements of the facts in other words.

## All runs in one place

Every model and variant, Base first, with the noise floor for each format at the bottom. Rates
are in %; ↓ marks lower is better; "taught" and "document-only" are the two halves of the
closed-book test; "document numbers" are closed-book questions answered by a document or section
number. **[base format]** rows are scored as base models (plain-text prompts, no chat format), so
their log-probabilities don't compare with the other rows'. The end-token columns are the
log-probability of the end-of-answer token alone. Runs trained once are read against the same
floor as those trained twice.

**Knowledge (closed-book: no passages in the prompt)**

<!-- headline-knowledge:start -->
| run | what it is | closed-book, taught (%) | closed-book, document-only (%) | closed-book, document numbers (%) | gold-answer log-prob, taught | gold-answer log-prob, document-only | end-token log-prob, taught | end-token log-prob, document-only |
|---|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 12.6 | 11.6 | 7.8 | -6.28 | -6.36 | -0.44 | -0.47 |
| `instruct-8b` | Ministral 3 8B Instruct (Mistral's chat model) | 11.4 | 7.7 | 3.1 | -7.31 | -7.71 | -0.70 | -0.87 |
| `mistral-large-3` | Mistral Large 3 (API, closed-book only) | 26.4 | 25.8 | 40.6 |  |  |  |  |
| `cpt-8b` | CPT, documents only [base format] | 15.6 | 13.6 | 12.5 | -5.84 | -5.68 | -0.40 | -0.45 |
| `cpt-8b-seed1` | CPT, documents only, seed 1 [base format] | 13.2 | 12.9 | 12.5 | -5.84 | -5.70 | -0.40 | -0.45 |
| `cpt-8b-replay10` | CPT + 10% general text (used next) [base format] | 15.0 | 11.0 | 12.5 | -5.93 | -5.70 | -0.43 | -0.46 |
| `sft-from-base` | SFT on Base (control) | 23.9 | 8.4 | 10.9 | -4.97 | -6.20 | -0.44 | -0.58 |
| `sft-from-base-seed1` | SFT on Base, seed 1 | 23.4 | 7.7 | 14.1 | -4.87 | -6.20 | -0.38 | -0.54 |
| `sft-from-cpt` | SFT on CPT (used next) | 28.7 | 11.6 | 20.3 | -4.70 | -5.89 | -0.42 | -0.53 |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 25.8 | 11.0 | 23.4 | -4.52 | -5.68 | -0.35 | -0.48 |
| `dpo` | DPO, first labels | 30.5 | 11.6 | 20.3 | -4.70 | -6.16 | -0.30 | -0.38 |
| `dpo-seed1` | DPO, first labels, seed 1 | 28.7 | 11.0 | 17.2 | -4.85 | -6.28 | -0.35 | -0.45 |
| `dpo-2ep` | DPO, 2 epochs (ablation; failed the merge check) | 25.8 | 12.9 | 14.1 | -7.82 | -10.57 | -0.20 | -0.28 |
| `dpo-strict` | DPO, corrected labels (served, bf16) | 30.5 | 11.0 | 18.8 | -4.81 | -6.27 | -0.32 | -0.42 |
| `grpo` | GRPO (not served) | 29.9 | 9.7 | 20.3 | -5.52 | -7.54 | -0.27 | -0.41 |
| `grpo-seed1` | GRPO, seed 1 | 28.7 | 10.3 | 18.8 | -5.52 | -7.65 | -0.30 | -0.45 |
| `dpo-strict-fp8` | served model in FP8 (passed the quality check) | 29.3 | 9.7 | 17.2 | -4.82 | -6.27 | -0.32 | -0.44 |
| `dpo-strict-fp8kv` | FP8 + FP8 KV cache (failed the quality check) | 26.4 | 11.6 | 17.2 | -4.87 | -6.33 | -0.33 | -0.45 |
| `dpo-strict-w4a16` | INT4 W4A16 (failed the quality check) | 27.5 | 8.4 | 10.9 | -5.12 | -6.70 | -0.30 | -0.36 |
|  | *floor, base-format rows: two CPT seeds' gap, or SE* | 2.8 | 2.7 | 4.1 | 0.03 | 0.03 | 0.01 | 0.01 |
|  | *floor, chat rows: two SFT seeds' gap, or SE* | 3.5 | 2.6 | 5.0 | 0.17 | 0.21 | 0.07 | 0.05 |

**Blank cells:**
- `cpt-8b-full` has no numbers: its weights were deleted before the closed-book task grew, and full-parameter weights can't be rebuilt from an adapter.
- `mistral-large-3` has no `gold_lp`, which is computed on the weights in the generation engine; Large 3 ran through the API.
<!-- headline-knowledge:end -->

**Behaviour (with passages: grounded answers, citations, refusals; and definitions)**

<!-- headline-behaviour:start -->
| run | what it is | grounded correct (%, judge) | citations valid (%) | citations supported (%, judge) | unanswerable answered (%) ↓ | answerable declined (%) ↓ | definitions (%, judge) |
|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 84.3 | 12.0 | 8.3 | 89.5 | 0.9 | 70.5 |
| `instruct-8b` | Ministral 3 8B Instruct (Mistral's chat model) | 89.8 | 83.3 | 78.7 | 1.3 | 7.4 | 78.6 |
| `cpt-8b` | CPT, documents only [base format] | 83.3 | 14.8 | 7.4 | 90.8 | 0.0 | 71.0 |
| `cpt-8b-seed1` | CPT, documents only, seed 1 [base format] | 79.6 | 10.2 | 3.7 | 93.4 | 0.0 | 70.0 |
| `cpt-8b-replay10` | CPT + 10% general text (used next) [base format] | 77.8 | 5.6 | 3.7 | 93.4 | 0.0 | 70.0 |
| `cpt-8b-full` | CPT, all weights (ablation) [base format] | 88.9 | 14.8 | 10.2 | 92.1 | 0.0 | 74.3 |
| `sft-from-base` | SFT on Base (control) | 92.6 | 100.0 | 87.0 | 6.6 | 0.0 | 83.3 |
| `sft-from-base-seed1` | SFT on Base, seed 1 | 89.8 | 99.1 | 88.0 | 1.3 | 0.9 | 82.9 |
| `sft-from-cpt` | SFT on CPT (used next) | 90.7 | 100.0 | 86.1 | 5.3 | 0.0 | 83.3 |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 93.5 | 98.2 | 88.0 | 1.3 | 0.9 | 85.7 |
| `dpo` | DPO, first labels | 91.7 | 100.0 | 87.0 | 1.3 | 0.0 | 83.8 |
| `dpo-seed1` | DPO, first labels, seed 1 | 90.7 | 100.0 | 88.0 | 2.6 | 0.0 | 84.8 |
| `dpo-2ep` | DPO, 2 epochs (ablation; failed the merge check) | 90.7 | 100.0 | 85.2 | 1.3 | 0.0 | 85.2 |
| `dpo-strict` | DPO, corrected labels (served, bf16) | 92.6 | 100.0 | 86.1 | 4.0 | 0.0 | 82.4 |
| `grpo` | GRPO (not served) | 93.5 | 100.0 | 88.0 | 1.3 | 0.0 | 84.3 |
| `grpo-seed1` | GRPO, seed 1 | 91.7 | 98.2 | 88.9 | 1.3 | 1.8 | 85.2 |
| `dpo-strict-fp8` | served model in FP8 (passed the quality check) | 90.7 | 100.0 | 85.2 | 5.3 | 0.0 | 82.9 |
| `dpo-strict-fp8kv` | FP8 + FP8 KV cache (failed the quality check) | 93.5 | 100.0 | 84.3 | 6.6 | 0.0 | 83.3 |
| `dpo-strict-w4a16` | INT4 W4A16 (failed the quality check) | 89.8 | 100.0 | 84.3 | 4.0 | 0.0 | 77.6 |
|  | *floor, base-format rows: two CPT seeds' gap, or SE* | 3.7 | 4.6 | 3.7 | 3.3 | 0.0 | 3.1 |
|  | *floor, chat rows: two SFT seeds' gap, or SE* | 2.8 | 1.8 | 3.3 | 3.9 | 0.9 | 2.6 |

**Blank cells:**
- `mistral-large-3` has no numbers: it is an API reference, run closed-book only.
<!-- headline-behaviour:end -->

**General capability and speed**

<!-- headline-general:start -->
| run | what it is | MMLU (%) | GSM8K (%) | HellaSwag (%) | perplexity, documents ↓ | perplexity, web text ↓ | first token, 1 request (ms) | per token, 1 request (ms) | latency measured on |
|---|---|---|---|---|---|---|---|---|---|
| `base-8b-hf` | Ministral 3 8B Base [base format] | 76.7 | 79.3 | 80.1 | 6.88 | 8.15 | 15.3 | 6.60 | 1 sample, GPU not recorded |
| `instruct-8b` | Ministral 3 8B Instruct (Mistral's chat model) | 76.1 | 85.5 | 80.1 |  |  | 17.7 | 6.60 | 1 sample, GPU not recorded |
| `cpt-8b` | CPT, documents only [base format] | 76.4 | 78.5 | 80.1 | 6.72 | 8.18 | 16.5 | 6.90 | 1 sample, GPU not recorded |
| `cpt-8b-seed1` | CPT, documents only, seed 1 [base format] | 76.5 | 78.6 | 80.0 | 6.72 | 8.17 |  |  |  |
| `cpt-8b-replay10` | CPT + 10% general text (used next) [base format] | 76.6 | 79.1 | 80.0 | 6.72 | 7.97 |  |  |  |
| `cpt-8b-full` | CPT, all weights (ablation) [base format] | 76.2 | 76.3 | 79.8 | 6.73 | 8.25 |  |  |  |
| `sft-from-base` | SFT on Base (control) | 76.7 | 79.2 | 79.5 | 7.01 | 8.23 | 18.4 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-base-seed1` | SFT on Base, seed 1 | 76.8 | 79.0 | 79.9 | 7.01 | 8.25 | 17.6 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-cpt` | SFT on CPT (used next) | 76.6 | 81.4 | 79.4 | 6.87 | 8.06 | 17.8 | 6.80 | 1 sample, GPU not recorded |
| `sft-from-cpt-seed1` | SFT on CPT, seed 1 | 77.0 | 79.2 | 80.0 | 6.87 | 8.07 | 16.2 | 6.90 | 1 sample, GPU not recorded |
| `dpo` | DPO, first labels | 76.7 | 81.0 | 79.6 | 6.88 | 8.07 | 17.3 | 6.80 | 1 sample, GPU not recorded |
| `dpo-seed1` | DPO, first labels, seed 1 | 76.5 | 80.7 | 79.5 | 6.89 | 8.07 |  |  |  |
| `dpo-2ep` | DPO, 2 epochs (ablation; failed the merge check) | 76.8 | 80.6 | 80.2 | 6.97 | 8.11 |  |  |  |
| `dpo-strict` | DPO, corrected labels (served, bf16) | 76.7 | 80.9 | 79.6 | 6.88 | 8.06 | 17.9 | 6.90 | H100, serving benchmark |
| `grpo` | GRPO (not served) | 76.7 | 82.0 | 79.8 | 6.93 | 8.09 | 25.5 | 6.70 | 1 sample, GPU not recorded |
| `grpo-seed1` | GRPO, seed 1 | 76.7 | 80.2 | 79.8 | 6.93 | 8.09 |  |  |  |
| `dpo-strict-fp8` | served model in FP8 (passed the quality check) |  |  |  |  |  | 35.2 | 4.70 | H100, serving benchmark |
|  | *floor, base-format rows: two CPT seeds' gap, or SE* | 0.3 | 1.1 | 0.4 | 0.02% | 0.23% |  |  |  |
|  | *floor, chat rows: two SFT seeds' gap, or SE* | 0.4 | 2.2 | 0.6 | 0.08% | 0.07% |  |  |  |

**Blank cells:**
- `mistral-large-3` has no numbers: it is an API reference, run closed-book only.
- `instruct-8b` has no perplexity: it wasn't measured.
- `dpo-strict-fp8`, `dpo-strict-fp8kv`, `dpo-strict-w4a16` have no lm-eval or perplexity run with the settings the other rows used; their quality-check GSM8K (run with a start-of-text token) and perplexity are in [`docs/stage6.md`](docs/stage6.md).
- Latency is measured per deployed checkpoint, not per seed or ablation; FP8-KV's and INT4's bench rows weren't run (Modal spend limit).
<!-- headline-general:end -->

[`docs/results.md`](docs/results.md) also has the earlier, lenient checker's columns, MMLU's four
subject groups, and perplexity on the training text and on 2026 reports.

## Reproduce

[`docs/reproduce.md`](docs/reproduce.md) has every command chain, the pinned versions and the
dataset hashes.

<!-- reproduce-table:start -->
| stage | command | training GPU-h | data (sha256 of its `SHA256SUMS`) |
|---|---|---|---|
| 0: the eval and baselines | `make reproduce-stage0` |  | `eval/tasks/` (committed, reviewed) |
| 1: the corpus | `make reproduce-stage1` |  | `data/processed`: 966d1e0c |
| 2: CPT | `make reproduce-stage2` | 5.40 | the corpus |
| 3: SFT | `make reproduce-stage3` | 1.97 | `data/sft`: 70f47740 |
| 4: DPO | `make reproduce-stage4` | 0.37 | `data/dpo/strict`: 5e3effaf |
| 5: GRPO | `make reproduce-stage5` | 1.08 | `data/grpo`: 4db8f7a6 |
| 6: serving | `make reproduce-stage6` | not totalled | `serve/bench_manifest.json` |
| all training | | 8.82 | |
<!-- reproduce-table:end -->

- **GPU-hours** count training only; evals, sampling and the serving benchmark aren't totalled.
- **Checking the numbers needs no GPU and no API key:**
  - `make reproduce-score` rescores every row from the committed generations and judge verdicts,
    and regenerates every table and figure.
  - `make audit` traces every number in this README and the diagrams to a results or data file,
    a config or the code, or, for a rule set in advance or a ruling on how to read a result, the
    decision log ([`notes/decisions.md`](notes/decisions.md)).

## Who wrote the data, licences, weights

**Who wrote the data:**
- **The test questions:** Mistral Large 3. Each was reviewed against its source passage with keep
  or reject verdicts ([rubric](notes/eval_review_rubric.md)).
- **SFT:**
  - Large 3 wrote the domain questions.
  - Large 3 and Medium 3.5 wrote the answers; Medium 3.5 wrote 415 of the 1,778 domain examples.
  - The refusals use one fixed sentence.
  - 500 general records come from the Tülu 3 mixture, minus its Claude-written subsets. That
    mixture's records were written by GPT-4o, GPT-3.5/4, Mixtral and people, except a math set
    since withdrawn (below).
- **DPO and GRPO:** the models' own answers, labelled and scored by rules.
- **Claude**, through Claude Code, built the tooling and reviewed generated records with keep or
  drop verdicts only. No training set contains text Claude wrote. The hand-written answers in
  `tests/` are fixtures and are never trained on.

**Licences:**

| what | licence | here |
|---|---|---|
| code | Apache-2.0 ([`LICENSE`](LICENSE)) | all of it |
| the corpus: 246 US federal documents | public domain (17 U.S.C. § 105); 27 reproduce third-party material (photo credits, figures), each hand-checked; copyrighted standards excluded | URLs and sha256 of the 251 candidates in [`data/sources.csv`](data/sources.csv); no PDFs |
| FineWeb-Edu (CPT general text) | ODC-By | not redistributed |
| Mistral output (test questions, SFT) | the customer's under Mistral's Commercial Terms (§3.1); labelled model-written (§3.2) | committed |
| Tülu 3 mixture: 500 general records, 50 probe prompts | ODC-BY-1.0; subsets below | row ids only (the withdrawn set's as sha256); `make sft-replay` rebuilds them |
| Ministral 3 8B Base | Apache 2.0 | not redistributed |

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
| Withdrawn math set | 46 | withdrawn after training: its licence restricts models trained on it |
| No Robots | 8 | CC-BY-NC-4.0 (non-commercial) |
<!-- replay-licences:end -->

### Weights

**Not published.**
- **Why:** they were trained on the last two subset rows above.
  - No Robots is non-commercial.
  - The withdrawn math set's licence restricts how models trained on it may be named.
- **Further questions for a release:** FLAN v2's 74 records (no licence on the card) and the GPT-4
  and GPT-4o subsets (OpenAI's terms).
- **The fix:**
  - swap those 54 records: OASST1 for No Robots, and a math source with an unrestricted licence
    and no GSM8K test overlap;
  - retrain SFT, DPO and FP8;
  - rerun the quality check.
- **Until then:** the checkpoints stay on the Modal volume.

### Demo

![The served FP8 model answers a grounded question with its citation, then declines an unanswerable one](docs/demo.gif)

**How it was recorded:** a live run of `serve/demo.py` against the FP8 server, replayed with its
real timing. Both answers match the saved generations.

**How the two items were picked:** by hand.
- **`gr-0038`:** `dpo-strict` answered 93 grounded items correctly with judge-supported citations
  (92 in FP8). Of the three with the shortest passages, this one has the most concrete answer.
- **`adv-0050`:** the third-shortest of the 73 unanswerable items it declined (72 in FP8).

**What they pass,** in both the bf16 and the FP8 run:
- `gr-0038` cites only its gold passage, which is GRPO's strict grounded check.
- `adv-0050` gives the exact refusal sentence.
- The closed-book strict checker applies to neither.
