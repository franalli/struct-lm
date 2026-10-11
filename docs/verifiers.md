# Verifiers and judges

Every number in this repo was decided by one of three kinds of evaluator, in Chip Huyen's terms
(*AI Engineering*, ch. 3):
- **exact evaluation**, a rule that compares an answer with a reference;
- **similarity** to a reference, which is looser;
- **AI as a judge**, a model grading against a rubric.

Each showed its own flaw here. What's left of the training signal at the end of the chain is
exact rules only. Two habits keep them honest: every rule gets an adversarial fixture suite before
it becomes a reward, and the answers it accepts are checked by hand before its scores are believed.

## Where each kind runs

| kind | evaluator | used for | how it goes wrong |
|---|---|---|---|
| exact | `scorers.qa_strict`: the whole gold after normalisation, one candidate, units compared | the strict closed-book score from Stage 5, the GRPO reward, `dpo-strict`'s labels | strict where an alias list is missing (one gold string per item); its tolerance rules are themselves code that can leak |
| exact | `cite_valid`, the abstain sentence, the format check | grounded citations, abstention, the GRPO format reward | none found; the sentence is fixed text |
| similarity | `scorers.qa_correct` (containment and numeric tolerance) | the closed-book columns from Stage 0 to 4 | accepts a range for a point value, the first number of a fraction and a child section |
| similarity | `sft_judge.same_fact` (substring) | Stage 4's closed-book pair labels | accepts any fragment of the gold, and judges multi-number golds on their first number |
| AI judge | Mistral Large 3, pointwise binary prompts (`eval/judge.py`) | `grounded_acc`, `cite_supported`, `vocab_recall`, the adversarial grade | not benchmarked; the Stage 0 hand-check of 40 verdicts found it errs strict |
| AI judge | Mistral Large 3, listwise rubric (`sft_judge.judge_list`) | benchmarked for Stage 4's pair labels; labels nothing | misses most of the defects a careful reader found (below) |

## The listwise judge, measured (Stage 4)

Before any preference pair was labelled, the listwise judge was measured on 166 answers whose
defects a reader with the full source passage had already found or cleared. Each answer was graded
as it would be in the pool: listed with three of the student's own answers to the same prompt.

| format | defects caught (recall) | clean answers passed | AUROC |
|---|---|---|---|
| grounded (60 defects, 60 clean) | 0.28 | 0.92 | 0.66 |
| definition (23 defects, 23 clean) | 0.09 | 0.87 | 0.54 |

- **Two other prompt variants** (quote the support first, show the whole page) caught no more
  defects.
- **What it misses is the defect a reader finds:** a dropped caveat, an OCR gap filled in, one
  example generalised into a rule. In several misses the judge's own reasoning names the flaw and
  the score is still 5.
- **Three biases, measured:**
  - the mean score falls about 0.5 point from the first listed answer to the fourth;
  - the student's answers score 0.8 (grounded) and 2.1 (definition) points below Large 3's own;
  - asked the same pair in both orders, it agreed with itself on 43-54% of prompts.
- **Why it labels no pairs:** at roughly 10% defect prevalence and this recall, an answer the judge
  flags is a real defect about a quarter of the time. The rejected side of judged pairs would be
  mostly good answers.

**The decision:** no judge in pair-building. Closed-book pairs are labelled by the verifier,
abstain pairs by the decline rule, grounded pairs by rules (`cite_valid`, cites the gold passage,
no false abstain). Definition pairs aren't built at all. Under the earlier rules, which capped
closed-book pairs at half the set, the same samples gave 114 pairs; the verifier-only set has 506
(`notes/decisions.md`, 2026-10-08).

**What it leaves on the eval:** `grounded_acc` and `cite_supported` use the same model with a
different, pointwise prompt that names the gold passage. This benchmark didn't test that prompt,
so the README reports `cite_supported` and draws no conclusion from it alone.

## Three holes in one day (Stage 5)

The closed-book verifier was about to become an RL reward, and an optimiser finds whatever slack a
proxy has (the RLHF Book's ch. 14 over-optimisation). The holes were found in this order:

1. **`same_fact` accepted fragments, found by auditing the answers it accepted before GRPO.**
   - "Section" was accepted for "Section 17.8.2"; 116 Stage 4 pool samples were accepted only
     that way.
   - "class 8" was accepted for "8 x 19": multi-number golds were judged on their first number.
   - The strict checker replaced it. Re-scored with it, 79 of the 458 closed-book chosen labels in
     the as-run DPO set were wrong, so DPO was retrained on strict labels (`dpo-strict`).
   - Fixtures: `("Section", "Section 17.8.2")` must fail; `("6.10.10", "Article 6.10.10")` must
     pass (`tests/test_qa_strict.py`).
2. **The strict checker let comma lists through, found by checking its code against its own
   one-answer rule, mid-run.**
   - "cripple wall, shear wall" earned full reward for the gold "cripple wall", in both orders, and
     so did slash and "and" lists.
   - The rollouts showed it wasn't learned: answer lines with several candidates stayed at the
     start's base rate (2.8% against 2.77%) and fell with step.
   - Fixtures: both orders of each list must fail; "ASCE/SEI 7-22 §12.11.2", "w/c" and "1,000-year
     flood" must pass. The suite went from 69 to 88 cases.
3. **A 2% relative tolerance accepted years, found by the hack audit of the top-reward rollouts.**
   - One task in the 622 has a bare-year gold. Its prompt, verbatim:

     > When did AASHTO initially include language in its standards that permits cold bending to
     > create camber in rolled beams used for highway bridges?
     > Reply with just the value or name, no explanation.

   - The gold is `2010` (`data/grpo/tasks.jsonl`, `fhwa-hif23003:p238:c0:f3:closed_book:2`), and
     2% of 2010 is 40 years. At step 33 `grpo` answered `1996` and got a reward of 1.0
     (`results/runs/grpo/rollouts.jsonl`).
   - After step 30, 10 of its 16 rollouts on that task were rewarded for a wrong year (1988, 1989,
     1990, 1994, 1996, 2004, 2014, 2020). This hole was learned, on one task.
   - Fixed: a bare-year gold needs the exact year. Fixture: `("1996", "2010", "value")` must fail.

The first two holes were closed before the policy could learn them; the third was learned on one
task of 622. No eval item has a year gold, so no eval row moved.

## The rule the repo ended with

1. **A verifier gets an adversarial fixture suite before it becomes a reward:** must-pass cases
   (aliases, normalisation, right answers with an added noun) and must-fail cases (fragments,
   lists in both orders, ranges, near-miss values), in `tests/`.
2. **The answers it accepts are audited before its scores are believed:** the samples it rewards,
   the top-reward rollouts first, are checked by hand, with verdicts only.
3. **A changed rule re-scores everything it graded,** and every changed verdict is checked against
   the gold by hand. The strict re-score moved the eval rows 0–1.6 points with no ordering change,
   and moved 79 DPO labels.
4. **A judge is measured on defects a careful reader found before it labels anything.** The
   listwise judge above was measured that way, and it labels no training data.

**At larger scale:** open-ended tasks have no exact rule, so the judge comes back. The work then is
calibrating it against subject-matter experts' labels, as here, before it labels anything.
