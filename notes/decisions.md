# Decisions

Every non-obvious choice, dated, with the alternative considered and why it lost.
Newest at the bottom. If a later result reverses a decision, add a new entry. Don't edit the old one.

Template:

```
## YYYY-MM-DD: <decision>
**Context:** what forced the choice
**Options:** A vs B (vs C)
**Chose:** X, because ...
**Evidence:** link to results/…, W&B run, or table row
**Revisit if:** ...
```

---

## 2026-09-26: Repo structure and stage order
**Context:** Project kickoff.
**Options:** SFT-only · SFT→DPO · CPT→SFT→DPO→GRPO
**Chose:** Full CPT→SFT→DPO→GRPO, each stage evaluated on the same KPI + regression suite,
so each one's marginal contribution is measurable (results/table.md).
**Revisit if:** a stage adds < noise on KPIs; drop it and say so in the write-up.

## 2026-09-26: Base model
**Context:** Need a base (not instruct) model small enough for LoRA on a single GPU.
**Chose:** Qwen/Qwen3-4B-Base (placeholder; confirm after tokenizer_coverage.py).
**Revisit if:** fertility on domain terms is much worse than alternatives.

## 2026-09-26: LoRA at every stage, merge between stages
**Context:** Compute budget; clean per-stage lineage.
**Chose:** LoRA per stage → train/merge.py → next stage's `init_from`. Adapters stay small
and comparable; merging avoids stacking adapters.
**Revisit if:** CPT val loss plateaus early (LoRA capacity limit), then try higher r or full FT.

## 2026-09-27: Seed corpus: US federal structural-engineering publications
**Context:** Need a domain corpus that can be redistributed with the write-up.
**Options:** textbooks / code documents (copyrighted) vs US federal works (public domain, 17 USC 105)
**Chose:** 20 federal PDFs (USACE EMs, NIST NEHRP briefs, FHWA NHI manuals, NASA standards,
FEMA), listed in `data/scripts/sources.csv`, roughly 3-4M tokens. Mix by role: USACE/NASA
for tables and numeric QA, NIST for vocabulary and procedures, FHWA for bulk tokens and worked examples.
**Rule:** NIST GCR reports are contractor-prepared. The current ones carry no copyright
notice, but check page 2 of any GCR added later and drop it if a notice appears.
**Next (Stage 1 expansion):** USACE Engineer Manuals index (100+ EMs), FHWA steel bridge
page (~20 more manuals), FEMA Building Science library (P-695, P-58-1, P-751, P-1050).

## 2026-09-27: Exclude ASCE 7 and the AISC Steel Construction Manual
**Context:** Both are the core references for building load combinations and steel design.
**Options:** add them · public-domain substitutes
**Chose:** substitutes. ASCE 7 and the AISC Manual are copyrighted, commercially sold works
(adoption into building codes doesn't put the text in the public domain), which would break
the corpus's redistributable, public-domain premise and risk verbatim reproduction of tables.
The model still learns clause references, since the federal documents cite ASCE 7 / AISC throughout.
**Added instead:** FEMA P-1050-1 (2015 NEHRP Provisions, amends ASCE 7-10), FEMA P-1051
(2015 design examples; replaces P-751, which has no working FEMA PDF), FEMA P-2082-1
(2020 NEHRP Provisions, basis of ASCE 7-22 seismic chapters); FHWA NHI-15-058, HIF-19-088,
HIF-13-020, IF-12-027 for steel.
**Not added:** FHWA Steel Bridge Design Handbook (HIF-16-002). FHWA has removed the PDFs
(404) and the handbook is now maintained by NSBA/AISC under AISC terms.

## 2026-09-27: Task generator and judge: Mistral Medium 3.5 (pinned), not Large
**Context:** The Mistral plan doesn't include `mistral-large-latest` (403 tier_not_allowed).
**Options:** upgrade plan · switch to the strongest available model
**Chose:** `mistral-medium-2604` for both `make_tasks.GEN_MODEL` and `judge.JUDGE_MODEL`,
pinned to the dated id so the judge can't drift between runs (the judge cache keys on it).
Rejected `ministral-8b`: no stronger than the 7B model under evaluation.
**Revisit if:** the plan gains Large. Switching models means regenerating the tasks and
re-judging every row in results/table.md.

## 2026-09-27: Removed Steel Bridge Design Handbook (2025 NSBA edition)
**Context:** 26 PDFs (b901-b920 chapters, b951-b956 design examples) were added to data/raw/.
**Finding:** every file carries "© AISC 2025 … All rights reserved … must not be reproduced
in any form without the written permission of the publisher."
**Chose:** deleted them; never added to sources.csv, so no chunks or eval items came from them.
Same reasoning as the ASCE 7 / AISC Manual exclusion above.
**Revisit if:** AISC grants written permission.

## 2026-09-27: Added Steel Bridge Design Handbook, 2015 FHWA edition (HIF-16-002)
**Context:** Public-domain substitute for the removed 2025 NSBA/AISC edition.
**Chose:** all 25 modules of FHWA-HIF-16-002 (19 volumes + Design Examples 1, 2A, 2B, 3, 4, 5),
from the USDOT National Transportation Library (ROSA P), which archives the FHWA originals.
Prepared by HDR, Inc. for FHWA; based on AASHTO LRFD 7th ed. (the NSBA edition is 10th ed.,
so specification article numbers and some values will differ).
**Rule:** contractor-prepared, like the NIST GCRs. Check each PDF's notice page after
download and drop any volume that carries a copyright notice.

## 2026-09-27: Cross-document dedup in extract.py
**Context:** Overlap check after adding the handbook: 131 exact-duplicate chunks (FHWA
boilerplate) and ~430 near-verbatim ones, almost all between successive editions
(FEMA P-1050-1 vs P-2082-1, 2015 vs 2020 NEHRP examples, NHI-04-041 vs NHI-15-058).
**Why it matters:** duplicated passages get extra CPT weight, and make_tasks.py reserves only
exact chunk ids, so an eval item's near-copy in another document would leak into training.
**Chose:** extract.py drops exact duplicates and chunks with >=90% of their word 8-grams already
seen in an EARLIER document (sources.csv order; first copy wins). 322 chunks dropped
(20,721 -> 20,399; 6.83M -> 6.76M tokens). Partial overlap (50-90%, e.g. design examples sharing
method text with different numbers) is kept. Residual: 72 chunks >=90% shared where the later
edition splits the text across chunk boundaries.
**No copyright notices** at document level in any of the 52 PDFs; in-body hits are credits for
third-party figures/photos reproduced with permission (image blocks aren't extracted).

## 2026-09-27: Base model -> Ministral 3 8B Base (2512), Tekken tokenizer everywhere
**Chose:** `mistralai/Ministral-3-8B-Base-2512`. Chunking (extract.py) tokenizes with its Tekken
tokenizer via mistral-common (AutoTokenizer fallback for other repos); vLLM runs with
tokenizer_mode="mistral" in run_eval.py, serve_vllm.sh and run_lm_eval.sh, so chunk budgets, eval
and serving all count tokens the way the model does. Tekken uses ~9% fewer tokens than the
Mistral-7B tokenizer on this corpus. Ministral 3 is multimodal: limit_mm_per_prompt={"image": 0}
keeps the vision encoder from reserving memory (vLLM ignores it for text-only models).
Chat models use llm.chat() so the checkpoint's own template is applied, not HF's.
**Deps:** transformers>=5 and vllm>=0.29 conflict with every llmcompressor release (compressed-tensors
pins), so AWQ moved to its own `quantize` extra, declared conflicting with `serve`/`eval` in
[tool.uv]; requires-python capped <3.14 (vLLM/llmcompressor). Modal image: hf_transfer replaced
by HF_XET_HIGH_PERFORMANCE (huggingface_hub 1.x dropped hf_transfer).
**Consequence:** chunks.jsonl must be re-extracted with the new tokenizer, which moves chunk
boundaries and ids, so eval/tasks must be regenerated afterwards.

## 2026-09-27: Training loads Ministral 3 as a VLM; LoRA on text layers only
**Context:** Ministral 3 is Mistral3ForConditionalGeneration (8.38B: text + 0.43B Pixtral vision
encoder + projector). transformers maps it to AutoModelForImageTextToText, not AutoModelForCausalLM.
**Chose:** keep the full architecture end to end (what vLLM serves). train/common.py picks the Auto
class from the config; merge.py loads base + PeftModel the same way and copies tekken.json and
processor_config.json into each merged stage. LoRA target_modules is a regex anchored on
`language_model` (single-quoted in YAML: `\.` is an invalid escape in double quotes), because the
vision tower also has q/k/v/o and MLP projections that bare names or all-linear would hit.
Verified on a meta-device model: 238 LoRA layers (34 x 7), 0 in vision; trainable 178M (CPT r=64),
89M (SFT r=32), 45M (DPO/GRPO r=16).

## 2026-09-27: Chunk token counting is join-aware
Tekken often tokenizes the space before a number as its own token, so summing standalone piece
counts overshot the 512 cap by one token per join (up to 589 on numeric tables). extract.py now
counts each piece with its leading space. Corpus: 19,337 chunks, 6.22M Tekken tokens, max 512.

## 2026-09-27: Closed-book QA filters in make_tasks.py
**Context:** hand-check of 30 domain_qa items: 15 bad, mostly context-bound ("Girder G2",
"Figure 5-1", "at that time", a document's own equation numbers). The verifier saw the passage,
so it resolved the ambiguity from context.
**Chose:** closed-book rules in the generation prompt; regex pre-filter for context-bound phrasing;
a blind check where the verifier sees ONLY the question; then the existing answer check.
n-qa-chunks 150 -> 250 to offset the stricter filtering. The blind check errs strict
(8/9 on known items; the miss rejected a good question).

## 2026-09-27: transformers pinned <5.17
**Context:** lm-eval on Modal died at vLLM model inspection for both Ministral 3 checkpoints:
`cannot import name 'PixtralRotaryEmbedding'`. transformers 5.17.0 renamed it to
`PixtralVisionRotaryEmbedding`; vLLM 0.29.0 and 0.30.0 `pixtral.py` both still import the old name.
**Chose:** `transformers>=5.10.4,<5.17` (floor is vLLM's own requirement) in pyproject.toml and the
Modal image. Lock now resolves 5.16.1. Lift the cap once a vLLM release imports the new name.
**Update (same day):** second hand-check of 30 on the regenerated set: 25 good, 2 weak, 3 bad
(was 15/30 bad). Added post-filters for trivia: notation/unit/abbreviation questions, answers
restated in the question (all-token comparison; an earlier word-only version wrongly dropped
"Which AASHTO LRFD article...? -> Article X" items and forced a regeneration), edition/citation-year
answers, and answers that are the passage's own document. Final: domain_qa 168, grounded 80,
vocab 400, adversarial 53, 844 reserved chunk ids.

## 2026-09-27: Generator and judge -> Mistral Large 3 (mistral-large-2512), pinned
**Context:** the Pro plan unlocked Large (the original design). Remaining QA failures after the
closed-book filters are judgment calls (general knowledge, underspecified), where the bigger model helps.
**Chose:** `mistral-large-2512` for make_tasks.GEN_MODEL and judge.JUDGE_MODEL (kept identical).
Rejected Magistral (only Small/Medium exist): a reasoning model adds variance to a temperature-0
0/1 judge for little gain on single-passage checks. Nothing had been judged yet, so no results are
invalidated. Also added deterministic QA filters: list questions ("name one..."), bare Greek-symbol
answers, and answer numbers that don't occur in the passage (caught a formula hallucinated from a
PyMuPDF-garbled equation).

## 2026-09-27: QA iteration rounds (Mistral Large 3) and hand-check reject list
Hand-checks of 30 fresh items per round: round 1 12 bad, round 2 10, round 3 7, round 4 5.
What worked (deterministic, verified on known items before applying): trivia/abbreviation/meta
regexes, a manual's own section numbers, bare-symbol answers, answer numbers not in the passage,
reference-list source chunks (reference-list shapes only; inline "(2014)" citations don't count),
single-number answers typed numeric (so "95%" matches "95 percent"), hyphens as spaces in
scorers.normalize. What didn't: a closed-book "Large already knows it" filter (a difficulty filter,
dropped good and bad alike); verifier clause (d) against worked-example values (all 5 known cases
survived it). Remaining failures are judgment calls, so reviewed-bad items go into
eval/tasks/qa_rejects.jsonl (question substring + reason), applied on every run.
Infra: LLM call cache (eval/.cache), --only qa, --workers (4; 8 rate-limits Large 3), 120 s request
timeout (a dead socket hung one run for ~7.5 h), stall-detecting monitors.
Final: domain_qa 183, grounded 80, vocab 400, adversarial 52.
**Full review (same day):** every domain_qa and few-shot item (186) hand-checked against its source
passage. 43 rejected (worked-example/case values 7, general knowledge 11, wrong or mismatched 8,
symbols/equation or chapter numbers 9, several valid answers 4, unscoreable/unverifiable 4), incl.
2 factually wrong reference answers (Category B threshold given as 12 ksi, table says 16; D/t "8.8"
is the coefficient of E/Fy). All rejects are in eval/tasks/qa_rejects.jsonl with reasons.
Final: domain_qa 140 + 3 few-shot, all human-reviewed.

## 2026-09-27: Full hand-review of grounded, adversarial and vocab; pipeline fixes
All 561 eval items across the five task files are now human-reviewed (0 unreviewed, 0 rejected
items present; rebuild byte-identical). Reject list: eval/tasks/rejects.jsonl (146 entries, task + reason).
**Adversarial (50):** first generation was 70% "What is the maximum allowable ..." vs 2% in grounded,
a phrasing shortcut to abstention. Generator now rotates 6 question types matching grounded and bans
giveaway words (now 4% vs 5%); n_adversarial 60 -> 110 for yield. New filter drops items whose passages
explicitly rebut the premise ("There are no criteria specified for ...": then "none" is a correct,
grounded answer). Non-prose seed passages dropped.
**Grounded (65):** 15 of 80 rejected; 11 had a non-prose gold passage (reference lists, report
documentation pages, flowcharts, author bios, glossaries, scrambled text). is_nonprose() post-filter
catches all 11 with no false positives; 4 content rejects (misframed, answer only in a non-gold passage).
**Vocab (303):** 97 of 400 removed: 15 terms absent from their source passage (term_in_passage), 8
mechanical duplicates (term_key: plural/hyphen/parenthetical), 74 manual (semantic duplicates,
passage-specific definitions, compositional phrases, 10 wrong definitions, generic terms).
**Infra fixes found on the way:** per-task RNG streams (QA rejects had been resampling every other
task); all new filters applied post-sampling/post-cap so they can only remove items; LLM output
parsing tolerates non-object list entries.
Final: domain_qa 140 + 3 few-shot, grounded 65, adversarial 50, vocab 303.

## 2026-09-27: Grounded and adversarial enlarged (65 -> 131, 50 -> 87), all reviewed
**Why:** at n=65/50, a rate's sampling noise is about +/-6 points, too coarse to compare stages.
**How:** a supplementary phase sampled AFTER all tasks from its own RNG streams (ids from 501), so no
earlier sample shifts: raising n_grounded directly would change vocab's pool (it excludes grounded's
chunks) and silently replace reviewed vocab items. Verified: extras=0 reproduces the reviewed set
byte-for-byte; with extras, every original item is unchanged. Defaults: --n-grounded-extra 80,
--n-adversarial-extra 110.
**Review:** grounded 13/79 rejected (7 not answerable from gold, mostly "purpose/rationale" questions
whose passage only applies a step; 3 non-prose golds; 1 meta; 2 reference lists), adversarial 8/45
(4 answerable by inference or partly answerable, 1 near-duplicate, 2 front-matter seeds, 1 derivable).
NONPROSE extended with front matter (forewords, report cover data, lists of tables, TOC dot leaders,
mission statements) and career-summary bios; both verified to remove only their targets. Reference
lists without citation-shaped entries (gr-0523, gr-0577) can't be caught by density without dropping
good items, so they stay manual rejects.
Final: domain_qa 140 + 3 few-shot, grounded 131, adversarial 87, vocab 303 = 664 items, all reviewed.
