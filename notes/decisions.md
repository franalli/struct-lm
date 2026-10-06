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
Rejected `ministral-8b`: a generator/judge should be stronger than the models it evaluates.
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
and serving all count tokens the way the model does.
Ministral 3 is multimodal: limit_mm_per_prompt={"image": 0}
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

## 2026-09-27: Grounded prompt: format line instead of worked example
**Why:** the worked example (question, passages, cited answer) made Ministral 3 8B Instruct abstain on
130/131 answerable grounded items: it treated the example's passages as the ones to use. Confirmed on
the Mistral API (ministral-8b-2512) before spending GPU time.
**Change:** prompts.GROUNDED_FORMAT, a one-line format example explicitly marked as unrelated to the
passages. API spot check: 12/12 grounded answered, 10/10 adversarial declined. Modal generations
redone (--generate-only). Instruct now: 9/131 grounded abstain, 121 cited; 86/87 adversarial abstain.
Base: 0/131 abstain but only 25 cite: it answers in prose and drops the [id] format.

## 2026-09-27: grounded_acc split from cite_supported; uncited answers scored by rule
**Why:** one prompt for both models (no per-model prompt tuning), so Base's lost citation format
would read as "wrong" under the strict metric. Splitting separates "doesn't know" from "doesn't cite",
which is the thing SFT is expected to fix.
**grounded_acc** (new judge rubric): correct per the gold passage and no unsupported claims,
citations ignored; declining scores 0. Same wording as the strict rubric minus the citation clause,
so the gap between the two is citation behaviour only.
**cite_supported fix:** a trial on 8 Base items found the strict judge passing uncited answers (3/5),
once quoting a citation the answer didn't contain. Now answers citing no provided passage id score 0
without a judge call, and the rubric lists the parsed cited ids instead of leaving the judge to find
them. table.md header gained the grounded_acc column; append_table refuses to append under a
mismatched header.

## 2026-09-27: Adversarial judge grades the answer only (passages removed from its prompt)
**Why:** Base's first scored run gave halluc_rate 0.058 although it used the abstain phrase on 2/87.
The judge, shown the passages and told they lack the answer, graded the passages instead of the
answer: "the maximum height difference is 10 feet", a bare section number and a citation-only
output all scored as refusals ("the passages do not provide any information ..."). On 14 Base
items the old rubric was wrong on 11; the new one right on all 14.
**Change:** adversarial_rubric(question, answer): classify the wording alone. Declining scores 1,
even when related information is mentioned; any value, clause, procedure, comparison or
explanation scores 0, as do empty and citation-only outputs. Passages aren't needed: by
construction they don't contain the answer, so any non-refusal is a hallucination.
**Also:** judge retries 5 -> 7 (backoff ~63 s): 8 Base verdicts had failed on 429 rate limits.
**Lesson (three judge fixes today):** give the judge only what the verdict depends on, and decide
by rule anything a rule can (citation presence, exact abstain phrase) before the judge sees it.

## 2026-09-27: Judge hand-check (40 verdicts); empty-answer guard; chat one-line generation
**Hand-check** (stratified, half pass / half fail per rubric):
- grounded_acc: 10/16 clearly right, 2 wrong (both false negatives: an extraction-scrambled
  subscript read as a mismatch; a claimed omission that wasn't one), 4 debatable (padding, a mild
  extrapolation, a nitpick, and gr-0060 whose hoop spacing is only in a figure: an item flaw).
- cite_supported: 10/12 right, 1 wrong (same subscript), 1 right verdict with a wrong reason.
- The judge errs strict: its misses are false negatives, so accuracy metrics are slight lower bounds.
**Found by the check, fixed by rule rather than prompt:**
- Instruct vocab: 273/303 outputs were only "**Term: x**" (the chat model opens with a header and the
  "\n" stop cut it there), and the judge passed 193 of them, quoting definitions that weren't in the
  output. Fix: prompts.CHAT_GEN drops the stop for chat on domain_qa/vocab; run_eval keeps the first
  line with content (scorers.answer_line; raw text kept as raw_output). API check 6/6 extracted.
  Instruct regenerated (--generate-only). Base generation unchanged.
- Empty or citation-only answers (Base: 15 adversarial, 2 grounded) are scored 0 by rule
  (scorers.substance): the judge had called 11 of the 15 refusals. Base halluc_rate 0.793 -> 0.920.

## 2026-09-27: Chat-mode lm-eval is invalid with tokenizer_mode=mistral; Instruct row uses chat-off
lm-eval 0.4.13 applies the chat template with tokenize=False and re-encodes the string with
add_special_tokens=False (vllm_causallms.py). mistral-common never parses control tokens from text,
so "<s>[INST]...[/INST]" reached the model as 18 ordinary tokens (verified locally). The CHAT=1 run
(12:14 UTC) is set aside in results/lm_eval/_invalid/. The Instruct row uses the chat-off run
(same format as Base). KPI eval is unaffected: vLLM llm.chat() tokenizes via mistral-common.
Possible fix, not applied: TOKENIZER_MODE=hf with CHAT=1 (tokenizer.json matches Tekken on 300/300
corpus chunks and parses control tokens), but the repo's chat_template.jinja prepends Mistral's
default system prompt, which the KPI chat path doesn't use.

## 2026-09-27: Chat format: always off for lm-eval, always on for the KPI eval (user decision)
lm-eval (MMLU/GSM8K/HellaSwag) never uses a chat template, for any checkpoint: every row stays
comparable, and lm-eval's chat path breaks Mistral control tokens (entry above). The KPI eval always
runs chat checkpoints (Instruct, SFT/DPO/GRPO) with --chat; base models without.
Enforced in code: run_lm_eval.sh exits on CHAT=1; modal_app's lm_eval takes no chat argument;
modal_app and run_eval refuse an Instruct checkpoint without --chat; merge_lm_eval skips
chat-template results files (e.g. an old run pulled back off the volume).

## 2026-09-27: Stage 1 corpus card: 246 documents, 20.6M Tekken tokens
**What:** the CPT corpus. `make data` runs download, extract, filter, dedup, pii, split, replay,
tokenizer_coverage and stats in order. Every number below comes from
`data/processed/stats.json` via `stats.py`, and `data/sources.csv` records each file's URL,
sha256, pages and final tokens. Train is 19.4M tokens (234 docs), val 1.2M (12 docs, 5.9%),
replay 1.9M FineWeb-Edu tokens. The processed files are not committed; they rebuild from
sources.csv plus the scripts.

**Sources (Stage 1 expansion, 52 -> 251 rows):**
- USACE Engineer Manuals, every EM 1110-2 (civil works) and EM 1110-1 (general engineering) on the
  live index. EM 1110-3 (22 mobilization manuals from 1984: pavements, water supply) was skipped.
  Akamai returns 403 to every script, so the index pages and PDFs were fetched in Chrome (DevTools
  MCP). download.py can't refetch them, and the same holds for fema.gov and ROSA P (the seed's
  Steel Bridge Design Handbook).
- FHWA steel and concrete bridge index pages (crawl_index.py).
- NIST NEHRP briefs 5-12 (Brief 14 is not published) and 15 contractor reports from the GCR 917
  series. 13 are earthquake engineering by ATC and the NEHRP Consultants Joint Venture: nonlinear
  analysis, soil-structure interaction, ground motions, nonstructural components, high-strength
  reinforcement. The other two cover lifeline performance and a resilience-standards symposium.
- FEMA: P-58-1/-2, P-695, P-751, P-1050-2, P-2006, P-795, NEHRP Design Examples vols. 2-3, and 35
  technical publications from the Building Science earthquake library (incl. P-58-4 to -7, P-2018,
  P-2091, P-2335 (listed as P-2355 until 2026-10-05), E-74; brochures, checklists, posters and
  forms skipped).
- Four FEMA documents aren't fetchable from fema.gov, so they come from other hosts, all
  scriptable:
  - P-695: NIST's NEHRP clearinghouse (nehrpsearch.nist.gov); FEMA's copy returns 404.
  - P-751: WBDG's federal facility criteria library (its public S3 file).
  - P-58-4 and P-58-5: ATC, which prepared them for FEMA. FEMA serves both truncated (7.2 of 8.0 MB
    and 3.5 of 13.3 MB against the length their linearisation headers declare, 0 readable pages);
    ATC's files match those lengths exactly (122 and 196 pages).
- NASA-STD-5002B and 5019A, so NASA has a non-eval document to hold out.
- The spec's 20-50M target needed more than the three named index pages. The EM 1110-2 series
  alone yielded 8.5M clean tokens, not 15-25M: many manuals are scanned, table-heavy, or PDF
  portfolios.
**Excluded for copyright** (`extract.py` flags ©/"copyright"/"all rights reserved" in the first 5
pages; all 31 were hand-checked, 27 in the final corpus plus the 4 excluded here):
- FHWA-HIF-19-102: © Lehigh University, all rights reserved.
- FEMA's 2025 seismic evaluation guidance: reproduces ICC code text "proprietary to and
  copyrighted by" ICC, the same reason as the ASCE 7/AISC exclusion.
- NIST GCR 15-917-35/36: © Fire Protection Research Foundation, and off-domain (cooking fires).
- Photo credits, "figures reproduced with permission" notes, and BSSC's "be alert to patent and
  copyright concerns" boilerplate were kept.
**No size floor (code review):** the spec's 200 KB download floor was dropped because it rejected
two real manuals, EM 1110-2-2102 (8k words) and EM 1110-1-4011 (13k words). A file is now rejected
only when it isn't a PDF or has no readable pages; short documents are left to the 2,000-word
minimum, which drops EM 1110-2-1304 (1,198 words). All 251 sources are accepted.

**Sources and pages** (extract.py; pages under 200 characters dropped, no OCR)

| publisher | docs | pages | kept | dropped: image-only | dropped: blank | tokens extracted | tokens final |
|---|---|---|---|---|---|---|---|
| FEMA | 49 | 15,713 | 14,220 | 890 | 603 | 8,492,849 | 4,781,434 |
| FHWA | 65 | 12,587 | 11,577 | 750 | 260 | 5,834,309 | 3,838,621 |
| NASA | 4 | 319 | 310 | 7 | 2 | 152,143 | 114,123 |
| NIST | 28 | 4,595 | 4,432 | 25 | 138 | 2,648,730 | 1,724,278 |
| USACE | 105 | 27,265 | 24,362 | 2,409 | 494 | 14,200,387 | 10,177,288 |
| **total** | 251 | 60,479 | 54,901 | 4,081 | 1,497 | 31,328,418 | 20,635,744 |

251 documents in sources.csv, 251 accepted (3.17 GB); rejected: none.
Header/footer lines removed: 123,528. Likely scanned (>50% of pages dropped): fhwa-sbdh-v03, usace-em-1110-2-1424, fema-nehrp-examples-v3, fema-p-2192, fema-p-1100-2c.

**Token funnel** (Tekken tokens)

| step | docs | tokens | removed |
|---|---|---|---|
| extracted | 251 | 31,328,418 |  |
| quality filter | 248 | 21,594,218 | 31.1% |
| exact dedup | 248 | 21,594,218 | 0.0% |
| near dedup (+ stubs under 2,000 words) | 246 | 20,637,997 | 4.4% |
| train | 234 | 19,421,232 |  |
| val | 12 | 1,214,512 | 5.9% of tokens |
| replay (FineWeb-Edu) | 1,780 | 1,944,824 | 10% of train |

**Quality filter** (filter.py; first failing rule counted)

| rule | paragraphs dropped | tokens dropped |
|---|---|---|
| alpha_ratio | 204,253 | 5,818,961 |
| word_length | 42,419 | 380,844 |
| short_lines | 37,346 | 1,323,281 |
| reference_list | 6,856 | 405,233 |
| symbol_ratio | 3,373 | 37,700 |
| numbered_lines | 2,875 | 170,349 |
| **paragraphs in / out** | 754,809 / 456,480 |  |

Documents dropped: fhwa-hif16010 (dictionary, 47,164 words, dictionary 0.649), usace-em-1110-2-1304-2021 (min_words, 1,198 words, dictionary 0.926), fema-nehrp-examples-v3 (min_words, 540 words, dictionary 0.937). Dictionary ratio of kept documents: min 0.743, median 0.925.

**Deduplication** (dedup.py)

Exact duplicate documents: none. Near-duplicate paragraphs: 34,827 of 456,480; tokens 21,594,218 -> 20,637,997 (4.4%). Dropped as under 2,000 words once duplicates were removed: fema-p-2192 (1,906 words), fema-p-1024-ra1 (44 words).

Most-duplicated across documents:

| docs | copies removed | paragraph |
|---|---|---|
| 59 | 74 | DEPARTMENT OF THE ARMY U.S. Army Corps of Engineers |
| 57 | 56 | The Federal Highway Administration (FHWA) provides high-quality information to serve Gover |
| 39 | 38 | Approved for public release; distribution is |
| 30 | 29 | Any opinions, findings, conclusions, or recommendations expressed in this publication do n |
| 28 | 27 | 8. Performing Organization Report No. |
| 25 | 24 | U.S. Department of Commerce |
| 25 | 24 | Notice This document is disseminated under the sponsorship of the U.S. Department of Trans |
| 24 | 23 | FOREWORD This handbook covers a full range of topics and design examples intended to provi |
| 23 | 22 | Form DOT F 1700.7 (8-72) Reproduction of completed pages authorized |
| 22 | 21 | ASCE American Society of Civil Engineers |

Most-duplicated by copies:

| docs | copies removed | paragraph |
|---|---|---|
| 6 | 1,212 | This publication is available free of charge from: https://doi.org/10.6028/NIST.GCR.17-917 |
| 1 | 484 | This publication is available free of charge from: https://doi.org/10.6028/NIST.GCR.22-917 |
| 1 | 299 | This publication is available free of charge from: https://doi.org/10.6028/NIST.GCR.18-917 |
| 1 | 239 | FEMA P-1051, NEHRP Recommended Provisions: Design Examples |
| 1 | 203 | FEMA P-1051, NEHRP Recommended Seismic Provisions: Design Examples |
| 1 | 202 | This publication is available free of charge from: https://doi.org/10.6028/NIST.GCR.22-917 |
| 1 | 164 | Chapter 7: Materials, Material Deficiencies, and Inspection Methods |
| 3 | 144 | 1 Aug 08 (Change 2) |
| 1 | 136 | Appendix H: Wood Archetype Collapse Results |
| 2 | 130 | Strength Property ACMR at DR |

**PII** (pii.py): EMAIL 58, PHONE 180, ID 0

**Tokenizer fit** (tokenizer_coverage.py, full report in `data/processed/tokenizer_coverage.md`; tokens per whitespace word)

|  | `mistralai/Ministral-3-8B-Base-2512` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
|---|---|---|
| corpus tokens | 20,635,744 | 20,635,744 |
| fertility: domain corpus | 1.44 | 1.44 |
| fertility: FineWeb-Edu | 1.338 | 1.338 |
| corpus / FineWeb-Edu | 1.08x | 1.08x |
| fertility: vocab_eval terms (303) | 1.433 | 1.433 |
| fertility: tfidf_top terms (500) | 1.116 | 1.116 |
| fertility: probes terms (18) | 3.207 | 3.207 |
| term words split into 4+ tokens | 34 / 843 | 34 / 843 |

Base and Instruct encode identically (corpus sample + every term word): **True**.

| designation | tokens |
|---|---|
| `ASCE 7-22` | 7 |
| `ASCE 7-16` | 7 |
| `EM 1110-2-2104` | 13 |
| `NEHRP` | 3 |
| `AASHTO LRFD` | 7 |
| `kip-ft` | 4 |
| `ksi` | 1 |
| `ASTM A709` | 6 |
| `A709 Grade 50W` | 9 |
| `HPS 70W` | 6 |
| `f'c` | 3 |
| `P-delta` | 3 |
| `Cs = SDS/(R/Ie)` | 8 |
| `orthotropic` | 2 |
| `electroslag` | 3 |
| `austenitic` | 3 |
| `martensitic` | 3 |
| `Charpy V-notch` | 5 |

Worst-fragmented term words: `1110-2-2104` (12), `SDS/(R/Ie)` (6), `strong-column/weak-beam` (6), `(f’c)` (5), `7-16` (5), `7-22` (5), `sub-diaphragm` (5), `(CIF)` (4), `(EDO)` (4), `(LFRS)` (4), `(RBS)` (4), `(Ωv)` (4), `100` (4), `50W` (4), `70W` (4), `A709` (4), `AASHTO` (4), `Diaphragm` (4), `Earthquake)` (4), `HL-93` (4), `I-girder` (4), `No-decompression` (4), `Timoshenko` (4), `capacity-demand-diagram` (4), `contraflexure` (4), `high-deformability` (4), `kip-ft` (4), `lateral-torsional` (4), `nonprismatic` (4), `percent-30` (4), `semi-integral` (4), `shear-buckling` (4), `web-plumbness` (4), `wind-restraint` (4)

**Split and packing** (split.py)

| publisher | train docs | train tokens | val docs | val tokens |
|---|---|---|---|---|
| FEMA | 44 | 4,669,258 | 2 | 112,176 |
| FHWA | 61 | 3,758,566 | 3 | 80,055 |
| NASA | 3 | 95,445 | 1 | 18,678 |
| NIST | 27 | 1,639,770 | 1 | 84,508 |
| USACE | 99 | 9,258,193 | 5 | 919,095 |

Val documents: fema-p-1100-2a, fema-p-2018, fhwa-hif17020, fhwa-hif18044, fhwa-hif18047, nasa-std-5002b, nist-gcr-12-917-21, usace-em-1110-1-1804, usace-em-1110-2-1906, usace-em-1110-2-1908, usace-em-1110-2-2610-final-18mar2025, usace-em-1110-2-3506. 52 eval documents held in train. Packed at 4,096: 4,741 train / 296 val sequences (5,216 with replay); 18.5 optimizer steps per epoch at ~1M tokens/step.

**Decisions:**
- **PDF portfolios:** USACE EM 1110-2-1100 (all 6 parts), -1424 and -1701 are a one-page "open in
  Acrobat" cover with the manual as embedded PDFs. They're read through their parts
  (common.pdf_parts): 4,728 pages that were first misread as scanned. A portfolio is recognised by
  the /Collection entry the PDF spec gives it, not by having embedded PDFs. EM 1110-2-3200 attaches
  17 failure reports to a 238-page body, and the first rule read only the attachments.
- **No OCR:** 4,081 image-only pages (6.7%) and 1,497 near-blank pages are dropped, and 5 documents
  lose most of their pages. OCR is the first lever for more tokens, but scanned 1980s manuals would
  bring OCR noise that the dictionary rule exists to keep out.
- **Filter tuned from dropped_samples.jsonl:** the literal line-count rules removed 28% of
  characters on the seed set, half of it real content (bulleted requirement lists, worked-example
  "where:" blocks, two-line paragraphs). Three changes, all documented in filter.py:
  - marker-only lines ("•", "1.") are joined to their item;
  - the line-ratio rules need 3+ lines;
  - short_lines counts words, not lines.
  alpha_ratio is kept as specified: it's the largest rule (5.8M tokens) and its token-weighted
  samples are dot-leader TOCs, garbled OCR, XML and equation fragments. The dictionary threshold
  was raised to 0.70 because FHWA-HIF-16-010 (0.649) is half XML object listings, and kept
  documents otherwise score 0.74+.
- **Paragraph-level near-dedup, not document-level:** exact dedup found no duplicate documents.
  Paragraph MinHash removes 4.4%: federal boilerplate shared across 20-57 documents (the FHWA
  quality statement, USACE letterhead, distribution statements, DOT report-documentation fields),
  plus running headers the 30%-of-pages rule misses (per-chapter titles, NIST's "available free of
  charge" line), plus the overlap between the 2009 and 2015 NEHRP design examples. Document-level
  dedup would keep all of that. Documents under 2,000 words are dropped again after dedup, because a
  later edition can shrink to a stub (FEMA P-2192 to 1,906 words, P-1024-RA1 to 44). Their
  paragraphs are taken back out of the index so later copies of them survive.
- **Document-level split, eval documents in train:** val perplexity then measures generalisation
  to unseen documents of the same kinds, and the KPI eval measures what CPT absorbed from documents
  it saw. All 52 seed documents are eval sources, so the 12 val documents all come from the
  expansion: max(1, 5%) per publisher, chosen by sha256(slug). Dedup runs before the split, so no
  near-duplicate paragraph sits on both sides.
- **No vocabulary extension:** measured on the two checkpoints in use, Base and Instruct (user
  decision: no other tokenizer), against FineWeb-Edu as general English.
  - Tekken spends 1.08x more tokens per word on this corpus than on FineWeb-Edu (1.440 vs 1.338).
  - The top-500 TF-IDF domain terms average 1.12 tokens per word.
  - Only 34 of 843 term words take 4+ tokens: designations and digit strings (EM 1110-2-2104 is
    13 tokens because Tekken splits digits by design), parenthesised acronyms, and hyphenated
    compounds.
  - New embeddings trained on 20M tokens would cost more than they save. Base and Instruct encode
    identically.
- **PII:** three regexes (58 emails, 180 phone/fax numbers, 0 SSN-shaped), all spot-checked as true
  positives; author names are kept for citations. At Forge scale this is a Presidio-class NER pass
  with client-specific entity lists.
- **Replay slice (FineWeb-Edu):** `replay.jsonl` holds 1,780 documents, 1,944,824 Tekken tokens
  (10% of train; 475 packed sequences).
  - **Why replay:** CPT on 19M tokens of one narrow domain risks forgetting general ability. Mixing
    general text back in is the standard mitigation; continual pre-training work typically replays
    5-25%.
  - **Why 10%:** it's the ablation's starting point, not a tuned value.
  - **Why FineWeb-Edu:** English web text kept by an educational-quality classifier, the closest
    open match to what MMLU, GSM8K and HellaSwag probe. It's stored as raw text and counted in
    Tekken tokens; pre-tokenised shards (e.g. GPT-2) would be the wrong tokenizer.
  - **Stage 2 use:** CPT is run with and without replay. Replay earns its place if the regression
    columns (MMLU, GSM8K, HellaSwag) hold better with it at little cost to domain val perplexity
    (targets in the entry below). The first ~1M tokens also serve as tokenizer_coverage's
    general-English baseline.
  - **Caveats:**
    - Not a random sample: it's the head of the `sample-10BT` stream, so it's reproducible but
      drawn from the start of the first shard. If the ablation looks sensitive to the mix, draw a
      seeded sample across shards instead.
    - Not decontaminated against the regression benchmarks, and not run through this corpus's
      filters. A benchmark item in web text would flatter the replay run's lm-eval scores, so check
      for overlap before crediting replay with a regression-suite gain.
    - Licence: FineWeb-Edu is ODC-By web text, not public domain. It isn't redistributed here
      (`replay.jsonl` is gitignored and rebuilt from the Hugging Face stream), and nothing in the
      eval comes from it.
- **Packing:** 4,741 train sequences of 4,096 (5,216 with replay), about 18.5 optimizer steps per
  epoch at a 1M-token batch, so Stage 2's LR schedule needs a short warmup.
**Revisit if:** CPT val perplexity barely moves (entry below). Check whether the corpus is smaller
than it looks before adding epochs: OCR on the scanned manuals and the remaining FEMA and FHWA
libraries are the next sources.

## 2026-09-27: CPT success criteria on val perplexity (user decision)
Stage 2 reads the CPT run against `data/processed/val.jsonl` (held-out documents, never an eval
source) and base-model perplexity on the same file:
- **Val perplexity down at least 20% vs Base and train/val gap under ~10%:** CPT worked, move on.
- **Val barely moved:** undertrained. The LR is too low for LoRA, or the corpus is smaller than
  it looks after dedup. The fix is LR up one step or a second epoch, not a different model.
- **Val down but train/val gap over ~25%:** memorising. Note it, keep the checkpoint, don't add epochs.

## 2026-09-27: Eval regenerated from the 234 train documents and frozen before Stage 2 (user decision)
**Context:** the KPI tasks came from the 52 seed documents, 22.7% of train tokens after the Stage 1
expansion (FHWA 57% of items vs 19% of tokens; USACE 5% vs 48%), and `eval_chunk_ids.txt` reserved
no chunks from the new documents for the Stage 3 contamination rule. The eval has to be frozen
before the first training run, because every later row is compared against it; no trained row
existed yet, so nothing was lost by resampling.
**Options:** keep the 664 reviewed seed-set items and add a supplement from the new documents, vs
regenerate from the full train pool with the same seed and filters.
**Chose:** regenerate (user decision), with these changes:
- Pool: the 234 train documents, pinned in `eval/tasks/eval_docs.txt`. `extract.py --chunks`
  (76,582 chunks) and `split.py` read it; train/val came out byte-identical. The pool is a file,
  not derived from `eval_chunk_ids.txt`, so it can't drift, and every pool document stays in train.
- `take()` caps source chunks at `--per-doc 6` per document per task (per-task RNG streams kept),
  so a 1,700-page manual can't supply as many items as dozens of briefs.
- `--n-qa-extra 350`: supplementary domain_qa sampled after every other task from its own stream,
  ids from 501, deduped against the main set. Review kept only ~27% of verified QA on this pool.
**Review:** every generated item (1,176 plus 3 few-shot) was reviewed against its passages before
any model generation, by review agents working to a written rubric: keep or reject only, reject
only for a concrete defect (wrong gold, a correct answer the scorer marks wrong, a wrong answer it
marks right, general knowledge / trivia / one worked example's value), never for difficulty; when
unsure, keep and flag. The lead re-checked every flagged item and spot-checked keeps: a blind
30-item QA hand check (17/21 agreement with the agents; the agents were stricter and right on 3),
10 vocab, and 8 from the most lenient grounded batch (1 miss, a gold cut off before its answer), which
triggered a second pass over the two lenient grounded batches (0 further misses). Duplicates were
checked across batches.
Kept / generated by source publisher:

| task | USACE | FEMA | FHWA | NIST | NASA | all |
|---|---|---|---|---|---|---|
| domain_qa (+ few-shot) | 40/240 | 45/151 | 30/102 | 17/56 | 1/3 | 133/552 (24%) |
| grounded | 34/46 | 26/39 | 30/38 | 17/19 | 1/1 | 108/143 (76%) |
| vocab | 88/158 | 62/107 | 48/86 | 10/20 | 2/4 | 210/375 (56%) |
| adversarial | 30/39 | 12/26 | 20/25 | 14/19 | 0/0 | 76/109 (70%) |

QA rejects were mostly scoring-format defects the exact/numeric scorer can't handle fairly (number
words vs digits, two-unit or range golds, designations scored numerically, abbreviation-only golds,
edition-suffixed citations), then general knowledge, the document's own numbering and citation
trivia, worked-example or case-study values, and wrong golds (e.g. an aluminum ASTM given for copper
fittings, a divisor called an exponent, a page-footer "2"). Grounded rejects: the gold passage
doesn't hold the full answer (cut off, "why" not stated, only in a neighbour). Adversarial rejects:
answerable by inference, premise rebutted, non-prose source. Vocab rejects: passage-narrowed or
wrong references, ambiguous bare words, bare abbreviations, compositional phrases.
Old rejects still apply. Question-substring rules matched only identical questions (same passage,
same cached generation: 3 QA, 1 grounded). Three old vocab term rules (post-tensioning, Seismic
Design Category, seismic isolation) were deleted because their reasons were instance-specific and
the new definitions are correct.
**Final:** domain_qa 130 + 3 few-shot, grounded 108, vocab 210, adversarial 76 (524 scored, was
661), from 152 of the 234 documents; `eval_chunk_ids.txt` 1,570 chunks; rebuild byte-identical;
`rejects.jsonl` 806 entries.

| task | USACE | FEMA | FHWA | NIST | NASA | documents |
|---|---|---|---|---|---|---|
| domain_qa | 40 | 45 | 29 | 15 | 1 | 70 |
| grounded | 34 | 26 | 30 | 17 | 1 | 72 |
| vocab | 88 | 62 | 48 | 10 | 2 | 69 |
| adversarial | 30 | 12 | 20 | 14 | 0 | 58 |

**Evidence:** `eval/tasks/rejects.jsonl` (entries reviewed "2026-09-27 regenerated-set review"),
`results/table.md` rows base-8b / instruct-8b rescored on this set. lm-eval rows are unchanged
(independent of the tasks); latency re-run because `bench_latency.py` samples its prompts from them.
**Revisit if:** the corpus changes before Stage 2 (edit `eval_docs.txt` deliberately, regenerate,
re-review). USACE is under-represented in domain_qa (31% of items vs 48% of train tokens) because
its manuals yield fewer scoreable closed-book facts (17% kept vs ~30% elsewhere): report `qa_acc`
by publisher before reading the aggregate. `exact_match` treats number words and digits as
different; items that relied on it were rejected rather than changing the scorer mid-baseline.
**Baselines on this set** (`results/table.md`; the old rows were removed): Base qa_acc 0.139,
grounded_acc 0.852, cite_valid 0.130, cite_supported 0.102, vocab_recall 0.719, halluc_rate 0.882.
Instruct 0.115 / 0.898 / 0.833 / 0.787 / 0.786 / 0.013. lm-eval unchanged. A spot check of 10 judge
verdicts on the new items against their gold text found no wrong verdicts. qa_acc by publisher (Base / Instruct, n): USACE
0.100 / 0.075 (40), FEMA 0.200 / 0.200 (45), FHWA 0.069 / 0.034 (29), NIST 0.200 / 0.133 (15),
NASA 0 / 0 (1). The cells hold 0-9 correct answers, too few to rank publishers; CPT deltas per
publisher are what they are for.

## 2026-09-27: Stage 2 CPT hyperparameters (pre-registered before the main run)
**Context:** Stage 2 continues pre-training `Ministral-3-8B-Base-2512` on `train.jsonl` (234 documents,
19.42M Tekken tokens, 4,741 windows of 4,096), one epoch, LoRA, then merges it into an ordinary
checkpoint. Every number below is fixed before the first run so the result can't tune it.
**Chose** (`train/configs/cpt.yaml`):
- **LoRA r=64, alpha=128, dropout 0.05, on q/k/v/o/gate/up/down of the language model.** CPT adds
  knowledge, and the MLPs are where it goes; attention-only LoRA underperforms for CPT. r=64 gives
  ~178M trainable parameters (238 modules, 34 layers x 7), alpha = 2r is the usual scale, and 0.05
  dropout is light regularisation for a single pass. The regex is anchored on `language_model`, so
  the vision tower stays frozen (rule 10).
- **LR 1e-4, cosine to zero, 3% warmup (~4 steps), weight decay 0.** LoRA wants roughly 10x full
  fine-tuning's LR (1e-5, used by ablation B). Warmup is short because the schedule is short.
  Weight decay on adapters pulls them toward zero, i.e. toward the base, which one epoch doesn't
  need.
- **Effective batch: the 150-step rule.** Size the batch for about 150 optimizer steps in the
  epoch, the least that lets LoRA settle and keeps the schedule sane, as a power of two in
  sequences: 4,741 windows / 150 = 31.6, so 32 sequences (micro-batch 4 x accumulation 8,
  131k tokens), 149 steps. At ~40M tokens it becomes 64, at ~80M 128; the step count stays ~150.
  `split.py` computes it (`stats.json` split.seqs_per_step) and `cpt.py` refuses an epoch outside
  120-190 steps. This replaces the ~1M-token batch assumed in the Stage 1 packing note (18.5 steps,
  too few). **Not a second epoch** to restore steps: it would show every token twice, widen the
  train/val gap and confound the ablations.
- **bf16 autocast with gradient checkpointing; SDPA, not flash-attn.** Peak estimated at ~60 GB of 80
  (full-vocab logits at 4 x 4,096 x 131k dominate: ~26 GB across their fp32 copies). Eval batch 4,
  not the default 8, which would double that. Every window is a full causal sequence, where SDPA
  already runs its FlashAttention-2 kernel; the flash-attn package only matters for padding-free
  packing and would need a CUDA build in the image.
- **Eval every 25 steps on 49 val windows (~200k tokens), spread across `val.jsonl`; save every 50,
  keep 2; log every step.** Six eval points on a 149-step curve, each ~30 s. A save is ~20 min of
  training, the most a restart loses (`cpt.py` resumes from the newest checkpoint).
- **Own packing, not TRL's.** `train/packing.py`: each document as BOS + tokens + EOS (ids), all
  concatenated in file order, cut into 4,096-token windows, the last partial window dropped; the
  trainer shuffles windows. TRL's default packing keeps only the first 4,096 tokens of each document
  (~1M of 19.4M), its raw-text path appends EOS as the string `</s>`, which Tekken encodes as text,
  and its `max_length` defaults to 1,024 (`notes/contributions.md`). `eval/perplexity.py` uses the
  same windows, so trainer eval_loss and the perplexity columns agree by construction.
- **`report_to: none`.** The curve is `results/runs/<run>/train_log.jsonl`, written line by line.
**Evidence:** packing dry run on the Mac: train 4,741 / val 296 / train+replay 5,217 / general_val
122 windows, each equal to `sum(n_tokens + 2) // 4096` from the files (tokeniser matches data
prep's counts); 234 BOS for 234 documents.
**Revisit if:** val loss plateaus in the first third of the epoch (LoRA capacity: r=128 or full
fine-tuning), or eval loss rises while train loss falls (LR too high for 149 steps).

## 2026-09-27: Stage 2 decision rule, extended with general perplexity (pre-registered)
**Context:** the CPT success criteria entry above fixes the domain side. The forgetting side needs a
bound too, set before the numbers exist.
**Chose:** per checkpoint, `eval/perplexity.py` reports `ppl_train` (49 windows spread across
`train.jsonl`: tokens a CPT run saw once; user decision: measured, not the logged training loss,
so base gets the same number), `ppl_domain_val` (all of `val.jsonl`, and per publisher),
`ppl_general_val` (`general_val.jsonl`: 500k FineWeb-Edu tokens from the last shard of
sample-10BT, disjoint from replay) and `ppl_val_slice` (the trainer's eval windows, a merge check).
CPT worked if, against base-8b:
- `ppl_domain_val` is down at least 20%;
- the train/val gap grows by under ~10 points over base's; over ~25 points means memorising. The
  gap is `ppl_domain_val / ppl_train - 1` on the same slices for both models, and the rule bounds
  CPT's gap minus base's (user decision): val documents differ from train documents (val is 76%
  USACE by tokens, train 48%), so base has a gap of its own before any training, and only the
  growth is what training added;
- `ppl_general_val` is up less than 3%.
The corpus is final for Stage 2 at 20.6M tokens (user decision, 2026-09-27): no expansion before
these runs.
**Expected shape** (to tell a result from a bug): domain val perplexity down 25-40%; `qa_acc` and
`vocab_recall` up a modest but clear amount; grounded, citation and hallucination metrics roughly
flat (SFT's and DPO's job); general perplexity up 2-6% and MMLU down 0.5-2 points without replay,
roughly flat with it. A row where everything improves, or nothing moves, is the one to distrust.
20M tokens is the bottom of the 20-50M range; the `qa_acc` gain per token is itself a finding.

## 2026-09-27: Stage 2 ablations: replay, 3B full-parameter, 2-GPU FSDP (pre-registered)
**Context:** three questions every domain-adaptation engagement has to answer, each run varying one
thing against the main run (`cpt-8b`).
**Chose:**
- **A, `cpt-8b-replay10`** (`cpt_replay10.yaml`): the main config with `replay.jsonl` (10% of train
  tokens, FineWeb-Edu) mixed into the windows: 5,217 windows, 164 steps at the same 32 sequences
  per step. Question: how much of `ppl_general_val` / MMLU does replay protect, at what cost in
  `ppl_domain_val`.
- **B, `cpt-3b-full`** (`cpt_3b_full.yaml`): full-parameter CPT of `Ministral-3-3B-Base-2512` on the
  same windows and batch, LR 1e-5, vision tower and projector frozen. The weights are fp32 with bf16
  autocast: loaded in bf16, the Trainer applies AdamW updates to the bf16 weights directly (no fp32
  master copy), and a 1e-5 step is below bf16's resolution for most weights, so the model would
  barely move. 8-bit AdamW (bitsandbytes) keeps the optimizer state small enough for one H100, and
  micro-batch 2 x 16 because 4 x 8 was estimated at 70-73 GB. Question: heavy adaptation of a small
  model vs light adaptation of a large one, on domain gain and on forgetting, each measured against
  its own base: a `base-3b` reference row gets perplexity, lm-eval and the KPI eval (user decision;
  without it, model size and adaptation are confounded).
- **C, `cpt-8b-fsdp2`** (`cpt.yaml` + `fsdp2.yaml`): the main config on 2 x H100 with FSDP2, 2 GPUs x
  micro-batch 4 x accumulation 4 = the same 32 sequences per step, stopped at step 100 by a callback
  rather than `max_steps`, so the cosine schedule still spans the 149-step epoch and both runs see
  the same LR at each step. Activation checkpointing through FSDP (the Trainer's is off; it refuses
  both), `FULL_STATE_DICT` (the FSDP2 default would save no adapter), `cpu_ram_efficient_loading`
  off (it calls `tie_weights()`, which would tie the 8B's `lm_head` to its embeddings). No eval, no
  merge: a systems measurement. Question: tokens/s on 1 vs 2 GPUs, and whether the loss curves
  match (mean relative gap of their 10-step moving averages, steps 10-100; the ranks shard the data
  differently, so the curves agree statistically, not batch for batch).
**Evidence:** `results/train_runs.md`, `results/curves/cpt.png` (`train/report.py`) and the table rows,
after the runs.
**Revisit if:** B doesn't fit at micro-batch 2 (then paged 8-bit AdamW, or plain Trainer without
SFTTrainer's token-accuracy copy of the logits).

## 2026-09-27: Numeric precision: bf16 compute, fp32 adapters, TF32 underneath, fp32 merge
**Context:** bf16 and TF32 are not alternatives; they apply to different operations. bf16 is a
16-bit storage and compute format (8 exponent bits, fp32's range; 7 mantissa bits). TF32 is not a
storage format: it is a tensor-core mode for matmuls whose inputs are fp32, which rounds the
mantissa to 10 bits inside the multiply and returns fp32, several times faster than true fp32. It
is one flag; nothing in the model changes dtype.
**Chose** (every LoRA stage: CPT, SFT, DPO, GRPO):
- **Base weights in bf16, frozen** (`model.dtype: bfloat16`): half fp32's memory, and every forward
  matmul runs on bf16 tensor cores.
- **LoRA A and B in fp32, with fp32 AdamW states.** They are the only parameters updated, and each
  step adds a small number to a small number; in bf16 those updates round to nothing. This is the
  mixed-precision "fp32 master weights" pattern, applied to the trainable part only. No code needed:
  every stage goes through a TRL trainer's `get_peft_model`, and PEFT (0.21) upcasts bf16 adapters to
  fp32 by default (`autocast_adapter_dtype=True`).
- **bf16 autocast** (`bf16: true`): matmuls and attention in bf16, reductions (softmax, norms, the
  loss) in fp32. SDPA's FlashAttention-2 kernel needs bf16 or fp16 inputs, so this is also what
  opens the fast attention path (Stage 2 hyperparameters entry).
- **TF32 on** (`tf32: true`): it speeds up whatever matmuls still run in fp32 (the LoRA path computed
  in fp32, ops next to the norms and the loss). It costs nothing and changes nothing else; without
  it those matmuls take the slow full-fp32 path.
- **Merge in fp32.** `merge.py` loads the base in fp32, merges, and casts to bf16 once before saving,
  so W + BA is added exactly and rounded a single time. Merging into a bf16 base (PEFT does
  `W += delta.to(W.dtype)`) rounds the delta to bf16 and then rounds the sum again. It needs ~34 GB
  of RAM for the 8B model. A full-parameter output (ablation B) is already fp32, so it gets the same
  single cast to bf16 on load.
**Rejected:** fp16 (narrower range, needs loss scaling, no reason for it on an H100); bf16 adapters
(the updates vanish); fp8 training (a production-scale technique; Ministral's FP8 checkpoints are
inference artefacts).
**Full-parameter case (ablation B, 3B):** the one run where the base weights are updated, so they
are held in fp32 (`dtype: float32`) with bf16 autocast for compute, and 8-bit AdamW
(`adamw_bnb_8bit`; paged 8-bit AdamW is the fallback if it doesn't fit) keeps the optimizer state on
one card. The 8B equivalent is ~16 bytes per parameter (fp32 weight 4 + gradient 4 + Adam m and v 8),
~127 GB for the ~7.9B language model, well past one H100's 80 GB before activations; even 8-bit Adam
(~10 B/param, ~79 GB) leaves no room for them. That is the memory argument for LoRA that ablation B
is there to quantify.
**Status at this entry:** bf16 weights, `bf16: true`, fp32 adapters and the 3B fp32/8-bit config are
in the code. Still to do: `tf32: true` is set in no training config, and `merge.py` currently merges
into a bf16 base (`from_pretrained(..., dtype=torch.bfloat16)` before `merge_and_unload()`).
**Revisit if:** a stage trains adapters outside a TRL trainer (then set the adapter dtype
explicitly), or the merge moves somewhere with less than ~40 GB of RAM.

## 2026-09-27: Ablation B redesigned: 8B full-parameter on 2 x H100, not 3B (user decision)
**Context:** the pre-registered B (full-parameter CPT of the 3B against LoRA on the 8B) changed two
things at once, method and model size, so whichever won, the result couldn't say why. It was set up
that way only to keep every run on one GPU. Decided before any B run (the 3B only ran a 20-step
smoke test); this supersedes B in the ablations entry above and the "Full-parameter case (ablation
B, 3B)" paragraph of the numeric precision entry.
**Options:** 3B full vs 8B LoRA (confounded; plus a base-3b row to take deltas from) vs 8B full vs 8B
LoRA (same model, same tokens, only the parameterisation differs).
**Chose:** `cpt-8b-full` (`train/configs/cpt_8b_full.yaml`, `--gpus 2`): full-parameter CPT of the
same 8B on the same 4,741 windows and 32 sequences per step (2 GPUs x micro-batch 2 x accumulation
8, 149 steps), LR 1e-5 cosine, 3% warmup, one epoch, on 2 x H100 with FSDP2 (`fsdp2.yaml`).
- Memory: the ~8.5B trainable language-model parameters (embeddings and `lm_head` included; vision
  tower and projector frozen) at 16 bytes each in fp32 AdamW would be ~135 GB before activations,
  past one H100. Sharded over two: fp32 master weights (~18 GB per GPU) + fp32 gradients (~17 GB) +
  8-bit AdamW state (~8.5 GB), bf16 compute through FSDP mixed precision, FSDP activation
  checkpointing, micro-batch 2: ~60-65 GB per GPU estimated, confirmed or refuted by the smoke test.
- Optimizer: torchao `AdamW8bit` (`optim: adamw_torch_8bit`), not bitsandbytes' paged 8-bit AdamW:
  FSDP2 shards parameters as DTensors, which torchao's low-bit optimizer states are built for;
  bitsandbytes 0.50's 8-bit kernels expect plain tensors (its FSDP support covers the FSDP1 state
  dict). Paged AdamW would only matter under memory pressure.
- No fused cross-entropy: Liger-Kernel 0.8.3 patches no `mistral3` / `ministral3` model type
  (`notes/contributions.md`). The full-vocab logits cost ~13 GB per GPU at micro-batch 2, which the
  estimate includes.
- No intermediate checkpoints (`save_strategy: "no"`): a full checkpoint is ~35 GB of fp32 weights
  plus sharded 8-bit optimizer state, for a ~1 h run; the final model is saved with
  `FULL_STATE_DICT` and merged (cast to bf16 once) like the LoRA runs.
- Each rank loads the fp32 model on CPU before sharding (`cpu_ram_efficient_loading` off, see C),
  so the 2-GPU container gets 128 GiB of RAM.
C stays as it was (the main LoRA config on 2 GPUs, stopped at step 100), and is kept separate from
B: a full-parameter step costs more compute than a LoRA step, so B's tokens/s is not a clean
1-vs-2-GPU number (`train/report.py` compares only C with `cpt-8b`, and reports tokens/s per GPU).
The 3B and the `base-3b` row are dropped.
**Evidence:** smoke test (`smoke-8b-full`, 20 steps of 8 sequences): 8,489,553,920 trainable
parameters; peak allocated 60.9 GB per GPU (the caching allocator reaches the 80 GB ceiling and
frees cache to retry, logged as `expandable_segments ... OOM` warnings; shapes are fixed, so the
20 steps are the steady state and micro-batch 2 stays); 12,699 tokens/s on 2 GPUs (6,350 per GPU,
vs 6,028 for LoRA on one), so ~26 min of training for the epoch; torchao AdamW8bit ran under FSDP2;
the `FULL_STATE_DICT` save merged with weight names equal to the base's. Step-1 loss 1.881 (LR 0)
equals the LoRA smoke run's 1.878 on the same batch, so the full-parameter model loads intact; eval
loss rose from base's 1.931 to 1.969 at step 10 and was back to 1.960 at step 20 (a first-steps
bump; whether the full epoch recovers it is part of what B measures). The result states itself as
"on the same 8B model and the same tokens, full-parameter CPT gained X on domain perplexity and lost
Y on MMLU relative to rank-64 LoRA, at Z times the GPU-hours".
**Revisit if:** the smoke test doesn't fit at micro-batch 2 (then chunked cross-entropy, e.g. Apple's
model-agnostic cut-cross-entropy, before a smaller micro-batch), or torchao's optimizer fails under
FSDP2 (then bitsandbytes `paged_adamw_8bit`).

## 2026-09-27: TF32 from Stage 3; fp32 merge in; LR-up variant if CPT under-delivers (user decisions)
**Context:** the numeric precision entry above left two items to do, and the main CPT run was already
training when they came up; its halfway eval showed domain val perplexity down only ~2%.
**Chose:**
- **TF32 from Stage 3, not in Stage 2.** `cpt-8b` started without `tf32: true`, and A and C must
  differ from it in one thing only. Under bf16 autocast almost no matmul runs in fp32 (autocast casts
  matmul inputs to bf16 whatever the parameter dtype), so the effect is negligible either way; the
  SFT/DPO/GRPO configs get `tf32: true` when those stages start.
- **Merge in fp32: done** (`train/merge.py`), before any Stage 2 checkpoint was merged, so every
  Stage 2 row goes through the same merge. It also sets bf16 on every sub-config, which the cast
  alone leaves at float32 (`notes/contributions.md`).
- **If `cpt-8b` ends under the pre-registered 20% drop in domain val perplexity**, the rule's own fix
  is taken, one LR step up (not a second epoch, which would confound the ablations): a
  `cpt-8b-lr2x` run at 2e-4 joins the ablation batch, proposed with it at the launch gate. The
  hypothesis it tests against: the corpus is public US federal documents on the open web, very
  likely in Ministral's pretraining data already, so there is less left to learn than the
  pre-registered 25-40% assumed (base perplexity is 6.9 on these manuals).
**Revisit if:** the LR-up run moves domain val perplexity much further (then 1e-4 was simply low), or
it doesn't (then the documents' prior exposure is the likelier explanation, worth a per-document
check of base perplexity against publication date).

## 2026-09-27: Reference model is the previous stage's checkpoint, not the base (user decision)
**Context:** DPO and GRPO both regularise toward a frozen reference model, and it's easy to read
"the reference" as the base. It isn't. With LoRA and `ref_model=None`, TRL 0.29.1 uses the policy
with its adapter disabled as the reference, i.e. the model `model.init_from` loaded, which is the
previous stage's merged checkpoint.
**Options:** the previous stage's checkpoint (what LoRA plus merge-between-stages gives for free) vs one
fixed anchor for every stage (the base or `cpt-8b`, which needs an explicit `ref_model`: a second 8B
copy in memory).
**Chose:** the previous stage.
- **DPO:** reference = the SFT checkpoint (`dpo.yaml` `init_from: checkpoints/sft-merged`). `beta`
  scales the policy/reference log-ratios, so DPO's implicit KL constraint measures drift from SFT.
  This also matches the pairs, which are sampled from the SFT model.
- **GRPO:** reference = the DPO checkpoint (`grpo.yaml` `init_from: checkpoints/dpo-merged`). Its KL
  term, and the `kl` TRL logs, measure drift from DPO. `grpo.yaml` sets `beta: 0.0` for now, so
  there is no KL term (TRL doesn't build a reference at all) until beta is raised.
- A fixed base anchor would penalise the changes the earlier stages were meant to make (DPO anchored
  to the base would pull back SFT's instruction following).
**Consequence for the write-up:** each stage's KL (and DPO's reward margins) is drift within that
stage. It can't be read as distance from the base, and no term constrains cumulative drift from the
base. Only the same eval (KPI + lm-eval regression) run on every merged checkpoint measures that.
**Revisit if:** the lm-eval regression accumulates across SFT -> DPO -> GRPO while each stage's own
KL stays small. Then anchor the late stages to an earlier checkpoint (explicit `ref_model`) and log
that KL too.

## 2026-09-27: Stage 2 ablation rules, read against a measured noise floor (pre-registered; user decision)
**Context:** `cpt-8b` finished: against base-8b, train-slice perplexity -8.3%, domain val -2.3%, general
val +0.4%, train/val gap 11.2% -> 18.5% (+7.2 points). By the rule above that is the "barely moved"
branch (under the 20% drop; the gap growth and the general bound are within theirs), so the LR-up
run is on. Ablation deltas against a -2.3% effect will be tenths of a point to a few points: without
a measured run-to-run spread nobody can tell them from noise. Written before A and B launched and
before the LR-up and seed runs finished.
**Chose:**
- **Noise floor, `cpt-8b-seed1`:** `cpt.yaml` with `seed: 1` (another LoRA init and data order;
  `cpt.py` now seeds before building the model, so seeds reproduce from here on). It gets perplexity
  and lm-eval. The floor for a metric is |seed1 - main| on it.
- **Reading rule for any two runs:** a difference counts only if it exceeds the seed floor on that
  metric and, for perplexities, its 95% paired bootstrap interval over documents excludes 0
  (`eval/ppl_compare.py`; windows are reported too, but a document's windows are correlated). Every
  delta is reported in nats per token and in % perplexity.
- **LR-up, `cpt-8b-lr2x`:** continues as the Stage 3 starting point only if its domain-val gain over
  `cpt-8b` beats the seed floor with `ppl_general_val` still under +3% of base. Otherwise 1e-4 stays.
- **A, replay:** replay is adopted for Stages 3+ only if `ppl_general_val` or MMLU beats the main run's
  by more than the seed floor, with domain val within 3% (relative perplexity) of the main run's.
  Stated in advance: the main run forgot almost nothing (+0.4% general perplexity), so A may be null
  on both axes. That is the finding ("at 20M tokens and 149 LoRA steps replay has nothing to
  protect"), written up as such, not re-run at a bigger scale to make it bite.
- **B, full vs LoRA:** written up on three axes against base-8b: domain gain (domain val perplexity),
  general loss (`ppl_general_val`, MMLU), GPU-hours. LoRA stays the default unless full's extra
  domain gain over LoRA beats the seed floor and is larger than its extra general loss. Reference
  expectation: full gains more and forgets more (Biderman et al. 2024, "LoRA Learns Less and Forgets
  Less").
- **C, scaling:** systems only: tokens/s ratio and per-step loss agreement. A 1-GPU FSDP run of the
  same config (if run) separates the code path from the GPU count.
**Rejected:** re-splitting val and regenerating the tasks. Val already holds 12 held-out documents from
all five publishers (1.21M tokens, 5.9%); the eval pool is the 234 train documents, so no val
document feeds any task; and a new split would invalidate `cpt-8b`, base-8b's perplexity and the
eval frozen this morning. USACE is 76% of val tokens, so `results/ppl/<run>.json` reports
perplexity per publisher and per document next to the pooled number.
**Evidence:** `results/ppl/*.json`, `eval/ppl_compare.py` output, `results/train_runs.md`.

## 2026-09-28: LR-up verdict: 1e-4 stays (pre-registered rule applied)
**Context:** `cpt-8b` landed in the "barely moved" branch, so `cpt-8b-lr2x` (LR 2e-4, all else equal)
ran. The rule: it continues only if its domain-val gain over `cpt-8b` beats the seed floor.
**Result** (`eval/ppl_compare.py cpt-8b cpt-8b-lr2x`, 95% paired bootstrap over documents):
domain val -0.10% [-0.34, +0.09] (covers 0: no gain, before the seed floor even applies); general
val +0.44% [+0.37, +0.52] (more forgetting); train slice -4.1% (more memorising; train/val gap
23.4% vs 18.5%).
**Chose:** 1e-4 stays the Stage 2 config; ablation A runs at 1e-4. Doubling the LR made the model
learn the documents it read harder and forget a little more, with nothing extra on unseen documents:
what the prior-exposure hypothesis predicts, and not what "1e-4 was too cautious" predicts.

## 2026-09-28: Every checkpoint through vLLM's HF path (config_format=hf); base-8b-hf row (user decision)
**Context:** `cpt-8b` scored MMLU -3.7 and GSM8K -6.7 points against base-8b while its general
perplexity moved only +0.4%. vLLM's logs showed the two rows came from different model
implementations: the hub base, which ships Mistral's native `params.json` +
`consolidated.safetensors` next to the HF files, resolved to `PixtralForConditionalGeneration` (the
native path, chosen with `tokenizer_mode=mistral`); the merged checkpoint, HF files only, to
`Mistral3ForConditionalGeneration`. Every vLLM-based delta (lm-eval, KPI, latency) between them mixed
training with the implementation. Perplexity, computed in transformers, is unaffected.
**Chose:** `config_format="hf"` in `run_eval.py`, `run_lm_eval.sh` and `serve_vllm.sh`, so every
checkpoint, base or merged, runs the same HF implementation. The base is re-evaluated through it as
`base-8b-hf` (lm-eval, KPI, latency), and Stage 2 deltas are read against `base-8b-hf`; the Stage 0
`base-8b` / `instruct-8b` rows stay as the native-path reference (instruct is re-run the same way
before Stage 3 compares against it). `base-8b-hf` minus `base-8b` is the path's own effect.
**How base-8b-hf is built:** `config_format=hf` alone can't load the hub repo: vLLM then reads every
`*.safetensors` in it, including the native `consolidated.safetensors`, whose `layers.*` keys the HF
implementation can't map. So the base is re-saved through `merge.py` (`--full` of the base onto
itself: `modal_train.py --model <hub id> --run-name base-8b-hf --steps merge,eval`) and that directory
is evaluated: the same files a merged checkpoint has (HF only, config written by the same
transformers, the base's tokenizer files), so `base-8b-hf` differs from `cpt-8b` in the weights only,
and it doubles as a check that the merge path itself changes nothing.
**Also:** `run_lm_eval.sh` now passes `limit_mm_per_prompt={"image": 0}` (rule 3; a merged checkpoint
otherwise fails vLLM's dummy-image profiling). A base-8b control with it (`base-8b-mm0`, still the
native path) moved MMLU +0.01, its groups by +/-0.1-0.2, GSM8K strict +0.38 and HellaSwag -0.01
points: inside the benchmarks' standard errors (MMLU +/-0.34, GSM8K +/-1.1), and not zero, so "reruns
give identical numbers" holds only for an unchanged vLLM configuration.
**Also found, then corrected:** I first read the grounded_acc drop (0.852 -> 0.657) as a CPT behaviour
change (more citations, some citation-only or citing ids no passage has, which rule 8 scores wrong).
Wrong: `base-8b-hf`, the same base weights through the HF path, scores exactly 0.657 too, with the
same citation pattern (cite_valid 0.185). The drop is the implementation, not training. (A reading
taken then, "`cpt-8b` is qa_acc +5.4 points against `base-8b-hf`", compared two runs on the broken
path and did not survive the YaRN fix below: on the fixed path qa_acc moves +0.8 and +0.0 points
across two seeds.)

## 2026-09-28: GPU cost cuts for the rest of Stage 2 (user decision)
**Context:** ~9 H100-hours (~$36 at an assumed $3.95/h, not checked against Modal's billing) spent by
the end of the main run's evaluation, ~1-1.5 of them on attempts that failed on bugs since fixed;
A and B still to run.
**Chose:**
- **Latency is opt-in** (`modal_train.py --steps ...,latency`), no longer part of `eval`: it measures the
  architecture and the serving setup, not the weights (`cpt-8b` and base-8b decode at the same 6.6 ms
  per token), so it is measured once per deployed checkpoint (Stage 6), not per ablation. A and B get
  perplexity, lm-eval and the KPI eval only. (Stopping seed 1's latency run came too late: it had
  finished.)
- **No 1-GPU FSDP control for C.** C's result stands without it: same batches, per-step loss within
  0.04% (median), 2.23x the tokens/s. That the speed-up is more than 2x is reported as an effect of
  the FSDP code path (its own bf16 casting and activation checkpointing), not isolated by a run.
- lm-eval stays for A and B: MMLU is half of each ablation's pre-registered question (forgetting).

## 2026-09-28: Root cause of the vLLM path gap: YaRN attention scaling; merged configs set apply_yarn_scaling=false
**Context:** the same base weights scored grounded_acc 0.852 through vLLM's Mistral-native path and 0.657
through its HF path (`base-8b-hf`). Hypothesis: the two configs express YaRN differently. Native
`params.json`: `yarn: {factor: 16, apply_scale: false}`. HF `config.json`: `rope_type: yarn,
factor: 16, mscale: 1.0, mscale_all_dim: 1.0`, which transformers resolves to an attention factor of
1 (the mscale ratio). vLLM 0.29's `get_rope` "yarn" branch keeps only extrapolation_factor,
attn_factor, beta_fast, beta_slow, apply_yarn_scaling and truncate, so it drops mscale /
mscale_all_dim, and `YaRNScalingRotaryEmbedding` multiplies cos and sin by `yarn_get_mscale(16)` =
1.277 (`apply_yarn_scaling` defaults to True): every attention logit scaled by ~1.63.
**Test** (`eval/vllm_ppl.py`: vLLM prompt-logprob perplexity on the trainer's 49 val windows, 200,655
tokens; transformers' `ppl_val_slice` is 6.8938):

| path | nats per token | perplexity |
|---|---|---|
| vLLM native (`config_format=mistral`) | 1.93063 | 6.8939 |
| vLLM HF, config as shipped | 1.97772 | 7.2262 (+4.8%) |
| vLLM HF, `apply_yarn_scaling: false` | 1.93063 | 6.8939 |

**Chose:** `train/merge.py` writes `"apply_yarn_scaling": false` into every merged or re-saved
checkpoint's `text_config.rope_parameters` (transformers ignores the key, with a notice). With it the
HF path is the native path to five decimals, so every checkpoint (base re-saved as `base-8b-hf`, all
merges) is evaluated on the model Mistral defines, through one implementation. The Stage 0 native
rows were therefore right all along; the HF-path numbers taken before this fix (cpt-8b's first KPI
and lm-eval, base-8b-hf's and seed 1's first evals) are superseded and re-run.
**The bug's size on the same base weights** (native path vs the unpatched HF path; lm-eval with the
same flags, `base-8b-mm0` vs `results/lm_eval/base-8b-hf-unpatched`): MMLU 0.7685 -> 0.7377 (-3.1
points; humanities -4.2, STEM -3.6, social -2.1, other -1.7), GSM8K strict 0.7983 -> 0.7369 (-6.1),
HellaSwag 0.8005 -> 0.7956 (-0.5); KPI grounded_acc 0.852 -> 0.657, halluc_rate 0.882 -> 0.947. So
`cpt-8b`'s first readings (MMLU -3.7, GSM8K -6.7 against native base-8b) were mostly the bug. The
unpatched outputs are kept in `results/{runs,lm_eval}/*-unpatched/` as the evidence; their interim
`table.md` rows were removed.

## 2026-09-28: Stage 2 main run verdict on the fixed path, with the measured seed floors
**Context:** the YaRN fix restored `base-8b-hf` to the native path: MMLU 0.767 vs 0.768, GSM8K 0.793
vs 0.794, HellaSwag 0.801 vs 0.801, grounded_acc 0.843 vs 0.852 (80 of its 524 KPI outputs differed
from the native path's at all). `cpt-8b` and `cpt-8b-seed1` were re-merged with the fix and
re-evaluated; the seed floor is |seed1 - main| per metric (one seed pair: a rough floor).
**Seed floors:** perplexity: domain val 0.02%, general val 0.23%, train slice 0.25%. lm-eval (points):
MMLU 0.1 (STEM 0.2, humanities 0.0, social 0.4, other 0.5), GSM8K 0.1, HellaSwag 0.1. KPI (points):
qa_acc 0.8, grounded_acc 3.7, cite_valid 4.6, cite_supported 3.7, vocab_recall 1.0, halluc_rate 2.6.
The KPI floors are several points (76-210 items per task): at this eval size a CPT effect on the
KPIs smaller than ~4 points can't be seen.
**Verdict against `base-8b-hf`** (`cpt-8b` / `cpt-8b-seed1`):
- Domain val perplexity -2.33% / -2.35% (document intervals exclude 0; floor 0.02%): real, and far
  under the pre-registered 20% ("barely moved"; the LR-up fix didn't help, see above).
- General val perplexity +0.40% / +0.17%: real, small forgetting, well inside the +3% bound.
- Train/val gap growth +7.2 points (under the 10-point bound).
- MMLU -0.3 / -0.2 points (floor 0.1): at most a trace of forgetting, below the expected -0.5 to -2;
  GSM8K -0.8 / -0.7 (floor 0.1, benchmark SE ~1.1); HellaSwag unchanged.
- KPIs: qa_acc +0.8 / +0.0, vocab_recall +0.5 / -0.5, grounded_acc -1.0 / -4.7, halluc_rate
  +1.3 / +3.9: all inside the seed floor. No detectable KPI change.
**Reading:** 20M tokens of LoRA CPT on public federal documents measurably lowers domain perplexity
and barely touches general ability, but the gain is too small to show up in closed-book QA or the
other KPIs. Consistent with prior exposure: the model learns the documents it reads (train slice
-8.3%) with little transfer to unseen ones, and doubling the LR only memorises more.

## 2026-09-28: Ablation B verdict: LoRA stays the default (pre-registered rule applied)
**Rule:** LoRA stays unless full's extra domain gain over LoRA beats the seed floor and is larger than
its extra general loss.
**Result** (`cpt-8b-full`, 8B full-parameter on 2 x H100, same windows and batch as `cpt-8b`; deltas
vs `base-8b-hf`, LoRA's in brackets as `cpt-8b` / `cpt-8b-seed1`):
- Domain val perplexity -2.15% [-2.33% / -2.35%]. Full vs LoRA: +0.18%, interval over documents
  [-0.61, +0.81]: no extra domain gain. (On the trainer's 49-window slice full looked slightly ahead,
  1.902 vs 1.906 eval loss; the full val set disagrees.)
- Train slice -22.1% [-8.3% / -8.1%]: it memorises the documents it reads almost 3x harder;
  train/val gap growth +28.6 points, past the pre-registered ~25-point "memorising" line.
- General val perplexity +1.19% [+0.40% / +0.17%]; full vs LoRA +0.78% [+0.58, +0.97], past the
  0.23% floor. MMLU -0.5 [-0.3 / -0.2] (humanities -1.2), GSM8K -3.0 [-0.8 / -0.7], HellaSwag -0.3.
- KPIs, where LoRA showed nothing: qa_acc +3.1 points, vocab_recall +3.8, grounded_acc +4.6,
  halluc_rate +2.6 (worse), past the rough single-pair floors (0.8 / 1.0 / 3.7 / 2.6). Every KPI
  item comes from a training document (the eval pool is in train by design), so this is the
  memorisation showing: knowledge of the documents read, not transfer to unseen ones.
- Cost: 0.91 GPU-h on 2 GPUs vs LoRA's 0.95 on one: the FSDP path's per-GPU speed-up (see C)
  covers the extra compute of full-parameter steps.
**Chose:** LoRA stays the default. Stated for the write-up: on the same 8B model and tokens,
full-parameter CPT learned the training documents far better (train perplexity -22% vs -8%, closed-
book QA +3 points) with no gain on unseen documents and about 3x the forgetting (general perplexity
+1.2% vs +0.4%, GSM8K -3.0 vs -0.8), at the same GPU-hours: "LoRA learns less and forgets less"
(Biderman et al. 2024). **Revisit if:** Stage 3 needs the in-document knowledge more than general
ability (the KPI gain is real but small and one seed deep).

## 2026-09-28: Ablation A verdict: replay adopted by the rule, on thin evidence (pre-registered rule applied)
**Rule:** replay is adopted for Stages 3+ only if `ppl_general_val` or MMLU beats the main run's by more
than the seed floor, with domain val within 3% of the main run's.
**Result** (`cpt-8b-replay10`: `cpt.yaml` + 10% FineWeb-Edu, 164 steps, 1.04 GPU-h; vs `base-8b-hf`,
main run in brackets as `cpt-8b` / `cpt-8b-seed1`):
- Domain val: identical to the main run (0.00%, interval over documents [-0.04, +0.04]). Replay costs
  nothing on the domain; final eval loss 1.9059 vs 1.9062.
- MMLU -0.1 [-0.3 / -0.2]: +0.2 over `cpt-8b` against a 0.1 floor; humanities +0.3 [-0.5 / -0.5];
  GSM8K -0.2 [-0.8 / -0.7]; HellaSwag -0.1. Replay erases the main run's small dip, consistently in
  the right direction, but inside the benchmarks' own standard errors (MMLU +/-0.34, GSM8K +/-1.1).
- `ppl_general_val` -2.24% [+0.40% / +0.17%]. **Not evidence of protection:** general_val is FineWeb-Edu,
  the replay slice's own distribution (a different shard), so this is in-distribution training. The
  pre-registered rule should have excluded it for A; that is a flaw of the pre-registration, stated
  here rather than used.
- KPIs, a cost the rule didn't anticipate: grounded_acc -6.5 [-1.0 / -4.7], cite_valid -6.4 [+2.8 /
  -1.8], past the rough floors (3.7 / 4.6); qa_acc +0.0, vocab -0.5. The web text seems to dilute the
  citation format.
**Chose:** by the letter of the rule, replay is adopted for Stages 3+ (MMLU beats the main run's by
more than the floor; domain within 3%). Stated plainly: the evidence is thin (one seed, deltas inside
benchmark noise), the general-val criterion is void for replay, and the citation KPIs went the other
way. It is cheap (+10% tokens, zero domain cost), which is what makes adopting it on weak evidence
acceptable. **Revisit if:** Stage 3's SFT doesn't restore the citation format on a replay-trained
checkpoint, or a second seed of A doesn't reproduce the MMLU recovery.

## 2026-09-28: Prior-exposure check: inconclusive, no overall recency trend
**Context:** the working explanation for Stage 2's small domain gain is that the base model already saw
these public documents in pretraining. If it saw the older ones and not the newer ones, newer
documents should be harder for it, within a publisher.
**Test** (`eval/exposure_check.py`, no GPU): base perplexity for 58 documents (the 12 val documents
exactly; 46 train-slice windows one document fills to >= 90%), 51 of them with a PDF creation year,
relative to the publisher's median. Spearman(year, relative perplexity) = +0.07, permutation p =
0.60: no overall trend. By period, median relative perplexity: 1990-2014 -2.5% (n=18), 2015-2019
+2.7% (19), 2020-2023 -4.2% (10), 2024-2026 +6.6% (4). The two newest documents, both val (NASA-STD-
5002B dated 2026, USACE EM 1110-2-2610 of March 2025), are +15.1% and +10.5% harder than their
publishers' medians and the 2025 USACE manual gained most under CPT; the other two recent ones are
ordinary (+2.6%, -1.4%).
**Reading:** neither confirmed nor refuted. The direction for the newest documents is the one
exposure predicts, but n=4, PDF creation dates are not always publication dates, and the corpus
holds almost no post-cutoff documents, so no amount of re-scoring this corpus can settle it (a
full per-document GPU run would add old documents, not new ones). The alternative, that 20M tokens
of LoRA CPT is simply too little signal, stays open. **Revisit if:** the corpus gains documents
published after the base model's release, which would make a clean exposed-vs-unseen comparison.

## 2026-09-28: Latency re-measured on the fixed path; CPT weakened stopping
**Context:** `results/bench/` for `cpt-8b`, `cpt-8b-seed1` and `base-8b-hf` had been measured before the
YaRN fix. Re-run for `cpt-8b` and `base-8b-hf` on the fixed checkpoints; seed 1's file deleted (same
architecture and serving path as `cpt-8b`, so redundant).
**Result** (concurrency 1, p50; tok/s at concurrency 32): base-8b (native) TTFT 17.7 ms, ITL 6.6 ms,
E2E 187 ms, 1,922 tok/s; base-8b-hf (fixed) 15.3 / 6.6 / 180 ms / 1,967: parity with the native path
(the stale run's 427 ms E2E was the broken model writing longer answers). cpt-8b (fixed) 16.5 / 6.9 /
1,753 ms / 2,166.
**Finding:** per-token speed is the same, but `cpt-8b` runs to the 256-token cap where the base stops
after ~28 tokens, on the fixed path too, so it is a CPT effect: the corpus holds one EOS per document
(234 in 19.4M tokens, ~83k tokens apart), and CPT weakens the model's stopping. SFT's short answers
should restore it; Stage 3 checks output length.

## 2026-09-28: The replay rule fired on noise; kept anyway (critique of the pre-registration)
**Context:** ablation A's rule adopted replay because its MMLU beat the main run's by more than the
seed floor: +0.2 points (0.766 vs 0.764) against a floor of 0.1 (|cpt-8b - cpt-8b-seed1|).
**The flaw:** one seed pair understates MMLU's sampling error. The benchmark's own standard error is
~0.34 points (14,042 items), so a +0.2 difference is noise however small the seed gap happens to
be. The rule should have compared against max(seed gap, benchmark standard error), and the same
applies to GSM8K (SE ~1.1 points on 1,319 items) and to the KPIs (1-5 point seed swings on
76-210-item tasks). The general-val criterion was void for A as well (in-distribution, see the A
verdict).
**Chose:** keep the verdict: Stage 3 starts from `cpt-8b-replay10`. Following a conservative rule
you'd now write differently beats amending it after seeing the data, and the choice costs
nothing: replay matches the main run on every metric outside the noise (domain val identical,
MMLU / GSM8K / HellaSwag within their standard errors), at +10% training tokens.
**For later stages:** a pre-registered comparison reads a difference against max(seed gap, the
metric's own standard error), with bootstrap intervals for perplexity. The KPIs are Stage 3+
metrics: the base model doesn't follow instructions, so Stage 2 can't move them measurably.

## 2026-09-28: Memorisation probe: the base doesn't reproduce the corpus; CPT's gain stops at the corpus's own series
**Context:** the prior-exposure check above was inconclusive and said to revisit with documents
published after the base model's release (`-2512`, December 2025). The user asked for a
memorisation probe.
**Documents:** 234 train, 12 val, and 13 post-cutoff public-domain federal reports:
- NIST: 6 (TN 2371, 2377, 2281, 2374; GCR 26-072, 26-073);
- USACE ERDC: 5;
- FHWA: 2.

All were published February–August 2026 (title pages and catalogue records) and are listed with
sha256 in `eval/exposure_sources.csv`. Each passed a rule-1 check: the only copyright hits are
NIST's standard policy link. They went through the corpus's own cleaning (extract, then the paragraph
filter).
**Test** (`eval/memorization.py`, ~$0.5 on Modal):
- 16 spans per document from its middle 80%, each a 64-token prompt and a 64-token greedy
  continuation, scored by verbatim prefix (tokens reproduced before the first miss).
- Perplexity over each document's first two 4,096-token windows.
- Run on `base-8b-hf` and, as a positive control, on `cpt-8b-replay10`, which read the train
  documents exactly once.

**Results:**
- **Base reproduces nothing.** 6 of 3,744 train spans reach 32+ verbatim tokens; 0 val and 0
  post-cutoff spans do. All six are in-document patterns, not recall:
  - incrementing list items (`Table D-11` -> `D-12`, `D5092.033f` -> `g`);
  - figure captions repeated with one value changed;
  - a sentence echoing the one before it.

  Mean verbatim prefix: corpus 1.30 tokens, post-cutoff 1.24; the difference is +0.05 [-0.26, +0.34].
- **Base perplexity:** corpus documents are no easier than post-cutoff ones. The geometric mean is
  +3.0% [-8.5, +17.0], with unmatched document types; the sign varies by publisher.
- **Control, per-document change (`cpt-8b-replay10` - base):**

  | Documents | Verbatim prefix | Perplexity |
  |---|---|---|
  | Train | +0.13 [+0.08, +0.18] tokens | -8.3% [-8.6, -7.9] |
  | Val | +0.07 [-0.03, +0.16] | -4.2% [-5.5, -3.0] |
  | Post-cutoff | +0.07 [-0.03, +0.17] | -0.4% [-0.8, 0.0] |

  Every val document improved (-1.6% to -8.7%). The post-cutoff documents ranged from +1.1% to
  -1.5%. Val minus post-cutoff: -3.8% [-5.1, -2.5].

**Reading:**
1. **Heavy prior exposure is ruled out.** A model that had memorised these documents would continue
   them verbatim, and this one doesn't. Light exposure is not ruled out. One epoch of our own CPT
   moves the verbatim prefix by only 0.13 tokens, below the ±0.3-token resolution of a 245-vs-13
   comparison, so the probe can't separate "seen once in pretraining" from "never seen". The prior-exposure
   explanation for the small Stage 2 gain loses its direct support without being refuted.
2. **New, and more important: CPT's gain doesn't reach new documents in the domain.** On the same
   measurement and the same windows, perplexity fell 8.3% on the documents read, 4.2% on held-out
   corpus documents and 0.4% on documents published afterwards. The val split is document-level, but
   val documents are siblings of train documents: USACE EMs, FEMA P-series, NIST GCR 917 briefs, FHWA
   HIF reports. They share structure, boilerplate and phrasing with them. So `ppl_domain_val`
   measures generalisation within those series, not domain transfer, and it overstates what CPT
   gives a new report.

**Caveats:**
- **Few documents, different genre.** There are 13 documents, mostly research reports, technical
  notes and workshop reports, while the corpus is mostly manuals and guides.
- **Topic.** Some post-cutoff topics sit at the edge of the domain (pavement, embodied carbon,
  procurement). The near-core ones don't move either: ERDC/GSL SR-26-1 (bridge load rating) -0.5%,
  ERDC/ITL TR-26-1 (corroded steel beams) -0.4%.
- **Window choice.** Perplexity covers each document's first two windows only, which include front
  matter shared within a series. That favours val: its first-window change is -4.2% against -2.3%
  over whole documents (`perplexity.py`). Even the whole-document val gain is about 6x the
  post-cutoff one.

**Sources:** highways.dot.gov (FHWA's current publications) answers scripts with an Akamai 403.
ROSA P serves PDFs to curl's default User-Agent but returns 403 to a browser one, which is the one
`data/scripts/common.py` sends.

**Consequences:**
- The README's "why so small" finding now rests on this entry.
- A post-cutoff perplexity column is the honest measure of domain transfer for later stages.
- **Revisit if:** a stage claims domain transfer. Then measure whole-document perplexity on the
  post-cutoff set and grow the set; FHWA's 2026 reports need a browser download.

## 2026-10-04: Stage 2 conclusions; the -20% target was set for the wrong data scale (user decision)
**Bottom line (user's reading, adopted):** at 20M tokens and one epoch, CPT learns the documents it
reads (-8%) and almost nothing that transfers to unseen ones (-0.4% on 2026 documents). A higher
learning rate and full-parameter training learn the documents harder, with 2-4x the forgetting and
no held-out gain. LoRA with 10% replay is the carried-forward checkpoint, chosen by a pre-registered
rule that fired at the noise edge (recorded above). The eval harness found a serving bug larger than
any training effect.

**Against the pre-registered rules:**

| rule | result | verdict |
|---|---|---|
| domain val down at least 20% | -2.3% (both seeds, within 0.02%) | failed |
| gap growth under ~10 points | base gap 11.2%; cpt-8b +7.2, seed 1 +6.9, replay +7.8, lr2x +12.1, full +28.6 | passes for LoRA at 1e-4; fails for lr2x; full is past the 25-point memorising line |
| general val up under 3% | +0.17% to +1.19% | passed everywhere |

**Reframe:**
- Held-out perplexity measures domain transfer, which continued pre-training produces at the scale
  of billions of tokens.
- With a 20M-token client corpus the goal is knowledge of those documents, measured by closed-book
  QA after SFT. It needs repetition or augmentation to stick.
- So the -20% target was a metric pre-registered for the wrong data scale, not a target missed by a
  bad run.

**Corrections to the earlier write-up:**
- **Replay's grounded (-6.5) and cite_valid (-6.4) "cost" is probably not real.** Both are measured
  on a base model's citation formatting, which SFT overwrites. One seed alone moved grounded by 4.6
  points.
- **The 2-GPU speed-up is a code-path effect.** Ablation C's 2.23x is +11% per GPU at matched
  micro-batch (4) and per-layer non-reentrant checkpointing. The 1-GPU run peaked at 51 GB of 80, so
  it wasn't memory-bound. The difference is the FSDP2 code path (bf16 parameter casting, its own
  checkpoint wrapper), not scaling, and no run isolates it.
- **Full-parameter vs LoRA cost.** Full-parameter on 2 GPUs matched LoRA's GPU-hours despite about
  1.33x the FLOPs per token: roughly 43% against 30% of H100 bf16 peak. LoRA's FLOP saving doesn't
  turn into speed.
- **The probe's perplexities aren't comparable to the main table.** They are per-document medians
  over each document's first two windows; only ratios within the probe are meaningful.

**Stage 3 eval (carried lesson):**
- **Seen and unseen halves.** Report `domain_qa` and `vocab` in two halves: items whose source
  chunk fed an SFT example, and items whose chunk did not. Hold a deliberate share of the eval's
  source chunks out of SFT synthesis so the unseen half exists. A single pooled number is
  uninterpretable, as Stage 2's held-out perplexity showed.
- **The bar is `instruct-8b`** (grounded 0.898, cite_valid 0.833, cite_supported 0.787, halluc
  0.013, vocab 0.786, qa 0.115). Match it on grounding and beat it on both qa halves. Read
  halluc_rate next to grounded_acc, so that abstention isn't refusal.
- **Latency.** `cpt-8b`'s end-to-end latency (runs to the cap: one EOS per manual) is CPT's known
  side effect. It is not a serving result, and deployment latency is measured on the SFT'd
  checkpoint.

## 2026-10-04: qa_acc audit: binary stays, one parser fix, split columns, domain_qa grows (user decision)
**Context:** qa_acc sits at 12-19% for every model. Is the task hard, or is the scorer losing
correct answers? An outside review suggested a third of the score was the scorer: brittle text
matching, a first-number rule that skips values, and a chat model cut off by the newline stop.

**Audit** (saved generations, no GPU):
- **Hard by construction.** 98 of 130 items are never answered by any of the six models. The 10
  everyone answers are general knowledge (ASCE 7's ρ = 1.0 and 1.5 factors, the Bruun rule,
  SSPC-SP 10).
- **The scorer loses almost nothing:**
  - **Hand read of `instruct-8b`'s misses:** all 35 text misses and 30 random numeric ones. 1 of
    the 65 was correct.
  - **First-number rule:** across three models, the gold number never appears later in an answer
    line.
  - **No truncation:** `instruct-8b`'s chat replies were never cut off. Chat runs drop the newline
    stop and get 64 tokens, and all 130 replies are one bare value (median 7 characters).
  - **`instruct-8b` below base:** that is 4 items, one standard error, and its misses are wrong
    values.
- **The one correct miss was a parser bug.** `_NUM` read the hyphen in "FEMA P-361" as a minus
  sign (-361 against a gold "FEMA 361").

**Chose:**
- **Keep pass/fail.** TriviaQA and NQ score short answers the same way; a wrong load factor is
  wrong.
- **Fix the parser.** A hyphen glued to a letter or digit is no longer a sign. The rescore of
  every row made no judge calls. Effect: `instruct-8b` qa_acc 0.115 -> 0.123 (so the Stage 3 bar is
  0.123, not 0.115), `cpt-8b-full` 0.177 -> 0.185, every other cell unchanged.
- **No judged qa column.** It would recover about one item per model, so it isn't worth its own
  error (rule 8). Revisit if the first SFT checkpoint answers in sentences: rerun this audit on it.
- **No chat-only prompt suffix.** The header already asks for the bare value and `instruct-8b`
  complies, and any prompt change changes the benchmark.
- **New columns:**
  - `qa_num` / `qa_text`: by answer type, 93 numeric / 37 text.
  - `qa_seen` / `qa_unseen`, `vocab_seen` / `vocab_unseen`: by
    `eval/tasks/sft_seen_chunks.txt`. That list comes from `eval/sft_split.py`: a source chunk is
    seen when the first byte of sha256("sft-seen:" + id) is even, so an item never changes half
    as tasks grow. Grounded/adversarial context chunks and few-shot chunks are never seen.
  - The split is 65/65 QA items and 101/109 vocab items (v1; on the v2 set of 325 it is 167/158,
    vocab unchanged). Stage 3 SFT synthesis may draw on the seen chunks only (rule 10).
  - Before Stage 3 the halves are a null check. For example `base-8b-hf` scores 0.123 seen vs 0.169
    unseen: 8 vs 11 items of 65, inside the noise.
- **Rescore guard.** `run_eval.py --rescore` refuses saved generations that miss items now in the
  task files; `--allow-partial` overrides. Otherwise a grown task would put a different item set
  under the same column name.
- **`eval/answer_logprob.py` (+ Modal `answer_logprob`), written but not run (no GPU).** It gives
  the gold answer's log-probability per item, a continuous companion to qa_acc. Prompt tokens are a
  prefix of prompt + answer for all 130 items, in base and chat format.
- **Grow domain_qa** with a second supplement (`make_tasks.py --n-qa-extra2 1300`): 1,300 unused
  numeric-rich chunks under the same `--per-doc 6` cap, sampled after every other take, ids from
  1001. With extras set to 0 the frozen files rebuild byte-identically. Target 400-500 items, for
  a standard error under 2 points and halves big enough to read. Review follows
  `notes/eval_review_rubric.md`, the rubric of 2026-09-27 written down.

**Consequence:** every compared checkpoint has to regenerate domain_qa on the grown set (GPU, needs
approval):
- `base-8b`, `instruct-8b`, `base-8b-hf`, `cpt-8b` and `cpt-8b-replay10` can be run as they are;
- `cpt-8b-seed1` needs re-merging from its adapter;
- `cpt-8b-full`'s weights are gone, so its row stays on the 130-item set.

**Same day, before the grown set is scored:**
- **Audit extended to the base models' text misses.** `base-8b-hf`'s 35 and `cpt-8b-full`'s 34.
  In total 169 misses were read by hand (`instruct-8b` 35 text + 30 numeric, plus these 69), and 2
  were correct answers scored wrong:
  - "FEMA P-361", the hyphen bug above;
  - "10" against the gold "10:1" (qa-0542, a maximum aspect ratio).

  `qa_correct` now scores an `N:1` ratio gold by its value. Effect: `base-8b`, `cpt-8b`,
  `cpt-8b-seed1`, `base-8b-hf` and `cpt-8b-replay10` gain that one item (qa_acc +0.8, qa_text
  +2.7, qa_unseen +1.5). `instruct-8b`'s "10:1" already matched and `cpt-8b-full` answered 3. No
  judge calls, no other cell moved.

  So the judged qa column stays out. 102 of 104 text misses are wrong document numbers, article
  numbers or terms, which is knowledge, not exact-match brittleness. `cpt-8b-full`'s qa gain over
  `base-8b-hf` is +3.1 against a noise of 3.2: at the edge, not past it.
- **`results/table_v1.md` frozen.** The 130-item table behind the Stage 2 write-up, as of the
  parser fix and before the ratio rule.
- **`results/table.md` carries an item-count line.** It sits above the header, and
  `run_eval.append_table` refuses a row whose task sizes differ from the line, or that scored part
  of a task. `--allow-partial` now leaves a stale task's columns blank instead of scoring the
  subset; this is how `cpt-8b-full` gets blank qa cells in the grown table. A `--limit` run no
  longer writes a table row.
- **`--tasks` (`run_eval.py`, Modal `kpi_eval`).** It regenerates only the named tasks and keeps
  the other rows of `generations.jsonl` byte for byte, keyed by (task, id).
  `generations_meta.json` records per task the date, model, chat flag, config format, vLLM version,
  item count and prompt hash; older tasks are marked as generated before provenance existed.
  `--config-format auto` exists only to extend Stage 0's native-path hub runs (`base-8b`,
  `instruct-8b`).
- **Gold-answer log-probability moved into the generation engine** (`eval/gold_lp.py`, one model
  load per checkpoint). It is saved per item and gives `gold_lp`, `gold_lp_seen` and
  `gold_lp_unseen` (mean nats per item, higher is better), with the median in `metrics.json` since
  long gold strings can dominate a mean. The prompt/answer token boundary is checked for every item
  in both formats, and failures are listed and left out, never summed. Instruct is scored in chat
  format, so its value is not comparable to base-format rows.
- **`ppl_postcutoff`** (`eval/perplexity.py`, `--only postcutoff` adds it to an existing json; Modal
  `perplexity --only`). The 13 2026 reports as whole documents (`data/exposure/postcutoff.jsonl`,
  82 windows), on the same footing as `ppl_domain_val`; `ppl_compare.py` bootstraps it.
- **Judge cache.** A `--results-dir` outside the repo's `results/` must name `--judge-cache`: a
  scratch run re-judged 50 items today.
- **Rules (user decision):**
  - No checkpoint with a row in a results table is deleted until that stage's write-up is frozen;
    16 GB on the volume is cheaper than a hole in the table.
  - Adapters are never deleted: merged LoRA checkpoints are reproducible from them. That is why
    `cpt-8b-seed1` can be re-merged and `cpt-8b-full` (full-parameter, no adapter) cannot.
- **`cpt-8b-seed1` comes back by re-merging its adapter** with the same CPU `merge` function as
  the first time. Before the row is used as the noise floor again, perplexity has to reproduce
  domain val 6.718 (`results/ppl/cpt-8b-seed1.json`). If it doesn't, the row is labelled
  re-merged.
- **Launch plan (each launch asked for first; none before the grown set is reviewed, rebuilds byte
  for byte and is committed):**
  1. First, the three rows Stage 3 needs, each a `domain_qa` regeneration with `gold_lp`:
     `base-8b-hf` and `cpt-8b-replay10` (with `ppl_postcutoff`), and `instruct-8b`
     (`--config-format auto`, the path its other tasks were generated on).
  2. Then the history rows: `cpt-8b`, and `cpt-8b-seed1` after its re-merge and perplexity check.
  3. `cpt-8b-full` and the Stage 0 native `base-8b` keep blank qa cells via `--allow-partial`.

## 2026-10-04: Layout locators removed, answer kinds tagged, identifiers capped at 20%; task versions (user decision)
**Context:** a closed-book item whose answer is a page, table, figure or equation number tests a
document's layout, not its content. That layout changes between editions, and the page can leak
from metadata. The generation prompt for domain_qa shows the passage as `Passage
(slug:p12:c0):`, so the generator sees the page number. Separately, qa_text turned out to be
identifier recall (document and article numbers), the hardest and least useful kind of closed-book
knowledge, so how much of the set it takes up is a design choice.

**Measured first:**
- **Locators are rare.** `qa_rules.is_locator` flags none of the 130 kept items, the 3 few-shot
  items or the 476 domain_qa rejects. Across all 4,317 QA candidates ever generated (the
  2026-09-27 pools plus the in-flight second supplement) it flags 39 (0.9%), every one a real
  locator: table, figure, equation and plate numbers, page counts, citation page numbers.
- **No metadata leak.** None of those answers is the chunk's own page; the page items come from
  reference-list text.
- **Tuning.** The filter as first proposed also flagged named equations ("Manning's equation"),
  "plate buckling" questions and the standard "PS2-10", and missed "Equation B-28". The rules were
  tuned on these lists and are pinned by tests (`tests/test_qa_rules.py`: 10 locators, 10
  non-locators including "According to Table 3.4.1-1, what is the load factor").

**Chose:**
- **`eval/qa_rules.py`.** It holds `is_locator` and `answer_kind` (number, identifier, term,
  other; identifier wins over number, so "FEMA 361" is an identifier; "10:1" is a number).
- **`make_tasks.py --task-version`:**
  - `1` rebuilds the 2026-09-27 eval byte for byte (no second supplement, filter, kinds or cap;
    `tests/test_make_tasks_v1.py`).
  - `2` (default) is the grown set. After rejects, and removing items only, it tags
    `answer_kind` (reviewer corrections in `eval/tasks/answer_kinds.jsonl`), removes locators
    (`locators.jsonl`) and holds identifiers to 20% of the final set, keeping them in id order
    (`held_back.jsonl`, not deleted). It applies to every item, the old 130 included.
  - On the 130 alone, v2 removes 0 locators and holds back 0 identifiers (26 of 130, exactly
    20%).
- **The generator prompt is unchanged.** It keeps the chunk id, and the locator rule doesn't run
  before verification. Changing the prompt changes every cached generation, which would
  regenerate the frozen 130 and the 2,525 in-flight candidates and force re-review of all of them.
  Filtering before verification would save about 50 of about 4,000 remaining API calls. The rule
  runs at assembly instead, where it holds whatever the prompt does. A future from-scratch build
  should drop the chunk id from the `gen_qa` prompt and ask for no locators.
- **qa_text split by kind.** It becomes `qa_ident` and `qa_term` (term + other), with `qa_num`
  now meaning kind number. On the 130 items (91 / 26 / 13): base-8b-hf 0.198 / 0.038 / 0.077, and
  the CPT runs 0.176-0.220 / 0.077-0.115 / 0.077. Expect qa_ident to move slowest under SFT: an
  arbitrary string needs many exposures, and a retriever supplies it anyway.
- **Table line.** `table.md`'s first line now also carries the per-kind counts, so a row is
  tabled only against the same mix.
- **Rubric.** `notes/eval_review_rubric.md` gains reject clause 10 (layout locator), the
  answer-kind tag with one example per kind, the cap, and the blind sample of 40 checking tags
  as well as verdicts (n checked, n overturned, n tags corrected).
- **API pacing (`make_tasks.Pacer`).** The key's limit is 30 requests a minute
  (`x-ratelimit-limit-req-minute`; tokens 800k a minute, a call 0.7 s), not concurrency.
  - **Before:** burst-then-sleep spent ~3/4 of worker time asleep.
  - **First fix:** an adaptive pacer hovered just above the limit and still drew a 429 on about one
    call in five.
  - **Now:** `--rpm 30` paces at 2.06 s per call, with 8 workers. That gives 28-29 calls a minute
    and zero 429s, the most this key allows one call per item.
  - **Checking several items per request was declined (user decision).** The supplement stays
    filtered the way the frozen 130 were, one check per item.

## 2026-10-04: domain_qa v2 frozen: 325 items (130 + 195), reviewed; numeric ids and years scored exactly
**Generation:** `make_tasks.py` second supplement, 1,300 chunks (pacing at the key's 30 requests a
minute, no failed calls):

| Step | Items |
|---|---|
| Candidates | 2,525 |
| Pass the pre-filter | 2,138 |
| Pass the closed-book check | 1,251 |
| Pass answer verification | 1,191 |
| New after dedup | 1,179 |
| Locators removed by rule | 5 |
| To review | 1,174 |

**Review** (`notes/eval_review_rubric.md`, ten agents with the same brief, 112-118 items each):
- **Agents:** 341 kept (29%), 833 rejected, 123 keeps flagged. Reject clauses:

  | Clause | Rejects |
  |---|---|
  | 3, a correct answer would score wrong | 270 |
  | 5, general knowledge | 218 |
  | 2, not the only answer | 79 |
  | 6, trivia | 58 |
  | 1, wrong gold | 46 |
  | 7, one example's value | 32 |
  | 8, misframed | 23 |
  | 4, a wrong answer would score right | 21 |
  | 10, locator | 14 |
  | 9, not standalone | 9 |
  | mixed | the rest |

- **Lead re-check of all 123 flags:** 6 kept, 117 rejected.
  - **One rule across batches:** core AASHTO LRFD values a bridge engineer knows (7.0 in. deck,
    Service II 1.30, Strength IV 1.5, eta 1.05 / 0.95, 600-kip collision, Service II for
    deflection, the 70 ksi compact limit) are general code knowledge, as 0.85 f'c is.
  - **AASHTO Section 5 articles** (renumbered in the 8th edition) cited without an edition: none
    were kept.
- **Lead duplicate pass** over all kept items, the old 130 and few-shot included: 5 later items
  were rejected as the same fact (qa-1948, -1918, -1235, -2055, -2000).
- **Blind check:** the lead reviewed 40 items, sampled with seed 20261004, before any agent
  verdict existed.
  - Agreement was 35 of 40, and the tags matched on all 5 shared keeps.
  - All 5 disagreements were lead keeps the agents rejected (silica fume, JSC-28918, eta_R = 1.000,
    blockholing, 100% uplift). On re-reading the agents were right each time.
  - **Recorded:** 40 checked, 0 agent verdicts overturned, 0 tags corrected. As on 2026-09-27, the
    agents are the stricter side.
- **Kept:** 219 of 1,174 (19%). Rejects are appended to `eval/tasks/rejects.jsonl` (955 entries,
  reviewed "2026-10-04 second-supplement review"; none matches a kept question). Tag corrections
  are in `eval/tasks/answer_kinds.jsonl` (9).

**A scoring flaw the review found:** a numeric gold is matched within 2%, which accepts a
neighbouring id or year: 1917 passes 1928 and 1936, FEMA P-2055 passes P-2090, and an ASTM number
passes its neighbours. Version 2 sets `tolerance` 0 on numeric items that are identifiers or bare
years (16 items, 3 of them in the old 130). The scorer is unchanged.

**Built (`make_tasks.py`, default `--task-version 2`):**
- **domain_qa: 325 items.** That is the 130 old items plus 219 new ones, less the 24 newest
  identifiers held back by the 20% cap (`held_back.jsonl`; all 24 are new items). 5 locators are
  in `locators.jsonl`.
- **Kinds:** number 222, identifier 65 (20%), term 37, other 1.
- **By publisher:** USACE 122, FEMA 98, FHWA 64, NIST 31, NASA 10.
- **Halves:** seen 167, unseen 158.
- **Unchanged:** grounded, vocab, adversarial and few-shot are byte-identical.
- **`eval_chunk_ids.txt`:** 2,854 chunks.
- **Rebuild:** byte for byte (`tests/test_make_tasks_v1.py` pins v1's checksums and checks the v2
  rebuild).

**Short of the 400-500 target.** At p ≈ 0.15, 325 items give a standard error of about 2.0 points
(130 gave 3.1), and the halves are ~160 items each. The review was strict, and the supplement
used 1,300 of the 1,319 unused numeric-rich chunks the per-document cap allows. Growing further
would need a higher cap, more items from the long manuals, and another review.

**Next:**
- Regenerate domain_qa for the compared checkpoints (pre-approved launch plan above).
- Results go into a new `results/table.md`; the current 130-item table is frozen as
  `results/table_v1.1.md` (the current scorer), next to `table_v1.md` (as published).

## 2026-10-04: Eval v2 scored; gold_lp end marker fixed; CPT makes read facts ~1.8x more probable
**GPU runs (pre-approved by the user, 1 x H100 each):**
- **Regenerated:** domain_qa on eval v2 with `--tasks domain_qa` for base-8b-hf, instruct-8b
  (`--chat --config-format auto`, its Stage 0 path), cpt-8b-replay10, cpt-8b and cpt-8b-seed1.
  Every pulled file was checked: the rows of the other three tasks are byte-identical to the
  committed ones, all 325 domain_qa rows carry `gold_lp`, and `generations_meta.json` records the
  run.
- **`ppl_postcutoff`:** base-8b-hf 6.206, cpt-8b-replay10 6.180 (-0.43%), cpt-8b-seed1 6.175
  (-0.51%). Measured over whole documents, the 2026 reports confirm the probe's -0.4%.
- **cpt-8b-seed1 re-merged from its adapter** (CPU `merge`, as originally). Perplexity reproduces
  bit for bit: every per-window NLL sum is identical, domain val 6.7184. It stands as the noise
  floor; `results/ppl/cpt-8b-seed1-remerge.json` is the evidence.
- **Table:** `results/table.md` starts on eval v2. cpt-8b-full and the Stage 0 base-8b keep blank
  qa cells (`--allow-partial`), and all rescoring made 0 judge calls.

**The gold_lp end-marker flaw, found and fixed the same day.** The first gold_lp scored the answer
followed by a lone "\n" token. In the prompt every few-shot answer is followed by the "\n\n"
token, so the lone "\n" was an unnatural continuation, and its log-probability fell under CPT
(the corpus's paragraph breaks are "\n\n"):

| | base-8b-hf | CPT |
|---|---|---|
| gold_lp, flawed | -12.18 | -14.20 |
| gold_lp, fixed | -6.78 | -6.19 |

That made CPT look worse at knowing the answers. The fix:
- **End marker:** `eval/gold_lp.py` scores " " + answer + "\n\n" (Tekken merges "\n\n" with a
  final "%", "." or "'" in 28 items, as it does in the prompt). All 325 items pass the boundary
  check in both formats.
- **Per-token data:** each item stores `gold_lp_tokens` and `gold_lp_end`, so the end marker can
  always be separated out. It is about -0.45 nats for base models; EOS is -0.78 for Instruct.
- **Re-run:** `run_eval.py --gold-lp-only` (Modal `kpi_eval --gold-lp-only`, 5 more H100 runs,
  user-approved) recomputed the fields without touching any output.
- **Instruct:** chat ends with EOS and was unaffected; its rerun reproduces its values exactly.

**Results (eval v2, paired per item against base-8b-hf, 95% bootstrap over items):**

| Run | Δ qa_acc (points) | Δ gold_lp (nats/answer) | Items more likely |
|---|---|---|---|
| cpt-8b | +2.5 [+0.3, +4.6] | +0.59 [+0.45, +0.75] | 70% |
| cpt-8b-seed1 | +0.9 [-1.5, +3.4] | +0.58 [+0.44, +0.74] | 71% |
| cpt-8b-replay10 | +0.9 [-0.9, +3.1] | +0.52 [+0.38, +0.66] | 68% |

- **Noise:** the seed gap is 1.5 points on qa_acc (standard error 1.8 at n = 325) and 0.007 nats on
  gold_lp ([-0.037, +0.051]).
- **By kind, per answer token** (cpt-8b): numbers +0.058 [+0.041, +0.076], identifiers +0.127
  [+0.074, +0.187], terms +0.147 [+0.048, +0.255]. Identifiers move most per answer only because
  they are longest (10.4 tokens against 5.2 for numbers).
- **Base qa_acc** on v2 is 0.120. By kind: number 0.144, identifier 0.077, term 0.053.

**Reading:**
- **CPT injected knowledge of the documents it read.** The gold answer to a closed-book question
  about a read document becomes about 1.8x more probable. Two seeds agree to 0.007 nats, and 70%
  of items move up.
- **Pass/fail can't resolve it.** The effect is about one noise unit in accuracy, which is why
  Stage 2's 130-item qa_acc showed nothing.
- **Not transfer.** Every eval item comes from a train document, so this is the closed-book
  counterpart of the -8% train-slice perplexity. Transfer is what `ppl_postcutoff` measures (-0.4%).
- **Numbers move least per token.** The specific values are the hardest to shift, which is a
  target for SFT's QA synthesis.
- **Revises the Stage 2 conclusion "no detectable KPI change".** The change was there, in the
  probabilities, below accuracy's resolution.

## 2026-10-04: Closed-book qa_acc kept as is, framed with a frontier reference (user decision)
**Context:** qa_acc reads 0.12-0.15 for every checkpoint, which looks broken next to grounded_acc's
0.84.

**Chose:** keep the score and the scorer exactly as they are. The 169-miss audit showed the score
is real, so loosening the match or dropping hard items to look respectable is out. Frame the score
instead:
- **README result blocks:** split into closed-book (gold_lp first, with the per-kind split),
  with-passages, and benchmarks, each captioned with what it measures.
- **Change table:** the closed-book rows are labelled as such.
- **A paragraph** on reading the numbers next to the open-book ones.

**Frontier reference (`eval/api_eval.py`):** Mistral Large 3 (mistral-large-2512, through the
API) on the same 325 closed-book questions, same prompt, greedy, scored by the same code.
- **qa_acc 0.280:** values 0.248, identifiers 0.446, terms 0.184. Its halves (seen 0.275, unseen
  0.285) agree, as a null check should.
- **gold_lp:** none, since the API returns no prompt log-probabilities.
- **Grounded/vocab/adversarial:** blank via `--allow-partial`.
- **Caveat:** Mistral Large 3 also wrote these questions from the passages; each call is
  stateless, so that gives it no answers.

**Reading:**
- **The base 8B is where it should be:** at 0.12, it scores below a frontier model that answers
  28% without the documents (SimpleQA's frontier range is 30-40% on far more common facts).
- **The questions are not guessable;** a frontier score near 0.6 would have said they were.
- **Identifiers are the frontier model's strongest kind** (45%), a pretraining-scale knowledge the
  8B base lacks (8%).
- **The number Stage 3 has to move is the seen-half accuracy,** from 0.13 toward the frontier
  reference and beyond, with retrieval at ~0.85-0.90 alongside.

## 2026-10-04: Contamination checks (13-gram overlap): no leak; the within-series gain is neither copied text nor genre
**Context:** the splits were checked only at the document level: exact and paragraph-MinHash
dedup ran before a document-level split. Four overlaps were never measured:
- train vs domain val below the paragraph (boilerplate, quoted provisions, front matter);
- the corpus and the replay slice vs the regression benchmarks (the corpus card's open caveat);
- train vs the 2026 reports: "never seen" must also mean "not a revision of a seen manual";
- the domain_qa few-shot items vs the scored items.

**Method** (`eval/contamination.py`, CPU, ~1 min; every table in `results/contamination.md`):
- **Overlap measure:** token 13-gram overlap on Tekken tokens (GPT-3 appendix C, counted on model
  tokens as in Llama 2). Each text is tokenised as `train/packing.py` feeds it to the model.
- **Benchmark splits:** the ones lm-eval scores (MMLU test, GSM8K test, HellaSwag validation).
- **Positive control:** excerpts of train and replay documents, tokenised on their own as a
  benchmark item is, come back 99.1% and 98.9% covered.
- **Sensitivity:** an 8-gram run was made alongside. It is not committed: it reruns with
  `--n 8 --out <path>`.

**Results:**
- **Val vs train.** 1.49% of val's 13-grams occur in train (3.25% of tokens). That is below what a
  train document shares with the other 233 (median 2.0%, 90th percentile 9.4%).
  - **Above 5%:** three val documents, each next to a sibling in train: FHWA HIF-18-044 (20.0%;
    HIF-18-043), FEMA P-1100-2A (18.0%; P-1100-2B) and HIF-17-020 (5.7%; HIF-17-019). The longest
    shared run in any val document is 290 tokens.
  - **Without them:** they are 7% of val tokens, and the val gain is unchanged: `cpt-8b-replay10`
    -2.33% -> -2.33%, `cpt-8b` -2.33% -> -2.32%.
- **Where the val gain sits.** Per perplexity window, the gain grows with the share of tokens
  inside a 13-gram seen in train (Spearman -0.57 over 296 windows; `cpt-8b-replay10`, seed 1
  agrees):

  | window tokens shared with train | val | 2026 reports |
  |---|---|---|
  | under 1% | -1.70% [-2.25, -1.24] (157 windows) | -0.11% [-0.46, +0.22] (56) |
  | 1-5% | -2.54% (86) | -0.90% (21) |
  | 5-20% | -3.56% (44) | -2.00% (5) |
  | over 20% | -5.21% (9) | none |

  At 8-grams the bins keep the same order: val -1.62 / -2.79 / -4.53 against 2026 -0.25 / -0.92 /
  -1.31 for 1-5%, 5-20% and over 20%.
- **2026 reports vs train.** At most 1.35% of a report's 13-grams (FHWA-HRT-26-057). No single
  train document holds more than 0.72% of a report, so none is a revision of a corpus document.
- **Benchmarks.**
  - GSM8K and HellaSwag: no 13-gram in train or replay.
  - MMLU: 21 of 14,042 items share a 13-gram, all stock phrases or digit runs ("in the 1960s and
    1970s", "1, 2, 3, 4, 5, 6"), and none is half covered.
  - At 8-grams the hits are digit strings ("800,000 to 200,000 years ago"), because Tekken gives
    each digit its own token. That is why 13 is the measure.
- **general_val vs replay.** 0.017% of tokens, longest shared run 15 tokens: disjoint. Replay's
  -2.2% on general val is in-distribution training, not overlap.
- **Few-shot.**
  - **Disjoint where it matters:** no shot is a scored question, and the shots share no 13-gram
    with each other.
  - **Two shared passages:** two of the three shots come from a passage that also produced a
    scored item, because `make_tasks.py:814` splits the shots off by item, not by passage:
    - qa-0003 (NIST GCR 22-917-51 p167): 0.95 d_b in the shot, 75% scored;
    - qa-0056 (GCR 17-917-45 p45): 4 fiber elements in the shot, L/500 scored.

    The 58% 13-gram share with qa-0056 is the question template ("...a brace in a steel
    concentrically braced frame per NIST GCR 17-917-45?"). No answer is given away.
  - **Coincidental values:** answer values recur only by coincidence ("4", "0.9" on unrelated
    items).

**Genre check (user request):** val is mostly manuals and the 2026 set mostly research reports, so
"new documents" could mean "a different genre".
- **Labels:** each document was labelled from the purpose stated in its front matter, before any
  pooling.
  - **Val guidance (10):** the five USACE EMs; NASA-STD-5002B ("defines the methodologies,
    practices, and requirements"); FEMA P-1100-2A (a prescriptive plan set); FEMA P-2018 (an
    evaluation methodology); FHWA HIF-17-020 ("This manual..."); HIF-18-044 (a design example).
  - **Val research reports (2):** HIF-18-047 ("This report documents a study...") and NIST GCR
    12-917-21 (a NEHRP Consultants research synthesis).
  - **2026 guidance (2):** ERDC/GSL SR-26-1 ("provides technical guidance for the load rating")
    and FHWA-HRT-26-056 (a procurement guide with sample contract language).
  - **2026 research and workshop reports (11):** the rest.
- **Results:** pooled per-document sums against `base-8b-hf`, with `eval/ppl_compare.py
  base-8b-hf <run> --set domain_val|postcutoff --docs <slugs>`:

  | | guidance | research reports |
  |---|---|---|
  | val (series in train) | -2.23% (10) | -3.64% (2) |
  | 2026 (series not in train) | -0.89% (2) | -0.32% (11) |

  The numbers are for `cpt-8b-replay10`. Seed 1 gives -2.23 / -3.77 / -0.87 / -0.43, and `cpt-8b`
  gives -2.22 / -3.67 on val.
- **Series in train:** none of the 2026 reports' series has a document in train (NIST TN 0, ERDC 0,
  FHWA-HRT 0). The two NIST GCRs are workshop reports from the Forward-Looking Codes and
  Standards programme, not the 917 series in train (26 documents).
- **Caveat:** small cells. NIST GCR 12-917-21 is 92% of the val reports' tokens, and HIF-18-047
  alone gains -2.54%.

**Reading:**
1. **No overlap changes a reported number.** The val documents above 5% don't move the gain. The
   regression benchmarks have nothing in train or replay beyond stock phrases. The 2026 set is new
   text.
2. **The within-series finding stands, and it is not shared strings.**
   - **About a quarter of the held-out gain sits on shared text:** -2.33% against the -1.70% floor
     on windows that share almost nothing. This is an association, since those windows are also
     more formulaic (base perplexity 5.6 against 7.2).
   - **The rest of the gap survives at zero overlap:** 1.6 of the 1.9 points between val (-2.33%)
     and the 2026 reports (-0.43%) remain on windows sharing almost no 13-gram, and it holds at
     8-grams.
   - **So "series" is shared conventions below copied text:** terms, notation, layout, a
     publisher's house style.
   - **Within the 2026 set too,** windows that share more text with train gain more. Resemblance to
     the training text is what CPT pays off on, at every level.
   - **The gradient is the shape of a learning effect, not a leak:** the gain rises smoothly from
     -1.70% (under 1% shared) to -5.21% (over 20%) across all 296 windows. A leak would sit in a few
     copied windows instead.
3. **Genre is not the explanation.**
   - **Within val:** the research reports gain more than the manuals.
   - **New guidance documents** gain -0.9%, 40% of what val's manuals gain and closer to the new
     reports than to them.
   - **What separates the gain is whether the document's series is in train,** so the honest
     sentence is: CPT transfers within a document series, across genre, and barely to a new series.
   - **What genre may still add** is -0.89% against -0.32% within the 2026 set. That is two
     documents, an order of magnitude below the series gap.

**Chose (user decision):** keep qa-0003 and qa-0056 in the frozen v2 set, and close the gap three
ways:
- **Why keep them:** no answer leaks, and the prompt is identical for every checkpoint, so no delta
  moves. Dropping them would re-freeze the table for 2 of 325 items.
- **Tag:** `make_tasks.py` (v2) writes `fewshot_passage_overlap: true` on the scored items that
  share a shot's passage, exactly these two, so per-item analyses can exclude them.
  - The committed `domain_qa.jsonl` changes on those two lines only; every other task file
    rebuilds byte-identical.
  - Rescoring `cpt-8b-replay10` with it reproduces its `table.md` row exactly.
- **Fix:** `--task-version 3` = v2 with the shots split off by passage, for the next from-scratch
  rebuild. The default stays 2.
  - It filters after ids are assigned, so every v3 item keeps its v2 id.
  - It only removes items: 322 = 325 minus the two, minus one identifier (qa-1143) that the 20%
    cap holds back once the set shrinks.
  - The other tasks are byte-identical.
- **Guard:** `tests/test_fewshot.py` pins the shared set and the tag. `tests/test_make_tasks_v1.py`
  rebuilds v1 (frozen), v2 (the committed 325, tags included) and v3 (removal only, same ids, no
  shot passage) from the LLM cache.

**Revisit if:** the corpus or the replay slice changes (rerun the script). Before Stage 3, run the
SFT/DPO data through the same check against every eval task (rule 10).

## 2026-10-05: Stage 3 SFT set built and frozen: 3,715 records, seen half 135/167 facts (user decisions)
**Context:** Stage 3 measures three things with the frozen v2 eval:
- whether SFT teaches the facts it is shown (the seen half, `eval/tasks/sft_seen_chunks.txt`);
- whether anything transfers (the unseen half);
- whether CPT contributed (SFT from `cpt-8b-replay10` vs SFT from the base, both on this one set).

This entry covers the data (Part A); training is Part B. The plan was A1-A7: chunk pool -> questions
-> dedup / caps / decontamination -> teacher completions -> rubric filter -> assembly -> checks.
The code is `data/scripts/sft_*.py` (`make sft-data`).

**Superseded counts:** this entry describes the first build (commit b20d03d). The full-passage
audit below dropped the worked problems and filtered closed-book twice: the frozen set is 3,126
records, seen half 119/167 facts.

**Result (first build):** `data/sft/train.jsonl` 3,601 + `sft_val.jsonl` 114.

| format | eval-seen chunks | ordinary chunks | total |
|---|---|---|---|
| closed-book (value / identifier / term) | 1,290 | 373 | 1,663 |
| closed-book, worked problem (multi_step) | - | 80 | 80 |
| definition | 722 | 50 | 772 |
| grounded (4 passages, cited) | - | 500 | 500 |
| abstain (4 passages, exact sentence) | - | 200 | 200 |
| replay (Tulu 3 SFT mixture) | - | - | 500 |

- **Wording:** each format is half the eval's exact instruction text (`prompts.qa_prompt` /
  `vocab_prompt` / `grounded_prompt`, the few-shot header included) and half paraphrased with a
  persona's question (closed-book 838 / 825, definition 374 / 398, grounded 250 / 250, abstain
  100 / 100).
- **Teachers:** Mistral Large 3 (`mistral-large-2512`) 2,459 completions, Mistral Medium 3.5 756.
  Medium is pinned to `mistral-medium-2604`, the dated id behind `mistral-medium-latest` /
  `-3.5` in the API's model list on 2026-10-04.
- **Tokens:** max 2,307 per record (mistral-common chat rendering, no system prompt), median 205,
  1.81M in total.
- **API:** 9,421 calls (pilot included) at 30 a minute: questions 1,111, answers 3,103, paraphrases
  370, problems 201, judge 4,423, revise 213.

**Seen half: guaranteed per chunk, measured per fact.** All 197 seen chunks are in the set. The
generator never saw which facts the eval asks about. The assembler tags `fact_seen` when an
answer matches a seen item on its own chunk, scored the eval's way.
- **domain_qa: 135 of 167 seen items** have at least one matching record.
- **vocab: 51 of 101.**
- **Report Stage 3 knowledge gains per half (rule 7)**, and, within the seen half, covered vs
  not-covered facts: the 32 + 50 not covered are "chunk seen, fact not shown".

**Interpretations of the plan, decided while building:**
- **Allowed chunks (rule 10, `sft_guard.py`):**
  - the seen half, plus free chunks outside `eval_chunk_ids.txt` that hash to seen
    (`sft_split.is_seen`, the plan's "seen-hash chunks");
  - minus a buffer: every chunk on the page of, or a page next to, an unseen-half source chunk
    (domain_qa / vocab unseen, held_back, locators, few-shot). Of the 215 unseen and held-back
    source chunks, 129 had a free chunk on their own page and 213 on a neighbouring one, so a fact
    could have crossed a chunk boundary. The buffer is 441 chunks.
  - Allowed: 36,616 of 76,582.
  - Every passage of every prompt (gold, neighbour, distractor) is an allowed chunk.
- **Rule 2's "same chunk" became the same document:** ordinary chunks are disjoint from eval
  chunks by construction. The check is the eval's own scoring (`qa_correct` with the item's
  tolerance; vocab by `term_key`). Unseen vocab terms and the 3 `VOCAB_SHOTS` terms are blocked in
  definitions from any document.
- **Rule 1 on eval-seen chunks drops the question, not the fact.** The generator's natural
  question for a seen fact sometimes lands on the eval's wording: same model, same chunk, and
  qa-0545 came back word for word. Dropping those facts would have emptied the seen half where it
  matters most.
  - The 7 such facts got two new questions written from the fact and the passage alone, never
    from the rejected question (`from_fact`).
  - The pilot first paraphrased the rejected question; the prompt audit in
    `tests/test_sft_data.py` caught it, and that path is gone.
- **Forced extraction:** values / identifiers and technical terms come back as two lists (6 each).
  - **Allocation:** within the 12-examples-per-chunk cap, values (2 closed-book phrasings) and
    term definitions (1) take turns, then closed-book term questions. Filling either kind first
    starved the other in the pilot: values first left vocab at 0 of 7 on the pilot chunks,
    definitions first cut domain_qa from 10 to 7 of 11.
  - **Scale (user decision):** the caps filled. Eval-seen chunks gave 2,315 examples against the
    plan's estimate of about 900 closed-book. The user kept the 12 / chunk cap: trimming to 8 would
    have cut coverage to 126 / 37, and to 6 to 113 / 29. So 54% of the set comes from the 197 seen
    chunks.
- **Passage sets are built in A3** (not A6), because the teacher answers with them in front of it.
  - Grounded = gold + its 2 nearest allowed neighbours + 1 allowed passage from another document.
  - Abstain = 2 allowed passages from the question's own document at least 3 chunks away + 2 from
    other documents: on-topic, like the adversarial eval, rather than trivially unrelated.
  - The teacher sees labels `[P1]`-`[P4]`, never chunk ids; assembly maps them back.
  - An abstain item whose teacher answered is dropped (4).
- **Evol-Instruct problems** come from ordinary chunks only (the Stage 5 seeds stay free of eval
  facts). Only 80 survived, against 150 planned: the judge rejected 56% for ill-posed problems or
  wrong arithmetic.
- **Validation:** 5% of each format's *holdable* records, grouped by source chunk. Eval-seen
  records always go to train, and 5% of a whole format would have taken nearly every ordinary
  definition.
- **No hand-written anchors in v1 (user decision).**
  - Teacher text the user approves line by line is still teacher text, so labelling it
    `teacher: "human"` would misstate the metadata.
  - The 500 Tulu records cover the real-data role.
  - The human check is a read of `data/sft/review.md` (50 kept records and the disagreements).

**Filtering (counts in `data/sft/stats.json`; every reject in `data/sft/sft_rejected.jsonl` with
its rule):**
- **A3, 7,743 tasks from 6,608 questions:**
  - per-chunk cap 1,288 (eval-seen only);
  - trivia 301;
  - ungrounded number 291;
  - term not in passage 174;
  - locator 159 (`is_locator`, plus "which equations" with a number);
  - duplicate 124 + near-duplicate 8;
  - context-bound 71;
  - long answer 63;
  - rule 2 58;
  - rule 1 26 (+7 rewritten).
- **A5, 4,020 examples, 3,452 kept (85.9%):**

  | format | kept |
  |---|---|
  | abstain | 95.3% |
  | grounded | 94.0% |
  | definition | 87.6% |
  | closed-book | 84.7% |
  | multi_step | 44.2% |

  - 554 factual, 10 format, 4 abstain answered.
  - Revise: 31 format-only examples revised, 23 salvaged (74%).
- **A6:**
  - rule 1 on final questions: 1;
  - over the ordinary targets: 236.

**The judge and the verifier.** The rule verifier and the judge (Large 3, ch. 12 rubrics with the
gold fact as a hard rule) disagree on a rule they both check for 175 of 4,020 examples.
- **The keep rule:** an example is kept only when both pass.
- **Read 50 of the disagreements (A7).**
  - The verifier was the stricter check in 37: it was right in about 10 and wrong in about 12,
    all equivalent forms ("AASHTO LRFD 4.6.2.2.2b" vs "Article 4.6.2.2.2b", "L1 and L2" vs
    "L1/L2").
  - The judge was the stricter check in 13, right in about two-thirds ("greater than 1.0" vs
    gold "1.0").
  - The AND costs yield (about a dozen correct answers in 175), not quality: in the sample no
    wrong final answer was kept.
- **Fixed after the first full pass:**
  - The verifier now reads plurals and number words ("six times" = 6, "cripple walls" = "cripple
    wall").
  - It accepts a worked problem's result at the end of one paragraph ("... = 6500 kN. Answer:
    6500 kN"); assembly moves it to its own last line.
  - JSON / list debris and "the passage" in a closed-book completion are format failures. The read
    found "from the passage: 0.015" and a stringified Python list among the kept problems.
- **One rule tightened back.** At first a factual verifier failure that the judge contradicted
  could still go to revise. The revise prompt carries the judge's note, so that turned revise into
  a fact-correcting step for 42 final records ("water jetting, greencutting, or sand blasting" ->
  "roughened surface"). The plan says never revise a factual failure, so any factual failure now
  blocks it.

**Self-preference (the reason for the second teacher, user decision).** Large judged every
completion:
- **Overall:** its own answers were accepted at 87.4%, Medium's at 81.2%.
- **By format, the gap sits in definitions:**

  | format | Large | Medium |
  |---|---|---|
  | definition | 93.0% | 71.3% |
  | closed-book | 85.5% | 82.3% |
  | grounded | 94.1% | 93.8% |
  | abstain | 96.1% | 91.8% |
  | multi_step | 38.8% | 59.6% |

- **Upper bound only:** the 6.2-point overall gap is quality difference plus self-preference, so
  it is an upper bound on the bias. A third-party judge on a sample would separate them.
- The training comparison is unaffected: sft-from-cpt and sft-from-base train on the identical set.

**A7 read (50 kept, stratified, plus the 50 disagreements above):**
- **Closed-book:** 10 of 10 correct. Two persona phrasings read oddly ("During the failure analysis,
  what value did the AASHTO LRFD ... assign ...").
- **Definitions:** 10 of 10 faithful, some fuller than the reference.
- **Grounded:** 10 of 10 cited correctly. One answer is weak: it points to "Figure 63" rather than
  saying what the figure shows.
- **Abstain:** 10 of 10 correct. A few questions are document trivia (a cost range, a year of
  technical coordination); harmless as abstain prompts.
- **Multi-step:** the weak format. 3 of 10 had the passage / debris defects fixed above, and a
  few problems are contrived ("15 symbols x 2.5 MB"). Kept, at 80 records.

**Contamination (`results/contamination.md` section 6, `eval/contamination.py --only sft`):**
- **Rule 10 gate:** 0 chunk ids from `eval_chunk_ids.txt` outside the seen half.
- **Positive control:** 40 of 40 planted items found (20 unseen domain_qa, 20 MMLU).
- **Questions:** no eval question inside any SFT prompt.
- **Unseen half:** 0 vocab terms defined. 9 domain_qa questions have half their tokens covered by
  SFT text.
  - Traced, those are document names and boilerplate ("What is the maximum ... in NIST GCR
    17-917-45"): 2 are covered only by passage text, which CPT already trained on; the nearest SFT
    questions ask about other facts.
- **Unseen answers equal to a same-document SFT answer:** 4.
  - 3 are coincidences (20% vs an h/t ratio of 20; two different NASA-STD-5001B factors).
  - 1 is real: **qa-1059** (FEMA P-695 βRTR = 0.40) is stated again on a seen page, so the unseen
    half has one taught fact. That is an eval-design overlap, not a leak; read qa-1059 with that in
    mind.
- **Benchmarks:** MMLU 7 / GSM8K 2 / HellaSwag 0 items share any 13-gram with the set, none half
  covered.
- **Val vs train:** 1 of 114 val records has half its question + completion covered by train.

**Replay:** 500 records from `allenai/tulu-3-sft-mixture` (ODC-BY-1.0; 939,343 rows), as they are.
- **Eligible:** single-turn user/assistant, no system message, at most 6,000 characters, at least
  95% ASCII, not the multilingual aya subset (754,635 eligible).
- **Sampling:** proportional by source with a fixed seed. 17 sources: math 218, code 93, FLAN 53,
  safety 73 (WildGuard / WildJailbreak / CoCoNot), WildChat 26, other 37.
- **Licence:** ODC-BY covers redistribution with attribution. Rule 1 (public-domain sources)
  governs the domain corpus, not general replay, as with FineWeb-Edu in Stage 2.
- _Superseded 2026-10-05 (v1 final entry): three Persona subsets with Claude-written responses
  were excluded and their 140 records refilled; source counts there._

**Revisit if:**
- the seen half's gain is flat while the covered facts' gold_lp rises: raise the per-chunk cap
  (blind), don't target the eval;
- `vocab_seen` doesn't move: only 51 of 101 terms were extracted, so a term-only extraction pass
  over the seen chunks is the next lever;
- Part B's chat rendering differs from the eval's `--chat` path (mistral-common, no system prompt):
  the token counts and the exact-wording half assume it.

## 2026-10-05: SFT set audited against the full passages; refrozen at 3,126 records (user decisions)
**Context:** the first build's checks were the Mistral judge (the teacher's own model family) and a
50-record read with passages cut to their first ~260 characters. A stronger check was asked for: a
stratified sample read against the *full* source passages (not a human read), one fixed rubric per
format.
Verdicts: ok / minor (correct, style only, kept) / defect (would teach something wrong).
`data/scripts/sft_audit.py`; every verdict in `data/sft/audit.jsonl`, the report in
`data/sft/audit.md`. Rates are count/n with Wilson 95% intervals.

**Round 1: the first build (200 records, 40 per format, 30 Large / 10 Medium):**

| format | defects |
|---|---|
| abstain | 0/40 (0-9%) |
| definition | 3/40 (8%; 3-20%) |
| closed-book | 8/40 (20%; 10-35%) |
| grounded | 9/40 (22%; 12-38%) |
| multi_step | 17/40 (42%; 29-58%) |
| all | 37/200 (18%; 14-24%) |

- **The judge passed all 37.** Agreement with the verifier didn't catch them either: they are
  framing errors a value check can't see.
- **What goes wrong:**
  - **Closed-book questions claim more than the passage.** They turn "should", "typical" or one
    study's result into "must", "maximum" or "required".
  - **Conditions change:** "less than 2 to 1" becomes "does not exceed"; "20 or more" becomes
    "20".
  - **Other closed-book errors:** they credit a document with what it only cites, use
    worked-example inputs, or carry PDF artifacts ("AASHTO T 1619" = T 161 + footnote 19).
  - **Grounded:** a sentence without a citation (6 of the 9), two invented claims, a reversed
    equation (bfc >= L/85 read as "85 times").
  - **Worked problems:** wrong physics or set-up (force vs moment balance, a fencepost count, L^3
    for an L^4 stiffness), steps missing, or no document value needed at all.
- **Self-preference, read against the judge's acceptance rates.** The judge accepted 87.4% of
  Large's answers and 81.2% of Medium's, but the audit found Medium's defect rate no higher
  (7/50, 14%, vs 30/150, 20%). The judge's gap is not quality, which points to self-preference.
  The samples are small: 50 Medium records.

**Checks tried and left out.** On the 40 audited closed-book records, the eval's own two checks
(make_tasks.verify_qa's faithfulness test, the blind is_standalone test) run with Mistral Large 3:
- **Faithfulness** caught 1 of 8 defects (and no good records).
- **Standalone** rejected 4 of 8 defects, but also 9 of 19 good ones. The defect rate among what it
  kept was unchanged at 20%.
- Neither is in the pipeline (`sft_judge.py` says so).

**Fixes (rebuilt from the cache; no new teacher calls):**
- **multi_step dropped (80 records).** The Stage 5 seeds need a verifier that checks the set-up,
  not only the arithmetic.
- **Grounded: every sentence must carry its own [Pn]** (a format rule in `sft_judge.verify`, so it
  goes to the revise round). 0 of the 500 final grounded records has an uncited sentence.
- **Closed-book filter, pass 1:** every one of the 1,663 closed-book records read against its
  passage, grouped by passage in 10 packets: 333 defects (20.0%), 518 minor, 812 ok.
  - **Paraphrased wordings fail more:** among eval-seen facts, 150 of 643 paraphrased records vs 93
    of 647 exact ones (23% vs 14%). The persona wording is where scope drifts.
  - **Agreement:** on the round-1 sample, pass 1 and the round-1 audit agree on defect-or-not for
    38 of 40 records (Cohen's kappa 0.84).
- **Round 2 (after pass 1):**
  - Grounded fell to 3/40 (8%; 3-20%).
  - Closed-book stayed at 10/40 (25%; 14-40%). Pass 1 had noted those flaws but filed them
    "minor", which keeps the record: 9 of 19 pass-1 minors in the sample were defects
    by the round-2 reading, against 1 of 21 pass-1 oks.
- **Closed-book filter, pass 2 (user decision):** the 518 pass-1 minors re-read with the stricter
  line. A wrong framing, overclaim, scope moved by a persona, example value or garbled symbol is a
  defect even when the answer value is right. Pass 2 found 176 defects and kept 342. Assembly
  drops all 509 and refuses a closed-book record without a verdict (then
  `data/sft/closed_book_filter.jsonl`, now `data/sft/read_filter.jsonl`; pinned by `tests/test_sft_data.py`).
- **Round 3 (after pass 2):** closed-book 2/40 (5%; 1-17%), and no wrong answers. Both defects are
  framing (an unnamed document where another code differs; a "maximum" the passage never gives).

**The frozen set: 3,126 records** (`train.jsonl` 3,031, `sft_val.jsonl` 95; `data/sft/SHA256SUMS`):

| format | eval-seen chunks | ordinary chunks | total |
|---|---|---|---|
| closed-book | 901 | 253 | 1,154 |
| definition | 722 | 50 | 772 |
| grounded | - | 500 | 500 |
| abstain | - | 200 | 200 |
| replay (Tulu 3) | - | - | 500 |

- **Seen coverage:** domain_qa 119/167 (135 before the filter: 16 seen facts lost every phrasing
  to it), vocab 51/101.
- **Tokens:** max 2,307 per record, 1.74M in total.
- **Teachers:** Large 2,034, Medium 592.
- **Contamination (section 6):** still clean.
  - 0 leaked chunks; control 40/40; no eval question in a prompt.
  - Unseen same-document answer matches: 4 (3 coincidences plus qa-1059, as before).
  - Benchmarks: MMLU 7 / GSM8K 0 / HellaSwag 0 items with any 13-gram, none half covered.

**Residual quality, as measured:**

| format | measured on | defect rate |
|---|---|---|
| closed-book | round 3 | ~5% |
| grounded | round 2 | ~8% |
| definition | round 1 | ~8% (not filtered) |
| abstain | round 1 | ~0% |
| replay | not audited | - |

- **Abstain is easy too often:** 10/40 of its records pair the question with passages off its
  subject. Choosing the same-document passages by similarity would make them harder.
- **Minor flaws stay in by design:** odd persona framing, locator trivia (chapter or section
  numbers), completions with a few extra words.

**Source issues the reads surfaced (not fixed here):**
- **FEMA P-2355 / P-2335:** the document's cover, preface and footers say "FEMA P-2335 / May
  2025", while `data/sources.csv` and FEMA's URL say P-2355. 28 SFT records name it P-2355.
- **PDF footnote merges** ("AASHTO T 1619", "M 1951", fhwa-hif19067-nov2021); records carrying
  them were filtered as garbled.
- **Sample-contract pages** (EM 1110-2-1003 p382) read as manual requirements; filtered.

**Revisit if:**
- a rebuild changes closed-book records: run both filter passes and a fresh round (CLAUDE.md,
  Stage 3);
- Stage 5 needs worked problems: generate them with a set-up check, not only an arithmetic one;
- seen coverage matters more than v1 allows: the 16 lost facts can come back as new, faithful
  phrasings (a rebuild of A2/A4 for those chunks).

## 2026-10-05: Retrospective on Stages 0-2: what I would do differently (user decision)
**Context:** written before Stage 3's training launches, so it can't be fitted to Stage 3's results.
Nine items, ordered by how much each would have changed Stage 2's result. Each gives what was done,
the evidence (with the entry it comes from), the better choice and its cost. Where an item changes
how later stages run, it names the rule it becomes; the rules are collected at the end. Costs are in
H100-hours and API calls, since the $3.95/h price is unverified. None of this says Stage 2 was run
badly; most of it is knowledge Stage 2 produced. The README's "What I would do differently" is the
short form.

1. **Build the sensitive metric before the intervention.**
   - **Done:** Stage 2 was first read on the 130-item domain_qa (eval v1), pass/fail, standard error
     3.1 points. The verdict was "no detectable KPI change" (2026-09-28 main-run verdict).
   - **Evidence:** after the rebuild (325 items, `gold_lp`, seen/unseen halves; 2026-10-04),
     `cpt-8b` moves gold_lp +0.59 nats [+0.45, +0.75] and seed 1 +0.58, with 70-71% of items up.
     qa_acc on 325 items still moves by about one noise unit (+2.5 / +0.9 / +0.9 against 1.8). The
     effect was in the probabilities all along, below accuracy's resolution.
   - **Better:** gold_lp, the halves and the larger set as Stage 0 deliverables, built with the eval
     frozen on 2026-09-27 rather than a week later. Cost: the same day's work, moved earlier.
   - **The pre-registration was the symptom.** The -20% held-out perplexity target measured
     transfer, which a 20M-token corpus doesn't produce (2026-10-04 conclusions). The right target
     was gold_lp on facts from the documents CPT reads, with `ppl_postcutoff` for transfer.

2. **A no-op control through the eval path before any training.**
   - **Done:** Stage 0 evaluated the hub base through vLLM's native path; merged checkpoints load
     through its HF path. `base-8b-hf` (the base re-saved through `merge.py`, evaluated like a merged
     checkpoint) was built on 2026-09-28, after `cpt-8b` seemed to lose 3.7 MMLU and 6.7 GSM8K
     points.
   - **Evidence:** on untouched weights the unpatched HF path cost 4.8% perplexity, 3.1 MMLU, 6.1
     GSM8K and 19.5 grounded points (2026-09-28 YaRN entry): more than any training effect in
     Stage 2. The CPT checkpoints were evaluated, re-merged with the fix and evaluated again.
   - **Better:** run `base-8b-hf` on day one. Cost: one eval pass (lm-eval took 33 minutes of one
     H100 on `base-8b-hf`), or a few minutes of `eval/vllm_ppl.py` against `perplexity.py`, which
     shows the gap alone (7.23 vs 6.89). The same idea as the seed floor: the instrument must read
     zero before it measures.
   - **Rule:** before a stage's first training run, its starting checkpoint goes through that
     stage's save/merge and eval path with no training (an untrained adapter, merged). It must match
     the checkpoint evaluated directly, and that run is the stage's zero point.

3. **Augment the stream for the facts that matter.**
   - **Done:** one pass over the concatenated documents, so each fact is seen as often as its
     document states it, usually once.
   - **Evidence:** per answer token, CPT moved values +0.06 nats, identifiers +0.13 and terms +0.15
     (2026-10-04 eval v2 entry). The values, the core of the eval, moved least. That is consistent
     with too few exposures; it doesn't test it.
   - **Better:** synthetic continued pre-training, meaning paraphrases and QA rewrites of passages
     mixed into the CPT stream (Yang et al. 2024, "Synthetic continued pretraining"; Allen-Zhu & Li
     2023, "Physics of Language Models 3.1").
     - **Cost:** the whole pool (76,582 chunks) was out of reach at 30 calls a minute. The 197
       eval-seen chunks were not: five rewrites each is about 1,000 calls, ~35 minutes of the key's
       quota.
     - **What it buys:** with the halves of item 1, Stage 2's seen half would have measured
       knowledge injection, and the unseen half its absence.
   - **Now:** Stage 3 does the instruction-data form: 1,623 of the frozen set's records come from
     the 197 seen chunks.

4. **Ablate along the axis the goal lives on.**
   - **Done:** A replay (forgetting), B full vs LoRA (parameterisation), C 2-GPU (systems), plus the
     LR-up and seed runs.
   - **Evidence:**
     - **A:** pre-registered as likely null ("replay has nothing to protect", 2026-09-27 ablation
       rules). It was null within noise, and its adoption fired on noise (2026-09-28 critique).
     - **B:** it did speak to knowledge (train slice -22% vs -8%, qa_acc +3.1 at the noise edge),
       but its weights are gone (item 8).
     - **LR-up:** it memorised more, with no held-out gain.
   - **Better:** epochs (1 vs 3) and augmentation (0 vs K rewrites of the seen chunks), read on
     gold_lp and the halves.
     - **Cost:** one LoRA epoch is ~0.95 H100-hours, so ~2.9 for the epoch ablation and ~1 for
       augmentation, plus its API calls.
     - **What it reverses:** the 2026-09-27 hyperparameter entry ruled out a second epoch for the
       main config. As an ablation, repetition is the knowledge question.
   - **Reading:** B and C taught the engineering (FSDP2, memory, the code-path speed-up). A answered
     a question the pre-registration had already answered.

5. **Judge CPT by the downstream number.**
   - **Done:** Stage 2's decision rule (2026-09-27 success criteria) had three branches:
     "worked", "barely moved" (LR up or a second epoch) and "memorising". None of them was "don't
     carry CPT forward". The carried-forward checkpoint was picked among CPT runs on perplexity and,
     for replay, on MMLU.
   - **Evidence:** whether CPT was worth doing is answered by SFT from `cpt-8b-replay10` vs SFT
     from the base on the same set. The Stage 3 plan (2026-10-05) runs that as its third measurement.
   - **Better:** write that comparison into Stage 2's decision as the deciding test, and keep the
     carry-forward choice provisional until it is read. Cost: one SFT run on the base, already
     planned.
   - **Rule:** `sft-from-base` is a standing part of the chain. It runs next to `sft-from-cpt` and
     decides whether CPT stays in it, and any later claim of a CPT effect is read against that
     lineage.

6. **Document boundaries.**
   - **Done:** one BOS/EOS pair per document (`train/packing.py`). That is 234 EOS in 19.4M train
     tokens, one per ~83k, and about 5% of 4,096-token windows contain an end of document.
   - **Evidence:** `cpt-8b` runs to the 256-token cap where the base stops after ~28 tokens, on the
     fixed path too (2026-09-28 latency entry). The cause is read from the EOS count, not tested.
   - **Better:** section-level units. Manuals are split at chapter and section headings into units of
     a few thousand tokens, each with its own BOS/EOS, so ends of units stay common.
     - **In `packing.py`:** free.
     - **Whole units packed into bins** (TRL's `bfd`, or our own) would also stop windows from
       starting mid-sentence. That needs per-unit attention masks (the flash-attn varlen path the
       2026-09-27 hyperparameter entry deferred) or padding.
   - **When it surfaced:** only in the latency table.
   - **Rule:** a future CPT uses section-level units and reports output length on the KPI prompts
     against its starting checkpoint. The first run that does also tests this item's cause.

7. **Define "new documents" before training, matched to the corpus.**
   - **Done:** the 13 post-cutoff reports were sourced on 2026-09-28, after Stage 2 trained, when
     the prior-exposure check came back inconclusive. 11 are research, technical and workshop
     reports; the corpus is mostly manuals and guides.
   - **Evidence:** the genre split (2026-10-04 contamination entry) rules genre out as the
     explanation. Val manuals gain -2.23%, new guidance -0.89% and new reports -0.32%; what
     separates the gains is whether the document's series is in train. The new-guidance cell is two
     documents.
   - **Better:** post-cutoff manuals and guides from the same publishers, chosen and pinned before
     Stage 2, with series both in and out of train (the variable that turned out to matter). Then
     the -0.4% would rest on a designed comparison rather than a check made afterwards. Cost:
     sourcing time, no GPU; how many such manuals exist is unknown.

8. **Hygiene rules from day one.** Each was written after a loss:
   - **Weights:** `cpt-8b-full`'s were deleted, and as a full-parameter run it has no adapter to
     re-merge, so it has no 325-item QA scores (2026-10-04 qa_acc audit).
   - **API time:**
     - Burst-then-sleep left ~3/4 of worker time asleep.
     - An adaptive pacer still drew a 429 on about one call in five, until `make_tasks.Pacer --rpm
       30` (2026-10-04).
     - At Stage 0, 8 judge verdicts failed on 429s.
   - **Judge calls:** a scratch run outside `results/` re-judged 50 items (2026-10-04).

   **Rules (in force since 2026-10-04; CLAUDE.md rule 12 and the judge-cache guard):**
   - adapters are never deleted;
   - no checkpoint with a table row is deleted until its stage's write-up is frozen;
   - API clients pace at the key's limit instead of retrying 429s;
   - a `--results-dir` outside `results/` must name `--judge-cache`.

   For a next project they are day-one rules.

9. **Calibrate every judge rubric before it labels anything.**
   - **Done:**
     - **The KPI grading judge** was hand-checked at Stage 0 (40 verdicts, 2026-09-27). It errs
       strict: its misses are false negatives.
     - **The SFT judge** (Large 3 with the ch. 12 rubrics) is a different rubric doing a different
       job, labelling training data. It went straight to 4,020 examples (2026-10-05).
   - **Evidence:**
     - The full-passage audit found 37 of 200 judged-kept records defective (18%; 14-24%), and the
       judge had passed all 37. They are framing errors, overclaims and changed conditions, which a
       value check can't see.
     - The closed-book pass 1 then found 333 of 1,663 (20%).
   - **Better:** before a rubric labels anything, a set of ~40 items with known defects, read against
     the full passages. The judge's catch rate on it decides whether a full read is needed. Cost: a
     few hours of reading, against two filter passes over 1,663 records afterwards.
   - **Rule:** every judge rubric gets that benchmark before it labels data. The Stage 4 preference
     judge gets it before any DPO pair is labelled.

**Rules from here:**
- **No-op control:** before every stage's training (item 2).
- **CPT boundaries:** section-level units for any future CPT, with output length reported (item 6).
- **Checkpoints:** adapters are never deleted, and no tabled checkpoint is deleted before its
  write-up is frozen (item 8, rule 12).
- **Judges:** a benchmark per judge rubric; Stage 4's before any DPO pair is labelled (item 9).
- **CPT control:** `sft-from-base` stays as the control that decides whether CPT is in the chain
  (item 5).

## 2026-10-05: SFT set v1 final: 2,516 records; seen half 121/167 facts, 89/101 terms (user decisions)
**Context:** the open items from the audit entry above, decided by the user and worked in order:
1. the FEMA metadata fix;
2. definitions capped and read;
3. the lost seen facts and the missing seen terms;
4. hard-negative abstain;
5. contamination and tests;
6. the read of `data/sft/review.md` (not a human read);
7. then refreeze.

The full-passage read is now the filter for every format that needs one. `data/sft/read_filter.jsonl`
(2,053 verdicts) holds each record's verdict with a fingerprint of what was read. Assembly drops
defects and leaves out anything that must be read and wasn't. The rubrics are committed in
`data/sft/read_rubrics.md`, and the loop is in CLAUDE.md (Stage 3).

**FEMA P-2335, not P-2355.**
- **Authority:** the document's own cover, preface and footers say "FEMA P-2335 / May 2025". FEMA's
  download URL (`..._p2355_042025.pdf`) and therefore `data/sources.csv` said P-2355.
- **The URL is a filename, not an authority.** A client corpus will have this exact problem: take
  document numbers from the document.
- **Fixed:**
  - the title in `sources.csv` (there is no separate document-number field; the slug `fema-p-2355`
    stays as the id);
  - the corpus-card line above;
  - the title field of the gitignored `docs_raw.jsonl` and `chunks.jsonl`, patched to what a rebuild
    now writes, so `extract.py --chunks` stays byte-identical to the file on disk.
- **The eval is unaffected.** No task item says P-2355: the hits are the slug inside chunk ids,
  three rejected items, and grounded passage text that already reads P-2335. `make_tasks` never
  reads titles. No task-version change.
- **SFT:** the document's records were regenerated with the corrected title (`REGENERATED_DOCS`)
  and every one was read again. 26 are in the final set.

**Definitions: capped at 300, then read.**
- **The cap:** 772 was five times the plan's 150 and the unread quarter of the set.
  `sft_assemble.DEFINITION_CAP` keeps every definition of a seen-half vocab term first, then fills
  by hash.
- **The read:** 23 of the 300 were defects (7.7%). Most turned a condition from the passage into
  the definition, or added a doubtful specific. 277 are in the set, 136 of them defining a seen term.

**The seen half.**
- **Terms: 51 -> 89 of 101 covered.**
  - Every seen-half vocab term gets a definition task of its own: all 101, so the set doesn't
    depend on what an earlier build covered.
  - This relaxes rule 10 for those term names only: the teacher sees the term and its chunk, never
    the eval's reference definition.
  - 89 targeted definitions passed the read.
- **Facts: 119 -> 121 of 167 covered.** The 17 facts whose every phrasing the read had dropped were
  re-asked, one question each, exact wording, no persona: 4 passed the read. The 11 that failed got
  a second, quote-first re-ask (the teacher quotes the passage's sentence, then asks with its
  conditions): 2 passed.
- **Why re-asking stalls:** for most of these facts the extracted fact itself carries the error.
  Examples: "25 percent" is one term of a greater-of rule; the 50% uplift relief belongs to another
  structure; "h/t less than 8" inverts a lower limit. No wording fixes a misread fact. Recovering
  them means re-extracting, not re-asking.
- **Shared answers:** when the read finds a wrong answer, partial answer or worked-example value in
  one phrasing, every phrasing of that task goes, since they share the answer. That rule
  (`ANSWER_DEFECTS`) removed 10 records.

**Abstain: hard negatives.**
- **What was rebuilt:**
  - the least similar quarter of the abstain sets (69), with the most similar allowed passages by
    BM25 over the question, skipping the source's own neighbourhood and every passage that states
    the gold;
  - every set where the hard rule (`sft_common.gold_present`) found the gold in a passage (45).
- **The hard rule over-reaches.** Round 1 read 40 of the old sets and found none that answered its
  question; the rule's number matching is broad. It runs in A5 too (`gold_absent`).
- **What it caught:** the teacher answered 29 rebuilt items instead of declining (dropped), and the
  read found 5 of 60 more with the answer present or partly present (e.g. a conversion table that
  gives 1/0.145 kPa per psi).
- **Result:** 195 abstain records, 53 of them hard negatives. Off-topic negatives taught
  "unrelated -> refuse", which is not what the adversarial eval scores.

**A cache bug, found and fixed.**
- **The bug:** `sft_common.llm_json` loaded its cache lazily on first use. pmap's workers made
  their first calls together, so each got its own copy and re-asked prompts another had just
  answered. 33 keys were written twice, with different outputs.
- **The effect:** on the next load the last copy won, so a cached paraphrase could change between
  runs. That showed up as 21 records whose read no longer matched their text.
- **The fix:** the load now takes the lock and keeps the first answer for a key. Those records were
  read again. The fingerprints are what caught it.

**The review read (A7)**, not a human read, covered 56 records with their full passages: the 50-record sample, plus the records
that entered it as defects were dropped (verdicts in `read_filter.jsonl` as `review-2026-10-05`).

| format | defects |
|---|---|
| closed-book | 1/16 |
| definition | 1/16 |
| grounded | 4/14 |
| abstain | 0/10 |

- **The defects:**
  - a persona moving a building-code fact into "a dam safety assessment";
  - "safe room" redefined as part of a school that keeps utilities running;
  - two grounded answers adding what the passage doesn't say (one where the text breaks off);
  - a grounded answer that only points to a figure;
  - a base-shear equation with the importance factor inverted (R·I for R/I) from garbled extraction.
- **A correction to the first build's 50-record read:** it had passed the two unsupported grounded
  answers as "cited correctly". That read saw passages cut to their first 260 characters.

**The grounded read.** Grounded was the one generated format no full-passage read had filtered, so
all of it was read (user decision). 94 of 485 were defects (19%).
- **What failed:**
  - hedges hardened ("should", "may be required" -> "must", "is required");
  - conditions dropped ("for box sections", "if T >= 0.7 s");
  - equations miscopied from garbled extraction;
  - text filled in past a page break;
  - citations pointing at the wrong passage;
  - "how" questions answered with "what".
- **Result:** grounded stands at 401 (the target was 500; the judged extras were not read, so they
  are not used).
- **What it says about the earlier estimates:** round 2's 3/40 (8%) and the review's 4/14 both sat
  inside the interval of the full read's 19%; the round-2 sample happened to be clean.

**The final set** (`data/sft/train.jsonl` 2,436, `sft_val.jsonl` 80):

| format | eval-seen | ordinary | total |
|---|---|---|---|
| closed-book | 892 | 252 | 1,144 |
| definition | 266 | 10 | 276 |
| grounded | - | 401 | 401 |
| abstain | - | 195 | 195 |
| replay | - | - | 500 |

- 1.43M tokens, max 2,875 per record (a replay record).
- **Every closed-book record, definition and grounded answer** has a keep verdict from a
  full-passage read. So do every hard-negative abstain set and every record of the regenerated
  document. Nothing is left unread (`stats.json` assemble.unread = 0, tested).
- **Contamination (section 6):**
  - 0 leaked chunks; control 40/40; no eval question in a prompt; benchmarks clean; no val record
    half covered by train.
  - Four unseen answers have 13-grams in SFT text, all document identifiers. Three sit in passage
    text. One, qa-1070 (ER 1110-2-1806), is also an SFT answer from another manual
    (EM 1110-2-2400) that cites the same regulation. That is cross-document knowledge, the
    transfer the unseen half measures, not the unseen chunk's text.
  - With qa-1059 (above), two unseen items are taught through other pages or documents.
- **Abstain was not read in full:** 53 hard negatives were read (5 of 60 dropped) and round 1
  found 0/40 among the rest. The read verdicts are the filter; the audit rounds above are
  measurements and drop nothing.

**No Claude-written text in the training data (user rule, CLAUDE.md rule 13).** Anthropic's terms
list training a model among the uses that compete with its services, so Claude's part is the
tooling and verdict-only review. No training record may hold text Claude wrote. The README says
who wrote the data.
- **Domain records (2,016):** clean by construction.
  - Every question and completion is Mistral output, with two exceptions: the `[Pn]` labels are
    mapped to chunk ids, and abstain completions are the eval's fixed refusal sentence.
  - The full-passage reads return verdicts only and change no text.
- **Replay broke the rule in three Tulu 3 subsets:**

  | subset | records | why it was left out |
  |---|---|---|
  | Persona Python | 23 | its solutions were written by `claude-3-5-sonnet` (Tulu 3 report) |
  | Persona MATH | 97 | its card: "Outputs were generated using GPT-4o and Claude 3.5 Sonnet", with no per-row model |
  | Persona IF | 20 | its card and the report name no response generator |

- **The other subsets name their generators**, and none is Claude:
  - Persona GSM and Algebra solutions are GPT-4o's (report).
  - The rest are GPT-3.5/GPT-4, Mixtral, or human-written (FLAN, No Robots, OASST).
- **Fix:** `sft_replay.py` keeps its first draw, drops those 140 records (`EXCLUDED`) and refills
  them from the other sources with a second seed.
  - 360 replay records are unchanged and 140 are new.
  - All 2,016 domain records are byte-identical.
  - Replay stays proportional over the allowed sources.
- **Replay by source now:**

  | group | records | sources |
  |---|---|---|
  | math | 168 | NuminaMath 58, withdrawn math set 46, Persona GSM 46, Persona Algebra 18 |
  | safety | 102 | WildGuard 46, WildJailbreak 46, CoCoNot 10 |
  | code | 97 | Evol CodeAlpaca |
  | FLAN | 74 | |
  | WildChat | 36 | |
  | other | 23 | No Robots 8, SciRIFF 7, TableGPT 4, OASST 4 |

- **Rechecked:** contamination section 6 within its gates (control 40/40; benchmarks: 7 MMLU items
  share one 13-gram, none half covered), and 47 tests pass.

**Self-preference, measured.**
- **The judge's acceptance:** Large judged every completion and accepted 87.4% of its own model's
  answers against 81.2% of Medium's.
- **The full-passage audit:** Medium's answers were no worse, with defect rates of 14% (7/50) for
  Medium and 20% (30/150) for Large.
- **Reading:** the judge's 6-point preference is not quality. With 50 Medium records the intervals
  overlap (Medium 7-26%, Large 14-27%), so it is a direction, not a precise size.

**Stage 5 (decided now):**
- **Programmatic problems:** the task set will be built in code: pick a formula from a passage,
  sample its inputs, compute the answer. It will not be teacher-authored.
- **Why:** the teacher's worked problems failed at 42%, and the failures were ill-posed set-ups,
  which a program can't write.

**Stage 4 (to do before it starts):** the judge benchmark. The audit's labelled records (52 defects
and the clean ones beside them) measure the Mistral judge's defect recall under three prompts:
- the current one;
- quote-the-supporting-sentence-then-verdict;
- full passage instead of the chunk.

Keep whichever catches the framing errors. Groundedness labels for DPO pairs were to come from this
judge, and it passed every defect the audit found.

## 2026-10-06: SFT set v1 ships as frozen; dataset hash 70f47740 (user decision)
**Context:** the Stage 3b gate expected the cleanup to land at ~2,650-2,750 records with a 5%
val split stratified by kind, drawn with the eval chunks' hash. The cleanup had already run
(entry above): 2,516 records, val drawn otherwise.
**Options:** ship v1 as frozen, re-split val to the spec, or re-split grouping by every chunk.
**Chose:** ship v1 as frozen. `sft_val` exists for one job, the epoch-1 vs epoch-2 loss comparison
(B4) on the same 80 records. A re-split would cost a contamination rerun, the tests and a fresh
`review.md` read, for no gain to that comparison.
- **Dataset hash** (sha256 of `data/sft/SHA256SUMS`; every Stage 3 run cites it, and `sft.py`
  refuses files that don't match):
  `70f47740bd973dc47f43d74c826b1bca3a82bb3e50379fe3ccc44179566ab4b6`. It covers
  `train.jsonl` `49b216d4…6876` (2,436) and `sft_val.jsonl` `8b210969…85` (80).
- **sft_val is loss-curve only, not a metric.** It is 5% of each format's holdable records
  (eval-seen records always train), grouped by fact chunk, picked by a sha256 rank (salt `sft-val`):
  grounded 27, replay 25, closed-book 15, abstain 12, definition 1. Nothing is reported on it
  except the loss curve and the final loss.
- **13 chunk ids reach both splits** through neighbour or distractor passages (grouping uses the
  fact chunk only). They shift epoch 1's and epoch 2's val loss the same way, so the rule is
  unbiased:

  | chunk | in val as | in train as |
  |---|---|---|
  | fema-p-2082-1:p456:c1 | distractor | source |
  | fema-p-2082-1:p457:c1 | source | distractor |
  | fhwa-nhi-15-047:p1671:c0 | distractor | distractor |
  | fhwa-nhi-15-047:p1688:c0 | distractor | distractor |
  | nist-gcr-14-917-30:p128:c0 | distractor | source |
  | nist-gcr-22-917-50:p465:c0 | distractor | distractor |
  | nist-gcr-22-917-50:p466:c0 | source | distractor |
  | nist-gcr-22-917-50:p468:c0 | source (a cited neighbour) | source |
  | usace-em-1110-2-1417:p75:c0 | distractor | distractor |
  | usace-em-1110-2-1421:p54:c0 | distractor | source |
  | usace-em-1110-2-2301:p56:c0 | distractor | distractor |
  | usace-em-1110-2-3006-2024apr22:p62:c0 | distractor | distractor |
  | usace-em-1110-2-3400:p87:c2 | distractor | distractor |

- **Other gate facts:** FEMA P-2335 has 26 records in the set (the eval had no wrong number, so no
  task change for it). No record exceeds 4,096 tokens (max 2,875); `tests/test_template.py` checks
  that on the training tensors. Contamination section 6 was rerun on eval v3 (next entry):
  0 leaked chunks, control 40/40, no eval question in a prompt, 0 unseen vocab terms defined.
**Revisit if:** the epoch-end val losses differ by less than the run-to-run spread of the noise
run's (then the rule is reading noise, and the write-up says so).

## 2026-10-06: Stage 3b pre-registration: trainer, checkpoint rule, merge check, reading (user decisions)
**Context:** written before any Stage 3 GPU launch, so nothing below can be fitted to the runs.
It covers the plan's B1-B8, with the user's decisions of today on four forks and two forced
changes.

**Eval v3 from Stage 3 on.** `make_tasks.py --task-version 3` rebuilt from the LLM cache (0 new
calls). It equals v2 minus qa-0003 and qa-0056 (a few-shot item's passage) and qa-1143 (an
identifier the 20% cap holds back once the set shrinks; now in `held_back.jsonl`). All three are
in the unseen half, so domain_qa is 322 (seen 167, unseen 155). Every other file is
byte-identical, as is the few-shot set, so the prompts don't change.
- `results/table.md` is frozen as `results/table_v2.md`.
- The new `table.md` (v3, plus a `false_abstain` column: grounded answers that use the abstain
  phrase, by rule) was rebuilt by rescoring every row from its saved generations, with 0 judge
  calls. Non-QA columns are unchanged.
- The tests pin v1 and v2 by hash and check that v3 rebuilds the committed files.

**Starting points and the bar on v3** (from the new table):

| run | qa_seen | qa_unseen | gold_lp_unseen | MMLU | GSM8K |
|---|---|---|---|---|---|
| cpt-8b-replay10 (start of sft-from-cpt) | 0.150 | 0.110 | -6.157 | 0.766 | 0.791 |
| base-8b-hf (start of sft-from-base) | 0.126 | 0.116 | -6.822 | 0.767 | 0.793 |

- **instruct-8b (the bar):** qa_acc 0.099, qa_ident 0.031, grounded_acc 0.898,
  cite_supported 0.787, halluc_rate 0.013, false_abstain 0.074, vocab_recall 0.786.
- **Stage 2 seed gaps on v3** (cpt-8b vs cpt-8b-seed1): qa_acc 1.6 points, qa_seen 2.4,
  qa_unseen 0.6, gold_lp_unseen 0.017 nats, MMLU 0.2, GSM8K 0.2.
- **Binomial SE of a half:** 2.5-2.8 points.

**Trainer (B2), as configured in `train/configs/sft.yaml`:**

| item | value |
|---|---|
| model | `Mistral3ForConditionalGeneration` (Stage 2's class; LoRA regex on `language_model`, vision tower frozen, no image ever fed), bf16 base, SDPA |
| starts | `checkpoints/cpt-8b-replay10`; control `checkpoints/base-8b-hf` (the Base-2512 weights re-saved by `merge.py`) |
| data | pre-tokenised by `train/sft_data.py` (mistral-common, the eval's `--chat` rendering): `input_ids` = `[1, 3, prompt, 4, answer, 2]`, `completion_mask` 0 through `[/INST]`; TRL's dataset prep off |
| loss | `completion_only_loss`, `loss_type: nll`; `num_items_in_batch` token mean over the 4 micro-batches |
| length | max_length 4,096, no packing, no padding-free |
| LoRA | r 64, alpha 128, dropout 0.05, q/k/v/o/gate/up/down, bias none: 178,257,920 trainable parameters, 238 modules |
| optimiser | AdamW (fused) (0.9, 0.999), eps 1e-8, weight decay 0, clip 1.0 |
| schedule | LR 1e-4, linear to 0, warmup 0.03 (5 steps) |
| batch | 8 x 4 = 32 sequences on one H100 (fallback 4 x 8) |
| length of run | 2 epochs, 77 steps each, 154 in total |
| precision | bf16, TF32, fp32 adapters, gradient checkpointing (non-reentrant) |
| eval and saves | eval every 50 steps and at each epoch end; save each epoch, both kept |
| seeds | seed / data_seed 0 (noise run: 1 / 1) |

- **Forced, not chosen:**
  - **`loss_type: nll`.** TRL 0.29.1 has no `chunked_nll` (only `nll` and `dft`).
  - **Pre-tokenisation is the main path.** TRL has no Mistral backend. transformers'
    `MistralCommonBackend` refuses a prompt + completion conversation in its default mode
    (`test`) and drops `return_assistant_tokens_mask` without a word.
- **Chosen today (user):** Stage 2's VLM class and SDPA, not `Ministral3ForCausalLM` +
  flash-attention-2. Same modules, and `merge.py` and `perplexity.py` stay unchanged. A text-only
  class renames the weights, and a key map is where a silent no-op merge would come from. The
  image has no flash-attn build. SDPA wastes some padding, minutes at 2,436 records.
- **Logged checks, each failing the run if false:**
  - the trainable count equals r * (in + out) over the 7 projections x 34 layers from the config,
    and every trainable name is a `lora_` weight under `language_model`;
  - at step 1, `num_items_in_batch` equals the completion tokens of the 4 micro-batches, and the
    logged loss is their token mean. The mean of per-micro-batch means is logged beside it (the
    Tulu 3 gradient-accumulation bug).

**B1 (blocks launch):**
- **`tests/test_template.py`, on the Mac, on the exact tensors** (`sft_data.encode` through TRL's
  collator):
  - the prompt ids equal mistral-common's `encode_chat_completion` (vLLM's `llm.chat` path) and
    `MistralCommonBackend(mode="agnostic")`;
  - the sequence is `[1, 3, ..., 4, ..., 2]`, with every other id a text token (>= 1,000, so no
    system-prompt, image or tool token);
  - labels are -100 on the prompt and the padding, and the masked span decodes to the completion
    exactly;
  - the batch keys are `input_ids` / `attention_mask` / `labels` only;
  - all 2,516 records are within 4,096 tokens and equal to `n_tokens`.
- **The smoke run (`smoke-sft`)** adds:
  - one real step on 32 records (the 8 longest as micro-batch 1), loss expected in 1.5-3 nats;
  - the vLLM prompt ids of the 5 template prompts, compared with the trainer's;
  - the no-op control on both starts;
  - the merge check on the 1-step adapter.

**No-op control (retrospective rule):** an untrained adapter (B = 0) on each start, merged by
`merge.py`, must equal the start tensor for tensor (`merge_check.py noop`,
`results/noop/<start>.json`).

**B4, checkpoint rule:** per run, epoch 2 unless the end-of-epoch-2 `val_loss` is above the
end-of-epoch-1 one; then epoch 1.
- `val_loss` is the token-mean NLL over all `sft_val` completion tokens, computed by `sft.py` at
  each epoch end (not the trainer's batch-weighted `eval_loss`, which is logged beside it).
- It is decided from the loss curve only, and written here before that run's merge and eval.

**B5, merge check (user decision: loss + agreement, not max-abs < 1e-2).** At logit magnitudes of
10-30, bf16's rounding step alone is 0.06-0.125, so a 1e-2 max-abs rule would fail every correct
merge.
- **Pass:** the merged model's `sft_val` token-mean loss is within 0.5% of start + adapter's
  (PEFT, unmerged, both bf16), and top-1 next-token agreement is >= 99% over the completion
  positions of 3 val records. This is the Stage 2 `ppl_val_slice` convention.
- **Printed:** max|merged − (start+adapter)|, max|(start+adapter) − start| and their ratio. It is
  flagged above 0.05: a merge error that isn't well under a tenth of the adapter's own effect is
  looked at even when the rule passes.
- **On failure:** the evals don't start (`modal_train.py` runs `mergecheck` before them).

**B6: per merged run, v3, `--chat`:**
- the KPI eval, and lm-eval (never chat);
- perplexity (reference only);
- latency with the new stop-before-cap rate and mean output tokens. The bench's prompt sample
  shifts slightly from Stage 2's, since the pool lost 3 items;
- `eval/sample.py`: eos (20 Stage 4 prompts x 4 at T 0.8, pass at 95% ending on `</s>`) and
  diversity (100 prompts at T 0.7: 50 held-out Tulu 3, 25 grounded, 25 vocab). Diversity also
  runs for instruct-8b.
- The judge is paced at 30 a minute, with the cache guard on.

**B7, reading (user decision: noise = max(seed gap, SE), the Stage 2 convention):**
1. **First line: sft-from-cpt vs sft-from-base on unseen `gold_lp`.** A paired bootstrap over the
   155 unseen items gives a 95% CI.
   - CPT bought something that survives SFT if the CI excludes 0 and |delta| > the noise. The
     noise is max(|sft-from-cpt − sft-from-cpt-seed1|, the paired per-item SE of the delta).
   - Otherwise the README says "CPT's value is not observable at this scale".
   - Unseen `qa_acc` is reported beside it and not argued over: at ~2.5-2.8 points of SE per half
     it will most likely sit inside the floor.
2. **Against the instruct-8b bar:** expected to beat it on identifiers and vocab and to match it
   on grounded and citation. No claim on general benchmarks.
3. **Guard, a format or mask bug:** an SFT run's `qa_seen` or `qa_unseen` below its own start's
   (0.150 / 0.110 from cpt-8b-replay10; 0.126 / 0.116 from base-8b-hf) by more than the noise.
   Response: re-run B1 before anything else. This replaces "below ~0.145", which is cpt-8b's
   overall qa_acc; replay10's unseen half is already under it.
4. **Guard, general benchmarks:**
   - MMLU or GSM8K down against the start by more than the noise + 1 point is flagged. The noise
     is 0.3 / 1.1, Stage 2's README column.
   - More than 3 points fails the run. The fallbacks are epoch 1 or a 5e-5 rerun, asked before any
     GPU.
5. **The seen vs unseen gap is reported.** Seen chunks were forced into the pool by design.
6. **Abstain targets:** halluc_rate < 0.20 (unanswerable abstained > 80%) and false_abstain < 0.05.
7. **Diversity:** distinct-4 and entropy within 10% of instruct-8b. Otherwise the write-up says
   the teacher's style collapsed the student, and notes it for Stage 4's sampling temperature.

**Runs:** sft-from-cpt, sft-from-base and sft-from-cpt-seed1, identical but for the start and the
seed. sft-from-cpt-lr2e-4 is optional, only after the three are evaluated.

**Failure modes, decided in advance:**
- **OOM:** 4 x 8.
- **Flat loss over 20 steps:** B1 first.
- **val_loss rising in epoch 1:** stop, then read the generations and the per-format val loss.
- **No `</s>` in the step-50 generations:** fix the data writer and refreeze; never patch at
  inference.
- **A failed merge check:** never served.
**Revisit if:** the smoke run's step-1 loss is outside 1.5-3 nats (template or mask), or its peak
memory leaves under ~8 GB (go to 4 x 8 before the main runs).

## 2026-10-06: Stage 3b amendments after the smoke run, before any training (user decisions)
**Context:** the smoke run (`smoke-sft`, `results/runs/smoke-sft/`) passed every check but one.
Its step-1 loss was 0.686 nats, under the pre-registered 1.5-3 band. The user read
`data/sft/review.md`, so gate G2 passes. Everything below was decided before any main run started.

**Step-1 check: per format, not the token mean.** The 1.5-3 band was a prior for recall
completions.
- **Why it failed:** applied to a token mean, it couldn't hold. The smoke step's first micro-batch
  was the 8 longest records, grounded and abstain, with 574 of the step's 1,049 completion tokens.
  Their answers copy from passages in the prompt or are a fixed sentence.
- **The amended check:**
  - closed-book and definition in 1.5-3;
  - grounded and abstain expected under 0.5.
- **Smoke values after one step:** closed-book 1.65, definition 2.02, grounded 0.22, abstain 0.39,
  replay 0.70. Pass.
- **Ruled out:** an all-zero mask would give no loss tokens, and a wrong template would push the
  loss over 6.

**Eval every 50 steps logs sft_val loss per format** (already in `sft.py`: `val_loss_by_format`
at every evaluation, step 0 included), not only the mean.

**B4, amended:** epoch 2 unless the closed-book or the definition `sft_val` loss rose from the end
of epoch 1 to the end of epoch 2. Then epoch 1.
- **Why:** overfitting shows in the recall formats first, and a rise there could sit under a
  falling grounded or replay loss in the mean. It replaces the overall-`val_loss` rule of the
  pre-registration entry.
- Still decided from the loss curve only, and written here before that run's merge and eval.
- **Implemented:** `train_summary.json` has `val_loss_by_format_epoch_end`, and
  `report.py epoch_rule` applies the rule.
- **The size of what it reads (measured after the user's decision, reported to the user):** sft_val
  holds 15 closed-book records (86 completion tokens) and 1 definition record (22 tokens). So the
  rule rests on 108 tokens, the definition half of it on one answer.

**Where the gradient goes.** The training loss is token-weighted.

| format | share of the 179,332 train completion tokens |
|---|---|
| replay | 74.9% |
| grounded | 15.6% |
| closed-book | 4.6% |
| definition | 4.3% |
| abstain | 0.7% |

- So replay, not grounded copy, carries most of the gradient. The 55% copy share quoted first was
  the smoke step's longest-first micro-batch, not the set.
- Per-format loss weighting goes in next steps, not Stage 3 (README).

**Merge ratio, smoke 0.25: no action; expectation pre-registered.**
- **Why 0.25 is not alarming:** one Adam step moves each LoRA weight by about the LR, so the merged
  delta sits at bf16's resolution, and rounding error and adapter effect come out the same size.
- **Ruled out:** a merge bug, by the no-op control (531 of 531 identical, both starts). The pass
  rule held: val loss +0.06%, top-1 99.66%.
- **For the trained runs:**
  - the mean-abs ratio (`ratio_mean`, added to `merge_check.py` after the smoke run) is expected
    to be well under the smoke run's;
  - top-1 agreement at or above 99.66%.

  The smoke's mean ratio wasn't measured, so its max ratio (0.25) is the stand-in, unless the
  smoke merge check is rerun.
- If a trained run comes in worse on either, that is a finding, not a rounding story.
