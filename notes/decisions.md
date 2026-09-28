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
  P-2091, P-2355, E-74; brochures, checklists, posters and forms skipped).
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
