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

## 2026-09-27: Stage 1 corpus card: 242 documents, 20.0M Tekken tokens
**What:** the CPT corpus. `make data` runs download, extract, filter, dedup, pii, split, replay,
tokenizer_coverage and stats in order. Every number below comes from
`data/processed/stats.json` via `stats.py`, and `data/sources.csv` records each file's URL,
sha256, pages and final tokens. Train is 18.8M tokens (230 docs), val 1.2M (12 docs, 6.1%),
replay 1.9M FineWeb-Edu tokens. The processed files are not committed; they rebuild from
sources.csv plus the scripts.

**Sources (Stage 1 expansion, 52 -> 249 rows):**
- USACE Engineer Manuals, every EM 1110-2 (civil works) and EM 1110-1 (general engineering) on the
  live index. EM 1110-3 (22 mobilization manuals from 1984: pavements, water supply) was skipped.
  Akamai returns 403 to every script, so the index pages and PDFs were fetched in Chrome (DevTools
  MCP). download.py can't refetch them, which is also the case for FEMA.
- FHWA steel and concrete bridge index pages (crawl_index.py).
- NIST NEHRP briefs 5-12 (Brief 14 is not published) and 15 NEHRP GCR reports (ATC).
- FEMA: P-58-1/-2, P-1050-2, P-2006, and 35 technical publications from the Building Science
  earthquake library (brochures, checklists, posters and forms skipped). P-695 and P-751 aren't in
  FEMA's library; the substitutes are P-795 (P-695's companion methodology) and the 2020 NEHRP
  Design Examples vols. 2-3.
- NASA-STD-5002B and 5019A, so NASA has a non-eval document to hold out.
- The spec's 20-50M target needed more than the three named index pages. EM 1110-2 yielded
  8.5M clean tokens, not 15-25M: many manuals are scanned, table-heavy, or PDF portfolios.
**Excluded for copyright** (`extract.py` flags ©/"copyright"/"all rights reserved" in the first 5
pages; all 31 were hand-checked, 27 in the final corpus plus the 4 excluded here):
- FHWA-HIF-19-102: © Lehigh University, all rights reserved.
- FEMA's 2025 seismic evaluation guidance: reproduces ICC code text "proprietary to and
  copyrighted by" ICC, the same reason as the ASCE 7/AISC exclusion.
- NIST GCR 15-917-35/36: © Fire Protection Research Foundation, and off-domain (cooking fires).
- Photo credits, "figures reproduced with permission" notes, and BSSC's "be alert to patent and
  copyright concerns" boilerplate were kept.
**Rejected on download:**
- Three files under 200 KB.
- FEMA P-58-4 and P-58-5: FEMA serves them truncated (7.2 of 8.0 MB and 3.5 of 13.3 MB against
  the length their linearisation headers declare), so they have 0 readable pages.

**Sources and pages** (extract.py; pages under 200 characters dropped, no OCR)

| publisher | docs | pages | kept | dropped: image-only | dropped: blank | tokens extracted | tokens final |
|---|---|---|---|---|---|---|---|
| FEMA | 45 | 14,058 | 12,643 | 879 | 536 | 7,582,974 | 4,243,632 |
| FHWA | 65 | 12,587 | 11,577 | 750 | 260 | 5,834,309 | 3,838,621 |
| NASA | 4 | 319 | 310 | 7 | 2 | 152,143 | 114,123 |
| NIST | 28 | 4,595 | 4,432 | 25 | 138 | 2,648,730 | 1,724,316 |
| USACE | 102 | 27,243 | 24,168 | 2,592 | 483 | 14,111,797 | 10,112,419 |
| **total** | 244 | 58,802 | 53,130 | 4,253 | 1,419 | 30,329,953 | 20,033,111 |

249 documents in sources.csv, 244 accepted (3.14 GB); rejected: usace-em-1110-2-1304-2021 (under 200 KB), usace-em-1110-2-2102 (under 200 KB), usace-em-1110-1-4011 (under 200 KB), fema-p-58-4 (no readable pages (truncated or corrupt file)), fema-p-58-5 (no readable pages (truncated or corrupt file)).
Header/footer lines removed: 120,491. Likely scanned (>50% of pages dropped): fhwa-sbdh-v03, usace-em-1110-2-1424, usace-em-1110-2-3200, fema-nehrp-examples-v3, fema-p-2192, fema-p-1100-2c.

**Token funnel** (Tekken tokens)

| step | docs | tokens | removed |
|---|---|---|---|
| extracted | 244 | 30,329,953 |  |
| quality filter | 242 | 20,913,702 | 31.0% |
| exact dedup | 242 | 20,913,702 | 0.0% |
| near dedup | 242 | 20,035,325 | 4.2% |
| train | 230 | 18,818,589 |  |
| val | 12 | 1,214,522 | 6.1% of tokens |
| replay (FineWeb-Edu) | 1,720 | 1,881,859 | 10% of train |

**Quality filter** (filter.py; first failing rule counted)

| rule | paragraphs dropped | tokens dropped |
|---|---|---|
| alpha_ratio | 194,178 | 5,611,924 |
| word_length | 40,409 | 368,395 |
| short_lines | 35,989 | 1,280,218 |
| reference_list | 6,812 | 403,568 |
| symbol_ratio | 3,312 | 37,301 |
| numbered_lines | 2,819 | 167,048 |
| **paragraphs in / out** | 725,916 / 441,223 |  |

Documents dropped: fhwa-hif16010 (dictionary, 47,164 words, dictionary 0.649), fema-nehrp-examples-v3 (min_words, 540 words, dictionary 0.937). Dictionary ratio of kept documents: min 0.743, median 0.926.

**Deduplication** (dedup.py)

Exact duplicate documents: none. Near-duplicate paragraphs: 32,996 of 441,223; tokens 20,913,702 -> 20,035,325 (4.2%).

Most-duplicated across documents:

| docs | copies removed | paragraph |
|---|---|---|
| 57 | 72 | DEPARTMENT OF THE ARMY U.S. Army Corps of Engineers |
| 57 | 56 | The Federal Highway Administration (FHWA) provides high-quality information to serve Gover |
| 37 | 36 | Approved for public release; distribution is |
| 28 | 27 | 8. Performing Organization Report No. |
| 27 | 26 | Any opinions, findings, conclusions, or recommendations expressed in this publication do n |
| 25 | 24 | U.S. Department of Commerce |
| 25 | 24 | Notice This document is disseminated under the sponsorship of the U.S. Department of Trans |
| 24 | 23 | FOREWORD This handbook covers a full range of topics and design examples intended to provi |
| 23 | 22 | Form DOT F 1700.7 (8-72) Reproduction of completed pages authorized |
| 22 | 21 | 2. Government Accession No. 3. Recipient’s Catalog No. |

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

**PII** (pii.py): EMAIL 56, PHONE 177, ID 0

**Tokenizer fit** (tokenizer_coverage.py, full report in `data/processed/tokenizer_coverage.md`; tokens per whitespace word)

|  | mistralai/Ministral-3-8B-Base-2512 | mistralai/Ministral-3-8B-Instruct-2512-BF16 |
|---|---|---|
| corpus tokens | 20,033,111 | 20,033,111 |
| fertility: domain corpus | 1.438 | 1.438 |
| fertility: FineWeb-Edu (~1M tokens) | 1.338 | 1.338 |
| fertility: vocab_eval terms (303) | 1.433 | 1.433 |
| fertility: tfidf_top terms (500) | 1.112 | 1.112 |
| fertility: probes terms (18) | 3.207 | 3.207 |
| term words with 4+ tokens | 34 / 842 | 34 / 842 |

Base and Instruct encode identically: True.

| designation | tokens |
|---|---|
| ASCE 7-22 | 7 |
| ASCE 7-16 | 7 |
| EM 1110-2-2104 | 13 |
| NEHRP | 3 |
| AASHTO LRFD | 7 |
| kip-ft | 4 |
| ksi | 1 |
| ASTM A709 | 6 |
| A709 Grade 50W | 9 |
| HPS 70W | 6 |
| f'c | 3 |
| P-delta | 3 |
| Cs = SDS/(R/Ie) | 8 |
| orthotropic | 2 |
| electroslag | 3 |
| austenitic | 3 |
| martensitic | 3 |
| Charpy V-notch | 5 |

Worst-fragmented term words: 1110-2-2104 (12), SDS/(R/Ie) (6), strong-column/weak-beam (6), (f’c) (5), 7-16 (5), 7-22 (5), sub-diaphragm (5), (CIF) (4), (EDO) (4), (LFRS) (4), (RBS) (4), (Ωv) (4), 100 (4), 50W (4), 70W (4), A709 (4), AASHTO (4), Diaphragm (4), Earthquake) (4), HL-93 (4), I-girder (4), No-decompression (4), Timoshenko (4), capacity-demand-diagram (4), contraflexure (4)

**Split and packing** (split.py)

| publisher | train docs | train tokens | val docs | val tokens |
|---|---|---|---|---|
| FEMA | 42 | 4,131,446 | 2 | 112,186 |
| FHWA | 61 | 3,758,566 | 3 | 80,055 |
| NASA | 3 | 95,445 | 1 | 18,678 |
| NIST | 27 | 1,639,808 | 1 | 84,508 |
| USACE | 97 | 9,193,324 | 5 | 919,095 |

Val documents: fema-p-1100-2a, fema-p-2018, fhwa-hif17020, fhwa-hif18044, fhwa-hif18047, nasa-std-5002b, nist-gcr-12-917-21, usace-em-1110-1-1804, usace-em-1110-2-1906, usace-em-1110-2-1908, usace-em-1110-2-2610-final-18mar2025, usace-em-1110-2-3506. 52 eval documents held in train. Packed at 4,096: 4,594 train / 296 val sequences (5,054 with replay); 17.9 optimizer steps per epoch at ~1M tokens/step.

**Decisions:**
- **PDF portfolios:** USACE EM 1110-2-1100 (all 6 parts), -1424 and -1701 are a one-page "open in
  Acrobat" cover with the manual as embedded PDFs. They're read through their parts
  (common.pdf_parts): 4,728 pages that were first misread as scanned.
- **No OCR:** 4,253 image-only pages (7.2%) and 1,419 near-blank pages are dropped, and 6 documents
  lose most of their pages. OCR is the first lever for more tokens, but scanned 1980s manuals would
  bring OCR noise that the dictionary rule exists to keep out.
- **Filter tuned from dropped_samples.jsonl:** the literal line-count rules removed 28% of
  characters on the seed set, half of it real content (bulleted requirement lists, worked-example
  "where:" blocks, two-line paragraphs). Three changes, all documented in filter.py:
  - marker-only lines ("•", "1.") are joined to their item;
  - the line-ratio rules need 3+ lines;
  - short_lines counts words, not lines.
  alpha_ratio is kept as specified: it's the largest rule (5.6M tokens) and its token-weighted
  samples are dot-leader TOCs, garbled OCR, XML and equation fragments. The dictionary threshold
  was raised to 0.70 because FHWA-HIF-16-010 (0.649) is half XML object listings, and kept
  documents otherwise score 0.74+.
- **Paragraph-level near-dedup, not document-level:** exact dedup found no duplicate documents.
  Paragraph MinHash removes 4.2%: federal boilerplate shared across 20-57 documents (the FHWA
  quality statement, USACE letterhead, distribution statements, DOT report-documentation fields),
  plus running headers the 30%-of-pages rule misses (per-chapter titles, NIST's "available free of
  charge" line). Document-level dedup would keep all of that.
- **Document-level split, eval documents in train:** val perplexity then measures generalisation
  to unseen documents of the same kinds, and the KPI eval measures what CPT absorbed from documents
  it saw. All 52 seed documents are eval sources, so the 12 val documents all come from the
  expansion: max(1, 5%) per publisher, chosen by sha256(slug). Dedup runs before the split, so no
  near-duplicate paragraph sits on both sides.
- **No vocabulary extension:**
  - Tekken spends 1.07x more tokens per word on this corpus than on FineWeb-Edu (1.438 vs 1.338).
  - The top-500 TF-IDF domain terms average 1.11 tokens per word.
  - Only 34 of 842 term words take 4+ tokens: designations and digit strings (EM 1110-2-2104 is
    13 tokens because Tekken splits digits by design), parenthesised acronyms, and hyphenated
    compounds.
  - New embeddings trained on 20M tokens would cost more than they save. Base and Instruct encode
    identically.
- **PII:** three regexes (56 emails, 177 phone/fax numbers, 0 SSN-shaped), all spot-checked as true
  positives; author names are kept for citations. At Forge scale this is a Presidio-class NER pass
  with client-specific entity lists.
- **Packing:** 4,594 train sequences of 4,096 (5,054 with replay), about 18 optimizer steps per
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
