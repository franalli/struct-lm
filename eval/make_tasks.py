"""Build the four domain eval tasks from the train corpus with Mistral Large 3 (GEN_MODEL).

Usage:  MISTRAL_API_KEY=... python eval/make_tasks.py
Input:  data/processed/chunks.jsonl (from data/scripts/extract.py --chunks: the documents
        pinned in eval/tasks/eval_docs.txt)
Output: eval/tasks/
          domain_qa.jsonl     {id, question, answer, answer_type, tolerance, source_chunk}
          grounded.jsonl      {id, question, context:[{chunk_id,text}x4], gold_chunk_ids}
          vocab.jsonl         {id, term, definition, source_chunk}
          adversarial.jsonl   {id, question, context:[{chunk_id,text}x3], why_unanswerable}
          fewshot.jsonl       3 QA pairs used in the domain_qa prompt, excluded from domain_qa
                              (v3: with every item from their passages)
          eval_chunk_ids.txt  every chunk id an eval item was built from (Stage 3 must not
                              generate SFT data from these, except the "seen" half that
                              eval/sft_split.py lists in sft_seen_chunks.txt: run it after this)

domain_qa grows in three samples, each taken after everything before it so no earlier item moves:
the main set (ids from 1), --n-qa-extra (from 501) and --n-qa-extra2 (from 1001, 2026-10-04).
Every item is hand-reviewed before any model generates on it (notes/eval_review_rubric.md).

Everything is seeded; re-running regenerates the same sample of chunks. Generated items are
verified by a second GEN_MODEL pass. Hand-check ~30 QA items afterwards; if more than
3 are bad, tighten the prompt and regenerate.

Cost: ~700 GEN_MODEL calls.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mistralai.client import Mistral
from prompts import VOCAB_SHOTS
from qa_rules import answer_kind, is_locator

GEN_MODEL = "mistral-large-2512"  # Mistral Large 3, pinned; must match judge.JUDGE_MODEL
client = Mistral(
    api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=120_000
)  # no default timeout: one dead socket hung a run for hours


FAILED_CALLS = 0  # calls that exhausted retries; their chunks silently yield no items


# Disk cache so iterating on filters/prompts only pays for calls whose prompt changed.
# Keyed on (model, prompt); failed calls ({}) are never cached, so they retry next run.
CACHE_PATH = pathlib.Path(__file__).parent / ".cache" / "llm_cache.jsonl"
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
if CACHE_PATH.exists():
    for _line in CACHE_PATH.open():
        _rec = json.loads(_line)
        _cache[_rec["key"]] = _rec["out"]


def llm_json(prompt: str, retries: int = 6) -> dict:
    """One GEN_MODEL call in JSON mode, through the disk cache. Returns the parsed object, or {}
    if every retry failed; callers read fields with .get() defaults, so {} yields no items.
    Thread-safe: pmap runs this from several workers, and cache writes take _cache_lock."""
    key = hashlib.sha256(f"{GEN_MODEL}\n{prompt}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    out = _llm_json_uncached(prompt, retries)
    if out:
        with _cache_lock:
            _cache[key] = out
            CACHE_PATH.parent.mkdir(exist_ok=True)
            with CACHE_PATH.open("a") as f:
                f.write(json.dumps({"key": key, "out": out}, ensure_ascii=False) + "\n")
    return out


class Pacer:
    """Client-side pacing shared by every worker: calls start at least `interval` seconds apart.
    main() pins the interval (and its floor) just under the key's requests-per-minute limit
    (--rpm); a 429 still stretches it x1.25, at most once per 10 s, and 10 successes in a row
    shrink it back toward the floor. Without pacing, workers burst into the limit together and all
    sleep 15-30 s (2026-10-04: 299 retries in 37 minutes, ~3/4 of worker time asleep)."""

    def __init__(self, interval: float = 1.0, floor: float = 0.25, ceiling: float = 4.0):
        self.interval, self.floor, self.ceiling = interval, floor, ceiling
        self.next_start, self.ok, self.last_slow, self.calls = 0.0, 0, 0.0, 0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        time.sleep(start - now)

    def result(self, throttled: bool) -> None:
        with self.lock:
            now = time.monotonic()
            if throttled:
                self.ok = 0
                if now - self.last_slow > 10:
                    self.interval, self.last_slow = min(self.ceiling, self.interval * 1.25), now
            else:
                self.ok, self.calls = self.ok + 1, self.calls + 1
                if self.calls % 50 == 0:
                    print(f"  pacer: {self.calls} calls, interval {self.interval:.2f} s")
                if self.ok >= 10:
                    self.interval, self.ok = max(self.floor, self.interval * 0.9), 0


# Mistral's limit for this key is 30 requests a minute (x-ratelimit-limit-req-minute, checked
# 2026-10-04; tokens are 800k a minute, never the constraint). Pace just under it: every 429 is a
# request spent for nothing. --rpm overrides when the limit changes.
RPM = 30
PACER = Pacer(interval=60 / RPM * 1.03, floor=60 / RPM * 1.03)


def _llm_json_uncached(prompt: str, retries: int) -> dict:
    """The API call itself, with retries. Every failure mode (rate limit, network error, invalid
    JSON) is retried the same way; after the last one, the failure is counted in FAILED_CALLS
    and reported at the end of the run by report_failures()."""
    global FAILED_CALLS
    for attempt in range(retries):
        PACER.wait()
        try:
            r = client.chat.complete(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                # A little variety in question phrasing; the cache makes each prompt's output
                # fixed after the first run, so reruns are still reproducible.
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            message = r.choices[0].message
            content = message.content if message else None
            if not isinstance(content, str):  # empty or non-text reply: retry like any error
                raise TypeError(f"model returned no text content: {content!r}")
            PACER.result(throttled=False)
            return json.loads(content)
        except Exception as e:  # noqa: BLE001  any failure (rate limit, network, bad JSON) is retried
            print(f"  retry {attempt + 1}: {str(e)[:160]}")
            if "429" in str(e):
                PACER.result(throttled=True)  # the pacer slows every worker; a short pause here
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(2**attempt)
    FAILED_CALLS += 1
    return {}


WORKERS = 8  # enough calls in flight for the Pacer's rate (each call takes seconds); --workers


def pmap(fn, items, workers=None):
    """Parallel map over threads (the work is network-bound). Results come back in input order,
    so callers zip them with their inputs."""
    with ThreadPoolExecutor(workers or WORKERS) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------- generators ---
def answer_type_of(answer: str) -> str:
    """One number -> "numeric" (scored by value, 2% tolerance), so "95%" matches "95 percent" and
    "0.7EcIg" matches "0.70EcIg". Several numbers (article/section/document ids such as
    5.10.9.3.2, ACI 318-19, EM 1110-2-6051) -> "exact". The generator's own label was
    inconsistent: 57 of 133 "exact" answers in round 1 were plain numbers."""
    return "numeric" if len(re.findall(r"\d[\d,]*(?:\.\d+)?", answer)) == 1 else "exact"


def gen_qa(chunk: dict) -> list[dict]:
    """Step 1 of domain_qa: up to 2 closed-book question/answer candidates from one chunk.

    This is the generous step. Candidates then go through, in order: the regex pre-filters
    (CONTEXT_BOUND, is_trivia, ungrounded_numbers, is_reference_list), the blind check
    (is_standalone), answer verification with the passage (verify_qa), dedup_questions, and
    the hand-review rejects. The model's own answer_type is overwritten by answer_type_of()."""
    out = llm_json(f"""You are building an exam from an engineering document. From the passage below, write up to 2
questions whose answer is a single short fact stated explicitly in the passage: a numeric value with
its unit, a defined term, a named method or component, or a specific requirement.

Rules:
- CLOSED BOOK: the question will be asked WITHOUT the passage, to someone who knows this document
  collection (FHWA, FEMA, USACE, NIST, NASA structural documents). It must identify exactly one fact
  on its own: name the structure type, load condition, code, material or quantity it refers to.
- Never refer to anything the reader can't see: no "the figure/table/equation/example/model/passage",
  no values computed in a specific worked example (e.g. a particular girder's section modulus,
  a bolt force in one splice design), no relative time ("at that time", "currently").
- When asking which article, section or clause governs something, name the standard or document
  (e.g. "Which AASHTO LRFD article...", "In ASCE 7...", "AWS D1.5 clause..."). Never ask for a
  document's own equation, figure or section numbering.
- The question must not be answerable by general engineering common sense; it must need these documents.
- The answer is at most 6 words. For numbers include the unit.
- Return {{"items": []}} if the passage has no such fact (worked calculations, tables of contents,
  boilerplate, references).

Return JSON: {{"items": [{{"question": "...", "answer": "...", "answer_type": "numeric" or "exact"}}]}}

Passage ({chunk["chunk_id"]}):
{chunk["text"]}""")
    items = [
        i
        for i in out.get("items", [])
        if isinstance(i, dict) and i.get("question") and i.get("answer")
    ]
    for i in items:
        i["source_chunk"] = chunk["chunk_id"]
        i["answer_type"] = answer_type_of(i["answer"])
    return items[:2]


def verify_qa(item: dict, chunk_text: str) -> bool:
    """Last LLM filter for domain_qa, run with the passage visible: is the answer right, unique,
    faithfully framed, and a general fact rather than one worked example's number? A missing or
    failed response counts as not ok, so a failed call drops the item instead of keeping it."""
    out = llm_json(f"""Passage:
{chunk_text}

Question: {item["question"]}
Proposed answer: {item["answer"]}

Is the proposed answer (a) stated explicitly in the passage, not inferred or assembled from a
reference list, (b) the only reasonable short answer to the question as written, and (c) is the
question's framing faithful to the passage? A question that calls something "required", "minimum",
"maximum" or "permitted" when the passage only describes, suggests or reports it is NOT faithful.
(d) Is the answer a general rule, requirement, definition, recommendation or finding, NOT a value
computed, chosen or assumed for one worked example or analysis in the passage (e.g. a stress
computed for one girder, the electrode or software used in one example, an example's material
property table)? If the passage is a worked example and the answer is specific to it, return false.
Return JSON: {{"ok": true or false, "reason": "one sentence"}}""")
    return bool(out.get("ok"))


# Cheap pre-filter for closed-book QA: phrasings that point at something only the passage has.
CONTEXT_BOUND = re.compile(
    r"\b(figure|fig\.|table|equation (?:number|no)|eq\. ?\d|this (?:example|design|bridge|model|"
    r"passage|document|section)|the (?:example|model|passage|design example)|at that time|"
    r"girder g\d|section [a-z]-\d|equation \d|"
    r"(?:fhwa|nhi|hif)[- ][\w-]{0,25}\s+sections?\b)\b",  # a manual's own section numbers
    re.IGNORECASE,
)


# Questions about notation, units, abbreviations, names or publication details: they test
# document trivia rather than engineering knowledge, and are often answerable by guessing.
TRIVIA = re.compile(
    r"\b(symbol|denoted|notation|unit (?:is |are )?(?:used )?(?:for|of)|what units?\b|abbreviat\w*|"
    r"number of pages|colloquial\w*|designated as|stands? for|acronym|known as|publication year|"
    r"year (?:was|is) .{0,40}published|published in (?:what|which) year)\b",
    re.IGNORECASE,
)


# answers that are just an edition or a citation year ("7th Edition", "AASHTO (2014)")
CITATION_ANSWER = re.compile(
    r"^\s*(?:(?:\d+(?:st|nd|rd|th)|\d{4})\s+edition|[A-Z][A-Za-z/&. ]*?\s?\(?(?:19|20)\d{2}[a-z]?\)?)\s*$",
    re.IGNORECASE,
)


# "Name one ..." questions have several correct answers, but exact_match only knows one.
LIST_QUESTION = re.compile(
    r"\b(name one|one example|what is one|give one|list one|an example of)\b", re.IGNORECASE
)
GREEK = re.compile(r"[Ͱ-Ͽ]")  # the Greek and Coptic Unicode block, for bare-symbol answers


REF_CITATION = re.compile(  # reference-list shapes only; a bare inline "(2014)" doesn't count
    r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)\s*\.|\b(?:19|20)\d{2}[a-z]?\.\s+[A-Z\u201c\"]|"
    r"\b(?:Vol|pp|No)\.\s?\d|,\s*(?:Chicago|Washington|Reston|Miami|New York|Farmington Hills), [A-Z]{2}\b"
)


def is_reference_list(passage: str) -> bool:
    """Bibliography chunks yield trivia ("AWS A5.36/A5.36M:2016", "AISC 2010a", a damping value from
    a paper title). Hand-check: reference chunks score 3.8-8.6 per 100 words, prose <= 0.6."""
    return len(REF_CITATION.findall(passage)) / max(1, len(passage.split())) * 100 >= 1.0


def ungrounded_numbers(item: dict, passage: str) -> list[str]:
    """Numbers in the answer that don't occur in the passage. PyMuPDF scrambles some equations
    (e.g. "( . b y p d f l 30 0 ="), and the generator then 'reconstructs' a formula that isn't
    there; the LLM verifier accepted one such answer (0.3 d_b f_y) in the hand-check."""
    nums = lambda s: [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]
    text = passage.replace(",", "")
    return [n for n in nums(item["answer"]) if n not in text]


def is_trivia(item: dict) -> bool:
    """Notation/unit/abbreviation questions, or answers already spelled out in the question
    (hand-check: "symbol for shear modulus -> G", "column weaker than beam -> weak-column/strong-beam").
    Compares ALL alphanumeric tokens incl. numbers and single letters, so "Which AASHTO LRFD
    article...? -> Article 3.6.3" is kept: "3", "6" aren't in the question."""
    if TRIVIA.search(item["question"]) or CITATION_ANSWER.match(item["answer"]):
        return True
    if LIST_QUESTION.search(item["question"]):  # "name one ..." has several right answers
        return True
    ans = item["answer"]
    if GREEK.search(ans) and len(re.sub(r"[\s_{}]", "", ans)) <= 5 and not re.search(r"[=.]", ans):
        return True  # bare symbol (θmax, α1), not a value like "φ = 0.9"
    # the answer is the passage's own document ("Which FEMA document ...? -> FEMA P-2012")
    alnum = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if alnum(item["answer"]) == alnum(item.get("source_chunk", "").split(":")[0]):
        return True
    toks = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    ans = toks(item["answer"])
    return bool(ans) and ans <= toks(item["question"])


def is_standalone(question: str) -> tuple[bool, str]:
    """Blind check: the verifier sees ONLY the question, as the model under test will.
    (verify_qa sees the passage, so it resolves ambiguity from context and can't catch this.)"""
    out = llm_json(f"""A closed-book exam about US federal structural engineering documents (FHWA, FEMA, USACE,
NIST, NASA) contains the question below. You see only the question, exactly as the examinee will.

Question: {question}

The examinee has studied these documents and the codes they cite (AASHTO LRFD, ASCE 7, ACI 318,
AISC, AWS D1.5, TMS 402, ...). Needing that knowledge is the POINT of the exam: naming a code,
article, section or document in the question is good, not a problem.

Is the question self-contained, i.e. does it pin down exactly one fact without also seeing a
particular passage, figure, table, worked example or model? Answer false only if ANY of these hold:
- it refers to something the examinee can't see (a particular girder, splice, figure, example,
  "the model", "the four categories"), or uses relative time;
- it asks for a document's internal equation/section numbering without naming the document;
- it could reasonably have several correct answers across different codes, documents or contexts
  (e.g. "the unit weight of concrete" without saying which document or concrete);
- a structural engineer would know the answer without these documents: standard material
  properties implied by the name (the yield of "Grade 50" steel), textbook definitions, or
  unit conventions;
- it asks what a document is about or what its title says;
- it asks about a modeling choice, assumption, coefficient or value used in ONE particular
  analysis, design example or formula, even if it names the document ("In FHWA-HIF-18-046, what
  element type models the tendons?" is false: that is one example's choice, not a requirement).
Return JSON: {{"standalone": true or false, "reason": "one sentence"}}""")
    return bool(out.get("standalone")), str(out.get("reason", ""))


def dedup_questions(items: list[dict]) -> list[dict]:
    """Same answer + mostly the same question words = the same fact sampled from two pages."""
    toks = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    kept: list[dict] = []
    for it in items:
        if any(
            it["answer"].strip().lower() == k["answer"].strip().lower()
            and len(toks(it["question"]) & toks(k["question"]))
            / max(1, len(toks(it["question"]) | toks(k["question"])))
            > 0.5
            for k in kept
        ):
            continue
        kept.append(it)
    return kept


def gen_grounded_question(chunk: dict) -> str:
    """One open-book question answerable from this chunk (the gold passage). Returns "" on
    failure, and main() skips those. No answer is stored: the judge grades against the gold
    passage directly (judge.grounded_rubric), so there's no reference answer to get wrong."""
    out = llm_json(f"""Write one question a practicing structural or civil engineer would ask, whose complete answer is
contained in the passage below and takes one to three sentences. It should ask about a requirement,
procedure, limit or rationale, not a trivia value. Do not write "according to the passage".
Return JSON: {{"question": "..."}}

Passage:
{chunk["text"]}""")
    return out.get("question", "")


def term_key(term: str) -> str:
    """Dedup key: 'stirrup'/'stirrups', 'line-girder'/'line girder', 'Reduced Beam Section (RBS)'/
    'reduced beam section' collide. (The generation-time key was only lowercased.)"""
    t = re.sub(r"[-/]", " ", re.sub(r"\s*\(.*?\)\s*", " ", term.lower()))
    return " ".join(
        w[:-1] if w.endswith("s") and len(w) > 3 and not w.endswith("ss") else w for w in t.split()
    )


def term_in_passage(term: str, passage: str) -> bool:
    """The vocab term (or, for "Full Name (ABBR)", either part) must occur in its source passage,
    allowing a plural/singular s. Otherwise the definition is the generator's own knowledge, not
    the corpus's usage (hand-review: "fish-belly flange", "nesting", "fixed and float supports"...)."""
    text = re.sub(r"\s+", " ", passage).lower()
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", term)
    variants = [m.group(1), m.group(2)] if m else [term]
    for v in variants:
        v = re.sub(r"\s+", " ", v.strip().lower())
        if len(v) < 2:
            continue
        stem = v[:-1] if v.endswith("s") and len(v) > 3 else v
        if re.search(rf"(?<![a-z0-9]){re.escape(stem)}s?(?![a-z0-9])", text):
            return True
    return False


def gen_terms(chunk: dict) -> list[dict]:
    """Up to 4 {term, definition, source_chunk} from one chunk. The definition is the reference
    the judge compares against, so the prompt asks for the passage's usage rather than a generic
    textbook one. main() then removes few-shot terms, duplicates, and terms not in the passage."""
    out = llm_json(f"""List up to 4 domain-specific technical terms that the passage below defines or uses in a specific
technical sense. Exclude generic words, units, organisation names and document numbers. For each,
give a one-sentence definition faithful to how the passage uses it.
Return JSON: {{"terms": [{{"term": "...", "definition": "..."}}]}}

Passage:
{chunk["text"]}""")
    terms = [
        t
        for t in out.get("terms", [])
        if isinstance(t, dict) and t.get("term") and t.get("definition")
    ]
    for t in terms:
        t["source_chunk"] = chunk["chunk_id"]
    return terms


# Rotated across items so unanswerable questions have the same mix of types as grounded ones;
# the first generation was 70% "What is the maximum allowable ..." (vs 2% in grounded): a phrasing
# shortcut to abstention.
ADV_QTYPES = [
    "a requirement or condition (what must be done, or when a provision applies)",
    "a procedure or method (how something is determined, checked or designed)",
    "a rationale (why a provision, practice or assumption exists)",
    "a limit, value or factor",
    "a comparison or distinction between two cases, systems or conditions",
    "a reference (which code, article, section or document governs something)",
]


def gen_adversarial(context: list[dict], qtype: str) -> dict:
    """One plausible question the passages do NOT answer, of the given type (ADV_QTYPES).
    Returns {question, why_unanswerable} or {} on failure. Generators often write questions
    that are actually answerable, so each one is checked afterwards by is_answerable() (a
    separate call that doesn't see the generator's reasoning) and premise_rebutted()."""
    passages = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in context)
    out = llm_json(f"""Write one question that a practicing engineer might plausibly ask about the topic of the passages
below, phrased as if the answer were in them, but whose answer is NOT contained in the passages.
It must sound like a natural question, not a trick.

Question type (pick the one that fits what the passages leave out): {qtype}
Phrase it exactly the way you would phrase an ANSWERABLE question of that type. Do NOT use the
words "allowable", "specific", "exact", "precise" or "numerical", and do not hint that the answer
is missing: the examinee must have to read the passages to discover it isn't there.
The answer must not be derivable from the passages either (no interpolation or arithmetic on values
they give).
Return JSON: {{"question": "...", "why_unanswerable": "one sentence"}}

Passages:
{passages}""")
    return out


# Sentences where a document says outright that something isn't specified; see premise_rebutted().
PREMISE_REBUTTAL = re.compile(
    r"\b(?:no criteria|not (?:been )?specifically developed|have not been (?:specifically )?developed|"
    r"elected not to (?:define|include|specify)|(?:is|are) not (?:currently )?(?:specified|defined|provided|"
    r"addressed|covered)|there (?:are|is) no (?:specific )?(?:criteria|requirements?|guidance|provisions?|"
    r"rating systems?))\b",
    re.IGNORECASE,
)
# Function words ignored when premise_rebutted() counts the words a question and a sentence share.
_STOP = {
    "what",
    "which",
    "how",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "to",
    "and",
    "or",
    "be",
    "by",
    "with",
    "on",
    "at",
    "as",
    "that",
    "this",
    "from",
    "per",
    "its",
    "it",
    "does",
    "do",
    "should",
    "used",
    "use",
    "when",
    "under",
    "into",
    "their",
    "than",
    "any",
}


# Markers of chunks that aren't prose: report documentation pages, document indexes, flowcharts
# and design-step charts, author bios, glossaries, and "term - definition. term - " lists.
NONPROSE = re.compile(
    r"Technical Report Documentation Page|Document Number\s+Document Title|\bFlowchart\b|"
    r"\bDesign Step [\d.]+ Chart\b|\b(?:is|was) (?:the )?(?:chair|director|a fellow|a member|a principal|"
    r"president) of\b|^\s*(?:GLOSSARY|Glossary)\b|\b[a-z-]+ - [a-z][^.]{5,80}\. [a-z-]+ - |"
    # front matter: forewords, report cover data, lists of tables/figures, TOC dot leaders,
    # mission statements (adversarial seeds adv-0502/0550 were title pages and a foreword)
    r"\bFOREWORD\b|Publication No\.|Performing Organization|List of (?:Tables|Figures)|\.{10,}|"
    r"\bmission is to\b|is (?:proud|pleased) to\b|"
    # author bios phrased as career summaries (gr-0503: "has been in private practice for 30 years")
    r"\bhas been (?:in (?:private )?practice|with [A-Z])|\bholds (?:a|an) (?:B\.?S|M\.?S|Ph\.?D)|"
    r"\bis a (?:licensed|registered) (?:professional|structural) engineer\b",
    re.MULTILINE,
)


def is_garbled(text: str) -> bool:
    """PyMuPDF output from scrambled pages ("e ows g n d i a t o n o e U w") is mostly 1-2 char pieces."""
    toks = re.findall(r"\S+", text)
    return len(toks) > 30 and sum(len(t) <= 2 for t in toks) / len(toks) > 0.45


def is_nonprose(text: str) -> bool:
    """Chunks that can't ground an answer: reference lists, report documentation pages, flowcharts,
    author bios, glossaries, scrambled text. Hand-review of grounded: 11 of 15 rejects had such a gold
    passage; this catches all 11 with no false positives on the 65 kept items."""
    return bool(NONPROSE.search(text)) or is_reference_list(text) or is_garbled(text)


def premise_rebutted(question: str, context: list[dict]) -> bool:
    """The passages explicitly say the thing asked about doesn't exist ("There are no criteria specified
    for sizing edge stiffeners"). Then "none" is a correct, grounded answer, and scoring it as a failed
    abstention would be wrong. Requires >=2 shared content words so unrelated "not addressed" text
    elsewhere in the passage doesn't count. Hand-review: caught all 3 such items, 1 borderline, 0 false."""
    words = lambda s: {
        w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2
    }
    qw = words(question)
    for c in context:
        for sent in re.split(r"(?<=[.;:])\s+", re.sub(r"\s+", " ", c["text"])):
            if PREMISE_REBUTTAL.search(sent) and len(qw & words(sent)) >= 2:
                return True
    return False


def is_answerable(question: str, context: list[dict]) -> bool:
    """Independent check that an adversarial question really is unanswerable. "Even partially"
    makes it strict, and a failed call defaults to True (answerable), so a question is only
    kept on a clear "no"."""
    passages = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in context)
    out = llm_json(f"""Passages:
{passages}

Question: {question}

Can the question be answered, even partially, from the passages alone?
Return JSON: {{"answerable": true or false, "reason": "one sentence"}}""")
    return bool(out.get("answerable", True))


# ---------------------------------------------------------------- helpers ---
def alpha_ratio(text: str) -> float:
    """Fraction of characters that are letters. Low values mean tables, equations or numeric
    dumps, which make poor question sources; main() requires > 0.6."""
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1)


def neighbours(chunk: dict, by_doc: dict[str, list[dict]], k: int = 2) -> list[dict]:
    """Up to k chunks adjacent in the same document (topically related, honest distractors)."""
    seq = by_doc[chunk["doc"]]
    i = next(n for n, c in enumerate(seq) if c["chunk_id"] == chunk["chunk_id"])
    cands = [c for c in seq[max(0, i - 2) : i + 3] if c["chunk_id"] != chunk["chunk_id"]]
    return cands[:k]


def slim(c: dict) -> dict:
    """A chunk reduced to what a task's context needs (id and text), dropping doc/n_tokens/etc."""
    return {"chunk_id": c["chunk_id"], "text": c["text"]}


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):4d} -> {path}")


# ------------------------------------------------------------------- main ---
def apply_rejects(out: pathlib.Path, task: str, items: list[dict]) -> list[dict]:
    """Human-in-the-loop: drop items rejected in hand-review (eval/tasks/rejects.jsonl).
    Matched by text because ids change per run: vocab by exact term, other tasks by question
    substring. Applied after sampling/caps, so a reject never pulls in an unreviewed replacement."""
    path = out / "rejects.jsonl"
    if not path.exists():
        return items
    rules = [json.loads(l) for l in path.open()]
    rules = [r["match"].lower().strip() for r in rules if r["task"] == task]
    if task == "vocab":
        kept = [i for i in items if i["term"].lower().strip() not in rules]
    else:
        kept = [i for i in items if not any(r in i["question"].lower() for r in rules)]
    if len(kept) != len(items):
        print(f"  {task}: {len(items) - len(kept)} hand-review rejects applied")
    return kept


IDENTIFIER_CAP = 0.20
YEAR = re.compile(r"^\s*(1[6-9]|20)\d{2}\s*$")  # a bare year: 1917, 2014


def finalize_qa_v2(out: pathlib.Path, items: list[dict]) -> list[dict]:
    """--task-version 2, after rejects (removal only, so no unreviewed item can enter):
      1. answer_kind on every item: qa_rules.answer_kind, or the reviewers' correction in
         eval/tasks/answer_kinds.jsonl ({"match": question, "kind"});
      2. layout locators removed (qa_rules.is_locator) -> locators.jsonl, with the reason;
      3. identifiers held to IDENTIFIER_CAP of the final set, keeping them in id order (main set
         first, then the supplements) -> the surplus to held_back.jsonl, not deleted.
    Identifier recall (which EM, which article) is the hardest closed-book knowledge and the
    least useful to a user, so a set dominated by it would test a card catalogue."""
    path = out / "answer_kinds.jsonl"
    fixes = [json.loads(line) for line in path.open()] if path.exists() else []
    for i in items:
        i["answer_kind"] = next(
            (f["kind"] for f in fixes if f["match"].lower().strip() in i["question"].lower()),
            answer_kind(i["answer"], i["answer_type"]),
        )
    # A numeric gold is scored within 2%, which for an id or a year accepts its neighbours:
    # FEMA P-2055 would pass P-2090, 1917 would pass 1928 and 1936 (review 2026-10-04). Those
    # are matched exactly.
    exact = 0
    for i in items:
        if i["answer_type"] == "numeric" and (
            i["answer_kind"] == "identifier" or YEAR.match(i["answer"])
        ):
            i["tolerance"], exact = 0.0, exact + 1
    print(f"  domain_qa v2: {exact} numeric ids and years scored exactly (tolerance 0)")
    locators = [{**i, "reason": r} for i in items if (r := is_locator(i["question"], i["answer"]))]
    items = [i for i in items if not is_locator(i["question"], i["answer"])]
    others = [i for i in items if i["answer_kind"] != "identifier"]
    idents = [i for i in items if i["answer_kind"] == "identifier"]
    keep = int(
        len(others) * IDENTIFIER_CAP / (1 - IDENTIFIER_CAP)
    )  # idents <= cap of the final set
    held = idents[keep:]
    held_ids = {i["id"] for i in held}
    items = [i for i in items if i["id"] not in held_ids]
    write_jsonl(out / "locators.jsonl", locators)
    write_jsonl(out / "held_back.jsonl", held)
    print(
        f"  domain_qa v2: {len(locators)} locators removed, {len(held)} identifiers held back "
        f"(cap {IDENTIFIER_CAP:.0%}: {len(idents) - len(held)} of {len(items)})"
    )
    return items


def report_failures() -> None:
    """Warn if any LLM call gave up: those chunks produced nothing, so the task set is smaller than
    intended and missing whatever those chunks covered. Failures aren't cached, so a rerun retries
    exactly those calls."""
    if FAILED_CALLS:
        print(
            f"WARNING: {FAILED_CALLS} LLM calls failed after retries; their chunks produced no "
            f"items, so the task set is smaller and skewed. Fix the cause and regenerate."
        )


def main() -> None:
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="eval/tasks")
    ap.add_argument(
        "--n-qa-chunks", type=int, default=250
    )  # stricter closed-book filters keep ~half
    ap.add_argument("--n-grounded", type=int, default=80)
    ap.add_argument("--n-vocab-chunks", type=int, default=120)  # -> up to 480 terms before dedup
    ap.add_argument("--max-vocab", type=int, default=400)
    ap.add_argument("--n-adversarial", type=int, default=110)  # ~50% pass the unanswerable check
    ap.add_argument(
        "--n-grounded-extra",
        type=int,
        default=80,
        help="supplementary grounded chunks, sampled after all tasks (no resampling)",
    )
    ap.add_argument(
        "--n-adversarial-extra",
        type=int,
        default=110,
        help="supplementary adversarial chunks, sampled after all tasks",
    )
    ap.add_argument(
        "--n-qa-extra",
        type=int,
        default=350,  # hand review kept ~27% of verified QA on the 234-document pool
        help="supplementary domain_qa chunks, sampled after all tasks (ids from 501)",
    )
    ap.add_argument(
        "--n-qa-extra2",
        type=int,
        default=1300,  # ~all unused numeric-rich chunks under --per-doc; ~24% survive review
        help="second domain_qa supplement, sampled after everything else (ids from 1001)",
    )
    ap.add_argument(
        "--per-doc", type=int, default=6, help="max source chunks per document, per task"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel LLM calls")
    ap.add_argument("--rpm", type=int, default=RPM, help="the API key's requests-per-minute limit")
    ap.add_argument(
        "--task-version",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="1: the 130-item domain_qa frozen 2026-09-27 (no second supplement, no locator "
        "filter, kinds or cap; results/table_v1.md); 2: the grown set (finalize_qa_v2; "
        "results/table_v2.md); 3 (default, committed from Stage 3): v2 with the few-shot split "
        "off by passage",
    )
    ap.add_argument(
        "--only",
        choices=["qa"],
        help="regenerate only domain_qa/fewshot; QA chunks are "
        "sampled first with the same seed, so the other tasks stay consistent "
        "(supplementary QA items need the full run)",
    )
    args = ap.parse_args()
    WORKERS = args.workers
    PACER.interval = PACER.floor = 60 / args.rpm * 1.03
    if args.task_version == 1:
        args.n_qa_extra2 = 0

    # One stream per task: QA keeps Random(seed) (unchanged sample); the others get their own, so a
    # change in one task (e.g. QA rejects shrinking the list before its shuffle) can't resample the rest.
    rng = random.Random(args.seed)
    rngs = {t: random.Random(f"{args.seed}-{t}") for t in ("grounded", "vocab", "adversarial")}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.chunks) as f:
        chunks = [json.loads(l) for l in f]
    # Chunks grouped by document, in file order (= reading order), for neighbours().
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc"], []).append(c)
    # Source chunks must be full-sized and mostly text: short chunks are page fragments, and
    # letter-poor ones are tables or equations. QA additionally wants numbers (>= 6), because
    # short numeric facts are the easiest answers to score exactly.
    eligible = [c for c in chunks if 250 <= c["n_tokens"] <= 600 and alpha_ratio(c["text"]) > 0.6]
    with_numbers = [c for c in eligible if len(re.findall(r"\d+\.?\d*", c["text"])) >= 6]
    print(f"{len(chunks)} chunks, {len(eligible)} eligible, {len(with_numbers)} numeric-rich")

    # Every chunk sampled as a source so far. take() never picks a chunk twice, so no two tasks
    # are built from the same chunk, and the set becomes the contamination list at the end.
    used: set[str] = set()

    def take(pool: list[dict], n: int, r: random.Random = rng) -> list[dict]:
        """n unused chunks in random order, at most --per-doc from any one document per call: a
        1,700-page manual would otherwise supply as many items as dozens of 40-page briefs."""
        pool = [c for c in pool if c["chunk_id"] not in used]
        r.shuffle(pool)
        picked: list[dict] = []
        per: dict[str, int] = {}
        for c in pool:
            if per.get(c["doc"], 0) < args.per_doc:
                picked.append(c)
                per[c["doc"]] = per.get(c["doc"], 0) + 1
                if len(picked) == n:
                    break
        used.update(c["chunk_id"] for c in picked)
        return picked

    text_of = {c["chunk_id"]: c["text"] for c in chunks}

    def build_qa(src: list[dict]) -> list[dict]:
        """Candidates from src chunks that pass the pre-filters, the blind check and answer
        verification (see gen_qa), in generation order."""
        candidates = [i for items in pmap(gen_qa, src) for i in items]
        n_gen = len(candidates)
        candidates = [
            i
            for i in candidates
            if not CONTEXT_BOUND.search(i["question"])
            and not is_trivia(i)
            and not ungrounded_numbers(i, text_of[i["source_chunk"]])
            and not is_reference_list(text_of[i["source_chunk"]])
        ]
        print(
            f"  {n_gen} candidates, {n_gen - len(candidates)} dropped by pre-filter "
            "(context-bound, trivia, list questions, numbers not in passage, reference lists)"
        )
        blind = pmap(lambda i: is_standalone(i["question"]), candidates)
        candidates = [i for i, (ok, _) in zip(candidates, blind) if ok]
        print(f"  {len(candidates)} pass the blind standalone check, verifying answers...")
        keep = pmap(lambda i: verify_qa(i, text_of[i["source_chunk"]]), candidates)
        verified = [i for i, ok in zip(candidates, keep) if ok]
        print(f"  {len(verified)} pass answer verification")
        return verified

    # 1. domain_qa ------------------------------------------------------------
    qa_chunks = take(with_numbers, args.n_qa_chunks)
    print("generating QA...")
    qa = build_qa(qa_chunks)
    n = len(qa)
    qa = dedup_questions(qa)
    print(f"  {len(qa)} after dedup ({n - len(qa)} near-duplicate facts)")
    qa = apply_rejects(out, "domain_qa", qa)
    # The first 3 verified items become the few-shot examples in every domain_qa prompt, and are
    # removed from the scored set so no scored item has its answer in the prompt. v1/v2 split them
    # off by item, which left two scored items from a shot's passage (qa-0003, qa-0056: other facts,
    # eval/contamination.py); v3 also drops the shots' passages from the scored set.
    rng.shuffle(qa)
    fewshot, qa = qa[:3], qa[3:]
    for n, i in enumerate(qa, 1):
        i["id"] = f"qa-{n:04d}"
        i["tolerance"] = 0.02
    shot_chunks = {s["source_chunk"] for s in fewshot}
    if args.task_version >= 3:  # after the ids, so every v3 item keeps its v2 id
        qa = [i for i in qa if i["source_chunk"] not in shot_chunks]
    write_jsonl(out / "domain_qa.jsonl", qa)
    write_jsonl(out / "fewshot.jsonl", fewshot)

    if args.only == "qa":
        report_failures()
        return

    def build_grounded(src: list[dict], r: random.Random, first_id: int) -> list[dict]:
        questions = pmap(gen_grounded_question, src)
        items = []
        for n, (c, q) in enumerate(zip(src, questions), first_id):
            if not q:
                continue
            # Context = gold + 2 same-document neighbours (related but not the answer) + 1 chunk
            # from another document (unrelated), shuffled so the gold passage isn't always first.
            distractor = r.choice([d for d in eligible if d["doc"] != c["doc"]])
            ctx = [slim(c)] + [slim(x) for x in neighbours(c, by_doc)] + [slim(distractor)]
            r.shuffle(ctx)
            items.append(
                {
                    "id": f"gr-{n:04d}",
                    "question": q,
                    "context": ctx,
                    "gold_chunk_ids": [c["chunk_id"]],
                }
            )
        return items

    # 2. grounded -------------------------------------------------------------
    gr_chunks = take(eligible, args.n_grounded, rngs["grounded"])
    print("generating grounded questions...")
    grounded = build_grounded(gr_chunks, rngs["grounded"], 1)

    # 3. vocab ----------------------------------------------------------------
    vc_chunks = take(eligible, args.n_vocab_chunks, rngs["vocab"])
    print("extracting terms...")
    terms = [t for ts in pmap(gen_terms, vc_chunks) for t in ts]
    # terms used as few-shot examples in prompts.vocab_prompt would leak their own answer
    seen: set[str] = {s["term"].lower() for s in VOCAB_SHOTS}
    vocab = []
    for t in terms:
        key = re.sub(r"\s+", " ", t["term"].lower().strip())
        if key in seen or len(key) < 3:
            continue
        seen.add(key)
        vocab.append(t)
    rngs["vocab"].shuffle(vocab)
    vocab = vocab[: args.max_vocab]
    for n, t in enumerate(vocab, 1):
        t["id"] = f"vc-{n:04d}"
    n = len(
        vocab
    )  # post-cap, removal only: filtering earlier would pull unreviewed terms under the cap
    vocab = [t for t in vocab if term_in_passage(t["term"], text_of[t["source_chunk"]])]
    print(f"  vocab: {n - len(vocab)} dropped (term not in its source passage)")
    seen_keys: set[str] = set()
    deduped = []
    for t in vocab:
        if term_key(t["term"]) not in seen_keys:
            seen_keys.add(term_key(t["term"]))
            deduped.append(t)
    print(f"  vocab: {len(vocab) - len(deduped)} dropped (normalized-term duplicates)")
    vocab = deduped
    vocab = apply_rejects(out, "vocab", vocab)
    write_jsonl(out / "vocab.jsonl", vocab)

    def build_adversarial(src: list[dict], first_id: int) -> list[dict]:
        # Context = source chunk + 2 neighbours, unshuffled (no gold passage to hide). All three
        # are on one topic, so the question sounds answerable and the model has to read them to
        # find out that it isn't.
        contexts = [[slim(c)] + [slim(x) for x in neighbours(c, by_doc)] for c in src]
        qtypes = [ADV_QTYPES[k % len(ADV_QTYPES)] for k in range(len(contexts))]
        gens = pmap(lambda cq: gen_adversarial(*cq), list(zip(contexts, qtypes)))
        answerable = pmap(
            lambda p: is_answerable(p[0].get("question", ""), p[1]), list(zip(gens, contexts))
        )
        items = []
        for g, ctx, ans in zip(gens, contexts, answerable):
            if g.get("question") and not ans and not premise_rebutted(g["question"], ctx):
                items.append(
                    {
                        "id": f"adv-{first_id + len(items):04d}",
                        "question": g["question"],
                        "context": ctx,
                        "why_unanswerable": g.get("why_unanswerable", ""),
                    }
                )
        return items

    # 4. adversarial ----------------------------------------------------------
    adv_chunks = take(eligible, args.n_adversarial, rngs["adversarial"])
    print("generating adversarial questions (with unanswerable check)...")
    adversarial = build_adversarial(adv_chunks, 1)

    # 4b. supplementary grounded + adversarial + domain_qa ----------------------
    # Sampled AFTER every task, from their own streams, so enlarging these two sets can't shift
    # any earlier sample (vocab's pool excludes grounded's chunks, so simply raising n_grounded
    # would resample, and silently replace, reviewed vocab items). Ids start at 501 to mark them.
    if args.n_grounded_extra:
        extra = take(eligible, args.n_grounded_extra, random.Random(f"{args.seed}-grounded-extra"))
        print(f"generating {len(extra)} supplementary grounded questions...")
        grounded += build_grounded(extra, random.Random(f"{args.seed}-grounded-extra-ctx"), 501)
    if args.n_adversarial_extra:
        extra = take(
            eligible, args.n_adversarial_extra, random.Random(f"{args.seed}-adversarial-extra")
        )
        print(f"generating {len(extra)} supplementary adversarial questions...")
        adversarial += build_adversarial(extra, 501)
    more: list[dict] = []
    more2: list[dict] = []
    if args.n_qa_extra:
        # Same pipeline and same stream rule; dedup runs against the main set (already deduped,
        # so it keeps all of it), so a supplementary item never repeats a fact asked there.
        extra = take(with_numbers, args.n_qa_extra, random.Random(f"{args.seed}-qa-extra"))
        print(f"generating QA from {len(extra)} supplementary chunks...")
        main_qa = fewshot + qa
        more = dedup_questions(main_qa + build_qa(extra))[len(main_qa) :]
        more = apply_rejects(out, "domain_qa", more)
        for n, i in enumerate(more, 501):
            i["id"] = f"qa-{n:04d}"
            i["tolerance"] = 0.02
        write_jsonl(out / "domain_qa.jsonl", qa + more)
    if args.n_qa_extra2:
        # Second supplement (2026-10-04, user decision): 130 items gave qa_acc a 3.1-point standard
        # error, wider than any Stage 2 effect, and too few for seen/unseen halves. Same pipeline,
        # its own stream, the last take(), so no earlier sample moves; deduped against every
        # earlier item, the few-shot included.
        extra = take(with_numbers, args.n_qa_extra2, random.Random(f"{args.seed}-qa-extra2"))
        print(f"generating QA from {len(extra)} second-supplement chunks...")
        earlier = fewshot + qa + more
        more2 = dedup_questions(earlier + build_qa(extra))[len(earlier) :]
        more2 = apply_rejects(out, "domain_qa", more2)
        for n, i in enumerate(more2, 1001):
            i["id"] = f"qa-{n:04d}"
            i["tolerance"] = 0.02
        write_jsonl(out / "domain_qa.jsonl", qa + more + more2)
    if args.task_version >= 2:
        # every item, the frozen 130 included: v2 is a new set, v1 stays reproducible
        final = finalize_qa_v2(out, qa + more + more2)
        for i in final:  # v2 only: v3 has no such item. Per-item analyses can exclude these.
            if i["source_chunk"] in shot_chunks:
                i["fewshot_passage_overlap"] = True
        write_jsonl(out / "domain_qa.jsonl", final)

    n = len(grounded)
    grounded = [g for g in grounded if not is_nonprose(text_of[g["gold_chunk_ids"][0]])]
    print(f"  grounded: {n - len(grounded)} dropped for non-prose gold passages")
    grounded = apply_rejects(out, "grounded", grounded)
    write_jsonl(out / "grounded.jsonl", grounded)

    n = len(adversarial)
    adversarial = [a for a in adversarial if not is_nonprose(a["context"][0]["text"])]
    print(f"  adversarial: {n - len(adversarial)} dropped for non-prose source passages")
    adversarial = apply_rejects(out, "adversarial", adversarial)
    write_jsonl(out / "adversarial.jsonl", adversarial)

    # 5. contamination list ---------------------------------------------------
    # Every chunk an eval item was built from or shows as context (sources, neighbours,
    # distractors). SFT/DPO/GRPO data generation must skip these, or the eval measures
    # memorised Q/A pairs. CPT still trains on them: closed-book QA tests exactly that knowledge.
    ids = sorted(
        used
        | {c["chunk_id"] for g in grounded for c in g["context"]}
        | {c["chunk_id"] for a in adversarial for c in a["context"]}
    )
    (out / "eval_chunk_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} chunk ids reserved for eval -> {out / 'eval_chunk_ids.txt'}")
    report_failures()


if __name__ == "__main__":
    main()
