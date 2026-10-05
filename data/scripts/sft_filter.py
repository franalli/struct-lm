"""A3: dedup, caps, decontamination and selection, before any answer is generated
-> data/sft/work/questions_kept.jsonl (one row per answer task) + data/sft/sft_rejected.jsonl

  .venv/bin/python data/scripts/sft_filter.py

No LLM calls; this is where the eval is read (through sft_guard), so the next step's teacher gets
tasks that already passed every eval check. In order:
  1. quality   closed-book questions: qa_rules.is_locator, make_tasks.CONTEXT_BOUND / is_trivia /
               ungrounded_numbers, at most 6 words of answer; definitions: the term occurs in the
               passage (make_tasks.term_in_passage)
  2. rule 1    (all chunks) the question shares > 50% of its word 4-grams with an eval question,
               or the term is an unseen vocab item's (sft_guard.rule1 / term_block). On eval-seen
               chunks a closed-book fact keeps its place and only loses the question: the
               generator's natural question for a seen fact often lands on the eval's wording, and
               dropping the fact would empty the seen half. A4 rewrites it (rewrite_only).
  3. rule 2    (ordinary chunks) the answer equals an eval answer from the same document
  4. dedup     exact (normalised question / term key), then MinHash on word 3-grams (>= 0.7)
  5. caps      final examples per chunk, counted after A4's paraphrases: 12 per eval-seen chunk
               (a fact = 2 closed-book phrasings, a term's definition 1; filled definitions
               first, then values, identifiers, terms), 4 per ordinary chunk (a value fact = 3)
  6. selection ordinary tasks per format up to target x headroom (what A4 pays for), alternating
               the exact and paraphrased chunk halves; targets scale with the pool (--pilot)
Grounded and abstain items get their passage sets here (A6 in the plan), because A4's teacher
answers with them in front of it: grounded = gold + 2 allowed neighbours within 4 positions + 1
allowed passage from another document, shuffled; abstain = 2 allowed passages from the question's
own document at least 3 positions from its chunk + 2 from other documents (on-topic, as the
adversarial eval is), checked for the answer by A4's teacher and A5's judge.
"""

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# eval/ modules (prompts, scorers, ...), as eval/run_eval.py imports them
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from datasketch import MinHash, MinHashLSH
from qa_rules import is_locator
from scorers import normalize
from sft_common import (
    SFT,
    WORK,
    h01,
    load_chunks,
    make_tasks,
    read_jsonl,
    update_stats,
    write_jsonl,
)
from sft_guard import Guard, words

# Final ordinary examples per format at full scale (900 chunks); A6 trims to these.
TARGETS = {
    "value": 110,
    "identifier": 60,
    "grounded": 500,
    "abstain": 200,
    "definition": 150,
    "multi_step": 150,
}
HEADROOM = {
    "value": 1.25,
    "identifier": 1.25,
    "grounded": 1.3,
    "abstain": 1.6,
    "definition": 1.3,
    "multi_step": 1.3,
}
MIN_ORDINARY_DEFINITIONS = 50  # forced terms may fill the definition target; keep some variety
PHRASINGS = {("eval_seen", "closed_book"): 2, ("ordinary", "value"): 3}
CAP = {"eval_seen": 12, "ordinary": 4}
MAX_FACTS = (
    12  # A2 extracts at most 6 values/identifiers and 6 terms; the 12-example cap binds first
)
# Bibliographic trivia make_tasks.TRIVIA misses (pilot: "What are the page numbers ... in the March
# 1972 issue of Water Power?"), and equation-number answers to "which equations" (is_locator only
# knows the singular).
BIBLIO = re.compile(
    r"\b(page numbers?|issue of|journal|proceedings|published (?:in|by))\b", re.IGNORECASE
)
EQUATIONS = re.compile(r"\bequations?\b", re.IGNORECASE)
PASSAGE_WORDS = re.compile(
    r"\b(passage|this document|the figure|the table|the excerpt)\b", re.IGNORECASE
)


def closed_book_rule(mt, q: str, a: str, chunk_id: str, passage: str) -> str | None:
    """The first quality rule a closed-book question fails, or None."""
    if not q:
        return "empty"
    if is_locator(q, a) or (EQUATIONS.search(q) and re.search(r"\d", a)):
        return "locator"
    if mt.CONTEXT_BOUND.search(q) or PASSAGE_WORDS.search(q):
        return "context_bound"
    if mt.is_trivia({"question": q, "answer": a, "source_chunk": chunk_id}) or BIBLIO.search(q):
        return "trivia"
    if mt.ungrounded_numbers({"answer": a}, passage):
        return "ungrounded_number"
    if len(a.split()) > 6:
        return "long_answer"
    return None


def shingles(text: str) -> set[bytes]:
    ws = words(text)
    return {" ".join(ws[i : i + 3]).encode() for i in range(max(1, len(ws) - 2))}


def main() -> None:
    mt = make_tasks()
    guard = Guard()
    by_id, by_doc = load_chunks()
    pool = read_jsonl(WORK / "pool.jsonl")
    chunk = {c["chunk_id"]: c for c in pool}
    order = {c["chunk_id"]: k for k, c in enumerate(pool)}
    items = sorted(
        read_jsonl(WORK / "questions.jsonl"), key=lambda q: (order[q["chunk_id"]], q["qid"])
    )
    rejects, kept = [], []

    def reject(it: dict, rule: str, eval_id: str | None = None) -> None:
        rejects.append(
            {
                "stage": "A3",
                "rule": rule,
                "qid": it["qid"],
                "chunk_id": it["chunk_id"],
                "kind": it["kind"],
                "question": it["question"] or it["term"],
                "answer": it["answer"],
                "eval_id": eval_id,
            }
        )

    # Each item yields a closed-book task (value / identifier, and forced terms: question -> term),
    # a definition task (terms) or a grounded task (procedure); judged separately below.
    tasks = []
    for it in items:
        c = chunk[it["chunk_id"]]
        if it["kind"] in ("value", "identifier") or (it["kind"] == "term" and it["question"]):
            tasks.append({**it, "format": "closed_book"})
        if it["kind"] == "term" and it["definition"]:
            tasks.append({**it, "format": "definition"})
        if it["kind"] == "procedure":
            tasks.append({**it, "format": "grounded"})
    rewritten = []  # eval-seen facts whose question rule 1 rejected
    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    seen_q, seen_terms = set(), set()
    for t in tasks:
        c = chunk[t["chunk_id"]]
        t["tid"] = f"{t['qid']}:{t['format']}"
        q, a = t["question"] or "", t["answer"]
        if t["format"] == "closed_book":
            rule = closed_book_rule(mt, q, a, t["chunk_id"], c["text"])
        elif t["format"] == "definition":
            rule = "term_not_in_passage" if not mt.term_in_passage(t["term"], c["text"]) else None
        else:
            rule = "empty" if not q else "context_bound" if PASSAGE_WORDS.search(q) else None
        if rule:
            reject(t, rule)
            continue
        if t["format"] == "definition":
            if hit := guard.term_block(t["term"]):
                reject(t, "rule1", hit)
                continue
        elif hit := guard.rule1(q):
            if c["origin"] == "eval_seen" and t["format"] == "closed_book":
                # the fact stays, the question goes: A4 writes two new phrasings of it (from this
                # question, never the eval's) and A6's rule-1 gate checks them again
                t["rewrite_only"] = True
                rewritten.append(t["qid"])
            else:
                reject(t, "rule1", hit)
                continue
        term = t["term"] if t["format"] == "definition" else None
        if (
            c["origin"] == "ordinary"
            and t["format"] != "grounded"
            and (hit := guard.rule2(c["doc"], a, term))
        ):
            reject(t, "rule2", hit)
            continue
        if t["format"] == "definition":
            key = mt.term_key(t["term"])
            if key in seen_terms:
                reject(t, "duplicate")
                continue
            seen_terms.add(key)
        else:
            key = normalize(q)
            if key in seen_q:
                reject(t, "duplicate")
                continue
            m = MinHash(num_perm=128)
            m.update_batch(list(shingles(q)))
            if lsh.query(m):
                reject(t, "near_duplicate")
                continue
            seen_q.add(key)
            lsh.insert(t["tid"], m)
        kept.append(t)

    # ---- forced chunks: every surviving fact, within 12 examples per chunk ----
    # Values / identifiers (2 closed-book phrasings each) and term definitions (1 each) take turns,
    # then closed-book term questions: the generator can't tell a domain_qa chunk from a vocab one
    # (the two halves share no chunk), and filling either kind first starved the other in the pilot.
    out, budget, facts = [], defaultdict(int), defaultdict(set)
    per_chunk = defaultdict(lambda: ([], [], []))
    for t in kept:
        if chunk[t["chunk_id"]]["origin"] == "eval_seen":
            slot = 1 if t["format"] == "definition" else 2 if t["kind"] == "term" else 0
            per_chunk[t["chunk_id"]][slot].append(t)
    forced_tasks = []
    for cid in sorted(per_chunk, key=order.get):
        facts_, defs, terms = per_chunk[cid]
        n = max(len(facts_), len(defs))
        forced_tasks += [t for k in range(n) for t in (facts_[k : k + 1] + defs[k : k + 1])] + terms
    for t in forced_tasks:
        cost = 2 if t["format"] == "closed_book" else 1
        new_fact = t["qid"] not in facts[t["chunk_id"]]
        if budget[t["chunk_id"]] + cost > CAP["eval_seen"] or (
            new_fact and len(facts[t["chunk_id"]]) >= MAX_FACTS
        ):
            reject(t, "cap")
            continue
        budget[t["chunk_id"]] += cost
        facts[t["chunk_id"]].add(t["qid"])
        out.append(
            {"rewrite_only": False, **t, "phrasings": 2 if t["format"] == "closed_book" else 1}
        )
    forced_defs = sum(t["format"] == "definition" for t in out)

    # ---- ordinary chunks: select per format, alternating the two wording halves ----
    n_ordinary = sum(c["origin"] == "ordinary" for c in pool)
    scale = n_ordinary / 900
    targets = dict(TARGETS)
    targets["definition"] = max(MIN_ORDINARY_DEFINITIONS, TARGETS["definition"] - forced_defs)
    want = {k: round(v * scale * HEADROOM[k]) for k, v in targets.items()}
    ordinary = [t for t in kept if chunk[t["chunk_id"]]["origin"] == "ordinary"]
    used = set()  # qids already given a format

    def candidates(pred, salt: str) -> list[dict]:
        cs = sorted(
            (t for t in ordinary if pred(t) and t["qid"] not in used),
            key=lambda t: h01(f"{salt}:{t['qid']}"),
        )
        halves = [
            [t for t in cs if chunk[t["chunk_id"]]["wording"] == w]
            for w in ("exact", "paraphrased")
        ]
        mixed = [t for pair in zip(*halves) for t in pair]
        longer = max(halves, key=len)
        return mixed + longer[len(mixed) // 2 :]

    allowed_far = [
        c
        for c in by_id.values()
        if guard.allowed(c["chunk_id"])
        and 150 <= c["n_tokens"] <= 600
        and not mt.is_nonprose(c["text"])
    ]

    def other_doc(r: random.Random, doc: str, n: int, taken: set) -> list[dict]:
        got = []
        while len(got) < n:
            c = r.choice(allowed_far)
            if c["doc"] != doc and c["chunk_id"] not in taken:
                got.append(c)
                taken.add(c["chunk_id"])
        return got

    def grounded_passages(t: dict) -> list[dict] | None:
        src = by_id[t["chunk_id"]]
        seq = by_doc[src["doc"]]
        near = sorted(
            (
                c
                for c in seq[max(0, src["pos"] - 4) : src["pos"] + 5]
                if c["chunk_id"] != src["chunk_id"]
                and guard.allowed(c["chunk_id"])
                and c["n_tokens"] >= 80
                and not mt.is_nonprose(c["text"])
            ),
            key=lambda c: (abs(c["pos"] - src["pos"]), c["pos"]),
        )[:2]
        if len(near) < 2:
            return None
        r = random.Random(f"sft-grounded:{t['qid']}")
        ctx = [
            src,
            *near,
            *other_doc(r, src["doc"], 1, {c["chunk_id"] for c in near} | {src["chunk_id"]}),
        ]
        r.shuffle(ctx)
        return ctx

    def abstain_passages(t: dict) -> list[dict] | None:
        src = by_id[t["chunk_id"]]
        same = [
            c
            for c in by_doc[src["doc"]]
            if abs(c["pos"] - src["pos"]) >= 3
            and guard.allowed(c["chunk_id"])
            and 150 <= c["n_tokens"] <= 600
            and not mt.is_nonprose(c["text"])
        ]
        if len(same) < 2:
            return None
        r = random.Random(f"sft-abstain:{t['qid']}")
        ctx = r.sample(same, 2)
        ctx += other_doc(r, src["doc"], 2, {c["chunk_id"] for c in ctx} | {src["chunk_id"]})
        r.shuffle(ctx)
        return ctx

    def take(
        fmt: str, pred, cost: int, extra=None, task_format: str | None = None, phrasings: int = 1
    ) -> int:
        n = 0
        for t in candidates(pred, f"sft-{fmt}"):
            if n >= want[fmt]:
                break
            if budget[t["chunk_id"]] + cost > CAP["ordinary"]:
                continue
            row = {
                **t,
                "format": task_format or t["format"],
                "phrasings": phrasings,
                "rewrite_only": False,
            }
            if extra:
                ctx = extra(t)
                if ctx is None:
                    continue
                row["passages"] = [
                    {"label": f"P{k + 1}", "chunk_id": c["chunk_id"], "text": c["text"]}
                    for k, c in enumerate(ctx)
                ]
                row["gold_label"] = next(
                    (p["label"] for p in row["passages"] if p["chunk_id"] == t["chunk_id"]), None
                )
            if fmt in ("abstain", "multi_step"):
                row["tid"] = f"{t['qid']}:{fmt}"
                row["kind"] = None if fmt == "abstain" else "multi_step"
            budget[t["chunk_id"]] += cost
            used.add(t["qid"])
            out.append(row)
            n += 1
        return n

    picked = {
        "grounded": take("grounded", lambda t: t["format"] == "grounded", 1, grounded_passages),
        "value": take(
            "value", lambda t: t["format"] == "closed_book" and t["kind"] == "value", 3, phrasings=3
        ),
        "identifier": take(
            "identifier", lambda t: t["format"] == "closed_book" and t["kind"] == "identifier", 1
        ),
        "definition": take("definition", lambda t: t["format"] == "definition", 1),
        "multi_step": take(
            "multi_step",
            lambda t: t["format"] == "closed_book" and t["kind"] == "value",
            1,
            task_format="closed_book",
        ),
        "abstain": take(
            "abstain",
            lambda t: (
                t["format"] in ("closed_book", "grounded") and t["kind"] in ("value", "procedure")
            ),
            1,
            abstain_passages,
            task_format="abstain",
        ),
    }

    for t in out:
        c = chunk[t["chunk_id"]]
        t.update(
            doc=c["doc"],
            title=c["title"],
            text=c["text"],
            origin=c["origin"],
            wording=c["wording"],
            persona=c["persona"],
            asker=c["asker"],
        )
        t.setdefault("passages", None)
        t.setdefault("gold_label", None)
    write_jsonl(WORK / "questions_kept.jsonl", out)
    write_jsonl(SFT / "sft_rejected.jsonl", rejects)
    stats = {
        "items": len(items),
        "tasks": len(tasks),
        "rejected": dict(Counter(r["rule"] for r in rejects)),
        "rejected_by_origin": dict(
            Counter(f"{chunk[r['chunk_id']]['origin']}:{r['rule']}" for r in rejects)
        ),
        "kept_forced": dict(Counter(t["format"] for t in out if t["origin"] == "eval_seen")),
        "forced_rule1_rewrite": sum(t.get("rewrite_only", False) for t in out),
        "forced_chunks_with_tasks": len({t["chunk_id"] for t in out if t["origin"] == "eval_seen"}),
        "ordinary_targets_with_headroom": want,
        "ordinary_picked": picked,
        "ordinary_wording": dict(
            Counter(f"{t['format']}:{t['wording']}" for t in out if t["origin"] == "ordinary")
        ),
    }
    update_stats("filter", stats)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
