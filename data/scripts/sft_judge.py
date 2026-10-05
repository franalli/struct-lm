"""A5: rubric filter, one judge call per example -> data/sft/work/judged.jsonl

  set -a; . ./.env; set +a
  .venv/bin/python data/scripts/sft_judge.py

Two independent checks per example, then a verdict:
  verifier  rules, no LLM (rule 8: decide by rule what a rule can decide): the gold fact by
            scorers.numeric_match / exact_match, [Pn] citations by scorers.citations_valid, the
            exact abstain sentence, a hedge pattern, answer length, one-sentence definitions,
            the "Answer:" line of a worked problem
  judge     TEACHER (Large 3) at temperature 0, one call: the format's rubric in the RLHF Book
            ch. 12 layout ([Hard Rule] / [Principle] / [Optional] / [Pitfall], each with a title,
            a one-line description and a weight), the item's gold fact inserted as a hard rule.
            JSON with the reasoning first, then each rule's verdict.
  keep      verifier passes, no hard rule fails, and the principle score (weighted mean of the
            1-5 principle grades, minus half of each pitfall's weight) is at least 4
(The eval's own closed-book checks, make_tasks.verify_qa's faithfulness test and the blind
is_standalone test, were tried here after the 2026-10-05 audit and left out: on the 40 audited
closed-book items the first caught 1 of 8 defects, the second rejected 9 of 19 good items for 4 of
8 defects, so neither lowered the defect rate; notes/decisions.md.)
Critique-and-revise, one round: an example whose only failures are format failures (citation
format, hedging, a reworded abstain sentence, extra words) goes back to its own teacher with the
failures listed, then through both checks again. Factual failures are never revised. A5 logs the
salvage rate, acceptance by format, by rule and by teacher (Large judging Medium's answers against
its own: the self-preference measurement), and every example where the verifier and the judge
disagree on a rule they both check.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# eval/ modules (prompts, scorers, ...), as eval/run_eval.py imports them
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from scorers import citations, citations_valid, exact_match, numeric_match
from sft_common import (
    ABSTAIN_REPLY,
    TEACHER,
    WORK,
    llm_json,
    pmap,
    read_jsonl,
    report_failures,
    set_rpm,
    update_stats,
    write_jsonl,
)

KEEP_SCORE = 4.0
HEDGE = re.compile(
    r"\b(approximately|approx\.|about|around|roughly|typically|usually|generally|likely|probably|"
    r"possibly|i think|i believe|it depends|or so|may vary|varies)\b|\?",
    re.IGNORECASE,
)
# The worked problem's result: the last "Answer: ..." in the text, running to its end. Teachers
# often write the steps as one paragraph ("... = 6500 kN. Answer: 6500 kN"); sft_assemble moves the
# final "Answer:" onto its own line, as the prompt asks. Leftover JSON is a format failure.
FINAL_ANSWER = re.compile(r"answer\s*:\s*([^\n]+?)\s*$", re.IGNORECASE)
JSON_DEBRIS = re.compile(r"""\[['"]|['"]\]|\{['"]|['"],\s*['"]|"answer"\s*:""", re.IGNORECASE)
# A closed-book completion can't lean on a passage the student never sees (review: "Default design
# story drift ratio from the passage: 0.015"). "a passage" as a physical opening is fine.
PASSAGE_REF = re.compile(
    r"\b(?:the|this|that) passage\b|\bpassages?\s+(?:states?|gives?|says|provides?)\b",
    re.IGNORECASE,
)
LABEL_CITE = re.compile(r"\[P[1-4]\]")
NUMBER_WORDS = {w: str(k) for k, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
)}  # fmt: skip


def same_fact(answer: str, gold: str, kind: str) -> bool:
    """The answer states the gold fact: numbers by value (2%, number words read as digits), text by
    scorers.exact_match either way, with plural s ignored ("cripple walls" = "cripple wall")."""
    digits = lambda t: re.sub(
        r"\b(" + "|".join(NUMBER_WORDS) + r")\b",
        lambda m: NUMBER_WORDS[m[1].lower()],
        t,
        flags=re.IGNORECASE,
    )
    if kind == "value" and re.search(r"\d", gold):
        return numeric_match(digits(answer), gold)
    sing = lambda t: re.sub(r"(?<=[a-z]{3})s\b", "", t.lower())
    a, g = sing(answer), sing(gold)
    return exact_match(a, g) or exact_match(g, a)


# (id, tag, class, weight, title, description). class "format" failures may be revised; "factual"
# never. {gold} / {what} / {label} are filled per item.
RUBRICS = {
    "closed_book": [
        (
            "H1",
            "Hard Rule",
            "factual",
            None,
            "Supported",
            "The answer is stated in the passage or follows directly from it.",
        ),
        (
            "H2",
            "Hard Rule",
            "factual",
            None,
            "Gold fact",
            'The answer gives "{gold}"{what}: the same value (within 2%), designation or term; a different unit spelling is fine.',
        ),
        (
            "H3",
            "Hard Rule",
            "factual",
            None,
            "Question fits",
            "Read without the passage, the question asks for exactly this fact and has it as its only reasonable answer: it names the document, structure, condition or quantity.",
        ),
        (
            "H5",
            "Hard Rule",
            "factual",
            None,
            "General fact",
            (
                "A general rule, requirement, definition, recommendation or finding of the "
                "document, NOT a value computed, chosen or assumed for one worked example in it."
            ),
        ),
        (
            "H4",
            "Hard Rule",
            "format",
            None,
            "No hedging",
            'The answer is stated plainly: no "approximately", "typically", "I think", alternatives or caveats.',
        ),
        (
            "P1",
            "Principle",
            "factual",
            3,
            "Precise",
            "As specific as the passage: the unit, condition or designation the question needs.",
        ),
        (
            "P2",
            "Principle",
            "format",
            2,
            "Bare answer",
            "Only the value, term or name, at most 6 words, no explanation.",
        ),
        (
            "P3",
            "Principle",
            "factual",
            2,
            "Natural question",
            "Reads like a question a real engineer would ask: clear, grammatical, not contrived.",
        ),
        (
            "O1",
            "Optional",
            "format",
            None,
            "Passage notation",
            "Uses the passage's own unit and notation.",
        ),
        (
            "X1",
            "Pitfall",
            "format",
            1,
            "Restates the question",
            "The answer repeats words of the question instead of just answering.",
        ),
        (
            "X2",
            "Pitfall",
            "factual",
            2,
            "Facts not in the passage",
            "The answer adds anything the passage doesn't say.",
        ),
    ],
    "multi_step": [
        (
            "H1",
            "Hard Rule",
            "factual",
            None,
            "Document value",
            "The value the solution takes from the document is the one the passage states.",
        ),
        (
            "H2",
            "Hard Rule",
            "factual",
            None,
            "Correct result",
            'The arithmetic is right and the final answer is "{gold}" (within 2%).',
        ),
        (
            "H3",
            "Hard Rule",
            "factual",
            None,
            "Self-contained problem",
            "The problem states every input except the document's value, names the document or code, and has one correct answer.",
        ),
        (
            "H4",
            "Hard Rule",
            "format",
            None,
            "Answer line",
            'The solution ends with a line "Answer: <value with unit>".',
        ),
        (
            "P1",
            "Principle",
            "factual",
            3,
            "Clear steps",
            "Each step shows its formula or operation and its numbers.",
        ),
        ("P2", "Principle", "format", 2, "Concise", "One line per step, no padding."),
        (
            "X1",
            "Pitfall",
            "factual",
            2,
            "Facts not in the passage",
            "Uses a document value the passage doesn't give.",
        ),
    ],
    "definition": [
        (
            "H1",
            "Hard Rule",
            "factual",
            None,
            "Faithful",
            (
                "Nothing in it contradicts how the passage uses the term; it may be more general "
                'than the reference and leave out its specifics, but not wrong (reference: "{gold}").'
            ),
        ),
        ("H2", "Hard Rule", "format", None, "One sentence", "Exactly one sentence."),
        (
            "P1",
            "Principle",
            "factual",
            3,
            "Precise",
            "States the defining mechanism or property, not a synonym or a vague gloss.",
        ),
        (
            "P2",
            "Principle",
            "format",
            2,
            "Stands alone",
            "Understandable without the passage; no citation, no mention of the passage or document.",
        ),
        (
            "O1",
            "Optional",
            "format",
            None,
            "Genus first",
            'Opens with the class of thing ("A member that ...", "The ratio of ...").',
        ),
        (
            "X1",
            "Pitfall",
            "factual",
            2,
            "Circular",
            "Defines the term with itself or its root word.",
        ),
        (
            "X2",
            "Pitfall",
            "factual",
            2,
            "Facts not in the passage",
            "Adds a claim the passage doesn't support.",
        ),
    ],
    "grounded": [
        (
            "H1",
            "Hard Rule",
            "factual",
            None,
            "Supported",
            "Every claim is supported by the passage it cites.",
        ),
        (
            "H2",
            "Hard Rule",
            "factual",
            None,
            "Answers from the gold passage",
            'Answers the question with the content of passage {label} (reference gist: "{gold}").',
        ),
        (
            "H3",
            "Hard Rule",
            "format",
            None,
            "Citations",
            "Every claim is followed by the bracketed label of a provided passage (P1-P4), and nothing else is in brackets.",
        ),
        (
            "H4",
            "Hard Rule",
            "factual",
            None,
            "Does not abstain",
            "Gives the answer (it is in the passages) instead of declining.",
        ),
        (
            "P1",
            "Principle",
            "factual",
            3,
            "Complete",
            "Covers what the question asks, as far as the passages say.",
        ),
        (
            "P2",
            "Principle",
            "format",
            2,
            "Direct",
            "One to three sentences, straight to the answer.",
        ),
        (
            "X1",
            "Pitfall",
            "factual",
            2,
            "Cites a non-supporting passage",
            "Cites a passage (e.g. an unrelated distractor) that doesn't support the claim.",
        ),
        (
            "X2",
            "Pitfall",
            "factual",
            2,
            "Facts not in the passages",
            "Adds anything none of the passages says.",
        ),
        ("X3", "Pitfall", "format", 1, "Restates the question", "Opens by repeating the question."),
    ],
    "abstain": [
        (
            "H1",
            "Hard Rule",
            "factual",
            None,
            "Warranted",
            "None of the passages contains the answer, even partially or by simple arithmetic on their values.",
        ),
        (
            "H2",
            "Hard Rule",
            "format",
            None,
            "Exact sentence",
            f'The reply is exactly "{ABSTAIN_REPLY}" and nothing else.',
        ),
        (
            "P1",
            "Principle",
            "factual",
            1,
            "Natural question",
            (
                "A clear, specific question an engineer could ask about this subject (judged on its "
                "own, not on these passages)."
            ),
        ),
    ],
}


def rubric_key(e: dict) -> str:
    return "multi_step" if e["kind"] == "multi_step" else e["format"]


def rubric_text(e: dict) -> tuple[str, list[tuple]]:
    items = RUBRICS[rubric_key(e)]
    what = f" ({e['what']})" if e.get("what") else ""
    gold = (e.get("gold") or "").replace('"', "'")
    lines = []
    for rid, tag, _, w, title, desc in items:
        weight = f" (weight {w})" if w else ""
        lines.append(
            f"[{tag}] {rid} {title}{weight}: {desc.format(gold=gold, what=what, label=e.get('gold_label'))}"
        )
    return "\n".join(lines), items


def shown(e: dict) -> str:
    """What the judge sees as the item's source: the passage, or the labelled passages."""
    if e["passages"]:
        return "\n\n".join(f"[{p['label']}]\n{p['text']}" for p in e["passages"])
    return f"Document: {e['title']}\nPassage:\n{e['text']}"


def asked(e: dict) -> str:
    if e["format"] == "definition":
        return f'Define the term "{e["term"]}" in one sentence.'
    if e["kind"] == "multi_step":
        return f'{e["question"]} (show the calculation; last line "Answer: ...")'
    if e["passages"]:
        return (
            f"{e['question']} (answer from passages P1-P4 only, citing labels; if absent, decline)"
        )
    return f"{e['question']} (closed book: the value, term or name only)"


def judge(e: dict) -> dict:
    text, items = rubric_text(e)
    out = llm_json(
        f"""You are grading one training example for a model that answers questions about US federal
structural engineering documents. Grade strictly against the rubric: decide each rule on its own.

Source the answer must rest on:
{shown(e)}

Question:
{asked(e)}

Answer to grade:
{e["answer"]}

Rubric:
{text}

Return JSON, the reasoning first:
{{"reasoning": "two to four sentences",
 "hard": {{"H1": true or false, ...}},
 "principles": {{"P1": 1-5, ...}},
 "optional": {{"O1": true or false}},
 "pitfalls": {{"X1": true if present, ...}}}}""",
        step="judge",
        model=TEACHER,
        temperature=0.0,
    )
    hard = {i[0]: bool(out.get("hard", {}).get(i[0], False)) for i in items if i[1] == "Hard Rule"}
    princ = {}
    for i in items:
        if i[1] == "Principle":
            try:
                princ[i[0]] = min(5, max(1, int(out.get("principles", {}).get(i[0], 1))))
            except (TypeError, ValueError):
                princ[i[0]] = 1
    pits = {
        i[0]: bool(out.get("pitfalls", {}).get(i[0], False)) for i in items if i[1] == "Pitfall"
    }
    w = {i[0]: i[3] for i in items}
    score = sum(w[k] * s for k, s in princ.items()) / sum(w[k] for k in princ) - sum(
        w[k] / 2 for k, hit in pits.items() if hit
    )
    cls = {i[0]: i[2] for i in items}
    fails = [k for k, ok in hard.items() if not ok] + [k for k, hit in pits.items() if hit]
    fails += [k for k, s in princ.items() if s <= 3]
    return {
        "ok": bool(out) and all(hard.values()) and score >= KEEP_SCORE,
        "failed_call": not out,
        "hard": hard,
        "principles": princ,
        "pitfalls": pits,
        "score": round(score, 2),
        "fails": fails,
        "factual": [k for k in fails if cls[k] == "factual"],
        "reasoning": str(out.get("reasoning", ""))[:600],
        "titles": {i[0]: i[4] for i in items},
    }


def verify(e: dict) -> dict:
    """Rule checks: {rule: (ok, class)}. The rules the judge also checks (gold fact, citations,
    abstain sentence, hedging) are compared with its verdicts as disagreements."""
    a, fmt, kind = e["answer"], e["format"], e["kind"]
    out = {}
    if fmt == "abstain":
        if citations(a) or len(a.split()) > 12:
            out["abstained"] = (False, "factual")  # it answered: the passages hold the answer
        else:
            # case and the final period don't count; the record carries ABSTAIN_REPLY itself
            same = a.strip().rstrip(".").lower() == ABSTAIN_REPLY.rstrip(".").lower()
            out["exact_sentence"] = (same, "format")
    elif fmt == "grounded":
        out["citations"] = (citations_valid(a, {p["label"] for p in e["passages"]}), "format")
        out["answered"] = ("not in the provided passages" not in a.lower(), "factual")
        # every sentence carries its own [Pn] (audit 2026-10-05: one citation at the end of two
        # sentences was 6 of 9 grounded defects, and the judge passed them all)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", a.strip()) if s.strip()]
        out["every_sentence_cited"] = (all(LABEL_CITE.search(s) for s in sentences), "format")
    elif fmt == "definition":
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", a.strip()) if s]
        out["one_sentence"] = (len(sentences) == 1, "format")
        out["no_passage_ref"] = (not PASSAGE_REF.search(a), "format")
    elif kind == "multi_step":
        m = FINAL_ANSWER.search(a)
        out["answer_line"] = (bool(m) and not JSON_DEBRIS.search(a), "format")
        out["no_passage_ref"] = (not PASSAGE_REF.search(a), "format")
        out["gold"] = (bool(m) and numeric_match(m[1], e["gold"]), "factual")
    else:
        out["gold"] = (same_fact(a, e["gold"], kind), "factual")
        out["no_passage_ref"] = (not PASSAGE_REF.search(a), "format")
        out["no_hedge"] = (not HEDGE.search(a), "format")
        out["short"] = (len(a.split()) <= 6, "format")
    return out


# verifier rule -> the judge's hard rule on the same thing, per rubric
SHARED = {
    "closed_book": {"gold": "H2", "no_hedge": "H4"},
    "multi_step": {"gold": "H2", "answer_line": "H4"},
    "grounded": {"citations": "H3", "answered": "H4"},
    "abstain": {"exact_sentence": "H2"},
    "definition": {"one_sentence": "H2"},
}


def revise(e: dict, v: dict, j: dict | None) -> str:
    problems = [f"- {rule.replace('_', ' ')}" for rule, (ok, _) in v.items() if not ok]
    if j:
        problems += [f"- {j['titles'][k]}" for k in j["fails"]]
        problems.append(f"Grader's note: {j['reasoning']}")
    prompt = e["gen_prompt"].rsplit("\nReturn JSON", 1)[0]
    out = llm_json(
        f"""{prompt}

A previous answer was:
{e["answer"]}

It has these format problems, which you must fix:
{chr(10).join(problems)}

Rewrite the answer fixing only these problems. Keep the same facts and add nothing new.
Return JSON: {{"answer": "..."}}""",
        step="revise",
        model=e["teacher"],
        temperature=0.2,
    )
    return str(out.get("answer", "")).strip()


def check(e: dict) -> dict:
    """Verifier, then the judge (skipped for an abstain item the teacher answered). Any factual
    failure, the verifier's or the judge's, rules out the revise round, even where the other check
    disagrees: letting the judge overrule the verifier there (tried first) turned revise into a
    fact-correcting step on 42 final records ("water jetting, ..." -> "roughened surface")."""
    v = verify(e)
    factual_v = [r for r, (ok, cls) in v.items() if not ok and cls == "factual"]
    j = None if factual_v and e["format"] == "abstain" else judge(e)
    disagree = []
    if j and not j["failed_call"]:
        for r, h in SHARED[rubric_key(e)].items():
            if r in v and v[r][0] != j["hard"].get(h, v[r][0]):
                disagree.append(f"{r}:verifier={v[r][0]}/judge={j['hard'][h]}")
    ok = all(ok for ok, _ in v.values()) and bool(j and j["ok"])
    return {
        "v": v,
        "j": j,
        "ok": ok,
        "factual": bool(factual_v or (j and j["factual"])),
        "disagree": disagree,
    }


def process(e: dict) -> dict:
    c = check(e)
    first, revised = c, False
    if not c["ok"] and not c["factual"] and not (c["j"] and c["j"]["failed_call"]):
        new = revise(e, c["v"], c["j"])
        if new and new != e["answer"]:
            e = {**e, "answer_before_revise": e["answer"], "answer": new}
            c, revised = check(e), True
    j = c["j"] or {}
    reason = (
        "kept"
        if c["ok"]
        else "judge_call_failed"
        if j.get("failed_call")
        else "abstain_answered"
        if "abstained" in c["v"] and not c["v"]["abstained"][0]
        else "factual"
        if c["factual"]
        else "format"
    )
    return {
        **e,
        "keep": c["ok"],
        "reason": reason,
        "revised": revised,
        "salvaged": revised and c["ok"],
        "verifier": {r: ok for r, (ok, _) in c["v"].items()},
        "judge": {k: j.get(k) for k in ("hard", "principles", "pitfalls", "score", "reasoning")}
        if j
        else None,
        "first_fails": (first["j"] or {}).get("fails", [])
        + [r for r, (ok, _) in first["v"].items() if not ok],
        "disagree": first["disagree"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=30)
    args = ap.parse_args()
    set_rpm(args.rpm)
    examples = read_jsonl(WORK / "answered.jsonl")
    print(f"judging {len(examples)} examples with {TEACHER}")
    out = pmap(process, examples)
    write_jsonl(WORK / "judged.jsonl", out)

    def rate(pred) -> dict:
        groups = Counter()
        kept = Counter()
        for e in out:
            k = pred(e)
            groups[k] += 1
            kept[k] += e["keep"]
        return {k: f"{kept[k]}/{n} ({kept[k] / n:.1%})" for k, n in sorted(groups.items())}

    revised = [e for e in out if e["revised"]]
    stats = {
        "examples": len(out),
        "kept": sum(e["keep"] for e in out),
        "acceptance_by_format": rate(lambda e: rubric_key(e)),
        "acceptance_by_origin": rate(lambda e: e["origin"]),
        "acceptance_by_teacher": rate(lambda e: e["teacher"]),
        "acceptance_by_teacher_and_format": rate(lambda e: f"{e['teacher']}:{rubric_key(e)}"),
        "reasons": dict(Counter(e["reason"] for e in out)),
        "first_pass_failed_rules": dict(
            Counter(f"{rubric_key(e)}:{r}" for e in out for r in e["first_fails"])
        ),
        "revised": len(revised),
        "salvaged": sum(e["salvaged"] for e in revised),
        "disagreements": dict(
            Counter(d.split(":")[0] + ":" + rubric_key(e) for e in out for d in e["disagree"])
        ),
        "judge_model": TEACHER,
    }
    update_stats("judge", stats)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    report_failures()


if __name__ == "__main__":
    main()
