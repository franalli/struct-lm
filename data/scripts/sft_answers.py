"""A4: the teacher's completions (the distillation step) -> data/sft/work/answered.jsonl

  set -a; . ./.env; set +a
  .venv/bin/python data/scripts/sft_answers.py

"Distillation" in the colloquial sense: a stronger model writes the completions the student is
trained to imitate (cross-entropy on its text in Stage 3's run; no logits, no teacher at training
time). The teacher reads the source passage for every format, so what it supplies is "how to
answer from this passage"; the knowledge reaches the student's weights through the facts in its
training examples.

Per task from questions_kept.jsonl, temperature 0.2, one call, teacher by hash of the task id
(TEACHER, or SECOND_TEACHER for SECOND_SHARE of them):
  closed_book  passage + question -> the value / designation / term only, no hedging
  multi_step   passage + an Evol-Instruct problem (written first, TEACHER at 0.7, from a value
               fact: the value plus 1-3 arithmetic steps, every other input stated) -> worked
               arithmetic ending "Answer: <value>"
  definition   passage + term -> one sentence
  grounded     4 passages labelled [P1]..[P4] + question -> 1-3 sentences, [Pn] after each claim
  abstain      the same, with 4 passages that lack the answer -> the exact abstain sentence
               (a teacher that answers instead has found the answer: A5 drops the item)
Passages carry labels, never chunk ids; sft_assemble maps the labels back to ids.

Paraphrases (TEACHER at 0.7, one call per chunk, the same answer reused): one more phrasing per
fact on eval-seen chunks, in a persona's voice; two more per value fact on ordinary chunks, in the
chunk's asker's voice. An eval-seen fact whose question rule 1 removed gets two new questions
written from the fact and the passage alone (from_fact), and the teacher answers the first.
"""

import argparse
from collections import Counter, defaultdict

from sft_common import (
    ABSTAIN_REPLY,
    PERSONAS,
    TEACHER,
    WORK,
    h01,
    llm_json,
    pick,
    pmap,
    read_jsonl,
    report_failures,
    set_rpm,
    teacher_for,
    update_stats,
    write_jsonl,
)


def source(t: dict) -> str:
    return f"Document: {t['title']}\nPassage:\n{t['text']}"


def labelled(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['label']}]\n{p['text']}" for p in passages)


def answer_prompt(t: dict, question: str | None) -> str:
    """The teacher's instruction for one task (stored with the example: A5's revise step reuses it)."""
    if t["format"] == "closed_book" and t["kind"] == "multi_step":
        return f"""You are solving an engineering problem with the document passage below as your source.

{source(t)}

Problem: {question}

Solve it step by step, showing the arithmetic, using the value the passage gives. Keep it short:
one line per step, as one string. End with a last line "Answer: <value with unit>".
Return JSON: {{"answer": "..."}}"""
    if t["format"] == "closed_book":
        return f"""You are answering a question about a US federal structural engineering document, using the passage below as your source.

{source(t)}

Question: {question}

Reply with the answer only: the value with its unit, or the term, name or designation, as the
passage states it, in at most 6 words. No explanation, no hedging, no restating the question.
Return JSON: {{"answer": "..."}}"""
    if t["format"] == "definition":
        return f"""{source(t)}

Define the term "{t["term"]}" in one precise sentence, faithful to how the passage uses it: say
what it is or does (the defining mechanism or property), not a synonym. Don't begin by repeating
the term, and don't mention the passage or the document.
Return JSON: {{"answer": "..."}}"""
    return f"""Answer the question using only the passages below. After each claim, cite the label of the
passage that supports it in square brackets, for example [P2]. Answer in one to three sentences,
directly, without restating the question. If the passages do not contain the answer, reply
exactly: {ABSTAIN_REPLY}

{labelled(t["passages"])}

Question: {question}
Return JSON: {{"answer": "..."}}"""


def paraphrase(group: list[dict]) -> dict[str, list[str]]:
    """Rewrites for one chunk's closed-book tasks: {tid: [rewrite, ...]}."""
    t0 = group[0]
    lines = []
    for n, t in enumerate(group, 1):
        k = t["phrasings"] - 1
        lines.append(
            f"{n}. [{k} rewrite{'s' if k > 1 else ''}, asked by {t['_asker']}] {t['question']}  (answer: {t['answer']})"
        )
    listing = "\n".join(lines)
    out = llm_json(
        f"""Rewrite each numbered question below as new questions that ask for exactly the same fact (the
same answer), in the voice of the asker given in brackets. Every rewrite must still stand alone
without any passage: keep the document, structure, load condition, material or quantity it names,
and its units and conditions. Change the wording and the sentence structure, not the meaning, and
never mention a passage, figure, table or page.

Document: {t0["title"]}

{listing}

Return JSON: {{"items": [{{"n": 1, "rewrites": ["...", "..."]}}]}}""",
        step="paraphrase",
        temperature=0.7,
    )
    res = {}
    for it in out.get("items", []):
        if not isinstance(it, dict):
            continue
        try:
            n = int(it.get("n") or 0)
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(group):
            continue
        t = group[n - 1]
        rw = [str(r).strip() for r in it.get("rewrites", []) if str(r).strip()]
        res[t["tid"]] = rw[: t["phrasings"] - 1]
    return res


def from_fact(t: dict) -> list[str]:
    """New questions for an eval-seen fact whose own question rule 1 removed (sft_filter): written
    from the fact and the passage, never from that question, so its wording can't carry over."""
    out = llm_json(
        f"""{source(t)}

Write {t["phrasings"]} different questions {t["_asker"]} would ask whose answer is exactly this fact:
{t["answer"]} ({t["what"] or "as the passage states it"}).
They are asked WITHOUT the passage: name the document, structure, load condition, material or
quantity so each pins down this one fact; never mention a passage, figure, table or page. Vary the
wording and the sentence structure between them.
Return JSON: {{"questions": ["...", "..."]}}""",
        step="paraphrase",
        temperature=0.7,
    )
    return [q for q in (as_text(x) for x in out.get("questions", [])) if q][: t["phrasings"]]


def reask_question(t: dict) -> list[str]:
    """One new question for a seen fact every earlier phrasing of which the full-passage read
    dropped: written from the fact, in the eval's neutral register, keeping the passage's own
    conditions and strength of obligation (the reads' commonest defects were "should" asked as
    "must" and scope moved by a persona)."""
    out = llm_json(
        f"""{source(t)}

Write one question a practicing structural or civil engineer would ask whose answer is exactly
this fact: {t["answer"]} ({t["what"] or "as the passage states it"}).
It is asked WITHOUT the passage, so:
- name the document and the structure, condition or quantity, so it has this one answer;
- keep the passage's own conditions, limits and strength of obligation: "should", "typically",
  "may" and "about" stay as they are; never turn them into "must", "required", "maximum" or
  "minimum", and never move the fact to another structure type or situation;
- no persona and no scenario; never mention a passage, figure, table or page.
Return JSON: {{"question": "..."}}""",
        step="paraphrase",
        temperature=0.7,
    )
    q = as_text(out.get("question"))
    return [q] if q else []


def reask_quoted(t: dict) -> list[str]:
    """The second re-ask: quote first, then ask. The quote makes the teacher carry the fact's own
    conditions, scope and limits into the question, which the first re-ask mostly dropped."""
    out = llm_json(
        f"""{source(t)}

The fact: {t["answer"]} ({t["what"] or "as the passage states it"}).
1. Quote, word for word, the sentence or sentences of the passage that state this fact, including
   every condition, scope, exception and limit they attach to it (which structure, which method,
   which case; "less than" vs "not more than"; footnotes such as "not applicable to ...").
2. Write one question a practicing structural or civil engineer would ask whose answer is exactly
   this fact. It is asked WITHOUT the passage: name the document, and carry every condition from
   your quote into the question, so the fact is true as asked. Keep the passage's strength of
   obligation ("should" stays "should"). No persona, no scenario, no mention of a passage, figure,
   table or page.
Return JSON: {{"quote": "...", "question": "..."}}""",
        step="paraphrase",
        temperature=0.7,
    )
    q = as_text(out.get("question"))
    return [q] if q else []


def as_text(x) -> str:
    """A JSON field as text: strings as they are, a list of steps one per line, anything else ""."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list) and all(isinstance(i, str) for i in x):
        return "\n".join(i.strip() for i in x).strip()
    return ""


def evolve(t: dict) -> dict:
    """An Evol-Instruct problem built on one value fact: {problem, final_answer} or {}."""
    out = llm_json(
        f"""{source(t)}

The passage gives this value: "{t["question"]}" -> {t["answer"]}.
Write one short engineering problem (2 to 4 sentences of plain text) that someone who knows this
document could solve without the passage. It must:
- name the document the value comes from, and NOT state the value itself: the solver has to know it;
- state every other input (dimensions, loads, other factors) with its unit;
- need the document's value plus one to three arithmetic steps (applying a factor, converting
  units, checking a limit) and ask for exactly one numeric result;
- make no other design assumptions, and never mention the passage, a figure, a table or a page.
Then solve it.
Return JSON: {{"problem": "the problem, one string", "solution": "the steps, one string", "final_answer": "one number with its unit"}}""",
        step="evol",
        temperature=0.7,
    )
    problem, final = as_text(out.get("problem")), as_text(out.get("final_answer"))
    if problem and final and t["answer"] not in problem:
        return {"problem": problem, "final_answer": final}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=30)
    args = ap.parse_args()
    set_rpm(args.rpm)
    tasks = read_jsonl(WORK / "questions_kept.jsonl")
    dropped = Counter()

    # phrasings beyond the first: eval-seen facts in a persona's voice, ordinary values in the asker's
    for t in tasks:
        forced = t["origin"] == "eval_seen"
        t["_asker"] = pick(f"sft-persona:{t['qid']}", PERSONAS) if forced else t["asker"]
    groups = defaultdict(list)
    for t in tasks:
        if t["phrasings"] > 1 and not t["rewrite_only"]:
            groups[t["chunk_id"]].append(t)
    fresh = [t for t in tasks if t["rewrite_only"]]
    print(f"paraphrases: {len(groups)} chunk calls, {len(fresh)} rule-1 facts re-asked")
    rewrites = {}
    for res in pmap(paraphrase, list(groups.values())):
        rewrites.update(res)
    for t, qs in zip(
        fresh,
        pmap(
            lambda t: (
                reask_quoted(t)
                if t["reask"] == 2
                else reask_question(t)
                if t["reask"]
                else from_fact(t)
            ),
            fresh,
        ),
    ):
        rewrites[t["tid"]] = qs
    for t in tasks:  # every phrasing of the task; the first is the one the teacher answers
        if t["rewrite_only"]:
            t["questions"] = rewrites.get(t["tid"], [])
        elif t["phrasings"] > 1:
            t["questions"] = [t["question"], *rewrites.get(t["tid"], [])]
        else:
            t["questions"] = [t["question"]]

    multi = [t for t in tasks if t["kind"] == "multi_step"]
    print(f"evol: {len(multi)} problem calls")
    for t, ev in zip(multi, pmap(evolve, multi)):
        if ev:
            t["answer"] = ev["final_answer"]
            t["question"], t["questions"] = ev["problem"], [ev["problem"]]
        else:
            t["questions"] = []
            dropped["evol_failed"] += 1
    dropped["no_question"] = sum(not t["questions"] for t in tasks if t["kind"] != "multi_step")
    tasks = [t for t in tasks if t["questions"]]

    print(f"answers: {len(tasks)} calls")

    def answer(t: dict) -> dict:
        question = t["questions"][0] if t["format"] != "definition" else None
        prompt = answer_prompt(t, question)
        teacher = teacher_for(t["tid"])
        out = llm_json(prompt, step="answer", model=teacher, temperature=0.2)
        return {
            "answer": as_text(out.get("answer")),
            "teacher": teacher,
            "gen_prompt": prompt,
        }

    examples = []
    for t, a in zip(tasks, pmap(answer, tasks)):
        if not a["answer"]:
            dropped["answer_failed"] += 1
            continue
        forced = t["origin"] == "eval_seen"
        questions = t["questions"]
        if t["phrasings"] > 1 and len(questions) < t["phrasings"]:
            dropped["paraphrase_missing"] += t["phrasings"] - len(questions)
        for i, q in enumerate(questions, 1):
            if forced and t["format"] == "closed_book":
                wording, persona = ("exact", None) if i == 1 else ("paraphrased", t["_asker"])
            elif forced:  # a forced term's definition: one phrasing, its half by hash
                wording = "exact" if h01(f"sft-wording:{t['qid']}") < 0.5 else "paraphrased"
                persona = None
            else:
                wording, persona = t["wording"], t["persona"]
            examples.append(
                {
                    "eid": f"{t['tid']}:{i}",
                    "tid": t["tid"],
                    "qid": t["qid"],
                    "format": t["format"],
                    "kind": t["kind"],
                    "chunk_id": t["chunk_id"],
                    "doc": t["doc"],
                    "origin": t["origin"],
                    "title": t["title"],
                    "text": t["text"],
                    "question": q,
                    "term": t["term"],
                    "gold": t["definition"] if t["format"] == "definition" else t["answer"],
                    "what": t["what"],
                    "phrasing": i,
                    "paraphrase_of": t["qid"] if i > 1 or t["rewrite_only"] else None,
                    "wording": wording,
                    "persona": persona,
                    "passages": t["passages"],
                    "gold_label": t["gold_label"],
                    "hard_negative": t["hard_negative"],
                    "reask": t["reask"],
                    "targeted": t["targeted"],
                    **a,
                }
            )
    write_jsonl(WORK / "answered.jsonl", examples)
    stats = {
        "tasks": len(tasks),
        "examples": dict(Counter(f"{e['origin']}:{e['format']}" for e in examples)),
        "teacher": dict(Counter(e["teacher"] for e in examples if e["phrasing"] == 1)),
        "dropped": dict(dropped),
        "teacher_model": TEACHER,
    }
    update_stats("answers", stats)
    print(stats)
    report_failures()


if __name__ == "__main__":
    main()
