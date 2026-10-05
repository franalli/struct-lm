"""A6 + A7: assemble, gate, split and freeze the SFT set
-> data/sft/train.jsonl, data/sft/sft_val.jsonl, data/sft/review.md, data/sft/SHA256SUMS

  .venv/bin/python data/scripts/sft_assemble.py
  .venv/bin/python eval/contamination.py --only sft      # then: section 6 of results/contamination.md

One record per line, in the trainer's conversational prompt-completion form: `prompt` (the user
turn) and `completion` (the assistant turn), each a one-message list; every other field is
metadata (tests/test_sft_data.py checks the schema).

Prompts. Half of each format carries the eval's own instruction text (wording "exact"): exactly
prompts.qa_prompt(question, fewshot) / vocab_prompt(term) / grounded_prompt(question, passages),
the text run_eval sends a chat model as its user turn. The other half (wording "paraphrased")
uses sft_common's QA / DEF / RAG templates with the persona's question. multi_step problems have no
eval counterpart (wording null). Passage labels [P1]..[P4] become chunk ids, in the prompt and in
the completion's citations.

Gates on every record (rejects appended to sft_rejected.jsonl with stage "A6"):
  rule 1 again on the final question (A4's paraphrases and Evol-Instruct problems are new text);
  every chunk id in a prompt or a citation is allowed (sft_guard); per-chunk caps (12 / 4).
Then ordinary records are trimmed to the format targets (sft_filter.TARGETS, scaled to the pool),
alternating the exact and paraphrased halves; eval-seen records are all kept, except that
definitions are capped at DEFINITION_CAP (seen-term definitions first). Multi-step problems are
dropped (2026-10-05 audit: 17/40 ill-posed).

Full-passage reads (data/sft/read_filter.jsonl, rubrics in data/sft/read_rubrics.md): every
closed-book record, definition and grounded answer, every hard-negative abstain set and every
record of a regenerated document needs a verdict for its current content; defects are dropped and unread
records are left out and listed in work/unread.jsonl (sft_audit.py read-packets reads them next).

fact_seen: the seen-half eval items (domain_qa / vocab, same chunk) a record's answer matches,
scored the eval's way. Coverage = share of the 167 + 101 seen items with at least one record: the
facts SFT showed, as opposed to the chunks it drew on. Extraction was blind to them; the re-asked
facts and the targeted seen-term definitions (sft_filter) are the two deliberate exceptions.

Validation: 5% of each format's holdable records (ordinary and replay) held out, by source
chunk, so no fact and none of its phrasings sits on both sides. Eval-seen records always go to
train, so they don't count toward the 5%: definitions are mostly eval-seen, and 5% of the format
would have taken nearly every ordinary one.
Tokens: mistral-common's chat encoding of [user, assistant] (gold_lp.token_ids, the eval's --chat
rendering, no system prompt); every record must fit 4,096 with the prompt a prefix of the whole.
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# eval/ modules (prompts, scorers, ...), as eval/run_eval.py imports them
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from gold_lp import token_ids
from prompts import grounded_prompt, qa_prompt, vocab_prompt
from sft_common import (
    ABSTAIN_REPLY,
    BASE,
    DEF_TEMPLATES,
    MULTI_STEP_SUFFIX,
    QA_TEMPLATES,
    RAG_TEMPLATES,
    READS,
    REGENERATED_DOCS,
    SFT,
    WORK,
    fingerprint,
    h01,
    pick,
    read_jsonl,
    update_stats,
    write_jsonl,
)
from sft_filter import CAP, MIN_ORDINARY_DEFINITIONS, TARGETS

# read categories about the answer, not the question: they hold for every phrasing of the task
ANSWER_DEFECTS = {"wrong_answer", "partial_answer", "worked_example_value"}
DEFINITION_CAP = 300  # user decision 2026-10-05: definitions were 772, a quarter of the set
from sft_guard import Guard

MAX_TOKENS = 4096
VAL_SHARE = 0.05
REVIEW = {"closed_book": 15, "definition": 10, "grounded": 10, "abstain": 10, "new": 5}
LABEL = re.compile(r"\[(P[1-4])\]")


def fmt_key(e: dict) -> str:
    return "multi_step" if e["kind"] == "multi_step" else e["format"]


def prompt_text(e: dict, guard: Guard) -> str:
    q, w = e["question"], e["wording"]
    if e["kind"] == "multi_step":
        return q + MULTI_STEP_SUFFIX
    if e["format"] == "closed_book":
        return (
            qa_prompt(q, guard.fewshot)
            if w == "exact"
            else pick(f"sft-tpl:{e['eid']}", QA_TEMPLATES).format(q=q)
        )
    if e["format"] == "definition":
        return (
            vocab_prompt(e["term"])
            if w == "exact"
            else pick(f"sft-tpl:{e['eid']}", DEF_TEMPLATES).format(t=e["term"])
        )
    ctx = [{"chunk_id": p["chunk_id"], "text": p["text"]} for p in e["passages"]]
    if w == "exact":
        return grounded_prompt(q, ctx)
    instr, layout = pick(f"sft-tpl:{e['eid']}", RAG_TEMPLATES)
    passages = "\n\n".join(layout.format(id=c["chunk_id"], text=c["text"]) for c in ctx)
    return instr.format(passages=passages, q=q)


def record(e: dict, guard: Guard) -> dict:
    label_id = {p["label"]: p["chunk_id"] for p in e["passages"] or []}
    completion = e["answer"]
    if e["format"] == "grounded":
        completion = LABEL.sub(lambda m: f"[{label_id[m[1]]}]", completion)
        cited = list(dict.fromkeys(label_id[c] for c in LABEL.findall(e["answer"])))
        sources, distractors = cited, [i for i in label_id.values() if i not in cited]
    elif e["format"] == "abstain":
        completion, sources, distractors = ABSTAIN_REPLY, [], list(label_id.values())
    elif e["kind"] == "multi_step":  # the result on its own last line, as MULTI_STEP_SUFFIX asks
        completion = "\n".join(s.strip() for s in completion.split("|") if s.strip())  # "a|b" steps
        completion = re.sub(
            r"[\s.,;]*answer\s*:\s*([^\n]+?)\s*$", r"\nAnswer: \1", completion, flags=re.IGNORECASE
        )
        sources, distractors = [e["chunk_id"]], []
    else:
        sources, distractors = [e["chunk_id"]], []
    multi = e["kind"] == "multi_step"
    if e["origin"] == "eval_seen":
        fs = guard.fact_seen(
            e["chunk_id"],
            e["answer"] if e["format"] == "closed_book" else None,
            e["term"] if e["format"] == "definition" else None,
        )
    else:
        fs = None
    return {
        "prompt": [{"role": "user", "content": prompt_text(e, guard)}],
        "completion": [{"role": "assistant", "content": completion}],
        "format": e["format"],
        "kind": e["kind"] if e["format"] != "definition" else "term",
        "source_chunks": sources,
        "distractors": distractors,
        "split": "seen",
        "persona": None if multi else e["persona"],
        "wording": None if multi else e["wording"],
        "paraphrase_of": e["paraphrase_of"],
        "teacher": e["teacher"],
        "rubric": {
            "hard": "pass",
            "principle": e["judge"]["score"],
            "verifier": "pass",
            "revised": e["revised"],
        },
        "origin": e["origin"],
        "question": e["term"] if e["format"] == "definition" else e["question"],
        "fact_id": e["qid"],
        "fact_seen": fs,
        "final_answer": e["gold"] if multi else None,
        "replay_source": None,
        "eid": e["eid"],
    }


def replay_record(r: dict) -> dict:
    user, assistant = r["messages"]
    return {
        "prompt": [{"role": "user", "content": user["content"]}],
        "completion": [{"role": "assistant", "content": assistant["content"]}],
        "format": "replay",
        "kind": None,
        "source_chunks": [],
        "distractors": [],
        "split": None,
        "persona": None,
        "wording": None,
        "paraphrase_of": None,
        "teacher": "allenai/tulu-3-sft-mixture",
        "rubric": None,
        "origin": "replay",
        "question": None,
        "fact_id": None,
        "fact_seen": None,
        "final_answer": None,
        "replay_source": f"{r['source']}#{r['id']}",
        "eid": f"replay:{r['id']}",
    }


def ids_in(rec: dict) -> set[str]:
    out = set(rec["source_chunks"]) | set(rec["distractors"])
    if rec["fact_id"]:
        out.add(rec["fact_id"].rsplit(":", 1)[0])
    return out


def review_md(sample: list[dict], record_of: dict) -> str:
    """The human read before a freeze: a stratified sample of the final set, each record as the
    student sees it (prompt and completion) with its source passage, plus its read verdict."""

    def block(e: dict) -> list[str]:
        r = record_of[e["eid"]]
        out = [
            f"### {e['eid']}",
            "",
            f"{fmt_key(e)} | {e['origin']} | wording {r['wording']} | teacher {e['teacher']}"
            + (" | re-asked" if e["reask"] else "")
            + (" | targeted term" if e["targeted"] else "")
            + (" | hard negative" if e["hard_negative"] else ""),
            "",
        ]
        if e["passages"]:
            for p in e["passages"]:
                gold = p["label"] == e["gold_label"]
                text = p["text"] if gold else p["text"][:400] + " ..."
                out += [f"- **[{p['chunk_id']}]**{' (source)' if gold else ''}: {text}"]
            if e["format"] == "abstain":
                out.append(f"- (the gold, from another chunk: {e['gold']})")
        else:
            out.append(f"Source passage `{e['chunk_id']}` ({e['title']}): {e['text']}")
        question = e["term"] if e["format"] == "definition" else e["question"]
        out += ["", f"**Asked:** {question}", f"**Target:** {r['completion'][0]['content']}"]
        return out + [""]

    lines = [
        "# SFT data: sample for the human read",
        "",
        (
            f"{len(sample)} records of the final set, stratified by format, each with its source "
            "passage. Every closed-book record and definition, every hard-negative abstain set and "
            "every record of a regenerated document was read against its passage before this "
            "(data/sft/read_filter.jsonl); this read is the human check of what survived."
        ),
        "",
    ]
    for e in sample:
        lines += block(e)
    return "\n".join(line.rstrip() for line in "\n".join(lines).split("\n")).rstrip() + "\n"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    guard = Guard()
    pool = read_jsonl(WORK / "pool.jsonl")
    judged = read_jsonl(WORK / "judged.jsonl")
    rejects = [r for r in read_jsonl(SFT / "sft_rejected.jsonl") if r.get("stage") == "A3"]

    def reject(e: dict, stage: str, rule: str, eval_id: str | None = None) -> None:
        rejects.append(
            {
                "stage": stage,
                "rule": rule,
                "qid": e["qid"],
                "eid": e["eid"],
                "chunk_id": e["chunk_id"],
                "kind": e["kind"],
                "question": e["question"] or e["term"],
                "answer": e["answer"],
                "eval_id": eval_id,
            }
        )

    for e in judged:
        if not e["keep"]:
            reject(e, "A5", e["reason"])
    candidates = []
    for e in (e for e in judged if e["keep"]):
        if e["kind"] == "multi_step":  # dropped from v1: 17 of 40 audited were defective
            reject(e, "A6", "multi_step_dropped")
            continue
        if e["format"] == "definition":
            hit = guard.term_block(e["term"])
        else:
            hit = guard.rule1(e["question"])
        ids = {e["chunk_id"]} | {p["chunk_id"] for p in e["passages"] or []}
        if hit:
            reject(e, "A6", "rule1", hit)
        elif not all(guard.allowed(i) for i in ids):
            reject(e, "A6", "chunk_not_allowed")
        else:
            candidates.append(e)

    # ---- eval-seen: all kept; ordinary: trimmed to targets, alternating the wording halves ----
    forced = [e for e in candidates if e["origin"] == "eval_seen"]
    scale = sum(c["origin"] == "ordinary" for c in pool) / 900
    forced_defs = sum(e["format"] == "definition" for e in forced)
    target = {k: round(v * scale) for k, v in TARGETS.items()}
    target["definition"] = round(
        max(MIN_ORDINARY_DEFINITIONS * scale, TARGETS["definition"] * scale - forced_defs)
    )
    ordinary = [e for e in candidates if e["origin"] == "ordinary"]
    by_task = defaultdict(list)
    for e in ordinary:
        by_task[e["tid"]].append(e)
    chosen = []
    for key in ("value", "identifier", "multi_step", "grounded", "abstain", "definition"):

        def match(t: list[dict], key: str = key) -> bool:
            e = t[0]
            k = e["kind"] if e["format"] == "closed_book" else e["format"]
            return k == key

        tasks = sorted(
            (t for t in by_task.values() if match(t)), key=lambda t: h01(f"sft-trim:{t[0]['tid']}")
        )
        halves = [[t for t in tasks if t[0]["wording"] == w] for w in ("exact", "paraphrased")]
        mixed = [t for pair in zip(*halves) for t in pair] + max(halves, key=len)[
            min(map(len, halves)) :
        ]
        for t in mixed[: target[key]]:
            chosen += t
    chosen_ids = {e["eid"] for e in chosen}
    for e in ordinary:
        if e["eid"] not in chosen_ids:
            reject(e, "A6", "over_target")
    selected = forced + chosen

    # ---- per-chunk caps on the final records (A3 budgeted them; this is the check) ----
    count = Counter()
    final = []
    for e in selected:
        c = e["chunk_id"]
        if e["reask"] or e["targeted"]:  # added on top of the budget by design (sft_filter)
            final.append(e)
            continue
        if count[c] >= CAP[e["origin"]]:
            reject(e, "A6", "cap")
            continue
        count[c] += 1
        final.append(e)

    # ---- definitions: at most DEFINITION_CAP, first every one that defines a seen-half vocab
    # term, then by hash. Before the reads, so the same records are chosen whether read or not.
    defs = [e for e in final if e["format"] == "definition"]
    covers = lambda e: bool(guard.fact_seen(e["chunk_id"], None, e["term"]))
    defs.sort(key=lambda e: (not covers(e), h01(f"sft-def-cap:{e['eid']}")))
    over = {e["eid"] for e in defs[DEFINITION_CAP:]}
    for e in defs[DEFINITION_CAP:]:
        reject(e, "A6", "definition_cap")
    final = [e for e in final if e["eid"] not in over]

    # ---- full-passage reads (sft_audit.py read-packets / read-merge, rubrics in read_rubrics.md):
    # a verdict holds for the content it read (fingerprint); defects out, minors kept. Every
    # closed-book record, definition and grounded answer is read, and every hard-negative abstain
    # set and record of a regenerated document. Unread ones stay out (work/unread.jsonl lists them for the next read).
    reads = {v["eid"]: v for v in read_jsonl(READS)} if READS.exists() else {}

    def must_read(e: dict) -> bool:
        return (
            e["format"] in ("closed_book", "definition", "grounded")
            or e["hard_negative"]
            or e["doc"] in REGENERATED_DOCS
        )

    def verdict(e: dict) -> str | None:
        v = reads.get(e["eid"])
        return v["verdict"] if v and v["fp"] == fingerprint(e) else None

    # A defect in the answer itself condemns every phrasing of the task: they share one answer.
    shared_answer = {
        e["tid"]
        for e in judged
        if verdict(e) == "defect" and set(reads[e["eid"]]["categories"]) & ANSWER_DEFECTS
    }
    unread = [e for e in final if must_read(e) and verdict(e) is None]
    for e in final:
        if verdict(e) == "defect":
            reject(e, "A6", "read_defect")
        elif e["tid"] in shared_answer:
            reject(e, "A6", "read_defect_shared_answer")
    final = [
        e
        for e in final
        if verdict(e) != "defect"  # whatever the format: a read that found a defect drops it
        and not (must_read(e) and verdict(e) is None)
        and e["tid"] not in shared_answer
    ]
    write_jsonl(WORK / "unread.jsonl", unread)
    if unread:
        print(f"UNREAD: {len(unread)} records left out; sft_audit.py read-packets, then read-merge")
    records = [record(e, guard) for e in final]
    records += [replay_record(r) for r in read_jsonl(WORK / "replay.jsonl")]

    # ---- tokens ----
    for r in records:
        enc = token_ids(BASE, True, r["prompt"][0]["content"], r["completion"][0]["content"])
        assert enc is not None, f"{r['eid']}: prompt tokens are not a prefix of prompt + completion"
        r["n_tokens"] = len(enc[0])
        assert r["n_tokens"] <= MAX_TOKENS, f"{r['eid']}: {r['n_tokens']} tokens"

    # ---- coverage of the seen half ----
    covered = {i for r in records for i in r["fact_seen"] or []}
    seen_qa = [i["id"] for i in guard.qa if i["source_chunk"] in guard.seen]
    seen_vocab = [i["id"] for i in guard.vocab if i["source_chunk"] in guard.seen]

    # ---- validation split: by source chunk, stratified by format; eval-seen always train ----
    group = lambda r: r["fact_id"].rsplit(":", 1)[0] if r["fact_id"] else r["eid"]
    by_format = defaultdict(list)
    for r in records:
        by_format[fmt_key(r) if r["format"] != "replay" else "replay"].append(r)
    val_groups = set()
    for f, rs in by_format.items():
        groups = defaultdict(list)
        for r in rs:
            if r["origin"] != "eval_seen":
                groups[group(r)].append(r)
        want = round(VAL_SHARE * sum(map(len, groups.values())))
        n = 0
        for g in sorted(groups, key=lambda g: h01(f"sft-val:{g}")):
            if n >= want:
                break
            if g in val_groups:
                continue
            val_groups.add(g)
            n += len(groups[g])
    forced_groups = {group(r) for r in records if r["origin"] == "eval_seen"}
    assert not val_groups & forced_groups
    order = lambda r: h01(f"sft-order:{r['eid']}")
    train = sorted((r for r in records if group(r) not in val_groups), key=order)
    val = sorted((r for r in records if group(r) in val_groups), key=order)
    for r in train + val:
        assert all(guard.allowed(i) for i in ids_in(r)), r["eid"]

    write_jsonl(SFT / "train.jsonl", train)
    write_jsonl(SFT / "sft_val.jsonl", val)
    write_jsonl(SFT / "sft_rejected.jsonl", rejects)
    (SFT / "SHA256SUMS").write_text(
        "".join(f"{sha256(SFT / n)}  {n}\n" for n in ("train.jsonl", "sft_val.jsonl"))
    )

    kept_judged = {e["eid"]: e for e in final}
    order = lambda e: h01(f"sft-review:{e['eid']}")
    new = sorted((e for e in final if e["reask"] or e["targeted"]), key=order)[: REVIEW["new"]]
    sample = list(new)
    for f in ("closed_book", "definition", "grounded", "abstain"):
        es = sorted((e for e in final if fmt_key(e) == f and e not in new), key=order)
        sample += es[: REVIEW[f]]
    (SFT / "review.md").write_text(review_md(sample, {r["eid"]: r for r in records}))

    toks = sorted(r["n_tokens"] for r in train + val)
    reads_used = Counter(reads[e["eid"]]["read"] for e in final if verdict(e))
    stats = {
        "train": len(train),
        "val": len(val),
        "by_format": dict(
            Counter(
                f"{fmt_key(r) if r['format'] != 'replay' else 'replay'}:{r['origin']}"
                for r in records
            )
        ),
        "val_by_format": dict(
            Counter(fmt_key(r) if r["format"] != "replay" else "replay" for r in val)
        ),
        "wording": dict(
            Counter(f"{fmt_key(r)}:{r['wording']}" for r in records if r["format"] != "replay")
        ),
        "teacher": dict(Counter(r["teacher"] for r in records)),
        "targets": target,
        "forced_records": len(forced),
        "seen_coverage": {
            "domain_qa": f"{len(covered & set(seen_qa))}/{len(seen_qa)}",
            "vocab": f"{len(covered & set(seen_vocab))}/{len(seen_vocab)}",
        },
        "tokens": {"max": toks[-1], "median": toks[len(toks) // 2], "total": sum(toks)},
        "unread": len(unread),
        "reads_used": dict(reads_used),
        "rejected": dict(Counter(f"{r['stage']}:{r['rule']}" for r in rejects)),
        "review": len(sample),
        "sha256": {n: sha256(SFT / n) for n in ("train.jsonl", "sft_val.jsonl")},
    }
    assert len(kept_judged) == len(final)
    update_stats("assemble", stats)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
