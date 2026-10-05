"""Quality audit of the frozen SFT set: a stratified sample read against the full source passages.

  .venv/bin/python data/scripts/sft_audit.py sample [2] # -> data/sft/work/audit[_r2]_<format>.md
  (auditors write data/sft/work/audit_verdicts_<format>.jsonl, one verdict per record)
  .venv/bin/python data/scripts/sft_audit.py report   # -> data/sft/audit.jsonl, data/sft/audit.md
  .venv/bin/python data/scripts/sft_audit.py filter-packets  # every closed-book record, 10 packets
  (auditors write data/sft/work/filter_cb_verdicts_<k>.jsonl)
  .venv/bin/python data/scripts/sft_audit.py filter-merge    # -> data/sft/closed_book_filter.jsonl

40 records per synthetic format (closed_book, multi_step, definition, grounded, abstain), 30 from
Mistral Large 3 and 10 from Medium 3.5 where there are enough, drawn by hash and disjoint from the
50 in review.md. Each packet holds the exact prompt and completion, the full source passage (or the
four passages a grounded / abstain prompt shows) and the gold fact A2 extracted. Verdicts:
  ok      correct and well-formed for its format
  minor   correct, but a style flaw that teaches nothing wrong (awkward phrasing, trivia, verbose)
  defect  would teach something wrong or malformed (wrong or unsupported fact, wrong citation,
          ambiguous or not-standalone question, abstain that the passages do answer, bad arithmetic)
The rates come with Wilson 95% intervals: 40 records bound a format's defect rate, they don't pin it.
"""

import json
import math
import sys
import textwrap
from collections import Counter, defaultdict

from sft_common import SFT, WORK, h01, read_jsonl, write_jsonl

FORMATS = ("closed_book", "multi_step", "definition", "grounded", "abstain")
PER_FORMAT, MEDIUM = 40, 10
REVIEWED = 10  # sft_assemble.review_md's per-format sample, excluded here


def fmt_key(r: dict) -> str:
    return "multi_step" if r["kind"] == "multi_step" else r["format"]


def wrap(text: str) -> str:
    return "\n".join(textwrap.fill(p, 160) for p in text.split("\n"))


def current() -> dict[str, dict]:
    return {
        r["eid"]: r for name in ("train.jsonl", "sft_val.jsonl") for r in read_jsonl(SFT / name)
    }


def sample(round_: int = 1) -> None:
    """Round 1: every synthetic format. Round 2 (after the fixes): closed_book and grounded again,
    a fresh draw that excludes round 1 and review.md."""
    records = current()
    judged = {e["eid"]: e for e in read_jsonl(WORK / "judged.jsonl")}
    tag = "" if round_ == 1 else f"_r{round_}"
    formats = {1: FORMATS, 2: ("closed_book", "grounded")}.get(round_, ("closed_book",))
    done = {v["eid"] for v in read_jsonl(SFT / "audit.jsonl")} if round_ > 1 else set()
    picked = {}
    for f in formats:
        es = sorted(
            (e for e in records.values() if fmt_key(e) == f),
            key=lambda r: h01(f"sft-review:{r['eid']}"),
        )
        pool = [r for r in es[REVIEWED:] if r["eid"] not in done]
        pool.sort(key=lambda r: h01(f"sft-audit{tag}:{r['eid']}"))
        medium = [r for r in pool if "medium" in r["teacher"]][:MEDIUM]
        large = [r for r in pool if "medium" not in r["teacher"]][: PER_FORMAT - len(medium)]
        picked[f] = [r["eid"] for r in large + medium]
        lines = [f"# Audit packet: {f} ({len(picked[f])} records)", ""]
        for k, eid in enumerate(sorted(picked[f], key=lambda e: h01(f"sft-audit-order:{e}")), 1):
            r, e = records[eid], judged[eid]
            lines += [
                f"## {k}. {eid}",
                "",
                f"format {fmt_key(r)} | origin {r['origin']} | wording {r['wording']}",
                "",
            ]
            if r["format"] in ("grounded", "abstain"):
                lines += [
                    "### Prompt (passages included, as the model sees them)",
                    "",
                    wrap(r["prompt"][0]["content"]),
                    "",
                ]
            else:
                lines += [f"### Source passage ({e['title']})", "", wrap(e["text"]), ""]
                lines += [
                    "### Prompt (as the model sees it)",
                    "",
                    wrap(r["prompt"][0]["content"]),
                    "",
                ]
            lines += [
                "### Completion (the training target)",
                "",
                wrap(r["completion"][0]["content"]),
                "",
            ]
            lines += [f"### Gold fact extracted in A2: {e['gold']}", ""]
            if r["format"] == "abstain":
                lines += [
                    f"(the question came from chunk {e['chunk_id']}, which is NOT among the passages)",
                    "",
                ]
        (WORK / f"audit{tag}_{f}.md").write_text("\n".join(lines) + "\n")
        print(f"{f}: {len(picked[f])} -> data/sft/work/audit{tag}_{f}.md")
    (WORK / f"audit_sample{tag}.json").write_text(json.dumps(picked, indent=1) + "\n")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(vs: list[dict], verdict: str = "defect") -> str:
    k, n = sum(v["verdict"] == verdict for v in vs), len(vs)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}; {lo:.0%}-{hi:.0%})" if n else "-"


ROUNDS = {
    1: "The set as first frozen (commit b20d03d): 40 records per synthetic format, 30 from Mistral "
    "Large 3 and 10 from Medium 3.5, disjoint from the 50 in review.md",
    2: "After the fixes (multi_step dropped, every grounded sentence cited, every closed-book record "
    "filtered by a full-passage reader): a fresh 40 closed-book + 40 grounded, "
    "disjoint from round 1 and review.md",
    3: "After the stricter second filter pass over the closed-book records pass 1 had kept as minor: "
    "a fresh 40 closed-book, disjoint from rounds 1-2 and review.md",
}


def collect(round_: int) -> list[dict]:
    tag = "" if round_ == 1 else f"_r{round_}"
    picked = json.loads((WORK / f"audit_sample{tag}.json").read_text())
    records, out = current(), []
    for f, eids in picked.items():
        got = {v["eid"]: v for v in read_jsonl(WORK / f"audit_verdicts{tag}_{f}.jsonl")}
        missing = set(eids) - set(got)
        assert not missing, f"round {round_} {f}: no verdict for {sorted(missing)[:3]}"
        for eid in eids:
            v, r = got[eid], records[eid]
            assert v["verdict"] in ("ok", "minor", "defect"), v
            out.append(
                {**v, "format": f, "teacher": r["teacher"], "origin": r["origin"], "round": round_}
            )
    return out


def report() -> None:
    """Round 1 stays as recorded in audit.jsonl (its multi_step records are no longer in the set);
    round 2 is collected when its sample exists."""
    path = SFT / "audit.jsonl"
    prev = [{**v, "round": v.get("round", 1)} for v in read_jsonl(path)] if path.exists() else []
    rounds = {}
    for rnd in sorted(ROUNDS):
        tag = "" if rnd == 1 else f"_r{rnd}"
        recorded = [v for v in prev if v["round"] == rnd]
        if recorded:  # frozen once reported: later fixes may drop the records it read
            rounds[rnd] = recorded
        elif (WORK / f"audit_sample{tag}.json").exists():
            rounds[rnd] = collect(rnd)
    out = [v for vs in rounds.values() for v in vs]
    write_jsonl(path, out)

    lines = [
        "# SFT set: full-passage audit",
        "",
        (
            "Records read against their full source passage(s) by an independent LLM reader (a different model "
            "family from the teacher and the judge; not a human read): `data/scripts/sft_audit.py`, "
            "verdicts in `data/sft/audit.jsonl`. Rates are count/n (share; Wilson 95% interval)."
        ),
    ]
    for rnd, vs in rounds.items():
        by = lambda key, vs=vs: {
            k: [v for v in vs if key(v) == k] for k in sorted({key(v) for v in vs})
        }
        lines += [
            "",
            f"## Round {rnd}",
            "",
            f"{ROUNDS[rnd]}.",
            "",
            "| format | defect | minor | ok |",
            "|---|---|---|---|",
        ]
        for f, fv in by(lambda v: v["format"]).items():
            lines.append(f"| {f} | {rate(fv)} | {rate(fv, 'minor')} | {rate(fv, 'ok')} |")
        lines.append(f"| all | {rate(vs)} | {rate(vs, 'minor')} | {rate(vs, 'ok')} |")
        lines += ["", "| teacher | defect | minor |", "|---|---|---|"]
        lines += [
            f"| {t} | {rate(tv)} | {rate(tv, 'minor')} |"
            for t, tv in by(lambda v: v["teacher"]).items()
        ]
        lines += ["", "| origin | defect | minor |", "|---|---|---|"]
        lines += [
            f"| {o} | {rate(ov)} | {rate(ov, 'minor')} |"
            for o, ov in by(lambda v: v["origin"]).items()
        ]
        cats = Counter(
            (v["format"], c) for v in vs if v["verdict"] != "ok" for c in v.get("categories", [])
        )
        lines += ["", "Categories (defect and minor):", ""]
        lines += [f"- {f}: {c} ({n})" for (f, c), n in sorted(cats.items())]
        for verdict in ("defect", "minor"):
            lines += ["", f"### Round {rnd}: every {verdict}", ""]
            for v in sorted(
                (v for v in vs if v["verdict"] == verdict), key=lambda v: (v["format"], v["eid"])
            ):
                lines.append(
                    f"- `{v['eid']}` ({v['format']}, {v['teacher']}): {', '.join(v.get('categories', []))}. {v['note']}"
                )
    (SFT / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


FILTER_PACKETS = 10
FILTER = SFT / "closed_book_filter.jsonl"


def filter_packets(pass_: int = 1) -> None:
    """Pass 1: every closed-book record of the current set, grouped by source chunk (the passage
    once, then each of its questions and phrasings), in FILTER_PACKETS balanced packets for readers
    who apply the closed-book rubric above. Pass 2: the records pass 1 called "minor", in 4 packets,
    for readers with the stricter defect line (round 2 of the audit found 9 of 19 such records
    defective: the pass-1 readers had noticed the flaw and filed it as minor)."""
    records = [r for name in ("train.jsonl", "sft_val.jsonl") for r in read_jsonl(SFT / name)]
    judged = {e["eid"]: e for e in read_jsonl(WORK / "judged.jsonl")}
    first = {v["eid"]: v["verdict"] for v in read_jsonl(FILTER)} if pass_ == 2 else {}
    by_chunk = defaultdict(list)
    for r in records:
        closed_book = r["format"] == "closed_book" and r["kind"] != "multi_step"
        if closed_book and (pass_ == 1 or first.get(r["eid"]) == "minor"):
            by_chunk[judged[r["eid"]]["chunk_id"]].append(r)
    n_packets = FILTER_PACKETS if pass_ == 1 else 4
    name = "filter_cb" if pass_ == 1 else "filter2_cb"
    packets = [[] for _ in range(n_packets)]
    for cid in sorted(by_chunk, key=lambda c: (-len(by_chunk[c]), h01(f"sft-filter:{c}"))):
        min(packets, key=lambda p: sum(len(by_chunk[c]) for c in p)).append(cid)
    for k, cids in enumerate(packets):
        n = sum(len(by_chunk[c]) for c in cids)
        lines = [f"# Closed-book filter packet {k} ({n} records, {len(cids)} passages)", ""]
        for cid in cids:
            e0 = judged[by_chunk[cid][0]["eid"]]
            lines += [f"## Passage {cid} ({e0['title']})", "", wrap(e0["text"]), ""]
            for r in sorted(by_chunk[cid], key=lambda r: r["eid"]):
                lines += [
                    f"### {r['eid']}",
                    f"wording {r['wording']} | gold fact (A2): {judged[r['eid']]['gold']}",
                    f"Q: {r['question']}",
                    f"A: {r['completion'][0]['content']}",
                    "",
                ]
        (WORK / f"{name}_{k}.md").write_text("\n".join(lines) + "\n")
        print(f"packet {k}: {n} records, {len(cids)} passages")


def filter_merge() -> None:
    """The readers' verdicts -> data/sft/closed_book_filter.jsonl (committed; sft_assemble drops
    the defects and refuses a closed-book record without a verdict). A pass-2 verdict replaces the
    pass-1 "minor" it re-read; the first verdict is kept as first_verdict."""
    out = []
    for k in range(FILTER_PACKETS):
        out += read_jsonl(WORK / f"filter_cb_verdicts_{k}.jsonl")
    assert len({v["eid"] for v in out}) == len(out), "duplicate verdicts"
    second = {}
    for k in range(4):
        path = WORK / f"filter2_cb_verdicts_{k}.jsonl"
        if path.exists():
            second.update({v["eid"]: v for v in read_jsonl(path)})
    if second:
        minors = {v["eid"] for v in out if v["verdict"] == "minor"}
        assert set(second) <= minors, "pass 2 re-reads pass-1 minors only"
        out = [
            {**second[v["eid"]], "pass": 2, "first_verdict": v["verdict"], "first_note": v["note"]}
            if v["eid"] in second
            else {**v, "pass": 1}
            for v in out
        ]
    assert all(v["verdict"] in ("ok", "minor", "defect") for v in out)
    write_jsonl(FILTER, sorted(out, key=lambda v: v["eid"]))
    print(Counter(v["verdict"] for v in out))


if __name__ == "__main__":
    cmds = {
        "sample": lambda: sample(int(sys.argv[2]) if len(sys.argv) > 2 else 1),
        "report": report,
        "filter-packets": lambda: filter_packets(int(sys.argv[2]) if len(sys.argv) > 2 else 1),
        "filter-merge": filter_merge,
    }
    cmds[sys.argv[1]]()
