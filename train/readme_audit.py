"""Every number in the README's prose traces to a file.

  .venv/bin/python train/readme_audit.py           # list untraced numbers; exit 1 if any
  .venv/bin/python train/readme_audit.py --words   # the prose word count

A number in the prose passes if:
- it has three or more significant digits and appears, at the precision written, in one of the
  README's generated blocks (train/report.py: the headline tables). Shorter numbers coincide with
  some table cell too easily (0.41 is in a dozen places), so they need the sidecar; or
- a row of notes/readme_numbers.tsv covers it: `number<TAB>context<TAB>source<TAB>in`, where `context` is
  a few words of the line it sits on (or of the line before and it, joined, for a wrapped
  sentence) and `source` a repo file that contains the number (the script checks that it does, at
  the precision written).

Prose is everything but code fences (the Mermaid diagram too), generated blocks, inline code, link
targets and HTML comments; hand-typed tables count as prose. Not numbers to trace: dates and years, versions, `Stage/rule/step/part
N`, section signs, identifiers with digits in them (H100, r64, FP8, GSM8K, 8B), and integers below
10 written without a decimal or a percent sign (counts such as "2 GPUs" or "1 df").

The word count leaves out tables too (headings included): the measure behind "under 3,000 words".
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
SIDECAR = REPO / "notes/readme_numbers.tsv"

BLOCK = re.compile(r"<!-- ([a-z0-9-]+):start -->.*?<!-- \1:end -->", re.DOTALL)
NUMBER = re.compile(
    r"(?<![\w.@/§])"  # not inside an identifier, a version, pass@k or a path
    r"[+−-]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(%|M|k|x|×)?"
    r"(?![\w.]\d|[A-Za-z0-9])"
)
# a stage, rule or section number, a model or licence version ("Medium 3.5", "CC-BY-4.0")
SKIP_BEFORE = re.compile(
    r"((Stage|stage|rule|Rule|steps?|Steps?|part|Part|§|ch\.|chapter|Section|Ministral|Mistral|Large"
    r"|Medium|Small|Llama|Tülu|Magistral|Apache)\s*(\d+\s*[-–]?)?|[A-Za-z]-)$"
)
CONFIDENCE = re.compile(r"95%(?= CI| confidence)")  # a confidence level, not a measurement
DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\b(19|20)\d{2}\b"
    r"|\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """(line number, text) of the README's prose."""
    keep = BLOCK.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    keep = re.sub(r"<!--.*?-->", "", keep, flags=re.DOTALL)
    out, fence = [], False
    for n, ln in enumerate(keep.splitlines(), 1):
        if ln.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        ln = re.sub(r"`[^`]*`", "", ln)
        ln = re.sub(r"\]\([^)]*\)", "]", ln)
        ln = re.sub(r"https?://\S+", "", ln)
        out.append((n, ln))
    return out


def numbers(line: str) -> list[tuple[str, float, int, str]]:
    """(as written, |value|, decimals, unit) of every number to trace in one line of prose."""
    dates = [m.span() for m in DATE.finditer(line)]
    out = []
    for m in NUMBER.finditer(line):
        if any(a <= m.start() < b for a, b in dates):
            continue
        if SKIP_BEFORE.search(line[: m.start()]) or CONFIDENCE.match(line, m.start()):
            continue
        if re.search(r"\bper\s*$", line[: m.start()]) or line[m.end() :].startswith(" U.S.C"):
            continue  # a unit ("per 1,000 requests") or a legal citation
        whole, frac, unit = m.group(1), m.group(2) or "", m.group(3) or ""
        value = float(whole.replace(",", "") + frac)
        if not frac and not unit and value < 10:
            continue
        if re.match(r"\s*\d+\.\s", line) and m.start() == len(line) - len(line.lstrip()):
            continue  # the marker of a numbered list item
        out.append((m.group(0), value, len(frac) - 1 if frac else 0, unit))
    return out


def significant(written: str) -> int:
    return len(re.sub(r"^0+", "", re.sub(r"\D", "", written)))


def values_in(text: str) -> list[float]:
    return [float(a.replace(",", "") + (b or "")) for a, b, *_ in NUMBER.findall(text)]


def matches(value: float, decimals: int, pool: list[float]) -> bool:
    """Written at `decimals` places, `value` is one of `pool` rounded (or a fraction as a percent)."""
    for y in pool:  # a fraction may be written as a percent; nothing else is scaled
        for v in (abs(y), abs(y) * 100) if abs(y) <= 1 else (abs(y),):
            if round(v, decimals) == value:
                return True
    return False


def sidecar() -> list[dict]:
    if not SIDECAR.exists():
        return []
    with SIDECAR.open() as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r.get("number")]


def source_kind(path: str) -> str:
    """data (results/, the data sets, eval/tasks/), code (scripts and configs), the decision log
    (registered rules and reading rulings only), or prose: the docs, CLAUDE.md, DEPLOY.md and the
    README restate numbers rather than compute them, so no row may point at them."""
    data = ("results/", "data/processed/", "data/sft/", "data/dpo/", "data/grpo/", "eval/tasks/")
    if path.startswith(data):
        return "data"
    if path.endswith((".py", ".yaml", ".yml")):
        return "code"
    return "decision log" if path == "notes/decisions.md" else "prose"


DIAGRAMS = REPO / "docs/diagrams/src"  # the stage diagrams' IR (make diagrams renders them)


def check(
    name: str, lines: list[tuple[int, str]], pool: list[float], rows: list[dict]
) -> list[str]:
    """Every number on `lines` is in `pool` (3+ significant digits) or covered by one of `rows`, and
    every row covers one. `name` labels the problems."""
    problems, used = [], set()
    for k, (n, line) in enumerate(lines):
        window = " ".join(((lines[k - 1][1] + " " if k else "") + line).split())  # wrapped context
        for written, value, decimals, _ in numbers(line):
            if significant(written) >= 3 and matches(value, decimals, pool):
                continue
            hits = [
                j
                for j, r in enumerate(rows)
                if numbers(f" {r['number']} ")[:1]
                and numbers(f" {r['number']} ")[0][1:3] == (value, decimals)
                and " ".join(r["context"].split()).lower() in window.lower()
            ]
            used.update(hits)
            if not hits:
                problems.append(f"{name}:{n}: {written}  | {line.strip()[:90]}")
    problems += [  # a row that covers nothing is a number the file no longer states
        f"{SIDECAR.name}: unused row {r['number']} ({r['context']}) for {name}"
        for j, r in enumerate(rows)
        if j not in used
    ]
    return problems


def diagram_lines(path: Path) -> list[tuple[int, str]]:
    """A diagram IR's node and edge labels, one line each (a label's line breaks joined)."""
    ir = json.loads(path.read_text())
    labels = [n["label"] for n in ir["nodes"]] + [e["label"] for e in ir["edges"] if e.get("label")]
    return [(k + 1, " ".join(label.split("\n"))) for k, label in enumerate(labels)]


def audit(path: Path = README) -> list[str]:
    """The README's prose and every stage diagram's labels, against the sidecar."""
    rows = sidecar()
    problems = []
    for r in rows:  # every sidecar row's number must be in its source, and the source must be data
        src = REPO / r["source"]
        if source_kind(r["source"]) == "prose":
            problems.append(
                f"{SIDECAR.name}: {r['number']} ({r['context']}) points at prose: {r['source']}"
            )
        num = numbers(f" {r['number']} ")
        if not src.exists():
            problems.append(f"{SIDECAR.name}: {r['number']}: no file {r['source']}")
        elif not num or not matches(num[0][1], num[0][2], values_in(src.read_text())):
            problems.append(f"{SIDECAR.name}: {r['number']} not in {r['source']}")
    text = path.read_text()
    pool = values_in("\n".join(m.group(0) for m in BLOCK.finditer(text)))
    readme_rows = [r for r in rows if (r.get("in") or "README.md") == "README.md"]
    problems += check(path.name, prose_lines(text), pool, readme_rows)
    if path == README:
        for ir in sorted(DIAGRAMS.glob("*.json")):
            name = ir.relative_to(REPO).as_posix()
            problems += check(name, diagram_lines(ir), [], [r for r in rows if r.get("in") == name])
    return problems


def word_count(path: Path = README) -> int:
    lines = prose_lines(path.read_text())
    return sum(len(ln.split()) for _, ln in lines if not ln.lstrip().startswith("|"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", action="store_true", help="print the prose word count")
    args = ap.parse_args()
    if args.words:
        print(word_count())
        return
    problems = audit()
    kinds = Counter(source_kind(r["source"]) for r in sidecar())
    done = "every number in the README and the stage diagrams traces to a file: " + ", ".join(
        f"{n} to {k}" for k, n in kinds.most_common()
    )
    print("\n".join(problems) or done)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
