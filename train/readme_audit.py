"""Every number in the README's prose traces to a file.

  .venv/bin/python train/readme_audit.py           # list untraced numbers; exit 1 if any
  .venv/bin/python train/readme_audit.py --words   # the prose word count

A number in the prose passes if:
- it has three or more significant digits and appears, at the precision written, in one of the
  README's generated blocks (train/report.py: the headline tables). Shorter numbers coincide with
  some table cell too easily (0.41 is in a dozen places), so they need the sidecar; or
- a row of notes/readme_numbers.tsv covers it: `number<TAB>context<TAB>source`, where `context` is
  a few words of the line it sits on and `source` a repo file that contains the number (the
  script checks that it does, at the precision written).

Prose excludes code fences (the Mermaid diagram too), generated blocks, tables, inline code, link
targets and HTML comments. Not numbers to trace: dates and years, versions, `Stage/rule/step/part
N`, section signs, identifiers with digits in them (H100, r64, FP8, GSM8K, 8B), and integers below
10 written without a decimal or a percent sign (counts such as "2 GPUs" or "1 df").

The word count is the same prose (headings included), the measure behind "under 3,000 words".
"""

import argparse
import csv
import re
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
    r"((Stage|stage|rule|Rule|step|Step|part|Part|§|ch\.|chapter|Section|Ministral|Mistral|Large"
    r"|Medium|Small|Llama|Tülu|Magistral|Apache)\s*|[A-Za-z]-)$"
)
CONFIDENCE = re.compile(r"95%(?= CI| confidence)")  # a confidence level, not a measurement
DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\b(19|20)\d{2}\b|\b\d{1,2} (Oct|Sep)\w*")


def prose_lines(text: str) -> list[tuple[int, str]]:
    """(line number, text) of the README's prose."""
    keep = BLOCK.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    keep = re.sub(r"<!--.*?-->", "", keep, flags=re.DOTALL)
    out, fence = [], False
    for n, ln in enumerate(keep.splitlines(), 1):
        if ln.startswith("```"):
            fence = not fence
            continue
        if fence or ln.lstrip().startswith("|"):
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


def audit(path: Path = README) -> list[str]:
    text = path.read_text()
    blocks = [m.group(0) for m in BLOCK.finditer(text)]
    block_pool = values_in("\n".join(blocks))
    rows = sidecar()
    problems = []
    for r in rows:  # every sidecar row's number must be in its source
        src = REPO / r["source"]
        num = numbers(f" {r['number']} ")
        if not src.exists():
            problems.append(f"{SIDECAR.name}: {r['number']}: no file {r['source']}")
        elif not num or not matches(num[0][1], num[0][2], values_in(src.read_text())):
            problems.append(f"{SIDECAR.name}: {r['number']} not in {r['source']}")
    for n, line in prose_lines(text):
        for written, value, decimals, _ in numbers(line):
            if significant(written) >= 3 and matches(value, decimals, block_pool):
                continue
            covered = any(
                numbers(f" {r['number']} ")[:1]
                and numbers(f" {r['number']} ")[0][1:3] == (value, decimals)
                and r["context"].lower() in line.lower()
                for r in rows
            )
            if not covered:
                problems.append(f"README.md:{n}: {written}  | {line.strip()[:90]}")
    return problems


def word_count(path: Path = README) -> int:
    return sum(len(ln.split()) for _, ln in prose_lines(path.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", action="store_true", help="print the prose word count")
    args = ap.parse_args()
    if args.words:
        print(word_count())
        return
    problems = audit()
    print("\n".join(problems) or "every number in the README's prose traces to a file")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
