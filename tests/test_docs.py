"""The write-up's files: README.md, docs/*.md and DEPLOY.md.

- Every generated block train/report.py writes sits exactly once in the file its BLOCK_FILES
  names (a block that moved without its writer would go stale), and no file carries an unknown one.
- Every relative link resolves to a file, and every #anchor to a heading in its target, by GitHub's
  slug rule (duplicate headings get -1, -2, ...)."""

import ast
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = [REPO / "README.md", REPO / "DEPLOY.md", *sorted((REPO / "docs").glob("*.md"))]
MARKER = re.compile(r"<!-- ([a-z0-9-]+):(start|end) -->")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def block_files() -> dict[str, str]:
    tree = ast.parse((REPO / "train/report.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BLOCK_FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("train/report.py has no BLOCK_FILES")


def prose(text: str) -> list[str]:
    """Lines outside code fences, inline code removed."""
    out, fence = [], False
    for ln in text.splitlines():
        if ln.startswith("```"):
            fence = not fence
        elif not fence:
            out.append(re.sub(r"`[^`]*`", "", ln))
    return out


def slugs(path: Path) -> set[str]:
    seen: Counter = Counter()
    out = set()
    fence = False
    for ln in path.read_text().splitlines():
        if ln.startswith("```"):
            fence = not fence
        if fence or not (m := re.match(r"^#+ (.*)", ln)):
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1))  # link text only
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        out.add(f"{slug}-{seen[slug]}" if seen[slug] else slug)
        seen[slug] += 1
    return out


def test_generated_blocks_live_where_report_writes_them():
    where = block_files()
    found: Counter = Counter()
    for f in FILES:
        for name, edge in MARKER.findall(f.read_text()):
            found[(name, edge, f.relative_to(REPO).as_posix())] += 1
    for name, path in where.items():
        for edge in ("start", "end"):
            assert found[(name, edge, path)] == 1, f"{path}: needs one <!-- {name}:{edge} -->"
    for (name, _, path), n in found.items():
        known = where.get(name) == path or (path == "DEPLOY.md" and name.startswith("deploy-"))
        assert known and n == 1, f"{path}: unexpected block marker {name} (x{n})"


def test_links_resolve():
    bad = []
    for f in FILES:
        for ln in prose(f.read_text()):
            for target in LINK.findall(ln):
                if re.match(r"(https?|mailto):", target):
                    continue
                file, _, anchor = target.partition("#")
                dest = (f.parent / file).resolve() if file else f
                if not dest.exists():
                    bad.append(f"{f.name}: {target} (no file)")
                elif anchor and dest.suffix == ".md" and anchor not in slugs(dest):
                    bad.append(f"{f.name}: {target} (no heading)")
    assert not bad, "\n".join(bad)


def test_headline_tables_share_rows_and_floors():
    """README part 3: the three tables list their runs in one order (HEADLINE_ROWS', with rows
    that have no number in a table left out), and each carries both format groups' floor rows."""
    text = (REPO / "README.md").read_text()
    order = None
    for name in ("headline-knowledge", "headline-behaviour", "headline-general"):
        block = text.split(f"<!-- {name}:start -->")[1].split(f"<!-- {name}:end -->")[0]
        runs = re.findall(r"^\| `([^`]+)` \|", block, re.MULTILINE)
        floors = re.findall(r"^\|  \| \*floor, ([a-z-]+) rows", block, re.MULTILINE)
        assert floors == ["base-format", "chat"], name
        order = order or runs
        assert [r for r in order if r in runs] == [r for r in runs if r in order], name
        assert runs[0] == "base-8b-hf", name
