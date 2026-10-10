"""Rescore every row of results/table.md from its committed generations and compare: the check a
reader can run with no GPU and no API key.

  .venv/bin/python eval/rescore_all.py     # -> results/_rescore/ (gitignored); exit 1 on a difference

Each run's generations.jsonl and metrics.json (for its model and chat flag) are copied into a
scratch results dir and scored there by `run_eval.py --rescore`, with the committed judge cache,
lm-eval outputs and perplexities, so results/ is never written. Every judged item must hit the
cache: a miss with no MISTRAL_API_KEY fails at once (judge.Judge builds its client lazily). The
scratch table must equal results/table.md row for row.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCRATCH = RESULTS / "_rescore"


def rows(table: Path) -> dict[str, list[str]]:
    lines = [ln for ln in table.read_text().splitlines() if ln.startswith("| ")]
    return {c[0]: c for c in ([x.strip() for x in ln.strip("|").split("|")] for ln in lines[1:])}


def main() -> None:
    committed = rows(RESULTS / "table.md")
    runs = [r for r in committed if r not in ("run", "smoke")]
    shutil.rmtree(SCRATCH, ignore_errors=True)
    for run in runs:
        src, dst = RESULTS / "runs" / run, SCRATCH / "runs" / run
        dst.mkdir(parents=True)
        for f in ("generations.jsonl", "metrics.json"):
            shutil.copy(src / f, dst / f)
        meta = json.loads((src / "metrics.json").read_text())
        cmd = [sys.executable, str(REPO / "eval/run_eval.py"), "--run-name", run, "--rescore",
               "--results-dir", str(SCRATCH), "--judge-cache", str(RESULTS / "judge_cache.jsonl"),
               "--lm-eval-dir", str(RESULTS / "lm_eval"), "--ppl-dir", str(RESULTS / "ppl"),
               "--allow-partial", "--model", meta["model"], *(["--chat"] if meta.get("chat") else [])]  # fmt: skip
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
        if out.returncode:
            raise SystemExit(f"{run}: run_eval failed\n{out.stderr[-2000:]}")
        if "judge calls this run: 0" not in out.stdout:
            raise SystemExit(f"{run}: the judge was called; the committed cache should cover it")
        print(f"rescored {run}")
    fresh = rows(SCRATCH / "table.md")
    diffs = [
        f"{run}: committed {committed[run]}\n{' ' * len(run)}  rescored  {fresh.get(run)}"
        for run in runs
        if fresh.get(run) != committed[run]
    ]
    head_ok = (SCRATCH / "table.md").read_text().splitlines()[0] == (
        RESULTS / "table.md"
    ).read_text().splitlines()[0]
    print("\n".join(diffs) or f"all {len(runs)} rows reproduce from the committed generations")
    if not head_ok:
        print("the task sizes line differs")
    raise SystemExit(1 if diffs or not head_ok else 0)


if __name__ == "__main__":
    main()
