"""Stage 2 training report: results/train_runs.md, results/curves/cpt.png and cpt_ppl.png.

  .venv/bin/python train/report.py [--usd-per-gpu-hour 3.95] [--scaling-run cpt-8b-fsdp2 --scaling-ref cpt-8b]

Reads results/runs/<run>/train_summary.json and train_log.jsonl for every run that has them
(pull each run with `modal volume get` first; smoke runs are skipped). The table has, per run:
GPUs, optimizer steps, tokens, steady-state tokens/s, wall time (including tokenising and model
load, i.e. what the GPUs are billed for), GPU-hours, dollars at --usd-per-gpu-hour (Modal's H100
list price: check modal.com/pricing and pass the current one), final train loss (mean of the last
10 steps) and final eval loss.

For the scaling run (ablation C: the LoRA config on 2 GPUs) against its 1-GPU reference: the
throughput ratio, and how closely the losses match, both per step and as the mean relative gap of
their 10-step moving averages from step 10 on. The global batch order doesn't depend on the number
of ranks (the same seeded permutation, split across them), so both runs see the same 32 windows at
every step and the per-step losses should agree to bf16 noise.

Figures (for the README): cpt.png has two panels, train loss (moving average) and val loss (the
trainer's evals on the 49 val windows, starting from base-8b's loss on the same windows at step 0);
cpt_ppl.png has each merged run's perplexity change vs base-8b on the train slice, domain val and
general val (results/ppl/<run>.json from eval/perplexity.py), with the pre-registered thresholds.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from ppl_compare import compare

RUNS = Path("results/runs")
PPL = Path("results/ppl")
BASE = "base-8b"
# Fixed colour per run (the entity, never its rank): categorical slots 1-5 of the reference
# palette, light mode. Yellow and magenta sit below 3:1 on the surface, so every line is
# direct-labelled and every bar carries its value.
COLORS = {
    "cpt-8b": "#2a78d6",
    "cpt-8b-replay10": "#eb6834",
    "cpt-8b-full": "#1baf7a",
    "cpt-8b-fsdp2": "#eda100",
    "cpt-8b-lr2x": "#e87ba4",
    "cpt-8b-seed1": "#2a78d6",  # the main config again: its colour, drawn dashed / hatched
}
TWIN = {"cpt-8b-seed1"}  # same config as another run, another seed
PPL_METRICS = [
    ("ppl_train", "train slice\n(seen once by CPT)"),
    ("ppl_domain_val", "domain val\n(held-out documents)"),
    ("ppl_general_val", "general val\n(FineWeb-Edu)"),
]
THRESHOLDS = {
    "ppl_domain_val": (-20.0, "pre-registered: -20%"),
    "ppl_general_val": (3.0, "bound: +3%"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SMOOTH = 10


def load(run_dir: Path) -> tuple[dict, list[dict]] | None:
    s, log = run_dir / "train_summary.json", run_dir / "train_log.jsonl"
    if not (s.exists() and log.exists()):
        return None
    summary = json.loads(s.read_text())
    if summary.get("smoke"):
        return None
    return summary, [json.loads(line) for line in log.open()]


def curve(log: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(steps, train loss) with one point per step: the last logged value if a step was logged
    twice (a resumed run replays nothing, but a crash between log and save can)."""
    by_step = {r["step"]: r["loss"] for r in log if "loss" in r}
    steps = np.array(sorted(by_step))
    return steps, np.array([by_step[s] for s in steps])


def smooth(y: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """Trailing moving average; the first k-1 points average what exists so far."""
    c = np.cumsum(np.insert(y, 0, 0.0))
    n = np.minimum(np.arange(1, len(y) + 1), k)
    return (c[1:] - c[np.arange(len(y)) + 1 - n]) / n


def table(runs: dict, usd: float) -> str:
    head = [
        "run",
        "GPUs",
        "steps",
        "tokens",
        "tokens/s",
        "tokens/s/GPU",
        "wall (h)",
        "GPU-h",
        "$",
        "final train loss",
        "final eval loss",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, _) in runs.items():
        cells = [
            name,
            s["gpus"],
            s["steps"],
            f"{min(s['steps'] * s['seqs_per_step'], s.get('windows', 1e12)) * 4096 / 1e6:.1f}M",
            f"{s['tokens_per_s']:,.0f}" if s.get("tokens_per_s") else "",
            f"{s['tokens_per_s'] / s['gpus']:,.0f}" if s.get("tokens_per_s") else "",
            f"{s['wall_s'] / 3600:.2f}",
            f"{s['gpu_hours']:.2f}",
            f"{s['gpu_hours'] * usd:.2f}",
            f"{s['final_train_loss']:.3f}" if s.get("final_train_loss") else "",
            f"{s['final_eval_loss']:.3f}" if s.get("final_eval_loss") else "",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def scaling(runs: dict, name: str, ref: str) -> str:
    """Ablation C only: B also runs on 2 GPUs, but full-parameter steps cost more compute than
    LoRA steps, so its tokens/s is not a 1-vs-2-GPU number."""
    out = []
    if name in runs and ref in runs:
        s, log = runs[name]
        r_s, r_log = runs[ref]
        (xa, ya), (xb, yb) = curve(log), curve(r_log)
        both = np.intersect1d(xa, xb)
        both = both[both >= SMOOTH]
        sa = dict(zip(xa, smooth(ya)))
        sb = dict(zip(xb, smooth(yb)))
        gap = np.mean([abs(sa[x] - sb[x]) / sb[x] for x in both]) if len(both) else float("nan")
        ra, rb = dict(zip(xa, ya)), dict(zip(xb, yb))
        step_gap = [abs(ra[x] / rb[x] - 1) for x in np.intersect1d(xa, xb)]
        ratio = s["tokens_per_s"] / r_s["tokens_per_s"]
        out.append(
            f"- **{name} vs {ref}:** {s['gpus']} GPUs give {ratio:.2f}x the tokens/s "
            f"({s['tokens_per_s']:,.0f} vs {r_s['tokens_per_s']:,.0f}); per-step losses differ "
            f"by {np.median(step_gap):.3%} (median) / {max(step_gap):.3%} (max) over "
            f"{len(step_gap)} steps, the {SMOOTH}-step moving averages by {gap:.2%} on average."
        )
    return "\n".join(out)


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, axis="y" if not xlabel else "both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def end_labels(ax, items: list[tuple[str, float, float, bool]]) -> None:
    """Direct labels at the curves' last points, nudged apart vertically so neighbours that end
    close together don't overlap. A run that stops early (the 2-GPU run, at step 100) ends on
    top of the longer curves, so its label goes just below its last point instead."""
    (x0, x1), (lo, hi) = ax.get_xlim(), ax.get_ylim()
    gap = 0.045 * (hi - lo)
    right = sorted((y, x, name) for name, x, y, early in items if not early)
    # one column right of the longest curve, so no line runs under a label
    x_text = max((x for _, x, _ in right), default=x0) + 0.02 * (x1 - x0)
    placed: list[float] = []
    for y, x, name in right:
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(name, (x, y), xytext=(x_text, y_text), va="center", fontsize=9, color=INK)
    for name, x, y, early in items:
        if early:
            ax.annotate(
                name,
                (x, y),
                xytext=(4, -14),
                textcoords="offset points",
                va="top",
                fontsize=9,
                color=INK,
                # it sits over the longer curves that continue past it: back it with the surface
                bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.85},
            )


def plot_loss(runs: dict, base_val_loss: float | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    last_step = max(curve(log)[0][-1] for _, log in runs.values())
    t_labels, v_labels = [], []
    for name, (_, log) in runs.items():
        color = COLORS.get(name, INK_2)
        x, y = curve(log)
        ys = smooth(y)
        ls = "--" if name in TWIN else "-"
        ax_t.plot(x, ys, color=color, lw=2, ls=ls, label=name, solid_capstyle="round")
        t_labels.append((name, x[-1], ys[-1], x[-1] < 0.8 * last_step))
        ev = sorted((r["step"], r["eval_loss"]) for r in log if "eval_loss" in r)
        if ev:
            if base_val_loss is not None:  # step 0 is the base model: its loss on the same windows
                ev = [(0, base_val_loss), *ev]
            ex, ey = zip(*ev)
            ax_v.plot(
                ex, ey, marker="o", ls=ls, lw=2, ms=6, color=color, mec=SURFACE, mew=1.5, label=name
            )
            v_labels.append((name, ex[-1], ey[-1], False))
    if base_val_loss is not None:
        ax_v.axhline(base_val_loss, color=INK_2, lw=1, ls="--")
        ax_v.annotate(
            "base-8b",
            (0, base_val_loss),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    style(
        ax_t,
        f"Train loss ({SMOOTH}-step moving average)",
        "optimizer step",
        "loss (nats per token)",
    )
    style(
        ax_v,
        "Val loss (49 held-out windows, ~200k tokens)",
        "optimizer step",
        "loss (nats per token)",
    )
    for ax, labels in ((ax_t, t_labels), (ax_v, v_labels)):
        ax.set_xlim(0, last_step * 1.3)  # room on the right for the direct labels
        end_labels(ax, labels)
    ax_t.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Stage 2 CPT: loss curves", color=INK, fontsize=11, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def plot_ppl(ppl: dict, out: Path) -> None:
    """Perplexity change vs base-8b per merged run, grouped by slice, with the pre-registered
    thresholds (domain val down at least 20%, general val up less than 3%)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = ppl[BASE]
    names = [r for r in [*COLORS, *sorted(ppl)] if r in ppl and not r.startswith("base")]
    names = list(dict.fromkeys(names))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / max(1, len(names))
    for j, name in enumerate(names):
        vals = [(ppl[name][m] / base[m] - 1) * 100 for m, _ in PPL_METRICS]
        xs = [i - 0.4 + width * (j + 0.5) for i in range(len(PPL_METRICS))]
        ax.bar(
            xs,
            vals,
            width,
            color=COLORS.get(name, INK_2),
            edgecolor=SURFACE,
            lw=1.5,
            hatch="//" if name in TWIN else None,
            label=name,
        )
        for x, v in zip(xs, vals):
            ax.annotate(
                f"{v:+.1f}%",
                (x, v),
                xytext=(0, 3 if v >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                color=INK_2,
            )
    for i, (m, _) in enumerate(PPL_METRICS):
        if m in THRESHOLDS:
            y, text = THRESHOLDS[m]
            ax.plot([i - 0.45, i + 0.45], [y, y], color=INK, lw=1, ls="--")
            ax.annotate(
                text,
                (i - 0.45, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(range(len(PPL_METRICS)), [label for _, label in PPL_METRICS])
    style(ax, f"Perplexity change vs {BASE} (lower is better)", "", "change (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def ppl_table(ppl: dict) -> str:
    """Each run's perplexity against base-8b on the three slices: the value, the change in %
    and in nats per token, and for the two val slices the 95% paired bootstrap interval over
    documents (eval/ppl_compare.py; empty when a run's json predates the per-document sums)."""
    base = ppl[BASE]
    head = [
        "run",
        "domain val",
        "change",
        "nats [95% CI, docs]",
        "general val",
        "change",
        "nats [95% CI, docs]",
        "train slice",
        "change",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    names = dict.fromkeys(
        [BASE, *[r for r in [*COLORS, *sorted(ppl)] if r in ppl and not r.startswith("base")]]
    )  # base-* rows are references, not runs
    for name in names:
        r = ppl[name]
        cells = [name]
        for key, set_name in (
            ("ppl_domain_val", "domain_val"),
            ("ppl_general_val", "general_val"),
            ("ppl_train", "train_slice"),
        ):
            cells.append(f"{r[key]:.3f}")
            if name == BASE:
                cells += ["", ""] if set_name != "train_slice" else [""]
                continue
            cells.append(f"{(r[key] / base[key] - 1) * 100:+.2f}%")
            if set_name == "train_slice":
                continue
            c = compare(base, r, set_name, "documents") if "sums" in r and "sums" in base else {}
            lo, hi = c.get("nats_ci", (None, None))
            nats = math.log(r[key] / base[key])
            cells.append(f"{nats:+.4f}" + (f" [{lo:+.4f}, {hi:+.4f}]" if c else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(path: Path, name: str, block: str) -> None:
    """Replace the README's generated block `name` (between <!-- name:start --> and
    <!-- name:end -->) with `block`, so its tables are always the collected ones. The figures keep
    their paths (results/curves/), so the images the README embeds update in place."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    text = path.read_text()
    if start not in text or end not in text:
        print(f"{path}: no {start} ... {end} block; not updated")
        return
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    path.write_text(f"{head}{start}\n{block.strip()}\n{end}{tail}")
    print(f"-> {path} ({name})")


def results_table(path: Path = Path("results/table.md")) -> str:
    """results/table.md without its smoke-test rows: the README's copy of the results."""
    lines = path.read_text().splitlines()
    return "\n".join(line for line in lines if not line.startswith("| smoke "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-gpu-hour", type=float, default=3.95)
    ap.add_argument("--scaling-run", default="cpt-8b-fsdp2")
    ap.add_argument("--scaling-ref", default="cpt-8b")
    ap.add_argument("--readme", default="README.md", help="'' to leave the README alone")
    args = ap.parse_args()
    runs = {d.name: r for d in sorted(RUNS.iterdir()) if d.is_dir() and (r := load(d))}
    if not runs:
        raise SystemExit(f"no train_summary.json + train_log.jsonl under {RUNS}")
    order = [*[r for r in COLORS if r in runs], *[r for r in runs if r not in COLORS]]
    runs = {r: runs[r] for r in order}
    md = "## Training runs\n\n" + table(runs, args.usd_per_gpu_hour)
    if cmp := scaling(runs, args.scaling_run, args.scaling_ref):
        md += "\n\n" + cmp
    md += (
        f"\n\n$ at {args.usd_per_gpu_hour} per GPU-hour (Modal's H100 list price as assumed, not "
        "checked against modal.com/pricing); wall time includes tokenising and model load.\n"
    )
    ppl = {
        f.stem: json.loads(f.read_text())
        for f in sorted(PPL.glob("*.json"))
        if not f.stem.startswith("smoke")
    }
    if BASE in ppl and len(ppl) > 1:
        md += f"\n## Perplexity vs {BASE}\n\n" + ppl_table(ppl) + "\n"
    Path("results/train_runs.md").write_text(md)
    print(md)
    if args.readme:
        # the README nests these under "#### Stage 2": its own headings go two levels down
        update_readme(Path(args.readme), "stage2-tables", re.sub(r"(?m)^## ", "##### ", md))
        update_readme(Path(args.readme), "results-table", results_table())
    base_val = math.log(ppl[BASE]["ppl_val_slice"]) if BASE in ppl else None
    plot_loss(runs, base_val, Path("results/curves/cpt.png"))
    if BASE in ppl and len(ppl) > 1:
        plot_ppl(ppl, Path("results/curves/cpt_ppl.png"))


if __name__ == "__main__":
    main()
