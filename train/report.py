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
import statistics
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
        per_gpu = s["tokens_per_s"] / s["gpus"], r_s["tokens_per_s"] / r_s["gpus"]
        out.append(
            f"- **{name} vs {ref}:** {s['gpus']} GPUs give {ratio:.2f}x the tokens/s "
            f"({s['tokens_per_s']:,.0f} vs {r_s['tokens_per_s']:,.0f}), so per GPU "
            f"{per_gpu[0] / per_gpu[1] - 1:+.0%} at the same micro-batch and per-layer "
            f"checkpointing (peak memory {s['peak_mem_gb']:.0f} vs {r_s['peak_mem_gb']:.0f} GB): "
            f"the FSDP2 code path, not scaling (finding 6 below); per-step losses differ "
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


RESULT_BLOCKS = (
    (
        (
            "Closed-book knowledge: no retrieval, no passage in the prompt; questions about facts "
            "on specific pages of the manuals (gold_lp: nats per answer, higher is better)"
        ),
        [
            "gold_lp",
            "gold_lp_seen",
            "gold_lp_unseen",
            "qa_acc",
            "qa_num",
            "qa_ident",
            "qa_term",
            "qa_seen",
            "qa_unseen",
        ],
    ),
    (
        (
            "With the passages: grounded answers and citations (4 passages given), abstention "
            "when the passages lack the answer (halluc_rate, lower is better), and definitions"
        ),
        [
            "grounded_acc",
            "cite_valid",
            "cite_supported",
            "halluc_rate",
            "vocab_recall",
            "vocab_seen",
            "vocab_unseen",
        ],
    ),
    (
        "General benchmarks (5-shot, no chat template) and perplexity (lower is better)",
        [
            "mmlu",
            "mmlu_stem",
            "mmlu_hum",
            "mmlu_soc",
            "mmlu_other",
            "gsm8k",
            "hellaswag",
            "ppl_train",
            "ppl_domain_val",
            "ppl_general_val",
            "ppl_postcutoff",
        ],
    ),
)


def results_table(path: Path = Path("results/table.md")) -> str:
    """results/table.md without its smoke-test rows, as three tables: closed-book knowledge,
    work with the passages given, and general benchmarks with perplexity. One table of 26 columns
    doesn't fit a page, and a bare qa_acc next to grounded_acc invites reading 0.13 against 0.84
    as a broken score rather than closed-book vs open-book. The table's first line records the
    task sizes it was scored on."""
    lines = path.read_text().splitlines()
    tag, lines = lines[0], [ln for ln in lines[1:] if not ln.startswith("| smoke ")]
    sizes = tag.removeprefix("<!-- items: ").removesuffix(" -->").split()
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head = cells[0]
    out = [f"Items per task: {', '.join(s.replace('=', ' ') for s in sizes)}."]
    for title, names in RESULT_BLOCKS:
        idx = [head.index(n) for n in names if n in head]
        rows = [[row[0], *(row[i] for i in idx)] for row in cells]
        rows = [rows[0], rows[1]] + [r for r in rows[2:] if any(r[1:])]  # rows with numbers here
        if len(rows) == 2:
            continue
        out.append(f"**{title}**\n\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    return "\n\n".join(out)


REF = "base-8b-hf"  # Stage 2 rows are read against the base evaluated through the same vLLM path
SEED_PAIR = ("cpt-8b", "cpt-8b-seed1")
DELTA_RUNS = ["cpt-8b", "cpt-8b-seed1", "cpt-8b-replay10", "cpt-8b-full"]
DELTA_ROWS = [  # (label, metrics.json key, kind, lm-eval task and stderr key or KPI task)
    ("domain val perplexity", "ppl_domain_val", "ppl", None),
    ("2026-report perplexity", "ppl_postcutoff", "ppl", None),
    ("general val perplexity", "ppl_general_val", "ppl", None),
    ("train slice perplexity", "ppl_train", "ppl", None),
    ("MMLU", "mmlu", "lm", ("mmlu", "acc_stderr,none")),
    ("GSM8K", "gsm8k", "lm", ("gsm8k", "exact_match_stderr,strict-match")),
    ("HellaSwag", "hellaswag", "lm", ("hellaswag", "acc_norm_stderr,none")),
    ("closed-book gold-answer log-prob (nats)", "gold_lp", "lp", None),
    ("closed-book qa_acc", "qa_acc", "kpi", "domain_qa"),
    ("grounded_acc (with passages)", "grounded_acc", "kpi", "grounded"),
    ("vocab_recall", "vocab_recall", "kpi", "vocab"),
    ("halluc_rate", "halluc_rate", "kpi", "adversarial"),
]


def paired_lp_se(ref: str, run: str) -> float:
    """Standard error of the per-item gold_lp difference run - ref on the same domain_qa items:
    the noise a paired comparison of a continuous score has to beat."""

    def load(r: str) -> dict:
        rows = (json.loads(line) for line in (RUNS / r / "generations.jsonl").open())
        return {x["id"]: x["gold_lp"] for x in rows if x["task"] == "domain_qa" and "gold_lp" in x}

    a, b = load(ref), load(run)
    d = [b[i] - a[i] for i in a if i in b]
    return statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0


def delta_table() -> str:
    """Each Stage 2 run's change vs base-8b-hf next to the noise it has to beat: max(the seed gap
    between two runs of the same config, the metric's own standard error). Perplexity changes in
    %, the rest in points. Empty when the reference or the seed pair isn't scored yet."""
    runs = {}
    for r in [REF, *DELTA_RUNS]:
        f = RUNS / r / "metrics.json"
        if f.exists():
            runs[r] = json.loads(f.read_text())
    if REF not in runs or not all(r in runs for r in SEED_PAIR):
        return ""
    ref = runs[REF]
    lm_files = sorted((Path("results/lm_eval") / REF).glob("**/results*.json"))
    lm = json.loads(lm_files[-1].read_text())["results"] if lm_files else {}
    names = [r for r in DELTA_RUNS if r in runs]
    head = ["metric", "base-8b-hf", *names, "noise"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, kind, extra in DELTA_ROWS:
        if ref.get(key) is None:
            continue

        change = {}
        for r, metrics in runs.items():
            v = metrics.get(key)
            if v is not None:
                change[r] = (
                    (v / ref[key] - 1) * 100
                    if kind == "ppl"
                    else (v - ref[key])
                    if kind == "lp"
                    else (v - ref[key]) * 100
                )
        # the seed gap needs both seeds measured; a missing one is no gap, not a gap of 0
        pair = [change.get(r) for r in SEED_PAIR]
        seed = abs(pair[0] - pair[1]) if None not in pair else None
        if kind == "lp":
            se = paired_lp_se(REF, SEED_PAIR[0])
        elif kind == "lm" and extra[0] in lm:
            se = lm[extra[0]].get(extra[1], 0) * 100
        elif kind == "kpi":
            n = ref.get("n", {}).get(extra, 0)
            se = math.sqrt(ref[key] * (1 - ref[key]) / n) * 100 if n else 0
        else:
            se = 0
        base = f"{ref[key]:.2f}" if kind in ("ppl", "lp") else f"{ref[key]:.3f}"
        cells = [label, base]
        fmt = {"ppl": "{:+.2f}%", "lp": "{:+.2f}"}.get(kind, "{:+.1f}")
        for r in names:
            v = change.get(r)
            cells.append("" if v is None else fmt.format(v))
        noise = max(x for x in (seed, se) if x is not None) if (seed or se) else None
        cells.append("" if noise is None else fmt.format(noise).lstrip("+"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def serving_table() -> str:
    """results/bench/<run>.json (serve/bench_latency.py): p50 at 1 concurrent request, and
    throughput at 1, 8 and 32. base-8b runs Mistral's native vLLM path, the rest the HF path."""
    rows = []
    for r in ["base-8b", "instruct-8b", "base-8b-hf", "cpt-8b"]:
        f = Path("results/bench") / f"{r}.json"
        if not f.exists():
            continue
        d = {x["concurrency"]: x for x in json.loads(f.read_text())}
        rows.append(
            [
                r,
                f"{d[1]['ttft_p50_ms']:.1f}",
                f"{d[1]['itl_p50_ms']:.1f}",
                f"{d[1]['e2e_p50_ms']:,.0f}",
                *(f"{d[c]['tok_per_s']:,.0f}" for c in (1, 8, 32)),
            ]
        )
    head = [
        "run",
        "TTFT p50 (ms)",
        "ITL p50 (ms)",
        "E2E p50 (ms)",
        "tok/s @1",
        "tok/s @8",
        "tok/s @32",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


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
        if not f.stem.startswith("smoke") and not f.stem.endswith("-remerge")
    }
    if BASE in ppl and len(ppl) > 1:
        md += f"\n## Perplexity vs {BASE}\n\n" + ppl_table(ppl) + "\n"
    if deltas := delta_table():
        md += (
            f"\n## Change vs {REF}, next to the noise\n\n{deltas}\n\nPerplexity in %, the "
            "gold-answer log-probability in nats per answer, the rest in points. noise = max(the "
            "seed gap cpt-8b vs cpt-8b-seed1, the metric's standard error: for base-8b-hf, or for "
            "the log-probability the paired per-item difference): a change smaller than it is not "
            "a result. QA rows are on the 325-item domain_qa (eval v2), so cpt-8b-full, whose "
            "weights were deleted, has none.\n"
        )
    Path("results/train_runs.md").write_text(md)
    print(md)
    if args.readme:
        # the README nests these under "#### Stage 2": its own headings go two levels down
        update_readme(Path(args.readme), "stage2-tables", re.sub(r"(?m)^## ", "##### ", md))
        update_readme(Path(args.readme), "results-table", results_table())
        update_readme(Path(args.readme), "serving-table", serving_table())
    base_val = math.log(ppl[BASE]["ppl_val_slice"]) if BASE in ppl else None
    plot_loss(runs, base_val, Path("results/curves/cpt.png"))
    if BASE in ppl and len(ppl) > 1:
        plot_ppl(ppl, Path("results/curves/cpt_ppl.png"))


if __name__ == "__main__":
    main()
