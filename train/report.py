"""Training report: results/train_runs.md, results/curves/cpt.png, cpt_ppl.png (Stage 2) and
sft.png, sft_kpi.png (Stage 3), plus the README's generated blocks.

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

Stage 3 (runs whose train_summary.json says stage "sft"; notes/decisions.md, Stage 3b
pre-registration): the run table with B4's epoch-end val losses and the epoch the rule picks;
sft.png (train loss and the token-mean sft_val loss, epoch boundary marked); sft_kpi.png (seen /
unseen gold_lp and qa_acc for the starts, the SFT runs and the instruct bar); the change against
each run's start next to the noise, max(seed gap, SE); B7's first line, the paired bootstrap of
unseen gold_lp, sft-from-cpt - sft-from-base; and the merge checks, no-op controls, </s> and
diversity results. They go into the README's <!-- stage3-tables --> block.

Stage 4 (runs whose train_summary.json says stage "dpo"; notes/decisions.md, Stage 4
pre-registration): the run table with the final dpo_val loss, reward accuracy and margin and the
checkpoint the rule picked (results/runs/<run>/b4.json); dpo.png (DPO loss, reward margin, reward
accuracy, train and dpo_val, and the chosen / rejected sequence log-probs); the win rate against
sft-from-cpt when eval/winrate.py has written it; and the merge checks, no-op control and
diversity. They go into the README's <!-- stage4-tables --> block.
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
SFT_COLORS = {
    "sft-from-cpt": "#2a78d6",
    "sft-from-base": "#eb6834",
    "sft-from-cpt-seed1": "#2a78d6",
    "sft-from-base-seed1": "#eb6834",
    "sft-from-cpt-lr2e-4": "#1baf7a",
}
DPO_COLORS = {
    "dpo": "#2a78d6",
    "dpo-seed1": "#2a78d6",
    "dpo-rpo": "#eb6834",
    "dpo-lnorm": "#1baf7a",
    "dpo-lr5e-6": "#eda100",
}
DPO_START = "sft-from-cpt"  # stage3-final: every DPO run's start and reference
# another seed of a config
TWIN = {"cpt-8b-seed1", "sft-from-cpt-seed1", "sft-from-base-seed1", "dpo-seed1"}
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
            "false_abstain",
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
    throughput at 1, 8 and 32. base-8b runs Mistral's native vLLM path, the rest the HF path.
    Stop rate (requests ended by </s> before the 256-token cap) and mean output tokens at
    concurrency 1 exist from Stage 3's bench on; earlier rows leave them blank."""
    rows = []
    for r in ["base-8b", "instruct-8b", "base-8b-hf", "cpt-8b", *SFT_COLORS]:
        f = Path("results/bench") / f"{r}.json"
        if not f.exists():
            continue
        d = {x["concurrency"]: x for x in json.loads(f.read_text())}
        stop, out_tok = d[1].get("stop_rate"), d[1].get("mean_out_tokens")
        rows.append(
            [
                r,
                f"{d[1]['ttft_p50_ms']:.1f}",
                f"{d[1]['itl_p50_ms']:.1f}",
                f"{d[1]['e2e_p50_ms']:,.0f}",
                "" if stop is None else f"{stop:.0%}",
                "" if out_tok is None else f"{out_tok:.0f}",
                *(f"{d[c]['tok_per_s']:,.0f}" for c in (1, 8, 32)),
            ]
        )
    head = [
        "run",
        "TTFT p50 (ms)",
        "ITL p50 (ms)",
        "E2E p50 (ms)",
        "stop before cap",
        "mean output tokens",
        "tok/s @1",
        "tok/s @8",
        "tok/s @32",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


# ------------------------------------------------------------------------------------- Stage 3 ---
SFT_SEED_PAIR = ("sft-from-cpt", "sft-from-cpt-seed1")
SFT_ROWS = [  # (label, metrics.json key, kind, half / items key, or lm-eval task + stderr key)
    ("unseen gold-answer log-prob (nats)", "gold_lp_unseen", "lp", "unseen"),
    ("seen gold-answer log-prob (nats)", "gold_lp_seen", "lp", "seen"),
    ("qa_unseen", "qa_unseen", "kpi", "qa_unseen"),
    ("qa_seen", "qa_seen", "kpi", "qa_seen"),
    ("qa_ident (identifiers)", "qa_ident", "kpi", "qa_identifier"),
    ("grounded_acc", "grounded_acc", "kpi", "grounded"),
    ("cite_supported", "cite_supported", "kpi", "grounded"),
    ("halluc_rate (lower is better)", "halluc_rate", "kpi", "adversarial"),
    ("false_abstain (lower is better)", "false_abstain", "kpi", "grounded"),
    ("vocab_seen", "vocab_seen", "kpi", "vocab_seen"),
    ("vocab_unseen", "vocab_unseen", "kpi", "vocab_unseen"),
    ("MMLU", "mmlu", "lm", ("mmlu", "acc_stderr,none")),
    ("GSM8K", "gsm8k", "lm", ("gsm8k", "exact_match_stderr,strict-match")),
    ("HellaSwag", "hellaswag", "lm", ("hellaswag", "acc_norm_stderr,none")),
]
SFT_COLUMNS = [
    "instruct-8b",
    "base-8b-hf",
    "sft-from-base",
    "sft-from-base-seed1",
    "cpt-8b-replay10",
    "sft-from-cpt",
    "sft-from-cpt-seed1",
    "sft-from-cpt-lr2e-4",
]


B4_FORMATS = ("closed_book", "definition")


def epoch_rule(summary: dict) -> int | None:
    """B4 (amended 2026-10-06, before training): epoch 2 unless the closed-book or the definition
    sft_val loss rose from the end of epoch 1 to the end of epoch 2, then epoch 1. Overfitting
    shows in the recall formats first, and the mixture's mean (83% replay tokens) would hide it."""
    v = summary.get("val_loss_by_format_epoch_end") or {}
    if "1" not in v or "2" not in v:
        return None
    return 1 if any(v["2"][f] > v["1"][f] for f in B4_FORMATS) else 2


def epochs_pair(by_epoch: dict) -> str:
    """'epoch-1 value / epoch-2 value' for whichever epoch ends were logged."""
    return " / ".join(f"{by_epoch[e]:.4f}" for e in ("1", "2") if e in by_epoch)


def sft_table(runs: dict, usd: float) -> str:
    head = [
        "run",
        "start",
        "steps",
        "tokens trained",
        "tokens/s",
        "wall (h)",
        "GPU-h",
        "$",
        "peak GB",
        "final train loss",
        "val_loss epoch 1 / 2",
        "closed-book 1 / 2",
        "definition 1 / 2",
        "B4 picks",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, _) in runs.items():
        v = s.get("val_loss_epoch_end") or {}
        vf = s.get("val_loss_by_format_epoch_end") or {}
        epoch = epoch_rule(s)

        cells = [
            name,
            Path(s["base"]).name,
            s["steps"],
            f"{s['tokens'] * s['steps'] / max(1, s['steps_per_epoch']) / 1e6:.2f}M",
            f"{s['tokens_per_s']:,.0f}" if s.get("tokens_per_s") else "",
            f"{s['wall_s'] / 3600:.2f}",
            f"{s['gpu_hours']:.2f}",
            f"{s['gpu_hours'] * usd:.2f}",
            f"{s['peak_mem_gb']:.0f}" if s.get("peak_mem_gb") else "",
            f"{s['final_train_loss']:.3f}" if s.get("final_train_loss") else "",
            epochs_pair(v),
            epochs_pair({e: x["closed_book"] for e, x in vf.items()}),
            epochs_pair({e: x["definition"] for e, x in vf.items()}),
            f"epoch {epoch}" if epoch else "",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def val_curve(log: list[dict], key: str = "val_loss") -> list[tuple[int, float]]:
    """(step, value of `key`) at each evaluation, the last value if a step repeats: SFT's
    token-mean sft_val loss by default, TRL's eval_* metrics for DPO."""
    return sorted({r["step"]: r[key] for r in log if key in r}.items())


def plot_sft_loss(runs: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    last_step = max(curve(log)[0][-1] for _, log in runs.values())
    t_labels, v_labels = [], []
    for name, (_, log) in runs.items():
        color, ls = SFT_COLORS.get(name, INK_2), "--" if name in TWIN else "-"
        x, y = curve(log)
        ys = smooth(y)
        ax_t.plot(x, ys, color=color, lw=2, ls=ls, label=name, solid_capstyle="round")
        t_labels.append((name, x[-1], ys[-1], False))
        if ev := val_curve(log):
            ex, ey = zip(*ev)
            ax_v.plot(ex, ey, marker="o", ls=ls, lw=2, ms=6, color=color, mec=SURFACE, mew=1.5)
            v_labels.append((name, ex[-1], ey[-1], False))
    epoch_step = next(iter(runs.values()))[0]["steps_per_epoch"]
    for ax in (ax_t, ax_v):
        ax.axvline(epoch_step, color=INK_2, lw=1, ls=":")
        ax.annotate(
            "end of epoch 1",
            (epoch_step, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    style(ax_t, f"Train loss ({SMOOTH}-step moving average)", "optimizer step", "nats per token")
    style(
        ax_v,
        "sft_val loss (token mean, 80 records; step 0 = the start)",
        "optimizer step",
        "nats per token",
    )
    for ax, labels in ((ax_t, t_labels), (ax_v, v_labels)):
        ax.set_xlim(0, last_step * 1.3)
        end_labels(ax, labels)
    fig.suptitle("Stage 3 SFT: loss curves", color=INK, fontsize=11, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def item_lp(run: str, half: str | None = None) -> dict[str, float]:
    """{domain_qa id: gold_lp} for a run, optionally one half (eval/tasks/sft_seen_chunks.txt)."""
    path = RUNS / run / "generations.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.open())
    lp = {x["id"]: x["gold_lp"] for x in rows if x["task"] == "domain_qa" and "gold_lp" in x}
    if half is None:
        return lp
    seen = set(Path("eval/tasks/sft_seen_chunks.txt").read_text().split())
    src = {}
    for line in Path("eval/tasks/domain_qa.jsonl").read_text().splitlines():
        r = json.loads(line)
        src[r["id"]] = r["source_chunk"]
    return {i: v for i, v in lp.items() if i in src and (src[i] in seen) == (half == "seen")}


def paired_lp(a: str, b: str, half: str, n_boot: int = 10_000) -> dict:
    """Mean per-item gold_lp difference b - a on one half, its paired SE and a 95% bootstrap
    interval over items (seeded): B7's first line with a = sft-from-base, b = sft-from-cpt."""
    x, y = item_lp(a, half), item_lp(b, half)
    ids = sorted(set(x) & set(y))
    if len(ids) < 2:
        return {}
    d = np.array([y[i] - x[i] for i in ids])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(d),
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "ci": (float(lo), float(hi)),
        "up": float((d > 0).mean()),
    }


def half_sizes() -> dict[str, int]:
    seen = set(Path("eval/tasks/sft_seen_chunks.txt").read_text().split())
    qa = [json.loads(line) for line in Path("eval/tasks/domain_qa.jsonl").read_text().splitlines()]
    vocab = [json.loads(line) for line in Path("eval/tasks/vocab.jsonl").read_text().splitlines()]
    return {
        "qa_seen": sum(r["source_chunk"] in seen for r in qa),
        "qa_unseen": sum(r["source_chunk"] not in seen for r in qa),
        "vocab_seen": sum(r["source_chunk"] in seen for r in vocab),
        "vocab_unseen": sum(r["source_chunk"] not in seen for r in vocab),
        "qa_identifier": sum(r.get("answer_kind") == "identifier" for r in qa),
    }


def lm_se(run: str, task: str, key: str) -> float:
    files = sorted((Path("results/lm_eval") / run).glob("**/results*.json"))
    res = json.loads(files[-1].read_text())["results"] if files else {}
    return res.get(task, {}).get(key, 0) * 100


def noise(m: dict, pair: tuple[str, str], key: str, kind: str, extra, sizes: dict) -> float | None:
    """max(|pair[0] - pair[1]|, pair[0]'s SE): binomial on its items, lm-eval's stderr, or for
    gold_lp the paired per-item SE of the seed twins' difference on that half. In points for
    rates, nats for gold_lp."""
    a, b = (m.get(r, {}).get(key) for r in pair)
    if a is None or b is None:
        return None
    if kind == "lp":
        return max(abs(a - b), paired_lp(pair[1], pair[0], extra).get("se", 0))
    if kind == "lm":
        return max(abs(a - b) * 100, lm_se(pair[0], *extra))
    n = sizes.get(extra) or m[pair[0]]["n"].get(extra, 0)
    return max(abs(a - b) * 100, math.sqrt(a * (1 - a) / n) * 100 if n else 0)


def sft_delta_table() -> str:
    """Absolute values per run, with the noise of Stage 3 (the sft-from-cpt seed twins) and of
    Stage 2 (cpt-8b vs cpt-8b-seed1) next to them, both max(seed gap, SE)."""
    m = {}
    for r in [*SFT_COLUMNS, "cpt-8b", "cpt-8b-seed1"]:
        f = RUNS / r / "metrics.json"
        if f.exists():
            m[r] = json.loads(f.read_text())
    if not all(r in m for r in SFT_SEED_PAIR):
        return ""
    sizes = half_sizes()
    cols = [r for r in SFT_COLUMNS if r in m]
    head = ["metric", *cols, "noise (Stage 3)", "noise (Stage 2)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, kind, extra in SFT_ROWS:
        n3 = noise(m, SFT_SEED_PAIR, key, kind, extra, sizes)
        if n3 is None:
            continue
        n2 = noise(m, ("cpt-8b", "cpt-8b-seed1"), key, kind, extra, sizes)
        unit = "{:.3f}" if kind == "lp" else "{:.1f}"
        cells = [label] + ["" if m[r].get(key) is None else f"{m[r][key]:.3f}" for r in cols]
        cells += [unit.format(n3), "" if n2 is None else unit.format(n2)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_lp(runs: list[str], half: str) -> dict[str, float]:
    """Per-item gold_lp averaged over runs (items every run has)."""
    per = [item_lp(r, half) for r in runs]
    ids = set.intersection(*(set(x) for x in per))
    return {i: sum(x[i] for x in per) / len(per) for i in ids}


def bootstrap_diff(a: dict, b: dict, n_boot: int = 10_000) -> dict:
    """Mean per-item difference b - a, its paired SE and 95% bootstrap CI over items (seeded)."""
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    boot = d[np.random.default_rng(0).integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(d),
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "ci": (float(lo), float(hi)),
        "up": float((d > 0).mean()),
    }


def b7_first_line() -> str:
    """B7's first line: gold_lp by half, CPT arm - base arm, against the noise.

    One seed per arm (pre-registered 2026-10-06): sft-from-cpt - sft-from-base, noise = max(the
    CPT arm's seed gap, the paired per-item SE). With a seed twin on both arms (2026-10-06,
    fixed before sft-from-base-seed1's results): the difference of the arm means, noise =
    max(its run-variance SD, sqrt(gap_cpt^2 + gap_base^2) / 2, and the paired per-item SE of the
    averaged runs); beyond the noise needs the item-bootstrap CI to exclude 0 and the difference
    to exceed the noise."""
    m = {}
    for r in SFT_COLORS:
        f = RUNS / r / "metrics.json"
        if f.exists():
            m[r] = json.loads(f.read_text())
    if not all(r in m for r in ("sft-from-cpt", "sft-from-base")):
        return ""
    two_arm = all(r in m for r in ("sft-from-cpt-seed1", "sft-from-base-seed1"))
    out = []
    for half in ("unseen", "seen"):
        key = f"gold_lp_{half}"
        if two_arm:
            cpt = ["sft-from-cpt", "sft-from-cpt-seed1"]
            base = ["sft-from-base", "sft-from-base-seed1"]
            c = bootstrap_diff(mean_lp(base, half), mean_lp(cpt, half))
            gap_c = abs(m[cpt[0]][key] - m[cpt[1]][key])
            gap_b = abs(m[base[0]][key] - m[base[1]][key])
            run_sd = math.sqrt(gap_c**2 + gap_b**2) / 2
            floor = max(run_sd, c["se"])
            label = "mean of 2 CPT-arm runs - mean of 2 base-arm runs"
            pairs = [m[x][key] - m[y][key] for x in cpt for y in base]
            # one-sided: the chance that a random split of the runs puts every CPT-arm run above
            # (or below) every base-arm run is 1 / C(n_cpt + n_base, n_cpt)
            side = (
                "above"
                if all(d > 0 for d in pairs)
                else "below"
                if all(d < 0 for d in pairs)
                else ""
            )
            odds = math.comb(len(cpt) + len(base), len(cpt))
            detail = (
                f"run-variance SD {run_sd:.3f} from seed gaps {gap_c:.3f} (CPT arm) and "
                f"{gap_b:.3f} (base arm), paired SE {c['se']:.3f}; the difference is "
                f"{c['mean'] / run_sd:.1f} run SD, indicative only: each arm's SD rests on one "
                f"seed pair (1 df). Single-run pairs {', '.join(f'{d:+.3f}' for d in pairs)}"
                + (
                    f"; every CPT-arm run {side} every base-arm run, an ordering with exact "
                    f"one-sided permutation probability 1 in {odds}"
                    if side
                    else ""
                )
            )
        else:
            c = paired_lp("sft-from-base", "sft-from-cpt", half)
            seed = (
                abs(m["sft-from-cpt"][key] - m["sft-from-cpt-seed1"][key])
                if "sft-from-cpt-seed1" in m
                else None
            )
            floor = max(x for x in (seed, c["se"]) if x is not None)
            label = "sft-from-cpt - sft-from-base"
            seed_txt = "" if seed is None else f"seed gap {seed:.3f}, CPT arm only; "
            detail = f"{seed_txt}paired SE {c['se']:.3f}"
        if not c:
            continue
        lo, hi = c["ci"]
        verdict = (
            "beyond the noise: CPT bought something that survives SFT"
            if (lo > 0 or hi < 0) and abs(c["mean"]) > floor
            else "inside the noise: CPT's value is not distinguishable at this scale"
        )
        out.append(
            f"- **{half} gold_lp, {label}:** {c['mean']:+.3f} nats per answer [95% CI over items "
            f"{lo:+.3f}, {hi:+.3f}; {c['n']} items, {c['up']:.0%} up]; noise {floor:.3f} "
            f"({detail}): {verdict}. The item CI conditions on these training runs; run variance "
            "enters only through the noise."
        )
    return "\n".join(out)


STAGE3_STARTS = ("cpt-8b-replay10", "base-8b-hf")
SFT_GATE = (
    "over all 11,351 sft_val completion positions against an fp32 reference, the argmax flips the "
    "merge adds over the unmerged bf16 model's own"
)


def checks_table(
    names=SFT_COLORS,
    starts: tuple[str, ...] = STAGE3_STARTS,
    gate: str = SFT_GATE,
    diversity: tuple[str, ...] = ("instruct-8b",),
) -> str:
    """Merge checks (B5), no-op controls of the stage's starts, </s> on sampled answers and
    diversity, per run (Stage 3's by default; stage4_md passes its own)."""
    out = []
    noop = sorted(Path("results/noop").glob("*.json")) if Path("results/noop").exists() else []
    noop = [f for f in noop if f.stem in starts]
    if noop:
        out.append("No-op control (an untrained adapter, merged, against its start):\n")
        out.append("| start | tensors | differ | max abs diff | passed |\n|---|---|---|---|---|")
        for f in noop:
            r = json.loads(f.read_text())
            out.append(
                f"| {f.stem} | {r['tensors']} | {r['differ']} | {r['max_abs_diff']} | "
                f"{'yes' if r['passed'] else 'NO'} |"
            )
    rows = []
    for name in names:
        f = RUNS / name / "merge_check.json"
        if f.exists():
            r = json.loads(f.read_text())
            if "added_flips" not in r:  # the first gate's format; superseded
                continue
            p = r.get("probes") or {}
            sha = (r.get("checkpoint_sha256") or {}).get("sha256", "")
            rows.append(
                f"| {name} | {r['added_flips']} of {r['added_flips_allowed']} | "
                f"{r['lp_error_ratio']:.3f} | {r['val_loss_rel_diff']:.2%} | "
                f"{r['merged_vs_unmerged_top1']:.2%} | {p.get('ratio_mean', '')} | `{sha[:12]}` | "
                f"{'yes' if r['passed'] else 'NO'} |"
            )
    if rows:
        out.append(
            f"\nMerge gate (B5, amended): {gate}, "
            "and its mean |delta log-prob| relative to the unmerged model's (max 1.5); val loss "
            "within 0.5%. Merged-vs-unmerged agreement and the 3-probe mean merge-error ratio are "
            "reported, not gated; sha256 of the merged checkpoint's file list:\n"
        )
        out.append(
            "| run | flips added | \\|dlp\\| ratio | val loss diff | merged vs unmerged top-1 | "
            "probe error ratio | checkpoint sha256 | passed |\n|" + "---|" * 8
        )
        out += rows
    rows = []
    for name in [*diversity, *names]:
        f = Path("results/diversity") / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        d, g = r.get("diversity", {}), r.get("diversity_general", {})
        dm, e = r.get("diversity_domain", {}), r.get("eos", {})
        rows.append(
            f"| {name} | {d.get('distinct_4', '')} | {d.get('entropy', '')} | "
            f"{d.get('mean_len', '')} | {g.get('distinct_4', '')} | {dm.get('distinct_4', '')} | "
            f"{d.get('stop_rate', '')} | "
            + (f"{e['eos_stop_rate']:.1%} of {e['samples']}" if e else "")
            + " |"
        )
    if rows:
        out.append(
            "\nDiversity (100 prompts at T 0.7: 50 general, 50 domain; distinct-4 and entropy over "
            "output tokens) and </s> on sampled answers (20 Stage 4 prompts x 4 at T 0.8):\n"
        )
        out.append(
            "| run | distinct-4 | entropy (bits) | mean length | distinct-4 general | "
            "distinct-4 domain | stopped (T 0.7) | stopped (eos job) |\n|" + "---|" * 8
        )
        out += rows
    return "\n".join(out)


CHAT_ROWS = (
    "instruct-8b",
    "sft-from-base",
    "sft-from-base-seed1",
    "sft-from-cpt",
    "sft-from-cpt-seed1",
    "sft-from-cpt-lr2e-4",
)


def plot_sft_kpi(out: Path) -> None:
    """Seen / unseen gold_lp (chat-format runs only: a base-format gold_lp isn't comparable) and
    qa_acc (every row) for the starts, the SFT runs and the instruct bar."""
    names = [r for r in SFT_COLUMNS if (RUNS / r / "metrics.json").exists()]
    if not any(r in names for r in SFT_COLORS):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    m = {r: json.loads((RUNS / r / "metrics.json").read_text()) for r in names}
    colors = {
        **SFT_COLORS,
        "instruct-8b": "#52514e",
        "base-8b-hf": "#c9c7c1",
        "cpt-8b-replay10": "#9fc3ef",
    }
    sizes = half_sizes()
    halves = (f"seen half ({sizes['qa_seen']})", f"unseen half ({sizes['qa_unseen']})")
    fig, (ax_lp, ax_qa) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    chat = [r for r in names if r in CHAT_ROWS]
    step = 0.7 / max(1, len(chat) - 1)
    for j, r in enumerate(chat):
        for i, k in enumerate(("gold_lp_seen", "gold_lp_unseen")):
            v = m[r].get(k)
            if v is None:
                continue
            x = i - 0.35 + step * j
            ax_lp.plot(
                x,
                v,
                "o",
                ms=9,
                color=colors.get(r, INK_2),
                mec=SURFACE,
                mew=1.5,
                fillstyle="left" if r in TWIN else "full",
            )
            ax_lp.annotate(
                f"{v:.2f}",
                (x, v),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_2,
            )
    ax_lp.set_xlim(-0.6, 1.6)
    ax_lp.set_xticks(range(2), halves)
    style(
        ax_lp, "Gold-answer log-prob, chat format (nats per answer, higher is better)", "", "nats"
    )

    width = 0.8 / len(names)
    for j, r in enumerate(names):
        vals = [m[r].get(k) for k in ("qa_seen", "qa_unseen")]
        xs = [i - 0.4 + width * (j + 0.5) for i in range(2)]
        err = [
            math.sqrt(v * (1 - v) / sizes[k]) * 100 if v is not None else 0
            for v, k in zip(vals, ("qa_seen", "qa_unseen"))
        ]
        ax_qa.bar(
            xs,
            [(v or 0) * 100 for v in vals],
            width,
            yerr=err,
            color=colors.get(r, INK_2),
            edgecolor=SURFACE,
            lw=1.5,
            hatch="//" if r in TWIN else None,
            error_kw={"ecolor": INK_2, "lw": 1, "capsize": 2},
        )
    ax_qa.set_xticks(range(2), halves)
    style(ax_qa, "Closed-book qa_acc (%, +-1 binomial SE)", "", "%")
    handles = [
        Patch(
            facecolor=colors.get(r, INK_2),
            hatch="//" if r in TWIN else None,
            edgecolor=SURFACE,
            label=r,
        )
        for r in names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(names),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        "Stage 3: closed-book knowledge by half (eval v3)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def stage3_md(runs: dict, usd: float) -> str:
    md = "## Training runs\n\n" + sft_table(runs, usd)
    md += (
        "\n\nB4 (pre-registered, amended before training): epoch 2 unless the closed-book or the "
        "definition sft_val loss (token mean) rose from epoch 1 to epoch 2. The overall val_loss "
        "is 83% replay tokens, so it is shown, not used. $ at "
        f"{usd} per GPU-hour (assumed).\n"
    )
    if deltas := sft_delta_table():
        md += (
            "\n## Results next to the noise\n\n" + deltas + "\n\nValues as fractions (gold_lp "
            "in nats per answer); noise in points (gold_lp in nats). Noise = max(the seed gap, the "
            "SE): Stage 3 for sft-from-cpt vs sft-from-cpt-seed1, Stage 2 for cpt-8b vs "
            "cpt-8b-seed1. The SE is binomial on the metric's items, lm-eval's stderr, or for "
            "gold_lp the paired per-item SE of the twins on that half. Starts: sft-from-cpt from "
            "cpt-8b-replay10, sft-from-base from base-8b-hf; instruct-8b is the bar.\n"
        )
    if first := b7_first_line():
        md += "\n## B7's first line: what CPT bought, measured after SFT\n\n" + first + "\n"
    if checks := checks_table():
        md += "\n## Checks\n\n" + checks + "\n"
    return md


def rule_pick(name: str) -> str:
    """The checkpoint Stage 4's rule picked, from results/runs/<run>/b4.json (written by the
    pipeline before the merge)."""
    f = RUNS / name / "b4.json"
    if not f.exists():
        return ""
    r = json.loads(f.read_text())
    if r.get("ref_step") is None:
        return f"final (step {r['steps']}; no earlier save)"
    vs = f"{r['eval_loss_end']:.4f} vs {r['eval_loss_ref']:.4f} at {r['ref_step']}"
    return f"step {r['picked_step']}" + (" (final)" if not r["checkpoint"] else "") + f": {vs}"


def dpo_table(runs: dict, usd: float) -> str:
    head = [
        "run",
        "start",
        "pairs",
        "steps",
        "tokens/s",
        "wall (h)",
        "GPU-h",
        "$",
        "peak GB",
        "final train loss",
        "dpo_val loss",
        "dpo_val reward accuracy",
        "dpo_val margin",
        "rule picks",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, _) in runs.items():
        e = s.get("final_eval") or {}

        def f(key: str, fmt: str = ".3f", e=e) -> str:
            return format(e[key], fmt) if key in e else ""

        cells = [
            name,
            Path(s["base"]).name,
            s["pairs"],
            s["steps"],
            f"{s['tokens_per_s']:,.0f}" if s.get("tokens_per_s") else "",
            f"{s['wall_s'] / 3600:.2f}",
            f"{s['gpu_hours']:.2f}",
            f"{s['gpu_hours'] * usd:.2f}",
            f"{s['peak_mem_gb']:.0f}" if s.get("peak_mem_gb") else "",
            f"{s['final_train_loss']:.3f}" if s.get("final_train_loss") else "",
            f("eval_loss"),
            f("eval_rewards/accuracies"),
            f("eval_rewards/margins"),
            rule_pick(name),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def winrate_table(names) -> str:
    """Win rate against the start on the dpo_judge prompts, when eval/winrate.py has written
    results/winrate/<run>_vs_sft-from-cpt.json. TODO(stage 4): fix the columns to winrate.py's
    output once it exists; until then whichever of these keys it has."""
    keys = ("win_rate", "wins", "ties", "losses", "n", "position_consistency")
    rows = []
    for name in names:
        f = Path("results/winrate") / f"{name}_vs_{DPO_START}.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append([name, *(r.get(k, "") for k in keys)])
    if not rows:
        return ""
    lines = ["| run | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def plot_dpo(runs: dict, out: Path) -> None:
    """Four panels: DPO loss, reward margin and reward accuracy (train as a moving average, dpo_val
    as points), and the chosen / rejected sequence log-probs (train)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0, 0], "loss", "eval_loss", "DPO loss (0.693 = no preference)", "loss"),
        (axes[0, 1], "rewards/margins", "eval_rewards/margins", "Reward margin", "beta x nats"),
        (axes[1, 0], "rewards/accuracies", "eval_rewards/accuracies", "Reward accuracy", "share"),
    ]
    last_step = max(curve(log)[0][-1] for _, log in runs.values())
    for ax, key, ekey, title, ylabel in panels:
        labels = []
        for name, (_, log) in runs.items():
            color, ls = DPO_COLORS.get(name, INK_2), "--" if name in TWIN else "-"
            pts = val_curve(log, key)
            if pts:
                x, y = np.array([p[0] for p in pts]), smooth(np.array([p[1] for p in pts]))
                ax.plot(x, y, color=color, lw=2, ls=ls, solid_capstyle="round")
                labels.append((name, x[-1], y[-1], False))
            if ev := val_curve(log, ekey):
                ex, ey = zip(*ev)
                ax.plot(ex, ey, marker="o", ls="none", ms=6, color=color, mec=SURFACE, mew=1.5)
        style(ax, title, "", ylabel)
        ax.set_xlim(0, last_step * 1.3)
        end_labels(ax, labels)
    ax = axes[1, 1]
    labels = []
    for name, (s, log) in runs.items():
        if s.get("length_norm"):  # per-token means: another scale
            continue
        color = DPO_COLORS.get(name, INK_2)
        for key, ls, tag in (("logps/chosen", "-", "chosen"), ("logps/rejected", ":", "rejected")):
            if pts := val_curve(log, key):
                x, y = np.array([p[0] for p in pts]), smooth(np.array([p[1] for p in pts]))
                ax.plot(x, y, color=color, lw=2, ls=ls)
                labels.append((f"{name} {tag}", x[-1], y[-1], False))
    style(ax, "Sequence log-prob (chosen solid, rejected dotted)", "optimizer step", "nats")
    ax.set_xlim(0, last_step * 1.3)
    end_labels(ax, labels)
    for a in axes[1]:
        a.set_xlabel("optimizer step", color=INK_2)
    fig.suptitle(
        f"Stage 4 DPO: train ({SMOOTH}-step moving average, lines) and dpo_val (points)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def stage4_md(runs: dict, usd: float) -> str:
    md = "## Training runs\n\n" + dpo_table(runs, usd)
    md += (
        "\n\nThe checkpoint rule (pre-registered): the final step unless the dpo_val loss at the "
        "end is above its value at step 50 (runs under 100 steps: the save nearest the midpoint). "
        f"dpo_val values at the last evaluation. $ at {usd} per GPU-hour (assumed).\n"
    )
    if wr := winrate_table(runs):
        md += f"\n## Win rate against {DPO_START}\n\n" + wr + "\n"
    # TODO(stage 4): the KPI change against sft-from-cpt next to max(seed gap, SE), as
    # sft_delta_table does for Stage 3, once the dpo rows are scored.
    checks = checks_table(
        names=list(runs),
        starts=(DPO_START,),
        gate="over every dpo_val completion position (chosen and rejected) against an fp32 "
        "reference, the argmax flips the merge adds over the unmerged bf16 model's own (at most "
        "0.1% of positions)",
        diversity=(DPO_START,),
    )
    if checks:
        md += "\n## Checks\n\n" + checks + "\n"
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-gpu-hour", type=float, default=3.95)
    ap.add_argument("--scaling-run", default="cpt-8b-fsdp2")
    ap.add_argument("--scaling-ref", default="cpt-8b")
    ap.add_argument("--readme", default="README.md", help="'' to leave the README alone")
    args = ap.parse_args()
    every = {d.name: r for d in sorted(RUNS.iterdir()) if d.is_dir() and (r := load(d))}
    if not every:
        raise SystemExit(f"no train_summary.json + train_log.jsonl under {RUNS}")
    runs = {k: v for k, v in every.items() if v[0].get("stage") not in ("sft", "dpo")}
    sft = {k: v for k, v in every.items() if v[0].get("stage") == "sft"}
    sft_runs = {r: sft[r] for r in [*[r for r in SFT_COLORS if r in sft], *sorted(sft)]}
    dpo = {k: v for k, v in every.items() if v[0].get("stage") == "dpo"}
    dpo_runs = {r: dpo[r] for r in [*[r for r in DPO_COLORS if r in dpo], *sorted(dpo)]}
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
            "a result. QA rows are on the 322-item domain_qa (eval v3; Stage 2 was first read on "
            "v2's 325, results/table_v2.md), so cpt-8b-full, whose weights were deleted, has "
            "none.\n"
        )
    md3 = stage3_md(sft_runs, args.usd_per_gpu_hour) if sft_runs else ""
    md4 = stage4_md(dpo_runs, args.usd_per_gpu_hour) if dpo_runs else ""
    Path("results/train_runs.md").write_text(
        "# Stage 2: CPT\n\n"
        + md
        + ("\n# Stage 3: SFT\n\n" + md3 if md3 else "")
        + ("\n# Stage 4: DPO\n\n" + md4 if md4 else "")
    )
    print(md + md3 + md4)
    if args.readme:
        # the README nests these under "#### Stage N": their own headings go two levels down
        update_readme(Path(args.readme), "stage2-tables", re.sub(r"(?m)^## ", "##### ", md))
        if md3:
            update_readme(Path(args.readme), "stage3-tables", re.sub(r"(?m)^## ", "##### ", md3))
        if md4:
            update_readme(Path(args.readme), "stage4-tables", re.sub(r"(?m)^## ", "##### ", md4))
        update_readme(Path(args.readme), "results-table", results_table())
        update_readme(Path(args.readme), "serving-table", serving_table())
    base_val = math.log(ppl[BASE]["ppl_val_slice"]) if BASE in ppl else None
    plot_loss(runs, base_val, Path("results/curves/cpt.png"))
    if BASE in ppl and len(ppl) > 1:
        plot_ppl(ppl, Path("results/curves/cpt_ppl.png"))
    if sft_runs:
        plot_sft_loss(sft_runs, Path("results/curves/sft.png"))
        plot_sft_kpi(Path("results/curves/sft_kpi.png"))
    if dpo_runs:
        plot_dpo(dpo_runs, Path("results/curves/dpo.png"))


if __name__ == "__main__":
    main()
