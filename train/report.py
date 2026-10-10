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
import hashlib
import json
import math
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
GRPO_COLORS = {"grpo": "#2a78d6", "grpo-seed1": "#2a78d6"}
GRPO_START = "dpo-strict"  # stage4-final (2026-10-09): GRPO's start
# another seed of a config
TWIN = {"cpt-8b-seed1", "sft-from-cpt-seed1", "sft-from-base-seed1", "dpo-seed1", "grpo-seed1"}
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


# where each generated block lives (docs/stageN.md hold the stage write-ups; their "## " headings
# are the blocks' own). DEPLOY.md's blocks are s6_deploy_blocks().
BLOCK_FILES = {
    "stage2-tables": "docs/stage2.md",
    "stage3-tables": "docs/stage3.md",
    "stage4-tables": "docs/stage4.md",
    "stage5-tables": "docs/stage5.md",
    "stage6-tables": "docs/stage6.md",
    "serving-table": "docs/stage6.md",
    "results-table": "docs/results.md",
    "headline-summary": "README.md",
    "headline-knowledge": "README.md",
    "headline-behaviour": "README.md",
    "headline-general": "README.md",
    "reproduce-table": "README.md",
    "replay-licences": "README.md",
}


def update_block(path: Path, name: str, block: str) -> None:
    """Replace the generated block `name` (between <!-- name:start --> and <!-- name:end -->) in
    `path` with `block`, so its tables are always the collected ones. A missing marker is an error:
    a block that moved without its writer would otherwise go stale silently. The figures keep their
    paths (results/curves/), so the images the docs embed update in place."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    text = path.read_text()
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path}: expected one {start} ... {end} block")
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
    strict = json.loads(QA_STRICT.read_text()) if QA_STRICT.exists() else {}
    for title, names in RESULT_BLOCKS:
        idx = [head.index(n) for n in names if n in head]
        rows = [[row[0], *(row[i] for i in idx)] for row in cells]
        if "qa_acc" in names:  # the strict checker beside the lenient columns of table.md
            keys = {"qa_acc": "qa_strict", "qa_num": "number_strict",
                    "qa_ident": "identifier_strict", "qa_term": "term_strict",
                    "qa_seen": "seen_strict", "qa_unseen": "unseen_strict"}  # fmt: skip
            # gold_lp's two parts per half (the answer tokens, the end token), from the
            # generations: a stage can move the end token (stopping) without the fact
            parts = [(p, h) for p in ("answer", "end") for h in (None, "seen", "unseen")]
            rows[0] = [*(f"{c} (lenient)" if c.startswith("qa_") else c for c in rows[0]),
                       *(f"{k} (strict)" for k in keys),
                       *(f"gold_lp {p}{f' {h}' if h else ''}" for p, h in parts)]  # fmt: skip
            rows[1] = [*rows[1], *["---"] * (len(keys) + len(parts))]
            for r in rows[2:]:
                st = strict.get(r[0], {})
                ok = st and not st.get("partial")
                r += [f"{st[k]:.3f}" if ok and st.get(k) is not None else "" for k in keys.values()]
                for p, h in parts:
                    v = item_lp(r[0], h, p)
                    r.append(f"{statistics.fmean(v.values()):.3f}" if v else "")
        rows = [rows[0], rows[1]] + [r for r in rows[2:] if any(r[1:])]  # rows with numbers here
        if len(rows) == 2:
            continue
        out.append(f"**{title}**\n\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    out.append(
        "qa_* (lenient) is `scorers.qa_correct`, the column `results/table.md` stores; qa_* (strict) "
        "is `scorers.qa_strict` (2026-10-09, the GRPO reward's rule: the whole gold, one candidate, "
        "units compared), re-scored from every row's saved generations by `eval/qa_strict.py`. No "
        "ordering changes between the two."
    )
    return "\n\n".join(out)


STRICT_KEYS = {  # metrics key -> eval/qa_strict.py's evals.json key (2026-10-09)
    "qa_strict": "qa_strict",
    "qa_strict_seen": "seen_strict",
    "qa_strict_unseen": "unseen_strict",
    "qa_strict_num": "number_strict",
    "qa_strict_ident": "identifier_strict",
    "qa_strict_term": "term_strict",
}


def load_metrics(run: str) -> dict | None:
    """results/runs/<run>/metrics.json, with the strict closed-book columns of
    results/qa_strict/evals.json added (eval/qa_strict.py; the qa_* keys are the lenient scorer,
    scorers.qa_correct)."""
    f = RUNS / run / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    strict = json.loads(QA_STRICT.read_text()).get(run, {}) if QA_STRICT.exists() else {}
    if not strict.get("partial"):  # a run scored on part of the items has no strict column
        m.update({k: strict[v] for k, v in STRICT_KEYS.items() if v in strict})
    for part, tag in (("answer", "ans"), ("end", "end")):  # gold_lp's two parts, per half
        for half, suffix in ((None, ""), ("seen", "_seen"), ("unseen", "_unseen")):
            v = item_lp(run, half, part)
            if v:
                m[f"gold_lp_{tag}{suffix}"] = statistics.fmean(v.values())
    return m


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
    ("  of it, the answer tokens", "gold_lp_ans", "lp_ans", None),
    ("  of it, the end token", "gold_lp_end", "lp_end", None),
    ("closed-book qa_strict", "qa_strict", "kpi", "domain_qa"),
    ("closed-book qa_acc (lenient)", "qa_acc", "kpi", "domain_qa"),
    ("grounded_acc (with passages)", "grounded_acc", "kpi", "grounded"),
    ("vocab_recall", "vocab_recall", "kpi", "vocab"),
    ("halluc_rate", "halluc_rate", "kpi", "adversarial"),
]


def paired_lp_se(ref: str, run: str, part: str = "total") -> float:
    """Standard error of the per-item gold_lp difference run - ref on the same domain_qa items:
    the noise a paired comparison of a continuous score has to beat."""
    a, b = item_lp(ref, None, part), item_lp(run, None, part)
    d = [b[i] - a[i] for i in a if i in b]
    return statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0


def delta_table() -> str:
    """Each Stage 2 run's change vs base-8b-hf next to the noise it has to beat: max(the seed gap
    between two runs of the same config, the metric's own standard error). Perplexity changes in
    %, the rest in points. Empty when the reference or the seed pair isn't scored yet."""
    runs = {}
    for r in [REF, *DELTA_RUNS]:
        if (v := load_metrics(r)) is not None:
            runs[r] = v
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
                    if kind in LP_PART
                    else (v - ref[key]) * 100
                )
        # the seed gap needs both seeds measured; a missing one is no gap, not a gap of 0
        pair = [change.get(r) for r in SEED_PAIR]
        seed = abs(pair[0] - pair[1]) if None not in pair else None
        if kind in LP_PART:
            se = paired_lp_se(REF, SEED_PAIR[0], LP_PART[kind])
        elif kind == "lm" and extra[0] in lm:
            se = lm[extra[0]].get(extra[1], 0) * 100
        elif kind == "kpi":
            n = ref.get("n", {}).get(extra, 0)
            se = math.sqrt(ref[key] * (1 - ref[key]) / n) * 100 if n else 0
        else:
            se = 0
        base = f"{ref[key]:.2f}" if kind == "ppl" or kind in LP_PART else f"{ref[key]:.3f}"
        cells = [label, base]
        fmt = "{:+.2f}" if kind in LP_PART else {"ppl": "{:+.2f}%"}.get(kind, "{:+.1f}")
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
    for r in ["base-8b", "instruct-8b", "base-8b-hf", "cpt-8b", *SFT_COLORS, "dpo"]:
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
    ("  of it, the answer tokens", "gold_lp_ans_unseen", "lp_ans", "unseen"),
    ("  of it, the end token", "gold_lp_end_unseen", "lp_end", "unseen"),
    ("seen gold-answer log-prob (nats)", "gold_lp_seen", "lp", "seen"),
    ("  of it, the answer tokens", "gold_lp_ans_seen", "lp_ans", "seen"),
    ("  of it, the end token", "gold_lp_end_seen", "lp_end", "seen"),
    ("qa_strict unseen", "qa_strict_unseen", "kpi", "qa_unseen"),
    ("qa_unseen (lenient)", "qa_unseen", "kpi", "qa_unseen"),
    ("qa_strict seen", "qa_strict_seen", "kpi", "qa_seen"),
    ("qa_seen (lenient)", "qa_seen", "kpi", "qa_seen"),
    ("qa_strict identifiers", "qa_strict_ident", "kpi", "qa_identifier"),
    ("qa_ident (lenient)", "qa_ident", "kpi", "qa_identifier"),
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


# gold_lp's parts (2026-10-09): the answer tokens and the one end token after them; the composite
# is their sum. A stage that changes the answer format moves the end token without the fact.
LP_PART = {"lp": "total", "lp_ans": "answer", "lp_end": "end"}


def lp_value(x: dict, part: str = "total") -> float:
    if part == "answer":
        return x["gold_lp"] - x["gold_lp_end"]
    return x["gold_lp_end"] if part == "end" else x["gold_lp"]


def item_lp(run: str, half: str | None = None, part: str = "total") -> dict[str, float]:
    """{domain_qa id: gold_lp} for a run, optionally one half (eval/tasks/sft_seen_chunks.txt);
    part "answer" is the answer tokens alone, "end" the end token alone."""
    path = RUNS / run / "generations.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.open())
    lp = {
        x["id"]: lp_value(x, part)
        for x in rows
        if x["task"] == "domain_qa" and x.get("gold_lp") is not None
    }
    if half is None:
        return lp
    seen = set(Path("eval/tasks/sft_seen_chunks.txt").read_text().split())
    src = {}
    for line in Path("eval/tasks/domain_qa.jsonl").read_text().splitlines():
        r = json.loads(line)
        src[r["id"]] = r["source_chunk"]
    return {i: v for i, v in lp.items() if i in src and (src[i] in seen) == (half == "seen")}


def paired_lp(a: str, b: str, half: str, n_boot: int = 10_000, part: str = "total") -> dict:
    """Mean per-item gold_lp difference b - a on one half, its paired SE and a 95% bootstrap
    interval over items (seeded): B7's first line with a = sft-from-base, b = sft-from-cpt."""
    x, y = item_lp(a, half, part), item_lp(b, half, part)
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
    if kind in LP_PART:
        se = paired_lp(pair[1], pair[0], extra, part=LP_PART[kind]).get("se", 0)
        return max(abs(a - b), se)
    if kind == "lm":
        return max(abs(a - b) * 100, lm_se(pair[0], *extra))
    n = sizes.get(extra) or m[pair[0]]["n"].get(extra, 0)
    return max(abs(a - b) * 100, math.sqrt(a * (1 - a) / n) * 100 if n else 0)


def sft_delta_table() -> str:
    """Absolute values per run, with the noise of Stage 3 (the sft-from-cpt seed twins) and of
    Stage 2 (cpt-8b vs cpt-8b-seed1) next to them, both max(seed gap, SE)."""
    m = {}
    for r in [*SFT_COLUMNS, "cpt-8b", "cpt-8b-seed1"]:
        if (v := load_metrics(r)) is not None:
            m[r] = v
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
        unit = "{:.3f}" if kind in LP_PART else "{:.1f}"
        cells = [label] + ["" if m[r].get(key) is None else f"{m[r][key]:.3f}" for r in cols]
        cells += [unit.format(n3), "" if n2 is None else unit.format(n2)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_lp(runs: list[str], half: str, part: str = "total") -> dict[str, float]:
    """Per-item gold_lp averaged over runs (items every run has)."""
    per = [item_lp(r, half, part) for r in runs]
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
        if (v := load_metrics(r)) is not None:
            m[r] = v
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
        split = ""
        if two_arm:
            parts = [
                bootstrap_diff(mean_lp(base, half, p), mean_lp(cpt, half, p))
                for p in ("answer", "end")
            ]
            split = (
                f" Of it, the answer tokens {parts[0]['mean']:+.3f} [{parts[0]['ci'][0]:+.3f}, "
                f"{parts[0]['ci'][1]:+.3f}] and the end token {parts[1]['mean']:+.3f} "
                f"[{parts[1]['ci'][0]:+.3f}, {parts[1]['ci'][1]:+.3f}]."
            )
        out.append(
            f"- **{half} gold_lp, {label}:** {c['mean']:+.3f} nats per answer [95% CI over items "
            f"{lo:+.3f}, {hi:+.3f}; {c['n']} items, {c['up']:.0%} up]; noise {floor:.3f} "
            f"({detail}): {verdict}.{split} The item CI conditions on these training runs; run "
            "variance enters only through the noise."
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
            "output tokens) and </s> on sampled answers (the eos job: 4 Stage 4 pool prompts per "
            "format x 4 at T 0.8; 20 prompts while the pool held replay, 16 after, 2026-10-08):\n"
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

    m = {r: load_metrics(r) for r in names}
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
    style(ax_qa, "Closed-book qa_acc, lenient scorer (%, +-1 binomial SE)", "", "%")
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
        f"{usd} per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).\n"
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
    """Win rate against the start on the 100 dpo_judge prompts (eval/winrate.py: Large 3, both
    orders, a split is a tie). Reported, not read (2026-10-08 amendment)."""
    head = [
        "run",
        "win rate",
        "SE",
        "ties",
        "identical greedy answers",
        "n",
        "position consistency",
    ]
    rows = []
    for name in names:
        f = Path("results/winrate") / f"{name}_vs_{DPO_START}.json"
        if f.exists():
            r = json.loads(f.read_text())
            o = r["overall"]
            rows.append([name, f"{o['win_rate']:.3f}", f"{o['se']:.3f}", o["ties"], o["identical"],
                         o["n"], f"{r['position_consistency']:.2f}"])  # fmt: skip
    if not rows:
        return ""
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


DPO_PAIR = ("dpo", "dpo-seed1")
DPO_ROWS = [  # the amended read (2026-10-08): primary verifier lines, then the guards
    ("halluc_rate (lower is better)", "halluc_rate", "kpi", "adversarial", "primary"),
    ("false_abstain (lower is better)", "false_abstain", "kpi", "grounded", "primary"),
    ("cite_valid", "cite_valid", "kpi", "grounded", "primary"),
    ("qa_seen (lenient, as registered)", "qa_seen", "kpi", "qa_seen", "primary"),
    ("qa_unseen (lenient, as registered)", "qa_unseen", "kpi", "qa_unseen", "primary"),
    ("qa_strict seen", "qa_strict_seen", "kpi", "qa_seen", "reported"),
    ("qa_strict unseen", "qa_strict_unseen", "kpi", "qa_unseen", "reported"),
    ("seen gold-answer log-prob (nats)", "gold_lp_seen", "lp", "seen", "primary"),
    ("  of it, the answer tokens", "gold_lp_ans_seen", "lp_ans", "seen", "reported"),
    ("  of it, the end token", "gold_lp_end_seen", "lp_end", "seen", "reported"),
    ("unseen gold-answer log-prob (nats)", "gold_lp_unseen", "lp", "unseen", "primary"),
    ("  of it, the answer tokens", "gold_lp_ans_unseen", "lp_ans", "unseen", "reported"),
    ("  of it, the end token", "gold_lp_end_unseen", "lp_end", "unseen", "reported"),
    ("MMLU", "mmlu", "lm", ("mmlu", "acc_stderr,none"), "guard"),
    ("GSM8K", "gsm8k", "lm", ("gsm8k", "exact_match_stderr,strict-match"), "guard"),
    ("grounded_acc (judge)", "grounded_acc", "kpi", "grounded", "reported"),
    ("cite_supported (judge)", "cite_supported", "kpi", "grounded", "reported"),
    ("vocab_recall (judge)", "vocab_recall", "kpi", "vocab", "reported"),
]


def dpo_delta_table() -> str:
    """Each DPO run against the start and the mean change of the two seeds, read two ways:
    - as written (pre-registered): noise = max(the DPO seed gap, the start's SE);
    - the Stage 3 way (2026-10-09 review): the floor also takes the start's own seed gap
      (sft-from-cpt vs sft-from-cpt-seed1), since the start is one run of a noisy stage.
    gold_lp gets the item-bootstrap 95% CI of the two-seed mean minus the start; beyond needs
    |change| > floor (and, for gold_lp, the CI excluding 0). One seed pair each: 1 df."""
    m = {}
    for r in (DPO_START, *DPO_PAIR, *SFT_SEED_PAIR):
        if (v := load_metrics(r)) is not None:
            m[r] = v
    if not all(r in m for r in (DPO_START, *DPO_PAIR, *SFT_SEED_PAIR)):
        return ""
    sizes = half_sizes()
    head = ["metric", "read", DPO_START, *DPO_PAIR, "change (mean of 2)", "noise as written",
            "beyond (as written)", "floor with the start's seed gap", "beyond (Stage 3 way)"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, kind, extra, role in DPO_ROWS:
        vals = [m[r].get(key) for r in (DPO_START, *DPO_PAIR)]
        if any(v is None for v in vals):
            continue
        start, a, b = vals
        scale = 1 if kind in LP_PART else 100
        change = ((a + b) / 2 - start) * scale
        n_start = noise(m, SFT_SEED_PAIR, key, kind, extra, sizes) or 0
        if kind in LP_PART:
            part = LP_PART[kind]
            n = max(abs(a - b), paired_lp(DPO_PAIR[1], DPO_PAIR[0], extra, part=part).get("se", 0))
            ci = bootstrap_diff(
                item_lp(DPO_START, extra, part), mean_lp(list(DPO_PAIR), extra, part)
            )["ci"]
            excl = ci[0] > 0 or ci[1] < 0
            note, unit = f"{change:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", "{:.3f}"
        else:
            if kind == "lm":
                n = max(abs(a - b) * 100, lm_se(DPO_START, *extra))
            else:
                k = sizes.get(extra) or m[DPO_START]["n"].get(extra, 0)
                n = max(abs(a - b) * 100, math.sqrt(start * (1 - start) / k) * 100 if k else 0)
            excl = True
            note, unit = f"{change:+.1f} pt", "{:.1f} pt"
        floor = max(n, n_start)
        cells = [label, role, *(f"{v:.3f}" for v in vals), note, unit.format(n),
                 "yes" if abs(change) > n and excl else "no", unit.format(floor),
                 "yes" if abs(change) > floor and excl else "no"]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


DPO_MORE = "dpo-2ep"  # the "more training" ablation (2026-10-08, post hoc): two epochs, epoch 2


def dpo_more_table() -> str:
    """dpo-2ep against dpo (one epoch, same data, seed and recipe) on the read's rows, next to the
    Stage 4 floor: max(the dpo / dpo-seed1 seed gap, dpo's SE; for gold_lp the twins' paired
    per-item SE), with the item-bootstrap CI for gold_lp."""
    m = {}
    for r in (DPO_PAIR[0], DPO_PAIR[1], DPO_MORE):
        if (v := load_metrics(r)) is not None:
            m[r] = v
    if not all(r in m for r in (*DPO_PAIR, DPO_MORE)):
        return ""
    sizes = half_sizes()
    one, more = DPO_PAIR[0], DPO_MORE
    head = [
        "metric",
        "read",
        f"{one} (1 epoch)",
        f"{more} (2 epochs)",
        "change",
        "floor",
        "beyond floor",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, kind, extra, role in DPO_ROWS:
        a, b, twin = m[one].get(key), m[more].get(key), m[DPO_PAIR[1]].get(key)
        if a is None or b is None or twin is None:
            continue
        if kind in LP_PART:
            part = LP_PART[kind]
            floor = max(abs(a - twin), paired_lp(DPO_PAIR[1], one, extra, part=part).get("se", 0))
            d = bootstrap_diff(item_lp(one, extra, part), item_lp(more, extra, part))
            change, beyond = b - a, abs(b - a) > floor and (d["ci"][0] > 0 or d["ci"][1] < 0)
            note, unit = f"{change:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", "{:.3f}"
        else:
            if kind == "lm":
                floor = max(abs(a - twin) * 100, lm_se(one, *extra))
            else:
                k = sizes.get(extra) or m[one]["n"].get(extra, 0)
                floor = max(abs(a - twin) * 100, math.sqrt(a * (1 - a) / k) * 100 if k else 0)
            change = (b - a) * 100
            beyond = abs(change) > floor
            note, unit = f"{change:+.1f} pt", "{:.1f} pt"
        cells = [
            label,
            role,
            f"{a:.3f}",
            f"{b:.3f}",
            note,
            unit.format(floor),
            "yes" if beyond else "no",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gate_flags(names) -> str:
    """Runs whose merge check failed but whose fp32-referenced diagnosis passed, evaluated under
    the 2026-10-09 rule (the row is flagged, not hidden)."""
    out = []
    for name in names:
        chk, diag = RUNS / name / "merge_check.json", RUNS / name / "merge_diagnose.json"
        if chk.exists() and diag.exists():
            c, d = json.loads(chk.read_text()), json.loads(diag.read_text())
            if not c.get("passed") and d.get("passed"):
                ratio = (
                    d["merged_bf16_vs_fp32_ref"]["mean_abs_lp_diff"]
                    / d["unmerged_bf16_vs_fp32_ref"]["mean_abs_lp_diff"]
                )
                out.append(
                    f"- **{name}: merge gate flagged.** Val-loss criterion "
                    f"{c['val_loss_rel_diff']:.2%} against 0.5% (merged {c['val_loss_merged']:.4f}, "
                    f"unmerged {c['val_loss_unmerged']:.4f} nats on sft_val). The merge-isolating "
                    f"criteria pass in `merge_diagnose`: {d['merge_adds_flips']} added flips of 11, "
                    f"|dlp| ratio {ratio:.2f} against 1.5. Evaluated under the rule fixed before the "
                    "diagnosis (notes/decisions.md, 2026-10-09)."
                )
    return "\n".join(out)


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
        f"dpo_val values at the last evaluation. $ at {usd} per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).\n"
    )
    if dt := dpo_delta_table():
        md += (
            f"\n## The read: change against {DPO_START}, next to the noise\n\n" + dt + "\n\n"
            "Rows marked primary are the amended read (2026-10-08, fixed before launch); guards must "
            "stay within the noise; judge-scored rows are reported, not read. As written: max(the DPO "
            "seed gap, the start's SE). The Stage 3 way also takes the start's own seed gap "
            f"({SFT_SEED_PAIR[0]} vs {SFT_SEED_PAIR[1]}): the start is one run. One seed pair each "
            "(1 df). The verdict uses the Stage 3 way.\n"
        )
    if mt := dpo_more_table():
        md += (
            f"\n## More training: {DPO_MORE} against {DPO_PAIR[0]} (ablation, not a candidate)\n\n"
            + mt
            + "\n\nSame data, seed and recipe; two epochs, epoch 2 evaluated. Floor: max(the "
            f"{DPO_PAIR[0]} / {DPO_PAIR[1]} seed gap, {DPO_PAIR[0]}'s SE), one seed pair.\n"
        )
    if wr := winrate_table(runs):
        md += f"\n## Win rate against {DPO_START} (reported, not read)\n\n" + wr + "\n"
    checks = checks_table(
        names=list(runs),
        starts=(DPO_START,),
        gate="over every sft_val completion position (11,351; dpo_val's 665 are too few for the "
        "0.1% line, 2026-10-08 freeze) against an fp32 reference, the argmax flips the merge adds "
        "over the unmerged bf16 model's own (at most 0.1% of positions)",
        diversity=(DPO_START,),  # the stage's own runs are added as names
    )
    if checks:
        md += "\n## Checks\n\n" + checks + "\n"
    if flags := gate_flags(runs):
        md += "\n" + flags + "\n"
    return md


# ---- Stage 5: GRPO ----------------------------------------------------------------------------

GRPO_PAIR = ("grpo", "grpo-seed1")
QA_STRICT = Path("results/qa_strict/evals.json")
PASSK = Path("results/passk")
GRPO_ROWS = [  # the pre-registered read (2026-10-09): primary, reported, guards
    ("qa_strict seen (the reward's rule)", "qa_strict_seen", "kpi", "qa_seen", "primary"),
    ("seen gold-answer log-prob (nats)", "gold_lp_seen", "lp", "seen", "primary"),
    ("  of it, the answer tokens", "gold_lp_ans_seen", "lp_ans", "seen", "reported"),
    ("  of it, the end token", "gold_lp_end_seen", "lp_end", "seen", "reported"),
    ("qa_acc seen (lenient)", "qa_seen", "kpi", "qa_seen", "reported"),
    ("qa_strict unseen", "qa_strict_unseen", "kpi", "qa_unseen", "reported"),
    ("qa_acc unseen (lenient)", "qa_unseen", "kpi", "qa_unseen", "reported"),
    ("unseen gold-answer log-prob (nats)", "gold_lp_unseen", "lp", "unseen", "reported"),
    ("  of it, the answer tokens", "gold_lp_ans_unseen", "lp_ans", "unseen", "reported"),
    ("  of it, the end token", "gold_lp_end_unseen", "lp_end", "unseen", "reported"),
    ("halluc_rate (lower is better)", "halluc_rate", "kpi", "adversarial", "guard"),
    ("false_abstain (lower is better)", "false_abstain", "kpi", "grounded", "guard"),
    ("cite_valid", "cite_valid", "kpi", "grounded", "guard"),
    ("MMLU", "mmlu", "lm", ("mmlu", "acc_stderr,none"), "guard"),
    ("GSM8K", "gsm8k", "lm", ("gsm8k", "exact_match_stderr,strict-match"), "guard"),
    ("grounded_acc (judge)", "grounded_acc", "kpi", "grounded", "reported"),
    ("cite_supported (judge)", "cite_supported", "kpi", "grounded", "reported"),
]


def metrics_with_strict(runs) -> dict:
    """metrics.json per run, plus the strict closed-book columns (eval/qa_strict.py)."""
    return {r: v for r in runs if (v := load_metrics(r)) is not None}


def change_row(m, label, key, kind, extra, start, pair, start_pairs, sizes):
    """(cells, beyond) for one metric: the mean of `pair` (or the one run) minus `start`, against
    max(the pair's seed gap, the start's SE, each start pair's own seed gap); gold_lp also needs
    its item-bootstrap 95% CI to exclude 0."""
    runs = [r for r in pair if r in m and m[r].get(key) is not None]
    if start not in m or m[start].get(key) is None or not runs:
        return None
    st = m[start][key]
    scale = 1 if kind in LP_PART else 100
    change = (sum(m[r][key] for r in runs) / len(runs) - st) * scale
    gaps = [noise(m, pp, key, kind, extra, sizes) for pp in start_pairs if all(r in m for r in pp)]
    if kind in LP_PART:
        part = LP_PART[kind]
        own = noise(m, pair, key, kind, extra, sizes) if len(runs) == 2 else 0
        ci = bootstrap_diff(item_lp(start, extra, part), mean_lp(runs, extra, part))["ci"]
        excl, note = ci[0] > 0 or ci[1] < 0, f"{change:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        floor, unit = max([own or 0, *(g or 0 for g in gaps)]), "{:.3f}"
    else:
        if kind == "lm":
            se = lm_se(start, *extra)
        else:
            k = sizes.get(extra) or m[start]["n"].get(extra, 0)
            se = math.sqrt(st * (1 - st) / k) * 100 if k else 0
        own = abs(m[runs[0]][key] - m[runs[1]][key]) * 100 if len(runs) == 2 else 0
        excl, note = True, f"{change:+.1f} pt"
        floor, unit = max([own, se, *(g or 0 for g in gaps)]), "{:.1f} pt"
    beyond = abs(change) > floor and excl
    return [note, unit.format(floor), "yes" if beyond else "no"], beyond


PASSK_ROWS_READ = [  # (label, scorer metric, half): paired per item, strict scorer, 8 x T 0.7
    ("pass@1 seen (sampled)", "pass@1", "seen"),
    ("pass@8 seen", "pass@n", "seen"),
    ("pass@1 unseen (sampled)", "pass@1", "unseen"),
    ("pass@8 unseen", "pass@n", "unseen"),
]


def passk_items(run: str) -> dict | None:
    f = PASSK / f"{run}.json"
    return {p["id"]: p for p in json.loads(f.read_text())["per_item"]} if f.exists() else None


def passk_change(start: str, runs, metric: str, half: str, gap_pairs=()) -> list[str] | None:
    """[change with its paired 95% CI, floor, beyond] for one pass@k line: the mean of `runs`
    minus `start` per item (strict scorer), item-bootstrapped; floor = the largest seed gap
    available (the runs' own pair, and each pair in gap_pairs); beyond needs |change| > floor and
    the CI excluding 0."""
    s, rs = passk_items(start), [passk_items(r) for r in runs]
    if s is None or any(r is None for r in rs):
        return None
    ids = sorted(i for i in s if s[i]["half"] == half)
    val = lambda d, i: float(d[i]["strict"][metric])
    diff = np.array([sum(val(r, i) for r in rs) / len(rs) - val(s, i) for i in ids])
    boot = diff[np.random.default_rng(0).integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pairs = [tuple(runs)] if len(runs) == 2 else []
    gaps = [abs(np.mean([val(a, i) - val(b, i) for i in ids]))
            for a, b in ([tuple(passk_items(r) for r in pp) for pp in (*pairs, *gap_pairs)])
            if a is not None and b is not None]  # fmt: skip
    floor, change = max(gaps, default=0.0) * 100, diff.mean() * 100
    beyond = abs(change) > floor and (lo > 0 or hi < 0)
    return [
        f"{change:+.1f} pt [{lo * 100:+.1f}, {hi * 100:+.1f}]",
        f"{floor:.1f} pt",
        "yes" if beyond else "no",
    ]


def grpo_delta_table() -> str:
    """Each GRPO run against the start (dpo-strict) and the mean change of the two seeds. Floor:
    max(the GRPO seed gap, the start's SE, the start's own seed gap). dpo-strict has no twin, so
    its gap is the lenient dpo / dpo-seed1 pair's (2026-10-09). One seed pair each: 1 df."""
    m = metrics_with_strict((GRPO_START, *GRPO_PAIR, *DPO_PAIR))
    if not all(r in m for r in (GRPO_START, *GRPO_PAIR)):
        return ""
    sizes = half_sizes()
    head = ["metric", "read", GRPO_START, *GRPO_PAIR, "change (mean of 2)", "floor", "beyond"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, kind, extra, role in GRPO_ROWS:
        row = change_row(m, label, key, kind, extra, GRPO_START, GRPO_PAIR, (DPO_PAIR,), sizes)
        if row is None:
            continue
        vals = [m[r].get(key) for r in (GRPO_START, *GRPO_PAIR)]
        cells = [label, role, *("" if v is None else f"{v:.3f}" for v in vals), *row[0]]
        lines.append("| " + " | ".join(cells) + " |")
    for label, metric, half in PASSK_ROWS_READ:
        ch = passk_change(GRPO_START, GRPO_PAIR, metric, half, (DPO_PAIR,))
        if ch is None:
            continue
        vals = [passk_items(r) for r in (GRPO_START, *GRPO_PAIR)]
        means = [
            np.mean([float(v[i]["strict"][metric]) for i in v if v[i]["half"] == half])
            for v in vals
        ]
        lines.append(
            "| " + " | ".join([label, "reported", *(f"{x:.3f}" for x in means), *ch]) + " |"
        )
    return "\n".join(lines)


def two_algorithm_table() -> str:
    """Same verifier, two algorithms, each measured from its own start, and the chain's cumulative
    change: DPO (dpo-strict, offline pairs, one run) from sft-from-cpt; GRPO (two seeds, on-policy
    groups) from dpo-strict, where it started; GRPO from sft-from-cpt (the chain). Each cell is the
    change and whether it clears its floor (shown): the run's own seed gap where it has a twin, the
    start's SE and the start's seed gap (dpo-strict borrows the lenient DPO pair's). pass@k lines are
    paired per item; sft-from-cpt's twin was not sampled, so their floors are the DPO and GRPO pairs'."""
    m = metrics_with_strict((DPO_START, *SFT_SEED_PAIR, GRPO_START, *DPO_PAIR, *GRPO_PAIR))
    if not all(r in m for r in (DPO_START, GRPO_START, *GRPO_PAIR)):
        return ""
    sizes = half_sizes()
    groups = [  # (start, runs, start's seed pairs for the floor)
        (DPO_START, (GRPO_START,), (SFT_SEED_PAIR, DPO_PAIR)),
        (GRPO_START, GRPO_PAIR, (DPO_PAIR,)),
        (DPO_START, GRPO_PAIR, (SFT_SEED_PAIR,)),
    ]
    head = ["metric", DPO_START, "DPO from its own start (sft-from-cpt)", "beyond (floor)",
            "GRPO from its own start (dpo-strict)", "beyond (floor)",
            "cumulative from sft-from-cpt (grpo)", "beyond (floor)"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    keep = (
        "qa_strict_seen",
        "qa_strict_unseen",
        "gold_lp_seen",
        "gold_lp_ans_seen",
        "gold_lp_end_seen",
        "gold_lp_unseen",
        "gold_lp_ans_unseen",
        "gold_lp_end_unseen",
        "halluc_rate",
        "mmlu",
        "gsm8k",
    )
    for label, key, kind, extra, _ in GRPO_ROWS:
        if key not in keep:
            continue
        cells = []
        for start, runs, pairs in groups:
            row = change_row(m, label, key, kind, extra, start, runs, pairs, sizes)
            if row is None:
                break
            cells += [row[0][0], f"{row[0][2]} ({row[0][1]})"]
        else:
            lines.append("| " + " | ".join([label, f"{m[DPO_START][key]:.3f}", *cells]) + " |")
    for label, metric, half in PASSK_ROWS_READ:
        cells = []
        for start, runs, pairs in groups:
            ch = passk_change(
                start, runs, metric, half, tuple(pp for pp in pairs if pp != SFT_SEED_PAIR)
            )
            if ch is None:
                break
            cells += [ch[0], f"{ch[2]} ({ch[1]})"]
        else:
            v = passk_items(DPO_START)
            st = np.mean([float(v[i]["strict"][metric]) for i in v if v[i]["half"] == half])
            lines.append("| " + " | ".join([label, f"{st:.3f}", *cells]) + " |")
    return "\n".join(lines)


PASSK_ROWS = ["sft-from-cpt", "dpo", "dpo-seed1", "dpo-strict", "grpo", "grpo-seed1", "instruct-8b"]


def passk_table() -> str:
    """pass@1, maj@8 and pass@8 (strict scorer) on the seen and unseen closed-book items, 8 samples
    at T 0.7 (eval/passk.py), with the per-item SE."""
    rows = {r: json.loads((PASSK / f"{r}.json").read_text()) for r in PASSK_ROWS
            if (PASSK / f"{r}.json").exists()}  # fmt: skip
    if not rows:
        return ""
    head = ["run", *(f"{h} {k}" for h in ("seen", "unseen") for k in ("pass@1", "maj@8", "pass@8"))]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r, d in rows.items():
        cells = [r]
        for h in ("seen", "unseen"):
            for k in ("pass@1", "maj@n", "pass@n"):
                cells.append(f"{d[h][f'strict_{k}']:.3f} ± {d[h][f'strict_{k}_se']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_passk(out: Path) -> None:
    """pass@1 against pass@8 per run (strict scorer), seen and unseen: RL that sharpens moves a
    point right (pass@1) without moving it up (pass@8)."""
    rows = {r: json.loads((PASSK / f"{r}.json").read_text()) for r in PASSK_ROWS
            if (PASSK / f"{r}.json").exists()}  # fmt: skip
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    color = {"sft-from-cpt": "#52514e", "instruct-8b": "#eda100", "dpo": "#eb6834",
             "dpo-seed1": "#eb6834", "dpo-strict": "#1baf7a", "grpo": "#2a78d6",
             "grpo-seed1": "#2a78d6"}  # fmt: skip
    for ax, half in zip(axes, ("seen", "unseen")):
        labels = []
        for r, d in rows.items():
            x, y = d[half]["strict_pass@1"], d[half]["strict_pass@n"]
            ax.errorbar(x, y, xerr=d[half]["strict_pass@1_se"], yerr=d[half]["strict_pass@n_se"],
                        fmt="o", ms=8, color=color.get(r, INK_2), mec=SURFACE, mew=1.5,
                        mfc="none" if r in TWIN else color.get(r, INK_2), elinewidth=1)  # fmt: skip
            labels.append((r, x, y, False))
        style(ax, f"{half} items: pass@1 vs pass@8 (strict)", "pass@1 (8 samples, T 0.7)", "pass@8")
        end_labels(ax, labels)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def grpo_table(runs: dict, usd: float) -> str:
    head = ["run", "start", "tasks", "steps", "stop", "rule picks", "grpo_val pass@1 / pass@8 at the pick",
            "final train reward", "final entropy", "mean length", "base drift", "wall (h)", "GPU-h", "$",
            "peak GB"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, _) in runs.items():
        b4 = RUNS / name / "b4.json"
        pick = json.loads(b4.read_text()) if b4.exists() else {}
        at = {e["step"]: e for e in s.get("val_curve", [])}.get(pick.get("picked_step"), {})
        stop = s.get("stop")
        cells = [
            name, Path(s["base"]).name, s["tasks"], s["steps"],
            f"step {stop['step']}: {stop['reason']}" if stop else "none",
            f"step {pick['picked_step']} (best {pick['best_step']})" if pick else "",
            f"{at['val_pass@1']:.3f} / {at['val_pass@8']:.3f}" if at else "",
            s.get("final_reward"), s.get("final_entropy"), s.get("final_mean_length"),
            "" if s.get("base_drift_max") is None else f"{s['base_drift_max']:.1e}",
            f"{s['wall_s'] / 3600:.2f}", f"{s['gpu_hours']:.2f}", f"{s['gpu_hours'] * usd:.2f}",
            f"{s['peak_mem_gb']:.0f}" if s.get("peak_mem_gb") else "",
        ]  # fmt: skip
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    return "\n".join(lines)


def plot_grpo(runs: dict, out: Path) -> None:
    """Six panels: train reward, grpo_val pass@1 / pass@8 (points), mean completion length,
    entropy, frac_reward_zero_std, and the rollout engine's log-prob gap to the policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0, 0], "reward", "Train reward (0.1 format + 0.9 correct + length)", "reward"),
        (axes[0, 2], "completions/mean_length", "Mean completion length", "tokens"),
        (axes[1, 0], "entropy", "Policy entropy", "nats / token"),
        (
            axes[1, 1],
            "zero_spread",
            "Task groups with identical rewards (from the rollouts)",
            "share",
        ),
        (
            axes[1, 2],
            "sampling/sampling_logp_difference/mean",
            "vLLM vs policy log-prob gap",
            "nats / token",
        ),
    ]
    last_step = max(s["steps"] for s, _ in runs.values())
    for ax, key, title, ylabel in panels:
        labels = []
        for name, (_, log) in runs.items():
            color, ls = GRPO_COLORS.get(name, INK_2), "--" if name in TWIN else "-"
            pts = [p for p in val_curve([r for r in log if "val_pass@1" not in r], key)]
            if pts:
                x, y = np.array([p[0] for p in pts]), smooth(np.array([p[1] for p in pts]))
                ax.plot(x, y, color=color, lw=2, ls=ls, solid_capstyle="round")
                labels.append((name, x[-1], y[-1], False))
        style(ax, title, "", ylabel)
        ax.set_xlim(0, last_step * 1.3)
        end_labels(ax, labels)
    ax, labels = axes[0, 1], []
    for name, (_, log) in runs.items():
        color = GRPO_COLORS.get(name, INK_2)
        for key, mk in (("val_pass@1", "o"), ("val_pass@8", "s")):
            if pts := val_curve(log, key):
                x, y = zip(*pts)
                ax.plot(x, y, marker=mk, ls="--" if name in TWIN else "-", lw=1, ms=6, color=color,
                        mec=SURFACE, mew=1.5)  # fmt: skip
                labels.append((f"{name} {key[4:]}", x[-1], y[-1], False))
    style(ax, "grpo_val pass@1 (circles) and pass@8 (squares)", "", "share of 50 tasks")
    ax.set_xlim(-5, last_step * 1.3)
    end_labels(ax, labels)
    for a in axes[1]:
        a.set_xlabel("optimizer step", color=INK_2)
    fig.suptitle(f"Stage 5 GRPO: train ({SMOOTH}-step moving average) and grpo_val (points)",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def stage5_md(runs: dict, usd: float) -> str:
    md = "## Training runs\n\n" + grpo_table(runs, usd)
    md += (
        "\n\nThe checkpoint rule (pre-registered): the best grpo_val pass@1 among the saves at or "
        "before any stop, ties within one SE to the earliest. grpo_val: 50 held-out tasks x 8 "
        f"samples at T 1.0. $ at {usd} per GPU-hour (Modal's H100 SXM5 list price, checked 2026-10-09).\n"
    )
    if dt := grpo_delta_table():
        md += (
            f"\n## The read: change against {GRPO_START}, next to the noise\n\n" + dt + "\n\n"
            "Rows marked primary are the pre-registered read (2026-10-09); guards must stay within "
            "the noise and their absolute lines; reported rows are not argued. Floor: max(the GRPO "
            f"seed gap, the start's SE, the start's own seed gap: {GRPO_START} has no twin, so the "
            f"lenient {DPO_PAIR[0]} / {DPO_PAIR[1]} gap). One seed pair each (1 df).\n"
        )
    if ta := two_algorithm_table():
        md += (
            "\n## Same verifier, two algorithms: each from its own start, and the chain's cumulative "
            f"change from {DPO_START}\n\n" + ta + "\n\n"
            "dpo-strict: 445 offline pairs from the SFT model's samples, labelled by the strict "
            "checker, one run. grpo: on-policy groups scored by the same checker, two seeds. Each "
            f"floor also takes {DPO_START}'s own seed gap. pass@k lines: paired per item (8 samples "
            "at T 0.7, the strict scorer), floor = the seed gaps sampled (grpo's pair; the lenient "
            f"dpo pair for dpo-strict); {DPO_START}'s twin was not sampled. Win rate: not run for "
            "these rows: the judge failed its benchmark and its two orders agreed at coin-flip.\n"
        )
    if pk := passk_table():
        md += (
            "\n## pass@k on the closed-book eval (strict scorer)\n\n" + pk + "\n\n"
            "8 samples per item at T 0.7 (eval/passk.py); ± is the SE over items.\n"
        )
    checks = checks_table(
        names=list(runs),
        starts=(GRPO_START,),
        gate="over every sft_val completion position (11,351) against an fp32 reference, the "
        "argmax flips the merge adds over the unmerged bf16 model's own (at most 0.1% of positions)",
        diversity=(GRPO_START,),
    )
    if checks:
        md += "\n## Checks\n\n" + checks + "\n"
    return md


# ------------------------------------------------------------------------------------- Stage 6 ---
SERVE = Path("results/serve")
S6_RUN = {  # variant -> the run its gate generations are filed under (serve/modal_serve.GATE_RUN)
    "bf16": "dpo-strict",
    "fp8": "dpo-strict-fp8",
    "fp8kv": "dpo-strict-fp8kv",
    "w4a16": "dpo-strict-w4a16",
}
S6_LABEL = {"bf16": "bf16", "fp8": "FP8", "fp8kv": "FP8 + FP8 KV", "w4a16": "INT4 W4A16"}
S6_COLORS = {"bf16": INK, "fp8": "#2a78d6", "fp8kv": "#8a5cd6", "w4a16": "#d6772a"}
# Pre-registered (notes/decisions.md, 2026-10-09, Stage 6): a variant ships if every line's
# |variant - bf16| is within the Stage 3 floor (report.noise on the sft-from-cpt pair)
S6_GATE = [  # (label, key, "rate" in points or "lp" in nats, floor)
    ("qa_strict unseen", "qa_strict_unseen", "rate", 2.6),
    ("qa_strict seen", "qa_strict_seen", "rate", 3.5),
    ("gold_lp answer tokens, unseen (nats)", "gold_lp_ans_unseen", "lp", 0.210),
    ("gold_lp answer tokens, seen (nats)", "gold_lp_ans_seen", "lp", 0.173),
    ("gold_lp end token, unseen (nats)", "gold_lp_end_unseen", "lp", 0.048),
    ("gold_lp end token, seen (nats)", "gold_lp_end_seen", "lp", 0.074),
    ("grounded_acc", "grounded_acc", "rate", 2.8),
    ("cite_valid", "cite_valid", "rate", 1.85),
    ("halluc_rate", "halluc_rate", "rate", 3.9),
    ("false_abstain", "false_abstain", "rate", 0.9),
    ("GSM8K (all 1,319, add_bos_token)", "gsm8k_gate", "rate", 2.2),
]
S6_REPORTED = [("qa_strict identifiers (reported)", "qa_strict_ident", "rate", 5.0)]
S6_EOS_PASS = 0.95
# decode bytes per token / 3.35 TB/s at concurrency 1 (7.42B linear + 0.54B untied lm_head params)
S6_ITL_FLOOR_MS = {"bf16": 4.75, "fp8": 2.5, "fp8kv": 2.5, "w4a16": 1.5}
# Step 0's estimate of concurrent 8,192-token sequences at --gpu-memory-utilization 0.90
S6_EST_CONCURRENCY = {"bf16": 44, "fp8": 50, "fp8kv": 100, "w4a16": 53}
# The Mistral Small 4 announcement's API price, $ per million (input, output) tokens, and its
# minimum self-hosting hardware (4x HGX H100; 2x H200 or 1x DGX B200), checked 2026-10-09
SMALL4_USD_PER_MTOK = (0.15, 0.60)
SMALL4_MIN_H100 = 4


def s6_metrics(variant: str) -> dict:
    """The gate's lines for one variant: the KPI metrics (load_metrics, scored on the Mac), the
    gate's GSM8K run, the eos job's </s> rate and vLLM's val-slice perplexity."""
    run = S6_RUN[variant]
    m = load_metrics(run) or {}
    files = sorted((SERVE / "gate/lm_eval" / run).glob("**/results*.json"))
    if files:
        res = json.loads(files[-1].read_text())["results"]["gsm8k"]
        m["gsm8k_gate"] = res["exact_match,strict-match"]
    eos = RUNS / run / "samples/eos.jsonl"
    if eos.exists():
        s = [x for r in map(json.loads, eos.read_text().splitlines()) for x in r["samples"]]
        m["eos_rate"] = statistics.fmean(
            x["finish_reason"] == "stop" and x["token_ids"][-1:] == [2] for x in s
        )
    if (ppl := Path("results/vllm_ppl") / f"{run}-hf.json").exists():
        m["vllm_ppl"] = json.loads(ppl.read_text())["ppl_val_slice"]
    return m


def s6_gate() -> tuple[str, dict]:
    """The gate table and each variant's verdict ("ships", "fails", or "incomplete")."""
    m = {v: s6_metrics(v) for v in S6_RUN}
    ref = m["bf16"]
    variants = [v for v in S6_RUN if v != "bf16" and m[v]]
    if not variants:
        return "", {}
    head = ["line", "bf16", *(f"{S6_LABEL[v]} − bf16" for v in variants), "floor"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    verdict = dict.fromkeys(variants, "ships")
    for label, key, kind, floor in S6_GATE + S6_REPORTED:
        gated = (label, key, kind, floor) in S6_GATE
        scale, fmt = (100, "{:+.1f}") if kind == "rate" else (1, "{:+.3f}")
        cells = [label, "" if ref.get(key) is None else f"{ref[key]:.3f}"]
        for v in variants:
            if ref.get(key) is None or m[v].get(key) is None:
                cells.append("")
                if gated:
                    verdict[v] = "incomplete" if verdict[v] == "ships" else verdict[v]
                continue
            d = (m[v][key] - ref[key]) * scale
            out = abs(d) > floor
            cells.append(f"**{fmt.format(d)}**" if out else fmt.format(d))
            if out and gated:
                verdict[v] = "fails"
        cells.append(f"{floor:g} pt" if kind == "rate" else f"{floor:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    cells = [
        "answers ending on </s> (eos job)",
        "" if "eos_rate" not in ref else f"{ref['eos_rate']:.2f}",
    ]
    for v in variants:
        e = m[v].get("eos_rate")
        cells.append("" if e is None else f"{e:.2f}" if e >= S6_EOS_PASS else f"**{e:.2f}**")
        if e is None:
            verdict[v] = "incomplete" if verdict[v] == "ships" else verdict[v]
        elif e < S6_EOS_PASS:
            verdict[v] = "fails"
    lines.append("| " + " | ".join([*cells, f">= {S6_EOS_PASS}"]) + " |")
    cells = [
        "vLLM val-slice perplexity (reported)",
        "" if "vllm_ppl" not in ref else f"{ref['vllm_ppl']:.3f}",
    ]
    for v in variants:
        p = m[v].get("vllm_ppl")
        cells.append(
            "" if p is None or "vllm_ppl" not in ref else f"{(p / ref['vllm_ppl'] - 1):+.2%}"
        )
    lines.append("| " + " | ".join([*cells, ""]) + " |")
    lines.append(
        "| " + " | ".join(["**verdict**", "", *(f"**{verdict[v]}**" for v in variants), ""]) + " |"
    )
    return "\n".join(lines), verdict


def s6_reps(variant: str, name: str) -> list[dict]:
    """The repeated runs of one bench config, each with its prefix-cache hit rate lifted out."""
    reps = [
        json.loads(f.read_text())
        for f in sorted((SERVE / "bench" / variant).glob(f"{name}-rep*.json"))
    ]
    for r in reps:
        r["hit_rate"] = (r.get("prefix_cache") or {}).get("hit_rate")
    return reps


def s6_mean(reps: list[dict], key: str) -> float | None:
    vals = [r[key] for r in reps if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def s6_spread(reps: list[dict], key: str = "request_throughput") -> float | None:
    """(max - min) / mean of a config's repeated runs: over 20% is the pre-registered rerun flag."""
    vals = [r[key] for r in reps if r.get(key)]
    return (max(vals) - min(vals)) / statistics.fmean(vals) if len(vals) > 1 else None


def s6_variants() -> list[str]:
    return [v for v in S6_RUN if (SERVE / "bench" / v).is_dir()]


def f1(x: float | None, fmt: str = "{:.1f}") -> str:
    return "" if x is None else fmt.format(x)


def s6_latency_table() -> str:
    head = ["variant", "concurrency", "TTFT p50 / p99 (ms)", "ITL p50 / p99 (ms)",
            "E2EL p50 / p99 (ms)", "req/s", "output tok/s", "goodput share", "run pair spread"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in s6_variants():
        for c in (1, 8, 32, 64):
            reps = s6_reps(v, f"c{c}")
            if not reps:
                continue
            itl50 = s6_mean(reps, "median_itl_ms")
            flag = (
                " (under the H100 floor)" if c == 1 and itl50 and itl50 < S6_ITL_FLOOR_MS[v] else ""
            )
            sp = s6_spread(reps)
            lines.append("| " + " | ".join([
                S6_LABEL[v], str(c),
                f"{f1(s6_mean(reps, 'median_ttft_ms'))} / {f1(s6_mean(reps, 'p99_ttft_ms'))}",
                f"{f1(itl50, '{:.2f}')}{flag} / {f1(s6_mean(reps, 'p99_itl_ms'), '{:.2f}')}",
                f"{f1(s6_mean(reps, 'median_e2el_ms'), '{:,.0f}')} / {f1(s6_mean(reps, 'p99_e2el_ms'), '{:,.0f}')}",
                f1(s6_mean(reps, "request_throughput"), "{:.2f}"),
                f1(s6_mean(reps, "output_throughput"), "{:,.0f}"),
                f1(s6_mean(reps, "goodput_share"), "{:.0%}"),
                "" if sp is None else f"{sp:.0%}" + (" (rerun)" if sp > 0.2 else ""),
            ]) + " |")  # fmt: skip
    return "\n".join(lines) if len(lines) > 2 else ""


def s6_load_table() -> str:
    head = ["variant", "offered req/s", "achieved req/s", "TTFT p50 / p99 (ms)", "E2EL p99 (ms)",
            "goodput share"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in s6_variants():
        for r in (1, 4, 16):
            reps = s6_reps(v, f"r{r}")
            if not reps:
                continue
            lines.append("| " + " | ".join([
                S6_LABEL[v], str(r), f1(s6_mean(reps, "request_throughput"), "{:.2f}"),
                f"{f1(s6_mean(reps, 'median_ttft_ms'))} / {f1(s6_mean(reps, 'p99_ttft_ms'))}",
                f1(s6_mean(reps, "p99_e2el_ms"), "{:,.0f}"),
                f1(s6_mean(reps, "goodput_share"), "{:.0%}"),
            ]) + " |")  # fmt: skip
    return "\n".join(lines) if len(lines) > 2 else ""


def s6_pair_table(rows: list[tuple[str, str, str]], cols: list[tuple[str, str, str]]) -> str:
    """rows: (label, variant, run name); cols: (header, key, format), means over the repeats."""
    lines = [
        "| " + " | ".join(["run", *(c[0] for c in cols)]) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for label, v, name in rows:
        reps = s6_reps(v, name)
        if reps:
            cells = [f1(s6_mean(reps, k), f) for _, k, f in cols]
            lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


def s6_prefix_table() -> str:
    return s6_pair_table(
        [("grounded, passages as evaluated (none shared)", "fp8", "grounded-unique-c8"),
         ("grounded, 4 questions per shared context (rag)", "fp8", "grounded-rag-c8")],
        [("TTFT p50 (ms)", "median_ttft_ms", "{:.1f}"), ("TTFT p99 (ms)", "p99_ttft_ms", "{:.1f}"),
         ("prefix-cache hit rate", "hit_rate", "{:.0%}"), ("E2EL p50 (ms)", "median_e2el_ms", "{:,.0f}")],
    )  # fmt: skip


def s6_spec_table() -> str:
    """FP8 with and without n-gram speculation, both served in the same container (fp8-ngram-base
    and fp8-ngram): the main sweep's FP8 rows ran on another host."""
    if not (SERVE / "bench" / "fp8-ngram").is_dir():
        return ""
    rows = []
    for d, label in (("grounded_unique", "grounded"), ("closedbook", "closed-book")):
        for c in (1, 8):
            rows += [(f"{label}, c={c}, FP8", "fp8-ngram-base", f"{d}-c{c}"),
                     (f"{label}, c={c}, FP8 + n-gram", "fp8-ngram", f"{d}-c{c}")]  # fmt: skip
    return s6_pair_table(
        rows,
        [("ITL p50 (ms)", "median_itl_ms", "{:.2f}"), ("TPOT p50 (ms)", "median_tpot_ms", "{:.2f}"),
         ("E2EL p50 (ms)", "median_e2el_ms", "{:,.0f}"), ("output tok/s", "output_throughput", "{:,.0f}")],
    )  # fmt: skip


def s6_memory_table() -> str:
    head = ["variant", "GPU", "weights (vLLM, GiB)", "KV cache (GiB)", "KV cache tokens",
            "8,192-token sequences: vLLM / Step 0 estimate"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in s6_variants():
        f = SERVE / "bench" / v / "server.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        srv = s["server"]
        lines.append("| " + " | ".join([
            S6_LABEL[v], f"{s['gpu']} (driver {s['driver']}, vLLM {s['vllm']})",
            f1(srv.get("weights_gib"), "{:.2f}"), f1(srv.get("kv_cache_gib"), "{:.2f}"),
            f1(srv.get("kv_cache_tokens"), "{:,.0f}"),
            f"{f1(srv.get('max_concurrency_8192'), '{:.1f}')} / {S6_EST_CONCURRENCY[v]}",
        ]) + " |")  # fmt: skip
    return "\n".join(lines) if len(lines) > 2 else ""


def s6_cost_table(usd: float, verdict: dict) -> str:
    """$ per 1,000 requests at each variant's goodput-maximising concurrency, against the Mistral
    API at Small 4's listed prices for the same tokens, and the sustained load at which one H100
    costs what the API would."""
    head = ["variant", "gate", "best concurrency", "goodput (req/s)", "$ / 1k requests (1 H100)",
            "API $ / 1k requests (Small 4 prices)", "break-even sustained req/s"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    p_in, p_out = SMALL4_USD_PER_MTOK
    for v in s6_variants():
        best = None
        for c in (1, 8, 32, 64):
            reps = s6_reps(v, f"c{c}")
            g = s6_mean(reps, "request_goodput")
            if g and (best is None or g > best[1]):
                best = (c, g, reps)
        if best is None:
            continue
        c, g, reps = best
        n_in = s6_mean(reps, "total_input_tokens") / s6_mean(reps, "completed")
        n_out = s6_mean(reps, "total_output_tokens") / s6_mean(reps, "completed")
        api_per_req = (n_in * p_in + n_out * p_out) / 1e6
        lines.append("| " + " | ".join([
            S6_LABEL[v], "reference" if v == "bf16" else verdict.get(v, "not run"), str(c),
            f"{g:.1f}", f"{usd / (g * 3600) * 1000:.4f}", f"{api_per_req * 1000:.4f}",
            f"{usd / 3600 / api_per_req:.1f}",
        ]) + " |")  # fmt: skip
    if len(lines) == 2:
        return ""
    return "\n".join(lines) + (
        f"\n\nOne H100 at ${usd}/h (Modal's H100 SXM5 list price, $0.001097/s, checked "
        f"2026-10-09); Small 4's API at ${p_in} / ${p_out} per million input / output tokens and its "
        f"self-hosting minimum of {SMALL4_MIN_H100} H100s (${SMALL4_MIN_H100 * usd:.2f}/h before any "
        "request), both from Mistral's Small 4 announcement. Tokens per "
        "request are the bench mix's measured means. The break-even is the sustained load above "
        "which the GPU is cheaper than the API; below it, an idle GPU costs the same per hour."
    )


def plot_serve(out_latency: Path, out_load: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    vs = s6_variants()
    if not vs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title, ylabel in (
        (axes[0], "ttft_ms", "Time to first token vs concurrency", "TTFT (ms, log)"),
        (axes[1], "itl_ms", "Inter-token latency vs concurrency", "ITL (ms, log)"),
    ):
        style(ax, title, "concurrent requests", ylabel)
        for v in vs:
            cs = [c for c in (1, 8, 32, 64) if s6_reps(v, f"c{c}")]
            if not cs:
                continue
            p50 = [s6_mean(s6_reps(v, f"c{c}"), f"median_{key}") for c in cs]
            p99 = [s6_mean(s6_reps(v, f"c{c}"), f"p99_{key}") for c in cs]
            ax.plot(cs, p50, "-o", color=S6_COLORS[v], lw=1.8, ms=4, label=f"{S6_LABEL[v]} p50")
            ax.plot(
                cs, p99, "--", color=S6_COLORS[v], lw=1.0, alpha=0.7, label=f"{S6_LABEL[v]} p99"
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 64], ["1", "8", "32", "64"])
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    out_latency.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_latency, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out_latency}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(
        axes[0], "Throughput vs offered load (Poisson)", "offered requests / s", "output tokens / s"
    )
    style(axes[1], "Tail latency vs offered load", "offered requests / s", "TTFT p99 (ms)")
    for v in vs:
        rs = [r for r in (1, 4, 16) if s6_reps(v, f"r{r}")]
        if not rs:
            continue
        tput = [s6_mean(s6_reps(v, f"r{r}"), "output_throughput") for r in rs]
        p99 = [s6_mean(s6_reps(v, f"r{r}"), "p99_ttft_ms") for r in rs]
        axes[0].plot(rs, tput, "-o", color=S6_COLORS[v], lw=1.8, ms=4, label=S6_LABEL[v])
        axes[1].plot(rs, p99, "-o", color=S6_COLORS[v], lw=1.8, ms=4, label=S6_LABEL[v])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], ["1", "4", "16"])
    axes[1].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_load, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out_load}")


def s6_deploy_blocks(usd: float) -> dict[str, str]:
    """DEPLOY.md's generated blocks: the same tables as the README's Stage 6 section."""
    gate, verdict = s6_gate()
    rows = [
        f"| {S6_LABEL[v]} | {'reference' if v == 'bf16' else verdict.get(v, 'not run')} |"
        for v in S6_RUN
        if v == "bf16" or v in verdict
    ]
    variants = "| variant | quality gate |\n|---|---|\n" + "\n".join(rows) if verdict else ""
    blocks = {
        "deploy-variants": variants,
        "deploy-gate": gate,
        "deploy-memory": s6_memory_table(),
        "deploy-latency": s6_latency_table(),
        "deploy-cost": s6_cost_table(usd, verdict),
    }
    return {k: v for k, v in blocks.items() if v}


def s6_repeat_note() -> str:
    """The FP8-KV variant's second domain_qa generation (dpo-strict-fp8kv-r2, another container):
    reported next to the registered reading, never in place of it."""
    if not (RUNS / "dpo-strict-fp8kv-r2/generations.jsonl").exists():
        return ""
    from qa_strict import run_strict

    ref = load_metrics(S6_RUN["bf16"]) or {}
    first = load_metrics(S6_RUN["fp8kv"]) or {}
    second = run_strict("dpo-strict-fp8kv-r2")
    if not all(k in d for d, k in ((ref, "qa_strict_seen"), (first, "qa_strict_seen"))):
        return ""
    d1 = (first["qa_strict_seen"] - ref["qa_strict_seen"]) * 100
    d2 = (second["seen"]["strict"] - ref["qa_strict_seen"]) * 100
    return (
        f"FP8 + FP8 KV, strict closed-book on the seen half, read twice: {d1:+.1f} points in the "
        f"registered generation, {d2:+.1f} in a second one on another container "
        f"(dpo-strict-fp8kv-r2, {second['seen']['n']} items). The verdict is the registered "
        "reading's; the second is the line's run-to-run spread, reported."
    )


def stage6_md(usd: float) -> str:
    gate, verdict = s6_gate()
    md = ""
    if gate:
        md += (
            "## Quality gate (pre-registered): change against bf16, next to the Stage 3 floor\n\n"
            + gate
            + "\n\n"
            "A variant ships if every gated line is within its floor (bold: beyond it) and the eos "
            "job ends >= 95% of answers on </s> (notes/decisions.md, 2026-10-09). The floor is the "
            "sft-from-cpt seed gap or the SE, whichever is larger: a cost under it is invisible to "
            "every other comparison here, which is what ships means, not that it costs nothing. "
            "GSM8K is the gate's own run (all 1,319, 5-shot, add_bos_token=True, bf16 rerun under "
            "the same flags), not the table's frozen-flag row.\n"
        )
        if note := s6_repeat_note():
            md += f"\n{note}\n"
    sections = [
        ("Serving memory (vLLM's own accounting at start-up)", s6_memory_table(), ""),
        ("Latency and throughput vs concurrency (`unique`: 360 eval requests, 60 / 30 / 10)",
         s6_latency_table(),
         ("Means of two runs per config, prefix cache reset before each; goodput share = requests "
          "with TTFT <= 500 ms and TPOT <= 25 ms. ITL at concurrency 1 under the decode floor "
          "(weight bytes / 3.35 TB/s: bf16 4.75, FP8 2.5, INT4 1.5 ms) would mean the run wasn't "
          "on an H100.")),
        ("Latency under load (Poisson arrivals, `unique`)", s6_load_table(), ""),
        ("Prefix caching: shared retrieval context (FP8, concurrency 8)", s6_prefix_table(),
         ("Same 108 questions in the same order; rag prompts are ~9% longer (gold passages run "
          "long), which counts against rag. TTFT is the comparison; output lengths differ.")),
        ("n-gram speculative decoding (FP8)", s6_spec_table(), ""),
        ("Cost", s6_cost_table(usd, verdict), ""),
    ]  # fmt: skip
    for title, body, note in sections:
        if body:
            md += f"\n## {title}\n\n{body}\n" + (f"\n{note}\n" if note else "")
    return md


# ------------------------------------------------------------------------- README headline ---
# Every stage and ablation with a row, in lifecycle order (base first), and what each row is. The
# Stage 0 base-8b (vLLM's Mistral-native path) is superseded by base-8b-hf; it stays in
# docs/results.md. cpt-8b-lr2x and cpt-8b-fsdp2 were training-only runs (docs/stage2.md).
HEADLINE_ROWS = [
    ("base-8b-hf", "Ministral 3 8B Base"),
    ("instruct-8b", "stock Instruct (comparison)"),
    ("mistral-large-3", "frontier reference (API, closed-book only)"),
    ("cpt-8b", "CPT"),
    ("cpt-8b-seed1", "CPT, seed 1"),
    ("cpt-8b-replay10", "CPT + 10% replay (shipped)"),
    ("cpt-8b-full", "CPT, all weights (ablation)"),
    ("sft-from-base", "SFT without CPT (control)"),
    ("sft-from-base-seed1", "SFT without CPT, seed 1"),
    ("sft-from-cpt", "SFT after CPT (shipped)"),
    ("sft-from-cpt-seed1", "SFT after CPT, seed 1"),
    ("dpo", "DPO, original labels"),
    ("dpo-seed1", "DPO, original labels, seed 1"),
    ("dpo-2ep", "DPO, 2 epochs (ablation; failed the merge check)"),
    ("dpo-strict", "DPO, corrected labels (shipped, bf16)"),
    ("grpo", "GRPO, step 25 (not shipped)"),
    ("grpo-seed1", "GRPO, seed 1"),
    ("dpo-strict-fp8", "shipped model in FP8 (passed the quality check)"),
    ("dpo-strict-fp8kv", "FP8 + FP8 KV cache (failed the quality check)"),
    ("dpo-strict-w4a16", "INT4 W4A16 (failed the quality check)"),
]
BASE_FORMAT = {"base-8b-hf", "cpt-8b", "cpt-8b-seed1", "cpt-8b-replay10", "cpt-8b-full"}
BLANKS = {  # block -> (runs, why their cells are blank), printed under the table
    "headline-knowledge": [
        (
            "cpt-8b-full",
            (
                "has no numbers: its weights were deleted before the closed-book task grew, "
                "and full-parameter weights can't be rebuilt from an adapter"
            ),
        ),
        (
            "mistral-large-3",
            (
                "has no `gold_lp`, which is computed on the weights in the generation engine; "
                "Large 3 ran through the API"
            ),
        ),
    ],
    "headline-behaviour": [
        ("mistral-large-3", "has no numbers: it is an API reference, run closed-book only")
    ],
    "headline-general": [
        ("mistral-large-3", "has no numbers: it is an API reference, run closed-book only"),
        ("instruct-8b", "has no perplexity: it wasn't measured"),
        (
            "dpo-strict-fp8, dpo-strict-fp8kv, dpo-strict-w4a16",
            (
                "have no lm-eval or perplexity run with the "
                "settings the other rows used; their quality-check GSM8K (one BOS) and perplexity are in "
                "[`docs/stage6.md`](docs/stage6.md)"
            ),
        ),
        (
            "",
            (
                "latency is measured per deployed checkpoint, not per seed or ablation; FP8-KV's and "
                "INT4's bench rows weren't run (Modal spend limit)"
            ),
        ),
    ],
}
FLOOR_ROWS = [  # (label, the seed pair whose gap, or the metric's SE, sets each column's floor)
    ("floor, base-format rows: two CPT seeds' gap, or SE", ("cpt-8b", "cpt-8b-seed1")),
    ("floor, chat rows: two SFT seeds' gap, or SE", SFT_SEED_PAIR),
]
HEADLINE = {  # block -> [(header, metrics key, kind, noise() extra)]; rates in %, gold_lp in nats
    "headline-knowledge": [
        ("closed-book seen (strict)", "qa_strict_seen", "kpi", "qa_seen"),
        ("closed-book unseen (strict)", "qa_strict_unseen", "kpi", "qa_unseen"),
        ("identifiers (strict)", "qa_strict_ident", "kpi", "qa_identifier"),
        ("gold_lp answer, seen", "gold_lp_ans_seen", "lp_ans", "seen"),
        ("gold_lp answer, unseen", "gold_lp_ans_unseen", "lp_ans", "unseen"),
        ("gold_lp end, seen", "gold_lp_end_seen", "lp_end", "seen"),
        ("gold_lp end, unseen", "gold_lp_end_unseen", "lp_end", "unseen"),
    ],
    "headline-behaviour": [
        ("grounded_acc (judge)", "grounded_acc", "kpi", "grounded"),
        ("cite_valid", "cite_valid", "kpi", "grounded"),
        ("cite_supported (judge, reported)", "cite_supported", "kpi", "grounded"),
        ("halluc_rate ↓", "halluc_rate", "kpi", "adversarial"),
        ("false_abstain ↓", "false_abstain", "kpi", "grounded"),
        ("vocab_recall (judge)", "vocab_recall", "kpi", "vocab"),
    ],
    "headline-general": [
        ("MMLU (no BOS)", "mmlu", "lm", ("mmlu", "acc_stderr,none")),
        ("GSM8K (no BOS)", "gsm8k", "lm", ("gsm8k", "exact_match_stderr,strict-match")),
        ("HellaSwag (no BOS)", "hellaswag", "lm", ("hellaswag", "acc_norm_stderr,none")),
        ("ppl domain val ↓", "ppl_domain_val", "ppl", None),
        ("ppl general val ↓", "ppl_general_val", "ppl", None),
    ],
}
SUMMARY = [  # the README's first table: the chain's checkpoints against the base and the bar
    ("base-8b-hf", "Ministral 3 8B Base"),
    ("instruct-8b", "stock Instruct"),
    ("sft-from-base", "SFT without CPT (control)"),
    ("sft-from-cpt", "CPT → SFT"),
    ("dpo-strict", "CPT → SFT → DPO (shipped)"),
    ("grpo", "CPT → SFT → DPO → GRPO (not shipped)"),
]
SUMMARY_COLS = [
    *HEADLINE["headline-knowledge"][:2],
    HEADLINE["headline-knowledge"][4],
    HEADLINE["headline-behaviour"][0],
    HEADLINE["headline-behaviour"][3],
    HEADLINE["headline-general"][1],
]


def headline_latency(run: str) -> tuple[str, str, str]:
    """(TTFT p50, ITL p50 at one request, source). Stage 6 measured dpo-strict (bf16) and its
    variants on a pinned H100 (results/serve/bench, mean of the repeats); earlier rows come from
    serve/bench_latency.py, one sample on an unrecorded GPU (results/bench). The two benches send
    different prompts, so compare rows within a source only."""
    variant = {r: v for v, r in S6_RUN.items()}.get(run)
    if variant and (reps := s6_reps(variant, "c1")):
        ttft, itl = s6_mean(reps, "median_ttft_ms"), s6_mean(reps, "median_itl_ms")
        return f1(ttft), f1(itl, "{:.2f}"), "H100, serving benchmark"
    f = Path("results/bench") / f"{run}.json"
    if f.exists():
        d = {x["concurrency"]: x for x in json.loads(f.read_text())}[1]
        return f"{d['ttft_p50_ms']:.1f}", f"{d['itl_p50_ms']:.2f}", "1 sample, GPU not recorded"
    return "", "", ""


def headline_cell(m: dict, key: str, kind: str) -> str:
    v = m.get(key)
    if v is None:
        return ""
    if kind in LP_PART:
        return f"{v:.2f}"
    return f"{v:.2f}" if kind == "ppl" else f"{v * 100:.1f}"


def headline_floor(m: dict, pair: tuple[str, str], key: str, kind: str, extra, sizes) -> str:
    """noise() for every kind but perplexity, whose floor is the seed pair's relative gap in %."""
    if kind == "ppl":
        a, b = (m.get(r, {}).get(key) for r in pair)
        return "" if a is None or b is None else f"{abs(a / b - 1) * 100:.2f}%"
    n = noise(m, pair, key, kind, extra, sizes)
    return "" if n is None else (f"{n:.2f}" if kind in LP_PART else f"{n:.1f}")


def headline_tables() -> dict[str, str]:
    """The README's results: three tables with the same rows (knowledge, behaviour, general and
    serving), each followed by one floor row per format group (base-format and chat-format gold_lp
    don't compare, so no pooled floor), and the summary table of part 1. Strict closed-book only:
    the lenient columns are in docs/results.md."""
    m = {r: v for r, _ in HEADLINE_ROWS if (v := load_metrics(r)) is not None}
    sizes = half_sizes()

    def table(rows, cols, latency: bool, name: str = "") -> str:
        head = ["run", "what it is", *(c[0] for c in cols)]
        if latency:
            head += ["TTFT p50 @1 (ms)", "ITL p50 @1 (ms)", "latency source"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        empty = [r for r, _ in rows if r not in m]
        for run, what in rows:
            if run not in m:
                continue
            fmt = " [base format]" if run in BASE_FORMAT else ""
            values = [headline_cell(m[run], k, kind) for _, k, kind, _ in cols]
            if latency:
                values += headline_latency(run)
            if not any(values):  # a row with no number in this table (left out, named below)
                empty.append(run)
                continue
            lines.append("| " + " | ".join([f"`{run}`", what + fmt, *values]) + " |")
        for label, pair in FLOOR_ROWS:
            if rows is SUMMARY and pair != SFT_SEED_PAIR:
                continue
            cells = [
                "",
                f"*{label}*",
                *(headline_floor(m, pair, k, kind, x, sizes) for _, k, kind, x in cols),
            ]
            if latency:
                cells += ["", "", ""]
            lines.append("| " + " | ".join(cells) + " |")
        notes = BLANKS.get(name, [])
        named = {r for runs, _ in notes for r in runs.split(", ")}
        if missing := [r for r in empty if r not in named]:
            notes = [
                *notes,
                (", ".join(missing), "have no numbers" if len(missing) > 1 else "has no numbers"),
            ]
        if notes:
            lines.append("\n**Blank cells:**")
            for runs, why in notes:
                who = ", ".join(f"`{r}`" for r in runs.split(", ")) if runs else ""
                lines.append(f"- {who} {why}." if who else f"- {why[0].upper()}{why[1:]}.")
        return "\n".join(lines)

    blocks = {
        name: table(HEADLINE_ROWS, cols, name == "headline-general", name)
        for name, cols in HEADLINE.items()
    }
    blocks["headline-summary"] = table(SUMMARY, SUMMARY_COLS, False)
    return blocks


SUMMARY_STATS = Path("results/summary_stats.json")  # eval/summary_stats.py
REPRODUCE = [  # (stage, make target, its results/train_runs.md section, data: a SHA256SUMS dir or a note)
    ("0: the eval and baselines", "reproduce-stage0", None, "`eval/tasks/` (committed, reviewed)"),
    ("1: the corpus", "reproduce-stage1", None, "data/processed"),
    ("2: CPT", "reproduce-stage2", "Stage 2: CPT", "the corpus"),
    ("3: SFT", "reproduce-stage3", "Stage 3: SFT", "data/sft"),
    ("4: DPO", "reproduce-stage4", "Stage 4: DPO", "data/dpo/strict"),
    ("5: GRPO", "reproduce-stage5", "Stage 5: GRPO", "data/grpo"),
    ("6: serving", "reproduce-stage6", None, "`serve/bench_manifest.json`"),
]
REPLAY_LICENCES = {  # Tülu 3 subset -> (name, the licence the mixture's card gives it)
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": ("Evol CodeAlpaca", "Apache 2.0"),
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": ("NuminaMath-TIR", "Apache 2.0"),
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": (
        "WildGuardMix",
        "Apache 2.0",
    ),
    "ai2-adapt-dev/oasst1_converted": ("OASST", "Apache 2.0"),
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k": ("WildJailbreak", "ODC-BY-1.0"),
    "allenai/tulu-3-sft-personas-math-grade": ("Persona GSM", "ODC-BY-1.0"),
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": ("WildChat (GPT-4)", "ODC-BY-1.0"),
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": ("Persona Algebra", "ODC-BY-1.0"),
    "ai2-adapt-dev/coconot_converted": ("CoCoNot", "ODC-BY-1.0"),
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k": ("SciRIFF", "ODC-BY-1.0"),
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k": ("TableGPT", "MIT"),
    "ai2-adapt-dev/flan_v2_converted": ("FLAN v2", "not given on the card"),
    "withdrawn": (
        "Withdrawn math set",
        "withdrawn after training: its licence restricts models trained on it",
    ),
    "ai2-adapt-dev/no_robots_converted": ("No Robots", "CC-BY-NC-4.0 (non-commercial)"),
}


def reproduce_table() -> str:
    """One row per stage: its make target, its training GPU-hours (summed from the training-run
    tables by eval/summary_stats.py) and its data set's hash, read live: the sha256 of the set's
    SHA256SUMS, which lists every file's own."""
    if not SUMMARY_STATS.exists():
        return ""
    cost = json.loads(SUMMARY_STATS.read_text())["training_cost"]
    lines = [
        "| stage | command | training GPU-h | data (sha256 of its `SHA256SUMS`) |",
        "|---|---|---|---|",
    ]
    for stage, target, section, data in REPRODUCE:
        c = cost.get(section) if section else None
        sums = Path(data) / "SHA256SUMS"
        if sums.exists():
            data = f"`{data}`: {hashlib.sha256(sums.read_bytes()).hexdigest()[:8]}"
        gpu = f"{c['gpu_h']:.2f}" if c else ("not totalled" if stage.startswith("6") else "")
        lines.append(f"| {stage} | `make {target}` | {gpu} | {data} |")
    lines.append(f"| all training | | {cost['all training']['gpu_h']:.2f} | |")
    return "\n".join(lines)


def replay_licence_table() -> str:
    """The Tülu 3 replay records by subset (data/sft/hosted/, counted by eval/summary_stats.py),
    with the licence the mixture's card gives each."""
    if not SUMMARY_STATS.exists():
        return ""
    counts = json.loads(SUMMARY_STATS.read_text())["licence_audit"]["replay_by_subset"]
    unknown = set(counts) - set(REPLAY_LICENCES)
    if unknown:
        raise SystemExit(f"replay subsets without a licence entry: {sorted(unknown)}")
    lines = ["| subset | records | licence |", "|---|---|---|"]
    for sub, (name, licence) in REPLAY_LICENCES.items():
        if counts.get(sub):
            lines.append(f"| {name} | {counts[sub]} | {licence} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-gpu-hour", type=float, default=3.95)
    ap.add_argument("--scaling-run", default="cpt-8b-fsdp2")
    ap.add_argument("--scaling-ref", default="cpt-8b")
    ap.add_argument(
        "--no-docs", action="store_true", help="leave README.md, docs/ and DEPLOY.md alone"
    )
    args = ap.parse_args()
    every = {d.name: r for d in sorted(RUNS.iterdir()) if d.is_dir() and (r := load(d))}
    if not every:
        raise SystemExit(f"no train_summary.json + train_log.jsonl under {RUNS}")
    # Stage 2 is every run without a later stage's tag (GRPO's section is written with its results)
    runs = {k: v for k, v in every.items() if v[0].get("stage") not in ("sft", "dpo", "grpo")}
    sft = {k: v for k, v in every.items() if v[0].get("stage") == "sft"}
    sft_runs = {r: sft[r] for r in [*[r for r in SFT_COLORS if r in sft], *sorted(sft)]}
    dpo = {k: v for k, v in every.items() if v[0].get("stage") == "dpo"}
    dpo_runs = {r: dpo[r] for r in [*[r for r in DPO_COLORS if r in dpo], *sorted(dpo)]}
    grpo = {k: v for k, v in every.items() if v[0].get("stage") == "grpo"}
    grpo_runs = {r: grpo[r] for r in [*[r for r in GRPO_COLORS if r in grpo], *sorted(grpo)]}
    order = [*[r for r in COLORS if r in runs], *[r for r in runs if r not in COLORS]]
    runs = {r: runs[r] for r in order}
    md = "## Training runs\n\n" + table(runs, args.usd_per_gpu_hour)
    if cmp := scaling(runs, args.scaling_run, args.scaling_ref):
        md += "\n\n" + cmp
    md += (
        f"\n\n$ at {args.usd_per_gpu_hour} per GPU-hour (Modal's H100 SXM5 list price, checked "
        "2026-10-09); wall time includes tokenising and model load.\n"
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
            f"\n## Change vs {REF}, next to the noise\n\n{deltas}\n\nlm-eval rows run 5-shot "
            "without a BOS token (the frozen flags send none, found in Stage 6): the changes stand, "
            "absolute values aren't comparable with published scores. Perplexity in %, the "
            "gold-answer log-probability in nats per answer, the rest in points. noise = max(the "
            "seed gap cpt-8b vs cpt-8b-seed1, the metric's standard error: for base-8b-hf, or for "
            "the log-probability the paired per-item difference): a change smaller than it is not "
            "a result. QA rows are on the 322-item domain_qa (eval v3; Stage 2 was first read on "
            "v2's 325, results/table_v2.md), so cpt-8b-full, whose weights were deleted, has "
            "none.\n"
        )
    md3 = stage3_md(sft_runs, args.usd_per_gpu_hour) if sft_runs else ""
    md4 = stage4_md(dpo_runs, args.usd_per_gpu_hour) if dpo_runs else ""
    md5 = stage5_md(grpo_runs, args.usd_per_gpu_hour) if grpo_runs else ""
    md6 = stage6_md(args.usd_per_gpu_hour)
    Path("results/train_runs.md").write_text(
        "# Stage 2: CPT\n\n"
        + md
        + ("\n# Stage 3: SFT\n\n" + md3 if md3 else "")
        + ("\n# Stage 4: DPO\n\n" + md4 if md4 else "")
        + ("\n# Stage 5: GRPO\n\n" + md5 if md5 else "")
        + ("\n# Stage 6: serving\n\n" + md6 if md6 else "")
    )
    print(md + md3 + md4 + md5 + md6)
    if not args.no_docs:
        blocks = {
            "stage2-tables": md,
            "stage3-tables": md3,
            "stage4-tables": md4,
            "stage5-tables": md5,
            "stage6-tables": md6,
            "results-table": results_table(),
            "serving-table": serving_table(),
            **headline_tables(),
            "reproduce-table": reproduce_table(),
            "replay-licences": replay_licence_table(),
        }
        for name, block in blocks.items():
            if block:
                update_block(Path(BLOCK_FILES[name]), name, block)
        if md6:
            for name, block in s6_deploy_blocks(args.usd_per_gpu_hour).items():
                update_block(Path("DEPLOY.md"), name, block)
    base_val = math.log(ppl[BASE]["ppl_val_slice"]) if BASE in ppl else None
    plot_loss(runs, base_val, Path("results/curves/cpt.png"))
    if BASE in ppl and len(ppl) > 1:
        plot_ppl(ppl, Path("results/curves/cpt_ppl.png"))
    if sft_runs:
        plot_sft_loss(sft_runs, Path("results/curves/sft.png"))
        plot_sft_kpi(Path("results/curves/sft_kpi.png"))
    if dpo_runs:
        plot_dpo(dpo_runs, Path("results/curves/dpo.png"))
    if grpo_runs:
        plot_grpo(grpo_runs, Path("results/curves/grpo.png"))
        plot_passk(Path("results/curves/passk.png"))
    plot_serve(Path("results/curves/serve_latency.png"), Path("results/curves/serve_load.png"))


if __name__ == "__main__":
    main()
