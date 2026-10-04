"""Was the corpus in the base model's pretraining data? A memorisation probe plus perplexity, by
document group: corpus documents (which the model may have read) against documents published after
its release (which it can't have), with the CPT checkpoint as the positive control (it read the
train documents once, the val documents never).

  # Mac: PDFs listed in eval/exposure_sources.csv, saved as data/exposure/<slug>.pdf
  .venv/bin/python eval/memorization.py build            # -> data/exposure/docs.jsonl
  modal volume put --force struct-lm data/exposure/docs.jsonl data/exposure/docs.jsonl
  modal run eval/modal_app.py::memorization --model /vol/checkpoints/base-8b-hf --run-name base-8b-hf

build: every train and val document from data/processed, plus the post-cutoff PDFs through the
corpus's own cleaning (extract.py's text, then filter.py's paragraph rules and reflow; no dedup or
PII pass, which change almost nothing), so all three groups look alike to the model.

run: per document, SPANS spans spread evenly over its middle 80% (front and back matter, where the
publisher's standard notices live, skipped). The model gets PROMPT tokens of the real text and
greedily writes TARGET more. Per span: the verbatim prefix (how many generated tokens reproduce
the real continuation before the first miss) and the share of positions that match; a span is
recalled when its verbatim prefix is at least RECALL tokens. Text a model trained on comes back
verbatim more often than text it never saw; formulaic passages (tables, standard clauses) come
back for both, so the comparison is between groups, not against zero. Per document too:
perplexity over its first PPL_WINDOWS windows (BOS + text + EOS, as in training), from vLLM's
prompt logprobs (the same numbers as transformers once the YaRN fix is in; eval/vllm_ppl.py).
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))  # repo or /root/train
from packing import SEQ_LEN, encoder

BASE = "mistralai/Ministral-3-8B-Base-2512"
SOURCES = ROOT / "eval/exposure_sources.csv"
DOCS = "data/exposure/docs.jsonl"
SPANS, PROMPT, TARGET, RECALL = 16, 64, 64, 32
PPL_WINDOWS = 2


def build(out: str) -> None:
    sys.path.insert(0, str(ROOT / "data/scripts"))
    from common import load_tokenizer, read_jsonl, split_pages
    from exposure_check import pdf_year
    from extract import extract_doc
    from filter import paragraph_rule

    encode, _ = load_tokenizer(BASE)
    # corpus year: the PDF's creationDate, which flags the 2026 revisions already in the corpus
    # (NASA-STD-5002B, FEMA P-58), not a clean post-cutoff control since earlier editions exist
    docs = [
        {
            "slug": d["slug"],
            "publisher": d["publisher"],
            "group": group,
            "year": pdf_year(d["slug"]),
            "text": d["text"],
        }
        for group in ("train", "val")
        for d in read_jsonl(f"data/processed/{group}.jsonl")
    ]
    for src in csv.DictReader(SOURCES.open()):
        raw, info = extract_doc(src, encode, Path(out).parent / f"{src['slug']}.pdf")
        if info["copyright_flag"]:  # rule 1: hand-check before keeping
            print(f"copyright flag: {src['slug']}")
        paras = [p for _, ps in split_pages(raw) for p in ps if not paragraph_rule(p)]
        text = "\n\n".join(" ".join(p.split("\n")) for p in paras)
        docs.append(
            {
                "slug": src["slug"],
                "publisher": src["publisher"],
                "group": "post-cutoff",
                "year": int(src["year"]),
                "text": text,
            }
        )
        print(f"{src['slug']}: {raw['n_tokens']} tokens extracted, {len(encode(text))} kept")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    # The post-cutoff documents alone, in the corpus schema, for eval/perplexity.py's
    # ppl_postcutoff (whole documents, the same windows as ppl_domain_val)
    post = [
        {
            "slug": d["slug"],
            "publisher": d["publisher"],
            "text": d["text"],
            "n_tokens": len(encode(d["text"])),
        }
        for d in docs
        if d["group"] == "post-cutoff"
    ]
    with open(Path(out).parent / "postcutoff.jsonl", "w") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in post)
    counts = {g: sum(d["group"] == g for d in docs) for g in ("train", "val", "post-cutoff")}
    print(f"{out}: {counts}")


def spans(ids: list[int]) -> list[tuple[list[int], list[int]]]:
    lo, hi = int(0.1 * len(ids)), int(0.9 * len(ids)) - PROMPT - TARGET
    if hi <= lo:
        return []
    starts = np.linspace(lo, hi, SPANS).astype(int)
    return [(ids[s : s + PROMPT], ids[s + PROMPT : s + PROMPT + TARGET]) for s in starts]


def summarise(rows: list[dict]) -> dict:
    rs = [r for r in rows if r["spans"]]
    prefixes = [p for r in rs for p in r["prefixes"]]
    return {
        "docs": len(rs),
        "spans": len(prefixes),
        "recall_rate": round(float(np.mean([p >= RECALL for p in prefixes])), 4),
        "mean_verbatim_prefix": round(float(np.mean(prefixes)), 2),
        "mean_token_match": round(float(np.mean([r["mean_token_match"] for r in rs])), 4),
        "median_ppl": round(float(np.median([r["ppl"] for r in rs])), 3),
    }


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    encode, bos, eos = encoder(args.model)
    docs = [json.loads(line) for line in Path(args.docs).read_text().splitlines()]
    for d in docs:
        d["ids"] = encode(d["text"])
    llm = LLM(
        model=args.model,
        tokenizer_mode="mistral",
        config_format=args.config_format,
        load_format="mistral" if args.config_format == "mistral" else "auto",
        limit_mm_per_prompt={"image": 0},
        dtype="bfloat16",
        max_model_len=SEQ_LEN + 8,
        gpu_memory_utilization=0.9,
        seed=0,
    )

    # memorisation: greedy continuations of every span of every document, one batch
    prompts, owners = [], []
    for i, d in enumerate(docs):
        for p, t in spans(d["ids"]):
            prompts.append(TokensPrompt(prompt_token_ids=[bos, *p]))
            owners.append((i, t))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=TARGET))
    per_doc = defaultdict(list)
    for (i, target), out in zip(owners, outs):
        gen = list(out.outputs[0].token_ids)
        prefix = next((k for k, (a, b) in enumerate(zip(gen, target)) if a != b), len(gen))
        per_doc[i].append((prefix, sum(a == b for a, b in zip(gen, target)) / TARGET))

    # perplexity over each document's first windows
    wins, wowner = [], []
    for i, d in enumerate(docs):
        ids = [bos, *d["ids"], eos]
        for w in range(max(1, min(PPL_WINDOWS, len(ids) // SEQ_LEN))):
            wins.append(ids[w * SEQ_LEN : (w + 1) * SEQ_LEN])
            wowner.append(i)
    louts = llm.generate(
        [TokensPrompt(prompt_token_ids=w) for w in wins],
        SamplingParams(max_tokens=1, prompt_logprobs=0),
    )
    nll, cnt = defaultdict(float), defaultdict(int)
    for i, w, out in zip(wowner, wins, louts):
        for pos in range(1, len(w)):  # position 0 has no logprob (nothing predicts it)
            nll[i] -= out.prompt_logprobs[pos][w[pos]].logprob
            cnt[i] += 1

    rows = []
    for i, d in enumerate(docs):
        s = per_doc.get(i, [])
        rows.append(
            {
                "slug": d["slug"],
                "publisher": d["publisher"],
                "group": d["group"],
                "year": d.get("year"),
                "spans": len(s),
                "prefixes": [p for p, _ in s],
                "recall_rate": round(float(np.mean([p >= RECALL for p, _ in s])), 4) if s else None,
                "mean_token_match": round(float(np.mean([m for _, m in s])), 4) if s else None,
                "ppl": round(math.exp(nll[i] / cnt[i]), 4),
            }
        )
    groups = {
        g: summarise([r for r in rows if r["group"] == g]) for g in ("train", "val", "post-cutoff")
    }
    result = {
        "model": args.model,
        "run_name": args.run_name,
        "settings": {
            "spans": SPANS,
            "prompt": PROMPT,
            "target": TARGET,
            "recall": RECALL,
            "ppl_windows": PPL_WINDOWS,
        },
        "groups": groups,
        "docs": rows,
    }
    out = Path(args.out_dir) / f"{args.run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(groups, indent=2))


def boot(x: np.ndarray, y: np.ndarray | None = None, n: int = 10_000) -> tuple[float, float, float]:
    """Mean of x (or mean(x) - mean(y)), with a 95% interval from resampling documents."""
    rng = np.random.default_rng(0)
    stat = lambda a, b: a.mean() - (b.mean() if b is not None else 0)
    draws = [
        stat(
            x[rng.integers(0, len(x), len(x))],
            None if y is None else y[rng.integers(0, len(y), len(y))],
        )
        for _ in range(n)
    ]
    return float(stat(x, y)), *np.percentile(draws, [2.5, 97.5]).tolist()


def report(args: argparse.Namespace) -> None:
    """Base: corpus documents against post-cutoff ones. Control: the base -> CPT change on the
    train documents (read once in CPT) against the val documents (never read). Per document:
    mean verbatim prefix (tokens reproduced before the first miss; more sensitive than the recall
    rate when little is recalled) and perplexity. Intervals resample documents."""
    runs = {r: json.loads((Path(args.out_dir) / f"{r}.json").read_text()) for r in args.runs}
    base, *others = args.runs
    prefix = lambda r: float(np.mean(r["prefixes"]))
    ci = lambda m, lo, hi, f="+.2f": f"{m:{f}} [{lo:{f}}, {hi:{f}}]"

    def split(rows: list[dict]) -> dict[str, list[dict]]:
        out = defaultdict(list)
        for r in rows:
            g = r["group"]
            if g != "post-cutoff" and (r["year"] or 0) >= 2026:
                g = "corpus-2026-revision"  # a new edition of an older document: not unseen
            out[g].append(r)
        return {g: [r for r in rs if r["spans"]] for g, rs in out.items()}

    for name, res in runs.items():
        print(f"\n{name}  (recall: verbatim prefix >= {RECALL} of {TARGET} tokens)")
        print(
            f"{'group':21} {'docs':>4} {'spans':>5} {'recalled':>8}  {'mean prefix (95% CI)':>21}  ppl"
        )
        for g, rs in split(res["docs"]).items():
            recalled = sum(p >= RECALL for r in rs for p in r["prefixes"])
            m = boot(np.array([prefix(r) for r in rs]))
            ppl = np.median([r["ppl"] for r in rs])
            n = sum(r["spans"] for r in rs)
            print(f"{g:21} {len(rs):4d} {n:5d} {recalled:8d}  {ci(*m, '.2f'):>21}  {ppl:.2f}")

    groups = split(runs[base]["docs"])
    corpus = [r for g in ("train", "val") for r in groups[g]]
    post = groups["post-cutoff"]
    m = boot(np.array([prefix(r) for r in corpus]), np.array([prefix(r) for r in post]))
    print(f"\n{base}: corpus - post-cutoff mean prefix = {ci(*m)} tokens")
    logp = lambda rs: np.log([r["ppl"] for r in rs])
    m = boot(logp(corpus), logp(post))
    print(
        f"{base}: corpus / post-cutoff ppl = {ci(*(np.expm1(m) * 100), '+.1f')}% (unmatched types)"
    )
    for pub in sorted({r["publisher"] for r in post}):
        c, q = (
            [r for r in corpus if r["publisher"] == pub],
            [r for r in post if r["publisher"] == pub],
        )
        print(
            f"  {pub:6} prefix {np.mean([prefix(r) for r in c]):.2f} vs {np.mean([prefix(r) for r in q]):.2f}"
            f"  median ppl {np.median([r['ppl'] for r in c]):.2f} vs {np.median([r['ppl'] for r in q]):.2f}"
            f"  (n={len(c)} vs {len(q)})"
        )

    for other in others:
        o = {r["slug"]: r for r in runs[other]["docs"] if r["spans"]}
        print(f"\n{other} - {base}, per-document change:")
        d_prefix, d_ppl = {}, {}
        for g in ("train", "val", "post-cutoff"):
            rs = [r for r in runs[base]["docs"] if r["spans"] and r["group"] == g]
            d_prefix[g] = np.array([prefix(o[r["slug"]]) - prefix(r) for r in rs])
            d_ppl[g] = np.log([o[r["slug"]]["ppl"] / r["ppl"] for r in rs])
            dp, dq = boot(d_prefix[g]), np.expm1(boot(d_ppl[g])) * 100
            print(f"  {g:12} prefix {ci(*dp)} tokens   ppl {ci(*dq, '+.1f')}%   (n={len(rs)})")
        dp = boot(d_prefix["train"], d_prefix["val"])
        dq = np.expm1(boot(d_ppl["train"], d_ppl["val"])) * 100
        print(f"  train - val  prefix {ci(*dp)} tokens   ppl {ci(*dq, '+.1f')}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=DOCS)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--run-name", required=True)
    r.add_argument("--config-format", default="hf", choices=("auto", "hf", "mistral"))
    r.add_argument("--docs", default=DOCS)
    r.add_argument("--out-dir", default="results/exposure")
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+", help="base run first, e.g. base-8b-hf cpt-8b-replay10")
    p.add_argument("--out-dir", default="results/exposure")
    args = ap.parse_args()
    {"build": lambda: build(args.out), "run": lambda: run(args), "report": lambda: report(args)}[
        args.cmd
    ]()


if __name__ == "__main__":
    main()
