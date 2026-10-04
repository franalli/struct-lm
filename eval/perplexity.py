"""Perplexity columns for results/table.md: results/ppl/<run>.json.

  python eval/perplexity.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b
  python eval/perplexity.py --model checkpoints/cpt-8b --run-name cpt-8b   # merged checkpoint
  (on Modal: train/modal_train.py --steps ppl, which runs this with cwd=/vol)

Windows come from train/packing.py, the ones cpt.py trains and evaluates on (BOS + document +
EOS, concatenated, 4,096-token windows), so the numbers compare across checkpoints and with the
trainer's eval_loss. Each is exp(mean NLL per predicted token), token-weighted:

  ppl_train        SLICE_WINDOWS windows spread across train.jsonl (~200k tokens a CPT run has seen
                   once; for the base, just more domain text)
  ppl_domain_val   all of val.jsonl, the held-out documents; by_publisher splits it by the
                   publisher of each predicted token (NASA is one 18.7k-token document: noisy)
  ppl_val_slice    the SLICE_WINDOWS val windows cpt.py evaluates on: should match
                   exp(final eval_loss) of the run to within ~1% (a merge check)
  ppl_general_val  general_val.jsonl, held-out FineWeb-Edu (the forgetting side)
  ppl_postcutoff   data/exposure/postcutoff.jsonl, the 13 federal reports published after the base
                   model (eval/exposure_sources.csv, built by eval/memorization.py build): domain
                   transfer to documents outside the corpus's own series, on the same footing as
                   ppl_domain_val. Skipped if the file is absent; --only postcutoff adds it to an
                   existing results/ppl/<run>.json without recomputing the other sets

ppl_domain_val / ppl_train - 1 is the train/val gap in the Stage 2 decision rule
(notes/decisions.md). One tokenizer for every model: the checkpoint's own tekken.json, which for
Ministral 3 3B and 8B is the same file.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))  # repo or /root/train
from common import auto_model_class
from packing import SLICE_WINDOWS, Packed, pack, spread

DATA = "data/processed"


@torch.no_grad()
def nll_sums(model, p: Packed, rows: np.ndarray, batch: int) -> dict:
    """Summed NLL and predicted-token counts over the windows in `rows`, three ways: per window
    (the unit of the window bootstrap), per document and per source (publisher). Each row's
    logits are upcast to fp32 one at a time, so the 131k-vocab upcast stays ~2 GB."""
    n_docs, n_src = len(p.doc_names), len(p.sources)
    win = np.zeros((len(rows), 2))
    doc_nll, doc_cnt = np.zeros(n_docs), np.zeros(n_docs)
    for i in range(0, len(rows), batch):
        idx = rows[i : i + batch]
        x = torch.from_numpy(p.ids[idx]).long().to(model.device)
        logits = model(input_ids=x).logits
        for j in range(len(idx)):
            lp = F.cross_entropy(logits[j, :-1].float(), x[j, 1:], reduction="none")
            lp = lp.double().cpu().numpy()
            doc = p.doc[idx[j], 1:]  # the document of each predicted token
            win[i + j] = lp.sum(), len(lp)
            doc_nll += np.bincount(doc, weights=lp, minlength=n_docs)
            doc_cnt += np.bincount(doc, minlength=n_docs)
        print(f"  {min(i + batch, len(rows))}/{len(rows)} windows", end="\r", flush=True)
    print()
    src = np.asarray(p.doc_source)
    return {
        "win": win,
        "doc_nll": doc_nll,
        "doc_cnt": doc_cnt,
        "src_nll": np.bincount(src, weights=doc_nll, minlength=n_src),
        "src_cnt": np.bincount(src, weights=doc_cnt, minlength=n_src),
    }


def ppl(nll, cnt) -> float:
    return round(math.exp(np.sum(nll) / np.sum(cnt)), 4)


def nats(nll, cnt) -> float:
    return round(float(np.sum(nll) / np.sum(cnt)), 5)


def add_postcutoff(result: dict, model, args) -> None:
    """ppl_postcutoff and its by-document, token and bootstrap-sum fields, added to result."""
    pc = pack([args.postcutoff], args.model)
    if len(pc.ids) != pc.expected:
        raise SystemExit(f"postcutoff: {len(pc.ids)} windows, n_tokens imply {pc.expected}")
    print(f"{args.run_name}: postcutoff ({len(pc.ids)} windows)")
    po = nll_sums(model, pc, np.arange(len(pc.ids)), args.batch)
    result["ppl_postcutoff"] = ppl(po["win"][:, 0], po["win"][:, 1])
    result["nats"]["postcutoff"] = nats(po["win"][:, 0], po["win"][:, 1])
    result["tokens"]["postcutoff"] = int(po["win"][:, 1].sum())
    result["by_doc_postcutoff"] = {
        name: {
            "publisher": pc.sources[pc.doc_source[k]],
            "ppl": ppl(po["doc_nll"][k], po["doc_cnt"][k]),
            "tokens": int(po["doc_cnt"][k]),
        }
        for k, name in enumerate(pc.doc_names)
        if po["doc_cnt"][k] > 0
    }
    result["sums"]["windows"]["postcutoff"] = po["win"].round(4).tolist()
    result["sums"]["docs"]["postcutoff"] = {
        n: [round(float(po["doc_nll"][k]), 4), int(po["doc_cnt"][k])]
        for k, n in enumerate(pc.doc_names)
        if po["doc_cnt"][k] > 0
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or merged checkpoint dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default="results/ppl")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--postcutoff", default="data/exposure/postcutoff.jsonl")
    ap.add_argument("--only", choices=["postcutoff"], help="add one set to the run's existing json")
    args = ap.parse_args()
    d = Path(args.data_dir)
    out = Path(args.out) / f"{args.run_name}.json"

    model = auto_model_class(args.model).from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()  # CPU: local tests only

    if args.only == "postcutoff":
        result = json.loads(out.read_text())
        add_postcutoff(result, model, args)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"ppl_postcutoff": result["ppl_postcutoff"]}))
        return

    train = pack([str(d / "train.jsonl")], args.model)
    val = pack([str(d / "val.jsonl")], args.model)
    general = pack([str(d / "general_val.jsonl")], args.model)
    for name, p in (("train", train), ("val", val), ("general_val", general)):
        if len(p.ids) != p.expected:
            raise SystemExit(f"{name}: {len(p.ids)} windows, n_tokens imply {p.expected}")

    print(f"{args.run_name}: train slice")
    tr = nll_sums(model, train, spread(len(train.ids)), args.batch)
    print(f"{args.run_name}: val ({len(val.ids)} windows)")
    va = nll_sums(model, val, np.arange(len(val.ids)), args.batch)
    print(f"{args.run_name}: general_val ({len(general.ids)} windows)")
    ge = nll_sums(model, general, np.arange(len(general.ids)), args.batch)
    sl = spread(len(val.ids), SLICE_WINDOWS)
    va_sl = va["win"][sl]

    result = {
        "run": args.run_name,
        "model": args.model,
        "ppl_train": ppl(tr["win"][:, 0], tr["win"][:, 1]),
        "ppl_domain_val": ppl(va["win"][:, 0], va["win"][:, 1]),
        "ppl_val_slice": ppl(va_sl[:, 0], va_sl[:, 1]),
        "ppl_general_val": ppl(ge["win"][:, 0], ge["win"][:, 1]),
        # mean NLL per predicted token (nats): deltas between runs are what the loss curves show
        "nats": {
            "train_slice": nats(tr["win"][:, 0], tr["win"][:, 1]),
            "domain_val": nats(va["win"][:, 0], va["win"][:, 1]),
            "val_slice": nats(va_sl[:, 0], va_sl[:, 1]),
            "general_val": nats(ge["win"][:, 0], ge["win"][:, 1]),
        },
        "by_publisher": {
            pub: {"ppl": ppl(va["src_nll"][k], va["src_cnt"][k]), "tokens": int(va["src_cnt"][k])}
            for k, pub in sorted(enumerate(val.sources), key=lambda kv: kv[1])
        },
        "by_doc": {
            name: {
                "publisher": val.sources[val.doc_source[k]],
                "ppl": ppl(va["doc_nll"][k], va["doc_cnt"][k]),
                "tokens": int(va["doc_cnt"][k]),
            }
            for k, name in enumerate(val.doc_names)
            if va["doc_cnt"][k] > 0
        },
        "tokens": {
            "train_slice": int(tr["win"][:, 1].sum()),
            "domain_val": int(va["win"][:, 1].sum()),
            "val_slice": int(va_sl[:, 1].sum()),
            "general_val": int(ge["win"][:, 1].sum()),
        },
        # raw sums for eval/ppl_compare.py's paired bootstrap: per window (4,096-token blocks)
        # and per document, [summed NLL, predicted tokens]; the same windows for every run
        "sums": {
            "windows": {
                "domain_val": va["win"].round(4).tolist(),
                "general_val": ge["win"].round(4).tolist(),
                "train_slice": tr["win"].round(4).tolist(),
            },
            "docs": {
                "domain_val": {
                    n: [round(float(va["doc_nll"][k]), 4), int(va["doc_cnt"][k])]
                    for k, n in enumerate(val.doc_names)
                    if va["doc_cnt"][k] > 0
                },
                "general_val": {
                    n: [round(float(ge["doc_nll"][k]), 4), int(ge["doc_cnt"][k])]
                    for k, n in enumerate(general.doc_names)
                    if ge["doc_cnt"][k] > 0
                },
            },
        },
    }
    result["train_val_gap"] = round(result["ppl_domain_val"] / result["ppl_train"] - 1, 4)
    if Path(args.postcutoff).exists():
        add_postcutoff(result, model, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k.startswith("ppl") or k == "train_val_gap"})
    )


if __name__ == "__main__":
    main()
