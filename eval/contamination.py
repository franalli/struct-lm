"""Contamination checks (13-gram overlap) between the CPT corpus and everything it is evaluated on.

  .venv/bin/python eval/contamination.py   # -> results/contamination.md, results/contamination.json

CPU only, a few minutes on the Mac. Reads data/processed/{train,val,replay,general_val}.jsonl,
data/exposure/postcutoff.jsonl, eval/tasks/{fewshot,domain_qa}.jsonl, results/ppl/<run>.json, and
the three benchmark splits lm-eval scores, from the Hugging Face Hub: cais/mmlu test,
openai/gsm8k test, Rowan/hellaswag validation (the configs recorded in results/lm_eval).

Method: GPT-3's 13-gram overlap (Brown et al., 2020, appendix C), counted on the model's own tokens
as Llama 2 does (Touvron et al., 2023, appendix A.6). Every text is tokenised with the base's Tekken
tokenizer as train/packing.py feeds it to the model, every run of 13 tokens is hashed to 64 bits
(at ~25M hashes a false match is a ~1e-5 event), and an eval text is measured against the hash set
of a reference corpus three ways:
  grams   the fraction of its 13-grams that occur in the reference
  tokens  the fraction of its tokens inside at least one such 13-gram (Llama 2's per-sample
          measure; it calls a sample dirty from 80%)
  run     the longest stretch of covered tokens: copied passages make long runs, shared phrasing
          (a standard sentence, a cited clause's title) short ones
The match is exact, so case, line-break and hyphenation differences are misses: a lower bound on
paraphrased overlap, the usual limit of n-gram checks. dedup.py removed near-duplicate paragraphs
(Jaccard >= 0.8 on 5-word shingles) across every corpus document before the split, so what this
finds between train and val is sharing below the paragraph.

Checks:
  1. domain val (12 documents) vs train, per document (with its top train source) and per
     perplexity window. Reference: each train document against the other 233, the overlap two
     documents of this corpus share anyway. Linked to results/ppl: the domain val change with and
     without the documents above --flag, and by window coverage, next to the 2026 reports' windows.
  2. the 13 post-cutoff (2026) reports vs train: a revision of a train manual would share most of
     its text with it.
  3. MMLU, GSM8K, HellaSwag vs train and vs replay (the FineWeb-Edu slice in cpt-8b-replay10).
  4. general_val vs replay (both FineWeb-Edu).
  5. the 3 domain_qa few-shot items vs the 325 scored items and each other; the scored items
     against each other.
  6. (--only sft, Stage 3) the SFT set (data/sft/train.jsonl) vs every eval item, question and
     answer apart and the seen / unseen halves apart (seen answers are in it by design; unseen
     should sit at the floor), vs sft_val.jsonl, and vs the three benchmarks; plus the chunk-id
     gate (rule 10) and a planted positive control. --only sft adds this section to the existing
     results/contamination.json and rewrites the .md, without recomputing sections 1-5.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from packing import SEQ_LEN, pack

BASE = "mistralai/Ministral-3-8B-Base-2512"
N = 13
MUL = np.uint64(0x9E3779B97F4A7C15)  # odd 64-bit multiplier of the rolling hash
REF = "base-8b-hf"  # rule 3: Stage 2+ rows compare against the base on the same path
RUNS = ("cpt-8b", "cpt-8b-seed1", "cpt-8b-replay10")
BINS = (0.0, 0.01, 0.05, 0.20, 1.0)  # window token coverage
DATA = ROOT / "data/processed"
SFT = ROOT / "data/sft"
POSTCUTOFF = ROOT / "data/exposure/postcutoff.jsonl"
TASKS = ROOT / "eval/tasks"


class Tok:
    """Tekken encode/decode plus the 13-gram hashes of a token sequence."""

    def __init__(self) -> None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tek = MistralTokenizer.from_hf_hub(BASE).instruct_tokenizer.tokenizer
        self.tek, self.bos, self.eos = tek, tek.bos_id, tek.eos_id
        # one random 64-bit value per token id, so the rolling hash sees no structure in the ids
        self.table = np.frombuffer(np.random.default_rng(0).bytes(8 * tek.n_words), np.uint64)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tek.encode(text, bos=False, eos=False), dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        return self.tek.decode([int(i) for i in ids])

    def grams(self, ids: np.ndarray) -> np.ndarray:
        """Hash of every N-token window of ids (none if the text is shorter than N tokens)."""
        n = len(ids) - N + 1
        if n <= 0:
            return np.empty(0, np.uint64)
        r = self.table[ids]
        h = np.zeros(n, np.uint64)
        for j in range(N):
            h = h * MUL + r[j : j + n]
        return h


class Index:
    """The 13-gram hashes of a set of reference documents, each hash once per document."""

    def __init__(self, grams: list[np.ndarray], names: list[str]) -> None:
        per = [np.unique(g) for g in grams]
        h = np.concatenate(per)
        doc = np.repeat(np.arange(len(per), dtype=np.int32), [len(p) for p in per])
        order = np.argsort(h, kind="stable")
        self.h, self.doc, self.names = h[order], doc[order], names
        self.keys, self.ndocs = np.unique(self.h, return_counts=True)

    def lookup(self, g: np.ndarray) -> np.ndarray:
        """Per hash in g, how many reference documents contain it (0: absent)."""
        if not len(g):
            return np.zeros(0, int)
        pos = np.minimum(np.searchsorted(self.keys, g), len(self.keys) - 1)
        return np.where(self.keys[pos] == g, self.ndocs[pos], 0)

    def sources(self, g: np.ndarray) -> np.ndarray:
        """Per reference document, how many of g's distinct hashes it contains."""
        g = np.unique(g)
        lo, hi = np.searchsorted(self.h, g, "left"), np.searchsorted(self.h, g, "right")
        n = hi - lo
        idx = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(n.sum())
        return np.bincount(self.doc[idx], minlength=len(self.names))


def coverage(hit: np.ndarray, n_tok: int) -> np.ndarray:
    """Token mask: True inside at least one hit 13-gram."""
    s = np.flatnonzero(hit)
    c = np.bincount(s, minlength=n_tok + 1) - np.bincount(s + N, minlength=n_tok + 1)
    return np.cumsum(c[:n_tok]) > 0


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def longest(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest covered stretch; (0, 0) if none."""
    starts, ends = runs(mask)
    if not len(starts):
        return 0, 0
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k] - starts[k])


def measure(ids: np.ndarray, g: np.ndarray, hit: np.ndarray) -> dict:
    cov = coverage(hit, len(ids))
    return {
        "tokens": len(ids),
        "grams": len(g),
        "gram_frac": float(hit.mean()) if len(g) else 0.0,
        "token_frac": float(cov.mean()) if len(ids) else 0.0,
        "run": longest(cov)[1],
    }


def jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def pct_change(nats: float) -> float:
    return (math.exp(nats) - 1) * 100


def boot_change(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int = 10_000) -> dict:
    """Pooled perplexity change of b vs a over units of [summed NLL, predicted tokens], paired,
    95% interval from resampling whole groups (documents), as eval/ppl_compare.py does."""
    ug, inv = np.unique(groups, return_inverse=True)
    A, B = np.zeros((len(ug), 2)), np.zeros((len(ug), 2))
    np.add.at(A, inv, a)
    np.add.at(B, inv, b)

    def delta(idx: np.ndarray) -> np.ndarray:
        a_, b_ = A[idx].sum(axis=-2), B[idx].sum(axis=-2)
        return b_[..., 0] / b_[..., 1] - a_[..., 0] / a_[..., 1]

    point = float(delta(np.arange(len(ug))))
    lo, hi = np.percentile(
        delta(np.random.default_rng(0).integers(0, len(ug), (n, len(ug)))), [2.5, 97.5]
    )
    return {
        "docs": len(ug),
        "tokens": int(A[:, 1].sum()),
        "ppl_ref": math.exp(A[:, 0].sum() / A[:, 1].sum()),
        "pct": pct_change(point),
        "ci": (pct_change(lo), pct_change(hi)),
    }


def ppl_runs(names: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for r in names:
        p = ROOT / "results/ppl" / f"{r}.json"
        if p.exists():
            out[r] = json.loads(p.read_text())
    return out


def window_coverage(tok: Tok, path: Path, ids: list[np.ndarray], covs: list[np.ndarray]):
    """Per perplexity window (train/packing.py over `path`), the covered share of its predicted
    tokens, and the document holding most of them. Asserts the windows match pack()'s."""
    p = pack([str(path)], BASE)
    flat = np.concatenate([np.r_[tok.bos, i, tok.eos] for i in ids])
    fcov = np.concatenate([np.r_[False, c, False] for c in covs])
    n = len(p.ids)
    if not np.array_equal(flat[: n * SEQ_LEN].reshape(n, SEQ_LEN), p.ids):
        raise SystemExit(f"{path}: token ids differ from train/packing.py's windows")
    win_cov = fcov[: n * SEQ_LEN].reshape(n, SEQ_LEN)[:, 1:].mean(axis=1)
    win_doc = np.array([np.bincount(row).argmax() for row in p.doc[:, 1:]])
    return win_cov, win_doc, p.doc_names


# ---- benchmark items, as lm-eval 0.4.13 renders them (configs in results/lm_eval) ----


def hellaswag_pre(text: str) -> str:
    """lm-eval's hellaswag preprocess()."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def benchmarks() -> dict[str, list[str]]:
    """Each scored item's text: the question with its choices (MMLU), the question with its
    worked solution (GSM8K), the context with its gold ending (HellaSwag)."""
    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    return {
        "MMLU (test)": [
            d["question"].strip() + "".join(f"\n{k}. {c}" for k, c in zip("ABCD", d["choices"]))
            for d in mmlu
        ],
        "GSM8K (test)": [f"Question: {d['question']}\nAnswer: {d['answer']}" for d in gsm],
        "HellaSwag (val)": [
            hellaswag_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
            + " "
            + hellaswag_pre(d["endings"][int(d["label"])])
            for d in hs
        ],
    }


def items_vs(tok: Tok, texts: list[str], refs: dict[str, Index], top: int = 3) -> dict:
    """Per reference, the share of items with any 13-gram in it and with >= 50% / 80% of their
    tokens covered; plus the items with the most coverage, for reading."""
    enc = [tok.encode(t) for t in texts]
    grams = [tok.grams(i) for i in enc]
    out: dict = {"items": len(texts), "checkable": int(sum(len(g) > 0 for g in grams))}
    cov_any = []
    for name, idx in refs.items():
        hits = [idx.lookup(g) > 0 for g in grams]
        covs = [coverage(h, len(i)) for h, i in zip(hits, enc)]
        frac = np.array([c.mean() if len(c) else 0.0 for c in covs])
        out[name] = {
            "any": int(sum(h.any() for h in hits)),
            "ge50": int((frac >= 0.5).sum()),
            "ge80": int((frac >= 0.8).sum()),
            "mean_token_frac": float(frac.mean()),
        }
        cov_any.append(covs)
    either = [np.logical_or.reduce(cs) for cs in zip(*cov_any)]
    frac = np.array([c.mean() if len(c) else 0.0 for c in either])
    out["either"] = {
        "any": int(sum(c.any() for c in either)),
        "ge50": int((frac >= 0.5).sum()),
        "ge80": int((frac >= 0.8).sum()),
        "mean_token_frac": float(frac.mean()),
    }
    examples = []
    for k in np.argsort(-frac, kind="stable")[:top]:
        if frac[k] == 0:
            break
        s, n = longest(either[k])
        where = [name for name, cs in zip(refs, cov_any) if cs[k][s : s + n].any()]
        examples.append(
            {
                "item": int(k),
                "token_frac": float(frac[k]),
                "found_in": where,
                "longest_span": tok.decode(enc[k][s : s + n]),
                "text": texts[k],
            }
        )
    out["examples"] = examples
    return out


def control(tok: Tok, docs: list[dict], idx: Index, k: int = 50, chars: int = 600) -> float:
    """Positive control: mean token coverage of ~600-character excerpts from the middle of k
    reference documents, tokenised on their own as benchmark items are. Near 100% means a
    benchmark item copied from the reference would be found; only the first token (no leading
    space) can differ."""
    fracs = []
    for d in docs[:k]:
        t = d["text"]
        s = t.find(" ", len(t) // 2) + 1
        e = t.rfind(" ", s, s + chars)
        ids = tok.encode(t[s:e])
        fracs.append(coverage(idx.lookup(tok.grams(ids)) > 0, len(ids)).mean())
    return float(np.mean(fracs))


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def qa_checks(tok: Tok) -> dict:
    fewshot, qa = jsonl(TASKS / "fewshot.jsonl"), jsonl(TASKS / "domain_qa.jsonl")
    text = lambda d: f"{d['question']}\n{d['answer']}"
    qa_g = [tok.grams(tok.encode(text(d))) for d in qa]
    qa_idx = Index(qa_g, [d["id"] for d in qa])
    qa_q = {norm(d["question"]) for d in qa}
    qa_chunks = {d["source_chunk"] for d in qa}
    qa_docs = {d["source_chunk"].split(":")[0] for d in qa}
    shots = []
    for k, d in enumerate(fewshot):
        g = tok.grams(tok.encode(text(d)))
        others = [tok.grams(tok.encode(text(e))) for j, e in enumerate(fewshot) if j != k]
        other = np.unique(np.concatenate(others)) if others else np.empty(0, np.uint64)
        shots.append(
            {
                "question": d["question"],
                "source_chunk": d["source_chunk"],
                "same_question_in_qa": norm(d["question"]) in qa_q,
                "same_chunk_in_qa": d["source_chunk"] in qa_chunks,
                "same_document_in_qa": d["source_chunk"].split(":")[0] in qa_docs,
                "gram_frac_in_qa": float((qa_idx.lookup(g) > 0).mean()) if len(g) else 0.0,
                "gram_frac_in_other_shots": float(np.isin(g, other).mean()) if len(g) else 0.0,
            }
        )
    # the scored items against each other: pairs sharing 13-grams
    pairs = []
    for k, g in enumerate(qa_g):
        shared = qa_idx.lookup(g) >= 2
        if not shared.any():
            continue
        counts = qa_idx.sources(g[shared])
        counts[k] = 0
        ug = len(np.unique(g))
        for j in np.flatnonzero(counts):
            if j > k:
                pairs.append(
                    {
                        "a": qa[k]["id"],
                        "b": qa[j]["id"],
                        "shared_of_a": float(counts[j] / ug),
                        "a_question": qa[k]["question"],
                        "b_question": qa[j]["question"],
                    }
                )
    pairs.sort(key=lambda p: -p["shared_of_a"])
    return {
        "fewshot": shots,
        "qa_items": len(qa),
        "qa_exact_duplicate_questions": len(qa) - len({norm(d["question"]) for d in qa}),
        "qa_pairs": pairs,
    }


# ---- Stage 3: the SFT set ----


def sft_checks(tok: Tok, bench: dict[str, list[str]]) -> dict:
    """Section 6. The SFT text a record trains on is its prompt plus its completion; "content" is
    the question (or term, or replay prompt) plus the completion, without instruction templates or
    passages, so train vs val measures repeated questions and answers, not shared boilerplate."""
    train, val = jsonl(SFT / "train.jsonl"), jsonl(SFT / "sft_val.jsonl")
    seen = set((TASKS / "sft_seen_chunks.txt").read_text().split())
    eval_ids = set((TASKS / "eval_chunk_ids.txt").read_text().split())

    def ids(r: dict) -> set[str]:
        out = set(r["source_chunks"]) | set(r["distractors"])
        return out | ({r["fact_id"].rsplit(":", 1)[0]} if r["fact_id"] else set())

    full = lambda r: r["prompt"][0]["content"] + "\n" + r["completion"][0]["content"]
    content = lambda r: (
        (r["question"] or r["prompt"][0]["content"]) + "\n" + r["completion"][0]["content"]
    )
    res: dict = {
        "train": len(train),
        "val": len(val),
        "leaked_chunks": sorted(set().union(*(ids(r) for r in train + val)) & (eval_ids - seen)),
    }
    idx = Index([tok.grams(tok.encode(full(r))) for r in train], [r["eid"] for r in train])

    qa, vocab = jsonl(TASKS / "domain_qa.jsonl"), jsonl(TASKS / "vocab.jsonl")
    grounded, adversarial = jsonl(TASKS / "grounded.jsonl"), jsonl(TASKS / "adversarial.jsonl")
    half = lambda i: "seen" if i["source_chunk"] in seen else "unseen"
    sets = {}
    for h in ("seen", "unseen"):
        sets[f"domain_qa {h}: questions"] = [i["question"] for i in qa if half(i) == h]
        sets[f"domain_qa {h}: answers"] = [i["answer"] for i in qa if half(i) == h]
        sets[f"vocab {h}: definitions"] = [i["definition"] for i in vocab if half(i) == h]
    sets["grounded: questions"] = [i["question"] for i in grounded]
    sets["adversarial: questions"] = [i["question"] for i in adversarial]
    res["eval"] = {k: items_vs(tok, v, {"sft": idx}) for k, v in sets.items()}

    # exact reuse: an eval question inside an SFT prompt; a domain_qa answer equal to an SFT
    # closed-book completion from the same document; a vocab term equal to an SFT definition's
    prompts_ = [norm(r["prompt"][0]["content"]) for r in train]
    cb = {}
    for r in train:
        if r["format"] == "closed_book" and r["fact_id"]:
            cb.setdefault(r["fact_id"].split(":")[0], set()).add(
                norm(r["completion"][0]["content"])
            )
    terms = {norm(r["question"]) for r in train if r["format"] == "definition"}
    exact = {}
    for h in ("seen", "unseen"):
        items = [i for i in qa if half(i) == h]
        exact[f"domain_qa {h}"] = {
            "items": len(items),
            "question_in_a_prompt": sum(
                any(norm(i["question"]) in p for p in prompts_) for i in items
            ),
            "answer_in_same_document": sum(
                norm(i["answer"]) in cb.get(i["source_chunk"].split(":")[0], set()) for i in items
            ),
        }
        v = [i for i in vocab if half(i) == h]
        exact[f"vocab {h}"] = {
            "items": len(v),
            "term_defined": sum(norm(i["term"]) in terms for i in v),
        }
    res["exact"] = exact

    tr_c = Index([tok.grams(tok.encode(content(r))) for r in train], [r["eid"] for r in train])
    res["val_vs_train"] = items_vs(tok, [content(r) for r in val], {"train": tr_c})
    res["benchmarks"] = {name: items_vs(tok, texts, {"sft": idx}) for name, texts in bench.items()}

    # positive control: 20 unseen domain_qa items and 20 MMLU items planted as SFT records
    planted = [f"{i['question']}\n{i['answer']}" for i in qa if half(i) == "unseen"][:20]
    planted += bench["MMLU (test)"][:20]
    ctl = Index(
        [tok.grams(tok.encode(full(r))) for r in train]
        + [tok.grams(tok.encode(t)) for t in planted],
        [r["eid"] for r in train] + [f"planted-{k}" for k in range(len(planted))],
    )
    m = items_vs(tok, planted, {"sft": ctl})
    res["control"] = {
        "planted": len(planted),
        "found_ge80": m["sft"]["ge80"],
        "checkable": m["checkable"],
    }
    return res


def sft_report(s: dict) -> list[str]:
    row = lambda name, m: [
        name, m["items"], m["checkable"], m["sft"]["any"], m["sft"]["ge50"], m["sft"]["ge80"],
        p(m["sft"]["mean_token_frac"]),
    ]  # fmt: skip
    head = [
        "eval items",
        "items",
        "checkable (>= 13 tokens)",
        "any 13-gram",
        ">= 50%",
        ">= 80%",
        "mean token share",
    ]
    out = [
        "",
        "## 6. SFT data (Stage 3) vs the eval and the benchmarks",
        "",
        (
            f"`data/sft/train.jsonl` ({s['train']:,} records; prompt + completion) as the reference."
            f" Chunk ids from eval_chunk_ids.txt outside the seen half: {len(s['leaked_chunks'])}"
            " (rule 10). Positive control: of"
            f" {s['control']['planted']} planted items (20 unseen domain_qa, 20 MMLU),"
            f" {s['control']['found_ge80']} are found with >= 80% of their tokens covered."
        ),
        "",
        table(head, [row(k, m) for k, m in s["eval"].items()]),
        "",
        "Exact reuse (normalised text):",
        "",
        table(
            [
                "eval half",
                "items",
                "question inside an SFT prompt",
                "answer = an SFT closed-book answer, same document",
            ],
            [
                [k, v["items"], v["question_in_a_prompt"], v["answer_in_same_document"]]
                for k, v in s["exact"].items()
                if "question_in_a_prompt" in v
            ],
        ),
        "",
        table(
            ["eval half", "items", "term defined in SFT"],
            [
                [k, v["items"], v["term_defined"]]
                for k, v in s["exact"].items()
                if "term_defined" in v
            ],
        ),
        "",
    ]
    v = s["val_vs_train"]
    out += [
        (
            f"sft_val ({s['val']} records, question + completion) vs train: {v['train']['any']} share"
            f" any 13-gram, {v['train']['ge50']} have >= 50% of their tokens covered,"
            f" {v['train']['ge80']} >= 80%."
        ),
        "",
        table(
            ["benchmark", "items", "any 13-gram", ">= 50%", ">= 80%"],
            [
                [k, m["items"], m["sft"]["any"], m["sft"]["ge50"], m["sft"]["ge80"]]
                for k, m in s["benchmarks"].items()
            ],
        ),
    ]
    return out


# ---- report ----


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def p(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def chg(c: dict | None) -> str:
    if not c:
        return ""
    return f"{c['pct']:+.2f}% [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"


def report(res: dict, flag: float) -> str:
    v, pc, loo = res["val"], res["postcutoff"], res["train_loo"]
    runs_ = res["runs"]
    out = [
        "# Contamination checks",
        "",
        "Generated by `eval/contamination.py`: token 13-gram overlap on the Tekken tokenizer",
        "(GPT-3 appendix C, measured on model tokens as in Llama 2 appendix A.6). Method and",
        "caveats in the script's docstring; the reading in `notes/decisions.md`. `grams` = share of",
        "a text's 13-grams found in the reference, `tokens` = share of its tokens inside one, `run` =",
        "longest covered stretch in tokens. Perplexity changes are vs `base-8b-hf`, with 95%",
        "intervals resampling documents.",
        "",
        "## 1. Domain val vs train",
        "",
        f"Reference: each train document against the other 233 shares a median {p(loo['median'])}",
        f"of its 13-grams (90th percentile {p(loo['p90'])}, max {p(loo['max'])} `{loo['max_doc']}`);",
        f"{loo['above_flag']} of 234 are above {p(flag, 0)}.",
        "",
    ]
    rows = []
    for slug, d in sorted(v.items(), key=lambda kv: -kv[1]["gram_frac"]):
        rows.append(
            [
                f"`{slug}`",
                f"{d['tokens']:,}",
                p(d["gram_frac"], 2),
                p(d["token_frac"], 2),
                d["run"],
                f"`{d['top_source']}` ({p(d['top_share'], 2)})",
                f"{d['ppl_ref']:.2f}" if d.get("ppl_ref") else "",
                f"{d['change']:+.2f}%" if d.get("change") is not None else "",
            ]
        )
    out += [
        table(
            [
                "val document",
                "tokens",
                "grams",
                "tokens covered",
                "run",
                "top train source (its share)",
                "base ppl",
                f"change ({res['main_run']})",
            ],
            rows,
        ),
        "",
        (
            f"All of domain val: {p(res['val_total']['gram_frac'], 2)} of 13-grams, "
            f"{p(res['val_total']['token_frac'], 2)} of tokens."
        ),
        "",
        f"### Domain val perplexity with and without the documents above {p(flag, 0)}",
        "",
    ]
    wo = res["with_without"]
    out += [
        table(
            ["documents", "n", "tokens", "base ppl"] + list(runs_),
            [
                [
                    k,
                    c[runs_[0]]["docs"],
                    f"{c[runs_[0]]['tokens']:,}",
                    f"{c[runs_[0]]['ppl_ref']:.3f}",
                ]
                + [chg(c.get(r)) for r in runs_]
                for k, c in wo.items()
            ],
        ),
        "",
        "### By window: the val gain against how much of each window is shared with train",
        "",
        "Every 4,096-token perplexity window (`train/packing.py`, the windows in `results/ppl`),",
        "binned by the share of its predicted tokens inside a 13-gram found in train; 2026 rows are",
        "the post-cutoff reports' windows on the same footing.",
        "",
    ]
    wruns = res["window_runs"]
    out += [
        table(
            ["windows", "coverage", "n", "docs", "base ppl"] + list(wruns),
            [
                [
                    b["set"],
                    b["bin"],
                    b["windows"],
                    b[wruns[0]]["docs"] if b.get(wruns[0]) else "",
                    f"{b[wruns[0]]['ppl_ref']:.3f}" if b.get(wruns[0]) else "",
                ]
                + [chg(b.get(r)) for r in wruns]
                for b in res["window_bins"]
            ],
        ),
        "",
        (
            f"Correlation of window coverage with the per-window change ({res['main_run']}, "
            f"nats): Spearman {res['window_corr']['spearman']:+.2f}, Pearson "
            f"{res['window_corr']['pearson']:+.2f} over {res['window_corr']['n']} windows."
        ),
        "",
        "## 2. 2026 reports vs train",
        "",
    ]
    rows = []
    for slug, d in sorted(pc.items(), key=lambda kv: -kv[1]["gram_frac"]):
        rows.append(
            [
                f"`{slug}`",
                f"{d['tokens']:,}",
                p(d["gram_frac"], 2),
                p(d["token_frac"], 2),
                d["run"],
                f"`{d['top_source']}` ({p(d['top_share'], 2)})",
                f"{d['change']:+.2f}%" if d.get("change") is not None else "",
            ]
        )
    out += [
        table(
            [
                "2026 report",
                "tokens",
                "grams",
                "tokens covered",
                "run",
                "top train source (its share)",
                f"change ({res['main_run']})",
            ],
            rows,
        ),
        "",
        "## 3. Benchmarks vs train and the replay slice",
        "",
        "The splits lm-eval scores. An item is checkable if it has at least 13 tokens; `any` = at",
        "least one 13-gram found (GPT-3's dirty), `>=80%` = at least 80% of its tokens covered",
        "(Llama 2's dirty). Positive control: 600-character excerpts of 50 train and 50 replay",
        (
            "documents, tokenised on their own like an item, come back"
            f" {p(res['control']['train'])} and {p(res['control']['replay'])} covered."
        ),
        "",
    ]
    rows = []
    for name, b in res["benchmarks"].items():
        for ref in ("train", "replay", "either"):
            r = b[ref]
            rows.append(
                [
                    name if ref == "train" else "",
                    b["items"] if ref == "train" else "",
                    ref,
                    f"{r['any']} ({p(r['any'] / b['items'], 2)})",
                    r["ge50"],
                    r["ge80"],
                    p(r["mean_token_frac"], 3),
                ]
            )
    out += [
        table(
            [
                "benchmark",
                "items",
                "vs",
                "any 13-gram",
                ">=50% tokens",
                ">=80% tokens",
                "mean tokens covered",
            ],
            rows,
        ),
        "",
        "Items with the most coverage (train or replay), and their longest matched span:",
        "",
    ]
    for name, b in res["benchmarks"].items():
        for e in b["examples"]:
            span = e["longest_span"].replace("\n", " ").replace("|", "/")
            out.append(
                f"- {name} #{e['item']} ({p(e['token_frac'])} covered, in "
                f'{" + ".join(e["found_in"])}): "{span[:200]}"'
            )
    gv = res["general_val"]
    out += [
        "",
        "## 4. general_val vs replay (both FineWeb-Edu)",
        "",
        (
            f"{gv['docs']} documents, {gv['tokens']:,} tokens: {p(gv['token_frac'], 3)} of tokens"
            f" covered; {gv['any']} documents share any 13-gram, {gv['ge50']} have >= 50% of"
            f" their tokens covered, {gv['ge80']} >= 80%. Longest run {gv['run']} tokens."
        ),
        "",
        "## 5. domain_qa few-shot items",
        "",
    ]
    q = res["qa"]
    out += [
        table(
            [
                "few-shot question",
                "same question in the 325",
                "same chunk",
                "same document",
                "13-grams in the 325",
                "13-grams in the other shots",
            ],
            [
                [
                    s["question"][:90] + ("..." if len(s["question"]) > 90 else ""),
                    s["same_question_in_qa"],
                    s["same_chunk_in_qa"],
                    s["same_document_in_qa"],
                    p(s["gram_frac_in_qa"]),
                    p(s["gram_frac_in_other_shots"]),
                ]
                for s in q["fewshot"]
            ],
        ),
        "",
        (
            f"The {q['qa_items']} scored items against each other: "
            f"{q['qa_exact_duplicate_questions']} exact duplicate questions; "
            f"{len(q['qa_pairs'])} pairs share a 13-gram (question + answer), "
            f"{sum(x['shared_of_a'] >= 0.5 for x in q['qa_pairs'])} share half or more. The top"
            " five (all pairs in `results/contamination.json`):"
        ),
        "",
    ]
    for x in q["qa_pairs"][:5]:
        out.append(
            f'- {x["a"]} / {x["b"]} ({p(x["shared_of_a"], 0)}): "{x["a_question"]}" / '
            f'"{x["b_question"]}"'
        )
    if "sft" in res:
        out += sft_report(res["sft"])
    return "\n".join(out) + "\n"


def main() -> None:
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flag",
        type=float,
        default=0.05,
        help="val documents above this share of 13-grams in train are dropped in the "
        "with/without comparison",
    )
    ap.add_argument("--main-run", default="cpt-8b-replay10")
    ap.add_argument(
        "--n",
        type=int,
        default=N,
        help="n-gram length; another n is a sensitivity run (the .md labels say 13): pass --out",
    )
    ap.add_argument("--out", default="results/contamination")
    ap.add_argument(
        "--only",
        choices=["sft"],
        help="add section 6 to the existing --out .json and rewrite the .md",
    )
    args = ap.parse_args()
    N = args.n
    tok = Tok()
    out = ROOT / args.out
    if args.only == "sft":
        res = json.loads(out.with_suffix(".json").read_text())
        print("SFT set vs eval, val and benchmarks")
        res["sft"] = sft_checks(tok, benchmarks())
        out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
        out.with_suffix(".md").write_text(report(res, args.flag))
        print(f"-> {out.with_suffix('.md')}, {out.with_suffix('.json')}")
        return

    print("tokenising train, val, 2026, replay, general_val")
    train, val, post = jsonl(DATA / "train.jsonl"), jsonl(DATA / "val.jsonl"), jsonl(POSTCUTOFF)
    replay, general = jsonl(DATA / "replay.jsonl"), jsonl(DATA / "general_val.jsonl")
    enc = {
        k: [tok.encode(d["text"]) for d in docs]
        for k, docs in (
            ("train", train),
            ("val", val),
            ("post", post),
            ("replay", replay),
            ("general", general),
        )
    }
    grams = {k: [tok.grams(i) for i in ids] for k, ids in enc.items()}
    tr_idx = Index(grams["train"], [d["slug"] for d in train])
    rp_idx = Index(grams["replay"], [d["id"] for d in replay])
    print(f"train: {len(tr_idx.keys):,} distinct 13-grams; replay: {len(rp_idx.keys):,}")

    # train against itself: a hash in 2+ train documents is shared with another document
    loo = np.array([float((tr_idx.lookup(g) >= 2).mean()) for g in grams["train"]])
    res: dict = {
        "n": N,
        "main_run": args.main_run,
        "train_loo": {
            "median": float(np.median(loo)),
            "p90": float(np.percentile(loo, 90)),
            "max": float(loo.max()),
            "max_doc": train[int(loo.argmax())]["slug"],
            "above_flag": int((loo > args.flag).sum()),
            "per_doc": {d["slug"]: float(x) for d, x in zip(train, loo)},
        },
    }

    ppl = ppl_runs((REF, *RUNS))
    ref = ppl[REF]
    runs_ = tuple(r for r in RUNS if r in ppl)
    res["runs"] = runs_

    def per_doc(key: str, docs: list[dict], set_name: str) -> tuple[dict, list[np.ndarray]]:
        out, covs = {}, []
        for d, ids, g in zip(docs, enc[key], grams[key]):
            hit = tr_idx.lookup(g) > 0
            m = measure(ids, g, hit)
            counts = tr_idx.sources(g[hit]) if hit.any() else np.zeros(len(train), int)
            m["top_source"] = train[int(counts.argmax())]["slug"]
            m["top_share"] = float(counts.max() / len(np.unique(g))) if len(g) else 0.0
            a = ref["sums"]["docs"].get(set_name, {}).get(d["slug"])
            b = (
                ppl.get(args.main_run, {})
                .get("sums", {})
                .get("docs", {})
                .get(set_name, {})
                .get(d["slug"])
            )
            if a:
                m["ppl_ref"] = math.exp(a[0] / a[1])
            if a and b:
                m["change"] = pct_change(b[0] / b[1] - a[0] / a[1])
            out[d["slug"]] = m
            covs.append(coverage(hit, len(ids)))
        return out, covs

    res["val"], val_covs = per_doc("val", val, "domain_val")
    res["postcutoff"], post_covs = per_doc("post", post, "postcutoff")
    tot_g = sum(len(g) for g in grams["val"])
    res["val_total"] = {
        "gram_frac": sum(m["gram_frac"] * m["grams"] for m in res["val"].values()) / tot_g,
        "token_frac": sum(c.sum() for c in val_covs) / sum(len(c) for c in val_covs),
    }

    # domain val perplexity with and without the flagged documents (per-document sums)
    flagged = [s for s, m in res["val"].items() if m["gram_frac"] > args.flag]
    names = list(ref["sums"]["docs"]["domain_val"])
    sets = {"all": names}
    if len(flagged) > 1:
        sets.update({f"without {s}": [x for x in names if x != s] for s in flagged})
    if flagged:
        sets[f"without {', '.join(flagged)}"] = [s for s in names if s not in flagged]
    res["flagged"] = flagged
    res["with_without"] = {}
    for label, keep in sets.items():
        a = np.array([ref["sums"]["docs"]["domain_val"][s] for s in keep])
        res["with_without"][label] = {
            r: boot_change(
                a,
                np.array([ppl[r]["sums"]["docs"]["domain_val"][s] for s in keep]),
                np.arange(len(keep)),
            )
            for r in runs_
        }

    # per window: coverage vs the change, val and the 2026 reports
    print("packing val and 2026 windows")
    v_cov, v_doc, _ = window_coverage(tok, DATA / "val.jsonl", enc["val"], val_covs)
    p_cov, p_doc, _ = window_coverage(tok, POSTCUTOFF, enc["post"], post_covs)
    res["window_coverage"] = {
        "domain_val": v_cov.round(5).tolist(),
        "postcutoff": p_cov.round(5).tolist(),
    }
    wruns = tuple(r for r in runs_ if "postcutoff" in ppl[r]["sums"]["windows"])
    res["window_runs"] = wruns
    bins = []
    for set_name, cov, docs in (("domain_val", v_cov, v_doc), ("postcutoff", p_cov, p_doc)):
        a_all = np.array(ref["sums"]["windows"][set_name])
        edges = [(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)] + [(0.0, 1.0)]
        for lo, hi in edges:
            sel = (cov >= lo) & ((cov < hi) if hi < 1.0 else (cov <= hi))
            if not sel.any():
                continue
            row = {
                "set": "val" if set_name == "domain_val" else "2026",
                "bin": "all" if (lo, hi) == (0.0, 1.0) else f"{p(lo, 0)}-{p(hi, 0)}",
                "windows": int(sel.sum()),
            }
            for r in wruns:
                b_all = np.array(ppl[r]["sums"]["windows"][set_name])
                row[r] = boot_change(a_all[sel], b_all[sel], docs[sel])
            bins.append(row)
    res["window_bins"] = bins
    a_all = np.array(ref["sums"]["windows"]["domain_val"])
    b_all = np.array(ppl[args.main_run]["sums"]["windows"]["domain_val"])
    dw = b_all[:, 0] / b_all[:, 1] - a_all[:, 0] / a_all[:, 1]
    rank = lambda x: np.argsort(np.argsort(x))
    res["window_corr"] = {
        "n": len(dw),
        "pearson": float(np.corrcoef(v_cov, dw)[0, 1]),
        "spearman": float(np.corrcoef(rank(v_cov), rank(dw))[0, 1]),
    }

    print("benchmarks")
    res["control"] = {"train": control(tok, train, tr_idx), "replay": control(tok, replay, rp_idx)}
    res["benchmarks"] = {
        name: items_vs(tok, texts, {"train": tr_idx, "replay": rp_idx})
        for name, texts in benchmarks().items()
    }

    # general_val vs replay
    gm = [measure(i, g, rp_idx.lookup(g) > 0) for i, g in zip(enc["general"], grams["general"])]
    res["general_val"] = {
        "docs": len(gm),
        "tokens": sum(m["tokens"] for m in gm),
        "token_frac": sum(m["token_frac"] * m["tokens"] for m in gm) / sum(m["tokens"] for m in gm),
        "any": sum(m["gram_frac"] > 0 for m in gm),
        "ge50": sum(m["token_frac"] >= 0.5 for m in gm),
        "ge80": sum(m["token_frac"] >= 0.8 for m in gm),
        "run": max(m["run"] for m in gm),
    }

    res["qa"] = qa_checks(tok)
    prev = out.with_suffix(".json")
    if prev.exists() and "sft" in (old := json.loads(prev.read_text())):
        res["sft"] = old["sft"]  # section 6 is refreshed by --only sft

    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    out.with_suffix(".md").write_text(report(res, args.flag))
    print(f"-> {out.with_suffix('.md')}, {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
