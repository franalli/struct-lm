"""How well do the tokenizers of the models we train fit the domain corpus?

Both checkpoints in use, Ministral 3 8B Base and Instruct, each loaded from its own repo via
mistral-common (Tekken, as vLLM and extract.py load it). Reports:
  - total corpus tokens (docs.jsonl)
  - fertility (tokens per whitespace word) on the corpus and on ~1M tokens of FineWeb-Edu
    (the head of replay.jsonl), i.e. how much more the domain text costs than general English
  - fertility on domain terms: the vocab eval terms, the top 500 corpus terms by TF-IDF against
    FineWeb-Edu, and a few designations (ASCE 7-22, EM 1110-2-2104, ...)
  - every term word that splits into 4+ tokens

Terms are counted as they appear mid-text: n("the " + term) - n("the"). This is the evidence for
the vocabulary-extension decision in notes/decisions.md.
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from common import BASE, load_tokenizer, read_jsonl, table, update_stats

MODELS = {
    "base": BASE,
    "instruct": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
}
TERM = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
PROBES = [
    "ASCE 7-22",
    "ASCE 7-16",
    "EM 1110-2-2104",
    "NEHRP",
    "AASHTO LRFD",
    "kip-ft",
    "ksi",
    "ASTM A709",
    "A709 Grade 50W",
    "HPS 70W",
    "f'c",
    "P-delta",
    "Cs = SDS/(R/Ie)",
    "orthotropic",
    "electroslag",
    "austenitic",
    "martensitic",
    "Charpy V-notch",
]
FINEWEB_TOKENS = 1_000_000


def tfidf_terms(corpus: list[str], reference: list[str], k: int) -> list[str]:
    """Top-k corpus terms by TF (corpus count) x IDF over the FineWeb-Edu documents."""
    tf, df, surface = Counter(), Counter(), {}
    for text in corpus:
        found = TERM.findall(text)
        for w in found:
            surface.setdefault(w.lower(), Counter())[w] += 1
        tf.update(w.lower() for w in found)
        df.update({w.lower() for w in found})
    ref_df = Counter()
    for text in reference:
        ref_df.update({w.lower() for w in TERM.findall(text)})
    n = len(reference)
    scored = [
        (c * math.log(n / (1 + ref_df[w])), w)
        for w, c in tf.items()
        if c >= 20 and df[w] >= 3 and len(w) >= 3
    ]
    return [surface[w].most_common(1)[0][0] for _, w in sorted(scored, reverse=True)[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/processed/docs.jsonl")
    ap.add_argument("--replay", default="data/processed/replay.jsonl")
    ap.add_argument("--vocab", default="eval/tasks/vocab.jsonl")
    ap.add_argument("--n-tfidf", type=int, default=500)
    ap.add_argument("--report", default="data/processed/tokenizer_coverage.md")
    args = ap.parse_args()

    corpus = [d["text"] for d in read_jsonl(args.docs)]
    replay = read_jsonl(args.replay)
    fineweb, n = [], 0
    for r in replay:
        if n >= FINEWEB_TOKENS:
            break
        fineweb.append(r["text"])
        n += r["n_tokens"]

    term_sets = {
        "vocab_eval": sorted({r["term"] for r in read_jsonl(args.vocab)}),
        "tfidf_top": tfidf_terms(corpus, [r["text"] for r in replay], args.n_tfidf),
        "probes": PROBES,
    }
    words = sorted({w for terms in term_sets.values() for t in terms for w in t.split()})
    corpus_words = sum(len(t.split()) for t in corpus)
    fineweb_words = sum(len(t.split()) for t in fineweb)

    stats: dict = {"term_counts": {k: len(v) for k, v in term_sets.items()}}
    encodings = {}
    for role, name in MODELS.items():
        encode, _ = load_tokenizer(name)
        corpus_tokens = sum(len(encode(t)) for t in corpus)  # the one full pass per model
        fineweb_tokens = sum(len(encode(t)) for t in fineweb)

        def n_tok(term: str, encode=encode) -> int:
            return len(encode("the " + term)) - len(encode("the"))

        per_word = {w: n_tok(w) for w in words}
        fragmented = sorted(
            ((n, w) for w, n in per_word.items() if n >= 4), key=lambda x: (-x[0], x[1])
        )
        stats[role] = {
            "model": name,
            "corpus_tokens": corpus_tokens,
            "fertility": {
                "corpus": round(corpus_tokens / corpus_words, 3),
                "fineweb_edu": round(fineweb_tokens / fineweb_words, 3),
            },
            "term_fertility": {
                k: round(sum(map(n_tok, ts)) / sum(len(t.split()) for t in ts), 3)
                for k, ts in term_sets.items()
            },
            "probes": {t: n_tok(t) for t in PROBES},
            "words_4plus_tokens": len(fragmented),
            "term_words": len(words),
            "worst_fragmented": [{"word": w, "tokens": n} for n, w in fragmented[:40]],
        }
        encodings[role] = [encode(t) for t in corpus[:20] + words]

    stats["base_instruct_identical"] = encodings["base"] == encodings["instruct"]
    stats["tfidf_top_terms"] = term_sets["tfidf_top"][:50]
    update_stats("tokenizer", stats)
    md = report(stats)
    Path(args.report).write_text(md)
    print(md)
    print(f"-> {args.report}")


def tables(stats: dict) -> str:
    """The fertility comparison, designations and worst-fragmented words (also in the corpus card)."""
    b, i = stats["base"], stats["instruct"]
    rows = [
        ["corpus tokens", b["corpus_tokens"], i["corpus_tokens"]],
        ["fertility: domain corpus", b["fertility"]["corpus"], i["fertility"]["corpus"]],
        ["fertility: FineWeb-Edu", b["fertility"]["fineweb_edu"], i["fertility"]["fineweb_edu"]],
        [
            "corpus / FineWeb-Edu",
            *(f"{s['fertility']['corpus'] / s['fertility']['fineweb_edu']:.2f}x" for s in (b, i)),
        ],
        *(
            [f"fertility: {k} terms ({n})", b["term_fertility"][k], i["term_fertility"][k]]
            for k, n in stats["term_counts"].items()
        ),
        [
            "term words split into 4+ tokens",
            *(f"{s['words_4plus_tokens']} / {s['term_words']}" for s in (b, i)),
        ],
    ]
    return "\n".join(
        [
            table(["", f"`{b['model']}`", f"`{i['model']}`"], rows),
            "",
            (
                "Base and Instruct encode identically (corpus sample + every term word): "
                f"**{stats['base_instruct_identical']}**."
            ),
            "",
            table(["designation", "tokens"], [[f"`{t}`", n] for t, n in b["probes"].items()]),
            "",
            "Worst-fragmented term words: "
            + ", ".join(f"`{x['word']}` ({x['tokens']})" for x in b["worst_fragmented"]),
        ]
    )


def report(stats: dict) -> str:
    """tokenizer_coverage.md: the fertility comparison and the worst-fragmented terms."""
    return "\n".join(
        [
            "# Tokenizer coverage",
            "",
            "Generated by `data/scripts/tokenizer_coverage.py` from `data/processed/docs.jsonl` (the",
            "clean corpus) and the head of `replay.jsonl` (~1M tokens of FineWeb-Edu). Both checkpoints",
            "in use, loaded from their own repos via mistral-common (Tekken, 131k vocabulary).",
            "Fertility is tokens per whitespace word; a term is counted as it appears mid-text:",
            'n("the " + term) - n("the").',
            "",
            tables(stats),
            "",
            "## Top corpus terms by TF-IDF against FineWeb-Edu",
            "",
            ", ".join(stats["tfidf_top_terms"]),
            "",
        ]
    )


if __name__ == "__main__":
    main()
