"""Quality filter: Gopher-style rules (Rae et al. 2021, app. A) adapted to engineering documents.

data/processed/docs_raw.jsonl -> data/processed/docs.jsonl ({slug, publisher, text, n_tokens}; dedup.py
and pii.py then rewrite it in place), plus dropped_samples.jsonl (50 random dropped paragraphs
per rule: read them, that's how a rule eating real content gets caught).

Paragraph rules (a paragraph is a text block; the first rule it fails is the one counted):
  alpha_ratio     letters / non-space characters < 0.6 (table debris, equation soup)
  word_length     mean word length < 3 or > 12
  short_lines     3+ lines, and > 40% of the words sit on lines under 6 words (tables of
                  contents, index pages, figure lists)
  numbered_lines  3+ lines, and > 30% of lines start with a digit or bullet and are under 6 words
  symbol_ratio    dot leaders, underscores, | … — • per word > 0.15
  reference_list  > 25% of lines contain a year in parentheses or "pp."
Tuned from dropped_samples.jsonl (the literal line-count rules dropped 28% of characters, half of
it real content): pymupdf puts list markers on their own line ("•" / "1." then the item), so
those are joined to their item first; the line rules need 3+ lines (single-line headings and
two-line paragraphs with a short wrap remainder aren't TOCs); short_lines counts words, not
lines, because worked examples interleave prose with one-token equation lines ("=", "k", "0.86").
Kept paragraphs are reflowed onto one line, so PDF line wraps don't become training text.
Equations and tables that pass are kept: engineering text needs them.

Document rules, after paragraph filtering:
  min_words   under 2,000 words
  dictionary  under 70% of words in an English stopword + dictionary list (scanned-and-garbled
              text and code listings fail this). Every source is English, so there is no
              language classifier; at Forge scale this would be fastText lid. Raised from 60%:
              kept documents score 0.81+ except one at 0.65 that is half XML object listings
              (FHWA-HIF-16-010, bridge information modelling), and 0.70 sits in that gap.
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from common import (
    BASE,
    load_tokenizer,
    read_jsonl,
    split_pages,
    update_stats,
    write_jsonl,
)

YEAR_OR_PP = re.compile(r"\((?:19|20)\d\d[a-z]?\)|\bpp\.")
SYMBOLS = re.compile(r"\.{3,}|_{3,}|[|…—•■□▪]")
BULLET = re.compile(r"\s*(?:\d|[•*‣◦▪■□–-])")
MARKER_LINE = re.compile(r"^((?:[•▪‣◦*–-]|\(?\d+(?:\.\d+)*[.)]?|\(?[a-z][.)]))\n", re.MULTILINE)
WORD = re.compile(r"[a-z]+")
FUNCTION_WORDS = (
    "a an the and or but if of to in on at by for with from as is are was were be been being it "
    "its this that these those which who whom what when where how not no shall should may must "
    "can will would could has have had do does did than then there their they we you he she"
)
STOPWORDS = set(FUNCTION_WORDS.split())
DICTIONARY = Path("/usr/share/dict/words")  # Linux: apt install wamerican
SAMPLES_PER_RULE = 50


def short(line: str) -> bool:
    return len(line.split()) < 6


def paragraph_rule(para: str) -> str | None:
    para = MARKER_LINE.sub(r"\1 ", para)
    words, lines = para.split(), para.split("\n")
    nonspace = sum(not c.isspace() for c in para)
    if sum(c.isalpha() for c in para) / nonspace < 0.6:
        return "alpha_ratio"
    if not 3 <= sum(len(w) for w in words) / len(words) <= 12:
        return "word_length"
    if len(lines) >= 3 and sum(len(ln.split()) for ln in lines if short(ln)) / len(words) > 0.4:
        return "short_lines"
    if (
        len(lines) >= 3
        and sum(bool(BULLET.match(ln)) and short(ln) for ln in lines) / len(lines) > 0.3
    ):
        return "numbered_lines"
    if len(SYMBOLS.findall(para)) / len(words) > 0.15:
        return "symbol_ratio"
    if sum(bool(YEAR_OR_PP.search(ln)) for ln in lines) / len(lines) > 0.25:
        return "reference_list"
    return None


class Dictionary:
    def __init__(self):
        self.words = STOPWORDS | {w.strip().lower() for w in DICTIONARY.open()}

    def known(self, w: str) -> bool:
        # web2 has few inflections: accept plurals and -ed/-ing forms of listed words
        return any(w[: len(w) - k] in self.words for k in (0, 1, 2, 3) if len(w) - k >= 2)

    def ratio(self, text: str) -> float:
        words = WORD.findall(text.lower())
        return sum(map(self.known, words)) / max(1, len(words))


class Reservoir:
    def __init__(self, k: int, seed: int = 0):
        self.k, self.rng, self.seen, self.items = k, random.Random(seed), 0, []

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        elif (j := self.rng.randrange(self.seen)) < self.k:
            self.items[j] = item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/docs_raw.jsonl")
    ap.add_argument("--out", default="data/processed/docs.jsonl")
    ap.add_argument("--samples", default="data/processed/dropped_samples.jsonl")
    ap.add_argument("--min-words", type=int, default=2000)
    ap.add_argument("--min-dictionary", type=float, default=0.7)
    args = ap.parse_args()

    encode, _ = load_tokenizer(BASE)
    dictionary = Dictionary()
    paras_dropped, tokens_dropped = Counter(), Counter()
    samples: dict[str, Reservoir] = {}
    docs_dropped, ratios = [], {}
    n_paras = tokens_in = 0

    def kept_docs():
        nonlocal n_paras, tokens_in
        for doc in read_jsonl(args.inp):
            tokens_in += doc["n_tokens"]
            kept = []
            for page_no, paras in split_pages(doc):
                for para in paras:
                    n_paras += 1
                    if rule := paragraph_rule(para):
                        paras_dropped[rule] += 1
                        tokens_dropped[rule] += len(encode(para))
                        samples.setdefault(rule, Reservoir(SAMPLES_PER_RULE)).add(
                            {"rule": rule, "slug": doc["slug"], "page": page_no, "text": para}
                        )
                    else:
                        kept.append(" ".join(para.split("\n")))
            text = "\n\n".join(kept)
            ratios[doc["slug"]] = ratio = round(dictionary.ratio(text), 3)
            n_words = len(text.split())
            reason = (
                "min_words"
                if n_words < args.min_words
                else "dictionary"
                if ratio < args.min_dictionary
                else None
            )
            if reason:
                docs_dropped.append(
                    {
                        "slug": doc["slug"],
                        "rule": reason,
                        "words": n_words,
                        "dictionary_ratio": ratio,
                    }
                )
                continue
            yield {
                "slug": doc["slug"],
                "publisher": doc["publisher"],
                "text": text,
                "n_tokens": len(encode(text)),
            }

    out = list(kept_docs())
    write_jsonl(args.out, out)
    write_jsonl(args.samples, (s for r in sorted(samples) for s in samples[r].items))

    tokens_out = sum(d["n_tokens"] for d in out)
    kept_paras = sum(len(d["text"].split("\n\n")) for d in out)
    kept_ratios = [ratios[d["slug"]] for d in out]
    update_stats(
        "filter",
        {
            "docs_in": len(ratios),
            "docs_out": len(out),
            "docs_dropped": docs_dropped,
            "paragraphs_in": n_paras,
            "paragraphs_out": kept_paras,
            "paragraphs_dropped": dict(paras_dropped.most_common()),
            "tokens_dropped_by_rule": dict(tokens_dropped.most_common()),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "dictionary_ratio_kept": {
                "min": min(kept_ratios),
                "median": sorted(kept_ratios)[len(kept_ratios) // 2],
            },
        },
    )
    print(f"docs {len(ratios)} -> {len(out)}; dropped: {docs_dropped}")
    print(
        f"paragraphs {n_paras:,} -> {kept_paras:,}; tokens {tokens_in:,} -> {tokens_out:,} "
        f"({1 - tokens_out / tokens_in:.1%} removed)"
    )
    print(f"{'rule':<16}{'paragraphs':>12}{'tokens':>12}")
    for rule, n in paras_dropped.most_common():
        print(f"{rule:<16}{n:>12,}{tokens_dropped[rule]:>12,}")
    print(f"samples -> {args.samples}")


if __name__ == "__main__":
    main()
