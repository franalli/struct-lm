"""PII scrub: rewrites data/processed/docs.jsonl in place, the last cleaning step.

Federal documents carry almost none, but the pipeline has the step. Emails -> [EMAIL],
US phone numbers -> [PHONE], SSN-shaped numbers -> [ID]; replacements are counted per type and
a sample of each is printed for a false-positive check. Person names are kept: they are the
authors of public documents, and stripping them would break citations. At Forge scale this is a
Presidio-class NER pass with client-specific entity lists, not three regexes.
"""

import argparse
import re
from collections import Counter

from common import BASE, load_tokenizer, read_jsonl, update_stats, write_jsonl

PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.IGNORECASE),
    # the last separator must be - or . so number columns ("100 200 3000") don't match
    "PHONE": re.compile(
        r"(?<![\w.-])(?:\(\s?[2-9]\d{2}\s?\)\s?|[2-9]\d{2}[-.\s])\d{3}[-.]\d{4}(?![\w-])"
    ),
    "ID": re.compile(r"(?<![\w.-])\d{3}-\d{2}-\d{4}(?![\w-])"),
}
SAMPLES = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/processed/docs.jsonl")
    args = ap.parse_args()

    encode, _ = load_tokenizer(BASE)
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {k: [] for k in PATTERNS}

    def scrub(para: str, slug: str) -> str:
        for kind, pattern in PATTERNS.items():
            for m in pattern.finditer(para):
                counts[kind] += 1
                if len(samples[kind]) < SAMPLES:
                    samples[kind].append(
                        f"[{slug}] ...{para[max(0, m.start() - 40) : m.end() + 40]}..."
                    )
            para = pattern.sub(f"[{kind}]", para)
        return para

    docs = read_jsonl(args.docs)  # read fully first: the output overwrites this file
    for i, d in enumerate(docs):
        n = sum(counts.values())
        text = scrub(d["text"], d["slug"])
        if sum(counts.values()) > n:
            docs[i] = {**d, "text": text, "n_tokens": len(encode(text))}

    n = write_jsonl(args.docs, docs)
    update_stats("pii", {"docs": n, "replacements": {k: counts[k] for k in PATTERNS}})
    print(f"{n} docs -> {args.docs}; replacements: {dict(counts)}")
    for kind, xs in samples.items():
        print(f"\n{kind} (first {len(xs)} of {counts[kind]}):")
        for x in xs:
            print("  " + " ".join(x.split()))


if __name__ == "__main__":
    main()
