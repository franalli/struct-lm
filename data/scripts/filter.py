"""Language ID, Gopher-style quality rules, and PII scrubbing.

Gopher rules follow Rae et al. 2021 (MassiveText), appendix A. Every rejected document is
logged with the rule that rejected it, so thresholds can be tuned from data, not guessed.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("•", "-", "*", "‣", "◦")

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<CARD>"),
]


def gopher_reject(text: str) -> str | None:
    """Return the name of the first failed rule, or None if the doc passes."""
    words = text.split()
    n = len(words)
    if not 50 <= n <= 100_000:
        return "word_count"
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:
        return "mean_word_length"
    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.1:
        return "symbol_ratio"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(BULLETS) for ln in lines) / len(lines) > 0.9:
            return "bullet_lines"
        if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
            return "ellipsis_lines"
    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:
        return "alpha_words"
    if len(STOP_WORDS & {w.lower() for w in words}) < 2:
        return "stop_words"
    return None


def scrub_pii(text: str) -> str:
    for pattern, token in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


class LangID:
    def __init__(self, model_path: str):
        import fasttext

        self.model = fasttext.load_model(model_path)

    def __call__(self, text: str) -> tuple[str, float]:
        labels, probs = self.model.predict(text.replace("\n", " ")[:2000])
        return labels[0].removeprefix("__label__"), float(probs[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/filtered.jsonl")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--lang-threshold", type=float, default=0.65)
    ap.add_argument(
        "--lid-model",
        default="data/raw/lid.176.bin",
        help="fastText LID model (https://fasttext.cc/docs/en/language-identification)",
    )
    args = ap.parse_args()

    lid = LangID(args.lid_model)
    reasons: Counter[str] = Counter()
    out = Path(args.out)
    rejects = out.with_suffix(".rejects.jsonl")

    with open(args.inp) as fin, out.open("w") as fout, rejects.open("w") as frej:
        for line in tqdm(fin, desc="filter"):
            doc = json.loads(line)
            lang, p = lid(doc["text"])
            reason = None if (lang == args.lang and p >= args.lang_threshold) else "language"
            reason = reason or gopher_reject(doc["text"])
            if reason:
                reasons[reason] += 1
                frej.write(json.dumps({"chunk_id": doc["chunk_id"], "reason": reason}) + "\n")
                continue
            reasons["kept"] += 1
            doc["text"] = scrub_pii(doc["text"])
            fout.write(json.dumps(doc) + "\n")

    print(json.dumps(dict(reasons), indent=2))


if __name__ == "__main__":
    main()
