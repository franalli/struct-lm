"""PDF -> page-anchored chunks of ~512 Ministral 3 (Tekken) tokens (pymupdf).

One JSONL record per chunk:
  {"chunk_id": "<slug>:p<page>:c<n>", "doc", "page", "page_label", "chunk", "text", "n_tokens",
   "title", "publisher", "url"}

page        1-based physical page in the PDF (what a PDF viewer's page box shows)
page_label  the printed page number when the PDF defines labels (e.g. "3-12", "B-4"), else null
chunk       0-based index within the page

Chunks never cross a page boundary, so every chunk cites exactly one page. Within a page,
text blocks (≈ paragraphs) are packed greedily up to --max-tokens; an oversized block is
split at sentence boundaries, and an oversized sentence at token boundaries.
Running headers/footers are removed before chunking.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf
from tqdm import tqdm

SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[])")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def page_blocks(page: pymupdf.Page) -> list[str]:
    """Text blocks in reading order, each joined into one paragraph string."""
    out = []
    for *_, text, _, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:  # 1 = image block
            continue
        text = HYPHEN_BREAK.sub(r"\1\2", text)
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        if text:
            out.append(text)
    return out


def boilerplate_key(block: str) -> str:
    return re.sub(r"\d+", "#", block.lower())


def strip_running_headers(pages: list[list[str]], min_frac: float = 0.3) -> list[list[str]]:
    """Drop short blocks that recur (digits masked) on >= min_frac of pages."""
    if len(pages) < 5:
        return pages
    counts: Counter[str] = Counter()
    for blocks in pages:
        counts.update({boilerplate_key(b) for b in blocks if len(b) < 120})
    threshold = max(3, min_frac * len(pages))
    repeated = {k for k, n in counts.items() if n >= threshold}
    return [[b for b in blocks if boilerplate_key(b) not in repeated] for blocks in pages]


def load_tokenizer(name: str):
    """(encode, decode) for the model being trained, without BOS/EOS.

    Mistral 3 checkpoints ship a Tekken tokenizer (tekken.json) that mistral-common loads
    exactly as vLLM's tokenizer_mode="mistral" does; anything else falls back to HF.
    """
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tek = MistralTokenizer.from_hf_hub(name).instruct_tokenizer.tokenizer
        print(f"tokenizer: {name} via mistral-common ({type(tek).__name__}, vocab {tek.n_words})")
        return (lambda s: tek.encode(s, bos=False, eos=False)), tek.decode
    except Exception as e:  # noqa: BLE001  repo without a mistral-common tokenizer file (any error)
        print(f"tokenizer: mistral-common can't load {name} ({str(e)[:100]}); using AutoTokenizer")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        return (lambda s: tok(s, add_special_tokens=False)["input_ids"]), tok.decode


class Chunker:
    def __init__(self, encode, decode, max_tokens: int):
        self.encode, self.decode = encode, decode
        self.max = max_tokens

    def n(self, text: str) -> int:
        return len(self.encode(text))

    def _pieces(self, block: str) -> list[tuple[str, int]]:
        """Split a block into pieces that each fit in max tokens."""
        n = self.n(block)
        if n <= self.max:
            return [(block, n)]
        pieces = []
        for sent in SENTENCE.split(block):
            ns = self.n(sent)
            if ns <= self.max:
                pieces.append((sent, ns))
                continue
            ids = self.encode(sent)
            for i in range(0, len(ids), self.max):
                window = ids[i : i + self.max]
                pieces.append((self.decode(window), len(window)))
        return pieces

    def chunk_page(self, blocks: list[str]) -> list[str]:
        chunks, cur, cur_n = [], [], 0
        for block in blocks:
            for piece, n in self._pieces(block):
                # Count the piece as it will appear after the " " join: Tekken often makes the
                # space its own token (e.g. before digits), so summing standalone counts
                # overshoots max by one token per join, ~80 on numeric-table pages.
                joined_n = self.n(" " + piece) if cur else n
                if cur and cur_n + joined_n > self.max:
                    chunks.append(" ".join(cur))
                    cur, cur_n, joined_n = [], 0, n
                cur.append(piece)
                cur_n += joined_n
        if cur:
            chunks.append(" ".join(cur))
        return chunks


class CrossDocDedup:
    """Drops exact-duplicate chunks and chunks whose word 8-grams mostly appeared in an
    earlier document (first copy in sources.csv order wins).

    Successive editions (FEMA P-1050-1 vs P-2082-1, NHI-04-041 vs NHI-15-058) and FHWA
    boilerplate repeat verbatim; left in, they get extra CPT weight and let an eval item's
    near-copy leak into training. Overlap within one document is not counted, and partial
    overlap (e.g. design examples sharing method text with different numbers) is kept.
    """

    NGRAM = 8
    WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.first_doc: dict[int, str] = {}  # 8-gram hash -> doc that first contained it
        self.texts: set[str] = set()

    def is_duplicate(self, doc: str, text: str) -> bool:
        norm = " ".join(self.WORD.findall(text.lower()))
        if norm in self.texts:
            return True
        w = norm.split()
        grams = {hash(" ".join(w[i : i + self.NGRAM])) for i in range(len(w) - self.NGRAM + 1)}
        if len(grams) >= 20:
            seen = sum(1 for g in grams if self.first_doc.get(g, doc) != doc)
            if seen / len(grams) >= self.threshold:
                return True
        self.texts.add(norm)
        for g in grams:
            self.first_doc.setdefault(g, doc)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    ap.add_argument("--tokenizer", default="mistralai/Ministral-3-8B-Base-2512")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--dup-threshold",
        type=float,
        default=0.9,
        help="drop a chunk when this fraction of its 8-grams already appeared in "
        "EARLIER documents (successive editions repeat whole sections)",
    )
    args = ap.parse_args()

    manifest = json.loads((Path(args.raw) / "manifest.json").read_text())
    chunker = Chunker(*load_tokenizer(args.tokenizer), args.max_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dedup = CrossDocDedup(args.dup_threshold)
    stats = []
    with out.open("w") as f:
        for src in tqdm(manifest, desc="extract"):
            slug = src["slug"]
            try:
                doc = pymupdf.open(src["path"])
            except Exception as e:  # noqa: BLE001  corrupt PDFs are expected; log and move on
                print(f"FAIL {slug}: {e}")
                continue
            with doc:
                labels = [p.get_label() or None for p in doc]
                pages = strip_running_headers([page_blocks(p) for p in doc])

            n_chunks = n_tokens = empty = n_dup = 0
            for page_no, (blocks, label) in enumerate(zip(pages, labels), start=1):
                if not blocks:
                    empty += 1
                    continue
                for c, text in enumerate(chunker.chunk_page(blocks)):
                    if dedup.is_duplicate(slug, text):
                        n_dup += 1  # chunk index c is kept, so surviving ids stay stable
                        continue
                    nt = chunker.n(text)
                    f.write(
                        json.dumps(
                            {
                                "chunk_id": f"{slug}:p{page_no}:c{c}",
                                "doc": slug,
                                "page": page_no,
                                "page_label": label,
                                "chunk": c,
                                "text": text,
                                "n_tokens": nt,
                                "title": src["title"],
                                "publisher": src["publisher"],
                                "url": src["url"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_chunks += 1
                    n_tokens += nt
            stats.append((slug, len(pages), empty, n_dup, n_chunks, n_tokens))

    print(f"\n{'doc':<26}{'pages':>7}{'empty':>7}{'dups':>6}{'chunks':>8}{'tokens':>11}")
    for slug, np_, ne, nd, nc, nt in stats:
        flag = "  <- scanned? needs OCR" if np_ and ne / np_ > 0.2 else ""
        print(f"{slug:<26}{np_:>7}{ne:>7}{nd:>6}{nc:>8}{nt:>11,}{flag}")
    tot = [sum(col) for col in zip(*[s[1:] for s in stats])] or [0] * 5
    print(f"{'TOTAL':<26}{tot[0]:>7}{tot[1]:>7}{tot[2]:>6}{tot[3]:>8}{tot[4]:>11,}  -> {out}")


if __name__ == "__main__":
    main()
