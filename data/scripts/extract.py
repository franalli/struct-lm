"""PDF -> text (pymupdf), in two shapes.

Default: one record per document, the input to the CPT corpus pipeline (filter -> dedup -> pii):
  data/processed/docs_raw.jsonl
  {"slug", "publisher", "title", "url", "text", "n_pages", "n_pages_kept", "page_offsets", "n_tokens"}

  text          kept pages joined by a blank line; text blocks (≈ paragraphs) within a page are
                also separated by a blank line and keep their PDF line breaks, which filter.py's
                line rules need (it reflows the paragraphs it keeps)
  page_offsets  [[page, char_offset], ...]: where each kept page (1-based physical page) starts

  Running headers/footers are stripped per document, then pages under 200 characters (covers,
  figure-only and scanned pages; no OCR) are dropped and counted. A PDF portfolio (cover page +
  embedded PDFs) is read through its embedded parts; page numbers run across the parts.

--chunks: page-anchored chunks of ~512 Tekken tokens, the input of eval/make_tasks.py:
  data/processed/chunks.jsonl
  {"chunk_id": "<slug>:p<page>:c<n>", "doc", "page", "page_label", "chunk", "text", "n_tokens",
   "title", "publisher", "url"}

  Frozen: built only from the documents in eval/tasks/eval_chunk_ids.txt, so corpus expansion
  can't change the file make_tasks.py samples from (the eval items are hand-reviewed).
  Chunks never cross a page boundary, so every chunk cites exactly one page. Within a page,
  text blocks are packed greedily up to --max-tokens; an oversized block is split at sentence
  boundaries, and an oversized sentence at token boundaries.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf
from common import (
    BASE,
    eval_docs,
    join_pages,
    load_tokenizer,
    pdf_parts,
    pdf_path,
    read_sources,
    update_stats,
    write_jsonl,
)
from tqdm import tqdm

SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[])")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
MIN_PAGE_CHARS = 200
COPYRIGHT = re.compile(r"©|\bcopyright\b|all rights reserved", re.IGNORECASE)


def page_blocks(page: pymupdf.Page, keep_lines: bool = False) -> list[str]:
    """Text blocks in reading order. keep_lines=False joins each block into one line (chunks);
    True keeps its lines, whitespace-normalised (documents)."""
    out = []
    for *_, text, _, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:  # 1 = image block
            continue
        text = HYPHEN_BREAK.sub(r"\1\2", text)
        if keep_lines:
            text = "\n".join(" ".join(ln.split()) for ln in text.split("\n") if ln.strip())
        else:
            text = re.sub(r"\s*\n\s*", " ", text).strip()
        if text:
            out.append(text)
    return out


# --- documents ------------------------------------------------------------------------------


def line_key(line: str) -> str:
    return " ".join(re.sub(r"\d+", "#", line.lower()).split())


def strip_boilerplate_lines(
    pages: list[list[str]], min_frac: float = 0.3, edge: int = 3
) -> tuple[list[list[str]], int]:
    """Drop running headers and footers: a line (digits masked, so page numbers and dated
    manual ids match) that is among the first or last `edge` lines of more than min_frac of the
    pages. Only page edges are candidates, so a recurring body line ("where:") survives."""
    if len(pages) < 5:
        return pages, 0

    def lines(blocks: list[str]) -> list[tuple[int, str]]:
        return [(i, ln) for i, b in enumerate(blocks) for ln in b.split("\n")]

    counts: Counter[str] = Counter()
    for blocks in pages:
        ls = lines(blocks)
        counts.update({line_key(ln) for _, ln in ls[:edge] + ls[-edge:]})
    threshold = max(2, min_frac * len(pages))
    repeated = {k for k, n in counts.items() if n > threshold}

    out, removed = [], 0
    for blocks in pages:
        ls = lines(blocks)
        keep: list[list[str]] = [[] for _ in blocks]
        for j, (i, ln) in enumerate(ls):
            if (j < edge or j >= len(ls) - edge) and line_key(ln) in repeated:
                removed += 1
            else:
                keep[i].append(ln)
        out.append(["\n".join(k) for k in keep if k])
    return out, removed


def extract_doc(src: dict, encode) -> tuple[dict, dict]:
    raw, has_image = [], []
    for part in pdf_parts(pdf_path(src["slug"])):
        with part:
            raw += [page_blocks(p, keep_lines=True) for p in part]
            has_image += [bool(p.get_images()) for p in part]
    pages, n_boiler = strip_boilerplate_lines(raw)

    kept, dropped_image, dropped_short = [], 0, 0
    for page_no, (blocks, image) in enumerate(zip(pages, has_image), start=1):
        if sum(len(b) for b in blocks) >= MIN_PAGE_CHARS:
            kept.append((page_no, blocks))
        elif image:
            dropped_image += 1
        else:
            dropped_short += 1
    text, offsets = join_pages(kept)
    doc = {
        **{k: src[k] for k in ("slug", "publisher", "title", "url")},
        "text": text,
        "n_pages": len(raw),
        "n_pages_kept": len(kept),
        "page_offsets": offsets,
        "n_tokens": len(encode(text)),
    }
    info = {
        "boilerplate_lines": n_boiler,
        "dropped_image_only": dropped_image,
        "dropped_short": dropped_short,
        # checked before header stripping: a copyright footer repeats on every page
        "copyright_flag": bool(COPYRIGHT.search("\n".join("\n".join(b) for b in raw[:5]))),
    }
    return doc, info


def write_docs(sources: list[dict], out: Path, encode) -> None:
    by_pub: dict[str, Counter] = {}
    boiler, scanned, flagged = 0, [], []

    def records():
        nonlocal boiler
        for src in tqdm(sources, desc="extract"):
            doc, info = extract_doc(src, encode)
            c = by_pub.setdefault(src["publisher"], Counter())
            c.update(
                docs=1,
                pages=doc["n_pages"],
                pages_kept=doc["n_pages_kept"],
                dropped_image_only=info["dropped_image_only"],
                dropped_short=info["dropped_short"],
                tokens=doc["n_tokens"],
            )
            boiler += info["boilerplate_lines"]
            if doc["n_pages_kept"] < 0.5 * doc["n_pages"]:
                scanned.append(src["slug"])
            if info["copyright_flag"]:
                flagged.append(src["slug"])
            yield doc

    n = write_jsonl(out, records())
    total = sum(by_pub.values(), Counter())
    update_stats(
        "extract",
        {
            "docs": n,
            "by_publisher": {p: dict(c) for p, c in sorted(by_pub.items())},
            "total": dict(total),
            "boilerplate_lines_removed": boiler,
            "likely_scanned": scanned,  # over half the pages dropped as < 200 characters
            "copyright_flags": flagged,  # hand-check each: © / "copyright" in the first 5 pages
        },
    )
    print(
        f"{n} docs, {total['pages_kept']}/{total['pages']} pages kept, "
        f"{total['tokens']:,} tokens -> {out}"
    )
    print(
        f"pages dropped: {total['dropped_image_only']} image-only, "
        f"{total['dropped_short']} blank/short; {boiler} header/footer lines removed"
    )
    print(f"likely scanned (>50% pages dropped): {scanned}")
    print(f"copyright flags (hand-check): {flagged}")


# --- eval chunks (frozen) -------------------------------------------------------------------


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


def write_chunks(sources: list[dict], out: Path, chunker: Chunker, dup_threshold: float) -> None:
    dedup = CrossDocDedup(dup_threshold)
    stats = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for src in tqdm(sources, desc="extract"):
            slug = src["slug"]
            try:
                doc = pymupdf.open(pdf_path(slug))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", action="store_true", help="write the eval's chunks.jsonl instead")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tokenizer", default=BASE)
    ap.add_argument("--max-tokens", type=int, default=512, help="--chunks only")
    ap.add_argument(
        "--dup-threshold",
        type=float,
        default=0.9,
        help="--chunks only: drop a chunk when this fraction of its 8-grams already appeared in "
        "EARLIER documents (successive editions repeat whole sections)",
    )
    args = ap.parse_args()

    encode, decode = load_tokenizer(args.tokenizer)
    sources = [s for s in read_sources() if s["sha256"]]  # download.py hashes only accepted files
    if args.chunks:
        keep = eval_docs()
        sources = [s for s in sources if s["slug"] in keep]
        out = args.out or Path("data/processed/chunks.jsonl")
        write_chunks(sources, out, Chunker(encode, decode, args.max_tokens), args.dup_threshold)
    else:
        write_docs(sources, args.out or Path("data/processed/docs_raw.jsonl"), encode)


if __name__ == "__main__":
    main()
