"""Helpers shared by the corpus scripts (imported via the script directory on sys.path).

Every document record holds its text as paragraphs separated by a blank line. docs_raw.jsonl
also has page_offsets = [[page_no, char_offset], ...] marking where each kept PDF page starts,
so any span can be traced back to its page; split_pages / join_pages convert between that and
[(page_no, [paragraph, ...]), ...]. docs.jsonl and the splits are {slug, publisher, text, n_tokens}.
"""

import csv
import json
from pathlib import Path

BASE = "mistralai/Ministral-3-8B-Base-2512"
SOURCES = Path("data/sources.csv")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
STATS = PROCESSED / "stats.json"
SOURCE_FIELDS = ["slug", "publisher", "title", "url", "sha256", "pages", "tokens"]


def read_sources(path: Path = SOURCES) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_sources(rows: list[dict], path: Path = SOURCES) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in SOURCE_FIELDS} for r in rows)


def pdf_path(slug: str) -> Path:
    return RAW / f"{slug}.pdf"


def pdf_parts(path: Path) -> list:
    """The documents holding a PDF's pages: the file itself, or, for a PDF portfolio (a one-page
    cover saying "open this PDF portfolio in Acrobat", the manual stored as embedded files, e.g.
    the USACE Coastal Engineering Manual), its embedded PDFs in stored order. Caller closes them."""
    import pymupdf

    doc = pymupdf.open(path)
    names = [n for n in doc.embfile_names() if n.lower().endswith(".pdf")]
    if not names:
        return [doc]
    parts = [pymupdf.open(stream=doc.embfile_get(n), filetype="pdf") for n in names]
    doc.close()
    return parts


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def update_stats(section: str, data: dict) -> None:
    """Each script owns one top-level key of stats.json; re-running it rewrites only that key."""
    stats = json.loads(STATS.read_text()) if STATS.exists() else {}
    stats[section] = data
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")


def split_pages(doc: dict) -> list[tuple[int, list[str]]]:
    text, offs = doc["text"], doc["page_offsets"]
    ends = [o for _, o in offs[1:]] + [len(text)]
    return [(p, text[o:e].strip().split("\n\n")) for (p, o), e in zip(offs, ends)]


def join_pages(pages: list[tuple[int, list[str]]]) -> tuple[str, list[list[int]]]:
    """Inverse of split_pages; pages left with no paragraphs are dropped."""
    parts, offs, pos = [], [], 0
    for page_no, paras in pages:
        paras = [p for p in paras if p.strip()]
        if not paras:
            continue
        body = "\n\n".join(paras)
        if parts:
            pos += 2  # the "\n\n" before this page
        offs.append([page_no, pos])
        parts.append(body)
        pos += len(body)
    return "\n\n".join(parts), offs


def load_tokenizer(name: str = BASE):
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


def eval_docs(path: str = "eval/tasks/eval_chunk_ids.txt") -> set[str]:
    """Slugs of every document an eval item was built from (chunk ids are <slug>:p<page>:c<n>)."""
    return {line.split(":")[0] for line in Path(path).read_text().split()}
