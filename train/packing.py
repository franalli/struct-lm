"""Next-token windows over JSONL documents, shared by cpt.py (training, trainer eval) and
eval/perplexity.py, so every loss and perplexity in Stage 2 is computed on the same windows.

Each document becomes BOS + its Tekken tokens + EOS, as token ids; the documents are
concatenated in file order and cut into fixed SEQ_LEN windows, and the last partial window is
dropped. data/scripts/split.py's packed() counts exactly these windows (sum(n_tokens + 2) //
SEQ_LEN), which is what cpt.py checks the result against.

Why not TRL's packing (notes/contributions.md): its default strategy ("bfd") keeps only the first
max_length tokens of each document, about 1M of the corpus's 19.4M tokens, and on raw text it
appends EOS as the string "</s>", which Tekken encodes as ordinary text rather than id 2.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEQ_LEN = 4096
SLICE_WINDOWS = 49  # ~200k tokens: the trainer's val slice and the ppl_train slice


@dataclass
class Packed:
    ids: np.ndarray  # (n_windows, seq_len) int32 token ids
    source: np.ndarray  # (n_windows, seq_len) int16 index into `sources`, per token
    sources: list[str]  # publisher of each source index ("general" for FineWeb-Edu rows)
    doc: np.ndarray  # (n_windows, seq_len) int32 index into `doc_names`, per token
    doc_names: list[str]  # slug (corpus) or id (FineWeb-Edu) of each document, in file order
    doc_source: list[int]  # the `sources` index of each document
    docs: int
    expected: int  # windows implied by the files' own n_tokens: sum(n_tokens + 2) // seq_len


def encoder(model: str):
    """(encode, bos_id, eos_id) for a hub id or a merged checkpoint directory: the Tekken
    tokenizer via mistral-common, as data/scripts/common.load_tokenizer counted n_tokens and as
    vLLM loads it with tokenizer_mode="mistral" (rule 3). encode adds no BOS/EOS."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    local = Path(model) / "tekken.json"
    mt = (
        MistralTokenizer.from_file(str(local))
        if local.exists()
        else MistralTokenizer.from_hf_hub(model)
    )
    tek = mt.instruct_tokenizer.tokenizer
    return (lambda s: tek.encode(s, bos=False, eos=False)), tek.bos_id, tek.eos_id


def pack(paths: list[str], model: str, seq_len: int = SEQ_LEN) -> Packed:
    encode, bos, eos = encoder(model)
    sources: list[str] = []
    doc_names: list[str] = []
    doc_source: list[int] = []
    ids, src, doc, expected_tokens = [], [], [], 0
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                pub = d.get("publisher", "general")
                if pub not in sources:
                    sources.append(pub)
                toks = np.asarray([bos, *encode(d["text"]), eos], dtype=np.int32)
                ids.append(toks)
                src.append(np.full(len(toks), sources.index(pub), dtype=np.int16))
                doc.append(np.full(len(toks), len(doc_names), dtype=np.int32))
                doc_names.append(str(d.get("slug") or d.get("id")))
                doc_source.append(sources.index(pub))
                expected_tokens += d["n_tokens"] + 2
    flat, flat_src, flat_doc = np.concatenate(ids), np.concatenate(src), np.concatenate(doc)
    n = len(flat) // seq_len
    return Packed(
        ids=flat[: n * seq_len].reshape(n, seq_len),
        source=flat_src[: n * seq_len].reshape(n, seq_len),
        sources=sources,
        doc=flat_doc[: n * seq_len].reshape(n, seq_len),
        doc_names=doc_names,
        doc_source=doc_source,
        docs=len(doc_names),
        expected=expected_tokens // seq_len,
    )


def spread(n_windows: int, k: int = SLICE_WINDOWS) -> np.ndarray:
    """k window indices spread evenly from the first window to the last, so a slice samples
    every part of the file (documents are contiguous) rather than the head of one document."""
    return np.unique(np.linspace(0, n_windows - 1, min(k, n_windows)).round().astype(int))
