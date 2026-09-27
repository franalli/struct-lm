"""Last step: record per-document tokens in sources.csv and print the corpus card.

Each earlier script writes its own section of data/processed/stats.json. This fills the tokens
column of data/sources.csv (Tekken tokens each document contributes to the clean corpus, 0 if it
was dropped) and prints the corpus card tables (markdown) for notes/decisions.md, so every number
in the card comes from stats.json, which is committed next to the scripts that produced it.
"""

import json

from common import STATS, read_jsonl, read_sources, write_sources


def table(header: list[str], rows: list[list]) -> str:
    fmt = [f"{x:,}" if isinstance(x, int) else str(x) for r in rows for x in r]
    cells = [fmt[i : i + len(header)] for i in range(0, len(fmt), len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


def main() -> None:
    tokens = {d["slug"]: d["n_tokens"] for d in read_jsonl("data/processed/docs.jsonl")}
    rows = read_sources()
    for r in rows:
        r["tokens"] = tokens.get(r["slug"], 0)
    write_sources(rows)

    s = json.loads(STATS.read_text())
    ex, fi, de, pii, sp, rp, tk = (
        s[k] for k in ("extract", "filter", "dedup", "pii", "split", "replay", "tokenizer")
    )

    print("**Sources and pages** (extract.py; pages under 200 characters dropped, no OCR)\n")
    rows = []
    for pub, c in ex["by_publisher"].items():
        fin = sp["by_publisher"].get(pub, {"train": {"tokens": 0}, "val": {"tokens": 0}})
        rows.append(
            [
                pub,
                c["docs"],
                c["pages"],
                c["pages_kept"],
                c["dropped_image_only"],
                c["dropped_short"],
                c["tokens"],
                fin["train"]["tokens"] + fin["val"]["tokens"],
            ]
        )
    t = ex["total"]
    rows.append(
        [
            "**total**",
            t["docs"],
            t["pages"],
            t["pages_kept"],
            t["dropped_image_only"],
            t["dropped_short"],
            t["tokens"],
            sp["train"]["tokens"] + sp["val"]["tokens"],
        ]
    )
    print(
        table(
            [
                "publisher",
                "docs",
                "pages",
                "kept",
                "dropped: image-only",
                "dropped: blank",
                "tokens extracted",
                "tokens final",
            ],
            rows,
        )
    )
    dl = s["download"]
    rejected = ", ".join(f"{r['slug']} ({r['reason']})" for r in dl["rejected"]) or "none"
    print(
        f"\n{dl['sources']} documents in sources.csv, {dl['downloaded']} accepted "
        f"({dl['bytes'] / 1e9:.2f} GB); rejected: {rejected}."
    )
    print(
        f"Header/footer lines removed: {ex['boilerplate_lines_removed']:,}. "
        f"Likely scanned (>50% of pages dropped): {', '.join(ex['likely_scanned']) or 'none'}."
    )

    print("\n**Token funnel** (Tekken tokens)\n")
    print(
        table(
            ["step", "docs", "tokens", "removed"],
            [
                ["extracted", fi["docs_in"], fi["tokens_in"], ""],
                [
                    "quality filter",
                    fi["docs_out"],
                    fi["tokens_out"],
                    f"{1 - fi['tokens_out'] / fi['tokens_in']:.1%}",
                ],
                [
                    "exact dedup",
                    fi["docs_out"] - len(de["exact_duplicates"]),
                    de["tokens_after_exact"],
                    f"{1 - de['tokens_after_exact'] / de['tokens_in']:.1%}",
                ],
                [
                    "near dedup",
                    de["docs_out"],
                    de["tokens_out"],
                    f"{1 - de['tokens_out'] / de['tokens_after_exact']:.1%}",
                ],
                ["train", sp["train"]["docs"], sp["train"]["tokens"], ""],
                [
                    "val",
                    sp["val"]["docs"],
                    sp["val"]["tokens"],
                    f"{sp['val_token_frac']:.1%} of tokens",
                ],
                ["replay (FineWeb-Edu)", rp["docs"], rp["tokens"], "10% of train"],
            ],
        )
    )

    print("\n**Quality filter** (filter.py; first failing rule counted)\n")
    rows = [[r, n, fi["tokens_dropped_by_rule"][r]] for r, n in fi["paragraphs_dropped"].items()]
    rows.append(
        ["**paragraphs in / out**", f"{fi['paragraphs_in']:,} / {fi['paragraphs_out']:,}", ""]
    )
    print(table(["rule", "paragraphs dropped", "tokens dropped"], rows))
    dropped = (
        ", ".join(
            f"{d['slug']} ({d['rule']}, {d['words']:,} words, dictionary {d['dictionary_ratio']})"
            for d in fi["docs_dropped"]
        )
        or "none"
    )
    print(
        f"\nDocuments dropped: {dropped}. Dictionary ratio of kept documents: min "
        f"{fi['dictionary_ratio_kept']['min']}, median {fi['dictionary_ratio_kept']['median']}."
    )

    print("\n**Deduplication** (dedup.py)\n")
    exact = "; ".join(f"{a} = {b}" for a, b in de["exact_duplicates"]) or "none"
    print(
        f"Exact duplicate documents: {exact}. Near-duplicate paragraphs: "
        f"{de['near_dup_paragraphs']:,} of {de['paragraphs_in']:,}; tokens "
        f"{de['tokens_in']:,} -> {de['tokens_out']:,} ({de['token_reduction']:.1%}).\n"
    )
    for title, key in (
        ("Most-duplicated across documents", "top_cross_document"),
        ("Most-duplicated by copies", "top_duplicated"),
    ):
        rows = [
            [x["docs_with_copy"], x["copies_removed"], x["text"][:90].replace("|", "/")]
            for x in de[key]
        ]
        print(f"{title}:\n\n" + table(["docs", "copies removed", "paragraph"], rows) + "\n")

    print("**PII** (pii.py): " + ", ".join(f"{k} {v}" for k, v in pii["replacements"].items()))

    print(
        "\n**Tokenizer fit** (tokenizer_coverage.py, full report in "
        "`data/processed/tokenizer_coverage.md`; tokens per whitespace word)\n"
    )
    roles = [r for r in ("base", "instruct") if r in tk]
    rows = [
        ["corpus tokens"] + [tk[r]["corpus_tokens"] for r in roles],
        ["fertility: domain corpus"] + [tk[r]["fertility"]["corpus"] for r in roles],
        ["fertility: FineWeb-Edu (~1M tokens)"]
        + [tk[r]["fertility"]["fineweb_edu"] for r in roles],
    ]
    rows += [
        [f"fertility: {k} terms ({tk['term_counts'][k]})"]
        + [tk[r]["term_fertility"][k] for r in roles]
        for k in tk["term_counts"]
    ]
    rows.append(
        ["term words with 4+ tokens"]
        + [f"{tk[r]['words_4plus_tokens']} / {tk[r]['term_words']}" for r in roles]
    )
    print(table([""] + [tk[r]["model"] for r in roles], rows))
    print(f"\nBase and Instruct encode identically: {tk['base_instruct_identical']}.\n")
    b = tk[roles[0]]
    print(table(["designation", "tokens"], [[p, n] for p, n in b["probes"].items()]))
    print(
        "\nWorst-fragmented term words: "
        + ", ".join(f"{x['word']} ({x['tokens']})" for x in b["worst_fragmented"][:25])
    )

    print("\n**Split and packing** (split.py)\n")
    rows = [
        [pub, v["train"]["docs"], v["train"]["tokens"], v["val"]["docs"], v["val"]["tokens"]]
        for pub, v in sp["by_publisher"].items()
    ]
    print(table(["publisher", "train docs", "train tokens", "val docs", "val tokens"], rows))
    print(
        f"\nVal documents: {', '.join(sp['val_docs'])}. {sp['eval_docs_in_train']} eval documents "
        f"held in train. Packed at {sp['seq_len']:,}: {sp['packed_seqs']['train']:,} train / "
        f"{sp['packed_seqs']['val']:,} val sequences ({rp['packed_seqs_train_plus_replay']:,} with "
        f"replay); {sp['steps_per_epoch_at_1M_tokens']} optimizer steps per epoch at ~1M tokens/step."
    )
    if sp["publishers_without_val"]:
        print(f"No held-out document for: {', '.join(sp['publishers_without_val'])}.")


if __name__ == "__main__":
    main()
