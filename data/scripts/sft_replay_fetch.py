"""Third-party Tülu 3 text, kept out of git: row ids and hashes in the repo, text from the source.

  .venv/bin/python data/scripts/sft_replay_fetch.py            # rebuild the inputs from data/sft/hosted/, eval/hosted/
  .venv/bin/python data/scripts/sft_replay_fetch.py --strip    # after a rebuild: inputs -> their hosted copies
  .venv/bin/python data/scripts/sft_replay_fetch.py --results  # after pulling results: third-party fields -> sha256

**Inputs** (rebuilt, gitignored, checked byte for byte):
- `data/sft/{train,sft_val}.jsonl`, the SFT set: git holds `data/sft/hosted/`, every record with
  the 500 Tülu 3 replay records' prompt and completion set to null and their row ids kept (`eid`
  "replay:<id>"). Rebuilt they must match data/sft/SHA256SUMS, the hashes every SFT run was
  trained on.
- `eval/{diversity_prompts,sft_template_prompts}.jsonl`, the probe prompts: git holds `eval/hosted/`
  with the 50 held-out Tülu prompts (format "general") and the replay template prompt blanked;
  rebuilt they must match eval/hosted/SHA256SUMS.

The default mode takes the rows' messages from allenai/tulu-3-sft-mixture (the 6 parquet shards,
1.4 GB, into the HF cache, as data/scripts/sft_replay.py reads them) and fills each record back in
the way sft_assemble.replay_record and eval/diversity.py built it. A mismatch exits 1.

**Results** (outputs, rewritten in place): the prompt ids and references sampling and training
stored for those rows (`prompt_token_ids`, `reference`, `prompt`) become `<field>_sha256`, the
sha256 of the field's JSON, so the text isn't re-hosted and tests/test_template.py still checks
the served prompt ids against the trainer's. The models' own outputs stay.

**Withdrawn rows** (46 math records whose licence restricts models trained on them, withdrawn
after training) keep only `sha256:<hex>` digests of their ids: `eid` and `replay_source` in the
SFT set, `id` and `source` in the probes and the results. The fill finds them by hashing each
mixture row's id and restores the ids, so the rebuilt files still match their hashes; `--strip` and
`--results` keep any row whose id digest the hosted copies hold hashed.

Why: the replay subsets carry their own licences (docs/reproduce.md lists them; No Robots is
CC-BY-NC-4.0), so the repo ships ids and this script rather than the text.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "data/sft"
EVAL = ROOT / "eval"
INPUTS = [  # (built dir, hosted dir, file, hash list)
    (SFT, SFT / "hosted", "train.jsonl", SFT / "SHA256SUMS"),
    (SFT, SFT / "hosted", "sft_val.jsonl", SFT / "SHA256SUMS"),
    (EVAL, EVAL / "hosted", "diversity_prompts.jsonl", EVAL / "hosted/SHA256SUMS"),
    (EVAL, EVAL / "hosted", "sft_template_prompts.jsonl", EVAL / "hosted/SHA256SUMS"),
]
RESULTS = [
    "results/runs/*/eval_generations.jsonl",
    "results/runs/*/samples/*.jsonl",
    "results/serve/*/served_check*.jsonl",
]
HASHED = ("prompt_token_ids", "reference", "prompt")
PREFIX = "sha256:"  # a withdrawn row's id, as the hosted copies keep it
REPO = "allenai/tulu-3-sft-mixture"
SHARDS = [f"data/train-{k:05d}-of-00006.parquet" for k in range(6)]


def third_party(r: dict) -> bool:
    """A Tülu 3 row: an SFT replay record, the replay template prompt, or a held-out probe."""
    return r.get("format") in ("replay", "general") or str(r.get("id", "")).startswith("replay:")


def tulu_id(r: dict) -> str:
    return (r.get("eid") or r["id"]).removeprefix("replay:")


def digest(s: str) -> str:
    return PREFIX + hashlib.sha256(s.encode()).hexdigest()


def key(r: dict) -> str:
    """The id the hosted copy keeps: an SFT record's eid, a probe's id (a digest if withdrawn)."""
    return r.get("eid") or r["id"]


def withdrawn() -> set[str]:
    """The id digests the hosted copies keep in place of withdrawn rows' ids."""
    rows = (r for _, hosted, name, _ in INPUTS for r in read(hosted / name))
    return {
        k
        for r in rows
        if isinstance(k := r.get("eid") or r.get("id"), str) and k.startswith(PREFIX)
    }


def hide(r: dict, hidden: set[str]) -> int:
    """Replace a withdrawn row's ids with their digests; 1 if it was one."""
    if r.get("eid") and digest(r["eid"]) in hidden:
        r["replay_source"], r["eid"] = digest(r["replay_source"]), digest(r["eid"])
        return 1
    if isinstance(r.get("id"), str) and digest(r["id"]) in hidden:
        r["id"] = digest(r["id"])
        if r.get("source"):
            r["source"] = digest(r["source"])
        return 1
    return 0


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip() -> None:
    sums: dict[Path, list[str]] = {}
    hidden = withdrawn()
    for built, hosted, name, sumfile in INPUTS:
        rows = read(built / name)
        for r in rows:
            if third_party(r):
                for k in ("prompt", "completion"):
                    if k in r:
                        r[k] = None
                hide(r, hidden)
        write(hosted / name, rows)
        if sumfile.parent == hosted:  # the probes' hashes are written here; SFT's are frozen
            sums.setdefault(sumfile, []).append(f"{sha(built / name)}  {name}")
        print(f"-> {hosted / name}: {sum(map(third_party, rows))} third-party rows blanked")
    for sumfile, lines in sums.items():
        sumfile.write_text("\n".join(lines) + "\n")


def fetch_messages(ids: set[str], hidden: set[str]) -> dict[str, tuple[str, str, list[dict]]]:
    """(id, source, messages) of each wanted row, by its id or, if withdrawn, its id digest."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    found: dict[str, tuple[str, str, list[dict]]] = {}
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        cols = ["id", "source", "messages"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=cols):
            for r in batch.to_pylist():
                for k in (r["id"], digest(r["id"]), digest(f"replay:{r['id']}")):
                    if k in ids or k in hidden:
                        found[k] = (r["id"], r["source"], r["messages"])
    if missing := (ids | hidden) - set(found):
        raise SystemExit(f"{len(missing)} replay ids not in {REPO}, e.g. {sorted(missing)[:3]}")
    return found


def fill() -> None:
    hosted = {(built, name): read(h / name) for built, h, name, _ in INPUTS}
    hidden = withdrawn()
    third = [r for rows in hosted.values() for r in rows if third_party(r)]
    found = fetch_messages({tulu_id(r) for r in third if key(r) not in hidden}, hidden)
    ok = True
    for built, _, name, sumfile in INPUTS:
        rows = hosted[(built, name)]
        for r in rows:
            if not third_party(r):
                continue
            if key(r) in hidden:  # a withdrawn row: its ids come back from the source
                tid, source, (user, assistant) = found[key(r)]
                if "eid" in r:
                    r["replay_source"], r["eid"] = f"{source}#{tid}", f"replay:{tid}"
                else:
                    r["id"], r["source"] = tid, source
            else:
                user, assistant = found[tulu_id(r)][2]
            if "completion" in r:  # an SFT record
                r["prompt"] = [{"role": "user", "content": user["content"]}]
                r["completion"] = [{"role": "assistant", "content": assistant["content"]}]
            else:  # a probe prompt: the user turn's text
                r["prompt"] = user["content"]
        write(built / name, rows)
        sums = dict(line.split()[::-1] for line in sumfile.read_text().splitlines())
        match = sha(built / name) == sums.get(name)
        ok &= match
        print(f"-> {built / name}: {'matches' if match else 'DIFFERS from'} {sumfile.name}")
    raise SystemExit(0 if ok else 1)


def hash_results() -> None:
    hidden = withdrawn()
    for pattern in RESULTS:
        for path in sorted(ROOT.glob(pattern)):
            rows, n = read(path), 0
            for r in rows:
                if not third_party(r):
                    continue
                n += hide(r, hidden)
                for k in HASHED:
                    if r.get(k) is not None:
                        blob = json.dumps(r[k], ensure_ascii=False).encode()
                        r[f"{k}_sha256"], r[k] = hashlib.sha256(blob).hexdigest(), None
                        n += 1
            if n:
                write(path, rows)
                print(f"-> {path.relative_to(ROOT)}: {n} fields hashed")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--strip", action="store_true", help="inputs -> their hosted copies")
    mode.add_argument("--results", action="store_true", help="third-party result fields -> sha256")
    args = ap.parse_args()
    strip() if args.strip else hash_results() if args.results else fill()


if __name__ == "__main__":
    main()
