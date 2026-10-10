"""The SFT set without re-hosting third-party text.

  .venv/bin/python data/scripts/sft_replay_fetch.py          # data/sft/hosted/ -> data/sft/{train,sft_val}.jsonl
  .venv/bin/python data/scripts/sft_replay_fetch.py --strip  # after a rebuild: data/sft/*.jsonl -> data/sft/hosted/

`data/sft/hosted/{train,sft_val}.jsonl` are what git holds: every SFT record, with the 500 Tülu 3
replay records' prompt and completion set to null and their row ids kept (`eid` "replay:<id>",
`replay_source` "<source>#<id>"). The default mode reads the ids, takes those rows' messages from
allenai/tulu-3-sft-mixture (the 6 parquet shards, 1.4 GB, into the HF cache, as
data/scripts/sft_replay.py reads them), fills each record back in the way sft_assemble.replay_record
built it, and writes data/sft/{train,sft_val}.jsonl. Both must match data/sft/SHA256SUMS byte for
byte, or the script exits 1: the training set is exactly the one every SFT row was trained on.

Why: the replay subsets carry their own licences (docs/reproduce.md lists them; No Robots is
CC-BY-NC-4.0), so the repo ships ids and this script rather than the text.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "data/sft"
HOSTED = SFT / "hosted"
FILES = ("train.jsonl", "sft_val.jsonl")
REPO = "allenai/tulu-3-sft-mixture"
SHARDS = [f"data/train-{k:05d}-of-00006.parquet" for k in range(6)]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def strip() -> None:
    for name in FILES:
        rows = read(SFT / name)
        for r in rows:
            if r["format"] == "replay":
                r["prompt"] = r["completion"] = None
        write(HOSTED / name, rows)
        print(
            f"-> {HOSTED / name}: {sum(r['format'] == 'replay' for r in rows)} replay records stripped"
        )


def fetch_messages(ids: set[str]) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    found: dict[str, list[dict]] = {}
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=20_000, columns=["id", "messages"]
        ):
            for r in batch.to_pylist():
                if r["id"] in ids:
                    found[r["id"]] = r["messages"]
    if missing := ids - set(found):
        raise SystemExit(f"{len(missing)} replay ids not in {REPO}, e.g. {sorted(missing)[:3]}")
    return found


def fill() -> None:
    hosted = {name: read(HOSTED / name) for name in FILES}
    ids = {
        r["eid"].removeprefix("replay:")
        for rows in hosted.values()
        for r in rows
        if r["format"] == "replay"
    }
    messages = fetch_messages(ids)
    sums = dict(line.split()[::-1] for line in (SFT / "SHA256SUMS").read_text().splitlines())
    ok = True
    for name, rows in hosted.items():
        for r in rows:
            if r["format"] == "replay":
                user, assistant = messages[r["eid"].removeprefix("replay:")]
                r["prompt"] = [{"role": "user", "content": user["content"]}]
                r["completion"] = [{"role": "assistant", "content": assistant["content"]}]
        write(SFT / name, rows)
        digest = hashlib.sha256((SFT / name).read_bytes()).hexdigest()
        match = digest == sums.get(name)
        ok &= match
        print(
            f"-> {SFT / name}: sha256 {digest[:12]} {'matches' if match else 'DIFFERS from'} SHA256SUMS"
        )
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", action="store_true", help="write data/sft/hosted/ from data/sft/")
    args = ap.parse_args()
    strip() if args.strip else fill()


if __name__ == "__main__":
    main()
