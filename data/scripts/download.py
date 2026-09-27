"""Fetch every PDF in sources.csv into data/raw/<slug>.pdf and write a sha256 manifest.

data/raw/ is not committed; sources.csv plus data/raw/manifest.json is the reproducible
record (sources.csv is also the corpus table in the write-up).

Files that already exist are skipped, so anything the script can't fetch (e.g. FEMA,
which blocks automated clients) can be downloaded in a browser and dropped into
data/raw/ as either <slug>.pdf or its original filename; the next run picks it up and hashes it.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
RETRY_SLEEP_S = 20  # USACE intermittently 403s, then serves the same file a minute later


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream to <dest>.part, verify it's a PDF, then rename, so a partial or HTML
    error page never masquerades as a finished download on the next run."""
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with tmp.open("rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise ValueError("response is not a PDF (likely a block or landing page)")
    tmp.rename(dest)


def fetch_with_retry(url: str, dest: Path) -> None:
    try:
        fetch(url, dest)
    except (requests.RequestException, ValueError) as e:
        print(f"  retrying in {RETRY_SLEEP_S}s after: {e}")
        time.sleep(RETRY_SLEEP_S)
        fetch(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(Path(__file__).with_name("sources.csv")))
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.sources, newline="") as f:
        sources = list(csv.DictReader(f))

    manifest, failed = [], []
    for i, src in enumerate(sources, 1):
        dest = out / f"{src['slug']}.pdf"
        # a browser download keeps the URL's filename (e.g. EM_1110-2-2104.pdf); accept it as-is
        browser_name = out / Path(urlparse(src["url"]).path).name
        if not dest.exists() and browser_name.suffix == ".pdf" and browser_name.exists():
            dest = browser_name
        status = "cached"
        if not dest.exists():
            print(f"[{i}/{len(sources)}] {src['slug']} <- {src['url']}")
            try:
                fetch_with_retry(src["url"], dest)
                status = "downloaded"
            except (requests.RequestException, ValueError) as e:
                print(f"  FAILED: {e}")
                failed.append(src)
                continue
        size_mb = dest.stat().st_size / 1e6
        print(f"[{i}/{len(sources)}] {src['slug']}: {status} ({size_mb:.1f} MB)")
        manifest.append(
            {**src, "path": str(dest), "bytes": dest.stat().st_size, "sha256": sha256(dest)}
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(sources)} in {out}/, manifest.json written")
    if failed:
        print("Download these in a browser and save to the path shown, then re-run:")
        for src in failed:
            print(f"  {out / (src['slug'] + '.pdf')}  <-  {src['url']}")


if __name__ == "__main__":
    main()
