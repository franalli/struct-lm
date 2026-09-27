"""Fetch every PDF in data/sources.csv into data/raw/<slug>.pdf; record sha256 and page count.

data/raw/ is not committed. sources.csv is the lineage record: slug, publisher, title, url, and
the sha256 and page count of the exact file that was processed (written back by this script).

Files that already exist are skipped, so anything the script can't fetch (e.g. FEMA, which
blocks automated clients) can be downloaded in a browser and dropped into data/raw/ as either
<slug>.pdf or its original filename; the next run renames it to <slug>.pdf and hashes it.

A file is rejected (its row kept, with no sha256, so extract.py skips it) when it isn't a PDF
(a block or landing page) or has no readable pages. There's no size floor: short documents are
judged on their text by filter.py's word minimum, not on file size.
"""

import argparse
import hashlib
import time
from pathlib import Path

import requests
from common import (
    HEADERS,
    RAW,
    SOURCES,
    pdf_parts,
    pdf_path,
    read_sources,
    update_stats,
    url_filename,
    write_sources,
)

ATTEMPTS = 3
RETRY_SLEEP_S = 20  # USACE intermittently 403s, then serves the same file a minute later
POLITE_SLEEP_S = 2  # between downloads; USACE rate-limits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path) -> None:
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("not a PDF (likely a block or landing page)")


def fetch(url: str, dest: Path) -> None:
    """Stream to <dest>.part, check it, then rename, so a partial download or an HTML error
    page never masquerades as a finished file on the next run."""
    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        check(tmp)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_with_retry(url: str, dest: Path) -> None:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fetch(url, dest)
        except (requests.RequestException, ValueError) as e:
            if attempt == ATTEMPTS:
                raise
            print(f"  attempt {attempt} failed ({e}); retrying in {RETRY_SLEEP_S}s")
            time.sleep(RETRY_SLEEP_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=SOURCES)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    sources = read_sources(args.sources)
    failed, rejected = [], []
    for i, src in enumerate(sources, 1):
        dest = pdf_path(src["slug"])
        # a browser download keeps the URL's filename (e.g. EM_1110-2-2104.pdf); adopt it
        browser_name = RAW / url_filename(src["url"])
        if not dest.exists() and browser_name.suffix == ".pdf" and browser_name.exists():
            browser_name.rename(dest)
        status = "cached"
        if not dest.exists():
            print(f"[{i}/{len(sources)}] {src['slug']} <- {src['url']}")
            try:
                fetch_with_retry(src["url"], dest)
                status = "downloaded"
                time.sleep(POLITE_SLEEP_S)
            except (requests.RequestException, ValueError) as e:
                print(f"  FAILED: {e}")
                src["sha256"] = src["pages"] = ""
                failed.append(src)
                continue
        try:
            check(dest)
            parts = pdf_parts(dest)
            src["pages"] = sum(d.page_count for d in parts)  # a portfolio counts its parts' pages
            for d in parts:
                d.close()
            if not src["pages"]:
                # e.g. FEMA serves P-58-4/P-58-5 cut short of the length their headers declare
                raise ValueError("no readable pages (truncated or corrupt file)")
        except Exception as e:  # noqa: BLE001  unreadable, or a bad manual drop-in
            # the row stays as a record, but with no sha256 extract.py skips it
            print(f"  REJECTED {dest}: {e}")
            src["sha256"] = src["pages"] = ""
            rejected.append({"slug": src["slug"], "reason": str(e)})
            continue
        src["sha256"] = sha256(dest)
        print(f"[{i}/{len(sources)}] {src['slug']}: {status} ({dest.stat().st_size / 1e6:.1f} MB)")
        # written after every file, so an interrupted run keeps the hashes it already computed
        write_sources(sources, args.sources)

    write_sources(sources, args.sources)
    ok = [s for s in sources if s["sha256"]]
    update_stats(
        "download",
        {
            "sources": len(sources),
            "downloaded": len(ok),
            "bytes": sum(pdf_path(s["slug"]).stat().st_size for s in ok),
            "rejected": rejected,
            "missing": [s["slug"] for s in failed],
        },
    )
    print(f"\n{len(ok)}/{len(sources)} in {RAW}/; sha256 + pages -> {args.sources}")
    for r in rejected:
        print(f"  rejected {r['slug']}: {r['reason']}")
    if failed:
        print("Download these in a browser and save to the path shown, then re-run:")
        for src in failed:
            print(f"  {pdf_path(src['slug'])}  <-  {src['url']}")


if __name__ == "__main__":
    main()
