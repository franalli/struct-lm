"""Add every PDF linked from an index page to data/sources.csv.

  crawl_index.py https://www.fhwa.dot.gov/bridge/steel/ --pattern '/(hif|nhi)\\d' --publisher FHWA
  crawl_index.py usace_em_index_page1.html --base https://www.publications.usace.army.mil/ \\
      --pattern 'EM[_ ]1110-2-' --publisher USACE

The index may be a URL or an HTML file saved from a browser (USACE's Akamai filter returns 403 to
scripts, so its index pages are saved from Chrome and --base resolves their relative links).
A link is kept when its URL (percent-decoded) ends in .pdf and matches --pattern. The title is
the text of the table row holding the link (index tables put the number, title and date in
cells), else the link text. The slug is <publisher>-<filename stem>, lowercased with runs of
other characters turned into "-" (EM_1110-2-2104.pdf -> usace-em-1110-2-2104). URLs and slugs
already in sources.csv are skipped, so re-running is harmless. Rows get sha256/pages when
download.py fetches them.
"""

import argparse
import html
import re
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from common import HEADERS, SOURCES, read_sources, url_filename, write_sources

LINK = re.compile(
    r"""<a\s[^>]*?href=["']([^"']+?\.pdf)["'][^>]*>(.*?)</a>""", re.IGNORECASE | re.DOTALL
)
ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)


def text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def slugify(publisher: str, url: str) -> str:
    stem = Path(url_filename(url)).stem
    return re.sub(r"[^a-z0-9]+", "-", f"{publisher}-{stem}".lower()).strip("-")


def links(page: str) -> list[tuple[str, str]]:
    """(href, title) for every .pdf link; the title is its table row's cells when it has one."""
    out, in_rows = [], set()
    for row in ROW.findall(page):
        cells = [c for c in (text(c) for c in CELL.findall(row)) if c]
        for href, _ in LINK.findall(row):
            out.append((href, " | ".join(cells)))
            in_rows.add(href)
    out += [(href, text(t)) for href, t in LINK.findall(page) if href not in in_rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", help="index page URL, or a saved .html file")
    ap.add_argument("--pattern", required=True, help="regex the PDF URL must match")
    ap.add_argument("--publisher", required=True)
    ap.add_argument("--base", help="base URL for relative links (default: the index URL)")
    args = ap.parse_args()

    if Path(args.index).exists():
        page, base = Path(args.index).read_text(errors="replace"), args.base
        if not base:
            ap.error("--base is required with a saved HTML file")
    else:
        r = requests.get(args.index, headers=HEADERS, timeout=30)
        r.raise_for_status()
        page, base = r.text, args.base or args.index

    rows = read_sources()
    seen_urls, seen_slugs = {r["url"] for r in rows}, {r["slug"] for r in rows}
    pattern = re.compile(args.pattern, re.IGNORECASE)
    found = links(page)
    added = 0
    for href, title in found:
        url = urljoin(base, html.unescape(href.strip()))
        slug = slugify(args.publisher, url)
        if not pattern.search(unquote(url)) or url in seen_urls or slug in seen_slugs:
            continue
        rows.append({"slug": slug, "publisher": args.publisher, "title": title or slug, "url": url})
        seen_urls.add(url)
        seen_slugs.add(slug)
        added += 1
        print(f"  + {slug:40} {(title or slug)[:70]}")

    write_sources(rows)
    print(f"{len(found)} PDF links, {added} new rows -> {SOURCES} ({len(rows)} total)")


if __name__ == "__main__":
    main()
