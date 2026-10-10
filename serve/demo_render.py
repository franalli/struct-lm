"""Render the demo's terminal capture to the README's GIF and a master MP4.

  .venv/bin/python serve/demo_render.py video-work/struct-lm/demo.cast   # -> docs/demo.gif + video-work/struct-lm/demo.mp4

The capture is an asciicast v2 file: serve/demo.py's stdout as it streamed from the served model
(serve/modal_demo.py), each chunk with its arrival time. This replays it on a virtual 100-column
terminal, handling the codes demo.py prints (clear screen, bold, dim, reset); text wraps by
character, as a terminal's does.
- **The GIF** gets one frame per change, held until the next one, so the streaming keeps its real
  timing. demo.py clears the screen at the start and the end, so the first and last frames are
  the same blank one and the GIF loops cleanly.
- **The MP4 master** is the same frames at 2x scale and 60 fps (ffmpeg).
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/Menlo.ttc"
BG, FG, BOLD, DIM, CURSOR = "#16181d", "#d8dade", "#ffffff", "#7c818c", "#5c6370"
ANSI = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


class Screen:
    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.clear()
        self.style = "fg"

    def clear(self) -> None:
        self.lines: list[list[tuple[str, str]]] = [[]]

    def write(self, text: str) -> None:
        pos = 0
        for m in ANSI.finditer(text):
            self.put(text[pos : m.start()])
            args, cmd = m.group(1), m.group(2)
            if cmd == "J" and args == "2":
                self.clear()
            elif cmd == "m":
                self.style = {"1": "bold", "2": "dim"}.get(args, "fg")
            pos = m.end()
        self.put(text[pos:])

    def put(self, text: str) -> None:
        for ch in text:
            if ch == "\n":
                self.lines.append([])
            elif ch >= " ":
                if len(self.lines[-1]) >= self.cols:
                    self.lines.append([])
                self.lines[-1].append((ch, self.style))
        self.lines = self.lines[-self.rows :]

    def render(self, scale: int) -> Image.Image:
        size = 15 * scale
        font = ImageFont.truetype(FONT, size, index=0)
        bold = ImageFont.truetype(FONT, size, index=1)
        cw = int(font.getbbox("M")[2])
        lh, pad = int(size * 1.35), 18 * scale
        img = Image.new("RGB", (self.cols * cw + 2 * pad, self.rows * lh + 2 * pad), BG)
        draw = ImageDraw.Draw(img)
        colour = {"fg": FG, "bold": BOLD, "dim": DIM}
        for y, line in enumerate(self.lines):
            for x, (ch, style) in enumerate(line):
                draw.text((pad + x * cw, pad + y * lh), ch, fill=colour[style],
                          font=bold if style == "bold" else font)  # fmt: skip
        y = len(self.lines) - 1
        x = len(self.lines[-1])
        draw.rectangle(
            [pad + x * cw, pad + y * lh, pad + (x + 1) * cw - 1, pad + y * lh + size], fill=CURSOR
        )
        return img


def frames(cast: Path, scale: int) -> list[tuple[Image.Image, float]]:
    """(frame, seconds held) for every change in the capture, with a short blank lead-in and tail."""
    lines = cast.read_text().splitlines()
    head = json.loads(lines[0])
    events = [json.loads(line) for line in lines[1:]]
    screen = Screen(head["width"], head["height"])
    out: list[tuple[Image.Image, float]] = [(screen.render(scale), 0.6)]
    for k, (t, kind, text) in enumerate(events):
        if kind != "o":
            continue
        screen.write(text)
        nxt = events[k + 1][0] if k + 1 < len(events) else t + 0.6
        out.append((screen.render(scale), max(nxt - t, 0.02)))
    return out


def main() -> None:
    cast = Path(sys.argv[1])
    gif = REPO / "docs/demo.gif"
    small = frames(cast, 1)
    first, *rest = (f.quantize(colors=16) for f, _ in small)
    first.save(gif, save_all=True, append_images=rest, loop=0, optimize=True,
               duration=[round(d * 1000) for _, d in small], disposal=1)  # fmt: skip
    print(f"-> {gif.relative_to(REPO)}: {len(small)} frames, {gif.stat().st_size / 1e6:.2f} MB")
    if not shutil.which("ffmpeg"):
        return
    mp4 = cast.with_suffix(".mp4")
    with tempfile.TemporaryDirectory() as tmp:
        listing, path = [], Path(tmp) / "0000.png"
        for k, (img, held) in enumerate(frames(cast, 2)):
            path = Path(tmp) / f"{k:04d}.png"
            img.save(path)
            listing += [f"file '{path}'", f"duration {held:.4f}"]
        listing.append(f"file '{path}'")  # the concat demuxer drops the last duration otherwise
        (Path(tmp) / "list.txt").write_text("\n".join(listing) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                        str(Path(tmp) / "list.txt"), "-vf", "fps=60,format=yuv420p", "-c:v", "libx264",
                        "-crf", "18", str(mp4)], check=True)  # fmt: skip
    print(f"-> {mp4}: master, 2x, 60 fps")


if __name__ == "__main__":
    main()
