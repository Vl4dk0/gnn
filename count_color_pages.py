#!/usr/bin/env python3
"""Count pages in thesis.pdf that contain colored pixels.

The script rasterizes each PDF page with Poppler's ``pdftoppm`` command,
parses the generated binary PPM image directly, and treats a page as colored
when enough pixels have meaningfully different RGB channel values.

Usage:
    python3 thesis/scripts/count_color_pages.py
    python3 thesis/scripts/count_color_pages.py --verbose
    python3 thesis/scripts/count_color_pages.py path/to/file.pdf --dpi 100
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
DEFAULT_PDF = THESIS_DIR / "thesis.pdf"


@dataclass(frozen=True)
class PageStats:
    page: int
    width: int
    height: int
    total_pixels: int
    colored_pixels: int
    color_ratio: float
    max_channel_delta: int
    is_colored: bool


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(
            f"error: {name!r} not found on PATH. Install Poppler "
            "(for example, `brew install poppler` on macOS)."
        )
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def page_count(pdf: Path, pdfinfo: str) -> int:
    proc = _run([pdfinfo, str(pdf)])
    if proc.returncode != 0:
        sys.exit(f"error: pdfinfo failed for {pdf}:\n{proc.stderr.strip()}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    if not match:
        sys.exit(f"error: could not read page count from pdfinfo output for {pdf}")
    return int(match.group(1))


def _next_ppm_token(data: bytes, offset: int) -> tuple[str, int]:
    whitespace = b" \t\r\n\v\f"

    while offset < len(data):
        byte = data[offset]
        if byte in whitespace:
            offset += 1
            continue
        if byte == ord("#"):
            newline = data.find(b"\n", offset)
            offset = len(data) if newline == -1 else newline + 1
            continue
        break

    start = offset
    while offset < len(data) and data[offset] not in whitespace:
        offset += 1

    if start == offset:
        raise ValueError("unexpected end of PPM header")
    return data[start:offset].decode("ascii"), offset


def _ppm_pixels(path: Path) -> tuple[int, int, memoryview]:
    data = path.read_bytes()
    magic, offset = _next_ppm_token(data, 0)
    if magic != "P6":
        raise ValueError(f"{path} is {magic!r}, expected binary PPM P6")

    width_s, offset = _next_ppm_token(data, offset)
    height_s, offset = _next_ppm_token(data, offset)
    maxval_s, offset = _next_ppm_token(data, offset)

    width = int(width_s)
    height = int(height_s)
    maxval = int(maxval_s)
    if maxval != 255:
        raise ValueError(f"{path} has maxval {maxval}, expected 255")

    if offset >= len(data) or data[offset] not in b" \t\r\n\v\f":
        raise ValueError(f"{path} has a malformed PPM header")
    offset += 1

    expected = width * height * 3
    pixels = memoryview(data)[offset:]
    if len(pixels) != expected:
        raise ValueError(
            f"{path} has {len(pixels)} pixel bytes, expected {expected}"
        )
    return width, height, pixels


def analyze_ppm(
    page: int,
    ppm: Path,
    channel_delta: int,
    min_color_pixels: int,
    min_color_ratio: float,
) -> PageStats:
    width, height, pixels = _ppm_pixels(ppm)

    colored_pixels = 0
    max_channel_delta = 0
    for index in range(0, len(pixels), 3):
        red = pixels[index]
        green = pixels[index + 1]
        blue = pixels[index + 2]
        delta = max(red, green, blue) - min(red, green, blue)
        if delta > max_channel_delta:
            max_channel_delta = delta
        if delta >= channel_delta:
            colored_pixels += 1

    total_pixels = width * height
    color_ratio = colored_pixels / total_pixels if total_pixels else 0.0
    is_colored = (
        colored_pixels >= min_color_pixels and color_ratio >= min_color_ratio
    )

    return PageStats(
        page=page,
        width=width,
        height=height,
        total_pixels=total_pixels,
        colored_pixels=colored_pixels,
        color_ratio=color_ratio,
        max_channel_delta=max_channel_delta,
        is_colored=is_colored,
    )


def render_page(pdf: Path, page: int, dpi: int, pdftoppm: str, out_dir: Path) -> Path:
    prefix = out_dir / f"page-{page}"
    proc = _run(
        [
            pdftoppm,
            "-q",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            str(pdf),
            str(prefix),
        ]
    )
    if proc.returncode != 0:
        sys.exit(f"error: pdftoppm failed on page {page}:\n{proc.stderr.strip()}")

    ppm = prefix.with_suffix(".ppm")
    if not ppm.exists():
        sys.exit(f"error: expected rendered file {ppm}, but it was not created")
    return ppm


def analyze_pdf(
    pdf: Path,
    dpi: int,
    channel_delta: int,
    min_color_pixels: int,
    min_color_ratio: float,
) -> list[PageStats]:
    pdfinfo = _tool("pdfinfo")
    pdftoppm = _tool("pdftoppm")
    pages = page_count(pdf, pdfinfo)

    stats: list[PageStats] = []
    with tempfile.TemporaryDirectory(prefix="thesis-color-pages-") as tmp:
        out_dir = Path(tmp)
        for page in range(1, pages + 1):
            ppm = render_page(pdf, page, dpi, pdftoppm, out_dir)
            try:
                stats.append(
                    analyze_ppm(
                        page,
                        ppm,
                        channel_delta,
                        min_color_pixels,
                        min_color_ratio,
                    )
                )
            except ValueError as exc:
                sys.exit(f"error: could not analyze page {page}: {exc}")
            finally:
                ppm.unlink(missing_ok=True)
    return stats


def _page_list(pages: list[int]) -> str:
    return ", ".join(str(page) for page in pages) if pages else "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rasterize a PDF and count pages with colored pixels."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to analyze (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="rasterization DPI; higher is slower but more precise (default: 72)",
    )
    parser.add_argument(
        "--channel-delta",
        type=int,
        default=20,
        help=(
            "minimum max(R,G,B)-min(R,G,B) for a pixel to count as colored "
            "(default: 20)"
        ),
    )
    parser.add_argument(
        "--min-color-pixels",
        type=int,
        default=100,
        help="minimum colored pixels required for a colored page (default: 100)",
    )
    parser.add_argument(
        "--min-color-ratio",
        type=float,
        default=0.0002,
        help="minimum colored-pixel ratio required for a colored page (default: 0.0002)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-page color statistics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON output",
    )
    args = parser.parse_args()

    pdf = args.pdf.expanduser().resolve()
    if not pdf.exists():
        sys.exit(f"error: PDF not found: {pdf}")

    stats = analyze_pdf(
        pdf=pdf,
        dpi=args.dpi,
        channel_delta=args.channel_delta,
        min_color_pixels=args.min_color_pixels,
        min_color_ratio=args.min_color_ratio,
    )
    colored_pages = [item.page for item in stats if item.is_colored]

    if args.json:
        print(
            json.dumps(
                {
                    "pdf": str(pdf),
                    "dpi": args.dpi,
                    "channel_delta": args.channel_delta,
                    "min_color_pixels": args.min_color_pixels,
                    "min_color_ratio": args.min_color_ratio,
                    "page_count": len(stats),
                    "colored_page_count": len(colored_pages),
                    "colored_pages": colored_pages,
                    "pages": [asdict(item) for item in stats],
                },
                indent=2,
            )
        )
        return 0

    print(f"PDF: {pdf}")
    print(f"Pages analyzed: {len(stats)} at {args.dpi} DPI")
    print(f"Colored pages: {len(colored_pages)}")
    print(f"Colored page numbers: {_page_list(colored_pages)}")

    if args.verbose:
        print()
        print("Per-page stats:")
        for item in stats:
            marker = "color" if item.is_colored else "gray "
            print(
                f"{item.page:>3}: {marker}  "
                f"{item.colored_pixels:>7} colored pixels  "
                f"{item.color_ratio:>8.5%}  "
                f"max delta {item.max_channel_delta:>3}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
