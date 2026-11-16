"""Look inside one PDF the way the ingestion sees it: font sizes, bold text, and what counts as small print.

When a document comes out wrong (a footnote left in the body, a table cell taken for a heading, a missing
paragraph), the cause is nearly always visible in the type sizes of the page. This tool prints them::

    python scripts/inspect_pdf.py eu-gdpr                      # the whole document in numbers, first two pages in detail
    python scripts/inspect_pdf.py eu-gdpr --pages 12-13        # every line of those pages with size, bold and tag
    python scripts/inspect_pdf.py eu-gdpr --find "Article 51"  # on which pages does this text occur?
    python scripts/inspect_pdf.py eu-gdpr --pages 12 --blocks  # and the blocks the ingestion makes of that page

The first argument is a document id from ``data/sources.csv`` (the file is looked up in ``data/raw``) or the path of
any PDF. Nothing is written; the output is meant to be pasted into a message.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from reglens.config import PROJECT_ROOT
from reglens.corpus.manifest import ManifestError, load_manifest
from reglens.ingest import layout, pdfread, structure
from reglens.ingest.pdfread import ExtractionError

MAX_TEXT = 100  # characters of a line shown in the table
MAX_MATCHES = 25
DEFAULT_PAGES = "1-2"


def _visible(text: str) -> str:
    """The text with characters a console cannot show (private-use glyphs, control codes) written as <U+XXXX>."""
    return "".join(
        f"<U+{ord(char):04X}>" if unicodedata.category(char) in ("Co", "Cc", "Cn", "Cs") and char != "\t" else char
        for char in text
    )


def _short(text: str, limit: int = MAX_TEXT) -> str:
    text = _visible(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_pages(spec: str, total: int) -> list[int]:
    """"5" -> [5]; "3-6" -> [3, 4, 5, 6]; "10-" -> from page 10 to the end. Pages outside the document are dropped."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:(-)\s*(\d*))?\s*", spec or "")
    if not match:
        raise ValueError(f"cannot read the page range {spec!r}; write a page (5), a range (3-6) or an open range (10-)")
    first = int(match.group(1))
    if not match.group(2):
        last = first
    else:
        last = int(match.group(3)) if match.group(3) else total
    if first < 1 or last < first:
        raise ValueError(f"the page range {spec!r} is empty or starts before page 1")
    pages = [number for number in range(first, last + 1) if number <= total]
    if not pages:
        raise ValueError(f"the document has only {total} page(s); {spec!r} is beyond the end")
    return pages


def find_file(target: str, raw_dir: Path, manifest: Path) -> Path:
    """The PDF to look at: ``target`` as a path, or as a document id from the manifest."""
    direct = Path(target)
    if direct.is_file():
        return direct
    ids: dict[str, str] = {}
    try:
        ids = {source.id: source.filename for source in load_manifest(manifest)}
    except ManifestError:
        pass
    if target in ids:
        path = raw_dir / ids[target]
        if not path.is_file():
            raise FileNotFoundError(f"{ids[target]} is not in {raw_dir} - run scripts/download_corpus.py first")
        return path
    similar = [identifier for identifier in ids if target.lower() in identifier.lower()][:6]
    hint = f" Did you mean: {', '.join(similar)}?" if similar else ""
    raise FileNotFoundError(f"{target!r} is neither a file nor a document id of the manifest.{hint}")


def _tag(line, body_size: float | None, bold_document: bool) -> str:
    size = getattr(line, "size", None)
    if size is None or body_size is None:
        return "?"
    tags = []
    if size >= structure.HEADING_SIZE_RATIO * body_size:
        tags.append("LARGE")
    elif layout._is_small(line, body_size):
        tags.append("small")
    if getattr(line, "bold", False) and not bold_document:
        tags.append("bold")
    if layout._footnote_match(str(line)) is not None:
        tags.append("number")
    return "+".join(tags)


def _size_table(pdf, out: Callable[[str], None]) -> tuple[float | None, bool]:
    """Print how the characters of the document are spread over font sizes. Returns body size and whether it is mostly bold."""
    by_size: Counter = Counter()
    bold_chars = typeset_chars = all_chars = 0
    for text, styles in zip(pdf.pages, pdf.styles):
        for line in layout.split_lines(text, styles=styles):
            characters = len(line.strip())
            all_chars += characters
            size = getattr(line, "size", None)
            if size:
                typeset_chars += characters
                by_size[round(size * 2) / 2] += characters
                bold_chars += characters if getattr(line, "bold", False) else 0
    typeset_pages = sum(1 for styles in pdf.styles if styles)
    out(f"pages: {len(pdf.pages)}   with readable font sizes: {typeset_pages}   bookmarks: {len(pdf.outline)}")
    if pdf.page_errors:
        out(f"pages that could not be read: {', '.join(pdf.page_errors[:5])}")
    if not typeset_chars:
        out("no font sizes could be read from this file: footnotes cannot be told from body text")
        return None, False
    out("characters by font size (the biggest share is the running text):")
    for size, count in by_size.most_common(8):
        out(f"  {size:5.1f} pt  {100 * count // max(1, typeset_chars):3d}%  {'#' * max(1, round(40 * count / typeset_chars))}")
    laid_out, report = layout.prepare_pages(pdf.pages, pdf.styles)
    bold_share = bold_chars / typeset_chars
    out(
        f"running text: {report.body_size} pt   bold share: {bold_share:.0%}   "
        f"footnotes found: {report.footnotes} on {report.footnote_pages} pages   "
        f"headers/footers removed: {sum(report.margin_lines_removed.values())}   empty pages: {report.empty_pages}"
    )
    return report.body_size, bold_share > structure.BOLD_BODY_SHARE


def _page_table(pdf, number: int, body_size: float | None, bold_document: bool, out: Callable[[str], None]) -> None:
    lines = layout.split_lines(pdf.pages[number - 1], styles=pdf.styles[number - 1])
    typeset = pdf.styles[number - 1] is not None
    out(f"\n--- page {number}: {len(lines)} lines{'' if typeset else ' (font sizes not readable on this page)'} ---")
    out("  line   size  bold  tag             text")
    for index, line in enumerate(lines, start=1):
        size = getattr(line, "size", None)
        shown = f"{size:5.1f}" if size is not None else "    ?"
        flag = "B" if getattr(line, "bold", False) else "-"
        out(f"  {index:4d} {shown}   {flag}    {_tag(line, body_size, bold_document):<14}  {_short(str(line))}")


def _find(pdf, needle: str, out: Callable[[str], None]) -> None:
    pattern = re.compile(r"\s+".join(re.escape(word) for word in needle.split()), re.IGNORECASE)
    hits = 0
    out(f"\npages containing {needle!r}:")
    for number, text in enumerate(pdf.pages, start=1):
        flat = " ".join(text.split())
        found = pattern.search(flat)
        if not found:
            continue
        hits += 1
        if hits <= MAX_MATCHES:
            start = max(0, found.start() - 40)
            out(f"  page {number:4d}: ...{_visible(flat[start : found.end() + 40])}...")
    out(f"  {hits} page(s)" + (f" (first {MAX_MATCHES} shown)" if hits > MAX_MATCHES else ""))


def _blocks(pdf, wanted: Sequence[int], out: Callable[[str], None]) -> None:
    pages, _report = layout.prepare_pages(pdf.pages, pdf.styles)
    blocks, _structure = structure.build_blocks(pages, pdf.outline)
    chosen = set(wanted)
    out("\nblocks the ingestion makes of these pages:")
    shown = 0
    for block in blocks:
        first, last = block.page or 0, block.page_end or block.page or 0
        if not chosen.intersection(range(first, last + 1)):
            continue
        shown += 1
        kind = f"{block.kind} L{block.level}" if block.kind == "heading" else block.kind
        span = f"p{first}" if first == last else f"p{first}-{last}"
        out(f"  [{kind:<11} {span:<7}] {_short(block.text, 150)}")
    if not shown:
        out("  (none)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspect_pdf", description="Show how the ingestion sees one PDF.")
    parser.add_argument("target", help="a document id from data/sources.csv, or the path of a PDF")
    parser.add_argument("--pages", default=DEFAULT_PAGES, help=f"pages to show line by line, e.g. 5, 3-6 or 10- (default {DEFAULT_PAGES})")
    parser.add_argument("--find", metavar="TEXT", help="list the pages that contain this text")
    parser.add_argument("--blocks", action="store_true", help="also show the blocks (headings, paragraphs, footnotes) made of those pages")
    parser.add_argument("--summary-only", action="store_true", help="only the numbers for the whole document")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "sources.csv"))
    return parser


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = find_file(args.target, Path(args.raw_dir), Path(args.manifest))
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"{path.name} is not a PDF; this tool looks at font sizes, which web pages do not have")
        pdf = pdfread.read_pdf(path)
        wanted = [] if args.summary_only else parse_pages(args.pages, len(pdf.pages))
    except (FileNotFoundError, ValueError) as exc:
        out(f"Problem: {exc}")
        return 2
    except ExtractionError as exc:
        out(f"Problem: {path.name} could not be read: {exc}")
        return 1
    out(f"{path.name}")
    body_size, bold_document = _size_table(pdf, out)
    if args.find:
        _find(pdf, args.find, out)
    for number in wanted:
        _page_table(pdf, number, body_size, bold_document, out)
    if args.blocks and wanted:
        _blocks(pdf, wanted, out)
    return 0


def run() -> int:  # pragma: no cover - thin wrapper for the script
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass
    return main()
