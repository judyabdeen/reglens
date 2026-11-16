"""Page-level clean-up of extracted PDF text: running headers and footers, page numbers, table-of-contents
lines and footnotes.

The input is one string per PDF page, exactly as the reader returns it, plus - when the PDF allows it - the
typography of every line (font size and weight). The output separates what belongs to the flowing text of the
document ("body") from what belongs to the page furniture:

* lines that repeat at the top or bottom of many pages (running titles, "Page 4 of 80", "L 119/32 EN Official
  Journal ...") are removed, because they would otherwise end up in the middle of sentences that continue on the
  next page. Some publishers' PDFs store the header in the middle of the page text or glue it to the end of a
  sentence, so a header that contains a number (a date, a page counter) is also cut out wherever it occurs;
* footnotes are split off so a paragraph that continues on the next page is not interrupted by them; they are
  kept (as their own blocks) because legal footnotes often carry real information. A footnote is recognised by
  its type: a block of small print at the foot of the page that starts with a note number. Text patterns alone
  cannot do this - a numbered list "1 ... 2 ..." or the recitals "(17) ... (18) ..." look exactly like footnotes -
  so without type sizes nothing is treated as a footnote.

Nothing in the body is ever dropped except those lines, and every removal is counted for the audit.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from reglens.ingest.models import TextLine
from reglens.ingest.textclean import normalize_chars, repair_footnote_markers

MARGIN_LINES = 3  # how many lines at the top and at the bottom of a page can be page furniture
MARGIN_MIN_PAGES = 3  # a line must repeat on at least this many pages ...
MARGIN_MIN_FRACTION = 0.10  # ... and on this share of all pages
MARGIN_MAX_CHARS = 120
MAX_NOTE_FRACTION = 0.8  # a block that fills more of the page than this is not a footnote zone but a page in small print
SMALL_RATIO = 0.93  # a footnote line is set at most this fraction of the size of the running text ...
MIN_SIZE_GAP = 0.6  # ... and at least this many points smaller, so rounding noise in the sizes never counts
MIN_TYPESET_SHARE = 0.5  # the sizes are used only if they are known for at least this share of the text
MIN_BODY_SHARE = 0.3  # a size that carries at least this share of the typeset text can be the running text
MAX_BODY_SIZE_JUMP = 1.5  # ... unless it is more than this much larger than the commonest size
FURNITURE_MIN_CHARS = 14  # a header must be at least this long to be cut out of the middle of a line

# Lines that name a part of a legal text must never be removed as "repeated page furniture".
STRUCTURAL_LINE = re.compile(
    r"^\s*(?:article|chapter|section|part|title|annex|appendix|schedule|recital|§)\s*[\dIVXLCivxlc.]+\b",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(
    r"^\s*(?:\d{1,4}|[ivxlc]{1,7}|(?:[Pp]age\s+)?\d{1,4}\s*(?:of|/)\s*\d{1,4}|[Pp]age\s+\d{1,4})\s*$"
)
_TOC_LINE = re.compile(r"^.{3,}?(?:\.{5,}|(?:\. ){5,})\s*[\divxlc]{1,5}\s*$")
_FOOTNOTE_START = re.compile(r"^\s*(?:\((\d{1,3})\)|(\d{1,3}))\s+(?=\S)")
_FOOTNOTE_GLUED = re.compile(r"^\s*(\d{1,3})(?=[A-Z\u201c\u2018\"])")
_FOOTNOTE_CONTINUES = re.compile(r"[.)\"”;:’]\s*$")


@dataclass
class RawNote:
    """One footnote: its number (if known) and its lines, still unjoined."""

    number: int | None
    lines: list[str]
    page: int


@dataclass
class PageLines:
    number: int  # 1-based PDF page number
    lines: list[str]  # body lines; leading spaces are kept (an indent can mark a new paragraph)
    notes: list[RawNote] = field(default_factory=list)


@dataclass
class LayoutReport:
    pages: int = 0
    empty_pages: int = 0
    empty_page_numbers: list[int] = field(default_factory=list)  # 1-based numbers of the pages counted in empty_pages
    margin_lines_removed: Counter = field(default_factory=Counter)  # margin_key -> pages it was removed from
    margin_examples: dict = field(default_factory=dict)  # margin_key -> the first line removed under that key
    page_numbers_removed: int = 0
    toc_lines_removed: int = 0
    footnote_pages: int = 0
    footnotes: int = 0
    body_size: float | None = None  # font size of the running text, when the PDF's type sizes were readable
    chars: Counter = field(default_factory=Counter)


def split_lines(
    page_text: str,
    stats: Counter | None = None,
    styles: list[tuple[float, bool] | None] | None = None,
) -> list[str]:
    """Normalise one page of extracted text and split it into lines (inner space runs collapsed).

    ``styles`` holds the ``(font size, bold)`` of each line of ``page_text`` (see ``pdfread``). When it matches the
    lines of the page, every non-empty line comes back as a ``TextLine`` that carries its typography.
    """
    text = repair_footnote_markers(normalize_chars(page_text, stats))
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if styles is not None and len(styles) != len(raw_lines):
        styles = None  # the lines no longer correspond to the typography; better no typography than a wrong one
    lines: list[str] = []
    for index, raw in enumerate(raw_lines):
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        indent = " " * min(len(raw) - len(raw.lstrip(" ")), 8)
        value = indent + " ".join(stripped.split())
        style = styles[index] if styles is not None else None
        lines.append(TextLine(value, style[0], style[1]) if style else value)
    return _trim_blank_edges(lines)


def _trim_blank_edges(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return lines[start:end]


def margin_key(line: str) -> str:
    """A fingerprint of a line that ignores digits and punctuation, so "L 119/32" and "L 119/33" match."""
    text = re.sub(r"\d+", "#", line.strip().lower())
    return " ".join(re.sub(r"[^\w#]+", " ", text).split())


def _edge_lines(lines: list[str]) -> list[int]:
    """Indexes of the lines that could be page furniture: the first and last few non-blank lines."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return sorted(set(filled[:MARGIN_LINES] + filled[-MARGIN_LINES:]))


def _numbers(line: str) -> tuple[int, ...]:
    return tuple(int(digits) for digits in re.findall(r"\d+", line))


def _numbers_move_with_the_page(seen: list[tuple[int, tuple[int, ...]]]) -> bool:
    """Is every number in a repeated line either the same on all pages or a page counter?

    A running header may contain "Page 3 of 80" or "L 119/32": one number moves in step with the page, the others
    stay. A repeated line whose numbers change in any other way is not page furniture - the usual example is a short
    footnote such as "12 Id." / "27 Id." at the foot of many pages.
    """
    width = len(seen[0][1])
    if width == 0:
        return True
    if any(len(numbers) != width for _page, numbers in seen):
        return False
    for position in range(width):
        if len({numbers[position] for _page, numbers in seen}) == 1:
            continue
        offsets = Counter(numbers[position] - page for page, numbers in seen)
        if offsets.most_common(1)[0][1] < 0.8 * len(seen):
            return False
    return True


def find_margin_keys(pages: list[list[str]]) -> set[str]:
    """Fingerprints of lines that repeat at the edges of many pages."""
    if len(pages) < MARGIN_MIN_PAGES:
        return set()
    seen: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    for page_index, lines in enumerate(pages):
        keys: dict[str, tuple[int, ...]] = {}
        for index in _edge_lines(lines):
            text = lines[index].strip()
            if len(text) > MARGIN_MAX_CHARS or STRUCTURAL_LINE.match(text):
                continue
            key = margin_key(text)
            if key and key not in keys:
                keys[key] = _numbers(text)
        for key, numbers in keys.items():
            seen.setdefault(key, []).append((page_index, numbers))
    needed = max(MARGIN_MIN_PAGES, math.ceil(MARGIN_MIN_FRACTION * len(pages)))
    return {key for key, found in seen.items() if len(found) >= needed and _numbers_move_with_the_page(found)}


def margin_examples(pages: list[list[str]], keys: set[str]) -> dict[str, str]:
    """For every margin key, the first line seen at a page edge under that key (digits as printed)."""
    examples: dict[str, str] = {}
    for lines in pages:
        for index in _edge_lines(lines):
            text = lines[index].strip()
            if len(text) > MARGIN_MAX_CHARS or STRUCTURAL_LINE.match(text):
                continue
            key = margin_key(text)
            if key in keys:
                examples.setdefault(key, text)
    return examples


_PIECE = re.compile(r"\d+|[^\W\d_]+|[^\w\s]")  # a number, a word or one punctuation mark


def furniture_regex(example: str) -> re.Pattern | None:
    """A pattern that finds a running header like ``example`` anywhere in a line, whatever its page number.

    Only headers that contain a number (a date, a page counter, an act number) and at least two words qualify. That
    makes a false match inside real text practically impossible, whereas a purely verbal running title such as
    "Guidelines on consent" could also be a phrase of the document.
    """
    pieces = _PIECE.findall(example)
    words = [piece for piece in pieces if piece[0].isalpha()]
    if len(example) < FURNITURE_MIN_CHARS or len(words) < 2 or not any(piece.isdigit() for piece in pieces):
        return None
    body = r"\s*".join(r"\d+" if piece.isdigit() else re.escape(piece) for piece in pieces)
    return re.compile(rf"(?<!\w){body}(?!\w)")


def furniture_patterns(examples: dict[str, str]) -> list[tuple[str, re.Pattern]]:
    patterns = []
    for key, example in examples.items():
        pattern = furniture_regex(example)
        if pattern is not None:
            patterns.append((key, pattern))
    return patterns


def remove_inline_furniture(lines: list[str], patterns: list[tuple[str, re.Pattern]], report: LayoutReport) -> list[str]:
    """Cut running headers out of the middle of lines; a line that consisted of nothing else disappears."""
    if not patterns:
        return lines
    result: list[str] = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        text = str(line)
        for key, pattern in patterns:
            found = pattern.search(text)
            if found:
                report.margin_examples.setdefault(key, found.group(0))
                text, hits = pattern.subn(" ", text)
                report.margin_lines_removed[key] += hits
        if text == line:
            result.append(line)
            continue
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        indent = " " * min(len(line) - len(line.lstrip(" ")), 8)
        size, bold = getattr(line, "size", None), getattr(line, "bold", False)
        result.append(TextLine(indent + cleaned, size, bold) if size is not None else indent + cleaned)
    return _trim_blank_edges(result)


def pick_body_size(by_size: Counter) -> float:
    """The size of the running text, from a count of characters per font size (sizes rounded to half a point).

    Usually the commonest size. But footnotes can outweigh the text above them (a short statement with long notes) and
    tables in a smaller size can outweigh the prose, while headings are large and rare. So among the sizes that carry
    at least 30 % of the typeset text the largest is taken - never one more than half as large again as the commonest.
    """
    total = sum(by_size.values())
    commonest = max(by_size, key=lambda candidate: (by_size[candidate], candidate))
    real = [size for size, characters in by_size.items() if characters >= MIN_BODY_SHARE * total and size <= MAX_BODY_SIZE_JUMP * commonest]
    return max(real, default=commonest)


def body_font_size(pages: list[list[str]]) -> float | None:
    """The size of the running text (see ``pick_body_size``).

    None unless the sizes are known for most of the text, because without them nothing can be called "small print".
    """
    by_size: Counter = Counter()
    total = typeset = 0
    for lines in pages:
        for line in lines:
            characters = len(line.strip())
            if not characters:
                continue
            total += characters
            size = getattr(line, "size", None)
            if size:
                typeset += characters
                by_size[round(size * 2) / 2] += characters
    if not total or typeset < MIN_TYPESET_SHARE * total:
        return None
    return pick_body_size(by_size)


def strip_margins(lines: list[str], keys: set[str], report: LayoutReport) -> list[str]:
    """Remove repeated furniture, bare page numbers at the page edges and table-of-contents lines."""
    drop: set[int] = set()
    edges = _edge_lines(lines)
    for index in edges:
        text = lines[index].strip()
        key = margin_key(text)
        if len(text) <= MARGIN_MAX_CHARS and not STRUCTURAL_LINE.match(text) and key in keys:
            drop.add(index)
            report.margin_lines_removed[key] += 1
            report.margin_examples.setdefault(key, text)
    # Bare page numbers: only the very first or last line counts, so a number inside the text is never touched.
    filled = [i for i, line in enumerate(lines) if line.strip() and i not in drop]
    for index in filled[:1] + filled[-1:]:
        if _PAGE_NUMBER.match(lines[index]) and index not in drop:
            drop.add(index)
            report.page_numbers_removed += 1
    kept = []
    for index, line in enumerate(lines):
        if index in drop:
            continue
        if _TOC_LINE.match(line.strip()):
            report.toc_lines_removed += 1
            continue
        kept.append(line)
    return _trim_blank_edges(kept)


def _footnote_match(line: str) -> tuple[int, str] | None:
    """``(number, rest of the line)`` when the line starts like a footnote, else None."""
    match = _FOOTNOTE_START.match(line)
    if match:
        return int(match.group(1) or match.group(2)), line[match.end() :]
    glued = _FOOTNOTE_GLUED.match(line)  # a raised note number set directly against the text: "1Regulation (EU) ..."
    if glued:
        return int(glued.group(1)), line[glued.end() :]
    return None


def _is_small(line: str, body_size: float) -> bool:
    size = getattr(line, "size", None)
    return size is not None and size <= SMALL_RATIO * body_size and size <= body_size - MIN_SIZE_GAP


@dataclass
class _Zone:
    body: list[str]
    continuation: list[str]  # small print above the first numbered note: the rest of the previous page's last note
    note_lines: list[str]


def _continues_previous(previous: RawNote | None, line: str) -> bool:
    """Can ``line`` (small print at the top of a footnote zone, without a number) continue the previous page's note?"""
    if previous is None or not previous.lines:
        return False
    return not _FOOTNOTE_CONTINUES.search(previous.lines[-1]) or line.lstrip()[:1].islower()


def _split_one_page(lines: list[str], body_size: float, previous: RawNote | None) -> _Zone | None:
    """Cut the footnote zone off the foot of a page, or return None if the page has none.

    The zone is the unbroken run of small print at the very end of the page. It starts with the first line that
    carries a note number; small lines above that, still inside the run, are the end of a note that began on the
    previous page. Anything else stays in the body.
    """
    start = None
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip():
            continue
        if not _is_small(lines[index], body_size):
            break
        start = index
    if start is None:
        return None
    zone = [index for index in range(start, len(lines)) if lines[index].strip()]
    if len(zone) > MAX_NOTE_FRACTION * sum(1 for line in lines if line.strip()):
        return None  # a whole page in small print, not a footnote zone
    first = next((position for position, index in enumerate(zone) if _footnote_match(lines[index])), None)
    if first is None:
        # No numbered note at all: only the tail of a note from the previous page can look like this.
        if _continues_previous(previous, lines[zone[0]]):
            return _Zone(lines[:start], [lines[index] for index in zone], [])
        return None
    lead = zone[:first]
    if lead and not _continues_previous(previous, lines[lead[0]]):
        return _Zone(lines[: zone[first]], [], [lines[index] for index in zone[first:]])
    return _Zone(lines[:start], [lines[index] for index in lead], [lines[index] for index in zone[first:]])


def _group_notes(lines: list[str], page: int) -> list[RawNote]:
    """Group footnote lines into notes. After the first, a line that starts with the next expected number starts a new note."""
    notes: list[RawNote] = []
    for line in lines:
        found = _footnote_match(line)
        expected = None if not notes else (notes[-1].number or 0) + 1
        if found is not None and (expected is None or found[0] == expected):
            notes.append(RawNote(found[0], [found[1].strip()], page))
        elif notes:
            notes[-1].lines.append(line.strip())
        else:
            notes.append(RawNote(None, [line.strip()], page))
    return notes


def _notes_look_real(spans: list[tuple[int, int]], pages_total: int) -> bool:
    """Footnotes show up on several pages and their numbers continue (or start again at 1) from page to page."""
    if not spans:
        return False
    if len(spans) == 1:
        # Only a short document may have footnotes on a single page, and then they start at 1 or form a run.
        first, last = spans[0]
        return pages_total < 4 and (first == 1 or last > first)
    fits = sum(1 for (_a, previous_last), (next_first, _b) in zip(spans, spans[1:])
               if next_first == 1 or previous_last < next_first <= previous_last + 4)
    return fits >= 0.5 * (len(spans) - 1)


def prepare_pages(
    raw_pages: list[str],
    styles: list[list[tuple[float, bool] | None] | None] | None = None,
) -> tuple[list[PageLines], LayoutReport]:
    """Turn raw page strings into body lines and footnotes, with a report of everything that was removed.

    ``styles`` is the typography of the pages as ``pdfread`` returns it (one entry per page, one pair per line).
    Without it nothing is treated as a footnote.
    """
    report = LayoutReport(pages=len(raw_pages))
    split = [
        split_lines(text, report.chars, styles[index] if styles is not None and index < len(styles) else None)
        for index, text in enumerate(raw_pages)
    ]
    report.empty_page_numbers = [number for number, lines in enumerate(split, start=1) if sum(len(line.strip()) for line in lines) < 50]
    report.empty_pages = len(report.empty_page_numbers)

    keys = find_margin_keys(split)
    patterns = furniture_patterns(margin_examples(split, keys))
    split = [remove_inline_furniture(lines, patterns, report) for lines in split]
    cleaned = [strip_margins(lines, keys, report) for lines in split]
    report.body_size = body_font_size(cleaned)

    pages: list[PageLines] = []
    spans: list[tuple[int, int]] = []
    previous: RawNote | None = None
    for number, lines in enumerate(cleaned, start=1):
        zone = _split_one_page(lines, report.body_size, previous) if lines and report.body_size else None
        if zone is None:
            pages.append(PageLines(number, lines))
            previous = None  # a note can only continue on the very next page
            continue
        notes = _group_notes(zone.note_lines, number)
        if zone.continuation and previous is not None:
            previous.lines.extend(line.strip() for line in zone.continuation)
        if notes and notes[0].number is not None:
            spans.append((notes[0].number, notes[-1].number or notes[0].number))
        pages.append(PageLines(number, _trim_blank_edges(zone.body), notes))
        if notes:
            previous = notes[-1]
    if not _notes_look_real(spans, len(raw_pages)):
        pages = [PageLines(page.number, cleaned[i]) for i, page in enumerate(pages)]
        spans = []
    report.footnote_pages = len(spans)
    report.footnotes = sum(len(page.notes) for page in pages)
    return pages, report
