"""Read a PDF with pypdf: the text of every page, how each line was typeset, and the bookmarks (outline).

Plain text extraction throws away what a reader uses to tell a footnote from a list item or a table cell from a
heading: the type size and weight. pypdf can report the font of every piece of text it extracts, so each page is
read once with that report switched on. The text itself is exactly what ``extract_text()`` returns - the pieces are
checked to add up to it - except for raised digits (footnote marks and exponents), which are written explicitly
because plain extraction glues them to the number or word before them.

Sizes need care. A piece of text arrives together with the matrices recorded when its run *started*, and some pypdf
versions deliver a run that follows a horizontal jump on the same line with the unit matrix instead: an 11-point line
of a Word-made PDF (``/F1 92 Tf`` under a ``0.12`` text matrix) then looks 92 points high. So the operators of the
page are watched as well; they are processed with the live matrices, and where a piece contradicts the operator that
drew it, the operator wins.

pypdf also hands over a whole run of text as one piece, however often the text matrix changed inside it, and a footnote
mark drawn at another scale (``Tm 6.96`` between two lines of ``Tm 11.04``) would vanish into the sentence. Where a
page does that, a harmless font selection is put in front of the text drawn at the new scale, in memory only; pypdf
flushes its text at each font selection, so the mark becomes a piece of its own. This is used only when it leaves the
text of the page exactly as pypdf reads it without the change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from reglens.ingest.models import OutlineEntry
from reglens.ingest.textclean import superscript_text
from reglens.pdfutil import describe_pdf_error, quiet_pypdf

_BOLD_FACE = re.compile(r"bold|black|heavy|demi", re.IGNORECASE)  # in the BaseFont name: "Arial-BoldMT", "Calibri,Bold"
_MARK = re.compile(r"\s*\d{1,3}(?:[,\u2013-]\d{1,3}){0,3}\s*")  # the text of a footnote mark: "7", "1,2", "3-5"
RAISED_RATIO = 0.78  # a footnote mark is set at most this fraction of the size of the text before it


class ExtractionError(Exception):
    """The document could not be read at all. The message says why and, where known, how to fix it."""


@dataclass
class PdfExtraction:
    pages: list[str]
    outline: list[OutlineEntry] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    metadata_title: str = ""
    # Per page, one ``(font size, bold)`` pair for every line of the page text (None for an empty line); the whole
    # page entry is None when the typography could not be read reliably. Line ``i`` of ``styles[p]`` belongs to line
    # ``i`` of ``pages[p].split("\n")``.
    styles: list[list[tuple[float, bool] | None] | None] = field(default_factory=list)


Piece = tuple[str, float, bool, float]  # text, effective font size, bold, baseline height


def _effective_size(cm, tm, font_size) -> float:
    """The size in points a piece of text is drawn at: the font size scaled by the text and page matrices."""
    a1, b1, c1, d1, _e1, _f1 = (float(v) for v in tm)
    a2, b2, c2, d2, _e2, _f2 = (float(v) for v in cm)
    return abs(float(font_size)) * math.hypot(c1 * a2 + d1 * c2, c1 * b2 + d1 * d2)


def _baseline(cm, tm) -> float:
    """The height of the text baseline on the page (the translation part of the combined text and page matrix)."""
    _a1, b1, _c1, d1, e1, f1 = (float(v) for v in tm)
    _a2, b2, _c2, d2, e2, f2 = (float(v) for v in cm)
    return e1 * b2 + f1 * d2 + f2


def _is_bold(font_dict) -> bool:
    try:
        return bool(_BOLD_FACE.search(str(font_dict.get("/BaseFont", "") or "")))
    except Exception:
        return False


def _compose_line(pieces: list[Piece]) -> tuple[str, tuple[float, bool] | None]:
    """Join the pieces of one line. Returns the text and its dominant ``(size, bold)``.

    A piece that is a short number, clearly smaller than the text before it and not lowered below the line, is a
    footnote mark or an exponent; ``superscript_text`` decides which and writes it out. A number is often drawn digit
    by digit ("2", "4" for 10^24), so directly adjacent raised digits are one number. A number right after an opening
    bracket is neither: Official Journal texts set the digit of "(1)" smaller than its brackets.
    """
    built = ""
    raised = ""  # the digits of a raised number, waiting to be written
    after = ""  # the space that followed them
    sizes: Counter = Counter()
    bold_chars = total_chars = 0
    base_size = base_baseline = None

    def write_raised() -> None:
        nonlocal built, raised, after
        if raised:
            built = built.rstrip() + superscript_text(built, raised) + after  # pypdf puts a space before a far-right piece
        raised = after = ""

    for text, size, bold, baseline in pieces:
        if (
            base_size is not None
            and 0 < size <= RAISED_RATIO * base_size
            and baseline >= base_baseline - 0.05 * base_size
            and _MARK.fullmatch(text)
            and not built.rstrip().endswith(("(", "["))  # the "1" of "(1)" is smaller in Official Journal texts; it is no mark
        ):
            digits = text.strip()
            if raised and (after or text[:1].isspace() or not (raised.isdigit() and digits.isdigit())):
                write_raised()  # not directly adjacent: this is a number of its own
            raised += digits
            after = text[len(text.rstrip()) :]
            continue
        write_raised()
        built += text
        visible = len(text.replace(" ", "").replace("\t", ""))
        if visible and size > 0:
            sizes[round(size, 1)] += visible
            total_chars += visible
            bold_chars += visible if bold else 0
            base_size, base_baseline = size, baseline
    write_raised()
    if not sizes:
        return built, None
    size = max(sizes, key=lambda candidate: (sizes[candidate], candidate))
    return built, (size, 2 * bold_chars >= total_chars)


def _compose_page(pieces: list[Piece]) -> tuple[str, list[tuple[float, bool] | None]]:
    lines: list[list[Piece]] = [[]]
    for text, size, bold, baseline in pieces:
        for number, part in enumerate(text.split("\n")):
            if number:
                lines.append([])
            if part:
                lines[-1].append((part, size, bold, baseline))
    texts, styles = [], []
    for line in lines:
        built, style = _compose_line(line)
        texts.append(built)
        styles.append(style)
    return "\n".join(texts), styles


SHOW_OPERATORS = (b"Tj", b"TJ", b"'", b'"')
MAX_PENDING = 5000  # operators waiting for their text; bounded so a page without text cannot grow it
SIZE_TOLERANCE = 0.03  # two sizes within 3 % are the same size


def _is_unit(cm, tm) -> bool:
    """True when both matrices are the identity: what pypdf hands over when it has lost track of a run's matrices."""
    unit = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    try:
        return tuple(float(v) for v in cm) == unit and tuple(float(v) for v in tm) == unit
    except Exception:
        return False


class _PageReader:
    """Collects the pieces of one page, with sizes checked against the operators that drew the text.

    ``before`` is pypdf's ``visitor_operand_before``: it sees every operator with the live matrices, so the size of each
    text-showing operator is noted, in order. ``visit`` is ``visitor_text``: it receives text that was drawn by some of
    the operators noted since the last piece with text. A piece normally describes itself (font size times its
    matrices). A piece that arrives with the unit matrices has lost its own, so it takes its baseline from the operator
    that drew it, and its size too when the operator's scale was a different one.
    """

    def __init__(self) -> None:
        self.pieces: list[Piece] = []
        self._bold_of: dict[int, bool] = {}
        self._font_size: float | None = None
        self._saved: list[float | None] = []
        self._pending: list[tuple[float, float, float]] = []  # (font size, effective size, baseline) per operator
        self._last: tuple[float, float, float] | None = None

    def before(self, operator, operands, cm, tm) -> None:
        try:
            if operator == b"Tf":
                self._font_size = float(operands[1])
            elif operator == b"q":
                self._saved.append(self._font_size)
            elif operator == b"Q":
                if self._saved:
                    self._font_size = self._saved.pop()
            elif operator in SHOW_OPERATORS and self._font_size is not None:
                if len(self._pending) >= MAX_PENDING:
                    del self._pending[: MAX_PENDING // 2]
                self._pending.append((self._font_size, _effective_size(cm, tm, self._font_size), _baseline(cm, tm)))
        except Exception:  # an odd operator must never stop the extraction
            pass

    def _operator_for(self, font_size: float) -> tuple[float, float, float] | None:
        """The operator that drew the text of the piece now arriving (None when that cannot be told)."""
        operators, self._pending = self._pending, []
        for candidate in operators:
            if abs(candidate[0] - font_size) < 0.01:  # a piece never spans a change of font size
                self._last = candidate
                return candidate
        if not operators and self._last is not None and abs(self._last[0] - font_size) < 0.01:
            return self._last  # a second piece of the same operator
        return None

    def visit(self, text, cm, tm, font_dict, font_size) -> None:
        if not text:
            return
        try:
            key = id(font_dict)
            if key not in self._bold_of:
                self._bold_of[key] = _is_bold(font_dict)
            size, baseline = _effective_size(cm, tm, font_size), _baseline(cm, tm)
            if text.strip():
                drawn_by = self._operator_for(float(font_size))
                if drawn_by is not None and _is_unit(cm, tm):  # the unit matrices say nothing; the operator knows
                    if abs(size - drawn_by[1]) > SIZE_TOLERANCE * drawn_by[1]:
                        size = drawn_by[1]
                    baseline = drawn_by[2]
            self.pieces.append((text, size, self._bold_of[key], baseline))
        except Exception:  # odd matrices or fonts: the text is still kept, without typography
            self.pieces.append((text, 0.0, False, 0.0))


def _scale_of(operands) -> float | None:
    """The vertical scale of the text matrix that a ``Tm`` operator sets."""
    try:
        return math.hypot(float(operands[2]), float(operands[3]))
    except Exception:
        return None


def _separate_scale_changes(page) -> tuple[object | None, bool]:
    """Parse the content of the page and put a font selection in front of text that is drawn at another text-matrix scale.

    The page is changed in memory only. Returns the original ``/Contents`` (to put back after reading) and whether
    anything was inserted; ``(None, False)`` when the page could not be parsed and is left as it is. The content is
    parsed exactly as pypdf does it itself (strings as raw bytes), so that nothing but the insertions can differ.
    """
    try:
        from pypdf.generic import ContentStream, NameObject

        original = page.get("/Contents")
        if original is None:
            return None, False
        content = ContentStream(original.get_object(), page.indirect_reference.pdf, "bytes")
        operations = []
        inserted = 0
        font = size = None
        saved: list[tuple] = []
        scale = 1.0  # vertical scale of the current text matrix
        shown: tuple | None = None  # font, size and scale of the text shown last
        for operands, operator in content.operations:
            if operator == b"Tf" and len(operands) >= 2:
                font, size = operands[0], operands[1]
            elif operator == b"q":
                saved.append((font, size))
            elif operator == b"Q":
                if saved:
                    font, size = saved.pop()
            elif operator == b"BT":
                scale = 1.0
            elif operator == b"Tm":
                found = _scale_of(operands)
                if found is not None:
                    scale = found
            elif operator in SHOW_OPERATORS and font is not None:
                if shown is not None and shown[:2] == (font, size) and abs(scale - shown[2]) > SIZE_TOLERANCE * max(scale, shown[2]):
                    operations.append(([font, size], b"Tf"))  # pypdf hands over its text at every font selection
                    inserted += 1
                shown = (font, size, scale)
            operations.append((operands, operator))
        content.operations = operations
        page[NameObject("/Contents")] = content
        return original, inserted > 0
    except Exception:  # a page that cannot be rewritten is simply read as it is
        return None, False


def _restore_contents(page, original) -> None:
    if original is None:
        return
    try:
        from pypdf.generic import NameObject

        page[NameObject("/Contents")] = original
    except Exception:
        pass


def _unix_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _without_repeats(pieces: list[Piece], plain: str) -> list[Piece] | None:
    """The pieces without the repeats pypdf reports for text drawn by a form (a header or footer is seen once inside
    the form and once more from the page, the second time with matrices of the page's own text). The second of two
    identical pieces in a row is dropped; None unless the rest then adds up to the text of the page."""
    kept = [piece for index, piece in enumerate(pieces) if not (index and piece[0].strip() and piece[0] == pieces[index - 1][0])]
    return kept if _unix_newlines("".join(piece[0] for piece in kept)) == plain else None


def _extract_with_pieces(page) -> tuple[str, list[Piece] | None] | None:
    """The text of the page and its pieces (None when they do not add up to the text); None when pypdf failed."""
    reader = _PageReader()
    try:
        plain = _unix_newlines(page.extract_text(visitor_text=reader.visit, visitor_operand_before=reader.before) or "")
    except Exception:  # reading with the font report failed: the caller takes the text alone, without typography
        return None
    pieces = reader.pieces
    if pieces and _unix_newlines("".join(piece[0] for piece in pieces)) != plain:
        pieces = _without_repeats(pieces, plain) or []
    return (plain, pieces) if pieces else (plain, None)  # pieces that do not add up: line and font cannot be matched


def read_page(page) -> tuple[str, list[tuple[float, bool] | None] | None]:
    """The text of one pypdf page and, when it could be read reliably, the typography of each of its lines.

    A page that draws text at several scales within one run is read a second time as it is, and its typography is
    taken from the first reading only if both give the same text: what is extracted must never depend on this trick.
    """
    original, separated = _separate_scale_changes(page)
    result = _extract_with_pieces(page)
    if original is not None:
        _restore_contents(page, original)
        if separated or result is None:
            as_it_is = _extract_with_pieces(page)
            if result is None or (as_it_is is not None and as_it_is[0] != result[0]):
                result = as_it_is
    if result is None:
        return _unix_newlines(page.extract_text() or ""), None
    plain, pieces = result
    return (plain, None) if pieces is None else _compose_page(pieces)


def _read_outline(reader) -> list[OutlineEntry]:
    """Flatten the bookmark tree. A nested list in pypdf's result holds the children of the item before it."""
    entries: list[OutlineEntry] = []

    def walk(items, level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            try:
                title = (item.title or "").strip()
                page = reader.get_destination_page_number(item)
            except Exception:  # a bookmark that points nowhere is simply skipped
                continue
            if title and page is not None and page >= 0:
                entries.append(OutlineEntry(title, level, page + 1))

    try:
        walk(reader.outline, 1)
    except Exception:  # a damaged bookmark tree must never stop the ingestion
        return []
    return entries


def read_pdf(path: Path) -> PdfExtraction:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is in requirements.txt
        raise ExtractionError('pypdf is not installed; run: pip install -r requirements.txt') from exc
    try:
        with quiet_pypdf():
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ExtractionError("the PDF is password-protected")
            pages: list[str] = []
            styles: list[list[tuple[float, bool] | None] | None] = []
            errors: list[str] = []
            for number, page in enumerate(reader.pages, start=1):
                try:
                    text, page_styles = read_page(page)
                    pages.append(text)
                    styles.append(page_styles)
                except Exception as exc:  # one bad page must not lose the whole document
                    pages.append("")
                    styles.append(None)
                    errors.append(f"page {number}: {type(exc).__name__}")
            outline = _read_outline(reader)
            title = ""
            try:
                title = str(getattr(reader.metadata, "title", "") or "")
            except Exception:
                title = ""
    except ExtractionError:
        raise
    except OSError as exc:
        raise ExtractionError(f"file cannot be read ({type(exc).__name__}) - close it if another program has it open") from exc
    except Exception as exc:  # pypdf raises many types on damaged files
        raise ExtractionError(describe_pdf_error(exc, "PDF could not be read")) from exc
    return PdfExtraction(pages=pages, outline=outline, page_errors=errors, metadata_title=title.strip(), styles=styles)
