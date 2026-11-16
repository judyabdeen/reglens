"""Find the structure of a document - headings, paragraphs, footnotes - in cleaned page lines.

PDF text has no markup, so structure is inferred. Headings are recognised by *patterns* that official documents
share, ordered from the most reliable to the least reliable:

1. PDF bookmarks (the outline), when they are detailed enough and match lines of the text;
2. legal divisions: "Article 5", "CHAPTER III", "Annex I", "Part 1: ...", "Appendix A: ...", "§ 7001. Definitions.",
   a California code section number such as "1798.100.", and the "Whereas:" that opens EU recitals;
3. numbered headings: "I. Summary", "2.1 Scope", "A.1 Title";
4. ALL-CAPS lines;
5. short Title Case lines that stand alone.

Rules 4 and 5 can misfire on tables and lists (every cell of a table that is written in Title Case looks like a
heading), so when the PDF's type sizes are known they must also be *set apart*: larger than the running text, or
bold. A safety valve additionally re-runs the detection with the weak rules switched off if the result looks
implausible (a heading for every few words). Paragraphs are rebuilt from the wrapped lines using sentence endings,
line length, indentation, list markers and blank lines.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from reglens.ingest.layout import PageLines, RawNote, pick_body_size
from reglens.ingest.models import FOOTNOTE, HEADING, PARAGRAPH, Block, OutlineEntry
from reglens.ingest.textclean import build_vocab, join_wrapped_lines, repair_spacing, repair_split_words

# --------------------------------------------------------------------------------------- patterns

_SENTENCE_END = re.compile(r"[.!?:;][\"'”’)\]]*(?:\s*\[\d{1,3}\])*$")  # a footnote marker such as "[12]" may follow
_LIST_MARKER = re.compile(r"^(?:[•*–-]\s|\((?:[a-z]{1,2}|[ivxlc]{1,6}|\d{1,3})\)\s|[a-z]\)\s|\d{1,3}[.)]\s)")
_LONE_MARKER = re.compile(r"^(?:[•*–-]|\((?:[a-z]{1,2}|[ivxlc]{1,6}|\d{1,3})\)|[a-z]\)|\d{1,3}[.)])$")

_KEYWORD = re.compile(
    r"^(?P<kw>(?i:article|chapter|section|part|title|annex|appendix|schedule|book|subpart))\s+"
    r"(?P<id>[IVXLC]{1,7}|\d{1,4}[A-Za-z]?(?:\.\d+)*|[A-Z])"
    r"(?:\s*(?P<sep>[:.–—-])\s*(?P<title>\S.*))?$"
)
_SECTION_SIGN = re.compile(r"^§{1,2}\s*(?P<num>\d+(?:\.\d+)*[A-Za-z]?)\.?\s+(?P<title>\S.*)$")
_CODE_SECTION = re.compile(r"^(?P<num>\d{4}\.\d{1,3}(?:\.\d{1,3})?)\.(?:\s+(?P<rest>\S.*))?$")
_DECIMAL = re.compile(r"^(?P<num>(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?P<title>[A-Z“\"(].*)$")
_ROMAN = re.compile(r"^(?P<num>[IVXLC]{1,6})\.\s+(?P<title>[A-Z“\"].*)$")
_ALPHA = re.compile(r"^(?P<num>[A-HJ-Z])\.\s+(?P<title>[A-Z“\"].*)$")
_WHEREAS = re.compile(r"^whereas:?$", re.IGNORECASE)
_ADOPTED = re.compile(r"^(?:HAS|HAVE) ADOPTED THIS (?:REGULATION|DIRECTIVE|DECISION)\s*:?$")  # ends the recitals of an EU act
_QUOTE_CLOSE = re.compile(r"’[;.,:)]*\s*$")  # the closing quotation mark at the end of a line, often followed by ";" or "."
_PARAGRAPH_NUMBER = re.compile(r"^(\d{1,3})\.?\s+(?=[A-Z“\"‘(])")  # "2 Each Party shall ..." / "2. Each Party shall ..."

_MONTHS = frozenset(
    "january february march april may june july august september october november december".split()
)
_SMALL_WORDS = frozenset(
    "a an the and or nor of in on at to for from by with without as via vs versus into onto upon over under "
    "per is are be not than that this these those its their".split()
)

# Rank of each heading scheme; smaller means higher in the hierarchy. Only the order matters: ranks are
# compressed to levels 1..k for the schemes a document really uses.
_KEYWORD_RANK = {
    "part": 1, "title": 1, "book": 1, "chapter": 2, "annex": 2, "appendix": 2, "schedule": 2,
    "recitals": 2, "enacting": 2, "section": 3, "subpart": 3, "code_section": 4, "section_sign": 4, "article": 5,
}  # fmt: skip
_OTHER_RANK = {"allcaps": 0.5, "titlecase": 0.7, "roman": 10, "alpha": 11}

MAX_QUOTE_PAGES = 12  # a quoted insertion that has lasted this many pages was surely closed in a way we did not see
MAX_NUMBERED_WORDS = 16
MAX_WRAPPED_NUMBERED_WORDS = 25  # a numbered heading that wraps over lines of one heading type may be longer
MAX_HEADING_CHARS = 140
MAX_WRAPPED_LINES = 4  # a heading in a narrow column wraps over at most this many lines
SAME_TYPE_TOLERANCE = 0.25  # points: two lines are set in the same size when they differ by less
MAX_HEADING_DENSITY = 2.0  # more than this many headings per 100 body words means the weak rules misfired
HEADING_SIZE_RATIO = 1.05  # a heading is typeset at least this much larger than the running text, or in bold
MIN_TYPESET_SHARE = 0.5  # type sizes are trusted only if they are known for at least this share of the text
BOLD_BODY_SHARE = 0.4  # if more than this share of the text is bold, boldness says nothing about headings
MIN_OUTLINE_MATCHES = 5
PARAGRAPH_MIN_LINE_CHARS = 25


# ------------------------------------------------------------------------------------- data classes


@dataclass(frozen=True)
class Line:
    text: str
    page: int
    indent: int = 0
    blank_before: bool = False
    size: float | None = None  # font size in points, when the PDF's typography was readable
    bold: bool = False


@dataclass(frozen=True)
class HeadingMatch:
    text: str
    scheme: str
    consumed: int = 1
    rest: str = ""  # what follows the heading on the same line and continues as body text


@dataclass
class StructureReport:
    rule_level: int = 2  # 2 = all rules, 1 = no Title Case / letter rules, 0 = legal and numbered headings only
    legal_act: bool = False  # three or more "Article N" headings: numbered paragraphs and Title Case are no headings
    outline_used: bool = False
    outline_matches: int = 0
    headings: int = 0
    schemes: Counter = field(default_factory=Counter)
    paragraphs: int = 0
    footnotes: int = 0


# -------------------------------------------------------------------------------------------- helpers


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text))


def _norm_title(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z’'-]*", text)


_MODAL_VERBS = frozenset({"shall", "must", "may", "should", "will", "would"})


def _heading_title_ok(title: str, *, max_words: int = MAX_NUMBERED_WORDS, allow_period: bool = False) -> bool:
    """Is ``title`` plausible as the text of a heading (short, not a sentence)?

    A title that ends with a colon or contains a modal verb ("2. Personal data shall be:") is the lead-in of a
    numbered paragraph, not a heading.
    """
    title = title.strip()
    if not 2 <= len(title) <= MAX_HEADING_CHARS:
        return False
    if len(title.split()) > max_words:
        return False
    if title[-1] in ",;:" or (title[-1] == "." and not allow_period):
        return False
    if _MODAL_VERBS & {word.lower() for word in title.split()}:
        return False
    first_word = (title.split()[0] if title.split() else "").lower().strip("“\"(")
    if first_word in _MONTHS:
        return False
    return any(c.isalpha() for c in title)


def _cut_bracket(text: str) -> bool:
    """Does the line open a square bracket it does not close ("[NIS"), or close one it never opened ("TAML.016]")?
    Such a line is the piece of a tag or a link that the page broke in two, not a heading."""
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


def _is_all_caps(text: str) -> bool:
    if _cut_bracket(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3 or len(text) > 100 or len(text.split()) > 14:
        return False
    if not all(c.isupper() for c in letters):
        return False
    return text[-1] not in ",;.:" and not text.endswith(("?", "!"))  # "HAVE ADOPTED THIS REGULATION:" is a lead-in


def _is_title_case(text: str) -> bool:
    if _cut_bracket(text):
        return False
    words = _words(text)
    if not 1 <= len(words) <= 12 or len(text) > 90:
        return False
    if text[-1] in ".,;!" or text[0].islower() or not any(c.islower() for c in text):
        return False
    if any(c.isdigit() for c in text):
        return False  # dates, captions such as "Table 3", list items
    significant = [w for w in words if w.lower() not in _SMALL_WORDS]
    if not significant or not words[0][0].isupper():
        return False
    return all(w[0].isupper() for w in significant)


# ------------------------------------------------------------------------------------ the detector


@dataclass
class _Context:
    p90: int
    rule_level: int
    outline: dict[int, list[tuple[str, int]]]
    outline_only: bool
    legal: bool = False  # legal-act mode: "1." and "2." number the paragraphs of an article; Title Case is no heading
    body_size: float | None = None  # font size of the running text; None when unknown (web pages, unreadable PDFs)
    bold_body: bool = False  # most of the text is bold, so bold cannot mark a heading


def _set_apart(line: Line, ctx: _Context) -> bool:
    """Is the line typeset like a heading - larger than the running text, or bold?

    A document without any typography (nothing could be read about its fonts) has only the pattern to go by, so the
    answer is yes. In a document that is typeset, a page whose fonts could not be read (a quirk of some pypdf versions)
    gives no reason to accept a Title Case or ALL CAPS line: table cells and figure labels look just like that.
    """
    if ctx.body_size is None:
        return True
    if line.size is None:
        return False
    return line.size >= HEADING_SIZE_RATIO * ctx.body_size or (line.bold and not ctx.bold_body)


def _typeset_as_heading(line: Line, ctx: _Context) -> bool:
    """Is the line *known* to be larger or bolder than the running text? Unlike ``_set_apart``, unknown typography
    says no - a line is only joined to the ones below it when the type tells that they belong together."""
    return ctx.body_size is not None and line.size is not None and _set_apart(line, ctx)


def _block_start(prev: Line | None, cur: Line, p90: int, prev_was_heading: bool) -> bool:
    """Does ``cur`` start a new block (so a heading is plausible here)?"""
    if prev is None or prev_was_heading or cur.blank_before:
        return True
    return ends_sentence(prev.text) or len(prev.text) < 0.6 * p90


def _next_ok(nxt: Line | None, *, list_ok: bool = False) -> bool:
    """May a heading be followed by ``nxt``? Not by a line that goes on in lower case - that would be a wrapped
    sentence - unless that line is the first item of a list ("a) ...", "(i) ...") and ``list_ok`` says so."""
    if nxt is None:
        return False
    return not nxt.text[:1].islower() or (list_ok and bool(_LIST_MARKER.match(nxt.text)))


def _following_title(lines: list[Line], i: int) -> tuple[str, int]:
    """The title that follows a bare "Article 5" / "CHAPTER III" line, and how many lines it takes."""
    if i + 1 >= len(lines):
        return "", 0
    first = lines[i + 1].text
    if (
        len(first) > MAX_HEADING_CHARS + 20
        or first[:1].islower()
        or _LIST_MARKER.match(first)
        or _KEYWORD.match(first)
        or first[-1] in ",;"
        or (first[-1] == "." and len(first.split()) > 4)
        or len(first.split()) > 25
    ):
        return "", 0
    parts, used = [first], 1
    while i + 1 + used < len(lines) and len(" ".join(parts)) < MAX_HEADING_CHARS:
        more = lines[i + 1 + used].text
        if more[:1].islower() and not ends_sentence(parts[-1]) and not lines[i + 1 + used].blank_before:
            parts.append(more)
            used += 1
        else:
            break
    return " ".join(parts), used


def _keyword_heading(lines: list[Line], i: int, ctx: _Context, starts: bool) -> HeadingMatch | None:
    match = _KEYWORD.match(lines[i].text)
    if not match:
        return None
    if not (starts or _typeset_as_heading(lines[i], ctx)):
        return None  # a wrapped sentence can start a line with "Annex I. In certain cases, ..."
    keyword = match.group("kw").lower()
    label = f"{keyword.capitalize()} {match.group('id')}"
    title = (match.group("title") or "").strip()
    consumed = 1
    if title:
        if len(title) > MAX_HEADING_CHARS or len(title.split()) > 25:
            return None
        title = title.rstrip(".")
    else:
        # A bare "Article 5" is a heading when a title line follows it; otherwise it must start a block
        # (a wrapped sentence can also leave "Article 5" alone on a line).
        title, extra = _following_title(lines, i)
        consumed += extra
        following = lines[i + 1] if i + 1 < len(lines) else None
        if not title and not (starts and _next_ok(following)):
            return None
    return HeadingMatch(f"{label} – {title}" if title else label, keyword, consumed)


def _numbered_title(text: str, ctx: _Context, max_words: int = MAX_NUMBERED_WORDS) -> HeadingMatch | None:
    """``text`` as a heading with a number in front ("2.1 Scope", "IV. Title", "B. Title"), if it reads like one."""
    decimal = _DECIMAL.match(text)
    if decimal and _heading_title_ok(decimal.group("title"), max_words=max_words):
        depth = decimal.group("num").count(".") + 1
        return HeadingMatch(text.rstrip("."), f"decimal{min(depth, 5)}")
    roman = _ROMAN.match(text)
    if roman and _heading_title_ok(roman.group("title"), max_words=max_words):
        return HeadingMatch(text, "roman")
    if ctx.rule_level >= 2:
        alpha = _ALPHA.match(text)
        if alpha and _heading_title_ok(alpha.group("title"), max_words=12):
            return HeadingMatch(text, "alpha")
    return None


def _numbered_heading(lines: list[Line], i: int, ctx: _Context, starts: bool) -> HeadingMatch | None:
    cur = lines[i]
    text = cur.text
    section_sign = _SECTION_SIGN.match(text)
    if section_sign and starts and _heading_title_ok(section_sign.group("title"), max_words=25, allow_period=True):
        title = section_sign.group("title").rstrip(".")
        return HeadingMatch(f"§ {section_sign.group('num')} – {title}", "section_sign")
    code = _CODE_SECTION.match(text)
    if code and starts:
        return HeadingMatch(f"Section {code.group('num')}", "code_section", 1, code.group("rest") or "")
    if ctx.legal or not starts:
        return None
    # A numbered heading in bold or larger type may wrap ("1.2. Respect for the rule of law, ... fairness and" /
    # "privacy"); its lines are then in the same type. The longest reading that is a heading wins.
    parts = _wrapped_title(lines, i, ctx) if _typeset_as_heading(cur, ctx) else [text]
    for count in range(len(parts), 0, -1):
        following = lines[i + count] if i + count < len(lines) else None
        if not _next_ok(following, list_ok=True) or (count == 1 and len(text) > 0.9 * ctx.p90):
            continue
        words = MAX_NUMBERED_WORDS if count == 1 else MAX_WRAPPED_NUMBERED_WORDS
        match = _numbered_title(" ".join(parts[:count]), ctx, words)
        if match:
            return HeadingMatch(match.text, match.scheme, count)
    return None


def _outline_heading(lines: list[Line], i: int, ctx: _Context) -> HeadingMatch | None:
    cur = lines[i]
    current = _norm_title(cur.text)
    if not current:
        return None
    for page in (cur.page, cur.page - 1):
        for norm, level in ctx.outline.get(page, ()):
            if current == norm:
                return HeadingMatch(cur.text, f"outline{level}")
            if norm.startswith(current + " ") and i + 1 < len(lines):
                if _norm_title(f"{cur.text} {lines[i + 1].text}") == norm:
                    return HeadingMatch(f"{cur.text} {lines[i + 1].text}", f"outline{level}", 2)
    return None


def _wrapped_title(lines: list[Line], i: int, ctx: _Context) -> list[str]:
    """The lines of a Title Case heading that wraps over several lines (a heading in a narrow column does).

    Only lines in exactly the type of the first one count - the same size and boldness, and the caller has already
    checked that this type is set apart from the running text - directly below it on the same page. Without readable
    typography nothing is joined.
    """
    first = lines[i]
    parts = [first.text]
    if first.size is None:
        return parts
    while len(parts) < MAX_WRAPPED_LINES and i + len(parts) < len(lines) and len(" ".join(parts)) < MAX_HEADING_CHARS:
        more = lines[i + len(parts)]
        if (
            more.page != first.page
            or more.blank_before
            or more.size is None
            or abs(more.size - first.size) >= SAME_TYPE_TOLERANCE
            or more.bold != first.bold
            or ends_sentence(parts[-1])
            or len(more.text) > 0.85 * ctx.p90
            or _LIST_MARKER.match(more.text)
            or _KEYWORD.match(more.text)
            or _numbered_start(more.text)
            or _cut_bracket(more.text)
        ):
            break
        parts.append(more.text)
    return parts


def _numbered_start(text: str) -> bool:
    """Does the line begin like a numbered heading of its own (1.2 Title, III. Title, § 5 Title, B. Title)?"""
    return bool(_DECIMAL.match(text) or _ROMAN.match(text) or _ALPHA.match(text) or _SECTION_SIGN.match(text) or _CODE_SECTION.match(text))


def detect_heading(lines: list[Line], i: int, ctx: _Context, prev_was_heading: bool) -> HeadingMatch | None:
    cur = lines[i]
    prev = lines[i - 1] if i else None
    nxt = lines[i + 1] if i + 1 < len(lines) else None

    outline = _outline_heading(lines, i, ctx) if ctx.outline else None
    if outline or ctx.outline_only:
        return outline

    starts = _block_start(prev, cur, ctx.p90, prev_was_heading)
    if _WHEREAS.match(cur.text):
        return HeadingMatch("Recitals", "recitals")
    if _ADOPTED.match(cur.text):
        return HeadingMatch("Enacting terms", "enacting")  # what follows is the binding text, no longer a recital
    keyword = _keyword_heading(lines, i, ctx, starts)
    if keyword:
        return keyword
    numbered = _numbered_heading(lines, i, ctx, starts)
    if numbered:
        return numbered
    if not starts or _LIST_MARKER.match(cur.text) or nxt is None:
        return None
    if _is_all_caps(cur.text) and _set_apart(cur, ctx):
        consumed, parts = 1, [cur.text]
        while i + consumed < len(lines) and len(" ".join(parts)) < 160:
            more = lines[i + consumed]
            if more.blank_before or not _is_all_caps(more.text) or _KEYWORD.match(more.text) or _LIST_MARKER.match(more.text):
                break
            parts.append(more.text)
            consumed += 1
        return HeadingMatch(" ".join(parts), "allcaps", consumed)
    if ctx.rule_level >= 2 and not ctx.legal and _is_title_case(cur.text) and not nxt.text[:1].islower():
        if len(cur.text) < 0.85 * ctx.p90 and _set_apart(cur, ctx):
            parts = _wrapped_title(lines, i, ctx)
            return HeadingMatch(" ".join(parts), "titlecase", len(parts))
    return None


# ------------------------------------------------------------------------------------- assembling


def _flatten(pages: list[PageLines]) -> list[Line]:
    """All body lines in reading order. A list marker that sits alone on its line is joined to the next line."""
    stream: list[Line] = []
    for page in pages:
        blank = False
        index = 0
        lines = page.lines
        while index < len(lines):
            raw = lines[index]
            index += 1
            if not raw.strip():
                blank = True
                continue
            text = raw.strip()
            indent = len(raw) - len(raw.lstrip(" "))
            size, bold = getattr(raw, "size", None), getattr(raw, "bold", False)
            if _LONE_MARKER.match(text) and index < len(lines) and lines[index].strip():
                following = lines[index]
                text = f"{text} {following.strip()}"
                index += 1
                size, bold = getattr(following, "size", size), getattr(following, "bold", bold)  # the marker joins the text
            stream.append(Line(text, page.number, indent, blank and bool(stream), size, bold))
            blank = False
    return stream


def _percentile_width(stream: list[Line]) -> int:
    widths = sorted(len(line.text) for line in stream if len(line.text) >= PARAGRAPH_MIN_LINE_CHARS)
    if not widths:
        return 80
    return widths[int(0.9 * (len(widths) - 1))]


def _body_style(stream: list[Line]) -> tuple[float | None, bool]:
    """The font size of the running text (see ``pick_body_size``) and whether that text is mostly bold."""
    by_size: Counter = Counter()
    total = typeset = bold_chars = 0
    for line in stream:
        characters = len(line.text)
        total += characters
        if line.size:
            typeset += characters
            by_size[round(line.size * 2) / 2] += characters
            bold_chars += characters if line.bold else 0
    if not total or typeset < MIN_TYPESET_SHARE * total:
        return None, False
    return pick_body_size(by_size), bold_chars > BOLD_BODY_SHARE * typeset


def _is_paragraph_break(prev: Line, cur: Line, p90: int) -> bool:
    if cur.blank_before or _LIST_MARKER.match(cur.text):
        return True
    if ends_sentence(prev.text):
        if len(prev.text) < 0.7 * p90:
            return True
        if cur.indent > 0 and prev.indent == 0:
            return True
    return len(prev.text) < 0.45 * p90 and cur.text[:1].isupper()


def _starts_next_number(paragraph: list[Line], line: Line) -> bool:
    """Does ``line`` open paragraph n+1 of a numbered series whose paragraph n is the one in progress?

    Treaties and statutes number their paragraphs "1 ... 2 ..." without a full stop; a paragraph that ends
    close to the right margin gives no other sign that the next one starts.
    """
    first, current = _PARAGRAPH_NUMBER.match(paragraph[0].text), _PARAGRAPH_NUMBER.match(line.text)
    return bool(first and current and int(current.group(1)) == int(first.group(1)) + 1)


def _finish(text: str, vocab: Counter) -> str:
    return repair_split_words(repair_spacing(text), vocab)


def _join(lines: list[str], vocab: Counter) -> str:
    text = lines[0]
    for nxt in lines[1:]:
        text = join_wrapped_lines(text, nxt, vocab)
    return _finish(text, vocab)


def _opens_quoted_division(text: str) -> bool:
    """Does the line open a quotation of a whole article or annex ("‘Article 75a")? An act that amends another one
    quotes the text it inserts, and the inserted text has headings of its own that are not the act's."""
    return text.startswith("‘") and bool(_KEYWORD.match(text[1:].lstrip()))


def _closes_quote(text: str) -> bool:
    return bool(_QUOTE_CLOSE.search(text))


def _article_number(text: str) -> int | None:
    match = _KEYWORD.match(text)
    if match and match.group("kw").lower() == "article" and match.group("id").isdigit():
        return int(match.group("id"))
    return None


def _assemble(stream: list[Line], notes_by_page: dict[int, list[RawNote]], ctx: _Context, vocab: Counter):
    """One pass over the lines. Returns the blocks (heading levels still 0) and the scheme of each heading."""
    blocks: list[Block] = []
    schemes: list[str] = []
    paragraph: list[Line] = []
    pending: list[RawNote] = []
    queued_pages = {page: False for page in notes_by_page}

    def queue_notes_before(page: int | None) -> None:
        for number in sorted(notes_by_page):
            if queued_pages[number] or (page is not None and number >= page):
                continue
            queued_pages[number] = True
            pending.extend(notes_by_page[number])

    def flush() -> None:
        if paragraph:
            text = _join([line.text for line in paragraph], vocab)
            if text:
                blocks.append(Block(PARAGRAPH, text, paragraph[0].page, paragraph[-1].page))
            paragraph.clear()
        for note in pending:
            body = _join(note.lines, vocab)
            if body:
                label = f"[{note.number}] " if note.number is not None else ""
                blocks.append(Block(FOOTNOTE, label + body, note.page, note.page))
        pending.clear()

    i = 0
    prev_was_heading = True
    last_page = None
    last_article = 0  # number of the last "Article N" heading of the act itself
    quote_page = None  # page where a quoted insertion began, while one is open
    resume_at = 0  # the number of the article that ends the quotation if its closing mark was not seen
    while i < len(stream):
        line = stream[i]
        if line.page != last_page:
            queue_notes_before(line.page)
            last_page = line.page
        if quote_page is None:
            if _opens_quoted_division(line.text):
                quote_page, resume_at = line.page, last_article + 1
        elif _article_number(line.text) == resume_at or line.page - quote_page > MAX_QUOTE_PAGES:
            quote_page = None  # the act's own next article is here, so the quotation is over
        inside_quote = quote_page is not None
        if inside_quote and _closes_quote(line.text):
            quote_page = None  # this is the last line of the quotation: still part of it
        heading = None if inside_quote else detect_heading(stream, i, ctx, prev_was_heading)
        if heading:
            flush()
            blocks.append(Block(HEADING, _finish(heading.text, vocab), line.page, line.page))
            schemes.append(heading.scheme)
            last_article = _article_number(heading.text) or last_article
            i += heading.consumed
            prev_was_heading = True
            if heading.rest:
                paragraph.append(Line(heading.rest, line.page, 0, False))
                prev_was_heading = False
            continue
        if paragraph and (_is_paragraph_break(paragraph[-1], line, ctx.p90) or _starts_next_number(paragraph, line)):
            flush()
        paragraph.append(line)
        prev_was_heading = False
        i += 1
    queue_notes_before(None)
    flush()
    return blocks, schemes


def _scheme_rank(scheme: str) -> float:
    if scheme in _KEYWORD_RANK:
        return _KEYWORD_RANK[scheme]
    if scheme in _OTHER_RANK:
        return _OTHER_RANK[scheme]
    if scheme.startswith("decimal"):
        return 12 + int(scheme[len("decimal") :])
    if scheme.startswith("outline"):
        return int(scheme[len("outline") :])
    return 20


def _assign_levels(blocks: list[Block], schemes: list[str]) -> list[Block]:
    ranks = sorted({_scheme_rank(s) for s in schemes})
    level_of = {rank: min(index + 1, 6) for index, rank in enumerate(ranks)}
    result, heading_index = [], 0
    for block in blocks:
        if block.kind == HEADING:
            level = level_of[_scheme_rank(schemes[heading_index])]
            heading_index += 1
            block = Block(HEADING, block.text, block.page, block.page_end, level)
        result.append(block)
    return result


def _outline_by_page(outline: list[OutlineEntry]) -> dict[int, list[tuple[str, int]]]:
    table: dict[int, list[tuple[str, int]]] = {}
    for entry in outline:
        norm = _norm_title(entry.title)
        if norm:
            table.setdefault(entry.page, []).append((norm, min(max(entry.level, 1), 6)))
    return table


def build_blocks(
    pages: list[PageLines],
    outline: list[OutlineEntry] | None = None,
    *,
    use_outline: bool = True,
) -> tuple[list[Block], StructureReport]:
    """Rebuild headings, paragraphs and footnotes from cleaned page lines."""
    stream = _flatten(pages)
    notes_by_page = {page.number: page.notes for page in pages if page.notes}
    texts = [line.text for line in stream]
    texts += [text for notes in notes_by_page.values() for note in notes for text in note.lines]
    vocab = build_vocab(texts)
    p90 = _percentile_width(stream)
    total_words = max(1, sum(len(line.text.split()) for line in stream))
    body_size, bold_body = _body_style(stream)

    def run(rule_level: int, table: dict, outline_only: bool, legal: bool = False):
        ctx = _Context(p90, rule_level, table, outline_only, legal, body_size, bold_body)
        return _assemble(stream, notes_by_page, ctx, vocab)

    # Pattern-based headings. If the weak rules produce an implausible number of headings, drop them.
    for rule_level in (2, 1, 0):
        blocks, schemes = run(rule_level, {}, False)
        if 100.0 * len(schemes) / total_words <= MAX_HEADING_DENSITY:
            break
    report = StructureReport(rule_level=rule_level)
    # In a legal act ("Article 1", "Article 2", ...) the structure is given by those labels: "1." and "2." number
    # the paragraphs of an article, and short Title Case lines ("The President") are signatures, not headings.
    report.legal_act = schemes.count("article") >= 3
    weak = [s for s in schemes if s in ("roman", "alpha", "titlecase") or s.startswith("decimal")]
    if report.legal_act and weak:
        blocks, schemes = run(rule_level, {}, False, legal=True)

    # Bookmarks win when they match enough lines and are about as detailed as the pattern-based result.
    outline_table = _outline_by_page(outline or []) if use_outline else {}
    if outline_table:
        outline_blocks, outline_schemes = run(rule_level, outline_table, True)
        report.outline_matches = len(outline_schemes)
        if len(outline_schemes) >= MIN_OUTLINE_MATCHES and len(outline_schemes) >= 0.7 * len(schemes):
            blocks, schemes = outline_blocks, outline_schemes
            report.outline_used = True

    blocks = _assign_levels(blocks, schemes)
    report.headings = len(schemes)
    report.schemes = Counter(schemes)
    report.paragraphs = sum(1 for b in blocks if b.kind == PARAGRAPH)
    report.footnotes = sum(1 for b in blocks if b.kind == FOOTNOTE)
    return blocks, report
