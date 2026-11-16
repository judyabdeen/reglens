"""Read a saved web page and return its main text as headings and paragraphs.

Web pages carry their structure in tags, so no guessing is needed - but they also carry menus, cookie banners,
breadcrumbs and footers that must not end up in the index. The approach:

1. use the page's ``<main>`` (or ``role="main"`` / ``<article>``) element if it has one, otherwise the ``<body>``;
2. drop non-content tags (scripts, forms, navigation, footers, sidebars) and elements whose class or id names
   boilerplate (cookie, breadcrumb, share, ...);
3. walk what is left in reading order: ``<h1>``-``<h6>`` become headings, paragraphs, list items and table rows
   become paragraphs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:  # beautifulsoup4 is in requirements.txt; if it is missing only the web pages fail, not the PDFs
    from bs4 import BeautifulSoup, NavigableString, Tag
    from bs4.element import CData, Comment, Declaration, Doctype, ProcessingInstruction
except ImportError:  # pragma: no cover - exercised only on a machine without beautifulsoup4
    BeautifulSoup = None

from reglens.ingest.models import HEADING, PARAGRAPH, Block
from reglens.ingest.pdfread import ExtractionError
from reglens.ingest.textclean import normalize_chars

DROP_TAGS = frozenset(
    "script style noscript template svg canvas iframe form button input select textarea nav footer aside "
    "dialog menu object embed video audio map".split()
)
BOILERPLATE_WORDS = frozenset(
    "cookie cookies banner breadcrumb breadcrumbs skip sidebar menu navbar nav footer social share sharing "
    "pagination feedback newsletter search modal popup toolbar masthead promo advert advertisement print "
    "subscribe backtotop".split()
)
BOILERPLATE_LINES = frozenset(
    {
        "skip to main content", "print this page", "share this page", "back to top", "on this page",
        "was this page useful?", "is this page useful?", "yes no", "return to top",
    }
)  # fmt: skip
HEADING_TAGS = {f"h{n}": n for n in range(1, 7)}
LEAF_BLOCKS = frozenset("p li blockquote pre dd dt figcaption caption summary address".split())
CONTAINERS = frozenset(
    "div section article main ul ol dl table thead tbody tfoot tr figure details header body html fieldset "
    "center hgroup".split()
)
BLOCKISH = frozenset(LEAF_BLOCKS | CONTAINERS | set(HEADING_TAGS))
_TOKEN = re.compile(r"[a-z]+")


@dataclass
class WebExtraction:
    title: str
    blocks: list[Block]
    root: str  # which element supplied the text: "main", "article", "body", ...
    chars: Counter = field(default_factory=Counter)


def decode_html(raw: bytes) -> str:
    """Decode saved HTML: UTF-8 (with or without a byte-order mark), else the declared charset, else cp1252."""
    for encoding in ("utf-8-sig",):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    declared = re.search(rb"charset=[\"']?([A-Za-z0-9_-]+)", raw[:4096])
    if declared:
        try:
            return raw.decode(declared.group(1).decode("ascii"))
        except (LookupError, UnicodeDecodeError):
            pass
    return raw.decode("cp1252", errors="replace")


MAX_BOILERPLATE_CHARS = 500


def _looks_like_boilerplate(tag: Tag) -> bool:
    """Hidden elements, or small elements whose class or id names a menu, banner, share box and the like.

    The size limit protects real content: a page wrapper called "content-with-sidebar" is large, a cookie banner is not.
    """
    if not isinstance(tag.attrs, dict):
        return False
    if tag.get("hidden") is not None or str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    names = tag.get("class") or []
    names = names if isinstance(names, list) else [names]
    words = set(_TOKEN.findall(" ".join([*names, str(tag.get("id", ""))]).lower()))
    if not words & BOILERPLATE_WORDS:
        return False
    return len(tag.get_text(" ", strip=True)) <= MAX_BOILERPLATE_CHARS


def _choose_root(soup: BeautifulSoup) -> tuple[Tag, str]:
    for finder, label in (
        (lambda: soup.find("main"), "main"),
        (lambda: soup.find(attrs={"role": "main"}), "main"),
        (lambda: soup.find("article"), "article"),
    ):
        found = finder()
        if found is not None and len(found.get_text(" ", strip=True)) > 400:
            return found, label
    return (soup.body or soup), "body"


def _clean_text(text: str, stats: Counter) -> str:
    return " ".join(normalize_chars(text, stats).split())


def extract_web_page(path: Path) -> WebExtraction:
    if BeautifulSoup is None:  # pragma: no cover
        raise ExtractionError("beautifulsoup4 is not installed; run: pip install -r requirements.txt")
    soup = BeautifulSoup(decode_html(path.read_bytes()), "html.parser")
    stats: Counter = Counter()
    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text(" ", strip=True), stats) if title_tag else ""

    root, root_name = _choose_root(soup)
    for tag in root.find_all(DROP_TAGS):
        tag.decompose()
    for line_break in root.find_all("br"):
        line_break.replace_with(" ")
    for tag in [t for t in root.find_all(True) if _looks_like_boilerplate(t)]:
        if not tag.decomposed:
            tag.decompose()

    paragraphs: list[tuple[str, str, int]] = []  # (kind, text, heading tag number)
    buffer: list[str] = []

    def flush() -> None:
        text = _clean_text(" ".join(buffer), stats)
        buffer.clear()
        if text and text.lower() not in BOILERPLATE_LINES:
            paragraphs.append((PARAGRAPH, text, 0))

    def emit(kind: str, text: str, number: int = 0) -> None:
        text = _clean_text(text, stats)
        if text and text.lower() not in BOILERPLATE_LINES:
            paragraphs.append((kind, text, number))

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, (Comment, Doctype, Declaration, ProcessingInstruction, CData)):
                    buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in HEADING_TAGS:
                flush()
                emit(HEADING, child.get_text(), HEADING_TAGS[name])
            elif name == "tr":
                flush()
                cells = [c.get_text() for c in child.find_all(["td", "th"])]
                emit(PARAGRAPH, " | ".join(_clean_text(c, stats) for c in cells if c.strip()))
            elif name in LEAF_BLOCKS or name in CONTAINERS:
                flush()
                if name in LEAF_BLOCKS and child.find(BLOCKISH) is None:
                    prefix = "• " if name == "li" else ""
                    emit(PARAGRAPH, prefix + child.get_text())
                else:
                    walk(child)
                    flush()
            elif name == "br":
                buffer.append(" ")
            elif child.find(BLOCKISH) is not None:
                flush()
                walk(child)
                flush()
            else:
                buffer.append(child.get_text())
        # text directly inside the container stays in `buffer` until the next block starts

    walk(root)
    flush()

    heading_numbers = sorted({number for kind, _text, number in paragraphs if kind == HEADING})
    level_of = {number: index + 1 for index, number in enumerate(heading_numbers)}
    blocks: list[Block] = []
    previous = None
    for kind, text, number in paragraphs:
        if (kind, text) == previous:
            continue  # the same line twice in a row (a page shows a title in two places)
        previous = (kind, text)
        blocks.append(Block(kind, text, level=level_of.get(number, 0) if kind == HEADING else 0))
    return WebExtraction(title=title, blocks=blocks, root=root_name, chars=stats)
