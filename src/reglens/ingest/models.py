"""Plain data classes shared by the ingestion modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

HEADING = "heading"
PARAGRAPH = "paragraph"
FOOTNOTE = "footnote"


@dataclass(frozen=True)
class Block:
    """One piece of a document, in reading order.

    ``level`` is only meaningful for headings (1 = highest). Pages are 1-based PDF page numbers; they are None
    for web pages.
    """

    kind: str
    text: str
    page: int | None = None
    page_end: int | None = None
    level: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, 0) or k in ("kind", "text")}

    @classmethod
    def from_dict(cls, data: dict) -> Block:
        return cls(
            kind=data["kind"],
            text=data["text"],
            page=data.get("page"),
            page_end=data.get("page_end"),
            level=data.get("level", 0),
        )


class TextLine(str):
    """A line of extracted PDF text that remembers how it was typeset.

    ``size`` is the dominant font size of the line in points (None when unknown) and ``bold`` says whether most of
    its characters are set in a bold face. The object *is* a ``str``, so every text operation works on it; slicing a
    list of lines or filtering it keeps the objects, so the typography travels with the line through the cleaning
    steps. Operations that build a new string (``strip``, ``join``) return a plain ``str``.
    """

    __slots__ = ("size", "bold")

    def __new__(cls, text: str = "", size: float | None = None, bold: bool = False) -> TextLine:
        line = super().__new__(cls, text)
        line.size = size
        line.bold = bold
        return line


@dataclass(frozen=True)
class OutlineEntry:
    """A bookmark of the PDF: its title, depth (1 = top) and the 1-based page it points to."""

    title: str
    level: int
    page: int
