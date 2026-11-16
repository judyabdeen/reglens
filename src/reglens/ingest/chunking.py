"""Split a structured document into chunks - the units that are embedded, stored and retrieved.

Two strategies are implemented so that they can be compared in the evaluation:

``structure`` (default)
    "Semantic" chunking in the sense of the assignment: the boundaries follow the meaning units the authors
    made - articles, chapters, numbered sections. Each section becomes one chunk if it fits; short neighbouring
    sections are merged until a chunk is useful; a section that is too long is cut at paragraph and sentence
    boundaries, with an overlap so that a sentence cut at the border is still complete in one of the two
    chunks. Every chunk records the path of headings above it, and the text that is embedded starts with that
    path ("breadcrumb") so a passage such as "The controller shall ..." still says which article it is from.
    Footnotes are collected into separate chunks, so long lists of case citations do not dilute the meaning of
    the passage they are attached to.

``fixed``
    The naive baseline: a sliding window of a fixed number of words with overlap, ignoring all structure.

Sizes are counted in words. With the default 320-word limit a chunk stays under the 512-token input limit of
the embedding model (legal English averages about 1.4 tokens per word).
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reglens.corpus.manifest import Source
from reglens.ingest.models import FOOTNOTE, HEADING, Block

STRATEGIES = ("structure", "fixed")
TEXT = "text"
NOTES = "notes"
MAX_NOTE_PAGE_GAP = 1  # footnotes this many pages apart or further do not share a chunk, so its page range stays true
_URL = re.compile(r"https?://\S+|www\.\S+")
_ABBREVIATIONS = frozenset(
    "no nos art arts sec secs para paras p pp fig figs eq al etc cf vs inc ltd co corp dr mr mrs ms st fed reg stat "
    "cir supp dkt ed eds vol ch cl resp approx e.g i.e u.s u.s.c c.f.r ibid id seq op cit viz ca dept gov ann rev "
    "sub subs subd pt div tit jan feb mar apr jun jul aug sep sept oct nov dec "
    "v exec pub cong rep sen gen admin assoc mgmt tech techs sols int intl natl mktg compl ex rel".split()
)
_SENTENCE_GAP = re.compile(r"([.!?])([\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class ChunkOptions:
    strategy: str = "structure"
    target_words: int = 240  # a chunk is closed once it reaches this size ...
    max_words: int = 320  # ... and never grows beyond this
    min_words: int = 60  # sections smaller than this are merged with their neighbour
    overlap_words: int = 40  # words repeated at the start of the next chunk when one section is cut

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, not {self.strategy!r}")
        if not 0 < self.min_words <= self.target_words <= self.max_words:
            raise ValueError("chunk sizes must satisfy 0 < min_words <= target_words <= max_words")
        if not 0 <= self.overlap_words < self.target_words:
            raise ValueError("overlap_words must be at least 0 and smaller than target_words")
        if self.max_words - self.overlap_words < self.min_words:
            raise ValueError("max_words minus overlap_words must be at least min_words")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    kind: str  # "text" or "notes" (footnotes)
    ordinal: int
    text: str  # what the reader and the language model see
    embed_text: str  # what is embedded: the same text with a short "document | section" prefix
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    n_words: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**{name: data.get(name) for name in cls.__dataclass_fields__ if name in data})

    @property
    def section(self) -> str:
        """Human-readable location, for citations: the last two headings of the path."""
        return " > ".join(self.section_path[-2:])


def context_prefix(chunk: Chunk) -> str:
    """The "document | section" line in front of the text in ``chunk.embed_text`` ("" for the fixed baseline)."""
    if chunk.kind == NOTES:  # "<document | section | footnotes>\n<notes>": the text starts with a "Footnotes:" label
        first = chunk.embed_text.partition("\n")[0].strip()
        return first if first.endswith("| footnotes") else ""
    body = _URL.sub("[link]", chunk.text)
    if body and chunk.embed_text.endswith(body):
        return chunk.embed_text[: len(chunk.embed_text) - len(body)].strip()
    return ""


def read_chunks(path: Path | str) -> list[Chunk]:
    """Read ``chunks.jsonl`` (one JSON object per line). A bad line is reported with its line number."""
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise TypeError("expected a JSON object")
                chunks.append(Chunk.from_dict(data))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}, line {number}: not a valid chunk record ({exc})") from exc
    return chunks


# ----------------------------------------------------------------------------------- text helpers


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, but not after abbreviations ("No.", "Art.", "U.S.C.") or initials ("J. D. Vance")."""
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_GAP.finditer(text):
        # Only the last few characters decide whether this is an abbreviation or an initial. Looking at all the
        # text since the last split would make a long run of abbreviations ("No. No. No. ...") quadratic.
        cut = max(start, match.start(1) - 40)
        tail = text[cut : match.start(1)]
        previous = re.search(r"([A-Za-z.]+)$", tail)
        if match.group(1) == "." and previous and previous.group(1).lower().strip(".") in _ABBREVIATIONS:
            continue
        if match.group(1) == "." and (re.search(r"\s[A-Z]$", tail) or (cut == start and re.fullmatch(r"[A-Z]", tail))):
            continue
        parts.append(text[start : match.end(2)].strip())
        start = match.end()
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _words(text: str) -> int:
    return len(text.split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def clean_path(source: Source, path: list[str] | tuple[str, ...]) -> list[str]:
    """Drop headings that merely repeat the document's own title (many documents print it as their first heading)."""
    title = _tokens(source.title)
    kept = []
    for heading in path:
        words = _tokens(heading)
        if words and title and len(words & title) / len(words | title) >= 0.6:
            continue
        kept.append(heading)
    return kept


def _tail(text: str, count: int) -> str:
    """About the last ``count`` words of ``text``, starting at a sentence start when there is one."""
    words = text.split()
    if count <= 0:
        return ""
    if len(words) <= count:
        return text
    tail = " ".join(words[-count:])
    gap = _SENTENCE_GAP.search(tail)
    if gap and gap.end() < 0.8 * len(tail):
        tail = tail[gap.end() :]
    return tail


def embed_prefix(source: Source, path: list[str] | tuple[str, ...], extra: str = "") -> str:
    """"Short document title | last two headings" - context that makes a bare passage self-explanatory."""
    title = " ".join(source.title.split()[:12])
    crumb = " > ".join(path[-2:])
    crumb = " ".join(crumb.split()[:18])
    parts = [part for part in (title, crumb, extra) if part]
    return " | ".join(parts)


def _make_embed_text(source: Source, path, text: str, extra: str = "") -> str:
    return f"{embed_prefix(source, path, extra)}\n{_URL.sub('[link]', text)}"


def _page_range(pages: list[int]) -> tuple[int | None, int | None]:
    return (min(pages), max(pages)) if pages else (None, None)


# ------------------------------------------------------------------------------- structure strategy


@dataclass
class _Para:
    text: str
    page: int | None
    page_end: int | None

    @property
    def words(self) -> int:
        return _words(self.text)


@dataclass
class _Unit:
    """One heading and the paragraphs under it, up to the next heading of any level."""

    path: tuple[str, ...]
    heading: str | None
    heading_page: int | None
    paras: list[_Para] = field(default_factory=list)
    notes: list[_Para] = field(default_factory=list)

    @property
    def words(self) -> int:
        return (_words(self.heading) if self.heading else 0) + sum(p.words for p in self.paras)

    @property
    def pages(self) -> list[int]:
        pages = [p for para in self.paras for p in (para.page, para.page_end) if p is not None]
        if self.heading_page is not None:
            pages.append(self.heading_page)
        return pages


@dataclass
class _Draft:
    """A chunk under construction."""

    path: tuple[str, ...]
    sections: list[tuple[str | None, list[_Para]]] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    overlap: str = ""

    @property
    def words(self) -> int:
        count = _words(self.overlap)
        for heading, paras in self.sections:
            count += (_words(heading) if heading else 0) + sum(p.words for p in paras)
        return count

    def text(self) -> str:
        parts = [self.overlap] if self.overlap else []
        for heading, paras in self.sections:
            if heading:
                parts.append(heading)
            parts.extend(p.text for p in paras)
        return "\n\n".join(parts)


def _build_units(blocks: list[Block]) -> list[_Unit]:
    units: list[_Unit] = []
    stack: list[tuple[int, str]] = []
    current = _Unit((), None, None)
    for block in blocks:
        if block.kind == HEADING:
            if current.paras or current.notes:
                units.append(current)
            stack = [(level, text) for level, text in stack if level < block.level]
            stack.append((block.level, block.text))
            current = _Unit(tuple(text for _level, text in stack), block.text, block.page)
        elif block.kind == FOOTNOTE:
            current.notes.append(_Para(block.text, block.page, block.page_end))
        else:
            current.paras.append(_Para(block.text, block.page, block.page_end))
    if current.paras or current.notes:
        units.append(current)
    return units


def _split_paragraph(para: _Para, limit: int, target: int) -> list[_Para]:
    """Cut a paragraph longer than ``limit`` words at sentence ends (a sentence longer than ``limit`` is cut at words)."""
    if para.words <= limit:
        return [para]
    target = min(target, limit)
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(para.text):
        count = _words(sentence)
        if count > limit:
            if current:
                pieces.append(" ".join(current))
                current, size = [], 0
            words = sentence.split()
            for start in range(0, len(words), target):
                pieces.append(" ".join(words[start : start + target]))
            continue
        if current and size + count > target:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += count
    if current:
        pieces.append(" ".join(current))
    return [_Para(piece, para.page, para.page_end) for piece in pieces]


def _split_unit(unit: _Unit, options: ChunkOptions) -> list[_Draft]:
    """Cut a section that is longer than ``max_words`` into drafts, with overlap between consecutive drafts.

    The overlap is added on top of every draft but the first, so the pieces are cut at ``max_words`` minus the
    overlap: no finished chunk is longer than ``max_words``.
    """
    limit = options.max_words - options.overlap_words
    target = min(options.target_words, limit)
    groups: list[list[_Para]] = []
    current: list[_Para] = []
    size = _words(unit.heading) if unit.heading else 0
    for para in unit.paras:
        for part in _split_paragraph(para, limit, target):
            if current and (size + part.words > limit or size >= target):
                groups.append(current)
                current, size = [], 0
            current.append(part)
            size += part.words
    if current:
        groups.append(current)
    if len(groups) > 1:  # a tiny last piece joins the one before it
        last_words = sum(p.words for p in groups[-1])
        before_words = sum(p.words for p in groups[-2])
        if last_words < options.min_words and before_words + last_words <= limit:
            groups[-2].extend(groups.pop())

    drafts: list[_Draft] = []
    previous_text = ""
    for index, group in enumerate(groups):
        draft = _Draft(unit.path)
        draft.sections.append((unit.heading if index == 0 else None, group))
        draft.pages = [p for para in group for p in (para.page, para.page_end) if p is not None]
        if index == 0 and unit.heading_page is not None:
            draft.pages.append(unit.heading_page)
        if index > 0:
            draft.overlap = _tail(previous_text, options.overlap_words)
        previous_text = " ".join(p.text for p in group)
        drafts.append(draft)
    return drafts


def _pack_units(units: list[_Unit], options: ChunkOptions) -> list[_Draft]:
    drafts: list[_Draft] = []
    pending: _Draft | None = None
    for unit in units:
        if not unit.paras:
            continue
        if unit.words > options.max_words:
            if pending:
                drafts.append(pending)
                pending = None
            drafts.extend(_split_unit(unit, options))
            continue
        if pending is not None and pending.words < options.min_words and pending.words + unit.words <= options.max_words:
            pending.sections.append((unit.heading, unit.paras))
            pending.pages.extend(unit.pages)
            continue
        if pending:
            drafts.append(pending)
        pending = _Draft(unit.path, [(unit.heading, unit.paras)], list(unit.pages))
    if pending:
        drafts.append(pending)
    if len(drafts) > 1 and drafts[-1].words < options.min_words // 2:
        last, before = drafts[-1], drafts[-2]
        if not last.overlap and before.words + last.words <= options.max_words:
            before.sections.extend(last.sections)
            before.pages.extend(last.pages)
            drafts.pop()
    return drafts


def _far_from(draft: _Draft, part: _Para) -> bool:
    """Is the note printed more than ``MAX_NOTE_PAGE_GAP`` pages after the last page of the notes gathered so far?
    (A chunk of notes from pages 8 to 16 would send the reader to the wrong page.)"""
    return bool(draft.pages) and part.page is not None and part.page - max(draft.pages) > MAX_NOTE_PAGE_GAP


def _pack_notes(units: list[_Unit], options: ChunkOptions) -> list[_Draft]:
    """Footnotes of consecutive sections are gathered into chunks of about ``target_words``, as long as they are
    printed on neighbouring pages."""
    drafts: list[_Draft] = []
    pending: _Draft | None = None
    for unit in units:
        for note in unit.notes:
            for part in _split_paragraph(note, options.max_words, options.target_words):
                if pending is not None and (pending.words + part.words > options.target_words or _far_from(pending, part)):
                    drafts.append(pending)
                    pending = None
                if pending is None:
                    pending = _Draft(unit.path, [(None, [])])
                pending.sections[0][1].append(part)
                pending.pages.extend(p for p in (part.page, part.page_end) if p is not None)
    if pending:
        drafts.append(pending)
    return drafts


def _chunk_structure(source: Source, blocks: list[Block], options: ChunkOptions) -> list[Chunk]:
    units = _build_units(blocks)
    chunks: list[Chunk] = []
    for ordinal, draft in enumerate(_pack_units(units, options), start=1):
        text = draft.text()
        start, end = _page_range(draft.pages)
        path = clean_path(source, draft.path)
        chunks.append(
            Chunk(
                chunk_id=f"{source.id}:{ordinal:04d}",
                doc_id=source.id,
                kind=TEXT,
                ordinal=ordinal,
                text=text,
                embed_text=_make_embed_text(source, path, text),
                section_path=path,
                page_start=start,
                page_end=end,
                n_words=_words(text),
            )
        )
    for ordinal, draft in enumerate(_pack_notes(units, options), start=1):
        body = draft.text()
        start, end = _page_range(draft.pages)
        path = clean_path(source, draft.path)
        chunks.append(
            Chunk(
                chunk_id=f"{source.id}:n{ordinal:03d}",
                doc_id=source.id,
                kind=NOTES,
                ordinal=ordinal,
                text="Footnotes:\n\n" + body,
                embed_text=_make_embed_text(source, path, body, "footnotes"),
                section_path=path,
                page_start=start,
                page_end=end,
                n_words=_words(body),
            )
        )
    return chunks


# --------------------------------------------------------------------------------- fixed strategy


def _chunk_fixed(source: Source, blocks: list[Block], options: ChunkOptions) -> list[Chunk]:
    words: list[str] = []
    starts: list[int] = []  # word offset at which each block begins
    paths: list[tuple[str, ...]] = []  # heading path in force at each block
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind == HEADING:
            stack = [(level, text) for level, text in stack if level < block.level]
            stack.append((block.level, block.text))
        starts.append(len(words))
        paths.append(tuple(text for _level, text in stack))
        words.extend(block.text.split())

    chunks: list[Chunk] = []
    step = options.target_words - options.overlap_words
    begin = 0
    while begin < len(words):
        end = min(begin + options.target_words, len(words))
        if len(words) - end < options.min_words and len(words) - begin <= options.max_words:
            end = len(words)  # do not leave a tiny tail
        first = bisect_right(starts, begin) - 1
        last = bisect_right(starts, end - 1) - 1
        pages = [p for b in blocks[first : last + 1] for p in (b.page, b.page_end) if p is not None]
        text = " ".join(words[begin:end])
        start_page, end_page = _page_range(pages)
        ordinal = len(chunks) + 1
        chunks.append(
            Chunk(
                chunk_id=f"{source.id}:{ordinal:04d}",
                doc_id=source.id,
                kind=TEXT,
                ordinal=ordinal,
                text=text,
                embed_text=_URL.sub("[link]", text),  # no document or section prefix: the baseline sees raw text only
                section_path=clean_path(source, paths[first]),
                page_start=start_page,
                page_end=end_page,
                n_words=end - begin,
            )
        )
        if end == len(words):
            break
        begin += step
    return chunks


def chunk_document(source: Source, blocks: list[Block], options: ChunkOptions | None = None) -> list[Chunk]:
    """Chunk one document's blocks with the chosen strategy."""
    options = options or ChunkOptions()
    if not blocks:
        return []
    if options.strategy == "fixed":
        return _chunk_fixed(source, blocks, options)
    return _chunk_structure(source, blocks, options)
