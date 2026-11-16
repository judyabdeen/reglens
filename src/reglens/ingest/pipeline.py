"""Run the ingestion for one document or for the whole corpus: read, clean, structure, audit, chunk, save.

Layout of the output directory (``data/processed`` by default, git-ignored):

``blocks/<id>.json``
    The cleaned document as a list of headings, paragraphs and footnotes, plus the audit figures. This is the
    expensive part (reading a PDF takes seconds), so it is cached: it is redone only when the source file or
    the extraction code changes.
``chunks.jsonl``
    One chunk per line - what the embedding step reads. Rewritten on every run, because it depends on the
    chunking options.
``ingest_report.json``
    One entry per document with counts and audit flags, plus the options that were used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from reglens.corpus.manifest import Source
from reglens.ingest import layout, models, pdfread, structure, textclean, webpage
from reglens.ingest.chunking import NOTES, TEXT, Chunk, ChunkOptions, chunk_document
from reglens.ingest.models import FOOTNOTE, HEADING, PARAGRAPH, Block

_FINGERPRINTED_MODULES = (textclean, layout, structure, webpage, pdfread, models)
_STOPWORDS = frozenset(
    "the of and to in a is that for shall be or by with as on this are from an which not it at its must may".split()
)
_GENERIC_TITLE_WORDS = frozenset(
    "guidelines guidance regulation commission recommendation council framework version summary statement "
    "draft with under that this from the for and".split()
)

LOW_CHARS_PER_PAGE = 600
MAX_EMPTY_PAGE_SHARE = 0.15
SERIOUS_EMPTY_PAGE_SHARE = 0.25  # blank pages are normal in print layouts, but not when this many pages are empty ...
SERIOUS_EMPTY_RUN = 4  # ... or when this many follow each other (a scanned part of the document)
MIN_HEADINGS = 3
MAX_HEADINGS_PER_100_WORDS = 1.0
LONG_PARAGRAPH_WORDS = 400
LONG_PARAGRAPH_SHARE = 0.30
MIN_ENGLISH_RATIO = 0.18  # prose has 35-40 % common words, telegraphic tables about 20 %, other languages and garbage under 10 %
MIN_TOKENS_PER_PAGE = 80
MAX_RUN_TOGETHER_RATIO = 0.003
MIN_TITLE_MATCH = 0.35
FOOTNOTE_HEAVY_SHARE = 0.6


@dataclass
class Flag:
    code: str
    level: str  # "warn" needs a look, "info" is context
    message: str


@dataclass
class DocResult:
    id: str
    status: str  # ok | failed | missing
    detail: str = ""
    format: str = ""
    cached: bool = False
    seconds: float = 0.0
    pages: int | None = None
    words: int = 0
    headings: int = 0
    paragraphs: int = 0
    footnotes: int = 0
    chunks: int = 0
    note_chunks: int = 0
    chunk_words_median: int = 0
    flags: list[Flag] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def warnings(self) -> list[Flag]:
        return [flag for flag in self.flags if flag.level == "warn"]


# --------------------------------------------------------------------------------------- caching


def extraction_fingerprint() -> str:
    """A short hash of the extraction code: when it changes, cached blocks are rebuilt."""
    digest = hashlib.sha256()
    for module in _FINGERPRINTED_MODULES:
        try:
            digest.update(Path(module.__file__).read_bytes())
        except (OSError, TypeError):
            digest.update(module.__name__.encode())
    return digest.hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _jsonable(value):
    if isinstance(value, Counter):
        return {str(k): v for k, v in value.most_common()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ----------------------------------------------------------------------------------------- audit


def _title_match(title: str, text: str) -> float:
    words = {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) >= 4 and w not in _GENERIC_TITLE_WORDS}
    if not words:
        return 1.0
    haystack = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(words & haystack) / len(words)


def _english_ratio(body: list[Block]) -> float:
    """Share of common English words on a typical page. The median over pages, so that a bibliography or a table of
    identifiers (which has few such words) does not make a document written in English look like another language."""
    by_page: dict = {}
    for block in body:
        by_page.setdefault(block.page, []).append(block.text)
    ratios = []
    for texts in by_page.values():
        tokens = re.findall(r"[a-z']+", " ".join(texts).lower())
        if len(tokens) >= MIN_TOKENS_PER_PAGE:
            ratios.append(sum(t in _STOPWORDS for t in tokens) / len(tokens))
    if ratios:
        return statistics.median(ratios)
    tokens = re.findall(r"[a-z']+", " ".join(b.text for b in body).lower())
    return sum(t in _STOPWORDS for t in tokens) / max(1, len(tokens))


def _longest_run(numbers: Sequence[int]) -> int:
    """The length of the longest stretch of consecutive numbers in a sorted list."""
    best = run = 0
    previous = None
    for number in numbers:
        run = run + 1 if previous is not None and number == previous + 1 else 1
        best = max(best, run)
        previous = number
    return best


def _page_list(numbers: Sequence[int], limit: int = 12) -> str:
    shown = ", ".join(str(number) for number in numbers[:limit])
    return shown + (", ..." if len(numbers) > limit else "")


def _empty_pages_flag(info: dict, pages: int) -> Flag | None:
    numbers = sorted(info.get("empty_page_numbers") or [])
    count = len(numbers) if numbers else info.get("empty_pages", 0)
    long_run = _longest_run(numbers) >= SERIOUS_EMPTY_RUN
    if not pages or (count <= MAX_EMPTY_PAGE_SHARE * pages and not long_run):
        return None
    serious = not numbers or long_run or count > SERIOUS_EMPTY_PAGE_SHARE * pages
    where = f" (pages {_page_list(numbers)})" if numbers else ""
    if serious:
        return Flag("EMPTY_PAGES", "warn", f"{count} of {pages} pages have almost no text{where} - scanned pages without a text layer?")
    return Flag("EMPTY_PAGES", "info", f"{count} of {pages} pages have almost no text{where} - blank and divider pages are normal in printed documents")


def _garbled_message(info: dict, total: int) -> str:
    lost = info.get("dropped_codes") or {}
    detail = f" ({', '.join(f'{code} x{count}' for code, count in lost.items())})" if lost else ""
    return f"{total} unreadable character codes were removed{detail}"


def audit_blocks(source: Source, blocks: list[Block], info: dict) -> list[Flag]:
    """Checks that tell whether the extracted text can be trusted. Warnings are what to look at first."""
    flags: list[Flag] = []
    body = [b for b in blocks if b.kind in (HEADING, PARAGRAPH)]
    notes = [b for b in blocks if b.kind == FOOTNOTE]
    words = sum(len(b.text.split()) for b in body)
    note_words = sum(len(b.text.split()) for b in notes)
    headings = sum(1 for b in blocks if b.kind == HEADING)
    pages = info.get("pages")

    if words == 0:
        return [Flag("NO_TEXT", "warn", "no text could be extracted (scanned images? an empty page?)")]
    if pages and info.get("chars_total", 0) / pages < LOW_CHARS_PER_PAGE:
        flags.append(Flag("LOW_TEXT", "warn", f"only {info['chars_total'] // pages} characters per page - mostly images or a scan?"))
    empty_flag = _empty_pages_flag(info, pages or 0)
    if empty_flag:
        flags.append(empty_flag)
    if info.get("page_errors"):
        flags.append(Flag("PAGE_ERRORS", "warn", f"{len(info['page_errors'])} page(s) could not be read: {', '.join(info['page_errors'][:3])}"))
    if (pages or 0) >= 5 or words >= 1500:
        if headings < MIN_HEADINGS:
            flags.append(Flag("NO_HEADINGS", "warn", f"only {headings} heading(s) found - chunks will not follow sections"))
    structural = sum(1 for b in blocks if b.kind == HEADING and layout.STRUCTURAL_LINE.match(b.text))
    loose = headings - structural  # "Article 12" and "Chapter IV" are expected in short legal texts; other headings are not
    if words and 100.0 * loose / words > MAX_HEADINGS_PER_100_WORDS:
        flags.append(Flag("MANY_HEADINGS", "warn", f"{headings} headings in {words} words - some lines may be mistaken for headings"))
    paragraph_words = [len(b.text.split()) for b in blocks if b.kind == PARAGRAPH]
    if paragraph_words:
        long_share = sum(w for w in paragraph_words if w > LONG_PARAGRAPH_WORDS) / max(1, sum(paragraph_words))
        if long_share > LONG_PARAGRAPH_SHARE:
            flags.append(Flag("LONG_PARAGRAPHS", "warn", f"{long_share:.0%} of the text is in paragraphs over {LONG_PARAGRAPH_WORDS} words - paragraph breaks were not found"))

    text = " ".join(b.text for b in body)
    tokens = re.findall(r"[a-z']+", text.lower())
    english = _english_ratio(body)
    if english < MIN_ENGLISH_RATIO:
        flags.append(Flag("NOT_ENGLISH", "warn", f"common English words make up only {english:.0%} of a typical page - wrong language or garbled?"))
    run_together = len(re.findall(r"[A-Za-z]{25,}", text)) / max(1, len(tokens))
    if run_together > MAX_RUN_TOGETHER_RATIO:
        flags.append(Flag("RUN_TOGETHER_WORDS", "warn", f"{run_together:.1%} of words are over 24 letters - spaces may be missing between words"))
    garbled = sum(info.get(key, 0) for key in ("cid_refs", "private_use_dropped", "replacement_chars"))
    if garbled > 20:
        flags.append(Flag("GARBLED_CHARS", "warn", _garbled_message(info, garbled)))
    head = " ".join(b.text for b in body[:60])[:6000]
    match = _title_match(source.title, head)
    if match < MIN_TITLE_MATCH:
        flags.append(Flag("TITLE_MISMATCH", "warn", f"only {match:.0%} of the title's key words appear at the start - is this the right document?"))
    if note_words > FOOTNOTE_HEAVY_SHARE * words:
        flags.append(Flag("NOTE_HEAVY", "info", f"footnotes are {note_words * 100 // max(1, words + note_words)}% of the text"))
    typeset, total_pages = info.get("typeset_pages"), info.get("pages")
    if typeset is not None and total_pages and typeset < 0.5 * total_pages:
        flags.append(Flag("NO_TYPOGRAPHY", "info", f"font sizes could be read for only {typeset} of {total_pages} pages: footnotes stay in the body text and table cells may be mistaken for headings"))
    if info.get("outline_used"):
        flags.append(Flag("OUTLINE", "info", f"headings taken from the PDF bookmarks ({info.get('outline_matches')} matched)"))
    if info.get("rule_level", 2) < 2:
        flags.append(Flag("WEAK_RULES_OFF", "info", "Title Case / letter headings were switched off because they produced too many headings"))
    return flags


# ----------------------------------------------------------------------------------- extraction


def extract_document(source: Source, path: Path) -> tuple[list[Block], dict]:
    """Read one file and return its blocks and the figures gathered on the way (JSON-friendly)."""
    if source.format == "pdf":
        pdf = pdfread.read_pdf(path)
        pages, layout_report = layout.prepare_pages(pdf.pages, pdf.styles)
        blocks, structure_report = structure.build_blocks(pages, pdf.outline)
        info = {
            "pages": layout_report.pages,
            "chars_total": sum(len(p) for p in pdf.pages),
            "empty_pages": layout_report.empty_pages,
            "empty_page_numbers": layout_report.empty_page_numbers,
            "page_errors": pdf.page_errors,
            "margin_lines_removed": {
                layout_report.margin_examples.get(key, key): count
                for key, count in layout_report.margin_lines_removed.most_common(8)
            },
            "page_numbers_removed": layout_report.page_numbers_removed,
            "toc_lines_removed": layout_report.toc_lines_removed,
            "footnote_pages": layout_report.footnote_pages,
            "body_font_size": layout_report.body_size,
            "typeset_pages": sum(1 for page_styles in pdf.styles if page_styles),
            "outline_entries": len(pdf.outline),
            "outline_matches": structure_report.outline_matches,
            "outline_used": structure_report.outline_used,
            "rule_level": structure_report.rule_level,
            "heading_schemes": dict(structure_report.schemes),
            "metadata_title": pdf.metadata_title,
            **dict(layout_report.chars),
        }
        return blocks, _tidy_info(info)
    page = webpage.extract_web_page(path)
    info = {"root": page.root, "page_title": page.title, **dict(page.chars)}
    return page.blocks, _tidy_info(info)


def _tidy_info(info: dict) -> dict:
    """Fold the per-glyph counters ("U+F0B7": 12) into one ``dropped_codes`` entry with the most frequent glyphs."""
    glyphs = {key: info.pop(key) for key in [key for key in info if key.startswith("U+")]}
    if glyphs:
        info["dropped_codes"] = dict(Counter(glyphs).most_common(6))
    return info


def _cache_path(out_dir: Path, source: Source) -> Path:
    return out_dir / "blocks" / f"{source.id}.json"


def _load_cached(path: Path, sha: str, fingerprint: str) -> tuple[list[Block], dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("source_sha256") != sha or data.get("fingerprint") != fingerprint:
        return None
    try:
        return [Block.from_dict(b) for b in data["blocks"]], data["info"]
    except (KeyError, TypeError):
        return None


def load_blocks(out_dir: Path, source_id: str) -> list[Block]:
    """Read the saved blocks of one document (used by the ``--show`` mode and by tests)."""
    data = json.loads((out_dir / "blocks" / f"{source_id}.json").read_text(encoding="utf-8"))
    return [Block.from_dict(b) for b in data["blocks"]]


# ------------------------------------------------------------------------------------ one document


def ingest_source(
    source: Source,
    raw_dir: Path,
    out_dir: Path,
    options: ChunkOptions,
    *,
    force: bool = False,
    fingerprint: str | None = None,
) -> tuple[DocResult, list[Chunk]]:
    """Ingest one document. Never raises for problems with the document itself: they become the result's status."""
    started = time.monotonic()
    result = DocResult(source.id, "ok", format=source.format)
    path = raw_dir / source.filename
    if not path.is_file():
        result.status, result.detail = "missing", f"{source.filename} is not in the raw folder - run: python scripts/download_corpus.py"
        return result, []
    fingerprint = fingerprint or extraction_fingerprint()
    try:
        sha = _sha256_file(path)
        cached = None if force else _load_cached(_cache_path(out_dir, source), sha, fingerprint)
        if cached is not None:
            blocks, info = cached
            result.cached = True
        else:
            blocks, info = extract_document(source, path)
            payload = {
                "id": source.id,
                "fingerprint": fingerprint,
                "source_sha256": sha,
                "format": source.format,
                "info": _jsonable(info),
                "blocks": [b.to_dict() for b in blocks],
            }
            _write_text_atomic(_cache_path(out_dir, source), json.dumps(payload, ensure_ascii=False))
    except pdfread.ExtractionError as exc:
        result.status, result.detail = "failed", str(exc)
        return result, []
    except OSError as exc:
        result.status, result.detail = "failed", f"file cannot be read ({type(exc).__name__}): {exc}"
        return result, []
    except Exception as exc:  # a surprising document must not stop the other 33
        result.status, result.detail = "failed", f"unexpected {type(exc).__name__}: {str(exc)[:150]}"
        return result, []

    result.info = info
    result.pages = info.get("pages")
    result.flags = audit_blocks(source, blocks, info)
    result.words = sum(len(b.text.split()) for b in blocks if b.kind in (HEADING, PARAGRAPH))
    result.headings = sum(1 for b in blocks if b.kind == HEADING)
    result.paragraphs = sum(1 for b in blocks if b.kind == PARAGRAPH)
    result.footnotes = sum(1 for b in blocks if b.kind == FOOTNOTE)
    chunks = chunk_document(source, blocks, options)
    result.chunks = sum(1 for c in chunks if c.kind == TEXT)
    result.note_chunks = sum(1 for c in chunks if c.kind == NOTES)
    sizes = [c.n_words for c in chunks if c.kind == TEXT]
    result.chunk_words_median = int(statistics.median(sizes)) if sizes else 0
    if not chunks:
        result.status, result.detail = "failed", "no text could be extracted"
    result.seconds = time.monotonic() - started
    return result, chunks


# ------------------------------------------------------------------------------------ the corpus


def _short(result: DocResult) -> str:
    if not result.ok:
        return result.status.upper()
    return "OK" + (" (cached)" if result.cached else "")


def run_ingest(
    sources: Sequence[Source],
    raw_dir: Path,
    out_dir: Path,
    options: ChunkOptions,
    *,
    force: bool = False,
    out: Callable[[str], None] = print,
    write_combined: bool = True,
    tag: str = "",
) -> tuple[list[DocResult], list[Chunk]]:
    """Ingest every source and print one line per document.

    With ``write_combined`` the chunks of all documents are written to ``chunks[-tag].jsonl`` and the audit to
    ``ingest_report[-tag].json``. A run over only some of the documents must not overwrite those (they would
    lose the other documents), so the caller turns it off then.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = extraction_fingerprint()
    results: list[DocResult] = []
    all_chunks: list[Chunk] = []
    total = len(sources)
    width = max((len(s.id) for s in sources), default=10)
    for number, source in enumerate(sources, start=1):
        out(f"[{number:>2}/{total}] {source.id:<{width}}  reading {source.format.upper()} ...")
        result, chunks = ingest_source(source, raw_dir, out_dir, options, force=force, fingerprint=fingerprint)
        results.append(result)
        all_chunks.extend(chunks)
        if result.ok:
            pages = f"{result.pages} pages, " if result.pages else ""
            out(
                f"        {_short(result)}: {pages}{result.words:,} words, {result.headings} headings, "
                f"{result.chunks} chunks (median {result.chunk_words_median} words) + {result.note_chunks} footnote chunks "
                f"[{result.seconds:.1f}s]"
            )
        else:
            out(f"        {_short(result)}: {result.detail}")
        for flag in result.flags:
            if flag.level == "warn":
                out(f"        WARNING {flag.code}: {flag.message}")

    if not write_combined:
        return results, all_chunks
    suffix = f"-{tag}" if tag else ""
    lines = (json.dumps(c.to_dict(), ensure_ascii=False) for c in all_chunks)
    _write_text_atomic(out_dir / f"chunks{suffix}.jsonl", "\n".join(lines) + ("\n" if all_chunks else ""))
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fingerprint": fingerprint,
        "options": asdict(options),
        "totals": {
            "documents": total,
            "ok": sum(r.ok for r in results),
            "failed": sum(r.status == "failed" for r in results),
            "missing": sum(r.status == "missing" for r in results),
            "chunks": sum(1 for c in all_chunks if c.kind == TEXT),
            "note_chunks": sum(1 for c in all_chunks if c.kind == NOTES),
            "words": sum(r.words for r in results),
        },
        "documents": [_jsonable(asdict(r)) for r in results],
    }
    _write_text_atomic(out_dir / f"ingest_report{suffix}.json", json.dumps(report, indent=2, ensure_ascii=False))
    return results, all_chunks


# -------------------------------------------------------------------------------------- reporting


def outline_text(blocks: Sequence[Block]) -> str:
    """The headings of a document, indented by level, with the page they start on."""
    lines = []
    for block in blocks:
        if block.kind == HEADING:
            where = f"p{block.page}" if block.page else "-"
            lines.append(f"{'  ' * (block.level - 1)}{block.text}  [{where}]")
    return "\n".join(lines)


def print_summary(results: Sequence[DocResult], chunks: Sequence[Chunk], out: Callable[[str], None] = print) -> None:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    text = [c for c in chunks if c.kind == TEXT]
    sizes = [c.n_words for c in text]
    out("")
    out(f"Ingested {len(ok)} of {len(results)} documents: {sum(r.words for r in ok):,} words, {len(text)} text chunks, "
        f"{len(chunks) - len(text)} footnote chunks.")
    if sizes:
        out(f"Text chunk size in words: median {int(statistics.median(sizes))}, smallest {min(sizes)}, largest {max(sizes)}.")
    flagged = [r for r in ok if r.warnings]
    if flagged:
        out("")
        out(f"{len(flagged)} document(s) have warnings (details above). Warnings are hints to look at, not errors.")
    if bad:
        out("")
        out("Not ingested:")
        for r in bad:
            out(f"  - {r.id}: {r.detail}")


def export_for_review(out_dir: Path, zip_path: Path, results: Sequence[DocResult]) -> int:
    """Pack the cleaned documents and the audit into one zip that can be attached to a chat. Returns its size."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            "RegLens processed text (no raw documents, no API keys).\n"
            "ingest_report.json - counts and audit flags per document\n"
            "outlines.txt       - the headings found in every document\n"
            "blocks/<id>.json   - the cleaned text of every document as headings, paragraphs and footnotes\n",
        )
        report = out_dir / "ingest_report.json"
        if report.is_file():
            archive.write(report, "ingest_report.json")
        outlines = []
        for result in results:
            blocks_file = out_dir / "blocks" / f"{result.id}.json"
            if result.ok and blocks_file.is_file():
                archive.write(blocks_file, f"blocks/{result.id}.json")
                blocks = load_blocks(out_dir, result.id)
                outlines.append(f"== {result.id} ({result.format}, {result.pages or '-'} pages, {result.headings} headings)\n{outline_text(blocks)}\n")
        archive.writestr("outlines.txt", "\n".join(outlines))
    return zip_path.stat().st_size
