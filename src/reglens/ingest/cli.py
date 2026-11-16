"""Command line for the ingestion step:  python scripts/ingest.py [options]"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from reglens.config import PROJECT_ROOT
from reglens.corpus.download import select_sources
from reglens.corpus.manifest import ManifestError, load_manifest
from reglens.ingest.chunking import STRATEGIES, ChunkOptions
from reglens.ingest.pipeline import (
    export_for_review,
    load_blocks,
    outline_text,
    print_summary,
    run_ingest,
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "processed"
DEFAULT_EXPORT = "reglens-processed-text.zip"


def _parser() -> argparse.ArgumentParser:
    defaults = ChunkOptions()
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Read the downloaded documents, clean them, split them into chunks and write data/processed/.",
    )
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "sources.csv"), help="path to sources.csv")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="where the downloaded files are")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="where the processed files are written")
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these source ids")
    parser.add_argument("--core-only", action="store_true", help="skip optional (priority 2) documents")
    parser.add_argument("--force", action="store_true", help="read the files again even if the saved text is current")
    parser.add_argument("--strategy", choices=STRATEGIES, default=defaults.strategy, help="chunking strategy")
    parser.add_argument("--target-words", type=int, default=defaults.target_words, help="chunk size to aim for")
    parser.add_argument("--max-words", type=int, default=defaults.max_words, help="largest chunk size")
    parser.add_argument("--min-words", type=int, default=defaults.min_words, help="smaller sections are merged")
    parser.add_argument("--overlap-words", type=int, default=defaults.overlap_words, help="overlap when a section is cut")
    parser.add_argument(
        "--tag",
        default=None,
        help="name the output files chunks-TAG.jsonl and ingest_report-TAG.json (default: none, or 'fixed' for the fixed strategy)",
    )
    parser.add_argument("--show", metavar="ID", help="print the headings and the first chunks of one document")
    parser.add_argument(
        "--export",
        nargs="?",
        const=DEFAULT_EXPORT,
        metavar="ZIP",
        help=f"also pack the cleaned text and the audit into a zip (default name {DEFAULT_EXPORT} inside the output folder)",
    )
    return parser


def _make_console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def _show(out_dir: Path, source_id: str, chunks) -> None:
    print(f"\n----- {source_id}: headings -----")
    print(outline_text(load_blocks(out_dir, source_id)) or "(no headings found)")
    mine = [c for c in chunks if c.doc_id == source_id][:3]
    print(f"\n----- {source_id}: first {len(mine)} chunks -----")
    for chunk in mine:
        pages = f"p{chunk.page_start}-{chunk.page_end}" if chunk.page_start else "web page"
        print(f"\n[{chunk.chunk_id}] {chunk.kind}, {chunk.n_words} words, {pages}")
        print(f"  section: {' > '.join(chunk.section_path) or '(top of the document)'}")
        print("  " + chunk.text[:700].replace("\n", "\n  "))


def _pypdf_version() -> str:
    try:
        from importlib.metadata import version

        return version("pypdf")
    except Exception:  # a version line must never stop the run
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    _make_console_safe()
    args = _parser().parse_args(argv)
    try:
        options = ChunkOptions(
            strategy=args.strategy,
            target_words=args.target_words,
            max_words=args.max_words,
            min_words=args.min_words,
            overlap_words=args.overlap_words,
        )
        sources = load_manifest(Path(args.manifest))
        only = list(args.only or [])
        if args.show and not only:
            only = [args.show]
        selected = select_sources(sources, only or None, args.core_only)
        if args.show and args.show not in {s.id for s in selected}:
            raise ValueError(f"--show {args.show}: that document is not part of this run")
    except (ManifestError, ValueError) as exc:
        print(f"Problem: {exc}")
        return 2
    if not selected:
        print("Nothing selected.")
        return 2

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    print("RegLens ingestion")
    print(f"Python    {sys.version.split()[0]}, pypdf {_pypdf_version()}")
    print(f"Reading   {raw_dir}")
    print(f"Writing   {out_dir}")
    print(
        f"Chunking  strategy={options.strategy}, target {options.target_words} / max {options.max_words} / "
        f"min {options.min_words} words, overlap {options.overlap_words}"
    )
    print("")
    tag = args.tag if args.tag is not None else ("" if options.strategy == "structure" else options.strategy)
    whole_corpus = len(selected) == len(sources)
    results, chunks = run_ingest(
        selected, raw_dir, out_dir, options, force=args.force, write_combined=whole_corpus, tag=tag
    )
    print_summary(results, chunks)
    if args.show:
        _show(out_dir, args.show, chunks)
    if args.export:
        target = Path(args.export)
        if not target.is_absolute() and target.parent == Path("."):
            target = out_dir / target
        size = export_for_review(out_dir, target, results)
        print(f"\nPacked the cleaned text for review: {target}  ({size / (1024 * 1024):.1f} MB)")
    if whole_corpus:
        print(f"\nChunks written to {out_dir / ('chunks-' + tag + '.jsonl' if tag else 'chunks.jsonl')}")
    else:
        print("\nPartial run: chunks.jsonl and ingest_report.json were not changed (run without --only to rebuild them).")
    return 0 if all(r.ok for r in results) else 1
