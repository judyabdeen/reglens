"""Command line:  python scripts/build_index.py  -  embed every chunk and fill the vector store(s)."""

from __future__ import annotations

import argparse
import platform
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from reglens.config import ConfigError, Settings, load_settings
from reglens.console import make_console_safe, short_duration
from reglens.embeddings import Embedder, EmbeddingError, is_hashing, make_embedder
from reglens.indexing import IndexInfo, IndexingError, build_index, file_sha256, info_path, validate_chunks
from reglens.ingest.chunking import read_chunks
from reglens.corpus.manifest import Source
from reglens.runtime import load_sources
from reglens.vectorstore import VectorStoreError, open_store

BACKEND_CHOICES = ("chroma", "numpy", "both")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Embed the chunks of data/processed/chunks.jsonl and store them in the vector database.",
    )
    parser.add_argument("--chunks", help="chunks file to index (default: CHUNKS_PATH, data/processed/chunks.jsonl)")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, help="vector store to fill (default: VECTORSTORE from .env; 'both' fills both)")
    parser.add_argument("--index-dir", help="folder for the index (default: VECTORSTORE_DIR, normally data/index)")
    parser.add_argument("--embedding-model", help="embedding model (default: EMBEDDING_MODEL from .env); 'hashing' needs no download")
    parser.add_argument("--batch-size", type=int, default=32, help="texts per batch inside the embedding model (default 32)")
    parser.add_argument("--fresh", action="store_true", help="delete the existing index first (use it if a build failed or the index is damaged)")
    parser.add_argument("--env-file", type=Path, help="use this .env file")
    return parser


def _progress_printer(write: Callable[[str], None], clock: Callable[[], float]) -> Callable[[str, int, int], None]:
    started = clock()
    state = {"stage": "", "step": -1}

    def report(stage: str, done: int, total: int) -> None:
        if stage != state["stage"]:
            state.update(stage=stage, step=-1)
        step = done * 10 // max(total, 1)  # a line at every 10 %
        if step != state["step"] or done == total:
            state["step"] = step
            write(f"  {stage:<9} {done:>5,} / {total:,}  ({done * 100 // max(total, 1)}%)  {short_duration(clock() - started)}")

    return report


def _describe(info: IndexInfo, where: Path) -> list[str]:
    lines = [
        f"  stored    {info.n_records:,} passages from {info.n_documents} documents ({info.n_text:,} text, {info.n_notes:,} footnote)",
        f"  vectors   {info.dim} numbers each, {info.embedding_model}, {info.space} similarity",
        f"  time      embedding {short_duration(info.seconds.get('embedding', 0))}, storing {short_duration(info.seconds.get('storing', 0))}",
        f"  location  {where}",
    ]
    if info.over_token_limit is not None and info.token_limit:
        share = 100 * info.over_token_limit / max(info.n_records, 1)
        lines.append(
            f"  length    {info.over_token_limit:,} passages ({share:.1f}%) are longer than the model's {info.token_limit}-token window "
            f"(longest {info.longest_tokens:,} tokens): the end of those passages is not part of their vector"
        )
    elif info.token_limit:
        lines.append(f"  length    the model reads {info.token_limit} tokens per passage (the overflow could not be measured)")
    return lines


def main(
    argv: Sequence[str] | None = None,
    *,
    make_embedder_fn: Callable[..., Embedder] = make_embedder,
    settings_loader: Callable[..., Settings] = load_settings,
    sources_loader: Callable[[Settings], dict[str, Source]] = load_sources,
    write: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    make_console_safe()
    args = _parser().parse_args(argv)
    try:
        settings = settings_loader(env_file=args.env_file)
    except ConfigError as exc:
        write(f"Problem: {exc}")
        return 2
    chunks_path = Path(args.chunks) if args.chunks else settings.chunks_path
    index_dir = Path(args.index_dir) if args.index_dir else settings.vectorstore_dir
    model_name = args.embedding_model or settings.embedding_model
    wanted = args.backend or settings.vectorstore_backend
    backends = ["chroma", "numpy"] if wanted == "both" else [wanted]
    if args.batch_size < 1:
        write("Problem: --batch-size must be at least 1")
        return 2
    if not chunks_path.is_file():
        write(f"Problem: {chunks_path} does not exist. Create the chunks first: python scripts/ingest.py")
        return 1

    write("RegLens index builder")
    write(f"Python    {platform.python_version()} on {platform.system()} {platform.release()}")
    write(f"Chunks    {chunks_path}")
    write(f"Index     {index_dir}  ({' and '.join(backends)})")
    if settings.vectorstore_dir_moved_from and not args.index_dir:
        write(f"          (the project is inside {settings.vectorstore_dir_moved_from}, so the index is kept outside it)")
    write(f"Embedding {model_name}")
    write("")
    try:
        chunks = read_chunks(chunks_path)
        sources = sources_loader(settings)
        if not sources:
            write(f"Problem: the manifest {settings.manifest_path} could not be read, so passages could not be cited. Restore data/sources.csv.")
            return 1
        validate_chunks(chunks, sources)  # before the slow part, so a problem shows up at once
        fingerprint = file_sha256(chunks_path)
        if args.fresh:
            for backend in backends:
                info_path(index_dir, backend).unlink(missing_ok=True)
                if (index_dir / backend).exists():
                    shutil.rmtree(index_dir / backend)
        stores = {backend: open_store(index_dir, backend) for backend in backends}  # also fails early if chromadb is missing
        embedder = make_embedder_fn(model_name, batch_size=args.batch_size)
        report = _progress_printer(write, clock)
        if is_hashing(model_name):
            write(f"Embedding {len(chunks):,} passages (the baseline model needs no download) ...")
        else:
            write(f"Embedding {len(chunks):,} passages (the first run downloads the model, about 130 MB for the default; this can take several minutes on a laptop) ...")
        mark = clock()
        vectors = embedder.embed_documents([chunk.embed_text for chunk in chunks], lambda done, total: report("embedding", done, total))
        embedding_seconds = clock() - mark
        for backend, store in stores.items():
            info = build_index(
                chunks, sources, embedder, store, info_file=info_path(index_dir, backend), progress=report, backend=backend,
                chunks_file=chunks_path, chunks_sha256=fingerprint, vectors=vectors, embedding_seconds=embedding_seconds,
            )  # fmt: skip
            write("")
            write(f"Built the {backend} index:")
            for line in _describe(info, index_dir / backend):
                write(line)
            write("")
    except (IndexingError, EmbeddingError, VectorStoreError, ValueError, OSError) as exc:
        write(f"\nProblem: {exc}")
        return 1
    write("Next: python scripts/ask.py \"How quickly must a personal data breach be reported?\"")
    return 0
