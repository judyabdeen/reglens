"""Build and open the vector index (assignment Steps 3-4: embed every chunk, store it, check that it works).

``build_index`` embeds the text of every chunk, stores vectors + text + metadata in a vector store and verifies the
result. It also writes a small ``<backend>.info.json`` file next to the store that records *how* the index was made
(embedding model, vector size, a fingerprint of ``chunks.jsonl``, library versions, timings). The file is deleted when a
build starts and written when it ends, so a build that was interrupted is recognised and never used by mistake.

``open_index`` is the other half: it refuses an index that is missing, unfinished, damaged, or built with a different
embedding model than the one that would now embed the questions (vectors of two different models cannot be compared).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata as metadata
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reglens.corpus.manifest import Source
from reglens.embeddings.base import Embedder
from reglens.ingest.chunking import NOTES, TEXT, Chunk
from reglens.vectorstore import VectorStore, VectorStoreError, open_store
from reglens.vectorstore.records import make_records

INFO_FORMAT = 1
PROBES = 5  # chunks used to check that the finished index returns what it was given
PROBE_HITS = 5
LIBRARIES = ("numpy", "chromadb", "sentence-transformers", "torch", "transformers")

Progress = Callable[[str, int, int], None]  # (stage, done, total)


class IndexingError(RuntimeError):
    """The chunks cannot be indexed, or the finished index failed its check."""


class IndexNotReady(RuntimeError):
    """There is no usable index. The message says what to do about it."""


@dataclass(frozen=True)
class IndexInfo:
    format: int
    embedding_model: str
    dim: int
    backend: str
    space: str
    n_records: int
    n_documents: int
    n_text: int
    n_notes: int
    chunks_file: str
    chunks_sha256: str
    built_at: str
    seconds: Mapping[str, float] = field(default_factory=dict)
    libraries: Mapping[str, str] = field(default_factory=dict)
    token_limit: int | None = None  # how many tokens of a chunk the embedding model reads
    over_token_limit: int | None = None  # chunks longer than that (their end is not embedded); None: not measured
    longest_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IndexInfo:
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})  # type: ignore[arg-type]


@dataclass(frozen=True)
class OpenedIndex:
    store: VectorStore
    info: IndexInfo
    warnings: tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------ small helpers


def info_path(directory: Path | str, backend: str) -> Path:
    return Path(directory) / f"{backend}.info.json"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in LIBRARIES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions[name] = metadata.version(name)
    return versions


def write_info(path: Path, info: IndexInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(info.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_info(path: Path) -> IndexInfo:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return IndexInfo.from_dict(data)
    except (OSError, ValueError, TypeError) as exc:
        raise IndexNotReady(f"{path.name} cannot be read ({type(exc).__name__}); rebuild the index: python scripts/build_index.py") from exc


def validate_chunks(chunks: Sequence[Chunk], sources: Mapping[str, Source]) -> None:
    """Raise ``IndexingError`` for chunks that cannot be indexed (cheap, so callers can run it before embedding)."""
    if not chunks:
        raise IndexingError("there are no chunks to index - run: python scripts/ingest.py")
    unknown = sorted({chunk.doc_id for chunk in chunks} - set(sources))
    if unknown:
        raise IndexingError(
            f"{len(unknown)} document(s) of the chunks are not in the manifest data/sources.csv ({', '.join(unknown[:5])}); "
            "their passages could not be cited with title, status and link"
        )
    seen: set[str] = set()
    for chunk in chunks:
        if not chunk.chunk_id:
            raise IndexingError("a chunk has no id")
        if chunk.chunk_id in seen:
            raise IndexingError(f"two chunks have the id {chunk.chunk_id!r}; ids must be unique")
        seen.add(chunk.chunk_id)
        if not chunk.text.strip() or not chunk.embed_text.strip():
            raise IndexingError(f"chunk {chunk.chunk_id!r} has no text")


def _check_vectors(vectors: np.ndarray, count: int, dim: int, name: str) -> None:
    if vectors.shape != (count, dim):
        raise IndexingError(f"{name} returned vectors of shape {vectors.shape}, expected {(count, dim)}")
    if not np.isfinite(vectors).all():
        raise IndexingError(f"{name} returned NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise IndexingError(f"{name} did not return unit-length vectors (norms between {norms.min():.3f} and {norms.max():.3f})")


def _token_report(embedder: Embedder, texts: Sequence[str]) -> tuple[int | None, int | None, int | None]:
    """(limit, chunks over the limit, longest chunk) in tokens. Best effort: a report must never stop a build."""
    try:
        limit = getattr(embedder, "max_tokens", None)
        counter = getattr(embedder, "token_counts", None)
        counts = counter(texts) if callable(counter) else None
    except Exception:
        return None, None, None
    if not counts:
        return (int(limit) if limit else None), None, None
    over = sum(1 for count in counts if limit and count > limit)
    return (int(limit) if limit else None), (over if limit else None), max(counts)


def _probe(store: VectorStore, vectors: np.ndarray, ids: Sequence[str]) -> None:
    """Ask the finished store for a few of its own vectors: each chunk must come back among the best hits."""
    count = len(ids)
    for position in sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})[:PROBES]:
        hits = store.query(vectors[position], PROBE_HITS)
        if ids[position] not in [hit.id for hit in hits]:
            raise IndexingError(
                f"the finished index does not return chunk {ids[position]!r} when asked for its own vector; "
                "the store is not working correctly"
            )


# ------------------------------------------------------------------------------------------------ building


def build_index(
    chunks: Sequence[Chunk],
    sources: Mapping[str, Source],
    embedder: Embedder,
    store: VectorStore,
    *,
    info_file: Path | None = None,
    progress: Progress | None = None,
    reset: bool = True,
    backend: str = "",
    chunks_file: Path | str | None = None,
    chunks_sha256: str | None = None,
    vectors: np.ndarray | None = None,
    embedding_seconds: float | None = None,
) -> IndexInfo:
    """Embed ``chunks`` and store them in ``store``. Returns (and, with ``info_file``, writes) what was built.

    ``vectors`` (one row per chunk, in order) skips the embedding step: it lets the same chunks go into two stores
    without embedding them twice. ``embedding_seconds`` is how long computing them took, for the record.
    """
    validate_chunks(chunks, sources)
    started = time.perf_counter()
    if info_file is not None:
        info_file.unlink(missing_ok=True)  # until the build ends, nobody may use this index
    if reset:
        store.reset()
    elif store.count():
        raise IndexingError("the store already contains vectors; build with reset=True to replace them")

    records = make_records(chunks, sources)
    texts = [chunk.embed_text for chunk in chunks]

    mark = time.perf_counter()
    if vectors is None:
        vectors = embedder.embed_documents(texts, (lambda done, total: progress("embedding", done, total)) if progress else None)
    embed_seconds = embedding_seconds if embedding_seconds is not None else time.perf_counter() - mark
    dim = int(embedder.dim)
    vectors = np.asarray(vectors, dtype=np.float32)
    _check_vectors(vectors, len(chunks), dim, getattr(embedder, "name", "the embedding model"))

    mark = time.perf_counter()
    store.add(records, vectors)
    store.flush()
    store_seconds = time.perf_counter() - mark
    if progress:
        progress("storing", len(records), len(records))

    if store.count() != len(records):
        raise IndexingError(f"the store holds {store.count()} records after adding {len(records)}")
    _probe(store, vectors, [r.id for r in records])
    limit, over, longest = _token_report(embedder, texts)

    info = IndexInfo(
        format=INFO_FORMAT,
        embedding_model=embedder.name,
        dim=dim,
        backend=backend or type(store).__name__,
        space=getattr(store, "space", "cosine"),
        n_records=len(records),
        n_documents=len({c.doc_id for c in chunks}),
        n_text=sum(c.kind == TEXT for c in chunks),
        n_notes=sum(c.kind == NOTES for c in chunks),
        chunks_file=str(chunks_file) if chunks_file else "",
        chunks_sha256=chunks_sha256 or "",
        built_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        seconds={"embedding": round(embed_seconds, 2), "storing": round(store_seconds, 2), "total": round(time.perf_counter() - started, 2)},
        libraries=library_versions(),
        token_limit=limit,
        over_token_limit=over,
        longest_tokens=longest,
    )
    if info_file is not None:
        write_info(info_file, info)
    return info


# ------------------------------------------------------------------------------------------------ opening


def open_index(
    directory: Path | str,
    backend: str,
    embedder: Embedder,
    *,
    chunks_path: Path | str | None = None,
    space: str = "cosine",
) -> OpenedIndex:
    """Open the finished index of ``backend`` in ``directory`` or raise ``IndexNotReady`` with the reason."""
    directory = Path(directory)
    path = info_path(directory, backend)
    if not path.is_file():
        if (directory / backend).exists():
            raise IndexNotReady(
                f"the {backend} index in {directory} was never finished (the build was interrupted or is still running). "
                "Rebuild it: python scripts/build_index.py"
            )
        raise IndexNotReady(f"there is no {backend} index in {directory} yet. Build it first: python scripts/build_index.py")
    info = read_info(path)
    if info.format != INFO_FORMAT:
        raise IndexNotReady(f"the index was written by a different version of RegLens (format {info.format}); rebuild it: python scripts/build_index.py")
    if info.embedding_model != embedder.name:
        raise IndexNotReady(
            f"the index was built with the embedding model {info.embedding_model!r}, but questions would now be embedded with "
            f"{embedder.name!r}; vectors of different models cannot be compared. Set EMBEDDING_MODEL={info.embedding_model} "
            "in .env, or rebuild the index: python scripts/build_index.py"
        )
    try:
        store = open_store(directory, backend, space=info.space)
        stored = store.count()
    except VectorStoreError as exc:
        raise IndexNotReady(f"the {backend} index is damaged ({exc}); rebuild it: python scripts/build_index.py") from exc
    if stored != info.n_records:
        raise IndexNotReady(
            f"the {backend} index holds {stored} records but should hold {info.n_records}; rebuild it: python scripts/build_index.py"
        )
    if int(embedder.dim) != info.dim:
        raise IndexNotReady(f"the index has vectors of size {info.dim}, the embedding model produces {embedder.dim}; rebuild the index")

    warnings: list[str] = []
    if chunks_path is not None and Path(chunks_path).is_file() and info.chunks_sha256:
        if file_sha256(chunks_path) != info.chunks_sha256:
            warnings.append(
                "chunks.jsonl changed after this index was built (the documents were ingested again?). "
                "The index still works, but it describes the old chunks: python scripts/build_index.py"
            )
    return OpenedIndex(store, info, tuple(warnings))
