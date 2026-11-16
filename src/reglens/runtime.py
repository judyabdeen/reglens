"""Wire the parts together: settings -> embedder + index + retriever + language model -> ``RAGPipeline``.

The command-line tools, the web interface and the evaluation all start from ``build_runtime``, so they open the index
the same way and fail in the same, readable way. A missing or broken language model is not an error here: the pipeline
then answers with the retrieved passages and says why (see ``RAGPipeline``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from reglens.config import ConfigError, Settings
from reglens.corpus.manifest import ManifestError, Source, load_manifest
from reglens.embeddings import Embedder, make_embedder
from reglens.indexing import IndexInfo, IndexNotReady, open_index
from reglens.llm.base import LLMClient, LLMError
from reglens.llm.factory import make_llm
from reglens.pipeline import RAGPipeline
from reglens.retrieval.rerank import CrossEncoderReranker, Reranker
from reglens.retrieval.retriever import Retriever


@dataclass(frozen=True)
class Runtime:
    pipeline: RAGPipeline
    retriever: Retriever
    embedder: Embedder
    index: IndexInfo
    settings: Settings
    llm_provider: str | None  # None: no language model could be set up (see llm_problem)
    llm_model: str | None
    llm_problem: str | None
    warnings: tuple[str, ...] = ()

    @property
    def has_llm(self) -> bool:
        return self.pipeline.llm is not None


def load_sources(settings: Settings) -> dict[str, Source]:
    """The manifest as {doc_id: Source}; an unreadable manifest gives an empty mapping (metadata is then missing)."""
    try:
        return {source.id: source for source in load_manifest(settings.manifest_path)}
    except (OSError, ManifestError):
        return {}


def build_runtime(
    settings: Settings,
    *,
    backend: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    min_interval: float = 0.0,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
) -> Runtime:
    """Open the index and assemble the pipeline. Raises ``IndexNotReady`` when there is no usable index.

    ``embedder``, ``reranker`` and ``llm`` can be passed in (tests, evaluation); by default they come from ``settings``.
    The embedding model and the re-ranking model are loaded the first time they are needed, not here.
    """
    backend = backend or settings.vectorstore_backend
    embedder = embedder or make_embedder(settings.embedding_model)
    opened = open_index(settings.vectorstore_dir, backend, embedder, chunks_path=settings.chunks_path)
    retriever = Retriever.from_store(opened.store, embedder, reranker or CrossEncoderReranker(settings.reranker_model))

    problem: str | None = None
    if llm is None:
        try:
            llm = make_llm(settings, provider, model=model, min_interval=min_interval)
        except (LLMError, ConfigError) as exc:
            problem = str(exc)
    pipeline = RAGPipeline(retriever, llm, llm_unavailable=problem)
    warnings = list(opened.warnings)
    if settings.vectorstore_dir_moved_from:
        warnings.append(
            f"The project is inside {settings.vectorstore_dir_moved_from}, so the index is kept in {settings.vectorstore_dir} "
            "(a sync client can lock database files while it uploads them)."
        )
    return Runtime(
        pipeline=pipeline,
        retriever=retriever,
        embedder=embedder,
        index=opened.info,
        settings=settings,
        llm_provider=getattr(llm, "provider", None),
        llm_model=getattr(llm, "model", None),
        llm_problem=problem,
        warnings=tuple(warnings),
    )


def describe(runtime: Runtime) -> Mapping[str, object]:
    """A few facts about the running system, for the interface footer and for logs (no secrets)."""
    info = runtime.index
    return {
        "documents": info.n_documents,
        "passages": info.n_records,
        "embedding_model": info.embedding_model,
        "vector_store": info.backend,
        "space": info.space,
        "reranker": runtime.settings.reranker_model,
        "language_model": f"{runtime.llm_provider}/{runtime.llm_model}" if runtime.has_llm else None,
        "built_at": info.built_at,
    }


__all__ = ["IndexNotReady", "Runtime", "build_runtime", "describe", "load_sources"]
