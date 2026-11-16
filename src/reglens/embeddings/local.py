"""Local embedding models through the ``sentence-transformers`` library (the default of RegLens).

The default model is ``BAAI/bge-small-en-v1.5``: 384-dimensional vectors, a 512-token input window, about 130 MB.
It runs on an ordinary laptop CPU, is free, and no document text leaves the machine. Search queries get the
instruction prefix the model was trained with ("Represent this sentence for searching relevant passages: ");
passages get none.

The library is imported only when the model is first needed, so the rest of RegLens (and its tests) work without it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence

import numpy as np

from reglens.embeddings.base import EmbeddingError, ProgressFn, normalize_rows

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GROUP_SIZE = 64  # texts embedded between two progress reports


def default_prefixes(model_name: str) -> tuple[str, str]:
    """The (query, passage) prefixes the model family expects."""
    name = model_name.lower()
    if "bge" in name and "-en" in name:
        return BGE_QUERY_PREFIX, ""
    if "e5" in name.split("/")[-1]:
        return "query: ", "passage: "
    return "", ""


def load_sentence_transformer(model_name: str, device: str | None = None):
    """Import sentence-transformers and load ``model_name`` (downloaded on first use, then cached)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # a harmless warning on Windows
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError("sentence-transformers is not installed - run: pip install -r requirements.txt") from exc
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:  # the library raises many kinds of errors (network, disk, format)
        raise EmbeddingError(
            f"could not load the embedding model {model_name!r} ({type(exc).__name__}: {exc}). The first run downloads "
            "it from huggingface.co (about 130 MB for the default model): check the internet connection, VPN or proxy."
        ) from exc


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        loader: Callable[..., object] = load_sentence_transformer,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        default_query, default_document = default_prefixes(model_name)
        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.document_prefix = default_document if document_prefix is None else document_prefix
        self._device = device
        self._loader = loader
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()  # one thread at a time inside the model

    # ---- loading
    def _loaded(self):
        with self._lock:
            if self._model is None:
                self._model = self._loader(self.name, self._device)
            return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._loaded()
        with self._lock:
            try:
                raw = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingError(f"the embedding model failed ({type(exc).__name__}: {exc})") from exc
        vectors = normalize_rows(np.asarray(raw, dtype=np.float32))  # normalise again: correctness must not depend on a flag
        if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise EmbeddingError("the embedding model returned an unexpected result (wrong shape or NaN values)")
        return vectors

    # ---- the Embedder interface
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._encode(["dimension probe"]).shape[1])
        return self._dim

    def embed_documents(self, texts: Sequence[str], progress: ProgressFn | None = None) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [self.document_prefix + text for text in texts]
        # Short texts first: batches then contain texts of similar length, which wastes less padding.
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        out = np.zeros((len(prepared), self.dim), dtype=np.float32)
        for start in range(0, len(order), GROUP_SIZE):
            ids = order[start : start + GROUP_SIZE]
            out[ids] = self._encode([prepared[i] for i in ids])
            if progress:
                progress(min(start + GROUP_SIZE, len(order)), len(order))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    # ---- optional: how much of each text the model can actually read
    @property
    def max_tokens(self) -> int | None:
        value = getattr(self._loaded(), "max_seq_length", None)
        return int(value) if isinstance(value, (int, float)) and value > 0 else None

    def token_counts(self, texts: Sequence[str]) -> list[int] | None:
        """Number of tokens of each (prefixed) text, or None if the model exposes no tokenizer."""
        tokenizer = getattr(self._loaded(), "tokenizer", None)
        if tokenizer is None:
            return None
        counts: list[int] = []
        for start in range(0, len(texts), 256):
            group = [self.document_prefix + t for t in texts[start : start + 256]]
            # verbose=False: the tokenizer would otherwise print its own "sequence longer than the maximum" warning for
            # every overlong passage; RegLens counts and reports those itself.
            encoded = tokenizer(group, add_special_tokens=True, truncation=False, verbose=False)
            counts.extend(len(ids) for ids in encoded["input_ids"])
        return counts
