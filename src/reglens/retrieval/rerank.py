"""Re-ranking with a cross-encoder (assignment section 5.3, "re-ranking models").

The first-stage search compares a query vector with chunk vectors that were computed *independently* of the query -
fast, but coarse. A cross-encoder reads the query and one passage **together** and outputs a relevance score; it is far
more precise and far too slow to run over thousands of chunks. So the usual design is two-stage: the fast search picks
a few dozen candidates and the cross-encoder puts the best of them first. Whether that helps on legal text is an
empirical question; the evaluation measures it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from typing import Protocol

DEFAULT_BATCH = 16


class RerankError(RuntimeError):
    """The re-ranking model could not be loaded or failed."""


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """A relevance score for each passage (higher = more relevant); same order as ``passages``."""
        ...


def load_cross_encoder(model_name: str, device: str | None = None):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankError("sentence-transformers is not installed - run: pip install -r requirements.txt") from exc
    try:
        return CrossEncoder(model_name, device=device)
    except Exception as exc:
        raise RerankError(
            f"could not load the re-ranking model {model_name!r} ({type(exc).__name__}: {exc}). The first run downloads "
            "it from huggingface.co (about 90 MB): check the internet connection."
        ) from exc


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = DEFAULT_BATCH,
        device: str | None = None,
        loader: Callable[..., object] = load_cross_encoder,
    ) -> None:
        self.name = model_name
        self._batch = batch_size
        self._device = device
        self._loader = loader
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            if self._model is None:
                self._model = self._loader(self.name, self._device)
            try:
                raw = self._model.predict(
                    [(query, passage) for passage in passages], batch_size=self._batch, show_progress_bar=False
                )
            except Exception as exc:
                raise RerankError(f"the re-ranking model failed ({type(exc).__name__}: {exc})") from exc
        scores = [float(value) for value in list(raw)]
        if len(scores) != len(passages) or any(value != value for value in scores):  # NaN check
            raise RerankError("the re-ranking model returned an unexpected result")
        return scores
