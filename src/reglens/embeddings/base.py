"""What an embedding model is, as far as the rest of RegLens is concerned (assignment section 2.2).

An embedder turns a text into a vector so that texts with similar meaning get vectors that point in similar
directions. Every embedder in RegLens returns **unit-length float32 vectors**: with unit vectors the cosine similarity
of two texts is just their dot product, and ranking by cosine similarity, by dot product and by (smallest) Euclidean
distance give the same order. That is why the vector store can use any of the three measures interchangeably.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

ProgressFn = Callable[[int, int], None]  # (texts done, texts in total)


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or produced unusable vectors."""


@runtime_checkable
class Embedder(Protocol):
    name: str  # identifies the model; stored with the index so a query is never embedded by a different model
    dim: int

    def embed_documents(self, texts: Sequence[str], progress: ProgressFn | None = None) -> np.ndarray:
        """One row per text, in the order given."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """A single vector for a search query (some models treat queries and passages differently)."""
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 (a row of zeros stays zero instead of becoming NaN)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms > 0, norms, 1.0)).astype(np.float32, copy=False)
