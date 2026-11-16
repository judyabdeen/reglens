"""A dependency-free embedder based on feature hashing ("the hashing trick").

It maps the words and word pairs of a text into a fixed number of slots, so two texts that share words get vectors
with a high cosine similarity. It knows **nothing about meaning** - "car" and "automobile" are unrelated to it - so it
is not a substitute for a real embedding model. RegLens uses it for three honest purposes:

* unit tests and demos that must run without downloading a model;
* a lexical baseline in the evaluation (how much does a real model add over word overlap?);
* a way to run the whole pipeline end to end when the Hugging Face download is blocked.

The hash is ``blake2b`` (not Python's ``hash``, which changes between runs), so vectors are reproducible.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from reglens.embeddings.base import ProgressFn, normalize_rows

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were which with "
    "shall may must any such not no".split()
)


@lru_cache(maxsize=500_000)
def _slot(feature: str, dim: int) -> tuple[int, int]:
    digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
    return digest % dim, 1 if (digest >> 63) & 1 else -1


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        content = [w for w in words if w not in _STOP]
        features = Counter(content)
        features.update(f"{a} {b}" for a, b in zip(content, content[1:], strict=False))
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in features.items():
            slot, sign = _slot(feature, self.dim)
            vector[slot] += sign * (1.0 + math.log(count))
        return vector

    def embed_documents(self, texts: Sequence[str], progress: ProgressFn | None = None) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            rows[index] = self._vector(text)
            if progress and ((index + 1) % 256 == 0 or index + 1 == len(texts)):
                progress(index + 1, len(texts))
        return normalize_rows(rows) if len(texts) else rows

    def embed_query(self, text: str) -> np.ndarray:
        return normalize_rows(self._vector(text))[0]
