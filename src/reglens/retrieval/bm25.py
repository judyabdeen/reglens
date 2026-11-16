"""BM25: the classic keyword ranking function, written from scratch.

BM25 scores a document for a query by adding up, for every query word, how rare the word is in the whole corpus (idf)
times how often it appears in the document, with diminishing returns for repetition (``k1``) and a correction for long
documents (``b``). It finds exact terms - article numbers, defined terms, regulation numbers - that a meaning-based
embedding can blur. RegLens combines it with the dense search (see ``fusion.py``).

    score(d, q) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avg|d|))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from reglens.retrieval.text import tokenize
from reglens.vectorstore.base import Record, Where
from reglens.vectorstore.filters import matches, normalize_where
from reglens.vectorstore.records import lexical_text


class BM25Index:
    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadata: Sequence[Mapping] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(ids) != len(texts) or (metadata is not None and len(metadata) != len(ids)):
            raise ValueError("ids, texts and metadata must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        self.k1, self.b = k1, b
        self._ids = list(ids)
        self._metadata = [dict(m) for m in metadata] if metadata is not None else [{} for _ in ids]
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths = np.zeros(len(ids), dtype=np.float64)
        for index, text in enumerate(texts):
            tokens = tokenize(text)
            lengths[index] = len(tokens)
            for token, count in Counter(tokens).items():
                docs, freqs = postings.setdefault(token, ([], []))
                docs.append(index)
                freqs.append(count)
        self._lengths = lengths
        n = len(ids)
        average = float(lengths.mean()) if n else 0.0
        self._norm = k1 * (1.0 - b + b * lengths / average) if average else np.zeros(n)
        self._postings = {t: (np.array(d, dtype=np.int64), np.array(f, dtype=np.float64)) for t, (d, f) in postings.items()}
        self._idf = {t: math.log(1.0 + (n - len(d) + 0.5) / (len(d) + 0.5)) for t, (d, _) in postings.items()}

    @classmethod
    def from_records(cls, records: Sequence[Record], **options) -> BM25Index:
        return cls([r.id for r in records], [lexical_text(r) for r in records], [r.metadata for r in records], **options)

    def __len__(self) -> int:
        return len(self._ids)

    def document_frequency(self, term: str) -> int:
        """In how many documents does this (already tokenised) term occur?"""
        posting = self._postings.get(term)
        return 0 if posting is None else len(posting[0])

    def search(self, query: str, k: int, where: Where | None = None) -> list[tuple[str, float]]:
        """Up to ``k`` (id, score) pairs, best first. Documents that share no term with the query are not returned."""
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            docs, freqs = posting
            scores[docs] += self._idf[term] * freqs * (self.k1 + 1.0) / (freqs + self._norm[docs])
        if normalize_where(where):
            scores[[not matches(m, where) for m in self._metadata]] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if not candidates.size:
            return []
        if k < candidates.size:
            candidates = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        candidates = candidates[np.lexsort((candidates, -scores[candidates]))]
        return [(self._ids[i], float(scores[i])) for i in candidates]
