"""Reciprocal rank fusion (RRF): merge several ranked lists into one.

A dense search and a keyword search return scores on different scales (cosine similarity versus BM25), so their
scores cannot be added. RRF ignores the scores and uses only the ranks: a document at rank r in a list earns
``weight / (k + r)``, and a document's final score is the sum over all lists. A document that is near the top of both
lists wins; one that only a single method likes still gets a fair chance. ``k`` = 60 is the value from the original
paper (Cormack, Clarke and Buettcher, 2009) and flattens the influence of the very first ranks.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ``rankings`` (lists of ids, best first). Returns (id, score), best first; ties keep first-seen order."""
    if k < 0:
        raise ValueError("k must not be negative")
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is needed")
    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        seen: set[str] = set()
        position = 0
        for item in ranking:
            if item in seen:  # a repeated id in one list counts once, at its best rank
                continue
            seen.add(item)
            position += 1
            scores[item] = scores.get(item, 0.0) + weight / (k + position)
    order = {item: i for i, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))
