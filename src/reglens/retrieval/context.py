"""Context window assembly (assignment section 2.4): choose what goes into the prompt.

The language model can only read so much, and more text is not always better: irrelevant passages distract it. The
assembler walks down the ranked chunks, keeps each one that still fits into the word budget, skips exact duplicates,
and numbers the survivors S1, S2, ... in rank order. These labels are what the model cites and what the interface
turns into clickable source cards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reglens.retrieval.retriever import Retrieved

DEFAULT_MAX_WORDS = 3000  # about 4,500 tokens of legal English
DEFAULT_MAX_ITEMS = 12


@dataclass(frozen=True)
class ContextItem:
    label: str  # "S1", "S2", ...
    retrieved: Retrieved

    @property
    def words(self) -> int:
        return len(self.retrieved.text.split())


@dataclass(frozen=True)
class Context:
    items: tuple[ContextItem, ...]
    dropped: tuple[str, ...] = ()  # ids of retrieved chunks that were left out (budget or duplicate)

    @property
    def words(self) -> int:
        return sum(item.words for item in self.items)

    def by_label(self, label: str) -> ContextItem | None:
        return next((item for item in self.items if item.label == label), None)


def assemble_context(
    results: Sequence[Retrieved], *, max_words: int = DEFAULT_MAX_WORDS, max_items: int = DEFAULT_MAX_ITEMS
) -> Context:
    if max_words < 1 or max_items < 1:
        raise ValueError("max_words and max_items must be at least 1")
    kept: list[Retrieved] = []
    dropped: list[str] = []
    seen_texts: set[str] = set()
    used = 0
    for result in results:  # already in rank order
        words = len(result.text.split())
        fingerprint = " ".join(result.text.split()).lower()
        if fingerprint in seen_texts or len(kept) >= max_items or (kept and used + words > max_words):
            dropped.append(result.id)
            continue
        seen_texts.add(fingerprint)
        kept.append(result)
        used += words
    items = tuple(ContextItem(f"S{i}", r) for i, r in enumerate(kept, start=1))
    return Context(items, tuple(dropped))
