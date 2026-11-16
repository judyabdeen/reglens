"""Turning text into the tokens of the keyword (BM25) index.

Legal text is full of numbers and defined terms that must match exactly ("72", "Annex III", "2026/1744"), so numbers
are kept as tokens, and the only normalisation is lower-casing and removing the plural "s". Dense embeddings handle
the rest of the vocabulary variation (see the evaluation of dense versus keyword versus hybrid retrieval).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have how i if in into is it its "
    "me my of on or our shall should so than that the their them then there these they this those to us was we were "
    "what when where which who whom why will with would you your".split()
)


def stem(token: str) -> str:
    """Remove a plural ending: authorities -> authority, processes -> process, controllers -> controller."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers without stop words, each with its plural ending removed."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
