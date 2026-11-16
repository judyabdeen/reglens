"""A cheap, programmatic check on the figures in an answer.

In legal answers the dangerous mistakes are wrong numbers: a deadline, an amount, an article number, a year. This
check lists every such figure in the answer that does not occur anywhere in the sources the model was given. A figure
on that list is not necessarily wrong (the model may have added two numbers, or the source may spell it out in
words), but it is exactly what a human reviewer should look at first, and in the evaluation it counts against the
answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_LIST_NUMBER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:[,  ]\d{3})+|\d+)(?:[.,]\d+)?(?![\w])")
_RUN = re.compile(r"\d+")
_UNIT = re.compile(
    r"^\s*(?:%|per ?cent|percent|days?|hours?|minutes?|weeks?|months?|years?|eur|euros?|€|usd|dollars?|\$|million|billion|thousand)",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[tuple[str, int]]:
    """(number with separators removed, end position) for every number in ``text``."""
    return [(re.sub(r"[,  ]", "", m.group(1)), m.end()) for m in _NUMBER.finditer(text)]


def figures(text: str) -> list[str]:
    """The figures worth checking: numbers of two or more digits, and any number followed by a unit."""
    cleaned = _LIST_NUMBER.sub("", text)  # source labels such as [S12] are never read as numbers: the S in front stops the match
    out: list[str] = []
    for number, end in _numbers(cleaned):
        if (len(number) >= 2 or _UNIT.match(cleaned[end : end + 16])) and number not in out:
            out.append(number)
    return out


def _known(text: str) -> set[str]:
    """Every number in ``text``, both as written with thousands separators ("35 000 000" -> 35000000) and as its parts."""
    return {number for number, _ in _numbers(text)} | set(_RUN.findall(text))


def unverified_figures(answer: str, sources: Iterable[str], question: str = "") -> list[str]:
    """Figures in ``answer`` that appear neither in any source nor in the question itself.

    A source is what the model was shown: its heading (title, version date, section, pages) and its passage. A model
    that writes "Regulation (EU) 2026/1744" has read that number in the heading, so it is not an invention.
    """
    known = set().union(*(_known(text) for text in sources), _known(question))
    return [number for number in figures(answer) if number not in known]
