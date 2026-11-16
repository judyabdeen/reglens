"""Reading the citations in an answer and checking them against the sources that were supplied.

The model is told to cite as ``[S2]`` or ``[S1][S3]``. Nothing about that is trusted: every label is checked against
the list of sources of this answer (a label that does not exist is a fabricated citation), and the share of
factual-looking sentences that carry a citation is measured. Both numbers are shown to the user and used by the
evaluation as a faithfulness signal.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from reglens.ingest.chunking import split_sentences

_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_LABEL = re.compile(r"S\d+")
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
MIN_CLAIM_WORDS = 6
# a full stop followed by citations: "... within 72 hours. [S1]" or "... hours.[S1][S2]"
_AFTER_STOP = re.compile(r"(?<=\S)([.!?])\s*((?:\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]\s*)+)")


def extract_labels(text: str) -> list[str]:
    """The distinct source labels cited in ``text``, in order of first appearance."""
    found: list[str] = []
    for group in _GROUP.findall(text):
        for label in _LABEL.findall(group):
            if label not in found:
                found.append(label)
    return found


def invalid_labels(labels: Collection[str], valid: Collection[str]) -> list[str]:
    """Cited labels that are not among the sources supplied (fabricated citations)."""
    return [label for label in labels if label not in set(valid)]


def strip_citations(text: str) -> str:
    return " ".join(_GROUP.sub("", text).split())


def attach_citations(line: str) -> str:
    """Move citations that follow a full stop in front of it, so that they stay with their own sentence.

    Models write "statement [S1]." and also "statement. [S1]". A sentence splitter cuts the second form after the full
    stop and would leave the citation in front of the next sentence, so the statement would count as uncited.
    """
    return _AFTER_STOP.sub(lambda m: f" {' '.join(m.group(2).split())}{m.group(1)} ", line).strip()


def claim_sentences(text: str) -> list[str]:
    """Sentences that state something (long enough, not a heading, not a pure caveat)."""
    claims: list[str] = []
    for line in text.splitlines():
        line = attach_citations(_BULLET.sub("", line.strip()))
        if not line or (line.endswith(":") and len(line.split()) < 12):
            continue
        for sentence in split_sentences(line):
            plain = strip_citations(sentence)
            if len(plain.split()) < MIN_CLAIM_WORDS or re.match(r"(?i)^(this is (not )?(legal advice|research)|note\b|caveat\b)", plain):
                continue
            claims.append(sentence)
    return claims


def citation_coverage(text: str) -> tuple[float | None, list[str]]:
    """(share of claim sentences that carry a citation, the sentences that do not). None if there are no claims."""
    claims = claim_sentences(text)
    if not claims:
        return None, []
    uncited = [c for c in claims if not _GROUP.search(c)]
    return (len(claims) - len(uncited)) / len(claims), uncited
