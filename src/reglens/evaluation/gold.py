"""The gold set: questions with the evidence and the facts a correct answer needs (assignment section 5.1).

Nothing in it is a number measured by a model. Each answerable question names

* ``gold_docs``  - the documents that contain the answer,
* ``evidence``   - regular expressions; a passage is *relevant* when it comes from a gold document and one of them matches it,
* ``facts``      - groups of alternative regular expressions; an answer is *correct* when every group matches somewhere in it.

Questions that the corpus cannot answer have ``expect_abstain``: the right behaviour is to say so.
``check_corpus`` proves that every answerable question has evidence in the real corpus, so a gold question cannot
silently be unanswerable by the documents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("number", "temporal", "rule", "definition", "scope", "multi-document", "out-of-scope")


class GoldError(ValueError):
    """The gold file is malformed."""


def normalise(text: str) -> str:
    """Collapse every run of whitespace so that line breaks inside the documents do not break a match."""
    return " ".join(text.split())


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        raise GoldError(f"{where}: invalid regular expression {pattern!r}: {exc}") from exc


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    category: str
    question: str
    gold_docs: tuple[str, ...] = ()
    evidence: tuple[re.Pattern[str], ...] = ()
    facts: tuple[tuple[re.Pattern[str], ...], ...] = ()
    expect_abstain: bool = False
    needs_all_docs: bool = False
    note: str = ""
    expected: str = ""  # the answer in one sentence, for the judge and for the reader of the report

    @property
    def answerable(self) -> bool:
        return not self.expect_abstain

    def is_relevant(self, doc_id: str, text: str) -> bool:
        """A passage is relevant when it is in a gold document and one evidence pattern matches it."""
        if doc_id not in self.gold_docs:
            return False
        text = normalise(text)
        return any(pattern.search(text) for pattern in self.evidence)

    def covered_evidence(self, passages: Iterable[tuple[str, str]]) -> int:
        """How many evidence patterns are matched by at least one (doc_id, text) passage."""
        prepared = [(doc, normalise(text)) for doc, text in passages if doc in self.gold_docs]
        return sum(any(pattern.search(text) for _, text in prepared) for pattern in self.evidence)

    def missing_facts(self, answer: str) -> list[int]:
        """Indexes of the fact groups that the answer does not state (empty = every fact is there)."""
        text = normalise(answer)
        return [i for i, group in enumerate(self.facts) if not any(pattern.search(text) for pattern in group)]


def parse_question(data: Mapping[str, object], where: str = "gold") -> GoldQuestion:
    unknown = set(data) - {"id", "category", "question", "gold_docs", "evidence", "facts", "expect_abstain", "needs_all_docs", "note", "expected"}
    if unknown:
        raise GoldError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
    for key in ("id", "category", "question"):
        if not isinstance(data.get(key), str) or not str(data[key]).strip():
            raise GoldError(f"{where}: {key!r} is missing")
    if data["category"] not in CATEGORIES:
        raise GoldError(f"{where}: category must be one of {', '.join(CATEGORIES)}, not {data['category']!r}")
    abstain = bool(data.get("expect_abstain", False))
    docs = tuple(data.get("gold_docs", ()))  # type: ignore[arg-type]
    evidence = tuple(_compile(p, f"{where}.evidence") for p in data.get("evidence", ()))  # type: ignore[union-attr]
    facts = tuple(
        tuple(_compile(p, f"{where}.facts") for p in group) for group in data.get("facts", ())  # type: ignore[union-attr]
    )
    if abstain and (docs or evidence or facts):
        raise GoldError(f"{where}: a question that expects an abstention must not list documents, evidence or facts")
    if not abstain:
        if not docs or not evidence or not facts or any(not group for group in facts):
            raise GoldError(f"{where}: an answerable question needs gold_docs, evidence and facts")
    return GoldQuestion(
        id=str(data["id"]), category=str(data["category"]), question=str(data["question"]).strip(), gold_docs=docs,
        evidence=evidence, facts=facts, expect_abstain=abstain, needs_all_docs=bool(data.get("needs_all_docs", False)),
        note=str(data.get("note", "")), expected=str(data.get("expected", "")),
    )


def load_gold(path: Path | str) -> list[GoldQuestion]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GoldError(f"cannot read the gold file {path}: {exc}") from exc
    questions: list[GoldQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldError(f"{path.name} line {number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(data, dict):
            raise GoldError(f"{path.name} line {number}: expected a JSON object")
        question = parse_question(data, f"{path.name} line {number}")
        if question.id in seen:
            raise GoldError(f"{path.name} line {number}: the id {question.id!r} is used twice")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise GoldError(f"{path.name} contains no questions")
    return questions


def check_corpus(questions: Sequence[GoldQuestion], passages: Iterable[tuple[str, str]]) -> list[str]:
    """Problems found when the gold set is held against the real passages (an empty list means it is sound)."""
    docs: dict[str, list[str]] = {}
    for doc_id, text in passages:
        docs.setdefault(doc_id, []).append(normalise(text))
    problems: list[str] = []
    for question in questions:
        if not question.answerable:
            continue
        for doc in question.gold_docs:
            if doc not in docs:
                problems.append(f"{question.id}: gold document {doc!r} is not in the corpus")
        for pattern in question.evidence:
            if not any(pattern.search(text) for doc in question.gold_docs for text in docs.get(doc, ())):
                problems.append(f"{question.id}: no passage of the gold documents matches the evidence {pattern.pattern!r}")
        for number, group in enumerate(question.facts, 1):  # a fact the documents never state could not be expected from an answer
            if not any(pattern.search(text) for pattern in group for doc in question.gold_docs for text in docs.get(doc, ())):
                problems.append(f"{question.id}: fact group {number} ({group[0].pattern!r}...) matches nothing in the gold documents")
        if question.needs_all_docs:
            for doc in question.gold_docs:
                if not any(pattern.search(text) for pattern in question.evidence for text in docs.get(doc, ())):
                    problems.append(f"{question.id}: {doc!r} has no matching passage, but the question needs every document")
    return problems
