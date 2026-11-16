"""Query transformation (assignment section 2.4): from the user's wording to what is actually searched.

A question as typed is often a poor search query: it can depend on the conversation ("and in the US?"), use everyday
words instead of legal terms, or be short and vague. This module builds a ``QueryPlan`` in up to four steps:

1. **Condense** (needs the model, only when there is a conversation): rewrite a follow-up question so that it stands on
   its own.
2. **Glossary** (free, instant): append official terms to the *keyword* query ("data leak" -> "personal data breach").
3. **Multi-query** (needs the model, optional): paraphrase the question a few times in the vocabulary of the
   regulations; every paraphrase is searched and the result lists are fused.
4. **HyDE** (needs the model, optional): let the model write a short passage that *would* answer the question (without
   caring whether it is right) and search with that passage - answers look more like answers than questions do.

A model failure never stops retrieval: the step is skipped and the plan says so.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from reglens.llm.base import LLMClient, LLMError, LLMRequest
from reglens.retrieval.glossary import expand

MAX_QUERY_CHARS = 400
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 600
_LIST_MARK = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")

CONDENSE_SYSTEM = (
    "You rewrite the last question of a conversation so that it can be understood without the conversation. "
    "Keep every name, article number, date and legal term. Do not answer the question and do not add information. "
    "Reply with the rewritten question only, on one line."
)
MULTI_QUERY_SYSTEM = (
    "You help search a library of EU and US regulation (GDPR, EU AI Act, NIST, HIPAA, CCPA) and official guidance. "
    "Write {n} different search queries that express the user's question in the vocabulary a regulation or guidance "
    "document would use. One query per line, no numbering, no explanations."
)
HYDE_SYSTEM = (
    "Write a short passage (about 80 words) in the style of a regulation or official guidance document that would "
    "answer the user's question. It only has to sound like such a document: it is used to find real documents, and "
    "it will never be shown to anyone. Do not invent article numbers. Reply with the passage only."
)


@dataclass(frozen=True)
class Turn:
    """One exchange of a conversation."""

    question: str
    answer: str


@dataclass(frozen=True)
class DenseQuery:
    text: str
    as_passage: bool = False  # embed like a document passage (HyDE) instead of like a search query


@dataclass(frozen=True)
class TransformOptions:
    glossary: bool = True
    condense: bool = True
    multi_query: int = 0  # number of paraphrases to add (0 = off)
    hyde: bool = False


@dataclass(frozen=True)
class QueryPlan:
    question: str  # as asked
    standalone: str  # after condensing the conversation (equal to ``question`` when nothing was changed)
    dense: tuple[DenseQuery, ...]
    lexical: str
    notes: tuple[str, ...] = ()  # what was done, in plain words, for the interface
    llm_calls: int = 0

    @classmethod
    def plain(cls, question: str) -> QueryPlan:
        """No transformation at all: the question is both the dense and the keyword query."""
        return cls(question, question, (DenseQuery(question),), question)


def _first_line(text: str, limit: int = MAX_QUERY_CHARS) -> str:
    for line in text.strip().splitlines():
        line = _LIST_MARK.sub("", line).strip().strip("\"'`*")
        if line:
            return line[:limit]
    return ""


def parse_queries(text: str, n: int, limit: int = MAX_QUERY_CHARS) -> list[str]:
    """Read a model's list of queries (one per line, maybe bulleted or numbered): at most ``n``, without repeats."""
    found: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARK.sub("", line).strip().strip("\"'`*")[:limit]
        if len(line) >= 8 and line.lower() not in (f.lower() for f in found):
            found.append(line)
        if len(found) == n:
            break
    return found


def format_history(history: Sequence[Turn]) -> str:
    recent = history[-HISTORY_TURNS:]
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer[:HISTORY_ANSWER_CHARS]}" for t in recent)


class QueryTransformer:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def _ask(self, system: str, user: str) -> str:
        assert self._llm is not None
        # a generous output limit: some models spend part of it on thinking before they write the answer
        return self._llm.generate(LLMRequest.single(system, user, max_output_tokens=1024)).text

    def plan(self, question: str, history: Sequence[Turn] = (), options: TransformOptions = TransformOptions()) -> QueryPlan:
        question = " ".join(question.split())
        if not question:
            raise ValueError("the question is empty")
        notes: list[str] = []
        calls = 0
        standalone = question

        needs_llm = (options.condense and history) or options.multi_query or options.hyde
        if needs_llm and self._llm is None:
            notes.append("no language model is available, so only the glossary was applied")

        if options.condense and history and self._llm is not None:
            calls += 1
            try:
                rewritten = _first_line(self._ask(CONDENSE_SYSTEM, f"{format_history(history)}\n\nLast question: {question}"))
            except LLMError as exc:
                notes.append(f"could not rewrite the follow-up question ({exc.kind}); the question was used as typed")
            else:
                if rewritten:
                    standalone = rewritten
                    if rewritten != question:
                        notes.append(f"follow-up rewritten as: {rewritten}")

        dense = [DenseQuery(standalone)]
        lexical, applied = expand(standalone) if options.glossary else (standalone, [])
        for name, terms in applied:
            notes.append(f"glossary ({name}): added {terms}")

        if options.multi_query and self._llm is not None:
            calls += 1
            try:
                variants = parse_queries(self._ask(MULTI_QUERY_SYSTEM.format(n=options.multi_query), standalone), options.multi_query)
            except LLMError as exc:
                notes.append(f"multi-query skipped ({exc.kind})")
            else:
                known = {standalone.lower()}
                for variant in variants:
                    if variant.lower() not in known:
                        known.add(variant.lower())
                        dense.append(DenseQuery(variant))
                notes.append(f"{len(dense) - 1} paraphrase(s) searched as well")

        if options.hyde and self._llm is not None:
            calls += 1
            try:
                passage = " ".join(self._ask(HYDE_SYSTEM, standalone).split())[:1500]
            except LLMError as exc:
                notes.append(f"hypothetical passage skipped ({exc.kind})")
            else:
                if passage:
                    dense.append(DenseQuery(passage, as_passage=True))
                    notes.append("a hypothetical answer passage was searched as well (HyDE)")

        return QueryPlan(question, standalone, tuple(dense), lexical, tuple(notes), calls)
