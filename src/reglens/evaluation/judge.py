"""An optional language-model judge for answer relevance, faithfulness and correctness (assignment section 5.1).

The programmatic checks in ``metrics.py`` need no model. The judge adds a reading of the answer by a language model,
on a fixed 1-5 scale. It is the same family of model that wrote the answer, which is a known bias (it tends to favour
its own phrasing); the report says so and the judge is switched on only with ``--judge``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from reglens.llm.base import LLMClient, LLMError, LLMRequest

JUDGE_SYSTEM = """You are a strict evaluator of answers about AI and data-protection regulation. You are given a QUESTION, the SOURCES that were available, the ANSWER to judge and, when known, the FACTS a correct answer must state.

Score three things from 1 (worst) to 5 (best):
- relevance: does the answer address what the question asks, without padding or off-topic material?
- faithfulness: is every claim in the answer supported by the SOURCES? Score 1-2 if it states anything the sources do not support, 5 only if every claim is supported.
- correctness: does the answer state the FACTS correctly (when FACTS are given)? If no FACTS are given, judge correctness from the SOURCES.

Reply with ONE line of JSON and nothing else: {"relevance": n, "faithfulness": n, "correctness": n, "reason": "one short sentence"}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)
MAX_SOURCE_WORDS = 1800


@dataclass(frozen=True)
class Judgement:
    relevance: int
    faithfulness: int
    correctness: int
    reason: str


def parse_judgement(text: str) -> Judgement | None:
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        values = [int(data[key]) for key in ("relevance", "faithfulness", "correctness")]
    except (ValueError, KeyError, TypeError):
        return None
    if any(not 1 <= v <= 5 for v in values):
        return None
    return Judgement(*values, reason=" ".join(str(data.get("reason", "")).split())[:300])


def build_prompt(question: str, answer: str, sources: list[str], facts: list[str]) -> str:
    words = 0
    kept: list[str] = []
    for index, text in enumerate(sources, 1):
        count = len(text.split())
        if words + count > MAX_SOURCE_WORDS:
            break
        words += count
        kept.append(f"[S{index}] {text}")
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none given)"
    return f"QUESTION:\n{question}\n\nSOURCES:\n" + ("\n\n".join(kept) or "(no sources)") + f"\n\nFACTS:\n{facts_block}\n\nANSWER:\n{answer}"


def judge(llm: LLMClient, question: str, answer: str, sources: list[str], facts: list[str]) -> Judgement | None:
    """One judgement, or ``None`` when the model fails or does not answer in the requested form."""
    request = LLMRequest.single(JUDGE_SYSTEM, build_prompt(question, answer, sources, facts), max_output_tokens=512)
    try:
        return parse_judgement(llm.generate(request).text)
    except LLMError:
        return None
