"""Scores for retrieval and for answers. Every number is computed from the run; nothing is estimated.

Retrieval (is the evidence among the first ``k`` passages?)
    hit@k       share of questions with at least one relevant passage in the first k
                (when a question needs every gold document, each of them must be there)
    MRR         mean of 1 / rank of the first relevant passage (0 when none is among the retrieved passages)
    coverage@k  mean share of the question's evidence patterns that the first k passages contain
    doc precision@k  mean share of the first k passages that come from a gold document

Answers
    correctness   every fact group of the gold question is stated in the answer; for questions the corpus cannot
                  answer, the answer abstains
    faithfulness  checked by the pipeline itself, without a language model: every claim carries a citation
                  (citation coverage), no cited label is invented, no figure is missing from the sources
                  ("grounded" = all three), and the share of cited passages that come from a gold document
    relevance     optional: a language-model judge (see judge.py)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from reglens.evaluation.gold import GoldQuestion
from reglens.generation.citations import strip_citations

KS = (1, 3, 5, 8, 12)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    text: str


def _first_rank(question: GoldQuestion, passages: Sequence[Passage]) -> int | None:
    """Rank of the first relevant passage; with ``needs_all_docs``, the rank at which every gold document has one."""
    if question.needs_all_docs:
        seen: dict[str, int] = {}
        for rank, passage in enumerate(passages, 1):
            if passage.doc_id not in seen and question.is_relevant(passage.doc_id, passage.text):
                seen[passage.doc_id] = rank
        wanted = set(question.gold_docs)
        # a gold document with no matching evidence anywhere cannot be required (check_corpus reports it)
        return max(seen.values()) if wanted and wanted <= set(seen) else None
    for rank, passage in enumerate(passages, 1):
        if question.is_relevant(passage.doc_id, passage.text):
            return rank
    return None


def score_retrieval(question: GoldQuestion, passages: Sequence[Passage], ks: Sequence[int] = KS) -> dict:
    """Retrieval scores of one answerable question; ``passages`` are in the order the retriever returned them."""
    first = _first_rank(question, passages)
    total = len(question.evidence) or 1
    return {
        "first_relevant_rank": first,
        "reciprocal_rank": (1.0 / first) if first else 0.0,
        "relevant_ranks": [r for r, p in enumerate(passages, 1) if question.is_relevant(p.doc_id, p.text)],
        "hit": {str(k): bool(first and first <= k) for k in ks},
        "coverage": {str(k): question.covered_evidence((p.doc_id, p.text) for p in passages[:k]) / total for k in ks},
        "doc_precision": {
            str(k): (sum(p.doc_id in question.gold_docs for p in passages[:k]) / len(passages[:k])) if passages[:k] else 0.0 for k in ks
        },
    }


def summarise_retrieval(scores: Iterable[Mapping], ks: Sequence[int] = KS) -> dict:
    scores = list(scores)
    if not scores:
        return {"n": 0}
    out: dict = {"n": len(scores), "mrr": fmean(s["reciprocal_rank"] for s in scores)}
    for name in ("hit", "coverage", "doc_precision"):
        out[name] = {str(k): fmean(float(s[name][str(k)]) for s in scores) for k in ks}
    return out


# ------------------------------------------------------------------------------------------------ answers


def is_grounded(mode: str, abstained: bool, coverage: float | None, invalid: Sequence, figures: Sequence) -> bool:
    """A grounded answer: from the documents, every claim cited, no invented citation, no figure missing from the sources."""
    return mode == "rag" and not abstained and (coverage is None or coverage >= 1.0) and not invalid and not figures


def score_answer(question: GoldQuestion, answer) -> dict:
    """Score an ``Answer`` (see ``reglens.generation.answer``) against its gold question.

    ``answer.mode`` is ``rag`` for a grounded answer and ``llm_only`` for the baseline. An answer that is not a language
    model answer at all (``extractive``: the model failed) is reported as ``failed`` and is not counted as wrong.
    """
    failed = answer.mode == "extractive" or bool(answer.error_kind)
    text = strip_citations(answer.text)
    abstained = bool(answer.abstained)
    if question.expect_abstain:
        correct = abstained
        missing: list[int] = []
    else:
        missing = question.missing_facts(text) if not abstained else list(range(len(question.facts)))
        correct = not missing and not abstained
    cited = [s for s in answer.sources if s.cited]
    gold_cited = [s for s in cited if s.doc_id in question.gold_docs]
    grounded = is_grounded(answer.mode, abstained, answer.citation_coverage, answer.invalid_citations, answer.unverified_figures)
    return {
        "failed": failed,
        "error_kind": answer.error_kind,
        "abstained": abstained,
        "correct": correct if not failed else None,
        "missing_facts": missing,
        "citation_coverage": answer.citation_coverage,
        "cited_labels": len(answer.cited),
        "invalid_citations": list(answer.invalid_citations),
        "unverified_figures": list(answer.unverified_figures),
        "grounded": grounded if answer.mode == "rag" and not abstained else None,
        "cited_from_gold": (len(gold_cited) / len(cited)) if cited and not question.expect_abstain else None,
        "latency_s": answer.timings.get("total"),
        "retrieve_s": answer.timings.get("retrieve"),
        "generate_s": answer.timings.get("generate"),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "truncated": answer.truncated,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    items = [v for v in values if v is not None]
    return fmean(items) if items else None


def _rate(values: Iterable[bool | None]) -> float | None:
    items = [v for v in values if v is not None]
    return (sum(items) / len(items)) if items else None


def summarise_answers(rows: Sequence[Mapping]) -> dict:
    """Aggregate the ``score_answer`` rows of one configuration. ``rows`` carry ``category`` and ``expect_abstain``."""
    scored = [r for r in rows if not r["failed"]]
    answerable = [r for r in scored if not r["expect_abstain"]]
    refusals = [r for r in scored if r["expect_abstain"]]
    return {
        "asked": len(rows),
        "failed": len(rows) - len(scored),
        "correct_rate": _rate(r["correct"] for r in answerable),
        "correct_count": sum(bool(r["correct"]) for r in answerable),
        "answerable": len(answerable),
        "abstained_when_unanswerable": _rate(r["correct"] for r in refusals),
        "abstain_count": sum(bool(r["correct"]) for r in refusals),
        "unanswerable": len(refusals),
        "wrongly_abstained": sum(bool(r["abstained"]) for r in answerable),
        "citation_coverage": _mean(r["citation_coverage"] for r in answerable),
        "grounded_rate": _rate(r["grounded"] for r in answerable),
        "invalid_citation_answers": sum(bool(r["invalid_citations"]) for r in scored),
        "unverified_figure_answers": sum(bool(r["unverified_figures"]) for r in scored),
        "cited_from_gold": _mean(r["cited_from_gold"] for r in answerable),
        "latency_s": _mean(r["latency_s"] for r in scored),
        "retrieve_s": _mean(r["retrieve_s"] for r in scored),
        "generate_s": _mean(r["generate_s"] for r in scored),
        "input_tokens": _mean(r["input_tokens"] for r in scored),
        "output_tokens": _mean(r["output_tokens"] for r in scored),
    }
