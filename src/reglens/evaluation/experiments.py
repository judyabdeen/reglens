"""The experiments of assignment sections 5.1-5.3, run on the real index.

Retrieval experiments need no language model: they feed every gold question to the retriever under different settings
and score what comes back. Generation experiments call the language model (paced, resumable) and score the answers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from reglens.corpus.manifest import Source
from reglens.embeddings.base import Embedder
from reglens.evaluation.gold import GoldQuestion
from reglens.generation.citations import strip_citations
from reglens.generation.grounding import unverified_figures
from reglens.evaluation.metrics import KS, Passage, is_grounded, score_answer, score_retrieval, summarise_answers, summarise_retrieval
from reglens.ingest.chunking import Chunk
from reglens.pipeline import AskOptions, RAGPipeline
from reglens.retrieval.query_transform import QueryTransformer, TransformOptions
from reglens.retrieval.rerank import RerankError, Reranker
from reglens.retrieval.retriever import RetrievalError, RetrievalOptions, Retriever
from reglens.vectorstore.numpy_store import NumpyStore
from reglens.vectorstore.records import make_records

DEPTH = max(KS)  # passages retrieved per question (the largest k that is reported)
FATAL_KINDS = ("auth", "forbidden", "quota", "not_found", "unavailable")  # more calls would only fail the same way


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    label: str
    mode: str = "hybrid"
    rerank: bool = False
    glossary: bool = True
    include_notes: bool = True
    multi_query: int = 0
    hyde: bool = False
    index: str = "main"  # which index: "main", or the name of a variant (hashing embeddings, fixed chunks)
    needs_llm: bool = False
    group: str = "search"  # the question the row answers: search | rerank | query | notes | embedding | chunking


MAIN_CONFIGS = (
    RetrievalConfig("lexical", "Keywords only (BM25)", mode="lexical", group="search"),
    RetrievalConfig("dense", "Meaning only (embeddings)", mode="dense", group="search"),
    RetrievalConfig("hybrid", "Hybrid: meaning + keywords (RRF)", group="search"),
    RetrievalConfig("hybrid_rerank", "Hybrid + cross-encoder re-ranking", rerank=True, group="rerank"),
    RetrievalConfig("hybrid_no_glossary", "Hybrid without the glossary", glossary=False, group="query"),
    RetrievalConfig("hybrid_multiquery", "Hybrid + 2 paraphrases (multi-query)", multi_query=2, needs_llm=True, group="query"),
    RetrievalConfig("hybrid_hyde", "Hybrid + hypothetical answer (HyDE)", hyde=True, needs_llm=True, group="query"),
    RetrievalConfig("hybrid_no_notes", "Hybrid without footnote passages", include_notes=False, group="notes"),
)


def variant_config(base: RetrievalConfig, index: str, label: str, group: str) -> RetrievalConfig:
    return replace(base, name=f"{base.name}@{index}", label=label, index=index, group=group)


# ------------------------------------------------------------------------------------------------ retrieval


def memory_retriever(chunks: Sequence[Chunk], sources: Mapping[str, Source], embedder: Embedder, reranker: Reranker | None = None,
                     progress: Callable[[int, int], None] | None = None) -> Retriever:
    """An in-memory index of ``chunks`` (nothing is written to disk): the basis of the embedding and chunking comparisons."""
    records = make_records(chunks, sources)
    vectors = np.asarray(embedder.embed_documents([c.embed_text for c in chunks], progress), dtype=np.float32)
    store = NumpyStore()
    store.add(records, vectors)
    return Retriever.from_store(store, embedder, reranker)


def run_retrieval(retriever: Retriever, questions: Sequence[GoldQuestion], config: RetrievalConfig, llm=None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Retrieve for every answerable question; returns the per-question scores and their summary (or a reason it was skipped)."""
    answerable = [q for q in questions if q.answerable]
    if config.needs_llm and llm is None:
        return {"config": config.name, "label": config.label, "group": config.group, "skipped": "needs a language model"}
    transformer = QueryTransformer(llm if config.needs_llm else None)
    transform = TransformOptions(glossary=config.glossary, condense=False, multi_query=config.multi_query, hyde=config.hyde)
    options = RetrievalOptions(mode=config.mode, top_k=DEPTH, candidates=40, rerank=config.rerank, rerank_pool=20, include_notes=config.include_notes)
    rows, seconds = [], []
    for done, question in enumerate(answerable, 1):
        started = time.perf_counter()
        try:
            plan = transformer.plan(question.question, (), transform)
            found = retriever.retrieve(plan, options)
        except (RerankError, RetrievalError) as exc:
            return {"config": config.name, "label": config.label, "group": config.group, "skipped": str(exc)}
        seconds.append(time.perf_counter() - started)
        passages = [Passage(r.id, str(r.metadata.get("doc_id", "")), r.text) for r in found]
        score = score_retrieval(question, passages)
        score.update(id=question.id, category=question.category, top=[p.chunk_id for p in passages[:5]], llm_calls=plan.llm_calls)
        rows.append(score)
        if progress:
            progress(done, len(answerable))
    summary = summarise_retrieval(rows)
    summary["seconds_per_question"] = sum(seconds) / len(seconds) if seconds else None
    summary["by_category"] = {
        cat: summarise_retrieval([r for r in rows if r["category"] == cat]) for cat in sorted({r["category"] for r in rows})
    }
    return {"config": config.name, "label": config.label, "group": config.group, "summary": summary, "questions": rows}


def store_agreement(first: Retriever, second: Retriever, questions: Sequence[GoldQuestion], k: int = 8) -> dict:
    """How closely two vector stores (Chroma and the exact NumPy search) agree on the first ``k`` passages of every question."""
    options = RetrievalOptions(mode="dense", top_k=k, candidates=max(k, 40))
    overlaps, same_top = [], []
    for question in (q for q in questions if q.answerable):
        a = [r.id for r in first.retrieve(question.question, options)]
        b = [r.id for r in second.retrieve(question.question, options)]
        overlaps.append(len(set(a) & set(b)) / max(len(a), 1))
        same_top.append(bool(a and b and a[0] == b[0]))
    n = len(overlaps)
    return {"questions": n, "k": k, "mean_overlap": sum(overlaps) / n if n else None, "same_first_passage": sum(same_top) / n if n else None}


# ------------------------------------------------------------------------------------------------ generation


@dataclass(frozen=True)
class GenerationConfig:
    name: str
    label: str
    kind: str  # "llm_only" or "rag"
    rerank: bool = False
    multi_query: int = 0


GENERATION_CONFIGS = {
    "llm_only": GenerationConfig("llm_only", "Language model alone (no retrieval)", "llm_only"),
    "rag": GenerationConfig("rag", "RegLens: hybrid retrieval + grounded prompt", "rag"),
    "rag_rerank": GenerationConfig("rag_rerank", "RegLens + re-ranking", "rag", rerank=True),
    "rag_multiquery": GenerationConfig("rag_multiquery", "RegLens + 2 paraphrases", "rag", multi_query=2),
}


def ask_options(config: GenerationConfig) -> AskOptions:
    return AskOptions(
        retrieval=RetrievalOptions(mode="hybrid", top_k=8, rerank=config.rerank),
        transform=TransformOptions(glossary=True, condense=False, multi_query=config.multi_query),
    )


def answer_one(pipeline: RAGPipeline, question: GoldQuestion, config: GenerationConfig):
    if config.kind == "llm_only":
        return pipeline.ask_llm_only(question.question)
    return pipeline.ask(question.question, options=ask_options(config))


def load_done(path: Path) -> dict[tuple[str, str], dict]:
    """Answers saved by an earlier run (so that a run that ran out of quota can be continued)."""
    done: dict[tuple[str, str], dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("score", {}).get("failed"):
                    done[(row["config"], row["id"])] = row
    return done


def run_generation(pipeline: RAGPipeline, questions: Sequence[GoldQuestion], configs: Iterable[GenerationConfig], path: Path,
                   progress: Callable[[str], None] = lambda message: None) -> tuple[list[dict], str | None]:
    """Ask every question under every configuration. Each answer is appended to ``path`` as soon as it exists.

    Returns (all rows, the reason the run stopped early or None). Questions already answered in ``path`` are not asked again.
    """
    done = load_done(path)
    rows: list[dict] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    stopped: str | None = None
    with path.open("a", encoding="utf-8") as sink:
        for config in configs:
            for number, question in enumerate(questions, 1):
                key = (config.name, question.id)
                if key in done:
                    rows.append(done[key])
                    continue
                if config.kind == "llm_only" and question.expect_abstain:
                    continue  # the baseline has no way to abstain, so it is only asked what the documents can answer
                progress(f"{config.name}  {question.id}  ({number}/{len(questions)})")
                answer = answer_one(pipeline, question, config)
                score = score_answer(question, answer)
                if config.kind == "llm_only":
                    # the baseline has no sources and no abstain marker: it is correct when it states the facts
                    score["grounded"] = None
                row = {
                    "config": config.name, "id": question.id, "category": question.category, "expect_abstain": question.expect_abstain,
                    "answer": answer.text, "cited": list(answer.cited), "sources": [s.chunk_id for s in answer.sources],
                    "source_texts": [s.text for s in answer.sources], "score": score, "notes": list(answer.notes),
                    "model": answer.model,
                }
                if not score["failed"]:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                rows.append(row)
                if score["failed"] and score["error_kind"] in FATAL_KINDS:
                    stopped = f"the language model reported {score['error_kind']} (see 'python scripts/check_setup.py'); the answers so far are saved - run the same command again to continue"
                    return rows, stopped
    return rows, stopped


def summarise_generation(rows: Sequence[dict]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for name in dict.fromkeys(r["config"] for r in rows):
        flat = [{**r["score"], "category": r["category"], "expect_abstain": r["expect_abstain"]} for r in rows if r["config"] == name]
        summary = summarise_answers(flat)
        summary["by_category"] = {}
        for cat in sorted({r["category"] for r in flat}):
            part = [r for r in flat if r["category"] == cat and not r["failed"] and not r["expect_abstain"]]
            if part:
                summary["by_category"][cat] = {"correct": sum(bool(r["correct"]) for r in part), "n": len(part)}
        summaries[name] = summary
    return summaries


def rescore(rows: Sequence[dict], questions: Sequence[GoldQuestion]) -> list[dict]:
    """Score the saved answers again against the current gold set, without asking the model anything.

    Used when a gold pattern turned out to be too narrow (it rejected a correct answer written in another form, such as
    "20,000,000" for "20 000 000"). Every configuration is re-scored with the same patterns. Returns the verdicts that
    changed, so that the report can show them instead of hiding the revision.
    """
    by_id = {q.id: q for q in questions}
    changed: list[dict] = []
    for row in rows:
        score, gold = row["score"], by_id.get(row["id"])
        if gold is None or score["failed"] or gold.expect_abstain:
            continue
        missing = list(range(len(gold.facts))) if score["abstained"] else gold.missing_facts(strip_citations(row["answer"]))
        correct = not missing and not score["abstained"]
        if correct != score["correct"]:
            changed.append({"config": row["config"], "id": row["id"], "before": bool(score["correct"]), "after": correct})
        score["correct"], score["missing_facts"] = correct, missing
    return changed


def recheck_figures(rows: Sequence[dict], questions: Sequence[GoldQuestion], headings: Mapping[str, str]) -> list[dict]:
    """Check the figures of the saved RegLens answers again, reading each source as the model saw it (heading and passage).

    ``headings`` maps a chunk id to the heading line shown above it (``reglens.generation.prompts.source_header``).
    Returns the answers whose list of unverified figures changed.
    """
    by_id = {q.id: q for q in questions}
    changed: list[dict] = []
    for row in rows:
        score, gold = row["score"], by_id.get(row["id"])
        if gold is None or score["failed"] or row["config"] == "llm_only":
            continue
        shown = [f"{headings.get(chunk_id, '')}\n{text}" for chunk_id, text in zip(row["sources"], row["source_texts"])]
        figures = unverified_figures(row["answer"], shown, gold.question)
        if figures != list(score["unverified_figures"]):
            changed.append({"config": row["config"], "id": row["id"], "before": list(score["unverified_figures"]), "after": figures})
        score["unverified_figures"] = figures
        score["grounded"] = is_grounded("rag", bool(score["abstained"]), score["citation_coverage"], score["invalid_citations"], figures) if not score["abstained"] else None
    return changed
