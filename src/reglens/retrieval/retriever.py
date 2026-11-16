"""The retriever (assignment section 2.4): from a query plan to the best-matching chunks.

Hybrid retrieval, step by step:

1. **Dense search** - embed the query and ask the vector store for the ``candidates`` nearest chunks (finds passages with
   the same *meaning*, whatever the wording).
2. **Keyword search** - rank all chunks with BM25 (finds the *exact words*: "Annex III", "2026/1744").
3. **Fusion** - merge the two ranked lists with reciprocal rank fusion.
4. **Re-ranking** (optional) - a cross-encoder re-scores the best ``rerank_pool`` candidates.
5. The top ``top_k`` chunks are returned, each with the scores that produced its place.

The same class runs the single-method modes ("dense", "lexical") that the evaluation compares against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from reglens.embeddings.base import Embedder
from reglens.retrieval.bm25 import BM25Index
from reglens.retrieval.fusion import reciprocal_rank_fusion
from reglens.retrieval.query_transform import QueryPlan
from reglens.retrieval.rerank import Reranker
from reglens.vectorstore.base import Record, VectorStore
from reglens.vectorstore.records import lexical_text

MODES = ("hybrid", "dense", "lexical")


class RetrievalError(RuntimeError):
    """A retrieval option cannot be honoured (for example re-ranking without a re-ranker)."""


@dataclass(frozen=True)
class RetrievalOptions:
    mode: str = "hybrid"
    top_k: int = 8  # chunks handed to the language model
    candidates: int = 40  # taken from each search before fusion
    rerank: bool = False
    rerank_pool: int = 20  # how many fused candidates the re-ranker re-scores
    where: Mapping | None = None  # metadata filter, e.g. {"jurisdiction": "EU"}
    include_notes: bool = True  # footnote chunks take part in the search
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        if self.top_k < 1 or self.candidates < 1 or self.rerank_pool < 1:
            raise ValueError("top_k, candidates and rerank_pool must be at least 1")
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("weights must not be negative")


@dataclass(frozen=True)
class Retrieved:
    """A retrieved chunk with the evidence for its position."""

    record: Record
    rank: int  # final position, 1 = best
    score: float  # what the final order is based on: re-rank score, else fused score, else the search's own score
    dense_score: float | None = None  # cosine similarity to the query
    dense_rank: int | None = None
    lexical_score: float | None = None  # BM25
    lexical_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def text(self) -> str:
        return self.record.text

    @property
    def metadata(self) -> Mapping:
        return self.record.metadata


def _ordinal(record: Record) -> int:
    value = record.metadata.get("ordinal")
    return value if isinstance(value, int) else 0


@dataclass
class _Evidence:
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    fused: float | None = None
    rerank: float | None = None
    record: Record | None = field(default=None, repr=False)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        records: Sequence[Record],
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._catalog = {r.id: r for r in records}
        self._lexical = BM25Index.from_records(records)
        self._order: dict[tuple[str, str], list[Record]] | None = None  # passages of each document in reading order (built on demand)

    @classmethod
    def from_store(cls, store: VectorStore, embedder: Embedder, reranker: Reranker | None = None) -> Retriever:
        """Read every record out of the vector store (once) to build the keyword index."""
        return cls(store, embedder, store.records(), reranker)

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def __len__(self) -> int:
        return len(self._catalog)

    def documents(self) -> list[tuple[str, str, int]]:
        """(doc_id, title, number of passages) of every indexed document, sorted by id."""
        found: dict[str, list] = {}
        for record in self._catalog.values():
            doc_id = str(record.metadata.get("doc_id", ""))
            entry = found.setdefault(doc_id, [str(record.metadata.get("title", doc_id)), 0])
            entry[1] += 1
        return sorted((doc_id, title, count) for doc_id, (title, count) in found.items())

    def catalog(self) -> list[dict]:
        """One entry per indexed document: its metadata (read from its passages) and its number of passages, sorted by id."""
        found: dict[str, dict] = {}
        for record in self._catalog.values():
            meta = record.metadata
            doc_id = str(meta.get("doc_id", ""))
            entry = found.setdefault(doc_id, {"doc_id": doc_id, "passages": 0})
            entry["passages"] += 1
            for key in ("title", "publisher", "jurisdiction", "doc_type", "status", "version_date", "url"):
                if key not in entry and meta.get(key) not in (None, ""):
                    entry[key] = str(meta[key])
        for entry in found.values():
            entry.setdefault("title", entry["doc_id"])
        return sorted(found.values(), key=lambda e: e["doc_id"])

    def record(self, chunk_id: str) -> Record | None:
        return self._catalog.get(chunk_id)

    def neighbours(self, chunk_id: str) -> tuple[Record | None, Record | None]:
        """The passages just before and just after ``chunk_id`` in the same document (of the same kind), if they exist."""
        record = self._catalog.get(chunk_id)
        if record is None:
            return None, None
        if self._order is None:
            groups: dict[tuple[str, str], list[Record]] = {}
            for each in self._catalog.values():
                groups.setdefault((str(each.metadata.get("doc_id", "")), str(each.metadata.get("kind", "text"))), []).append(each)
            self._order = {
                key: sorted(members, key=lambda r: (_ordinal(r), r.id)) for key, members in groups.items()
            }
        members = self._order[(str(record.metadata.get("doc_id", "")), str(record.metadata.get("kind", "text")))]
        at = next(i for i, each in enumerate(members) if each.id == chunk_id)
        return (members[at - 1] if at > 0 else None, members[at + 1] if at + 1 < len(members) else None)

    def _embed(self, text: str, as_passage: bool):
        return self._embedder.embed_documents([text])[0] if as_passage else self._embedder.embed_query(text)

    def retrieve(self, query: QueryPlan | str, options: RetrievalOptions = RetrievalOptions()) -> list[Retrieved]:
        plan = QueryPlan.plain(query) if isinstance(query, str) else query
        if options.rerank and self._reranker is None:
            raise RetrievalError("re-ranking was requested, but no re-ranking model is loaded")
        where = dict(options.where or {})
        if not options.include_notes:
            where.setdefault("kind", "text")
        where_or_none = where or None

        evidence: dict[str, _Evidence] = {}
        rankings: list[list[str]] = []
        weights: list[float] = []

        if options.mode in ("hybrid", "dense"):
            share = options.dense_weight / len(plan.dense)  # paraphrases together weigh as much as one dense query
            for dense_query in plan.dense:
                hits = self._store.query(self._embed(dense_query.text, dense_query.as_passage), options.candidates, where_or_none)
                for rank, hit in enumerate(hits, start=1):
                    info = evidence.setdefault(hit.id, _Evidence(record=self._catalog.get(hit.id, hit.record)))
                    if info.dense_score is None or hit.score > info.dense_score:
                        info.dense_score = hit.score
                    if info.dense_rank is None or rank < info.dense_rank:
                        info.dense_rank = rank
                rankings.append([h.id for h in hits])
                weights.append(share)

        if options.mode in ("hybrid", "lexical"):
            found = self._lexical.search(plan.lexical, options.candidates, where_or_none)
            for rank, (rid, score) in enumerate(found, start=1):
                info = evidence.setdefault(rid, _Evidence(record=self._catalog[rid]))
                info.lexical_score, info.lexical_rank = score, rank
            rankings.append([rid for rid, _ in found])
            weights.append(options.lexical_weight)

        fused = reciprocal_rank_fusion([r for r in rankings if r], k=options.rrf_k, weights=[w for r, w in zip(rankings, weights, strict=True) if r])
        used_lists = sum(1 for r in rankings if r)
        for rid, score in fused:
            evidence[rid].fused = score

        pool = options.rerank_pool if options.rerank else options.top_k
        ordered = [rid for rid, _ in fused[: max(pool, options.top_k)]]
        if options.rerank:
            ordered = ordered[: options.rerank_pool]
            scores = self._reranker.score(plan.standalone, [lexical_text(evidence[rid].record) for rid in ordered])  # type: ignore[union-attr]
            for rid, value in zip(ordered, scores, strict=True):
                evidence[rid].rerank = value
            ordered = sorted(ordered, key=lambda rid: -evidence[rid].rerank)  # stable: equal scores keep the fused order
        ordered = ordered[: options.top_k]

        results = []
        for rank, rid in enumerate(ordered, start=1):
            info = evidence[rid]
            if info.rerank is not None:
                score = info.rerank
            elif used_lists > 1:
                score = info.fused  # type: ignore[assignment]
            elif info.dense_score is not None:  # one non-empty list only: the score of that search
                score = info.dense_score
            elif info.lexical_score is not None:
                score = info.lexical_score
            else:
                score = info.fused  # type: ignore[assignment]
            results.append(
                Retrieved(
                    record=info.record,  # type: ignore[arg-type]
                    rank=rank,
                    score=float(score),
                    dense_score=info.dense_score,
                    dense_rank=info.dense_rank,
                    lexical_score=info.lexical_score,
                    lexical_rank=info.lexical_rank,
                    fused_score=info.fused if used_lists > 1 else None,
                    rerank_score=info.rerank,
                )
            )
        return results
