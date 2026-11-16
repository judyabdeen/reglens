"""The result of one question: the answer, the sources behind it and everything measured on the way."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from reglens.generation.prompts import page_text, status_text
from reglens.retrieval.context import ContextItem

RAG = "rag"  # retrieval + language model
EXTRACTIVE = "extractive"  # retrieval only: the language model was not used or not available
LLM_ONLY = "llm_only"  # the language model alone, without retrieval (the baseline of the comparison)


@dataclass(frozen=True)
class Source:
    """One numbered passage that was put into the prompt (or shown in place of an answer)."""

    label: str
    chunk_id: str
    doc_id: str
    title: str
    publisher: str
    jurisdiction: str
    doc_type: str
    status: str
    version_date: str
    section: str
    page_start: int | None
    page_end: int | None
    kind: str  # "text" or "notes"
    url: str
    text: str
    rank: int
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    cited: bool = False
    dense_rank: int | None = None  # position in the meaning search (1 = best), None if that search did not find it
    lexical_rank: int | None = None  # position in the keyword search
    fused_score: float | None = None  # reciprocal-rank-fusion score of the two searches

    @property
    def status_text(self) -> str:
        return status_text(self.status)

    @property
    def pages(self) -> str:
        return page_text(self.page_start, self.page_end)

    @property
    def location(self) -> str:
        """Where in the document the passage is, for a citation line."""
        return ", ".join(part for part in (self.section, self.pages) if part)

    @property
    def is_law(self) -> bool:
        """False for drafts, consultations and proposals: texts that are not (yet) binding."""
        return self.status in ("in_force", "final")

    @classmethod
    def from_item(cls, item: ContextItem, *, cited: bool = False) -> Source:
        r, m = item.retrieved, item.retrieved.metadata
        return cls(
            label=item.label,
            chunk_id=r.id,
            doc_id=str(m.get("doc_id", "")),
            title=str(m.get("title", m.get("doc_id", r.id))),
            publisher=str(m.get("publisher", "")),
            jurisdiction=str(m.get("jurisdiction", "")),
            doc_type=str(m.get("doc_type", "")),
            status=str(m.get("status", "")),
            version_date=str(m.get("version_date", "")),
            section=str(m.get("section", "")),
            page_start=m.get("page_start"),
            page_end=m.get("page_end"),
            kind=str(m.get("kind", "text")),
            url=str(m.get("url", "")),
            text=r.text,
            rank=r.rank,
            score=r.score,
            dense_score=r.dense_score,
            lexical_score=r.lexical_score,
            rerank_score=r.rerank_score,
            cited=cited,
            dense_rank=r.dense_rank,
            lexical_rank=r.lexical_rank,
            fused_score=r.fused_score,
        )


@dataclass(frozen=True)
class Answer:
    question: str
    standalone_question: str
    mode: str
    text: str
    abstained: bool = False  # the model said the sources do not answer the question
    sources: tuple[Source, ...] = ()
    cited: tuple[str, ...] = ()  # labels the answer cites
    invalid_citations: tuple[str, ...] = ()  # cited labels that do not exist
    unverified_figures: tuple[str, ...] = ()  # figures in the answer that are not in any source
    citation_coverage: float | None = None  # share of factual sentences that carry a citation
    uncited_sentences: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    fell_back_from: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    plan_notes: tuple[str, ...] = ()  # what query transformation did
    notes: tuple[str, ...] = ()  # warnings for the user
    error: str | None = None  # why the language model was not used
    error_kind: str | None = None  # an LLMError kind (quota, auth, ...) or "unavailable"; lets the interface pick a hint
    timings: dict[str, float] = field(default_factory=dict)  # seconds: transform, retrieve, generate, total
    options: dict = field(default_factory=dict)  # the settings of this run, for the evaluation log

    def to_dict(self) -> dict:
        data = asdict(self)
        for source in data["sources"]:
            source["status_text"] = status_text(source["status"])
            source["pages"] = page_text(source["page_start"], source["page_end"])
        return data

    @property
    def cited_sources(self) -> tuple[Source, ...]:
        return tuple(s for s in self.sources if s.cited)
