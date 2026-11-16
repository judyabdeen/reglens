"""A small invented corpus for testing retrieval, generation and the command line without the real documents.

The texts are synthetic and shortened; they only imitate the *shape* of the real regulations (an original act, an
amending act, guidance with a different legal status, a US rule). The two dates in the AI Act pair are the ones of the
flagship scenario of the project.
"""

from __future__ import annotations

import re

import numpy as np

from reglens.corpus.manifest import Source
from reglens.embeddings.base import normalize_rows
from reglens.ingest.chunking import Chunk


def source(id_, title, jurisdiction, doc_type, status, version="2024", publisher="Publisher", fmt="pdf"):
    return Source(
        id=id_, title=title, publisher=publisher, jurisdiction=jurisdiction, doc_type=doc_type, status=status,
        version_date=version, priority="1", fetch="auto", format=fmt, url=f"https://example.org/{id_}.pdf",
        landing_url=f"https://example.org/{id_}", filename=f"{id_}.pdf", verification="test", notes="",
    )


SOURCES = [
    source("gdpr", "General Data Protection Regulation (GDPR)", "EU", "regulation", "in_force", "2016"),
    source("ai-act", "AI Act (as originally published)", "EU", "regulation", "in_force", "2024"),
    source("omnibus", "Digital Omnibus on AI amending the AI Act", "EU", "regulation", "in_force", "2026"),
    source("draft-hr", "Draft guidelines on high-risk classification", "EU", "guidelines", "draft", "2026"),
    source("hhs-breach", "HHS Breach Notification Rule", "US", "web_guidance", "final", "2024", fmt="html"),
    source("ccpa", "California Consumer Privacy Act", "US", "statute", "in_force", "2023"),
]

# (doc_id, section_path, page_start, page_end, text)
_RAW = [
    ("gdpr", ["Chapter IV", "Article 33 - Notification of a personal data breach to the supervisory authority"], 52, 52,
     "In the case of a personal data breach, the controller shall without undue delay and, where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to the supervisory authority."),
    ("gdpr", ["Chapter III", "Article 17 - Right to erasure"], 43, 44,
     "The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay and the controller shall have the obligation to erase personal data without undue delay where the data are no longer necessary."),
    ("gdpr", ["Chapter III", "Article 22 - Automated individual decision-making, including profiling"], 46, 46,
     "The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her."),
    ("ai-act", ["Chapter III", "Article 6 - Classification rules for high-risk AI systems"], 60, 61,
     "An AI system shall be considered to be high-risk where it is intended to be used as a safety component of a product or where it is referred to in Annex III, such as AI systems used for recruitment or selection of natural persons."),
    ("ai-act", ["Chapter XIII", "Article 113 - Entry into force and application"], 140, 141,
     "This Regulation shall apply from 2 August 2026. Chapter III obligations for high-risk AI systems shall therefore apply from 2 August 2026."),
    ("omnibus", ["Article 1", "Amendments to Regulation (EU) 2024/1689"], 10, 11,
     "In Article 113, the obligations for high-risk AI systems referred to in Annex III shall apply from 2 December 2027 and the obligations for high-risk AI systems referred to in Annex I shall apply from 2 August 2028."),
    ("draft-hr", ["Section 3", "Employment"], 21, 21,
     "Draft guidance: AI systems that rank or filter job applicants are likely to fall under Annex III point 4 on employment and workers management."),
    ("hhs-breach", ["Breach Notification Rule"], None, None,
     "Following a breach of unsecured protected health information, covered entities must notify affected individuals without unreasonable delay and in no case later than 60 days after discovery of the breach."),
    ("ccpa", ["1798.120"], 12, 12,
     "A consumer shall have the right, at any time, to direct a business that sells or shares personal information about the consumer to third parties not to sell or share the consumer's personal information."),
    ("ccpa", ["1798.105"], 9, 9,
     "A consumer shall have the right to request that a business delete any personal information about the consumer which the business has collected from the consumer."),
]


def chunks() -> list[Chunk]:
    by_doc = {s.id: s for s in SOURCES}
    counters: dict[str, int] = {}
    out = []
    for doc_id, path, p1, p2, text in _RAW:
        counters[doc_id] = counters.get(doc_id, 0) + 1
        crumb = " > ".join(path[-2:])
        prefix = f"{by_doc[doc_id].title} | {crumb}"
        out.append(
            Chunk(
                chunk_id=f"{doc_id}:{counters[doc_id]:04d}", doc_id=doc_id, kind="text", ordinal=counters[doc_id], text=text,
                embed_text=f"{prefix}\n{text}", section_path=list(path), page_start=p1, page_end=p2, n_words=len(text.split()),
            )
        )
    return out


def by_source() -> dict[str, Source]:
    return {s.id: s for s in SOURCES}


# --------------------------------------------------------------------------------------------- a "semantic" embedder
# Words that mean the same thing share one axis, so "leak" and "breach" are close although they share no letters.
CONCEPTS = [
    ("breach", "leak", "hack", "incident"),
    ("authority", "regulator", "watchdog"),
    ("notify", "report", "inform", "tell"),
    ("erase", "erasure", "delete", "forget"),
    ("applicant", "recruitment", "resume", "candidate", "hiring", "job"),
    ("high-risk", "annex"),
    ("sell", "sells", "sale", "shares"),
    ("health", "patient", "medical", "hipaa"),
]


class ConceptEmbedder:
    """Maps text to the number of words of each concept (plus a little noise-free identity), then normalises."""

    name = "concepts-test"

    def __init__(self) -> None:
        self.dim = len(CONCEPTS) + 1
        self.queries: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z0-9-]+", text.lower())
        vector = np.zeros(self.dim, dtype=np.float32)
        for i, group in enumerate(CONCEPTS):
            vector[i] = sum(w in group for w in words)
        vector[-1] = 0.05  # keeps every vector non-zero
        return vector

    def embed_documents(self, texts, progress=None):
        rows = normalize_rows(np.array([self._vector(t) for t in texts], dtype=np.float32)) if len(texts) else np.zeros((0, self.dim), dtype=np.float32)
        if progress:
            progress(len(texts), len(texts))
        return rows

    def embed_query(self, text):
        self.queries.append(text)
        return normalize_rows(self._vector(text))[0]


def build(embedder=None):
    """(store, retriever, embedder, records) over the invented corpus, using an exact in-memory store."""
    from reglens.retrieval import Retriever
    from reglens.vectorstore import NumpyStore
    from reglens.vectorstore.records import make_records

    embedder = embedder or ConceptEmbedder()
    cs = chunks()
    records = make_records(cs, by_source())
    store = NumpyStore()
    store.add(records, embedder.embed_documents([c.embed_text for c in cs]))
    return store, Retriever(store, embedder, store.records()), embedder, records


# --------------------------------------------------------------------------------------------- a fake language model


class FakeLLM:
    """Replies with scripted texts (or the result of a function of the request) and records every request."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, *replies, truncated=False, fell_back_from=None):
        self.replies = list(replies)
        self.requests = []
        self.truncated = truncated
        self.fell_back_from = fell_back_from

    def generate(self, request):
        from reglens.llm.base import LLMResponse, Usage

        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        text = reply(request) if callable(reply) else reply
        return LLMResponse(
            text=text, provider=self.provider, model=self.model, finish_reason="STOP", usage=Usage(120, 40), latency_s=0.01,
            truncated=self.truncated, fell_back_from=self.fell_back_from,
        )
