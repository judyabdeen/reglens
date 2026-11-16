"""Retrieval and orchestration (assignment section 2.4): query transformation, hybrid search, re-ranking, context."""

from reglens.retrieval.bm25 import BM25Index
from reglens.retrieval.context import Context, ContextItem, assemble_context
from reglens.retrieval.fusion import reciprocal_rank_fusion
from reglens.retrieval.query_transform import DenseQuery, QueryPlan, QueryTransformer, TransformOptions, Turn
from reglens.retrieval.rerank import CrossEncoderReranker, Reranker, RerankError
from reglens.retrieval.retriever import MODES, RetrievalError, RetrievalOptions, Retrieved, Retriever

__all__ = [
    "BM25Index",
    "Context",
    "ContextItem",
    "CrossEncoderReranker",
    "DenseQuery",
    "MODES",
    "QueryPlan",
    "QueryTransformer",
    "RerankError",
    "Reranker",
    "RetrievalError",
    "RetrievalOptions",
    "Retrieved",
    "Retriever",
    "TransformOptions",
    "Turn",
    "assemble_context",
    "reciprocal_rank_fusion",
]
