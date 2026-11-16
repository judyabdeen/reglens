"""Embedding models: turn text into unit-length vectors (assignment section 2.2)."""

from reglens.embeddings.base import Embedder, EmbeddingError, normalize_rows
from reglens.embeddings.hashing import HashingEmbedder
from reglens.embeddings.local import SentenceTransformerEmbedder

__all__ = ["Embedder", "EmbeddingError", "HashingEmbedder", "SentenceTransformerEmbedder", "is_hashing", "make_embedder", "normalize_rows"]


def is_hashing(model_name: str) -> bool:
    """True for the dependency-free baseline (``hashing`` or ``hashing-256``), which needs no download."""
    return model_name == "hashing" or model_name.startswith("hashing-")


def make_embedder(model_name: str, **options) -> Embedder:
    """``hashing`` or ``hashing-256`` gives the dependency-free embedder; anything else is a Hugging Face model name."""
    if is_hashing(model_name):
        _, _, size = model_name.partition("-")
        return HashingEmbedder(int(size)) if size.isdigit() else HashingEmbedder()
    return SentenceTransformerEmbedder(model_name, **options)
