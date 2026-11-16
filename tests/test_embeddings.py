from __future__ import annotations

import sys

import numpy as np
import pytest

from reglens.embeddings import HashingEmbedder, SentenceTransformerEmbedder, is_hashing, make_embedder, normalize_rows
from reglens.embeddings.base import Embedder, EmbeddingError
from reglens.embeddings.hashing import _slot
from reglens.embeddings.local import BGE_QUERY_PREFIX, default_prefixes, load_sentence_transformer

# ----------------------------------------------------------------------------------------------- normalisation


def test_normalize_rows_gives_unit_length_and_keeps_zero_rows():
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]]))
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 0.0, 1.0])
    assert not np.isnan(out).any()


def test_normalize_rows_accepts_a_single_vector_and_does_not_change_its_input():
    original = np.array([3.0, 4.0], dtype=np.float32)
    out = normalize_rows(original)
    assert out.shape == (1, 2)
    assert list(original) == [3.0, 4.0]


# ----------------------------------------------------------------------------------------------- hashing embedder


def test_hashing_embedder_is_deterministic_across_instances_and_processes():
    a = HashingEmbedder(256).embed_documents(["personal data breach notification"])
    b = HashingEmbedder(256).embed_documents(["personal data breach notification"])
    assert np.array_equal(a, b)


def test_hashing_slot_values_are_stable():
    # recorded once with blake2b(digest_size=8); they must never change, or saved indexes would silently break
    # (and Python's per-process hash() would fail this test immediately)
    assert [_slot(w, 512) for w in ("breach", "controller", "72")] == [(169, -1), (108, 1), (28, -1)]


def test_hashing_embedder_vectors_are_unit_length_and_reflect_word_overlap():
    emb = HashingEmbedder()
    docs = emb.embed_documents(
        [
            "the controller shall notify the supervisory authority of a personal data breach",
            "banana smoothie recipe with oats and honey",
            "notify the authority about a data breach",
        ]
    )
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    query = emb.embed_query("breach notification to the authority")
    scores = docs @ query
    assert scores[0] > scores[1] and scores[2] > scores[1]
    assert abs(scores[1]) < 0.2


def test_hashing_embedder_handles_empty_and_symbol_only_text():
    emb = HashingEmbedder(64)
    assert np.allclose(emb.embed_query(""), 0.0)
    assert np.allclose(emb.embed_query("!!! ???"), 0.0)
    assert emb.embed_documents([]).shape == (0, 64)


def test_hashing_embedder_progress_and_validation():
    seen = []
    HashingEmbedder(64).embed_documents(["a b"] * 600, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (600, 600) and seen[0] == (256, 600)
    with pytest.raises(ValueError):
        HashingEmbedder(4)


def test_make_embedder():
    assert make_embedder("hashing").dim == 512
    assert make_embedder("hashing-128").dim == 128
    assert make_embedder("hashing-xyz").name == "hashing-512"
    assert isinstance(make_embedder("BAAI/bge-small-en-v1.5"), SentenceTransformerEmbedder)


def test_is_hashing_recognises_only_the_baseline_names():
    assert is_hashing("hashing") and is_hashing("hashing-256") and is_hashing("hashing-xyz")
    assert not is_hashing("BAAI/bge-small-en-v1.5")
    assert not is_hashing("hashingx") and not is_hashing("my-hashing-model") and not is_hashing("")
    assert isinstance(HashingEmbedder(), Embedder)


# ----------------------------------------------------------------------------------------------- sentence-transformers


class FakeModel:
    """Embeds a text as [length, first-letter code, 1, 0 ...]: different texts give different rows."""

    max_seq_length = 512

    def __init__(self, scale: float = 1.0):
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.scale = scale

        class Tokenizer:
            verbose_seen: list[bool] = []

            def __call__(inner, texts, add_special_tokens=True, truncation=False, verbose=True):
                inner.verbose_seen.append(verbose)
                return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}

        self.tokenizer = Tokenizer()
        self.tokenizer.verbose_seen = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.kwargs.append(kwargs)
        rows = [[len(t) * self.scale, ord(t[0]) if t else 0, 1.0, 0.5] for t in texts]
        return np.array(rows, dtype=np.float32)


def embedder_with(model, **options):
    loads = []

    def loader(name, device):
        loads.append((name, device))
        return model

    return SentenceTransformerEmbedder("some/model", loader=loader, **options), loads


def test_prefix_defaults_by_model_family():
    assert default_prefixes("BAAI/bge-small-en-v1.5") == (BGE_QUERY_PREFIX, "")
    assert default_prefixes("BAAI/bge-base-en-v1.5")[0] == BGE_QUERY_PREFIX
    assert default_prefixes("intfloat/e5-small-v2") == ("query: ", "passage: ")
    assert default_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    assert default_prefixes("BAAI/bge-m3") == ("", "")  # the multilingual BGE takes no instruction


def test_queries_get_the_query_prefix_and_passages_do_not():
    model = FakeModel()
    emb, _ = embedder_with(model, query_prefix="Q: ", document_prefix="P: ")
    emb.embed_query("what is x")
    emb.embed_documents(["alpha"])
    texts = [call for call in model.calls if "dimension probe" not in call[0]]
    assert texts == [["Q: what is x"], ["P: alpha"]]


def test_the_model_is_loaded_lazily_and_only_once():
    model = FakeModel()
    emb, loads = embedder_with(model)
    assert loads == []
    emb.embed_query("a")
    emb.embed_documents(["b", "c"])
    assert emb.dim == 4
    assert loads == [("some/model", None)]


def test_documents_come_back_in_input_order_although_they_are_embedded_by_length():
    model = FakeModel()
    emb, _ = embedder_with(model, document_prefix="")
    texts = ["a" * 50, "b", "c" * 20, "d" * 7]
    out = emb.embed_documents(texts)
    reference = normalize_rows(np.array([[len(t), ord(t[0]), 1.0, 0.5] for t in texts], dtype=np.float32))
    assert np.allclose(out, reference)


def test_vectors_are_normalised_even_if_the_model_does_not_do_it():
    emb, _ = embedder_with(FakeModel(scale=100.0))
    out = emb.embed_documents(["abc", "defg"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(emb.embed_query("hello")), 1.0, atol=1e-5)


def test_encode_is_called_with_explicit_options():
    model = FakeModel()
    emb, _ = embedder_with(model, batch_size=7)
    emb.embed_documents(["x"])
    assert model.kwargs[-1] == {"batch_size": 7, "convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": False}


def test_progress_is_reported_in_groups():
    emb, _ = embedder_with(FakeModel())
    seen = []
    emb.embed_documents(["t%d" % i for i in range(150)], progress=lambda d, t: seen.append((d, t)))
    assert seen == [(64, 150), (128, 150), (150, 150)]


def test_empty_input():
    emb, _ = embedder_with(FakeModel())
    assert emb.embed_documents([]).shape == (0, 4)


def test_model_failures_become_embedding_errors():
    class Broken(FakeModel):
        def encode(self, texts, **kwargs):
            raise RuntimeError("out of memory")

    emb, _ = embedder_with(Broken())
    with pytest.raises(EmbeddingError, match="out of memory"):
        emb.embed_query("x")


@pytest.mark.parametrize("bad", [np.full((1, 4), np.nan, dtype=np.float32), np.zeros((3, 4), dtype=np.float32)])
def test_unusable_results_are_rejected(bad):
    class Odd(FakeModel):
        def encode(self, texts, **kwargs):
            return np.array(bad)

    emb, _ = embedder_with(Odd())
    with pytest.raises(EmbeddingError):
        emb.embed_query("x")


def test_token_counts_and_window():
    emb, _ = embedder_with(FakeModel(), document_prefix="")
    assert emb.max_tokens == 512
    assert emb.token_counts(["one two three", "four"]) == [5, 3]


def test_token_counting_does_not_let_the_tokenizer_print_its_own_length_warning():
    model = FakeModel()
    emb, _ = embedder_with(model, document_prefix="")
    emb.token_counts(["one two three"])
    assert model.tokenizer.verbose_seen == [False]


def test_token_counts_without_a_tokenizer():
    model = FakeModel()
    model.tokenizer = None
    model.max_seq_length = None
    emb, _ = embedder_with(model)
    assert emb.token_counts(["a"]) is None and emb.max_tokens is None


def test_loading_without_the_library_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes "import sentence_transformers" fail
    with pytest.raises(EmbeddingError, match="pip install"):
        load_sentence_transformer("x")


def test_loading_failure_explains_the_download(monkeypatch):
    class FakeModule:
        @staticmethod
        def SentenceTransformer(name, device=None):  # noqa: N802 - mimics the library's class name
            raise OSError("offline")

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeModule)
    with pytest.raises(EmbeddingError, match="huggingface.co"):
        load_sentence_transformer("BAAI/bge-small-en-v1.5")
