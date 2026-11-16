from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import ragfixtures
from fakechroma import FakeChromaClient

from reglens.embeddings import HashingEmbedder
from reglens.indexing import (
    INFO_FORMAT,
    IndexInfo,
    IndexingError,
    IndexNotReady,
    build_index,
    file_sha256,
    info_path,
    library_versions,
    open_index,
    read_info,
)
from reglens.ingest.chunking import Chunk
from reglens.vectorstore import ChromaStore, NumpyStore, open_store


def built(tmp_path, *, backend="numpy", embedder=None, chunks=None, **options):
    """Build the invented corpus into ``tmp_path`` the way scripts/build_index.py does."""
    embedder = embedder or HashingEmbedder()
    store = open_store(tmp_path, backend)
    info = build_index(
        chunks if chunks is not None else ragfixtures.chunks(), ragfixtures.by_source(), embedder, store,
        info_file=info_path(tmp_path, backend), backend=backend, **options,
    )  # fmt: skip
    return store, info, embedder


# ----------------------------------------------------------------------------------------------- building


def test_a_build_stores_every_chunk_with_its_metadata_and_describes_itself(tmp_path):
    store, info, _ = built(tmp_path, chunks_file="chunks.jsonl", chunks_sha256="abc123")
    assert store.count() == 10 == info.n_records
    assert (info.n_documents, info.n_text, info.n_notes) == (6, 10, 0)
    assert (info.embedding_model, info.dim, info.backend, info.space) == ("hashing-512", 512, "numpy", "cosine")
    assert info.format == INFO_FORMAT and info.chunks_file == "chunks.jsonl" and info.chunks_sha256 == "abc123"
    assert info.built_at.endswith("Z") and set(info.seconds) == {"embedding", "storing", "total"}
    assert "numpy" in info.libraries and info.token_limit is None and info.over_token_limit is None
    record = store.get(["omnibus:0001"])[0]
    assert record.metadata["title"] == "Digital Omnibus on AI amending the AI Act" and record.metadata["status"] == "in_force"
    assert record.metadata["page_start"] == 10 and record.metadata["context"].startswith("Digital Omnibus")


def test_the_info_file_is_saved_next_to_the_store_and_reads_back(tmp_path):
    _, info, _ = built(tmp_path)
    path = info_path(tmp_path, "numpy")
    assert path.name == "numpy.info.json" and path.parent == tmp_path
    assert read_info(path) == info
    assert json.loads(path.read_text())["n_records"] == 10
    assert not list(tmp_path.glob("*.tmp"))  # written atomically: no half-written temporary file stays behind


def test_a_build_that_stops_half_way_leaves_no_info_file_so_the_index_is_not_used(tmp_path):
    built(tmp_path)  # a finished index is there ...

    class Dies(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            raise RuntimeError("the laptop went to sleep")

    with pytest.raises(RuntimeError, match="went to sleep"):
        built(tmp_path, embedder=Dies())
    assert not info_path(tmp_path, "numpy").exists()
    with pytest.raises(IndexNotReady, match="never finished"):
        open_index(tmp_path, "numpy", HashingEmbedder())


def test_progress_is_reported_for_both_stages(tmp_path):
    seen = []
    built(tmp_path, progress=lambda stage, done, total: seen.append((stage, done, total)))
    assert seen[0][0] == "embedding" and seen[-1] == ("storing", 10, 10)
    assert ("embedding", 10, 10) in seen


def test_rebuilding_replaces_the_old_content(tmp_path):
    built(tmp_path)
    store, info, _ = built(tmp_path, chunks=ragfixtures.chunks()[:4])
    assert store.count() == 4 == info.n_records and len(store.records()) == 4


def test_a_non_empty_store_is_not_overwritten_unless_a_reset_is_asked_for(tmp_path):
    store, _, embedder = built(tmp_path)
    with pytest.raises(IndexingError, match="already contains"):
        build_index(ragfixtures.chunks(), ragfixtures.by_source(), embedder, store, reset=False)


def test_chunks_that_cannot_be_indexed_are_rejected_before_anything_is_embedded(tmp_path):
    embedder = HashingEmbedder()
    store = NumpyStore()
    sources = ragfixtures.by_source()
    good = ragfixtures.chunks()
    with pytest.raises(IndexingError, match="no chunks"):
        build_index([], sources, embedder, store)
    with pytest.raises(IndexingError, match="unique"):
        build_index([good[0], good[0]], sources, embedder, store)
    with pytest.raises(IndexingError, match="no id"):
        build_index([Chunk("", "gdpr", "text", 1, "t", "t")], sources, embedder, store)
    with pytest.raises(IndexingError, match="no text"):
        build_index([Chunk("gdpr:9", "gdpr", "text", 1, "   ", "x")], sources, embedder, store)
    with pytest.raises(IndexingError, match="no text"):
        build_index([Chunk("gdpr:9", "gdpr", "text", 1, "x", " ")], sources, embedder, store)
    assert store.count() == 0


def test_chunks_of_documents_missing_from_the_manifest_are_rejected():
    sources = ragfixtures.by_source()
    del sources["ccpa"], sources["gdpr"]
    with pytest.raises(IndexingError, match=r"2 document\(s\).*not in the manifest.*ccpa, gdpr"):
        build_index(ragfixtures.chunks(), sources, HashingEmbedder(), NumpyStore())


def test_precomputed_vectors_skip_the_embedding_step_and_are_checked_like_any_others():
    embedder, chunks = HashingEmbedder(), ragfixtures.chunks()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])

    class Forbidden(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            raise AssertionError("must not embed again")

    first, second = NumpyStore(), NumpyStore()
    info_a = build_index(chunks, ragfixtures.by_source(), embedder, first, vectors=vectors)
    info_b = build_index(chunks, ragfixtures.by_source(), Forbidden(), second, vectors=vectors)
    assert info_a.n_records == info_b.n_records == 10 and info_b.seconds["embedding"] < 0.05
    query = embedder.embed_query("breach notification")
    assert [h.id for h in first.query(query, 5)] == [h.id for h in second.query(query, 5)]
    with pytest.raises(IndexingError, match="shape"):
        build_index(chunks, ragfixtures.by_source(), embedder, NumpyStore(), vectors=vectors[:5])


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda n, d: np.ones((n - 1, d), dtype=np.float32) / np.sqrt(d), "shape"),
        (lambda n, d: np.full((n, d), np.nan, dtype=np.float32), "NaN"),
        (lambda n, d: np.full((n, d), 2.0, dtype=np.float32), "unit-length"),
    ],
)
def test_vectors_that_break_the_embedder_contract_are_rejected(make, message):
    class Bad(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            return make(len(texts), self.dim)

    with pytest.raises(IndexingError, match=message):
        build_index(ragfixtures.chunks(), ragfixtures.by_source(), Bad(), NumpyStore())


def test_a_store_that_cannot_find_what_it_was_given_fails_the_build(tmp_path):
    class Forgetful(NumpyStore):
        def query(self, vector, k, where=None):
            return []

    with pytest.raises(IndexingError, match="does not return chunk"):
        build_index(ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), Forgetful())


def test_a_store_that_loses_records_fails_the_build():
    class Losing(NumpyStore):
        def count(self):
            return super().count() - 1

    with pytest.raises(IndexingError, match="holds 9 records after adding 10"):
        build_index(ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), Losing())


def test_the_build_also_works_with_the_chroma_adapter():
    store = ChromaStore("unused", client=FakeChromaClient())
    info = build_index(ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), store, backend="chroma")
    assert info.backend == "chroma" and store.count() == 10 and info.dim == 512


# ----------------------------------------------------------------------------------------------- tokens


class Measuring(HashingEmbedder):
    """A hashing embedder that pretends to have a 512-token window and a tokenizer."""

    def __init__(self, counts=None, limit=512):
        super().__init__()
        self._counts, self.max_tokens = counts, limit

    def token_counts(self, texts):
        return self._counts if self._counts is not None else [len(t.split()) for t in texts]


def test_chunks_longer_than_the_models_window_are_counted(tmp_path):
    counts = [100, 600, 513, 512, 40, 40, 40, 40, 40, 900]
    _, info, _ = built(tmp_path, embedder=Measuring(counts))
    assert (info.token_limit, info.over_token_limit, info.longest_tokens) == (512, 3, 900)


def test_the_token_report_never_stops_a_build(tmp_path):
    class Broken(Measuring):
        def token_counts(self, texts):
            raise RuntimeError("tokenizer missing")

    _, info, _ = built(tmp_path, embedder=Broken())
    assert info.n_records == 10 and info.token_limit is None and info.over_token_limit is None

    _, info, _ = built(tmp_path, embedder=Measuring(counts=[]))  # no counts: the limit is known, the overflow is not
    assert info.token_limit == 512 and info.over_token_limit is None and info.longest_tokens is None


# ----------------------------------------------------------------------------------------------- opening


def test_a_finished_index_opens(tmp_path):
    built(tmp_path)
    opened = open_index(tmp_path, "numpy", HashingEmbedder())
    assert opened.store.count() == 10 and opened.info.n_records == 10 and opened.warnings == ()


def test_no_index_yet_says_how_to_build_one(tmp_path):
    with pytest.raises(IndexNotReady, match=r"no numpy index in .* yet.*build_index\.py"):
        open_index(tmp_path, "numpy", HashingEmbedder())
    assert not (tmp_path / "numpy").exists()  # opening must not create an empty index as a side effect


def test_an_index_of_another_embedding_model_is_refused_with_the_way_out(tmp_path):
    built(tmp_path)
    with pytest.raises(IndexNotReady) as error:
        open_index(tmp_path, "numpy", HashingEmbedder(256))
    message = str(error.value)
    assert "'hashing-512'" in message and "'hashing-256'" in message and "EMBEDDING_MODEL=hashing-512" in message


def test_the_same_model_name_with_another_vector_size_is_refused(tmp_path):
    built(tmp_path)

    class Wider(HashingEmbedder):
        def __init__(self):
            super().__init__(256)
            self.name = "hashing-512"

    with pytest.raises(IndexNotReady, match="vectors of size 512"):
        open_index(tmp_path, "numpy", Wider())


def test_an_index_that_lost_its_data_is_reported_as_damaged(tmp_path):
    built(tmp_path)
    for file in (tmp_path / "numpy").iterdir():
        file.unlink()
    with pytest.raises(IndexNotReady, match="damaged|holds 0 records"):
        open_index(tmp_path, "numpy", HashingEmbedder())


def test_an_index_with_a_different_record_count_is_reported(tmp_path):
    built(tmp_path)
    path = info_path(tmp_path, "numpy")
    data = json.loads(path.read_text())
    data["n_records"] = 11
    path.write_text(json.dumps(data))
    with pytest.raises(IndexNotReady, match="holds 10 records but should hold 11"):
        open_index(tmp_path, "numpy", HashingEmbedder())


def test_an_unreadable_or_foreign_info_file_is_refused(tmp_path):
    built(tmp_path)
    path = info_path(tmp_path, "numpy")
    path.write_text("{ not json")
    with pytest.raises(IndexNotReady, match="cannot be read"):
        open_index(tmp_path, "numpy", HashingEmbedder())
    path.write_text("[1, 2, 3]")
    with pytest.raises(IndexNotReady, match="cannot be read"):
        open_index(tmp_path, "numpy", HashingEmbedder())
    data = built(tmp_path)[1].to_dict() | {"format": INFO_FORMAT + 1}
    path.write_text(json.dumps(data))
    with pytest.raises(IndexNotReady, match="different version"):
        open_index(tmp_path, "numpy", HashingEmbedder())


def test_a_changed_chunks_file_gives_a_warning_but_the_index_still_opens(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text("first version\n", encoding="utf-8")
    built(tmp_path, chunks_file=chunks_file, chunks_sha256=file_sha256(chunks_file))
    assert open_index(tmp_path, "numpy", HashingEmbedder(), chunks_path=chunks_file).warnings == ()
    chunks_file.write_text("second version\n", encoding="utf-8")
    opened = open_index(tmp_path, "numpy", HashingEmbedder(), chunks_path=chunks_file)
    assert len(opened.warnings) == 1 and "changed after this index was built" in opened.warnings[0]
    assert open_index(tmp_path, "numpy", HashingEmbedder(), chunks_path=tmp_path / "gone.jsonl").warnings == ()


def test_the_two_backends_do_not_disturb_each_other(tmp_path):
    built(tmp_path, backend="numpy")
    with pytest.raises(IndexNotReady, match="no chroma index"):
        open_index(tmp_path, "chroma", HashingEmbedder())


# ----------------------------------------------------------------------------------------------- helpers


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 1_000_000)  # more than one read block
    assert file_sha256(path) == hashlib.sha256(b"abc" * 1_000_000).hexdigest()
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_index_info_ignores_fields_from_the_future():
    _, _ = None, None
    base = {
        "format": 1, "embedding_model": "m", "dim": 3, "backend": "numpy", "space": "cosine", "n_records": 1, "n_documents": 1,
        "n_text": 1, "n_notes": 0, "chunks_file": "", "chunks_sha256": "", "built_at": "2026-01-01T00:00:00Z",
    }  # fmt: skip
    info = IndexInfo.from_dict(base | {"something_new": 5})
    assert info.embedding_model == "m" and info.to_dict()["token_limit"] is None and "something_new" not in info.to_dict()


def test_library_versions_lists_what_is_installed():
    versions = library_versions()
    assert versions["numpy"] == np.__version__
    assert all(isinstance(v, str) for v in versions.values()) and "definitely-not-a-package" not in versions


def test_footnote_chunks_are_stored_and_counted_separately(tmp_path):
    note = Chunk("gdpr:9001", "gdpr", "notes", 9, "Footnotes: 1 See Article 4.", "GDPR | footnotes\nFootnotes: 1 See Article 4.", ["Chapter I"], 5, 5, 5)
    store, info, _ = built(tmp_path, chunks=[*ragfixtures.chunks(), note])
    assert (info.n_records, info.n_text, info.n_notes) == (11, 10, 1)
    assert store.get(["gdpr:9001"])[0].metadata["kind"] == "notes"
