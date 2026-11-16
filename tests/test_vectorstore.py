from __future__ import annotations

import numpy as np
import pytest
from fakechroma import FakeChromaClient

from reglens.corpus.manifest import Source
from reglens.ingest.chunking import Chunk
from reglens.vectorstore import ChromaStore, NumpyStore, Record, VectorStoreError, open_store
from reglens.vectorstore.base import clean_metadata
from reglens.vectorstore.filters import matches, normalize_where, to_chroma_where
from reglens.vectorstore.records import chunk_record, lexical_text, make_records

# ----------------------------------------------------------------------------------------------- test data


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def sample():
    records = [
        Record("a:1", "alpha text", {"doc_id": "a", "jurisdiction": "EU", "status": "in_force", "ordinal": 1}),
        Record("a:2", "beta text", {"doc_id": "a", "jurisdiction": "EU", "status": "in_force", "ordinal": 2}),
        Record("b:1", "gamma text", {"doc_id": "b", "jurisdiction": "US", "status": "final", "ordinal": 1}),
        Record("c:1", "delta text", {"doc_id": "c", "jurisdiction": "EU", "status": "draft", "ordinal": 1}),
    ]
    vectors = np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0), unit(0, 1, 0), unit(0.5, 0.5, 0.5)])
    return records, vectors


@pytest.fixture(params=["numpy", "chroma"])
def store(request, tmp_path):
    if request.param == "numpy":
        return NumpyStore(tmp_path / "n")
    return ChromaStore(tmp_path / "c", client=FakeChromaClient())


@pytest.fixture()
def filled(store):
    records, vectors = sample()
    store.add(records, vectors)
    return store


# ----------------------------------------------------------------------------------------------- the contract, both stores


def test_count_get_and_records(filled):
    assert filled.count() == 4
    assert [r.id for r in filled.records()] == ["a:1", "a:2", "b:1", "c:1"]  # sorted by id
    got = filled.get(["c:1", "nope", "a:1"])
    assert [r.id for r in got] == ["c:1", "a:1"]  # asked order, unknown ids skipped
    assert got[1].text == "alpha text" and got[1].metadata["jurisdiction"] == "EU"
    assert filled.get([]) == []


def test_nearest_neighbour_order_and_scores(filled):
    hits = filled.query(unit(1, 0, 0), k=3)
    assert [h.id for h in hits] == ["a:1", "a:2", "c:1"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert hits[1].score == pytest.approx(float(unit(0.9, 0.1, 0) @ unit(1, 0, 0)), abs=1e-5)  # cosine similarity
    assert hits[0].record.text == "alpha text" and hits[0].record.metadata["doc_id"] == "a"


def test_k_larger_than_the_store_returns_everything(filled):
    assert len(filled.query(unit(1, 0, 0), k=50)) == 4


def test_filters(filled):
    q = unit(1, 0, 0)
    assert [h.id for h in filled.query(q, 10, {"jurisdiction": "US"})] == ["b:1"]
    assert [h.id for h in filled.query(q, 10, {"jurisdiction": "EU", "status": ["in_force", "draft"]})] == ["a:1", "a:2", "c:1"]
    assert filled.query(q, 10, {"jurisdiction": "UK"}) == []
    assert filled.query(q, 10, {"missing_key": "x"}) == []
    assert [h.id for h in filled.query(q, 1, {"jurisdiction": "EU"})] == ["a:1"]
    assert len(filled.query(q, 10, {})) == 4  # an empty filter filters nothing


def test_invalid_filters_are_rejected(filled):
    with pytest.raises(ValueError):
        filled.query(unit(1, 0, 0), 3, {"status": []})


def test_query_validates_k_and_works_on_an_empty_store(store):
    assert store.query(unit(1, 0, 0), 3) == []
    with pytest.raises(ValueError):
        store.query(unit(1, 0, 0), 0)


def test_reset_empties_the_store(filled):
    filled.reset()
    assert filled.count() == 0 and filled.records() == [] and filled.query(unit(1, 0, 0), 3) == []
    records, vectors = sample()
    filled.add(records[:1], vectors[:1])  # and it can be filled again
    assert filled.count() == 1


def test_reset_on_a_fresh_store_is_fine(store):
    store.reset()
    assert store.count() == 0


def test_duplicate_ids_are_rejected(filled):
    records, vectors = sample()
    with pytest.raises(ValueError, match="uplicate|unique"):
        filled.add(records[:1], vectors[:1])
    with pytest.raises(ValueError):
        filled.add([Record("z", "t", {"k": 1}), Record("z", "t", {"k": 1})], vectors[:2])
    assert filled.count() == 4  # nothing was half-added


def test_bad_vectors_are_rejected(store):
    with pytest.raises(ValueError):
        store.add([Record("x", "t", {"k": 1})], np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        store.add([Record("x", "t", {"k": 1})], np.array([[1.0, float("nan"), 0.0]], dtype=np.float32))
    assert store.count() == 0


def test_adding_in_several_batches(store):
    records, vectors = sample()
    store.add(records[:2], vectors[:2])
    store.add(records[2:], vectors[2:])
    assert store.count() == 4
    assert [h.id for h in store.query(unit(0, 1, 0), 1)] == ["b:1"]


def test_metadata_with_unsupported_types_is_cleaned(store):
    store.add([Record("x", "t", {"a": None, "b": ("t", 1), "c": 3, "d": True, "e": 1.5, "f": "s"})], np.array([unit(1, 0, 0)]))
    meta = store.get(["x"])[0].metadata
    assert "a" not in meta and meta["b"] == "('t', 1)" and meta["c"] == 3 and meta["d"] is True and meta["e"] == 1.5


def test_clean_metadata():
    assert clean_metadata({"a": None, "b": [1], "c": "x"}) == {"b": "[1]", "c": "x"}


# ----------------------------------------------------------------------------------------------- both stores agree


def test_both_stores_return_the_same_ranking_for_random_data(tmp_path):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(200, 24)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [Record(f"r{i:03d}", f"text {i}", {"group": i % 3, "doc_id": f"d{i % 5}"}) for i in range(200)]
    exact = NumpyStore()
    chroma = ChromaStore(tmp_path / "c", client=FakeChromaClient(), batch_size=64)
    exact.add(records, vectors)
    chroma.add(records, vectors)
    for _ in range(10):
        q = rng.normal(size=24).astype(np.float32)
        q /= np.linalg.norm(q)
        a, b = exact.query(q, 10, {"group": [0, 2]}), chroma.query(q, 10, {"group": [0, 2]})
        assert [h.id for h in a] == [h.id for h in b]
        assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-5)


@pytest.mark.parametrize("space", ["cosine", "ip", "l2"])
def test_cosine_dot_product_and_euclidean_rank_unit_vectors_identically(space):
    """The claim made in the report: for unit-length vectors the three measures give the same order."""
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [Record(f"r{i:03d}", "t", {"k": 1}) for i in range(300)]
    reference = NumpyStore(space="cosine")
    other = NumpyStore(space=space)
    reference.add(records, vectors)
    other.add(records, vectors)
    for _ in range(10):
        q = rng.normal(size=16).astype(np.float32)
        q /= np.linalg.norm(q)
        assert [h.id for h in reference.query(q, 15)] == [h.id for h in other.query(q, 15)]


def test_cosine_is_not_the_same_as_dot_product_for_vectors_of_different_length():
    records = [Record("short", "t", {"k": 1}), Record("long", "t", {"k": 1})]
    vectors = np.array([[1.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    query = np.array([1.0, 0.1], dtype=np.float32)
    cosine, dot = NumpyStore(space="cosine"), NumpyStore(space="ip")
    cosine.add(records, vectors)
    dot.add(records, vectors)
    assert cosine.query(query, 1)[0].id == "short"
    assert dot.query(query, 1)[0].id == "long"  # the longer vector wins on raw dot product


def test_unknown_space_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        NumpyStore(space="manhattan")
    with pytest.raises(ValueError):
        ChromaStore(tmp_path, space="manhattan", client=FakeChromaClient())


def test_ties_are_broken_by_insertion_order():
    store = NumpyStore()
    store.add([Record(f"r{i}", "t", {"k": 1}) for i in range(5)], np.tile(unit(1, 0), (5, 1)))
    assert [h.id for h in store.query(unit(1, 0), 3)] == ["r0", "r1", "r2"]


# ----------------------------------------------------------------------------------------------- NumpyStore persistence


def test_numpy_store_survives_a_restart(tmp_path):
    records, vectors = sample()
    first = NumpyStore(tmp_path / "idx")
    first.add(records, vectors)
    assert not (tmp_path / "idx" / "records.jsonl").exists()  # nothing is written before flush()
    first.flush()
    second = NumpyStore(tmp_path / "idx")
    assert second.count() == 4
    assert [h.id for h in second.query(unit(1, 0, 0), 2)] == ["a:1", "a:2"]
    assert second.get(["b:1"])[0].metadata["jurisdiction"] == "US"
    assert not list((tmp_path / "idx").glob("*.tmp*"))  # temporary files were renamed


def test_numpy_store_flush_without_changes_writes_nothing(tmp_path):
    NumpyStore(tmp_path / "idx").flush()
    assert not (tmp_path / "idx").exists()


def test_numpy_store_reset_deletes_the_saved_files(tmp_path):
    records, vectors = sample()
    store = NumpyStore(tmp_path / "idx")
    store.add(records, vectors)
    store.flush()
    store.reset()
    assert NumpyStore(tmp_path / "idx").count() == 0


def test_damaged_saved_index_is_reported(tmp_path):
    folder = tmp_path / "idx"
    records, vectors = sample()
    store = NumpyStore(folder)
    store.add(records, vectors)
    store.flush()
    (folder / "records.jsonl").write_text("{broken", encoding="utf-8")
    with pytest.raises(VectorStoreError, match="damaged"):
        NumpyStore(folder)
    store.flush()  # unchanged -> still nothing to write; the broken file stays broken
    (folder / "records.jsonl").write_text('{"id": "x", "text": "t"}\n', encoding="utf-8")
    with pytest.raises(VectorStoreError, match="inconsistent"):
        NumpyStore(folder)


def test_query_dimension_mismatch_is_reported():
    store = NumpyStore()
    records, vectors = sample()
    store.add(records, vectors)
    with pytest.raises(ValueError, match="dimensions"):
        store.query(np.array([1.0, 0.0], dtype=np.float32), 3)
    with pytest.raises(ValueError, match="dimensions"):
        store.add([Record("new", "t", {"k": 1})], np.array([[1.0, 0.0]], dtype=np.float32))


# ----------------------------------------------------------------------------------------------- the Chroma adapter itself


def test_chroma_uses_the_documented_collection_options(tmp_path):
    client = FakeChromaClient()
    ChromaStore(tmp_path, client=client)
    options = client.opened_with[0]
    assert options["configuration"] == {"hnsw": {"space": "cosine"}}
    assert options["embedding_function"] is None  # Chroma must never embed anything itself


def test_chroma_falls_back_to_the_old_metadata_option(tmp_path):
    client = FakeChromaClient(supports_configuration=False)
    store = ChromaStore(tmp_path, client=client)
    assert client.opened_with[-1]["metadata"] == {"hnsw:space": "cosine"}
    assert store.space == "cosine" and client.collections["reglens"].space == "cosine"


def test_chroma_adds_in_batches_and_sends_python_floats(tmp_path):
    client = FakeChromaClient()
    store = ChromaStore(tmp_path, client=client, batch_size=3)
    records = [Record(f"r{i}", "t", {"k": 1}) for i in range(8)]
    store.add(records, np.tile(unit(1, 0), (8, 1)))
    collection = client.collections["reglens"]
    assert collection.add_calls == 3 and collection.max_batch == 3 and store.count() == 8


def test_chroma_reset_recreates_the_collection_and_ignores_a_missing_one(tmp_path):
    client = FakeChromaClient()
    store = ChromaStore(tmp_path, client=client)
    client.collections.clear()  # the collection vanished: reset must still work
    store.reset()
    assert "reglens" in client.collections


def test_chroma_reset_does_not_hide_real_errors(tmp_path):
    class Broken(FakeChromaClient):
        def delete_collection(self, name):
            raise RuntimeError("disk is full")

    store = ChromaStore(tmp_path, client=Broken())
    with pytest.raises(RuntimeError, match="disk is full"):
        store.reset()


def test_chroma_l2_scores_are_negated_distances(tmp_path):
    store = ChromaStore(tmp_path, client=FakeChromaClient(), space="l2")
    store.add([Record("x", "t", {"k": 1}), Record("y", "t", {"k": 1})], np.array([unit(1, 0), unit(0, 1)]))
    hits = store.query(unit(1, 0), 2)
    assert [h.id for h in hits] == ["x", "y"]
    assert hits[0].score == pytest.approx(0.0, abs=1e-6) and hits[1].score == pytest.approx(-2.0, abs=1e-5)


def test_chroma_lists_more_records_than_one_page(tmp_path, monkeypatch):
    import reglens.vectorstore.chroma_store as module

    monkeypatch.setattr(module, "PAGE", 7)
    store = ChromaStore(tmp_path, client=FakeChromaClient())
    n = 30
    store.add([Record(f"r{i:02d}", "t", {"k": 1}) for i in range(n)], np.tile(unit(1, 0), (n, 1)))
    assert [r.id for r in store.records()] == [f"r{i:02d}" for i in range(n)]


def test_chroma_missing_library_gives_an_install_hint(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(VectorStoreError, match="pip install"):
        ChromaStore(tmp_path)


def test_open_store_gives_each_backend_its_own_folder(tmp_path):
    numpy_store = open_store(tmp_path, "numpy")
    assert isinstance(numpy_store, NumpyStore)
    with pytest.raises(ValueError):
        open_store(tmp_path, "pinecone")


# ----------------------------------------------------------------------------------------------- filters


def test_normalize_where():
    assert normalize_where(None) == {} and normalize_where({}) == {}
    assert normalize_where({"a": "x", "b": ["y", "z"], "c": 3, "d": True}) == {"a": ["x"], "b": ["y", "z"], "c": [3], "d": [True]}
    for bad in ({"a": []}, {"": "x"}, {"a": {"nested": 1}}, {"a": [1, [2]]}, {"a": None}, {1: "x"}):
        with pytest.raises(ValueError):
            normalize_where(bad)


def test_matches():
    meta = {"jurisdiction": "EU", "status": "final", "ordinal": 3}
    assert matches(meta, None) and matches(meta, {})
    assert matches(meta, {"jurisdiction": "EU"}) and matches(meta, {"status": ["draft", "final"]})
    assert not matches(meta, {"jurisdiction": "US"}) and not matches(meta, {"absent": "x"})
    assert matches(meta, {"jurisdiction": "EU", "ordinal": 3}) and not matches(meta, {"jurisdiction": "EU", "ordinal": 4})


def test_to_chroma_where():
    assert to_chroma_where(None) is None and to_chroma_where({}) is None
    assert to_chroma_where({"a": "x"}) == {"a": "x"}
    assert to_chroma_where({"a": ["x", "y"]}) == {"a": {"$in": ["x", "y"]}}
    assert to_chroma_where({"a": "x", "b": ["y", "z"]}) == {"$and": [{"a": "x"}, {"b": {"$in": ["y", "z"]}}]}


# ----------------------------------------------------------------------------------------------- records


def make_chunk(**kw):
    base = dict(chunk_id="doc:0007", doc_id="doc", kind="text", ordinal=7, text="The controller shall act.",
                embed_text="Doc title | Chapter 1 > Article 5\nThe controller shall act.", section_path=["Chapter 1", "Article 5"],
                page_start=12, page_end=13, n_words=4)
    base.update(kw)
    return Chunk(**base)


def make_source(**kw):
    base = dict(id="doc", title="Doc title", publisher="Pub", jurisdiction="EU", doc_type="regulation", status="in_force",
                version_date="2024", priority="1", fetch="auto", format="pdf", url="https://example.org/doc.pdf",
                landing_url="https://example.org/doc", filename="doc.pdf", verification="x", notes="")
    base.update(kw)
    return Source(**base)


def test_chunk_record_carries_everything_citations_and_filters_need():
    record = chunk_record(make_chunk(), make_source())
    assert record.id == "doc:0007" and record.text == "The controller shall act."
    assert dict(record.metadata) == {
        "doc_id": "doc", "kind": "text", "ordinal": 7, "section": "Chapter 1 > Article 5", "context": "Doc title | Chapter 1 > Article 5",
        "n_words": 4, "page_start": 12, "page_end": 13, "title": "Doc title", "publisher": "Pub", "jurisdiction": "EU",
        "doc_type": "regulation", "status": "in_force", "version_date": "2024", "url": "https://example.org/doc",
    }


def test_chunk_record_for_a_web_page_without_pages_or_a_section():
    record = chunk_record(make_chunk(page_start=None, page_end=None, section_path=[], embed_text="Doc title\nThe controller shall act."), make_source(landing_url=""))
    assert "page_start" not in record.metadata and "section" not in record.metadata
    assert record.metadata["url"] == "https://example.org/doc.pdf" and record.metadata["context"] == "Doc title"


def test_chunk_record_without_a_known_source_still_works():
    record = chunk_record(make_chunk(page_end=None), None)
    assert record.metadata["page_end"] == 12 and "title" not in record.metadata


def test_make_records_and_lexical_text():
    records = make_records([make_chunk(), make_chunk(chunk_id="x:1", doc_id="x")], {"doc": make_source()})
    assert [r.id for r in records] == ["doc:0007", "x:1"] and "title" in records[0].metadata and "title" not in records[1].metadata
    assert lexical_text(records[0]) == "Doc title | Chapter 1 > Article 5\nThe controller shall act."
    assert lexical_text(Record("r", "plain", {})) == "plain"


def test_a_byte_string_is_not_accepted_as_a_list_of_filter_values():
    from reglens.vectorstore.filters import normalize_where

    with pytest.raises(ValueError, match="expected a value or a list of values, not bytes"):
        normalize_where({"jurisdiction": b"EU"})


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_vectors_of_another_size_than_the_stored_ones_are_rejected(tmp_path):
    store = NumpyStore(tmp_path / "n")
    records, vectors = sample()
    store.add(records, vectors)
    with pytest.raises(ValueError, match="4 dimensions but the store holds 3"):
        store.add([Record("x:1", "t", {"k": 1})], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert store.count() == 4


def test_cosine_similarity_does_not_depend_on_the_length_of_the_query():
    store = NumpyStore(space="cosine")
    records, vectors = sample()
    store.add(records, vectors)
    short, long = store.query(unit(1, 0.2, 0), 4), store.query(unit(1, 0.2, 0) * 7.0, 4)
    assert [h.id for h in short] == [h.id for h in long]
    assert [h.score for h in long] == pytest.approx([h.score for h in short], abs=1e-6) and max(h.score for h in long) <= 1.0 + 1e-6


def test_records_are_listed_in_id_order_whatever_the_order_they_were_added_in(store):
    records, vectors = sample()
    order = [3, 0, 2, 1]
    store.add([records[i] for i in order], vectors[order])
    assert [r.id for r in store.records()] == ["a:1", "a:2", "b:1", "c:1"]


def test_adding_nothing_changes_nothing_and_saves_nothing(tmp_path):
    store = NumpyStore(tmp_path / "n")
    store.add([], np.zeros((0, 3), dtype=np.float32))
    store.flush()
    assert store.count() == 0 and not (tmp_path / "n").exists()
