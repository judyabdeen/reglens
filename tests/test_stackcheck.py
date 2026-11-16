from __future__ import annotations

import importlib.metadata as metadata

import numpy as np
import ragfixtures
from fakechroma import FakeChromaClient
from ragfixtures import FakeLLM

import reglens.stackcheck as stack
from reglens.config import load_settings
from reglens.embeddings import HashingEmbedder
from reglens.llm.base import LLMError
from reglens.vectorstore import ChromaStore, NumpyStore


def status_of(checks):
    return {c.name: c.status for c in checks}


# ----------------------------------------------------------------------------------------------- libraries


def test_missing_required_libraries_fail_and_missing_optional_ones_warn():
    installed = {"numpy": "2.0", "chromadb": "1.0.1", "sentence-transformers": "5.1", "torch": "2.4", "transformers": "4.50"}

    def version(name):
        if name in installed:
            return installed[name]
        raise metadata.PackageNotFoundError(name)

    checks = {c.name: c for c in stack.check_libraries(version)}
    assert checks["chromadb"].status == "OK" and checks["chromadb"].detail == "1.0.1"
    assert checks["huggingface-hub"].status == "WARN" and "pip install -r requirements.txt" in checks["huggingface-hub"].detail
    del installed["torch"]
    assert {c.name: c.status for c in stack.check_libraries(version)}["torch"] == "FAIL"


def test_the_accelerator_check_never_crashes():
    check = stack.check_accelerator()
    assert check.name == "Compute device" and check.status in ("OK", "SKIP", "FAIL")


def test_a_cloud_synced_project_folder_gets_a_warning(tmp_path):
    plain = load_settings(root=tmp_path, environ={})
    assert stack.check_sync_folder(plain).status == "OK"
    synced = load_settings(root=tmp_path / "OneDrive" / "p", environ={"LOCALAPPDATA": str(tmp_path / "local")})
    check = stack.check_sync_folder(synced)
    assert check.status == "WARN" and "OneDrive" in check.detail and str(tmp_path / "local") in check.detail and "C:\\dev\\reglens" in check.detail


# ----------------------------------------------------------------------------------------------- chroma


def chroma_factory():
    client = FakeChromaClient()  # one "server" for every open of the same path, like a real persistent folder
    return lambda path: ChromaStore(path, client=client)


def test_a_working_chroma_passes_the_round_trip():
    (check,) = stack.check_chroma(chroma_factory())
    assert check.name == "ChromaDB" and check.status == "OK" and "5 records" in check.detail and "same results as the exact store" in check.detail


def test_chroma_that_forgets_everything_after_reopening_fails():
    client = FakeChromaClient()
    opened = []

    def factory(path):
        opened.append(path)
        return ChromaStore(path, client=client if len(opened) == 1 else FakeChromaClient())  # a different, empty database

    (check,) = stack.check_chroma(factory)
    assert check.status == "FAIL" and "after reopening" in check.detail


def test_chroma_that_returns_other_results_than_the_exact_store_fails():
    class Reversed(NumpyStore):
        def query(self, vector, k, where=None):
            return list(reversed(super().query(vector, k, where)))

    (check,) = stack.check_chroma(lambda path: Reversed())
    assert check.status == "FAIL" and "vs exact" in check.detail


def test_a_chroma_that_cannot_be_opened_is_reported_with_its_message():
    def factory(path):
        raise RuntimeError("chromadb is not installed")

    (check,) = stack.check_chroma(factory)
    assert check.status == "FAIL" and "RuntimeError: chromadb is not installed" in check.detail


# ----------------------------------------------------------------------------------------------- models


class Measuring(HashingEmbedder):
    max_tokens = 100

    def token_counts(self, texts):
        return [50] * (len(texts) - 2) + [150, 300]


def test_a_working_embedding_model_passes_and_projects_the_time_for_the_corpus():
    texts = [c.embed_text for c in ragfixtures.chunks()] * 10
    clock = iter(float(i) for i in range(100))  # one "second" per reading
    checks = stack.check_embedding(Measuring(), texts, clock=lambda: next(clock))
    named = {c.name: c for c in checks}
    assert named["Embedding model hashing-512"].status == "OK" and "512 numbers per vector" in named["Embedding model hashing-512"].detail
    assert "ranks the breach passage first" in named["Embedding model hashing-512"].detail
    speed = named["Embedding speed"]
    assert speed.status == "OK" and "48.0 passages per second" in speed.detail and "all 100 passages" in speed.detail
    window = named["Chunk length"]
    assert window.status == "WARN" and "2 of 100 chunks (2.0%) are longer than the 100-token window" in window.detail and "300 tokens" in window.detail


def test_chunk_lengths_inside_the_window_are_fine():
    class Fits(Measuring):
        def token_counts(self, texts):
            return [10] * len(texts)

    named = {c.name: c for c in stack.check_embedding(Fits(), ["some chunk text"] * 5)}
    assert named["Chunk length"].status == "OK" and "0 of 5 chunks" in named["Chunk length"].detail


def test_without_chunks_the_speed_is_still_measured():
    named = {c.name: c for c in stack.check_embedding(HashingEmbedder())}
    assert named["Embedding speed"].status == "OK" and "no chunks.jsonl yet" in named["Embedding speed"].detail and "Chunk length" not in named


def test_an_embedding_model_that_cannot_load_fails_with_its_reason():
    class Offline(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            raise OSError("offline: could not reach huggingface.co")

    (check,) = stack.check_embedding(Offline())
    assert check.status == "FAIL" and "offline" in check.detail


def test_an_embedding_model_with_wrong_vectors_or_wrong_ranking_fails():
    class NotNormalised(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            return super().embed_documents(texts, progress) * 3

    (check,) = stack.check_embedding(NotNormalised())
    assert check.status == "FAIL" and "unexpected vectors" in check.detail

    class Backwards(HashingEmbedder):
        def embed_query(self, text):
            return super().embed_query("flour eggs bowl bake mixture golden")

    (check,) = stack.check_embedding(Backwards())
    assert check.status == "FAIL" and "not to the breach passage" in check.detail


def test_an_unmeasurable_window_is_a_skip_not_a_failure():
    named = {c.name: c for c in stack.check_embedding(HashingEmbedder(), ["x"] * 3)}
    assert named["Chunk length"].status == "SKIP"

    class Broken(Measuring):
        def token_counts(self, texts):
            raise RuntimeError("no tokenizer")

    named = {c.name: c for c in stack.check_embedding(Broken(), ["x"] * 3)}
    assert named["Chunk length"].status == "WARN" and "no tokenizer" in named["Chunk length"].detail


class Reranker:
    name = "stub-reranker"

    def __init__(self, scores=None, error=None):
        self.scores, self.error = scores, error

    def score(self, query, passages):
        if self.error:
            raise self.error
        return self.scores or [3.0, -2.0][: len(passages)]


def test_the_reranker_check():
    assert stack.check_reranker(Reranker()).status == "OK"
    assert stack.check_reranker(Reranker([1.0, 2.0])).status == "FAIL"
    assert stack.check_reranker(Reranker([1.0])).status == "FAIL"
    failed = stack.check_reranker(Reranker(error=RuntimeError("cannot download")))
    assert failed.status == "FAIL" and "cannot download" in failed.detail


# ----------------------------------------------------------------------------------------------- language model


def test_the_language_model_check(tmp_path):
    settings = load_settings(root=tmp_path, environ={})
    ok = stack.check_language_model(settings, lambda s: FakeLLM("ready"))
    assert ok.status == "OK" and "fake/fake-1 answered 'ready'" in ok.detail

    def no_key(s):
        raise LLMError("auth", "GEMINI_API_KEY is empty - add your key")

    assert stack.check_language_model(settings, no_key).status == "WARN"
    assert stack.check_language_model(settings, lambda s: FakeLLM(LLMError("auth", "key rejected"))).status == "FAIL"
    assert stack.check_language_model(settings, lambda s: FakeLLM(LLMError("not_found", "model retired"))).status == "FAIL"
    busy = stack.check_language_model(settings, lambda s: FakeLLM(LLMError("quota", "daily quota used up")))
    assert busy.status == "WARN" and "quota: daily quota used up" in busy.detail
    assert stack.check_language_model(settings, lambda s: FakeLLM(lambda request: 1 / 0)).status == "FAIL"
    fallback = stack.check_language_model(settings, lambda s: FakeLLM("ready", fell_back_from="fake-0"))
    assert "fake-0 was unavailable" in fallback.detail


# ----------------------------------------------------------------------------------------------- orchestration


def test_all_checks_together_with_stand_ins(tmp_path):
    settings = load_settings(root=tmp_path, environ={"GEMINI_API_KEY": ""})
    settings.chunks_path.parent.mkdir(parents=True)
    import json

    settings.chunks_path.write_text("\n".join(json.dumps(c.to_dict()) for c in ragfixtures.chunks()) + "\n", encoding="utf-8")
    said = []
    checks = stack.run_stack_checks(
        settings, online=False, embedder=Measuring(), reranker=Reranker(), chroma_factory=chroma_factory(),
        llm_factory=lambda s: FakeLLM("ready"), say=said.append,
    )  # fmt: skip
    status = status_of(checks)
    assert status["ChromaDB"] == "OK" and status["Embedding model hashing-512"] == "OK" and status["Re-ranking model stub-reranker"] == "OK"
    assert status["Language model"] == "SKIP"  # offline: no call
    assert status["Project folder"] == "OK" and "chunks.jsonl" not in status
    assert any("Loading the embedding model" in line for line in said) and any("ChromaDB" in line for line in said)
    assert any("tokens" in c.detail for c in checks if c.name == "Chunk length")


def test_the_loading_message_mentions_a_download_only_for_models_that_are_downloaded(tmp_path):
    for model, downloads in (("hashing-512", False), ("BAAI/bge-small-en-v1.5", True)):
        settings = load_settings(root=tmp_path, environ={"EMBEDDING_MODEL": model, "GEMINI_API_KEY": ""})
        said: list[str] = []
        stack.run_stack_checks(
            settings, online=False, rerank=False, llm=False, embedder=HashingEmbedder(), chroma_factory=chroma_factory(), say=said.append,
        )  # fmt: skip
        loading = next(line for line in said if "Loading the embedding model" in line)
        assert model in loading
        assert ("downloads about 130 MB" in loading) is downloads
        assert ("needs no download" in loading) is not downloads


def test_the_language_model_is_called_when_online_and_skipped_on_request(tmp_path, monkeypatch):
    settings = load_settings(root=tmp_path, environ={})
    monkeypatch.setattr(stack, "check_reachability", lambda urls: [stack.Check("huggingface.co", "OK", "reachable (HTTP 200)")])
    kwargs = dict(embedder=HashingEmbedder(), reranker=Reranker(), chroma_factory=chroma_factory(), llm_factory=lambda s: FakeLLM("ready"))
    checks = stack.run_stack_checks(settings, online=True, **kwargs)
    status = status_of(checks)
    assert status["Language model"] == "OK" and status["huggingface.co"] == "OK" and status["chunks.jsonl"] == "WARN"
    checks = stack.run_stack_checks(settings, online=True, llm=False, rerank=False, **kwargs)
    assert status_of(checks)["Language model"] == "SKIP" and status_of(checks)["Re-ranking model"] == "SKIP"


def test_an_unreadable_chunks_file_is_a_warning_not_a_crash(tmp_path):
    settings = load_settings(root=tmp_path, environ={})
    settings.chunks_path.parent.mkdir(parents=True)
    settings.chunks_path.write_text("{broken\n", encoding="utf-8")
    checks = stack.run_stack_checks(settings, online=False, embedder=HashingEmbedder(), reranker=Reranker(), chroma_factory=chroma_factory())
    assert status_of(checks)["chunks.jsonl"] == "WARN"


def test_main_prints_a_block_to_paste_and_fails_when_something_is_broken(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr(stack, "make_embedder", lambda name: HashingEmbedder())
    monkeypatch.setattr(stack, "check_chroma", lambda factory=None: [stack.Check("ChromaDB", "FAIL", "RuntimeError: chromadb is not installed")])
    code = stack.main(["--offline", "--no-rerank", "--no-llm", "--env-file", str(env_file)])
    out = capsys.readouterr().out
    assert code == 1
    assert "RegLens stack check" in out and "[FAIL] ChromaDB: RuntimeError: chromadb is not installed" in out
    assert "--- copy everything between these lines" in out and "FAIL: ChromaDB:" in out
    assert out.isascii()
