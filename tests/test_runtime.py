from __future__ import annotations

import pytest
import ragfixtures
from ragfixtures import FakeLLM

from reglens.config import PROJECT_ROOT, load_settings
from reglens.embeddings import HashingEmbedder
from reglens.generation.answer import EXTRACTIVE, RAG
from reglens.indexing import IndexNotReady, build_index, info_path
from reglens.llm.base import LLMError
from reglens.pipeline import AskOptions
from reglens.retrieval import RetrievalOptions
from reglens.runtime import build_runtime, describe, load_sources
from reglens.vectorstore import open_store


class LengthReranker:
    """Prefers short passages: enough to see that the re-ranking stage ran."""

    name = "length-reranker"

    def score(self, query, passages):
        return [-float(len(p)) for p in passages]


def settings_for(tmp_path, **env):
    environ = {"VECTORSTORE": "numpy", "VECTORSTORE_DIR": str(tmp_path / "index"), "GEMINI_API_KEY": "", **env}
    return load_settings(root=tmp_path, environ=environ)


def index_for(settings, embedder=None):
    store = open_store(settings.vectorstore_dir, "numpy")
    return build_index(
        ragfixtures.chunks(), ragfixtures.by_source(), embedder or HashingEmbedder(), store,
        info_file=info_path(settings.vectorstore_dir, "numpy"), backend="numpy",
    )  # fmt: skip


def test_the_runtime_answers_questions_from_the_index(tmp_path):
    settings = settings_for(tmp_path)
    index_for(settings)
    llm = FakeLLM("Notify within 72 hours [S1].")
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=llm)
    assert runtime.has_llm and (runtime.llm_provider, runtime.llm_model, runtime.llm_problem) == ("fake", "fake-1", None)
    answer = runtime.pipeline.ask("How quickly must we report a personal data breach to the supervisory authority?")
    assert answer.mode == RAG and answer.sources[0].chunk_id == "gdpr:0001" and answer.cited == ("S1",)
    assert len(runtime.retriever) == 10 and runtime.warnings == ()


def test_the_reranker_is_wired_in(tmp_path):
    settings = settings_for(tmp_path)
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=FakeLLM("x [S1]"))
    assert runtime.retriever.has_reranker
    options = AskOptions(retrieval=RetrievalOptions(rerank=True, top_k=3), use_llm=False)
    answer = runtime.pipeline.ask("personal data controller", options=options)
    assert len(answer.sources) == 3 and all(s.rerank_score is not None for s in answer.sources)
    scores = [s.rerank_score for s in answer.sources]
    assert scores == sorted(scores, reverse=True) and [s.score for s in answer.sources] == scores  # re-ranker decides the order
    assert [s.rank for s in answer.sources] == [1, 2, 3]


def test_the_default_reranker_is_the_configured_cross_encoder_and_loads_lazily(tmp_path):
    settings = settings_for(tmp_path, RERANKER_MODEL="some/cross-encoder")
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), llm=FakeLLM("x"))  # nothing is downloaded here
    assert runtime.retriever.has_reranker and runtime.retriever._reranker.name == "some/cross-encoder"


def test_a_missing_key_is_not_an_error_but_answers_come_from_the_passages(tmp_path):
    settings = settings_for(tmp_path)
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker())
    assert not runtime.has_llm and runtime.llm_provider is None and "GEMINI_API_KEY" in runtime.llm_problem
    answer = runtime.pipeline.ask("right to erasure delete data")
    assert answer.mode == EXTRACTIVE and answer.sources and answer.error_kind == "unavailable"
    assert "GEMINI_API_KEY" in answer.error


def test_a_configuration_problem_with_the_model_is_reported_not_raised(tmp_path):
    settings = settings_for(tmp_path, LLM_PROVIDER="openai", OPENAI_API_KEY="sk-not-a-real-key")
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker())
    assert not runtime.has_llm and "OPENAI_MODEL" in runtime.llm_problem
    assert "sk-not-a-real-key" not in repr(runtime) and "sk-not-a-real-key" not in repr(runtime.settings)


def test_a_key_in_the_settings_builds_a_real_client_without_calling_the_network(tmp_path):
    settings = settings_for(tmp_path, GEMINI_API_KEY="fake-key-for-test", GEMINI_MODEL="gemini-test-1", GEMINI_FALLBACK_MODEL="gemini-test-2")
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker())
    assert runtime.has_llm and runtime.llm_provider == "gemini" and runtime.llm_model == "gemini-test-1"
    other = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), provider="anthropic")
    assert not other.has_llm and "ANTHROPIC_API_KEY" in other.llm_problem  # asking for another provider uses its key


def test_a_missing_index_stops_with_a_clear_message(tmp_path):
    with pytest.raises(IndexNotReady, match="build_index.py"):
        build_runtime(settings_for(tmp_path), embedder=HashingEmbedder(), llm=FakeLLM("x"))


def test_an_index_in_a_synced_folder_is_explained_in_a_warning(tmp_path):
    root = tmp_path / "OneDrive" / "Desktop" / "reglens"
    root.mkdir(parents=True)
    local = tmp_path / "AppData"
    settings = load_settings(root=root, environ={"VECTORSTORE": "numpy", "LOCALAPPDATA": str(local), "GEMINI_API_KEY": ""})
    assert str(settings.vectorstore_dir).startswith(str(local))
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=FakeLLM("x"))
    assert len(runtime.warnings) == 1 and "OneDrive" in runtime.warnings[0] and str(local) in runtime.warnings[0]


def test_a_stale_chunks_file_is_passed_on_as_a_warning(tmp_path):
    settings = settings_for(tmp_path)
    settings.chunks_path.parent.mkdir(parents=True)
    settings.chunks_path.write_text("old\n", encoding="utf-8")
    from reglens.indexing import file_sha256

    store = open_store(settings.vectorstore_dir, "numpy")
    build_index(
        ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), store,
        info_file=info_path(settings.vectorstore_dir, "numpy"), backend="numpy", chunks_sha256=file_sha256(settings.chunks_path),
    )  # fmt: skip
    settings.chunks_path.write_text("new\n", encoding="utf-8")
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=FakeLLM("x"))
    assert any("changed after this index was built" in w for w in runtime.warnings)


def test_describe_reports_facts_without_secrets(tmp_path):
    settings = settings_for(tmp_path, GEMINI_API_KEY="fake-key-for-test")
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker())
    facts = describe(runtime)
    assert facts["documents"] == 6 and facts["passages"] == 10 and facts["embedding_model"] == "hashing-512"
    assert facts["vector_store"] == "numpy" and facts["language_model"].startswith("gemini/gemini-")
    assert "fake-key-for-test" not in str(facts)
    assert describe(build_runtime(settings_for(tmp_path), embedder=HashingEmbedder()))["language_model"] is None


def test_the_manifest_is_loaded_as_a_mapping_by_document_id():
    sources = load_sources(load_settings(root=PROJECT_ROOT, environ={}))
    assert len(sources) == 34 and sources["eu-gdpr"].jurisdiction == "EU" and sources["eu-ai-omnibus"].doc_type == "regulation"


def test_an_unreadable_manifest_gives_an_empty_mapping(tmp_path):
    assert load_sources(load_settings(root=tmp_path, environ={})) == {}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sources.csv").write_text("not,a,manifest\n1,2,3\n", encoding="utf-8")
    assert load_sources(load_settings(root=tmp_path, environ={})) == {}


def test_llm_errors_at_question_time_still_degrade(tmp_path):
    settings = settings_for(tmp_path)
    index_for(settings)
    llm = FakeLLM(LLMError("quota", "daily limit reached"))
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=llm)
    answer = runtime.pipeline.ask("controller notify supervisory authority")
    assert answer.mode == EXTRACTIVE and answer.error_kind == "quota"


def test_the_retriever_lists_its_documents(tmp_path):
    settings = settings_for(tmp_path)
    index_for(settings)
    runtime = build_runtime(settings, embedder=HashingEmbedder(), reranker=LengthReranker(), llm=FakeLLM("x"))
    documents = runtime.retriever.documents()
    assert [d[0] for d in documents] == ["ai-act", "ccpa", "draft-hr", "gdpr", "hhs-breach", "omnibus"]
    assert dict((d[0], d[2]) for d in documents) == {"ai-act": 2, "ccpa": 2, "draft-hr": 1, "gdpr": 3, "hhs-breach": 1, "omnibus": 1}
    assert documents[0][1] == "AI Act (as originally published)"
