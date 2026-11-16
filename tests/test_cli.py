from __future__ import annotations

import json

import pytest
import ragfixtures
from ragfixtures import FakeLLM

from reglens.cli import ask as ask_cli
from reglens.cli import build_index as build_cli
from reglens.config import load_settings
from reglens.embeddings import EmbeddingError, HashingEmbedder
from reglens.indexing import IndexNotReady, info_path, open_index
from reglens.llm.base import LLMError
from reglens.runtime import build_runtime
from reglens.vectorstore import open_store


class StubReranker:
    name = "stub"

    def score(self, query, passages):
        return [float(len(p)) for p in passages]


def write_chunks(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c.to_dict()) for c in ragfixtures.chunks()) + "\n", encoding="utf-8")
    return path


class Env:
    """A temporary project folder with settings that point into it."""

    def __init__(self, tmp_path, **env):
        self.root = tmp_path
        self.chunks = write_chunks(tmp_path / "data" / "processed" / "chunks.jsonl")
        environ = {"VECTORSTORE": "numpy", "GEMINI_API_KEY": "", **env}
        self.settings = load_settings(root=tmp_path, environ=environ)

    def loader(self, env_file=None):
        return self.settings

    def run_build(self, *argv, embedder=None, sources=None):
        lines: list[str] = []
        code = build_cli.main(
            list(argv), make_embedder_fn=lambda name, **kw: embedder or HashingEmbedder(), settings_loader=self.loader,
            sources_loader=lambda settings: ragfixtures.by_source() if sources is None else sources, write=lines.append,
        )  # fmt: skip
        return code, "\n".join(lines)


# ----------------------------------------------------------------------------------------------- build_index


def test_build_index_builds_and_reports(tmp_path):
    env = Env(tmp_path)
    code, out = env.run_build()
    assert code == 0, out
    assert "Embedding 10 passages" in out and "Built the numpy index:" in out
    assert "10 passages from 6 documents" in out and "hashing-512" in out and "Next: python scripts/ask.py" in out
    assert "embedding" in out and "100%" in out
    opened = open_index(env.settings.vectorstore_dir, "numpy", HashingEmbedder(), chunks_path=env.chunks)
    assert opened.info.n_records == 10 and opened.warnings == ()


def test_build_index_mentions_a_download_only_for_models_that_are_downloaded(tmp_path):
    code, out = Env(tmp_path / "baseline").run_build("--embedding-model", "hashing-512")
    assert code == 0, out
    assert "needs no download" in out and "130 MB" not in out
    code, out = Env(tmp_path / "default").run_build()
    assert code == 0, out
    assert "downloads the model, about 130 MB" in out and "needs no download" not in out


def test_build_index_with_both_backends_embeds_once(tmp_path, monkeypatch):
    from fakechroma import FakeChromaClient

    from reglens.vectorstore import ChromaStore, NumpyStore

    env = Env(tmp_path)
    calls: list[int] = []

    class Counting(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            calls.append(len(texts))
            return super().embed_documents(texts, progress)

    def fake_open(directory, backend="chroma", *, space="cosine"):
        if backend == "chroma":
            return ChromaStore(directory / backend, client=FakeChromaClient())
        return NumpyStore(directory / backend, space=space)

    monkeypatch.setattr(build_cli, "open_store", fake_open)
    code, out = env.run_build("--backend", "both", embedder=Counting())
    assert code == 0, out
    assert calls == [10]  # embedded once, stored twice
    assert "Built the chroma index:" in out and "Built the numpy index:" in out
    assert info_path(env.settings.vectorstore_dir, "chroma").is_file() and info_path(env.settings.vectorstore_dir, "numpy").is_file()


def test_build_index_reports_the_token_overflow(tmp_path):
    class Measuring(HashingEmbedder):
        max_tokens = 5

        def token_counts(self, texts):
            return [3, 9, 9, 9, 3, 3, 3, 3, 3, 12]

    code, out = Env(tmp_path).run_build(embedder=Measuring())
    assert code == 0 and "4 passages (40.0%) are longer than the model's 5-token window (longest 12 tokens)" in out


def test_build_index_fresh_removes_a_damaged_index(tmp_path):
    env = Env(tmp_path)
    assert env.run_build()[0] == 0
    damaged = env.settings.vectorstore_dir / "numpy" / "vectors.npy"
    damaged.write_bytes(b"garbage")
    code, out = env.run_build()
    assert code == 1 and "Problem:" in out  # the old, broken index is in the way
    code, out = env.run_build("--fresh")
    assert code == 0, out
    assert open_index(env.settings.vectorstore_dir, "numpy", HashingEmbedder()).info.n_records == 10


def test_build_index_problems_are_reported_in_plain_words(tmp_path):
    env = Env(tmp_path)
    code, out = env.run_build("--chunks", str(tmp_path / "nothing.jsonl"))
    assert code == 1 and "does not exist" in out and "scripts/ingest.py" in out

    code, out = env.run_build(sources={})
    assert code == 1 and "manifest" in out and "could not be read" in out

    other = ragfixtures.by_source()
    del other["gdpr"]
    code, out = env.run_build(sources=other)
    assert code == 1 and "not in the manifest" in out and "gdpr" in out

    class NoModel(HashingEmbedder):
        def embed_documents(self, texts, progress=None):
            raise EmbeddingError("could not load the embedding model 'x' (OSError: offline)")

    code, out = env.run_build(embedder=NoModel())
    assert code == 1 and "could not load the embedding model" in out

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"chunk_id": "x"}\n', encoding="utf-8")
    code, out = env.run_build("--chunks", str(bad))
    assert code == 1 and "line 1" in out

    code, out = env.run_build("--batch-size", "0")
    assert code == 2 and "--batch-size" in out
    assert not info_path(env.settings.vectorstore_dir, "numpy").exists()


def test_build_index_names_the_cloud_folder_when_it_moved_the_index(tmp_path):
    root = tmp_path / "OneDrive" / "project"
    env = Env.__new__(Env)
    env.root, env.chunks = root, write_chunks(root / "data" / "processed" / "chunks.jsonl")
    env.settings = load_settings(root=root, environ={"VECTORSTORE": "numpy", "LOCALAPPDATA": str(tmp_path / "local")})
    code, out = env.run_build()
    assert code == 0 and "inside OneDrive" in out and str(tmp_path / "local") in out


# ----------------------------------------------------------------------------------------------- ask


def make_runtime(tmp_path, *replies, key=False, **env):
    environment = Env(tmp_path, **env)
    store = open_store(environment.settings.vectorstore_dir, "numpy")
    from reglens.indexing import build_index

    build_index(
        ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), store,
        info_file=info_path(environment.settings.vectorstore_dir, "numpy"), backend="numpy",
    )  # fmt: skip
    llm = FakeLLM(*replies) if replies else None
    runtime = build_runtime(environment.settings, embedder=HashingEmbedder(), reranker=StubReranker(), llm=llm)
    return environment, runtime, llm


def run_ask(environment, runtime, *argv, inputs=()):
    lines: list[str] = []
    queue = list(inputs)

    def read_line(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    code = ask_cli.main(
        list(argv), build=lambda settings, **kw: runtime, settings_loader=environment.loader, read_line=read_line, write=lines.append,
    )  # fmt: skip
    return code, "\n".join(lines)


def test_ask_prints_the_answer_the_sources_and_the_checks(tmp_path):
    environment, runtime, llm = make_runtime(tmp_path, "The controller must notify the supervisory authority within 48 hours [S1][S9].")
    code, out = run_ask(environment, runtime, "How quickly must we report a personal data breach to the supervisory authority?")
    assert code == 0
    assert "ANSWER     (fake/fake-1" in out and "within 48 hours [S1][S9]." in out
    assert " * [S1] General Data Protection Regulation (GDPR)" in out and "   [S2]" in out
    assert "EU regulation | in force | 2016 | Chapter IV > Article 33" in out and "p. 52" in out
    assert "cites S9, which do not exist among the sources" in out and out.count("cites S9") == 1  # said once, not twice
    assert "WARNING: figures in the answer that appear in no source: 48" in out
    assert "100% of the factual sentences carry a citation" in out and "https://example.org/gdpr" in out
    assert len(llm.requests) == 1


def test_ask_marks_drafts_and_shows_full_passages_on_request(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "Draft says so [S1].")
    code, out = run_ask(environment, runtime, "resume ranking job applicants", "--filter", "doc_id=draft-hr", "--full")
    flat = " ".join(out.split())  # long lines are wrapped to the width of the console
    assert code == 0 and "DRAFT - not adopted" in out and "rank or filter job applicants are likely to fall under Annex III point 4 on employment" in flat


def test_ask_without_a_language_model_shows_the_passages_and_how_to_fix_it(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path)
    code, out = run_ask(environment, runtime, "right to erasure of personal data")
    assert code == 0 and "no language-model answer" in out and "GEMINI_API_KEY" in out and "Fix: Add your API key" in out
    assert "[S1]" in out and "gdpr:0002" not in out  # chunk ids are internal; the reader sees titles and sections


def test_ask_no_llm_is_not_an_error(tmp_path):
    environment, runtime, llm = make_runtime(tmp_path, "unused")
    code, out = run_ask(environment, runtime, "personal data breach", "--no-llm")
    assert code == 0 and "no language-model answer" in out and "Why:" not in out and llm.requests == []


def test_ask_failing_model_shows_the_kind_specific_hint(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, LLMError("quota", "daily limit reached"))
    code, out = run_ask(environment, runtime, "personal data breach")
    assert code == 0 and "Why: daily limit reached" in out and "Fix: The free quota of this model is used up" in out


def test_ask_both_shows_the_baseline_below_the_rag_answer(tmp_path):
    environment, runtime, llm = make_runtime(tmp_path, "With sources [S1].", "From memory.")
    code, out = run_ask(environment, runtime, "When do the AI Act's high-risk obligations apply?", "--both")
    assert code == 0 and out.index("With sources [S1].") < out.index("the same question, without the documents") < out.index("From memory.")
    assert "nothing here has been checked against a source" in out
    assert len(llm.requests) == 2 and llm.requests[1].messages[0].content == "When do the AI Act's high-risk obligations apply?"


def test_ask_llm_only_uses_no_retrieval(tmp_path):
    environment, runtime, llm = make_runtime(tmp_path, "From memory.")
    code, out = run_ask(environment, runtime, "anything", "--llm-only")
    assert code == 0 and "From memory." in out and "SOURCES" not in out and len(llm.requests) == 1


def test_ask_llm_only_needs_a_model(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path)
    code, out = run_ask(environment, runtime, "anything", "--both")
    assert code == 1 and "needs a language model" in out and "GEMINI_API_KEY" in out


def test_ask_json_output_is_machine_readable(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "Within 72 hours [S1].")
    code, out = run_ask(environment, runtime, "personal data breach supervisory authority", "--json")
    data = json.loads(out)
    assert code == 0 and data["mode"] == "rag" and data["sources"][0]["label"] == "S1" and data["cited"] == ["S1"]

    environment, runtime, _ = make_runtime(tmp_path / "second", "A [S1].", "B.")
    code, out = run_ask(environment, runtime, "personal data breach", "--both", "--json")
    data = json.loads(out)
    assert set(data) == {"rag", "llm_only"} and data["llm_only"]["text"] == "B."


def test_ask_search_options_reach_the_pipeline(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "x [S1].")
    code, out = run_ask(
        environment, runtime, "personal data", "--json", "--mode", "lexical", "--k", "2", "--rerank", "--filter", "jurisdiction=US",
        "--filter", "status=in_force,final", "--no-notes", "--no-glossary", "--max-words", "500", "--max-tokens", "300",
    )  # fmt: skip
    options = json.loads(out)["options"]
    assert options["retrieval"]["mode"] == "lexical" and options["retrieval"]["top_k"] == 2 and options["retrieval"]["rerank"] is True
    assert options["retrieval"]["where"] == {"jurisdiction": "US", "status": ["in_force", "final"]}
    assert options["retrieval"]["include_notes"] is False and options["transform"]["glossary"] is False
    assert options["max_context_words"] == 500 and options["max_output_tokens"] == 300


def test_ask_rejects_bad_filters_and_questions(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "x")
    for bad in ("nonsense", "color=red", "status="):
        code, out = run_ask(environment, runtime, "question", "--filter", bad)
        assert code == 2 and "--filter" in out
    code, out = run_ask(environment, runtime, "   ")
    assert code == 2 and "empty" in out


def test_ask_lists_documents(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "x")
    code, out = run_ask(environment, runtime, "--documents")
    assert code == 0 and "6 documents, 10 passages" in out and "omnibus" in out and "Digital Omnibus on AI amending the AI Act" in out


def test_ask_without_an_index_says_what_to_do(tmp_path):
    environment = Env(tmp_path)

    def build(settings, **kw):
        raise IndexNotReady("there is no numpy index in X yet. Build it first: python scripts/build_index.py")

    lines: list[str] = []
    code = ask_cli.main(["question"], build=build, settings_loader=environment.loader, write=lines.append)
    assert code == 1 and "build_index.py" in lines[0]


def test_ask_reports_a_model_that_cannot_be_loaded(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "x")

    class Offline(HashingEmbedder):
        def embed_query(self, text):
            raise EmbeddingError("could not load the embedding model 'x' (OSError: offline)")

    runtime.retriever._embedder = Offline()
    code, out = run_ask(environment, runtime, "personal data breach", "--mode", "dense")
    assert code == 1 and "Problem: could not load the embedding model" in out


def test_the_chat_keeps_the_conversation_and_understands_commands(tmp_path):
    def reply(request):
        if "rewrite" in request.system.lower():
            return "Which deadline applies to a breach of health information in the US?"
        return "Sixty days [S1]." if "60 days" in request.messages[0].content else "Within 72 hours [S1]."

    environment, runtime, llm = make_runtime(tmp_path, "Within 72 hours [S1].", reply)
    code, out = run_ask(
        environment, runtime,
        inputs=["", "/help", "How quickly must we report a data breach to the supervisory authority?", "and in the US?", "/docs", "/clear", "   ", "/quit"],
    )  # fmt: skip
    assert code == 0
    assert "RegLens: 10 passages from 6 documents" in out and "Language model: fake/fake-1" in out
    assert "/clear   forget the conversation" in out and "6 documents, 10 passages" in out and "(conversation cleared)" in out
    assert "Searched   Which deadline applies to a breach of health information in the US?" in out
    assert out.count("ANSWER     (fake/fake-1") == 2


def test_the_chat_survives_a_bad_question_and_ends_at_end_of_input(tmp_path):
    environment, runtime, _ = make_runtime(tmp_path, "ok [S1].")
    code, out = run_ask(environment, runtime, inputs=["x" * 3000, "personal data breach"])
    assert code == 0 and "Problem: the question is too long" in out and "ANSWER" in out


def test_the_chat_banner_explains_a_missing_model_and_warnings(tmp_path):
    root = tmp_path / "OneDrive" / "p"
    environment = Env.__new__(Env)
    environment.root, environment.chunks = root, write_chunks(root / "data" / "processed" / "chunks.jsonl")
    environment.settings = load_settings(root=root, environ={"VECTORSTORE": "numpy", "LOCALAPPDATA": str(tmp_path / "local"), "GEMINI_API_KEY": ""})
    store = open_store(environment.settings.vectorstore_dir, "numpy")
    from reglens.indexing import build_index

    build_index(ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), store, info_file=info_path(environment.settings.vectorstore_dir, "numpy"), backend="numpy")
    runtime = build_runtime(environment.settings, embedder=HashingEmbedder(), reranker=StubReranker())
    code, out = run_ask(environment, runtime, inputs=[])
    assert code == 0 and "No language model (GEMINI_API_KEY is empty" in out and "Note: The project is inside OneDrive" in out


def test_parse_filters_directly():
    assert ask_cli.parse_filters([]) == {}
    assert ask_cli.parse_filters(["jurisdiction=EU", "doc_type = guidelines , regulation "]) == {"jurisdiction": "EU", "doc_type": ["guidelines", "regulation"]}
    with pytest.raises(ValueError, match="unknown key"):
        ask_cli.parse_filters(["title=GDPR"])
