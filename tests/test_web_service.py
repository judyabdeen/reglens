from __future__ import annotations

import json
import logging
import threading

import pytest
import ragfixtures
from ragfixtures import FakeLLM
from webfixtures import BREACH, SECRET, Reranker, build_the_index, make_service, scripted, settings_for

from reglens.embeddings import EmbeddingError, HashingEmbedder
from reglens.generation.prompts import LLM_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT
from reglens.indexing import IndexNotReady
from reglens.llm.base import LLMError
from reglens.retrieval.query_transform import CONDENSE_SYSTEM
from reglens.runtime import build_runtime
from reglens.web import service as web
from reglens.web.service import (
    DEFAULT_OPTIONS,
    Busy,
    NotFound,
    NotReady,
    RequestError,
    Unavailable,
    WebService,
    json_safe,
    parse_ask_request,
)

# ----------------------------------------------------------------------------------------------- parsing


def test_a_minimal_request_gets_the_defaults():
    request = parse_ask_request({"question": "  What   is\nGDPR? "})
    assert request.question == "What is GDPR?" and request.mode == "rag" and request.history == ()
    r, t = request.options.retrieval, request.options.transform
    assert (r.mode, r.top_k, r.candidates, r.rerank_pool, r.rerank, r.where, r.include_notes) == ("hybrid", 8, 40, 20, False, None, True)
    assert (t.glossary, t.condense, t.multi_query, t.hyde) == (True, True, 0, False)
    assert request.options.use_llm is True
    assert DEFAULT_OPTIONS == {"search": "hybrid", "top_k": 8, "rerank": False, "glossary": True, "condense": True, "multi_query": 0, "hyde": False, "include_notes": True}


@pytest.mark.parametrize("mode, uses_model", [("rag", True), ("compare", True), ("passages", False), ("llm_only", False)])
def test_the_mode_decides_whether_the_pipeline_may_call_the_model(mode, uses_model):
    assert parse_ask_request({"question": "q", "mode": mode}).options.use_llm is uses_model


def test_every_setting_reaches_the_pipeline_options():
    request = parse_ask_request({
        "question": "q",
        "options": {"search": "dense", "top_k": 12, "rerank": True, "glossary": False, "condense": False, "multi_query": 2, "hyde": True, "include_notes": False},
        "filters": {"jurisdiction": "EU"},
    })  # fmt: skip
    r, t = request.options.retrieval, request.options.transform
    assert (r.mode, r.top_k, r.candidates, r.rerank_pool, r.rerank, r.include_notes) == ("dense", 12, 40, 20, True, False)
    assert dict(r.where) == {"jurisdiction": "EU"}
    assert (t.glossary, t.condense, t.multi_query, t.hyde) == (False, False, 2, True)


def test_the_candidate_pools_never_shrink_below_the_number_of_passages():
    assert parse_ask_request({"question": "q", "options": {"top_k": 12}}).options.retrieval.candidates == 40


@pytest.mark.parametrize(
    "payload, field",
    [
        (["question"], None),
        (None, None),
        ("question", None),
        ({}, "question"),
        ({"question": ""}, "question"),
        ({"question": "  \n "}, "question"),
        ({"question": 5}, "question"),
        ({"question": "x" * 2001}, "question"),
        ({"question": "q", "extra": 1}, "extra"),
        ({"question": "q", "mode": "everything"}, "mode"),
        ({"question": "q", "history": "hi"}, "history"),
        ({"question": "q", "history": ["hi"]}, "history"),
        ({"question": "q", "history": [{"question": 1, "answer": "a"}]}, "history question"),
        ({"question": "q", "history": [{"question": "q", "answer": "a" * 6001}]}, "history answer"),
        ({"question": "q", "options": "fast"}, "options"),
        ({"question": "q", "options": {"speed": 1}}, "options.speed"),
        ({"question": "q", "options": {"search": "fuzzy"}}, "options.search"),
        ({"question": "q", "options": {"top_k": 0}}, "options.top_k"),
        ({"question": "q", "options": {"top_k": 13}}, "options.top_k"),
        ({"question": "q", "options": {"top_k": "8"}}, "options.top_k"),
        ({"question": "q", "options": {"top_k": 2.5}}, "options.top_k"),
        ({"question": "q", "options": {"top_k": True}}, "options.top_k"),
        ({"question": "q", "options": {"multi_query": -1}}, "options.multi_query"),
        ({"question": "q", "options": {"multi_query": 4}}, "options.multi_query"),
        ({"question": "q", "options": {"rerank": "yes"}}, "options.rerank"),
        ({"question": "q", "options": {"hyde": 1}}, "options.hyde"),
        ({"question": "q", "options": {"include_notes": None}}, "options.include_notes"),
        ({"question": "q", "filters": "EU"}, "filters"),
        ({"question": "q", "filters": {"colour": "red"}}, "filters"),
        ({"question": "q", "filters": {"status": 5}}, "filters.status"),
        ({"question": "q", "filters": {"status": ["final", 5]}}, "filters.status"),
        ({"question": "q", "filters": {"doc_id": [f"d{i}" for i in range(61)]}}, "filters.doc_id"),
    ],
)  # fmt: skip
def test_a_malformed_request_is_refused_with_the_field_that_is_wrong(payload, field):
    with pytest.raises(RequestError) as caught:
        parse_ask_request(payload)
    assert caught.value.status == 400 and caught.value.kind == "bad_request" and caught.value.field == field
    assert str(caught.value)


def test_the_boundaries_of_the_numeric_settings_are_accepted():
    for options in ({"top_k": 1, "multi_query": 0}, {"top_k": 12, "multi_query": 3}):
        parse_ask_request({"question": "q", "options": options})
    assert parse_ask_request({"question": "x" * 2000}).question == "x" * 2000


def test_the_history_keeps_the_last_turns_and_drops_empty_ones():
    turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(9)] + [{"question": "  ", "answer": "a"}, {"question": "q", "answer": ""}]
    history = parse_ask_request({"question": "and in the US?", "history": turns}).history
    assert [(t.question, t.answer) for t in history] == [("q5", "a5"), ("q6", "a6"), ("q7", "a7"), ("q8", "a8")]  # the last 6 entries, minus the empty ones
    spaced = parse_ask_request({"question": "q", "history": [{"question": " a \n b ", "answer": "  c  d "}]}).history
    assert (spaced[0].question, spaced[0].answer) == ("a b", "c d")


def test_filters_are_normalised():
    where = parse_ask_request({
        "question": "q",
        "filters": {"jurisdiction": [" EU ", "US", "EU"], "status": "final", "doc_type": [], "doc_id": ["", "  "]},
    }).options.retrieval.where  # fmt: skip
    assert dict(where) == {"jurisdiction": ["EU", "US"], "status": "final"}  # duplicates removed, a single value stays a text, empty lists vanish
    assert parse_ask_request({"question": "q", "filters": {}}).options.retrieval.where is None


def test_filter_values_are_checked_against_the_documents_of_the_index():
    catalog = [{"doc_id": "a", "jurisdiction": "EU", "status": "final"}, {"doc_id": "b", "jurisdiction": "US", "status": "draft"}]
    good = parse_ask_request({"question": "q", "filters": {"jurisdiction": ["EU", "US"], "doc_id": "a"}}, catalog)
    assert dict(good.options.retrieval.where) == {"jurisdiction": ["EU", "US"], "doc_id": "a"}
    with pytest.raises(RequestError) as caught:
        parse_ask_request({"question": "q", "filters": {"jurisdiction": "Mars"}}, catalog)
    assert "Mars" in str(caught.value) and "Choose from: EU, US." in str(caught.value) and caught.value.field == "filters.jurisdiction"
    many = [{"doc_id": f"d{i}"} for i in range(13)]
    with pytest.raises(RequestError) as caught:
        parse_ask_request({"question": "q", "filters": {"doc_id": "zzz"}}, many)
    assert "Choose from" not in str(caught.value)  # a long list is not printed


def test_json_safe_keeps_plain_values_and_removes_what_json_cannot_carry():
    data = {"a": (1, 2.5, float("nan")), "b": {"c": float("inf"), "d": None, "e": True, "f": "text", "g": [float("-inf")]}}
    assert json_safe(data) == {"a": [1, 2.5, None], "b": {"c": None, "d": None, "e": True, "f": "text", "g": [None]}}
    json.dumps(json_safe(data), allow_nan=False)


# ----------------------------------------------------------------------------------------------- the pages


def test_status_describes_the_running_system_and_never_a_secret(tmp_path):
    service = make_service(tmp_path, FakeLLM("x"))
    status = service.status()
    assert status["ready"] is True and status["app"] == "RegLens" and status["defaults"] == DEFAULT_OPTIONS
    assert status["index"]["documents"] == 6 and status["index"]["passages"] == 10 and status["index"]["embedding_model"] == "hashing-512"
    assert status["llm"] == {"available": True, "provider": "fake", "model": "fake-1", "problem": None}
    assert status["limits"]["top_k"] == [1, 12] and status["limits"]["question_chars"] == 2000 and status["warnings"] == []
    assert SECRET not in json.dumps(status)


def test_status_reports_a_missing_language_model_in_plain_words(tmp_path):
    llm = make_service(tmp_path, None).status()["llm"]
    assert llm["available"] is False and "GEMINI_API_KEY is empty" in llm["problem"] and llm["model"] is None


def test_a_missing_index_is_reported_and_retried_later(tmp_path):
    calls, now = [], [100.0]

    def factory(settings):
        calls.append(1)
        if len(calls) < 3:
            raise IndexNotReady("There is no index yet. Build it first: python scripts/build_index.py")
        return build_runtime(settings, embedder=HashingEmbedder(), reranker=Reranker(), llm=FakeLLM("x"))

    settings = settings_for(tmp_path)
    build_the_index(settings)
    service = WebService(settings, runtime_factory=factory, retry_seconds=3.0, clock=lambda: now[0])
    first = service.status()
    assert first["ready"] is False and first["problem"] == "There is no index yet. Build it first." and first["fix"] == web.FIX_HINT
    assert "index" not in first and first["defaults"] == DEFAULT_OPTIONS
    for call in (service.documents, lambda: service.passage("gdpr:0001"), lambda: service.ask({"question": "q"})):
        with pytest.raises(NotReady) as caught:
            call()
        assert caught.value.status == 503 and caught.value.kind == "not_ready"
    assert len(calls) == 1  # a page that polls does not make the server reopen the index every time
    now[0] += 3.5
    assert service.status()["ready"] is False and len(calls) == 2
    now[0] += 3.5
    ready = service.status()
    assert ready["ready"] is True and len(calls) == 3 and "problem" not in ready
    assert service.status()["ready"] is True and len(calls) == 3  # built once, then kept


@pytest.mark.parametrize("error", [OSError("disk is locked"), EmbeddingError("model missing")])
def test_other_startup_failures_are_also_shown_not_crashed_on(tmp_path, error):
    def factory(settings):
        raise error

    service = WebService(settings_for(tmp_path), runtime_factory=factory)
    status = service.status()
    assert status["ready"] is False and status["problem"] == web.tidy_problem(error)
    assert str(error).lower() in status["problem"].lower()


@pytest.mark.parametrize(
    "message, expected",
    [
        ("there is no numpy index in data/index yet. Build it first: python scripts/build_index.py",
         "There is no numpy index in data/index yet. Build it first."),
        ("the index was written by a different version of RegLens (format 1); rebuild it: python scripts/build_index.py",
         "The index was written by a different version of RegLens (format 1); rebuild it."),
        ("the numpy index in data/index was never finished (the build was interrupted). Rebuild it: python scripts/build_index.py",
         "The numpy index in data/index was never finished (the build was interrupted). Rebuild it."),
        ("the index was built with the embedding model 'a', but questions would now use 'b'. Set EMBEDDING_MODEL=a in .env, "
         "or rebuild the index: python scripts/build_index.py",
         "The index was built with the embedding model 'a', but questions would now use 'b'. Set EMBEDDING_MODEL=a in .env, or rebuild the index."),
        ("the index has vectors of size 512, the embedding model produces 384; rebuild the index",
         "The index has vectors of size 512, the embedding model produces 384; rebuild the index."),
        ("  disk   is\nlocked  ", "Disk is locked."),
        ("Already a sentence!", "Already a sentence!"),
        ("python scripts/build_index.py", "RegLens could not open its library."),
        ("", "RegLens could not open its library."),
    ],
)  # fmt: skip
def test_a_start_up_problem_becomes_a_sentence_without_the_command(message, expected):
    """The command is shown in its own box on the page, so the sentence must not repeat it."""
    assert web.tidy_problem(message) == expected
    assert "build_index.py" not in web.tidy_problem(message)


def test_the_real_missing_index_message_reads_well_on_the_page(tmp_path):
    from reglens.indexing import open_index

    with pytest.raises(IndexNotReady) as caught:
        open_index(tmp_path / "nothing-here", "numpy", HashingEmbedder())
    sentence = web.tidy_problem(caught.value)
    assert sentence.startswith("There is no numpy index in ") and sentence.endswith(" yet. Build it first.")
    assert "build_index.py" not in sentence


def test_documents_lists_the_indexed_documents(tmp_path):
    documents = make_service(tmp_path, FakeLLM("x")).documents()["documents"]
    assert [d["doc_id"] for d in documents] == sorted(s.id for s in ragfixtures.SOURCES)
    ai_act = next(d for d in documents if d["doc_id"] == "ai-act")
    assert ai_act["passages"] == 2 and ai_act["status"] == "in_force" and ai_act["jurisdiction"] == "EU"


def test_a_passage_comes_with_the_passages_around_it(tmp_path):
    service = make_service(tmp_path, FakeLLM("x"))
    page = service.passage("gdpr:0002")
    assert page["passage"]["chunk_id"] == "gdpr:0002" and page["passage"]["pages"] == "pp. 43-44"
    assert page["passage"]["status_text"] == "in force" and "Article 17" in page["passage"]["section"] and page["passage"]["text"].startswith("The data subject")
    assert page["before"]["chunk_id"] == "gdpr:0001" and page["after"]["chunk_id"] == "gdpr:0003"
    assert service.passage("omnibus:0001")["before"] is None and service.passage("gdpr:0003")["after"] is None
    with pytest.raises(NotFound) as caught:
        service.passage("nope:1")
    assert caught.value.status == 404
    for bad in ("", None, 5, "x" * 201):
        with pytest.raises(RequestError):
            service.passage(bad)


# ----------------------------------------------------------------------------------------------- questions


def test_a_question_gets_a_grounded_answer_with_its_evidence(tmp_path, caplog):
    llm = scripted()
    service = make_service(tmp_path, llm)
    with caplog.at_level(logging.INFO, logger="reglens.web"):
        result = service.ask({"question": BREACH})
    assert result["mode"] == "rag" and result["baseline"] is None and result["elapsed"] >= 0
    answer = result["answer"]
    assert answer["mode"] == "rag" and answer["text"] == "Notify within 72 hours [S1]." and answer["cited"] == ["S1"] and answer["hint"] is None
    top = answer["sources"][0]
    assert top["chunk_id"] == "gdpr:0001" and top["cited"] is True and top["status_text"] == "in force" and top["pages"] == "p. 52"
    assert top["dense_rank"] is not None and top["fused_score"] > 0
    assert answer["options"]["retrieval"]["top_k"] == 8 and answer["timings"]["total"] > 0
    json.dumps(result, allow_nan=False)
    assert "answered mode=rag" in caplog.text and BREACH not in caplog.text  # what people ask is not written to the log
    assert SECRET not in json.dumps(result) and SECRET not in caplog.text


def test_passages_only_never_calls_the_model(tmp_path):
    llm = scripted()
    result = make_service(tmp_path, llm).ask({"question": BREACH, "mode": "passages"})
    assert result["answer"]["mode"] == "extractive" and result["answer"]["sources"] and result["answer"]["hint"] is None and result["baseline"] is None
    assert llm.requests == []


def test_the_filters_of_the_request_narrow_the_search(tmp_path):
    result = make_service(tmp_path, scripted()).ask({"question": "personal data", "mode": "passages", "filters": {"jurisdiction": "US"}})
    assert {s["jurisdiction"] for s in result["answer"]["sources"]} == {"US"}
    with pytest.raises(RequestError):
        make_service(tmp_path / "other", scripted()).ask({"question": "q", "filters": {"jurisdiction": "Mars"}})


def test_compare_asks_the_model_alone_the_standalone_question(tmp_path):
    llm = scripted(standalone="How quickly must a data breach be reported in the US?")
    history = [{"question": BREACH, "answer": "Within 72 hours [S1]."}]
    result = make_service(tmp_path, llm).ask({"question": "and in the US?", "mode": "compare", "history": history})
    answer, baseline = result["answer"], result["baseline"]
    assert answer["mode"] == "rag" and baseline["mode"] == "llm_only"
    assert answer["standalone_question"] == "How quickly must a data breach be reported in the US?"
    assert baseline["question"] == answer["standalone_question"] and baseline["text"] == "Within 72 hours." and baseline["sources"] == []
    systems = [request.system for request in llm.requests]
    assert systems == [CONDENSE_SYSTEM, SYSTEM_PROMPT, LLM_ONLY_SYSTEM_PROMPT]
    assert llm.requests[2].messages[0].content == answer["standalone_question"]  # the model alone sees no sources


def test_compare_does_not_ask_again_when_the_model_just_failed(tmp_path):
    llm = FakeLLM(LLMError("quota", "daily quota used up"))
    result = make_service(tmp_path, llm).ask({"question": BREACH, "mode": "compare"})
    assert result["answer"]["mode"] == "extractive" and result["answer"]["error_kind"] == "quota" and result["answer"]["sources"]
    assert "free quota" in result["answer"]["hint"] and result["baseline"] is None
    assert len(llm.requests) == 1


def test_without_a_language_model_the_page_gets_passages_and_a_hint(tmp_path):
    result = make_service(tmp_path, None).ask({"question": BREACH, "mode": "compare"})
    assert result["answer"]["mode"] == "extractive" and result["answer"]["error_kind"] == "unavailable" and result["answer"]["sources"]
    assert "No language model is set up" in result["answer"]["hint"] and result["baseline"] is None
    assert "GEMINI_API_KEY is empty" in result["answer"]["error"]
    alone = make_service(tmp_path / "again", None).ask({"question": BREACH, "mode": "llm_only"})
    assert alone["baseline"]["error_kind"] == "unavailable" and "No language model is set up" in alone["baseline"]["hint"]


def test_every_failure_kind_has_a_hint_the_page_can_show():
    from reglens.llm import base

    kinds = {getattr(base, name) for name in ("AUTH", "FORBIDDEN", "QUOTA", "RATE_LIMITED", "OVERLOADED", "SERVER", "TIMEOUT", "NETWORK", "BAD_REQUEST", "NOT_FOUND", "BLOCKED", "EMPTY", "UNKNOWN")}
    assert kinds | {"unavailable"} == set(web.HINTS)
    assert all(len(text) > 20 and text.endswith(".") for text in web.HINTS.values())


def test_the_model_alone_can_be_asked_without_any_retrieval(tmp_path):
    llm = scripted()
    result = make_service(tmp_path, llm).ask({"question": BREACH, "mode": "llm_only"})
    assert result["answer"] is None and result["baseline"]["text"] == "Within 72 hours." and result["baseline"]["mode"] == "llm_only"
    assert [request.system for request in llm.requests] == [LLM_ONLY_SYSTEM_PROMPT]


def test_an_invalid_request_costs_nothing(tmp_path):
    llm = scripted()
    service = make_service(tmp_path, llm, max_concurrent=1)
    for payload in ({"question": ""}, {"question": "q", "mode": "x"}, {"question": "q", "options": {"top_k": 99}}):
        with pytest.raises(RequestError):
            service.ask(payload)
    assert llm.requests == []
    assert service.ask({"question": BREACH})["answer"]["mode"] == "rag"  # the single slot was not used up


def test_a_second_question_while_one_is_running_is_turned_away_politely(tmp_path):
    started, release = threading.Event(), threading.Event()

    def slow(request):
        started.set()
        assert release.wait(5)
        return "Notify within 72 hours [S1]."

    service = make_service(tmp_path, FakeLLM(slow), max_concurrent=1)
    results = []
    worker = threading.Thread(target=lambda: results.append(service.ask({"question": BREACH})))
    worker.start()
    assert started.wait(5)
    with pytest.raises(Busy) as caught:
        service.ask({"question": BREACH})
    assert caught.value.status == 429 and "still being answered" in str(caught.value)
    release.set()
    worker.join(5)
    assert results and results[0]["answer"]["cited"] == ["S1"]
    assert service.ask({"question": BREACH})["answer"]["mode"] == "rag"  # the slot is free again


class BrokenQueries(HashingEmbedder):
    def embed_query(self, text):
        raise EmbeddingError("the embedding model could not be loaded")


def test_a_failing_search_component_is_reported_and_frees_the_slot(tmp_path):
    service = make_service(tmp_path, scripted(), embedder=BrokenQueries(), max_concurrent=1)
    for _ in range(2):  # the second attempt proves the slot was released
        with pytest.raises(Unavailable) as caught:
            service.ask({"question": BREACH})
        assert caught.value.status == 503 and "could not be loaded" in str(caught.value)


def test_warming_up_never_raises(tmp_path, caplog):
    make_service(tmp_path, scripted()).warm_up()  # fine
    broken = make_service(tmp_path / "b", scripted(), embedder=BrokenQueries())
    with caplog.at_level(logging.WARNING, logger="reglens.web"):
        broken.warm_up()
    assert "warm-up failed: EmbeddingError" in caplog.text

    def refuses(settings):
        raise IndexNotReady("no index")

    WebService(settings_for(tmp_path / "c"), runtime_factory=refuses).warm_up()


def test_an_unexpected_bug_is_not_hidden_as_a_friendly_problem(tmp_path):
    def broken(settings):
        raise ZeroDivisionError("a real bug")

    service = WebService(settings_for(tmp_path), runtime_factory=broken)
    with pytest.raises(ZeroDivisionError):
        service.status()
