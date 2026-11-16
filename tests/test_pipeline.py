from __future__ import annotations

import json
import threading

import pytest
from ragfixtures import ConceptEmbedder, FakeLLM, build

from reglens.generation.answer import EXTRACTIVE, LLM_ONLY, RAG
from reglens.generation.prompts import LLM_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT
from reglens.llm.base import LLMError
from reglens.pipeline import MAX_QUESTION_CHARS, NO_PASSAGES, AskOptions, RAGPipeline
from reglens.retrieval import RerankError, RetrievalOptions, TransformOptions, Turn

BREACH_Q = "How quickly must we report a data leak to the regulator?"


def pipeline(*replies, llm=None, reason=None, embedder=None):
    _, retriever, embedder, _ = build(embedder)
    model = llm if llm is not None else (FakeLLM(*replies) if replies else None)
    return RAGPipeline(retriever, model, llm_unavailable=reason), model, embedder


def test_a_grounded_answer_with_its_sources_and_checks():
    pipe, llm, _ = pipeline("The controller must notify the supervisory authority within 72 hours [S1].")
    answer = pipe.ask(BREACH_Q)
    assert answer.mode == RAG and not answer.abstained and answer.error is None
    assert answer.text == "The controller must notify the supervisory authority within 72 hours [S1]."
    top = answer.sources[0]
    assert top.label == "S1" and top.chunk_id == "gdpr:0001" and top.cited is True
    assert answer.cited == ("S1",) and answer.invalid_citations == () and answer.unverified_figures == ()
    assert answer.citation_coverage == 1.0
    assert (answer.provider, answer.model) == ("fake", "fake-1") and answer.input_tokens == 120 and answer.output_tokens == 40
    assert set(answer.timings) == {"transform", "retrieve", "generate", "total"} and answer.timings["total"] >= answer.timings["generate"]
    assert all(not s.cited for s in answer.sources[1:] if s.label != "S1")
    assert top.title.startswith("General Data Protection") and top.status_text == "in force" and top.pages == "p. 52"
    assert "Article 33" in top.location and top.is_law and top.url.endswith("gdpr")


def test_sources_carry_the_rank_of_each_search_and_the_fused_score():
    pipe, _, _ = pipeline()
    answer = pipe.ask(BREACH_Q, options=AskOptions(use_llm=False))
    top = answer.sources[0]
    assert top.chunk_id == "gdpr:0001" and top.dense_rank == 1 and top.fused_score > 0
    assert all(s.dense_rank is not None or s.lexical_rank is not None for s in answer.sources)
    exported = answer.to_dict()["sources"][0]
    assert (exported["dense_rank"], exported["lexical_rank"], exported["fused_score"]) == (top.dense_rank, top.lexical_rank, top.fused_score)


def test_the_prompt_carries_the_rules_the_sources_and_the_standalone_question():
    pipe, llm, _ = pipeline("Answer [S1].")
    pipe.ask(BREACH_Q)
    request = llm.requests[0]
    assert request.system == SYSTEM_PROMPT
    prompt = request.messages[0].content
    assert prompt.startswith("SOURCES\n\n[S1] General Data Protection Regulation (GDPR) | EU regulation | status: in force")
    assert prompt.endswith(f"QUESTION\n{BREACH_Q}") and "not later than 72 hours" in prompt
    assert request.max_output_tokens == 2048 and request.temperature is None


def test_flagship_scenario_both_the_original_act_and_the_amendment_reach_the_model():
    pipe, llm, _ = pipeline("Annex III systems: 2 December 2027 [S1]; originally 2 August 2026 [S2].")
    answer = pipe.ask("When do the AI Act's high-risk obligations apply?")
    ids = [s.chunk_id for s in answer.sources]
    assert "omnibus:0001" in ids and "ai-act:0002" in ids
    prompt = llm.requests[0].messages[0].content
    assert "2 December 2027" in prompt and "2 August 2026" in prompt
    assert "AI Act (as originally published)" in prompt and "Digital Omnibus on AI amending the AI Act" in prompt
    assert answer.unverified_figures == ()  # both dates are in the sources


def test_draft_documents_are_labelled_in_the_prompt_and_in_the_answer_object():
    pipe, llm, _ = pipeline("Draft guidance says so [S1].")
    answer = pipe.ask("Is a resume ranking tool high-risk? hiring candidate", options=AskOptions(retrieval=RetrievalOptions(where={"doc_id": "draft-hr"})))
    assert [s.chunk_id for s in answer.sources] == ["draft-hr:0001"]
    assert "status: DRAFT - not adopted" in llm.requests[0].messages[0].content
    assert answer.sources[0].is_law is False and answer.sources[0].status_text == "DRAFT - not adopted"


def test_the_model_can_say_that_the_sources_do_not_answer():
    pipe, _, _ = pipeline("NOT_IN_SOURCES: the sources only cover breach notification.")
    answer = pipe.ask(BREACH_Q)
    assert answer.abstained is True and answer.text == "the sources only cover breach notification."
    assert answer.cited == () and answer.mode == RAG
    pipe, _, _ = pipeline("not_in_sources:")
    assert pipe.ask(BREACH_Q).text == "The indexed documents do not answer this question."


def test_fabricated_citations_and_unverified_figures_are_flagged():
    pipe, _, _ = pipeline("Notify within 48 hours [S1][S9]. Fines reach 20 million euros [S2].")
    answer = pipe.ask(BREACH_Q)
    assert answer.cited == ("S1", "S2") and answer.invalid_citations == ("S9",)
    assert answer.unverified_figures == ("48", "20")
    assert any("S9" in n for n in answer.notes)
    assert answer.citation_coverage == 1.0 and answer.uncited_sentences == ()


def test_uncited_sentences_are_reported():
    pipe, _, _ = pipeline("The controller must notify the authority promptly [S1]. The processor must always help the controller with that.")
    answer = pipe.ask(BREACH_Q)
    assert answer.citation_coverage == 0.5 and answer.uncited_sentences == ("The processor must always help the controller with that.",)


def test_a_failing_model_degrades_to_the_retrieved_passages():
    pipe, _, _ = pipeline(LLMError("quota", "daily limit reached", provider="fake", model="fake-1"))
    answer = pipe.ask(BREACH_Q)
    assert answer.mode == EXTRACTIVE and answer.sources and answer.sources[0].chunk_id == "gdpr:0001"
    assert answer.error_kind == "quota" and "daily limit reached" in answer.error and any("quota" in n for n in answer.notes)
    assert "passages are shown below" in answer.text and answer.cited == ()
    assert "generate" in answer.timings


def test_no_model_configured_still_returns_sources_and_says_why():
    pipe, _, _ = pipeline(reason="GEMINI_API_KEY is empty - add your key to the .env file")
    answer = pipe.ask(BREACH_Q)
    assert answer.mode == EXTRACTIVE and answer.sources and "GEMINI_API_KEY" in answer.error and "GEMINI_API_KEY" in answer.notes[0]
    assert answer.error_kind == "unavailable"


def test_retrieval_only_on_request_is_not_an_error():
    pipe, llm, _ = pipeline("never used")
    answer = pipe.ask(BREACH_Q, options=AskOptions(use_llm=False))
    assert answer.mode == EXTRACTIVE and answer.error is None and answer.error_kind is None and llm.requests == [] and answer.sources


def test_nothing_found_gives_a_clear_answer_without_calling_the_model():
    pipe, llm, _ = pipeline("never used")
    answer = pipe.ask("zzzz qqqq", options=AskOptions(retrieval=RetrievalOptions(mode="lexical")))
    assert answer.abstained and answer.text == NO_PASSAGES and answer.sources == () and llm.requests == []


def test_truncation_and_model_fallback_are_reported():
    llm = FakeLLM("Cut off mid sen", truncated=True, fell_back_from="fake-0")
    pipe, _, _ = pipeline(llm=llm)
    answer = pipe.ask(BREACH_Q)
    assert answer.truncated and answer.fell_back_from == "fake-0"
    assert any("cut off" in n for n in answer.notes) and any("fake-0 was unavailable" in n for n in answer.notes)


def test_the_context_budget_is_reported():
    pipe, llm, _ = pipeline("Answer [S1].")
    answer = pipe.ask("personal data", options=AskOptions(max_context_words=40, retrieval=RetrievalOptions(top_k=8)))
    assert len(answer.sources) < 8 and any("did not fit the context budget" in n for n in answer.notes)


def test_follow_up_questions_are_rewritten_and_the_rewrite_is_what_is_searched():
    seen = {}

    def reply(request):
        if "rewrite" in request.system.lower():
            return "Which deadline applies to a breach of health information in the US?"
        seen["prompt"] = request.messages[0].content
        return "Sixty days [S1]."

    pipe, llm, embedder = pipeline(reply)
    history = [Turn(BREACH_Q, "72 hours [S1].")]
    answer = pipe.ask("and in the US?", history)
    assert answer.standalone_question == "Which deadline applies to a breach of health information in the US?"
    assert answer.question == "and in the US?"
    assert embedder.queries[-1] == answer.standalone_question
    assert seen["prompt"].endswith("QUESTION\nWhich deadline applies to a breach of health information in the US?")
    assert any("follow-up rewritten" in n for n in answer.plan_notes) and len(llm.requests) == 2
    assert answer.sources[0].jurisdiction == "US"


class BrokenReranker:
    name = "broken"

    def score(self, query, passages):
        raise RerankError("could not load the re-ranking model 'x' (OSError: offline)")


def test_a_re_ranker_that_cannot_load_degrades_to_the_first_stage_order():
    _, retriever, _, _ = build()
    retriever._reranker = BrokenReranker()
    pipe = RAGPipeline(retriever, FakeLLM("Within 72 hours [S1]."))
    options = AskOptions(retrieval=RetrievalOptions(rerank=True))
    answer = pipe.ask(BREACH_Q, options=options)
    assert answer.mode == RAG and answer.sources[0].chunk_id == "gdpr:0001"
    assert any("re-ranking model could not be used" in n and "offline" in n for n in answer.notes)
    assert all(s.rerank_score is None for s in answer.sources)
    assert answer.options["retrieval"]["rerank"] is True  # the request is recorded as asked, the note says what happened


def test_the_options_of_a_run_are_recorded():
    pipe, _, _ = pipeline("Answer [S1].")
    options = AskOptions(retrieval=RetrievalOptions(mode="dense", top_k=3, where={"jurisdiction": "EU"}), transform=TransformOptions(glossary=False))
    recorded = pipe.ask(BREACH_Q, options=options).options
    assert recorded["retrieval"]["mode"] == "dense" and recorded["retrieval"]["top_k"] == 3
    assert recorded["retrieval"]["where"] == {"jurisdiction": "EU"} and recorded["transform"]["glossary"] is False


def test_the_answer_is_json_serialisable_for_the_web_interface():
    pipe, _, _ = pipeline("Within 72 hours [S1].")
    data = json.loads(json.dumps(pipe.ask(BREACH_Q).to_dict()))
    first = data["sources"][0]
    assert first["status_text"] == "in force" and first["pages"] == "p. 52" and first["cited"] is True
    assert data["mode"] == "rag" and isinstance(data["timings"], dict)


def test_options_cannot_be_passed_in_the_place_of_the_history():
    pipe, _, _ = pipeline("x")
    with pytest.raises(TypeError, match="options=AskOptions"):
        pipe.ask(BREACH_Q, AskOptions())


def test_question_validation():
    pipe, _, _ = pipeline("x")
    with pytest.raises(ValueError, match="empty"):
        pipe.ask("   \n ")
    with pytest.raises(ValueError, match="too long"):
        pipe.ask("a" * (MAX_QUESTION_CHARS + 1))
    assert pipe.ask("  " + BREACH_Q + "\n\n").question == BREACH_Q


# ----------------------------------------------------------------------------------------------- the baseline


def test_llm_only_mode_uses_no_retrieval_and_a_neutral_prompt():
    pipe, llm, embedder = pipeline("From memory: 2 August 2026.")
    before = len(embedder.queries)
    answer = pipe.ask_llm_only("When do the AI Act's high-risk obligations apply?")
    assert answer.mode == LLM_ONLY and answer.sources == () and answer.cited == ()
    assert answer.text == "From memory: 2 August 2026." and len(embedder.queries) == before
    request = llm.requests[0]
    assert request.system == LLM_ONLY_SYSTEM_PROMPT and request.messages[0].content == "When do the AI Act's high-risk obligations apply?"


def test_llm_only_mode_degrades_gracefully():
    pipe, _, _ = pipeline(reason="no key")
    unavailable = pipe.ask_llm_only("question?")
    assert "no key" in unavailable.error and unavailable.error_kind == "unavailable"
    pipe, _, _ = pipeline(LLMError("overloaded", "busy"))
    answer = pipe.ask_llm_only("question?")
    assert answer.error == "busy" and answer.error_kind == "overloaded" and answer.mode == LLM_ONLY
    assert any("overloaded" in n for n in answer.notes)
    with pytest.raises(ValueError):
        pipe.ask_llm_only("")


# ----------------------------------------------------------------------------------------------- concurrency


def test_many_questions_at_once_do_not_interfere():
    pipe, _, _ = pipeline(lambda request: "Within 72 hours [S1].")
    results, errors, lock = [], [], threading.Lock()

    def work(question):
        try:
            answer = pipe.ask(question)
            with lock:
                results.append((question, answer.sources[0].chunk_id))
        except Exception as exc:  # pragma: no cover - only reached when the test fails
            with lock:
                errors.append(exc)

    questions = [BREACH_Q, "right to erase delete data", "consumer sells personal information", "hiring candidate resume"] * 3
    threads = [threading.Thread(target=work, args=(q,)) for q in questions]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(results) == 12
    expected = {BREACH_Q: "gdpr:0001", "right to erase delete data": "gdpr:0002"}
    assert all(chunk == expected[q] for q, chunk in results if q in expected)


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_retrieval_only_without_any_model_is_not_reported_as_a_problem():
    pipe, _, _ = pipeline(reason="GEMINI_API_KEY is empty")
    answer = pipe.ask(BREACH_Q, options=AskOptions(use_llm=False))
    assert answer.mode == EXTRACTIVE and answer.error is None and answer.error_kind is None and answer.notes == ()


def test_a_question_of_exactly_the_length_limit_is_accepted():
    pipe, _, _ = pipeline("x [S1].")
    assert pipe.ask("a" * MAX_QUESTION_CHARS).question == "a" * MAX_QUESTION_CHARS


def test_whitespace_inside_a_question_is_normalised():
    pipe, _, _ = pipeline("x [S1].")
    assert pipe.ask("How   quickly\nmust we report\t a data   leak to the regulator?").question == BREACH_Q


def test_the_reason_for_a_missing_model_is_forgotten_when_a_model_is_given():
    pipe, _, _ = pipeline("x", reason="stale reason")
    assert pipe.llm_unavailable is None
    assert pipeline(reason="no key")[0].llm_unavailable == "no key"


def test_the_output_limit_reaches_the_model_in_both_modes():
    pipe, llm, _ = pipeline("x [S1].", "y")
    pipe.ask(BREACH_Q, options=AskOptions(max_output_tokens=300))
    pipe.ask_llm_only(BREACH_Q, 123)
    assert [r.max_output_tokens for r in llm.requests] == [300, 123]


def test_a_figure_that_the_user_introduced_is_not_reported_when_the_answer_repeats_it():
    pipe, _, _ = pipeline("No limit of 99 hours is mentioned in the sources [S1].")
    answer = pipe.ask("Is there a limit of 99 hours for reporting a data leak to the regulator?")
    assert answer.unverified_figures == ()
    pipe, _, _ = pipeline("A limit of 99 hours applies [S1].")
    assert pipe.ask(BREACH_Q).unverified_figures == ("99",)  # the same sentence without the user's figure is flagged


def test_a_long_answer_after_the_refusal_marker_is_shown_as_an_answer():
    """Seen with the real model: it wrote NOT_IN_SOURCES and then a full cited answer. That is an answer, not a refusal."""
    long = "NOT_IN_SOURCES: " + "The sources do not mention the AI Act here. " + "Notification must happen within 72 hours of becoming aware of the breach [S1]. " * 6
    pipe, _, _ = pipeline(long)
    answer = pipe.ask("How quickly must we report a personal data breach to the supervisory authority?")
    assert answer.abstained is False and answer.text.startswith("The sources do not mention") and "72 hours" in answer.text
    assert any("began by saying" in note for note in answer.notes) and answer.cited == ("S1",)
    short, _, _ = pipeline("NOT_IN_SOURCES: the sources cover breach notification [S1].")
    assert short.ask("How quickly must we report a personal data breach to the supervisory authority?").abstained is True
