"""The evaluation: the gold set, the scores, the report and the command (assignment Step 5)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import ragfixtures
from ragfixtures import FakeLLM

from reglens.cli import evaluate as evaluate_cli
from reglens.config import PROJECT_ROOT, load_settings
from reglens.embeddings import HashingEmbedder
from reglens.evaluation import judge as judge_module
from reglens.evaluation.experiments import GENERATION_CONFIGS, MAIN_CONFIGS, recheck_figures, rescore, run_generation, run_retrieval, store_agreement, summarise_generation
from reglens.evaluation.gold import GoldError, check_corpus, load_gold, parse_question
from reglens.evaluation.metrics import Passage, score_answer, score_retrieval, summarise_answers, summarise_retrieval
from reglens.evaluation.report import render
from reglens.indexing import build_index, info_path
from reglens.llm.base import LLMError
from reglens.retrieval.rerank import RerankError
from reglens.runtime import build_runtime
from reglens.vectorstore import open_store

GOLD_FILE = PROJECT_ROOT / "eval" / "gold.jsonl"


def question(**changes):
    data = {
        "id": "t1", "category": "number", "question": "How quickly must a breach be reported?", "gold_docs": ["gdpr"],
        "evidence": ["not later than 72 hours"], "facts": [["72 hours"]], "expected": "72 hours.",
    }
    data.update(changes)
    return parse_question(data)


# ------------------------------------------------------------------------------------------------ the gold set


def test_the_shipped_gold_set_is_well_formed():
    questions = load_gold(GOLD_FILE)
    assert len({q.id for q in questions}) == len(questions) >= 25
    assert {q.category for q in questions} >= {"number", "temporal", "multi-document", "out-of-scope"}
    assert all(q.expected for q in questions)
    assert sum(not q.answerable for q in questions) >= 3  # the documents cannot answer some questions on purpose
    flagship = next(q for q in questions if q.id == "q04")
    assert flagship.gold_docs == ("eu-ai-omnibus",) and flagship.missing_facts("It applies from 2 December 2027.") == []
    assert flagship.missing_facts("It applies from 2 August 2026.") == [0]  # the out-of-date answer is caught


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"category": "gossip"}, "category must be one of"),
        ({"evidence": ["("]}, "invalid regular expression"),
        ({"facts": []}, "needs gold_docs, evidence and facts"),
        ({"gold_docs": []}, "needs gold_docs, evidence and facts"),
        ({"colour": "red"}, "unknown field"),
        ({"question": " "}, "'question' is missing"),
        ({"expect_abstain": True}, "must not list documents"),
    ],
)
def test_a_malformed_gold_question_is_refused(changes, message):
    with pytest.raises(GoldError, match=message):
        question(**changes)


def test_the_gold_file_is_read_with_clear_errors(tmp_path):
    path = tmp_path / "gold.jsonl"
    good = {"id": "a", "category": "out-of-scope", "question": "Recipe?", "expect_abstain": True}
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(GoldError, match="line 3.*used twice"):
        load_gold(path)
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(GoldError, match="line 1: not valid JSON"):
        load_gold(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(GoldError, match="no questions"):
        load_gold(path)
    with pytest.raises(GoldError, match="cannot read"):
        load_gold(tmp_path / "missing.jsonl")


def test_relevance_ignores_line_breaks_and_other_documents():
    q = question()
    assert q.is_relevant("gdpr", "must notify it not\nlater   than 72 hours after")
    assert not q.is_relevant("ccpa", "not later than 72 hours")  # right words, wrong document
    assert not q.is_relevant("gdpr", "within three days")


def test_the_gold_set_is_checked_against_the_corpus():
    q = question()
    assert check_corpus([q], [("gdpr", "notify not later than 72 hours after")]) == []
    problems = check_corpus([q], [("gdpr", "nothing relevant here"), ("ccpa", "not later than 72 hours")])
    assert any("evidence" in p for p in problems) and any("fact group" in p for p in problems)
    assert any("not in the corpus" in p for p in check_corpus([q], [("ccpa", "x")]))
    both = question(gold_docs=["gdpr", "hhs"], evidence=["72 hours", "60 days"], needs_all_docs=True)
    assert any("'hhs' has no matching passage" in p for p in check_corpus([both], [("gdpr", "72 hours"), ("hhs", "30 days")]))
    refusal = parse_question({"id": "r", "category": "out-of-scope", "question": "Recipe?", "expect_abstain": True})
    assert check_corpus([refusal], []) == []


# ------------------------------------------------------------------------------------------------ retrieval scores


def passages(*items):
    return [Passage(f"c{i}", doc, text) for i, (doc, text) in enumerate(items, 1)]


def test_retrieval_scores_follow_the_rank_of_the_first_relevant_passage():
    q = question()
    found = passages(("gdpr", "other"), ("ccpa", "not later than 72 hours"), ("gdpr", "not later than 72 hours"))
    score = score_retrieval(q, found)
    assert score["first_relevant_rank"] == 3 and score["reciprocal_rank"] == pytest.approx(1 / 3)
    assert score["hit"] == {"1": False, "3": True, "5": True, "8": True, "12": True}
    assert score["coverage"]["1"] == 0 and score["coverage"]["3"] == 1
    assert score["doc_precision"]["3"] == pytest.approx(2 / 3)
    assert score["relevant_ranks"] == [3]


def test_nothing_relevant_scores_zero():
    score = score_retrieval(question(), passages(("gdpr", "x"), ("gdpr", "y")))
    assert score["first_relevant_rank"] is None and score["reciprocal_rank"] == 0 and not any(score["hit"].values())
    assert score_retrieval(question(), [])["doc_precision"]["8"] == 0


def test_a_question_that_needs_every_document_waits_for_the_last_one():
    q = question(gold_docs=["gdpr", "hhs"], evidence=["72 hours", "60 days"], needs_all_docs=True)
    one = passages(("gdpr", "72 hours"), ("gdpr", "72 hours again"))
    both = passages(("gdpr", "72 hours"), ("x", "-"), ("hhs", "60 days"))
    assert score_retrieval(q, one)["first_relevant_rank"] is None
    assert score_retrieval(q, both)["first_relevant_rank"] == 3
    assert score_retrieval(q, both)["coverage"]["3"] == 1 and score_retrieval(q, both)["coverage"]["1"] == 0.5


def test_retrieval_scores_are_averaged():
    q = question()
    good = score_retrieval(q, passages(("gdpr", "not later than 72 hours")))
    bad = score_retrieval(q, passages(("gdpr", "x")))
    summary = summarise_retrieval([good, bad])
    assert summary["n"] == 2 and summary["mrr"] == 0.5 and summary["hit"]["1"] == 0.5 and summary["hit"]["8"] == 0.5
    assert summarise_retrieval([]) == {"n": 0}


# ------------------------------------------------------------------------------------------------ answer scores


def answer(text, *, abstained=False, mode="rag", cited=("S1",), invalid=(), figures=(), coverage=1.0, error_kind=None, docs=("gdpr",)):
    sources = tuple(SimpleNamespace(label=f"S{i}", doc_id=doc, cited=f"S{i}" in cited, chunk_id=f"{doc}:{i}", text="t") for i, doc in enumerate(docs, 1))
    return SimpleNamespace(
        text=text, abstained=abstained, mode=mode, sources=sources, cited=tuple(cited), invalid_citations=tuple(invalid),
        unverified_figures=tuple(figures), citation_coverage=coverage, error_kind=error_kind, timings={"total": 2.0, "retrieve": 0.5, "generate": 1.5},
        input_tokens=100, output_tokens=20, truncated=False, notes=(),
    )


def test_a_correct_cited_answer_scores_well():
    score = score_answer(question(), answer("Within 72 hours [S1]."))
    assert score["correct"] and score["grounded"] and score["cited_from_gold"] == 1 and score["missing_facts"] == []
    assert not score["failed"] and score["latency_s"] == 2.0


def test_citation_markers_do_not_hide_or_fake_a_fact():
    assert score_answer(question(), answer("Within 72 [S1] hours."))["correct"]  # the marker is removed before matching
    assert not score_answer(question(), answer("Within three days [S1]."))["correct"]


def test_an_answer_is_not_grounded_with_a_problem_the_pipeline_found():
    assert score_answer(question(), answer("72 hours.", coverage=0.5))["grounded"] is False
    assert score_answer(question(), answer("72 hours [S9].", invalid=("S9",)))["grounded"] is False
    assert score_answer(question(), answer("72 hours [S1].", figures=("99",)))["grounded"] is False


def test_citing_another_document_lowers_the_gold_share():
    score = score_answer(question(), answer("72 hours [S1][S2].", cited=("S1", "S2"), docs=("gdpr", "ccpa")))
    assert score["cited_from_gold"] == 0.5


def test_abstaining_is_right_only_when_the_documents_cannot_answer():
    refusal = parse_question({"id": "r", "category": "out-of-scope", "question": "Recipe?", "expect_abstain": True})
    assert score_answer(refusal, answer("The documents do not cover this.", abstained=True, cited=()))["correct"]
    assert not score_answer(refusal, answer("Boil pasta [S1]."))["correct"]
    assert not score_answer(question(), answer("The documents do not cover this.", abstained=True, cited=()))["correct"]  # a wrong abstention


def test_a_failed_model_call_is_not_counted_as_a_wrong_answer():
    failed = score_answer(question(), answer("Passages are shown below.", mode="extractive", error_kind="quota", cited=()))
    assert failed["failed"] and failed["correct"] is None
    rows = [{**score_answer(question(), answer("72 hours [S1].")), "category": "number", "expect_abstain": False}, {**failed, "category": "number", "expect_abstain": False}]
    summary = summarise_answers(rows)
    assert summary["asked"] == 2 and summary["failed"] == 1 and summary["correct_rate"] == 1 and summary["answerable"] == 1


def test_answers_are_summarised_per_configuration():
    good = {**score_answer(question(), answer("72 hours [S1].")), "category": "number", "expect_abstain": False}
    bad = {**score_answer(question(), answer("Soon [S1].", invalid=("S7",))), "category": "number", "expect_abstain": False}
    refusal_q = parse_question({"id": "r", "category": "out-of-scope", "question": "Recipe?", "expect_abstain": True})
    refused = {**score_answer(refusal_q, answer("No.", abstained=True, cited=())), "category": "out-of-scope", "expect_abstain": True}
    summary = summarise_answers([good, bad, refused])
    assert summary["correct_count"] == 1 and summary["answerable"] == 2 and summary["abstain_count"] == 1 and summary["unanswerable"] == 1
    assert summary["invalid_citation_answers"] == 1 and summary["latency_s"] == 2.0 and summary["input_tokens"] == 100


# ------------------------------------------------------------------------------------------------ the judge


def test_the_judge_reply_is_parsed_strictly():
    ok = judge_module.parse_judgement('Sure: {"relevance": 5, "faithfulness": 4, "correctness": 3, "reason": "good\\nanswer"}')
    assert (ok.relevance, ok.faithfulness, ok.correctness, ok.reason) == (5, 4, 3, "good answer")
    for bad in ("", "no json", '{"relevance": 9, "faithfulness": 4, "correctness": 3}', '{"relevance": "x"}', '{"relevance": 5}'):
        assert judge_module.parse_judgement(bad) is None


def test_the_judge_prompt_is_bounded_and_failures_give_none():
    prompt = judge_module.build_prompt("Q?", "A.", ["word " * 1000, "word " * 1000], ["fact one"])
    assert "[S1]" in prompt and "[S2]" not in prompt and "- fact one" in prompt and "(no sources)" not in prompt
    assert "(none given)" in judge_module.build_prompt("Q?", "A.", [], [])
    assert judge_module.judge(FakeLLM(LLMError("quota", "x", provider="fake")), "Q?", "A.", [], []) is None
    ok = FakeLLM('{"relevance": 5, "faithfulness": 5, "correctness": 5, "reason": "r"}')
    assert judge_module.judge(ok, "Q?", "A.", ["s"], []).correctness == 5


# ------------------------------------------------------------------------------------------------ runs on the invented corpus

GOLD_RECORDS = [
    {"id": "g1", "category": "number", "question": "How quickly must we report a personal data breach to the supervisory authority?", "gold_docs": ["gdpr"],
     "evidence": ["not later than 72 hours"], "facts": [["72 hours"]], "expected": "72 hours."},
    {"id": "g2", "category": "temporal", "question": "From when do the high-risk obligations in Annex III apply?", "gold_docs": ["omnibus"],
     "evidence": ["2 December 2027"], "facts": [["2 December 2027"]], "expected": "2 December 2027."},
    {"id": "g3", "category": "multi-document", "question": "Compare breach notification deadlines of the GDPR and the HHS rule", "gold_docs": ["gdpr", "hhs-breach"],
     "evidence": ["72 hours", "60 days"], "facts": [["72 hours"], ["60 days"]], "needs_all_docs": True, "expected": "72 hours and 60 days."},
    {"id": "g4", "category": "out-of-scope", "question": "What is a good recipe for a vegetarian lasagne?", "expect_abstain": True, "expected": "Abstain."},
]


@pytest.fixture
def gold_file(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in GOLD_RECORDS) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def runtime_parts(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(c.to_dict()) for c in ragfixtures.chunks()) + "\n", encoding="utf-8")
    environ = {"VECTORSTORE": "numpy", "VECTORSTORE_DIR": str(tmp_path / "index"), "GEMINI_API_KEY": "", "EMBEDDING_MODEL": "hashing", "CHUNKS_PATH": str(chunks_path)}
    settings = load_settings(root=tmp_path, environ=environ)
    store = open_store(settings.vectorstore_dir, "numpy")
    build_index(ragfixtures.chunks(), ragfixtures.by_source(), HashingEmbedder(), store, info_file=info_path(settings.vectorstore_dir, "numpy"), backend="numpy")
    return settings


class LengthReranker:
    """Prefers short passages: enough to see that the re-ranking stage ran."""

    name = "length-reranker"

    def score(self, query, passages):
        return [-float(len(p)) for p in passages]


class BrokenReranker:
    name = "broken"

    def score(self, query, passages):
        raise RerankError("the re-ranking model could not be loaded")


def runtime_for(settings, llm=None, reranker=None):
    return build_runtime(settings, embedder=HashingEmbedder(), llm=llm or FakeLLM("x"), reranker=reranker or LengthReranker())


def scripted_llm():
    """Answers like the pipeline expects: grounded for the questions of the corpus, an abstention for the recipe."""

    def reply(request):
        prompt = request.messages[0].content
        if "lasagne" in prompt:
            return "NOT_IN_SOURCES: the sources cover data protection."
        if "high-risk obligations" in prompt:
            return "The obligations apply from 2 December 2027 [S1]."
        return "Notify within 72 hours [S1]. The HHS rule allows 60 days [S2]."

    return FakeLLM(reply)


def test_retrieval_runs_score_every_answerable_question(runtime_parts):
    runtime = runtime_for(runtime_parts)
    questions = load_gold(_write(runtime_parts, GOLD_RECORDS))
    by_name = {c.name: c for c in MAIN_CONFIGS}
    run = run_retrieval(runtime.retriever, questions, by_name["hybrid"])
    assert [q["id"] for q in run["questions"]] == ["g1", "g2", "g3"]  # the question the documents cannot answer is not scored here
    assert run["summary"]["n"] == 3 and run["summary"]["hit"]["8"] == 1.0 and run["summary"]["by_category"]["number"]["n"] == 1
    assert run["questions"][0]["first_relevant_rank"] == 1
    lexical = run_retrieval(runtime.retriever, questions, by_name["lexical"])
    assert lexical["summary"]["n"] == 3


def test_a_configuration_that_needs_a_model_or_a_reranker_is_skipped_with_a_reason(runtime_parts):
    runtime = runtime_for(runtime_parts)
    questions = load_gold(_write(runtime_parts, GOLD_RECORDS))
    by_name = {c.name: c for c in MAIN_CONFIGS}
    assert "needs a language model" in run_retrieval(runtime.retriever, questions, by_name["hybrid_hyde"], llm=None)["skipped"]
    ranked = run_retrieval(runtime.retriever, questions, by_name["hybrid_rerank"])
    assert ranked["summary"]["n"] == 3  # the stand-in re-ranker works, so this configuration is scored
    broken = runtime_for(runtime_parts, reranker=BrokenReranker())
    assert "re-ranking model could not be loaded" in run_retrieval(broken.retriever, questions, by_name["hybrid_rerank"])["skipped"]


def test_two_stores_over_the_same_vectors_agree(runtime_parts):
    runtime = runtime_for(runtime_parts)
    questions = load_gold(_write(runtime_parts, GOLD_RECORDS))
    agreement = store_agreement(runtime.retriever, runtime.retriever, questions)
    assert agreement["questions"] == 3 and agreement["mean_overlap"] == 1 and agreement["same_first_passage"] == 1


def _write(settings, records):
    path = settings.chunks_path.parent / "gold-copy.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_generation_is_saved_resumed_and_scored(runtime_parts, tmp_path):
    runtime = runtime_for(runtime_parts, scripted_llm())
    questions = load_gold(_write(runtime_parts, GOLD_RECORDS))
    path = tmp_path / "out" / "generation.jsonl"
    configs = [GENERATION_CONFIGS["llm_only"], GENERATION_CONFIGS["rag"]]
    rows, stopped = run_generation(runtime.pipeline, questions, configs, path)
    assert stopped is None
    assert len(rows) == 3 + 4  # the baseline is not asked what the documents cannot answer
    summaries = summarise_generation(rows)
    rag = summaries["rag"]
    assert rag["correct_count"] == 3 and rag["answerable"] == 3 and rag["abstain_count"] == 1 and rag["grounded_rate"] == 1
    assert "llm_only" in summaries and summaries["llm_only"]["unanswerable"] == 0
    calls = len(runtime.pipeline.llm.requests)
    again, _ = run_generation(runtime.pipeline, questions, configs, path)  # everything is already saved
    assert len(runtime.pipeline.llm.requests) == calls and len(again) == len(rows)
    assert len(path.read_text().splitlines()) == len(rows)


def test_a_quota_error_stops_the_run_and_keeps_what_was_answered(runtime_parts, tmp_path):
    questions = load_gold(_write(runtime_parts, GOLD_RECORDS))
    llm = FakeLLM("Notify within 72 hours [S1].", LLMError("quota", "free quota used up", provider="fake"))
    runtime = runtime_for(runtime_parts, llm)
    path = tmp_path / "generation.jsonl"
    rows, stopped = run_generation(runtime.pipeline, questions, [GENERATION_CONFIGS["rag"]], path)
    assert stopped and "quota" in stopped and len(rows) == 2
    assert len(path.read_text().splitlines()) == 1  # only the real answer was saved, the failure will be asked again


# ------------------------------------------------------------------------------------------------ the command


def run_command(settings, gold, tmp_path, *argv, llm=None):
    lines: list[str] = []
    code = evaluate_cli.main(
        ["--gold", str(gold), "--out", str(tmp_path / "out"), "--min-interval", "0", *argv],
        build=lambda s, **kw: runtime_for(s, llm),
        settings_loader=lambda **kw: settings,
        write=lines.append,
    )
    return code, "\n".join(lines)


def test_the_command_only_validates_when_asked(runtime_parts, gold_file, tmp_path):
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--validate")
    assert code == 0 and "4 questions (3 answerable, 1 unanswerable)" in text and "every expected passage and fact exists" in text
    assert not (tmp_path / "out").exists()


def test_the_command_refuses_a_gold_set_the_documents_cannot_answer(runtime_parts, tmp_path):
    bad = dict(GOLD_RECORDS[0], id="bad", evidence=["a sentence that is nowhere"])
    code, text = run_command(runtime_parts, _write(runtime_parts, [bad]), tmp_path)
    assert code == 1 and "do not match the corpus" in text and "a sentence that is nowhere" in text


def test_the_command_writes_a_report_from_retrieval_alone(runtime_parts, gold_file, tmp_path):
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--embedding-baseline")
    assert code == 0
    out = tmp_path / "out"
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {r["config"] for r in results["retrieval"]} >= {"lexical", "dense", "hybrid", "dense@hashing", "hybrid@hashing"}
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# RegLens evaluation" in report and "Hybrid: meaning + keywords" in report and "Keywords only (BM25)" in report
    assert "Run on" in text and "Saved:" in text and "hybrid_rerank" not in report


def test_the_command_evaluates_a_fixed_chunk_file(runtime_parts, gold_file, tmp_path):
    fixed = tmp_path / "fixed.jsonl"
    fixed.write_text("\n".join(json.dumps(c.to_dict()) for c in ragfixtures.chunks()[:6]) + "\n", encoding="utf-8")
    code, _ = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--fixed-chunks", str(fixed))
    assert code == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert any(r["config"] == "hybrid@fixed" for r in results["retrieval"])
    assert results["extras"][0]["chunking"]["fixed"]["chunks"] == 6


def test_the_command_generates_judges_and_reports(runtime_parts, gold_file, tmp_path):
    llm = scripted_llm()
    verdicts = FakeLLM(lambda request: '{"relevance": 5, "faithfulness": 4, "correctness": 5, "reason": "ok"}')
    llm.replies.append(llm.replies[0])  # unchanged: answers use the scripted reply
    # one model plays both parts: it replies with a verdict when it is the judge's prompt
    answering = llm.replies[0]
    llm.replies = [lambda request: verdicts.replies[0](request) if "strict evaluator" in request.system else answering(request)]
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--generate", "--judge", llm=llm)
    assert code == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    summaries = results["generation"]["summaries"]
    assert summaries["rag"]["correct_count"] == 3 and summaries["rag"]["abstain_count"] == 1
    assert {"rag", "llm_only"} <= set(results["generation"]["judged"])
    assert "Language-model judge" in text and "Where RegLens was wrong" in text and "No gold question was answered incorrectly" in text
    assert (tmp_path / "out" / "judge.jsonl").is_file()


def test_the_command_continues_after_a_quota_stop(runtime_parts, gold_file, tmp_path):
    llm = FakeLLM("Notify within 72 hours [S1].", LLMError("quota", "used up", provider="fake"))
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--generate", llm=llm)
    assert code == 3 and "Stopped early" in text and "run the same command again" in text
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--generate", llm=scripted_llm())
    assert code == 0 and "Stopped early" not in text


def test_the_command_explains_unusable_options(runtime_parts, gold_file, tmp_path):
    code, text = run_command(runtime_parts, gold_file, tmp_path, "--judge")
    assert code == 2 and "--judge needs --generate" in text
    code, text = run_command(runtime_parts, gold_file / "missing", tmp_path)
    assert code == 2 and "Problem:" in text


def test_the_report_renders_whatever_parts_a_run_has():
    meta = {"date": "2026-11-16", "index": "10 passages", "embedding": "m", "llm": None, "n_questions": 4, "n_answerable": 3, "n_unanswerable": 1, "n_documents": 6}
    text = render({"meta": meta})
    assert "# RegLens evaluation" in text and "Limits of this evaluation" in text and "4 questions" in text
    run = {"config": "hybrid", "label": "Hybrid", "group": "search", "summary": {"n": 1, "mrr": 0.5, **{k: {str(i): 0.5 for i in (1, 3, 5, 8, 12)} for k in ("hit", "coverage", "doc_precision")}, "seconds_per_question": 0.1, "by_category": {}}}
    assert "50%" in render({"meta": meta, "retrieval": [run, {"config": "x", "label": "Skipped one", "skipped": "no"}]})


def test_saved_answers_can_be_scored_again_without_asking_the_model(runtime_parts, gold_file, tmp_path):
    llm = scripted_llm()
    code, _ = run_command(runtime_parts, gold_file, tmp_path, "--no-rerank", "--generate", llm=llm)
    assert code == 0
    out = tmp_path / "out"
    # the gold pattern for g1 turns out to be too narrow: it wanted "72 hours", the answer said "seventy-two hours"
    records = [dict(r) for r in GOLD_RECORDS]
    records[0]["facts"] = [["seventy-two hours", "72 hours"]]
    widened = _write(runtime_parts, records)
    calls = len(llm.requests)
    lines: list[str] = []
    assert evaluate_cli.main(["--gold", str(widened), "--out", str(out), "--rescore"], write=lines.append) == 0
    assert len(llm.requests) == calls  # nothing was asked again
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["meta"]["rescored"]["changed"] == []  # same verdicts: the answers already said 72 hours
    assert "Re-scored" in "\n".join(lines)


def test_rescoring_reports_the_verdicts_that_changed():
    narrow = question(facts=[["seventy-two hours"]])
    row = {"config": "rag", "id": "t1", "answer": "Within 72 hours [S1].", "score": {**score_answer(narrow, answer("Within 72 hours [S1].")), }}
    assert row["score"]["correct"] is False
    wide = question(facts=[["seventy-two hours", "72 hours"]])
    changed = rescore([row], [wide])
    assert changed == [{"config": "rag", "id": "t1", "before": False, "after": True}] and row["score"]["correct"] is True
    assert rescore([row], [wide]) == []  # a second pass changes nothing
    abstained = {"config": "rag", "id": "t1", "answer": "72 hours", "score": {"failed": False, "abstained": True, "correct": False, "missing_facts": []}}
    assert rescore([abstained], [wide]) == [] and abstained["score"]["missing_facts"] == [0]  # a refusal is never a correct answer


def test_rescoring_without_saved_answers_explains_what_to_run(tmp_path, gold_file):
    lines: list[str] = []
    assert evaluate_cli.main(["--gold", str(gold_file), "--out", str(tmp_path / "none"), "--rescore"], write=lines.append) == 1
    assert "--generate" in "\n".join(lines)


def test_a_figure_that_the_source_heading_shows_is_not_an_invention():
    """The model is shown "title | version | section" above each passage; a number it copies from there is supported."""
    gold = question()
    row = {"config": "rag", "id": "t1", "answer": "It is Regulation (EU) 2026/1744 [S1]. Fines reach 99 million [S1].", "sources": ["c1"], "source_texts": ["Text with no numbers."],
           "score": {"failed": False, "abstained": False, "unverified_figures": ["2026", "1744", "99"], "citation_coverage": 1.0, "invalid_citations": [], "grounded": False}}
    changed = recheck_figures([row], [gold], {"c1": "Regulation (EU) 2026/1744 amending the AI Act | version: adopted 2026-07-27"})
    assert row["score"]["unverified_figures"] == ["99"]  # the invented figure is still caught
    assert changed == [{"config": "rag", "id": "t1", "before": ["2026", "1744", "99"], "after": ["99"]}] and row["score"]["grounded"] is False
    baseline = {"config": "llm_only", "id": "t1", "answer": "99", "sources": [], "source_texts": [], "score": {"failed": False, "unverified_figures": [], "grounded": None}}
    assert recheck_figures([baseline], [gold], {}) == []  # the model alone has no sources to check against


def test_the_pipeline_reads_the_heading_when_it_checks_figures():
    from reglens.generation.grounding import unverified_figures
    from reglens.generation.prompts import source_header

    heading = source_header({"title": "Regulation (EU) 2026/1744", "version_date": "adopted 2026-07-27", "status": "in_force"}, "x")
    assert "2026/1744" in heading and "2026-07-27" in heading
    assert unverified_figures("Regulation 2026/1744 of 27 July 2026.", [f"{heading}\nSome passage."]) == []
    assert unverified_figures("Regulation 2026/1744.", ["Some passage."]) == ["2026", "1744"]
