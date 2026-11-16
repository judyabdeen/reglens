from __future__ import annotations

import math

import pytest
from ragfixtures import ConceptEmbedder, build, chunks

from reglens.llm.base import LLMError
from reglens.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    QueryPlan,
    QueryTransformer,
    RerankError,
    RetrievalError,
    RetrievalOptions,
    TransformOptions,
    Turn,
    assemble_context,
    reciprocal_rank_fusion,
)
from reglens.retrieval.glossary import GLOSSARY, expand
from reglens.retrieval.query_transform import DenseQuery, format_history, parse_queries
from reglens.retrieval.text import stem, tokenize

# ----------------------------------------------------------------------------------------------- tokens


def test_tokenize_lowercases_removes_stop_words_and_plurals():
    assert tokenize("The Controllers shall notify the Supervisory Authorities of 72 hours") == ["controller", "notify", "supervisory", "authority", "of"][:0] + ["controller", "notify", "supervisory", "authority", "72", "hour"]


def test_numbers_and_references_stay_searchable():
    assert tokenize("Regulation (EU) 2026/1744, Annex III, Article 6(1)(a)") == ["regulation", "eu", "2026", "1744", "annex", "iii", "article", "6", "1"][:0] + ["regulation", "eu", "2026", "1744", "annex", "iii", "article", "6", "1"]


@pytest.mark.parametrize("word, expected", [("authorities", "authority"), ("processes", "process"), ("boxes", "box"), ("matches", "match"), ("controllers", "controller"),
                                            ("data", "data"), ("class", "class"), ("status", "status"), ("analysis", "analysis"), ("is", "is"), ("gdpr", "gdpr"), ("ass", "ass")])
def test_stem(word, expected):
    assert stem(word) == expected


# ----------------------------------------------------------------------------------------------- BM25


def small_index():
    docs = {
        "a": "personal data breach notification to the supervisory authority within 72 hours",
        "b": "right to erasure of personal data",
        "c": "breach breach breach breach breach",
        "d": "unrelated text about oranges and apples " * 6,
    }
    return BM25Index(list(docs), list(docs.values()), [{"group": "x" if k in "ab" else "y"} for k in docs])


def test_bm25_ranks_the_document_with_the_rare_matching_terms_first():
    hits = small_index().search("breach notification authority", 4)
    assert [i for i, _ in hits][0] == "a"
    assert all(score > 0 for _, score in hits)
    assert "d" not in [i for i, _ in hits]  # shares no term with the query


def test_bm25_score_matches_the_formula():
    index = BM25Index(["d1", "d2"], ["alpha beta", "gamma"], k1=1.5, b=0.75)
    # N=2, df(alpha)=1 -> idf = ln(1 + (2-1+0.5)/(1+0.5)) = ln 2 ; avgdl = 1.5 ; |d1| = 2 -> norm = 1.5*(0.25+0.75*2/1.5)
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / 1.5)
    expected = math.log(2) * 1 * 2.5 / (1 + norm)
    assert index.search("alpha", 1)[0][1] == pytest.approx(expected)


def test_repeating_a_word_has_diminishing_returns():
    index = BM25Index(["once", "many", "pad"], ["breach", "breach " * 30, "other words here"])
    scores = dict(index.search("breach", 3))
    assert scores["many"] < scores["once"] * 3  # saturation (and the length penalty)


def test_bm25_respects_filters_and_k():
    index = small_index()
    assert [i for i, _ in index.search("personal data", 5, {"group": "y"})] == []
    assert [i for i, _ in index.search("personal data", 5, {"group": "x"})] == ["b", "a"] or {i for i, _ in index.search("personal data", 5, {"group": "x"})} == {"a", "b"}
    assert len(index.search("personal data breach", 1)) == 1
    with pytest.raises(ValueError):
        index.search("x", 0)


def test_bm25_is_deterministic_for_equal_scores():
    index = BM25Index(["a", "b", "c"], ["same words", "same words", "same words"])
    assert [i for i, _ in index.search("same", 3)] == ["a", "b", "c"]


def test_bm25_edge_cases():
    assert BM25Index([], []).search("anything", 3) == []
    assert BM25Index([], []).search("anything", 3, {"k": "v"}) == []
    assert small_index().search("the of and", 3) == []  # only stop words
    assert small_index().search("", 3) == []
    with pytest.raises(ValueError):
        BM25Index(["a", "a"], ["x", "y"])
    with pytest.raises(ValueError):
        BM25Index(["a"], ["x", "y"])
    assert len(small_index()) == 4 and small_index().document_frequency("breach") == 2


# ----------------------------------------------------------------------------------------------- fusion


def test_rrf_scores_and_order():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63) and fused["d"] == pytest.approx(1 / 63)
    order = [i for i, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])]
    assert order[:2] == ["a", "b"] and order[2:] == ["c", "d"]  # ties keep first-seen order


def test_a_document_in_both_lists_beats_one_at_the_top_of_a_single_list():
    order = [i for i, _ in reciprocal_rank_fusion([["x", "both", "y"], ["z", "w", "both"]])]
    assert order[0] == "both"  # 1/62 + 1/63 beats the 1/61 of a single first place
    assert order[1:3] == ["x", "z"]


def test_rrf_weights_and_duplicates_and_validation():
    assert [i for i, _ in reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 3.0])] == ["b", "a"]
    assert dict(reciprocal_rank_fusion([["a", "a", "a"]]))["a"] == pytest.approx(1 / 61)  # repeats count once
    assert reciprocal_rank_fusion([]) == [] and reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


# ----------------------------------------------------------------------------------------------- glossary


@pytest.mark.parametrize("name", [r.name for r in GLOSSARY])
def test_every_glossary_rule_matches_its_examples_and_not_its_counter_examples(name):
    rule = next(r for r in GLOSSARY if r.name == name)
    assert rule.examples, "a rule without examples cannot be tested"
    for text in rule.examples:
        assert rule.pattern.search(text), f"{name} should match {text!r}"
    for text in rule.counter_examples:
        assert not rule.pattern.search(text), f"{name} must not match {text!r}"


def test_glossary_rule_names_are_unique_and_terms_are_plain_words():
    names = [r.name for r in GLOSSARY]
    assert len(names) == len(set(names))
    for r in GLOSSARY:
        assert r.terms.strip() == r.terms and "  " not in r.terms and r.terms


def test_expand_appends_official_terms_once():
    query, applied = expand("We had a data leak - who is the regulator?")
    assert query.startswith("We had a data leak - who is the regulator?")
    assert "supervisory" in query and "breach" in query
    assert [n for n, _ in applied] == ["breach", "regulator"]
    words = query.split()
    assert words.count("supervisory") == 1  # a term that two rules add is added once


def test_expand_does_nothing_for_other_questions():
    assert expand("What is the capital of France?") == ("What is the capital of France?", [])


def test_glossary_never_contains_deadlines_or_amounts():
    import re

    for r in GLOSSARY:
        assert not re.search(r"\d", r.terms.replace("2024/1689", "").replace("2026/1744", "")), f"{r.name} leaks a number"


# ----------------------------------------------------------------------------------------------- query transformation


class ScriptedLLM:
    provider, model = "fake", "fake-1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def generate(self, request):
        from reglens.llm.base import LLMResponse

        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, provider=self.provider, model=self.model)


def test_plain_plan():
    plan = QueryPlan.plain("hello world")
    assert plan.dense == (DenseQuery("hello world"),) and plan.lexical == "hello world" and plan.llm_calls == 0


def test_glossary_only_plan_needs_no_model():
    plan = QueryTransformer(None).plan("  We had a data   leak\n")
    assert plan.question == "We had a data leak" and plan.standalone == plan.question
    assert plan.dense == (DenseQuery("We had a data leak"),)  # the dense query is left alone
    assert "supervisory" in plan.lexical
    assert plan.llm_calls == 0 and any("glossary (breach)" in n for n in plan.notes)


def test_glossary_can_be_switched_off():
    plan = QueryTransformer(None).plan("data leak", options=TransformOptions(glossary=False))
    assert plan.lexical == "data leak" and plan.notes == ()


def test_follow_up_questions_are_condensed_with_the_history():
    llm = ScriptedLLM('"What about the United States?"\nextra line')
    history = [Turn("How fast must we report a breach in the EU?", "Within 72 hours [S1].")]
    plan = QueryTransformer(llm).plan("and in the US?", history)
    assert plan.standalone == "What about the United States?" and plan.question == "and in the US?"
    assert plan.dense[0].text == "What about the United States?" and plan.llm_calls == 1
    assert "How fast must we report" in llm.requests[0].messages[0].content and "Last question: and in the US?" in llm.requests[0].messages[0].content


def test_condensing_is_skipped_without_history_or_when_the_model_fails():
    llm = ScriptedLLM()
    assert QueryTransformer(llm).plan("a question").llm_calls == 0 and llm.requests == []
    failing = ScriptedLLM(LLMError("rate_limited", "slow down"))
    plan = QueryTransformer(failing).plan("and in the US?", [Turn("q", "a")])
    assert plan.standalone == "and in the US?" and any("rate_limited" in n for n in plan.notes)


def test_multi_query_adds_paraphrases_without_repeats():
    llm = ScriptedLLM("1. notification of a personal data breach to the authority\n- Notification of a personal data breach to the authority\n* time limit for reporting a breach\n\nshort")
    plan = QueryTransformer(llm).plan("report a breach", options=TransformOptions(multi_query=3))
    assert [d.text for d in plan.dense] == ["report a breach", "notification of a personal data breach to the authority", "time limit for reporting a breach"]
    assert all(not d.as_passage for d in plan.dense) and plan.llm_calls == 1


def test_hyde_adds_a_passage_query():
    llm = ScriptedLLM("The controller shall notify the supervisory authority   without undue delay.")
    plan = QueryTransformer(llm).plan("report a breach", options=TransformOptions(hyde=True))
    assert plan.dense[1] == DenseQuery("The controller shall notify the supervisory authority without undue delay.", as_passage=True)


def test_failures_of_optional_steps_do_not_break_the_plan():
    llm = ScriptedLLM(LLMError("overloaded", "busy"), LLMError("quota", "no more"))
    plan = QueryTransformer(llm).plan("report a breach", options=TransformOptions(multi_query=2, hyde=True))
    assert len(plan.dense) == 1 and plan.llm_calls == 2
    assert any("multi-query skipped" in n for n in plan.notes) and any("hypothetical passage skipped" in n for n in plan.notes)


def test_missing_model_is_explained_in_the_notes():
    plan = QueryTransformer(None).plan("x y", [Turn("a", "b")], TransformOptions(multi_query=2))
    assert any("no language model" in n for n in plan.notes) and plan.llm_calls == 0


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        QueryTransformer(None).plan("   ")


def test_parse_queries_and_history_formatting():
    assert parse_queries("- a long enough query\n- A LONG ENOUGH QUERY\n2) another good one", 5) == ["a long enough query", "another good one"]
    assert parse_queries("one two three four", 1) == ["one two three four"]
    assert parse_queries("x", 3) == []
    turns = [Turn(f"q{i}", "a" * 1000) for i in range(5)]
    text = format_history(turns)
    assert "q4" in text and "q2" in text and "q1" not in text  # the last three turns only
    assert max(len(line) for line in text.splitlines()) < 700  # long answers are cut


# ----------------------------------------------------------------------------------------------- retriever


def ids(results):
    return [r.id for r in results]


def test_dense_search_finds_a_synonym_that_shares_no_word_with_the_query():
    _, retriever, _, _ = build()
    plan = QueryPlan.plain("leak regulator")  # neither word occurs in Article 33, whose text speaks of a breach and an authority
    dense = ids(retriever.retrieve(plan, RetrievalOptions(mode="dense", top_k=3)))
    lexical = ids(retriever.retrieve(plan, RetrievalOptions(mode="lexical", top_k=3)))
    assert dense[0] == "gdpr:0001"
    assert lexical == []


def test_keyword_search_finds_an_exact_term_that_the_embedding_cannot_distinguish():
    _, retriever, _, _ = build()
    plan = QueryPlan.plain("2 December 2027")
    assert ids(retriever.retrieve(plan, RetrievalOptions(mode="lexical", top_k=1))) == ["omnibus:0001"]


def test_hybrid_contains_what_either_method_finds():
    _, retriever, _, _ = build()
    plan = QueryPlan.plain("data leak regulator 72 hours")
    hybrid = ids(retriever.retrieve(plan, RetrievalOptions(mode="hybrid", top_k=5)))
    dense = ids(retriever.retrieve(plan, RetrievalOptions(mode="dense", top_k=5)))
    lexical = ids(retriever.retrieve(plan, RetrievalOptions(mode="lexical", top_k=5)))
    assert hybrid[0] == "gdpr:0001"
    assert set(dense[:2]) | set(lexical[:2]) <= set(hybrid)


def test_evidence_is_reported_for_each_result():
    _, retriever, _, _ = build()
    results = retriever.retrieve("breach notification authority 72 hours", RetrievalOptions(top_k=3))
    top = results[0]
    assert top.rank == 1 and top.id == "gdpr:0001"
    assert top.dense_score is not None and top.lexical_score is not None and top.fused_score is not None
    assert top.dense_rank >= 1 and top.lexical_rank == 1 and top.rerank_score is None
    assert top.score == top.fused_score
    assert [r.rank for r in results] == [1, 2, 3] and results[0].score >= results[1].score


def test_single_method_modes_report_their_own_score():
    _, retriever, _, _ = build()
    dense = retriever.retrieve("data leak", RetrievalOptions(mode="dense", top_k=1))[0]
    assert dense.score == dense.dense_score and dense.lexical_score is None and dense.fused_score is None
    lexical = retriever.retrieve("breach", RetrievalOptions(mode="lexical", top_k=1))[0]
    assert lexical.score == lexical.lexical_score and lexical.dense_score is None


def test_filters_apply_to_both_searches():
    _, retriever, _, _ = build()
    for mode in ("hybrid", "dense", "lexical"):
        results = retriever.retrieve("breach notification", RetrievalOptions(mode=mode, top_k=10, where={"jurisdiction": "US"}))
        assert results and {r.metadata["jurisdiction"] for r in results} == {"US"}
    results = retriever.retrieve("high-risk", RetrievalOptions(top_k=10, where={"status": ["draft"]}))
    assert ids(results) == ["draft-hr:0001"]
    assert retriever.retrieve("breach", RetrievalOptions(where={"jurisdiction": "UK"})) == []


def test_notes_can_be_excluded():
    store, retriever, embedder, records = build()
    from reglens.retrieval import Retriever
    from reglens.vectorstore.base import Record

    note = Record("gdpr:n001", "Footnotes: breach breach breach", {"doc_id": "gdpr", "kind": "notes", "jurisdiction": "EU"})
    store.add([note], embedder.embed_documents(["breach breach breach"]))
    retriever = Retriever(store, embedder, store.records())
    assert "gdpr:n001" in ids(retriever.retrieve("breach", RetrievalOptions(top_k=3)))
    assert "gdpr:n001" not in ids(retriever.retrieve("breach", RetrievalOptions(top_k=3, include_notes=False)))


def test_paraphrases_are_searched_and_fused():
    _, retriever, embedder, _ = build()
    plan = QueryPlan("q", "q", (DenseQuery("erase delete forget"), DenseQuery("hiring resume candidate")), "zzzz")
    result = ids(retriever.retrieve(plan, RetrievalOptions(mode="dense", top_k=4)))
    assert {"gdpr:0002", "draft-hr:0001"} <= set(result)
    assert len(embedder.queries) >= 2


def test_hyde_passages_are_embedded_like_documents():
    class Spy(ConceptEmbedder):
        def __init__(self):
            super().__init__()
            self.documents = []

        def embed_documents(self, texts, progress=None):
            self.documents.extend(texts)
            return super().embed_documents(texts, progress)

    embedder = Spy()
    _, retriever, _, _ = build(embedder)
    embedder.documents.clear()
    retriever.retrieve(QueryPlan("q", "q", (DenseQuery("passage text about a breach", as_passage=True),), "q"), RetrievalOptions(mode="dense"))
    assert embedder.documents == ["passage text about a breach"] and embedder.queries == []


def test_rerank_reorders_the_candidates_and_reports_its_scores():
    class FavourOmnibus:
        name = "fake"

        def score(self, query, passages):
            return [10.0 if "2 December 2027" in p else float(len(p) % 7) for p in passages]

    store, _, embedder, records = build()
    from reglens.retrieval import Retriever

    retriever = Retriever(store, embedder, records, FavourOmnibus())
    plain = retriever.retrieve("when do high-risk obligations apply", RetrievalOptions(top_k=3))
    reranked = retriever.retrieve("when do high-risk obligations apply", RetrievalOptions(top_k=3, rerank=True, rerank_pool=8))
    assert reranked[0].id == "omnibus:0001" and reranked[0].rerank_score == 10.0 and reranked[0].score == 10.0
    assert all(r.rerank_score is not None for r in reranked)
    assert plain[0].rerank_score is None
    assert len(reranked) == 3 and retriever.has_reranker


def test_rerank_pool_limits_what_the_reranker_sees():
    seen = []

    class Spy:
        name = "spy"

        def score(self, query, passages):
            seen.append(len(passages))
            return [0.0] * len(passages)

    store, _, embedder, records = build()
    from reglens.retrieval import Retriever

    Retriever(store, embedder, records, Spy()).retrieve("personal data", RetrievalOptions(top_k=2, rerank=True, rerank_pool=4))
    assert seen == [4]


def test_rerank_is_stable_for_equal_scores():
    class Flat:
        name = "flat"

        def score(self, query, passages):
            return [1.0] * len(passages)

    store, retriever0, embedder, records = build()
    from reglens.retrieval import Retriever

    base = ids(retriever0.retrieve("breach notification", RetrievalOptions(top_k=4)))
    flat = ids(Retriever(store, embedder, records, Flat()).retrieve("breach notification", RetrievalOptions(top_k=4, rerank=True, rerank_pool=10)))
    assert flat == base


def test_rerank_without_a_model_is_an_error():
    _, retriever, _, _ = build()
    with pytest.raises(RetrievalError, match="re-rank"):
        retriever.retrieve("x", RetrievalOptions(rerank=True))


def test_options_are_validated():
    for bad in ({"mode": "magic"}, {"top_k": 0}, {"candidates": 0}, {"rerank_pool": 0}, {"dense_weight": -1.0}):
        with pytest.raises(ValueError):
            RetrievalOptions(**bad)


def test_no_match_gives_an_empty_list():
    _, retriever, _, _ = build()
    assert retriever.retrieve("zzzzqqq", RetrievalOptions(mode="lexical")) == []
    assert len(retriever) == len(chunks())


# ----------------------------------------------------------------------------------------------- cross-encoder wrapper


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = None

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs = list(pairs)
        self.batch = batch_size
        return [float(len(p[1])) for p in pairs]


def test_cross_encoder_scores_pairs_and_loads_lazily():
    model, loads = FakeCrossEncoder(), []
    reranker = CrossEncoderReranker("some/cross", batch_size=4, loader=lambda name, device: (loads.append(name), model)[1])
    assert loads == [] and reranker.score("q", []) == []
    assert reranker.score("question", ["aa", "bbbb"]) == [2.0, 4.0]
    reranker.score("question", ["c"])
    assert loads == ["some/cross"] and model.batch == 4


def test_cross_encoder_failures_are_reported():
    class Broken(FakeCrossEncoder):
        def predict(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RerankError, match="boom"):
        CrossEncoderReranker("m", loader=lambda *a: Broken()).score("q", ["p"])

    class Nan(FakeCrossEncoder):
        def predict(self, pairs, **k):
            return [float("nan")] * len(pairs)

    with pytest.raises(RerankError):
        CrossEncoderReranker("m", loader=lambda *a: Nan()).score("q", ["p"])

    class Short(FakeCrossEncoder):
        def predict(self, pairs, **k):
            return [1.0]

    with pytest.raises(RerankError):
        CrossEncoderReranker("m", loader=lambda *a: Short()).score("q", ["p", "q"])


def test_cross_encoder_missing_library(monkeypatch):
    import sys

    from reglens.retrieval.rerank import load_cross_encoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankError, match="pip install"):
        load_cross_encoder("x")


# ----------------------------------------------------------------------------------------------- context assembly


def retrieved(n=5, words=100, prefix="t"):
    from reglens.retrieval.retriever import Retrieved
    from reglens.vectorstore.base import Record

    return [Retrieved(Record(f"id{i}", f"{prefix}{i} " + "word " * (words - 1), {"doc_id": "d"}), rank=i + 1, score=1.0 / (i + 1)) for i in range(n)]


def test_context_labels_follow_the_rank_order():
    context = assemble_context(retrieved(3))
    assert [i.label for i in context.items] == ["S1", "S2", "S3"]
    assert [i.retrieved.id for i in context.items] == ["id0", "id1", "id2"]
    assert context.by_label("S2").retrieved.id == "id1" and context.by_label("S9") is None
    assert context.words == 300 and context.dropped == ()


def test_context_respects_the_word_budget_but_always_keeps_the_best_chunk():
    context = assemble_context(retrieved(5, words=100), max_words=250)
    assert [i.retrieved.id for i in context.items] == ["id0", "id1"]
    assert context.dropped == ("id2", "id3", "id4")
    huge = assemble_context(retrieved(2, words=500), max_words=100)
    assert [i.retrieved.id for i in huge.items] == ["id0"]  # the first chunk is kept even though it exceeds the budget


def test_context_skips_a_chunk_that_does_not_fit_and_tries_the_next_one():
    from reglens.retrieval.retriever import Retrieved
    from reglens.vectorstore.base import Record

    rs = [Retrieved(Record("big", "w " * 200, {}), 1, 1.0), Retrieved(Record("bigger", "w " * 300, {}), 2, 0.9), Retrieved(Record("small", "w " * 20, {}), 3, 0.8)]
    context = assemble_context(rs, max_words=250)
    assert [i.retrieved.id for i in context.items] == ["big", "small"] and context.dropped == ("bigger",)


def test_context_skips_exact_duplicates_and_caps_the_count():
    from reglens.retrieval.retriever import Retrieved
    from reglens.vectorstore.base import Record

    rs = [Retrieved(Record("a", "Same   text here", {}), 1, 1.0), Retrieved(Record("b", "same text HERE", {}), 2, 0.9), Retrieved(Record("c", "other", {}), 3, 0.8)]
    assert [i.retrieved.id for i in assemble_context(rs).items] == ["a", "c"]
    assert len(assemble_context(retrieved(10, words=5), max_items=3).items) == 3
    with pytest.raises(ValueError):
        assemble_context([], max_words=0)
    assert assemble_context([]).items == ()


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_rrf_accepts_k_zero_but_not_a_negative_k():
    assert reciprocal_rank_fusion([["a", "b"]], k=0) == [("a", 1.0), ("b", 0.5)]
    with pytest.raises(ValueError, match="k must not be negative"):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_rrf_ties_keep_the_order_in_which_the_ids_were_first_seen_not_alphabetical_order():
    fused = reciprocal_rank_fusion([["b"], ["a"]])
    assert [item for item, _ in fused] == ["b", "a"] and fused[0][1] == fused[1][1]
    assert [item for item, _ in reciprocal_rank_fusion([["c", "a"], ["b"]])] == ["c", "b", "a"]


def test_rrf_needs_one_weight_per_ranking():
    with pytest.raises(ValueError, match="one weight per ranking"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def _passage(identifier, words):
    from reglens.retrieval.retriever import Retrieved
    from reglens.vectorstore.base import Record

    return Retrieved(Record(identifier, " ".join(f"{identifier}w{i}" for i in range(words)), {}), rank=1, score=1.0)


def test_a_passage_that_exactly_fills_the_word_budget_is_kept():
    from reglens.retrieval import assemble_context

    context = assemble_context([_passage("a", 3), _passage("b", 3), _passage("c", 1)], max_words=6)
    assert [item.retrieved.id for item in context.items] == ["a", "b"] and context.dropped == ("c",) and context.words == 6


def test_the_first_passage_is_kept_even_when_it_alone_exceeds_the_budget():
    from reglens.retrieval import assemble_context

    context = assemble_context([_passage("big", 50), _passage("small", 1)], max_words=10)
    assert [item.retrieved.id for item in context.items] == ["big"] and context.dropped == ("small",)


def test_context_limits_must_be_positive():
    from reglens.retrieval import assemble_context

    with pytest.raises(ValueError):
        assemble_context([_passage("a", 1)], max_items=0)
    with pytest.raises(ValueError):
        assemble_context([_passage("a", 1)], max_words=0)


def test_bm25_scores_follow_the_textbook_formula_with_the_default_parameters():
    texts = ["alpha beta", "alpha alpha gamma delta epsilon", "zeta eta theta"]
    index = BM25Index(["d0", "d1", "d2"], texts)
    assert (index.k1, index.b) == (1.5, 0.75)
    n, df, average = 3, 2, (2 + 5 + 3) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(tf, length):
        return idf * tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * length / average))

    got = dict(index.search("alpha", 5))
    assert got["d0"] == pytest.approx(score(1, 2)) and got["d1"] == pytest.approx(score(2, 5)) and "d2" not in got


def test_repeating_a_word_in_the_query_does_not_change_the_scores():
    index = BM25Index(["a", "b"], ["data breach notice", "breach of contract"])
    assert index.search("breach breach breach", 5) == index.search("breach", 5)


def test_stemming_boundaries():
    assert [stem(w) for w in ("ties", "dies", "lies")] == ["tie", "die", "lie"]  # four letters: the "ies" rule needs five
    assert [stem(w) for w in ("authorities", "parties")] == ["authority", "party"]
    assert [stem(w) for w in ("wishes", "fishes", "batches", "taxes", "addresses")] == ["wish", "fish", "batch", "tax", "address"]
    assert [stem(w) for w in ("gas", "bus", "tax", "its")] == ["gas", "bus", "tax", "its"]  # too short to be a plural
    assert [stem(w) for w in ("controllers", "systems", "ai", "analysis", "status", "process")] == ["controller", "system", "ai", "analysis", "status", "process"]


# ------------------------------------------------------------------ the fusion stage, checked against an independent computation


def _lists(retriever, plan, **options):
    dense = ids(retriever.retrieve(plan, RetrievalOptions(mode="dense", top_k=10, candidates=10, **options)))
    lexical = ids(retriever.retrieve(plan, RetrievalOptions(mode="lexical", top_k=10, candidates=10, **options)))
    return dense, lexical


@pytest.mark.parametrize(("k", "dense_weight", "lexical_weight"), [(60, 1.0, 1.0), (5, 1.0, 1.0), (60, 2.0, 0.5), (0, 1.0, 3.0)])
def test_hybrid_scores_are_the_reciprocal_rank_fusion_of_the_two_searches(k, dense_weight, lexical_weight):
    _, retriever, _, _ = build()
    plan = QueryPlan.plain("data leak regulator 72 hours")
    dense, lexical = _lists(retriever, plan)
    expected = reciprocal_rank_fusion([dense, lexical], k=k, weights=[dense_weight, lexical_weight])
    got = retriever.retrieve(plan, RetrievalOptions(top_k=10, candidates=10, rrf_k=k, dense_weight=dense_weight, lexical_weight=lexical_weight))
    assert [(r.id, r.fused_score) for r in got] == [(i, pytest.approx(s)) for i, s in expected]


def test_several_dense_queries_together_weigh_as_much_as_one():
    _, retriever, _, _ = build()
    first, second, keywords = "leak regulator", "erase delete forget", "72 hours breach"
    plan = QueryPlan("q", "q", (DenseQuery(first), DenseQuery(second)), keywords)
    dense_one = ids(retriever.retrieve(QueryPlan.plain(first), RetrievalOptions(mode="dense", top_k=10, candidates=10)))
    dense_two = ids(retriever.retrieve(QueryPlan.plain(second), RetrievalOptions(mode="dense", top_k=10, candidates=10)))
    lexical = ids(retriever.retrieve(QueryPlan.plain(keywords), RetrievalOptions(mode="lexical", top_k=10, candidates=10)))
    expected = reciprocal_rank_fusion([dense_one, dense_two, lexical], weights=[0.5, 0.5, 1.0])
    got = retriever.retrieve(plan, RetrievalOptions(top_k=10, candidates=10))
    assert [(r.id, r.fused_score) for r in got] == [(i, pytest.approx(s)) for i, s in expected]


def test_with_several_dense_queries_each_chunk_reports_its_best_score_and_best_rank():
    _, retriever, _, _ = build()
    first, second = "leak regulator", "erase delete forget"
    options = RetrievalOptions(mode="dense", top_k=10, candidates=10)
    single = [{r.id: (r.dense_score, r.dense_rank) for r in retriever.retrieve(QueryPlan.plain(q), options)} for q in (first, second)]
    both = retriever.retrieve(QueryPlan("q", "q", (DenseQuery(first), DenseQuery(second)), "q"), options)
    assert len(both) == 10
    for result in both:
        scores, ranks = zip(single[0][result.id], single[1][result.id], strict=True)
        assert result.dense_score == pytest.approx(max(scores)) and result.dense_rank == min(ranks)


def test_a_search_that_finds_nothing_does_not_count_as_a_second_opinion():
    _, retriever, _, _ = build()
    results = retriever.retrieve("leak regulator", RetrievalOptions(top_k=3))  # the keyword search finds nothing here
    assert results[0].id == "gdpr:0001" and results[0].lexical_score is None
    assert results[0].fused_score is None and results[0].score == results[0].dense_score


def test_the_keyword_search_uses_the_keyword_query_of_the_plan_not_the_question():
    _, retriever, _, _ = build()
    plan = QueryPlan("something unrelated", "something unrelated", (DenseQuery("something unrelated"),), "personal data breach supervisory authority")
    assert ids(retriever.retrieve(plan, RetrievalOptions(mode="lexical", top_k=1))) == ["gdpr:0001"]
    assert retriever.retrieve(QueryPlan.plain("something unrelated"), RetrievalOptions(mode="lexical")) == []


def test_candidates_limits_what_each_search_returns():
    _, retriever, _, _ = build()
    assert len(retriever.retrieve("data leak", RetrievalOptions(mode="dense", top_k=8, candidates=2))) == 2
    assert len(retriever.retrieve("personal data", RetrievalOptions(mode="lexical", top_k=8, candidates=2))) == 2
    assert len(retriever.retrieve("personal data", RetrievalOptions(mode="lexical", top_k=8, candidates=8))) > 2


def test_the_reranker_reads_each_passage_together_with_its_document_and_section_line():
    from reglens.vectorstore.records import lexical_text

    seen = []

    class Spy:
        name = "spy"

        def score(self, query, passages):
            seen.extend(passages)
            return [0.0] * len(passages)

    store, _, embedder, records = build()
    from reglens.retrieval import Retriever

    retriever = Retriever(store, embedder, records, Spy())
    results = retriever.retrieve("breach notification", RetrievalOptions(top_k=3, rerank=True, rerank_pool=4))
    assert len(seen) == 4 and all("\n" in passage for passage in seen)
    by_id = {record.id: record for record in records}
    assert set(seen) >= {lexical_text(by_id[r.id]) for r in results}
    contexts = {record.metadata["context"] for record in records}
    assert all(passage.split("\n", 1)[0] in contexts for passage in seen)  # first line: "document | section"


@pytest.mark.parametrize(
    "bad",
    [{"mode": "fuzzy"}, {"top_k": 0}, {"candidates": 0}, {"rerank_pool": 0}, {"dense_weight": -0.1}, {"lexical_weight": -1.0}],
)
def test_retrieval_options_are_validated(bad):
    with pytest.raises(ValueError):
        RetrievalOptions(**bad)
    assert RetrievalOptions(dense_weight=0.0, lexical_weight=0.0).top_k == 8


# ----------------------------------------------------------------------------------------- catalogue and neighbours


def test_the_catalog_lists_every_document_with_its_metadata_and_size():
    import ragfixtures

    _, retriever, _, _ = build()
    catalog = retriever.catalog()
    assert [entry["doc_id"] for entry in catalog] == sorted(source.id for source in ragfixtures.SOURCES)
    gdpr = next(entry for entry in catalog if entry["doc_id"] == "gdpr")
    assert gdpr["passages"] == 3 and gdpr["jurisdiction"] == "EU" and gdpr["status"] == "in_force" and gdpr["doc_type"] == "regulation"
    assert gdpr["url"] == "https://example.org/gdpr" and gdpr["title"].startswith("General Data Protection") and gdpr["version_date"] == "2016"
    assert sum(entry["passages"] for entry in catalog) == len(retriever)


def test_a_document_without_metadata_is_listed_under_its_id():
    from reglens.embeddings import HashingEmbedder
    from reglens.retrieval import Retriever
    from reglens.vectorstore import NumpyStore, Record

    retriever = Retriever(NumpyStore(), HashingEmbedder(), [Record("x:1", "text", {"doc_id": "x"})])
    assert retriever.catalog() == [{"doc_id": "x", "passages": 1, "title": "x"}]


def test_a_passage_can_be_looked_up_and_has_neighbours_in_reading_order():
    _, retriever, _, _ = build()
    assert retriever.record("gdpr:0002").text.startswith("The data subject shall have the right to obtain")
    assert retriever.record("nope") is None
    before, after = retriever.neighbours("gdpr:0002")
    assert (before.id, after.id) == ("gdpr:0001", "gdpr:0003")
    assert retriever.neighbours("gdpr:0001")[0] is None and retriever.neighbours("gdpr:0001")[1].id == "gdpr:0002"
    assert retriever.neighbours("gdpr:0003")[1] is None
    assert retriever.neighbours("omnibus:0001") == (None, None)  # the only passage of its document
    assert retriever.neighbours("nope") == (None, None)


def test_neighbours_stay_inside_one_document_and_one_kind():
    from reglens.embeddings import HashingEmbedder
    from reglens.retrieval import Retriever
    from reglens.vectorstore import NumpyStore, Record

    records = [
        Record("d:3", "three", {"doc_id": "d", "kind": "text", "ordinal": 3}),
        Record("d:1", "one", {"doc_id": "d", "kind": "text", "ordinal": 1}),
        Record("d:2", "note", {"doc_id": "d", "kind": "notes", "ordinal": 2}),
        Record("e:1", "other", {"doc_id": "e", "kind": "text", "ordinal": 1}),
        Record("f:b", "b", {"doc_id": "f"}),  # no ordinal: falls back to the id
        Record("f:a", "a", {"doc_id": "f"}),
    ]
    retriever = Retriever(NumpyStore(), HashingEmbedder(), records)

    def ids(chunk_id):
        return [record.id if record else None for record in retriever.neighbours(chunk_id)]

    assert ids("d:1") == [None, "d:3"] and ids("d:3") == ["d:1", None]
    assert ids("d:2") == [None, None] and ids("e:1") == [None, None]
    assert ids("f:a") == [None, "f:b"] and ids("f:b") == ["f:a", None]
