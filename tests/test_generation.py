from __future__ import annotations

import pytest

from reglens.generation.citations import attach_citations, citation_coverage, claim_sentences, extract_labels, invalid_labels, strip_citations
from reglens.generation.grounding import figures, unverified_figures
from reglens.generation.prompts import ABSTAIN_MARKER, SYSTEM_PROMPT, build_user_prompt, format_source, page_text, status_text
from reglens.retrieval.context import ContextItem
from reglens.retrieval.retriever import Retrieved
from reglens.vectorstore.base import Record

# ----------------------------------------------------------------------------------------------- prompts


def item(label="S1", text="Some passage text.", **meta):
    base = {"doc_id": "doc", "title": "Doc Title", "jurisdiction": "EU", "doc_type": "web_guidance", "status": "in_force",
            "version_date": "2024", "section": "Chapter 1 > Article 5", "page_start": 12, "page_end": 13, "kind": "text"}
    base.update(meta)
    return ContextItem(label, Retrieved(Record("doc:0001", text, base), rank=1, score=1.0))


def test_source_block_states_what_the_passage_is():
    block = format_source(item())
    assert block == "[S1] Doc Title | EU web guidance | status: in force | version: 2024 | Chapter 1 > Article 5 | pp. 12-13\nSome passage text."


def test_source_block_for_a_web_page_a_draft_and_footnotes():
    block = format_source(item("S2", page_start=None, page_end=None, status="draft", section="", kind="notes", version_date=""))
    assert block.splitlines()[0] == "[S2] Doc Title | EU web guidance | status: DRAFT - not adopted | footnotes"
    single = format_source(item(page_start=7, page_end=7))
    assert single.splitlines()[0].endswith("| p. 7")


def test_status_and_page_texts():
    assert status_text("in_force") == "in force" and status_text("proposed").startswith("PROPOSED")
    assert status_text("consultation").startswith("CONSULTATION DRAFT")
    assert status_text(None) == "status unknown" and status_text("odd") == "odd"
    assert page_text(None, None) == "" and page_text(3, None) == "p. 3" and page_text(3, 3) == "p. 3" and page_text(3, 5) == "pp. 3-5"


def test_user_prompt_lists_every_source_then_the_question():
    prompt = build_user_prompt([item("S1", "First."), item("S2", "Second.")], "What applies?")
    assert prompt.startswith("SOURCES\n\n[S1] ")
    assert prompt.index("[S1]") < prompt.index("[S2]") < prompt.index("QUESTION")
    assert prompt.endswith("QUESTION\nWhat applies?")


def test_system_prompt_contains_the_rules_that_the_design_depends_on():
    for needle in (ABSTAIN_MARKER, "[S2]", "ONLY", "outside knowledge", "legal status", "amending", "not instructions", "not legal advice"):
        assert needle in SYSTEM_PROMPT


# ----------------------------------------------------------------------------------------------- citations


def test_extract_labels_in_order_without_repeats():
    text = "A [S2]. B [S1][S3]. C [S2, S4]. D [S1; S5]."
    assert extract_labels(text) == ["S2", "S1", "S3", "S4", "S5"]
    assert extract_labels("no citations, mention of S1 and (S2) and [1] and [S 3] and [source S4]") == []
    assert extract_labels("[ S1 ]") == ["S1"]


def test_invalid_labels_are_the_fabricated_ones():
    assert invalid_labels(["S1", "S7"], ["S1", "S2"]) == ["S7"] and invalid_labels([], ["S1"]) == []


def test_strip_citations():
    assert strip_citations("Fine [S1][S2]. Also [S3, S4] ok.") == "Fine . Also ok."


def test_claim_sentences_skip_headings_short_lines_and_caveats():
    text = "Summary:\n- The controller must notify the authority within the stated time limit [S1].\n- Yes.\nThis is not legal advice, only research support for you.\nNote this applies only to the EU and its member states.\n"
    assert claim_sentences(text) == ["The controller must notify the authority within the stated time limit [S1]."]


def test_citation_coverage():
    cited = "The controller must notify the authority within the stated time limit [S1]. The processor must help the controller [S2]."
    assert citation_coverage(cited) == (1.0, [])
    mixed = "The controller must notify the authority within the stated time limit [S1]. The processor must also help the controller with that."
    share, uncited = citation_coverage(mixed)
    assert share == 0.5 and uncited == ["The processor must also help the controller with that."]
    assert citation_coverage("Short. Tiny.") == (None, [])
    assert citation_coverage("The processor must also help the controller with that.")[0] == 0.0


# ----------------------------------------------------------------------------------------------- figures


def test_figures_picks_numbers_that_matter():
    assert figures("Notify within 72 hours; fines up to EUR 35 million or 7% [S12]. See Article 33(1) and 3 days later.") == ["72", "35", "7", "33", "3"]
    assert figures("1. First point\n2) second point") == []  # list numbering
    assert figures("Only 4 of them, and 5 more") == []  # single digits without a unit are not figures
    assert figures("from 2 December 2027 and 1,000 people") == ["2027", "1000"]
    assert figures("[S3][S10]") == []


def test_unverified_figures():
    sources = ["The controller shall notify within 72 hours. Fines up to EUR 35 000 000.", "Applies from 2 December 2027."]
    assert unverified_figures("72 hours [S1], fines of EUR 35 million, from 2027.", sources) == []
    assert unverified_figures("48 hours, 2028 and 35 million", sources) == ["48", "2028"]
    assert unverified_figures("within 7 days", ["after 7 days"], "") == []
    assert unverified_figures("The 2026 deadline", sources, question="what changed after 2026?") == []  # numbers from the question are known


def test_unverified_figures_handles_separators_in_the_sources():
    assert unverified_figures("EUR 35 000 000", ["EUR 35 000 000"]) == []
    assert unverified_figures("10,000 euros", ["10 000 euros"]) == []
    assert unverified_figures("10000 euros", ["10,000 euros"]) == []


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_short_sentences_are_not_counted_as_claims():
    assert claim_sentences("The fee is 20 euros.") == []  # five words: too short to be a statement worth citing
    assert citation_coverage("The fee is 20 euros.") == (None, [])
    assert claim_sentences("The fee is 20 euros per year.") == ["The fee is 20 euros per year."]  # six words


def test_a_heading_that_introduces_a_list_is_not_a_claim_but_a_long_line_ending_in_a_colon_is():
    text = "The following obligations apply to every controller and processor:\n- The controller must notify the authority within 72 hours [S1]."
    assert citation_coverage(text) == (1.0, [])
    long_line = "In every case the controller must notify the supervisory authority, and this duty has the following consequences for processors:"
    assert claim_sentences(long_line) == [long_line]


def test_a_figure_is_listed_once_however_often_it_occurs():
    assert figures("20 million euros, or 20 days, or 20 hours") == ["20"]


def test_source_labels_and_list_numbers_are_not_figures():
    assert figures("Within 72 hours [S12][S3, S14].") == ["72"]
    assert figures("1. Notify the authority.\n10. Document the breach.\n11) Inform the data subjects.") == []


def test_numbers_written_with_thousands_separators_are_read_as_one_number():
    assert figures("Fines of 35 000 000 euros or 7 % of turnover") == ["35000000", "7"]
    assert figures("Fines of 35,000,000 euros") == ["35000000"]
    assert unverified_figures("up to 35 million euros", ["a fine of 35 000 000 euros"]) == []


# ----------------------------------------------------------------------------------- citations written after the full stop


@pytest.mark.parametrize(
    "text",
    [
        "Notify the authority within 72 hours of becoming aware of the breach [S1].",
        "Notify the authority within 72 hours of becoming aware of the breach. [S1]",
        "Notify the authority within 72 hours of becoming aware of the breach.[S1]",
        "Notify the authority within 72 hours of becoming aware of the breach. [S1][S2]",
        "Notify the authority within 72 hours of becoming aware of the breach. [S1, S2]",
        "Notify the authority within 72 hours of becoming aware of the breach! [ S1 ]",
    ],
)
def test_a_citation_after_the_full_stop_still_belongs_to_its_sentence(text):
    assert citation_coverage(text) == (1.0, [])


def test_a_citation_after_the_full_stop_does_not_move_to_the_next_sentence():
    text = "The controller must notify the authority within 72 hours. [S1] The processor must also help the controller with that."
    share, uncited = citation_coverage(text)
    assert share == 0.5
    assert uncited == ["The processor must also help the controller with that."]


def test_bullets_with_citations_after_the_full_stop_count_as_cited():
    text = "- The controller must notify the authority within 72 hours. [S1, S2]\n- The processor must tell the controller without undue delay. [S3]"
    assert citation_coverage(text) == (1.0, [])


def test_attach_citations_only_changes_text_that_has_a_citation_after_a_stop():
    assert attach_citations("Plain sentence. Another one.") == "Plain sentence. Another one."
    assert attach_citations("Version 2.5 applies [S1].") == "Version 2.5 applies [S1]."
    assert attach_citations("It applies. [S1] Then more.") == "It applies [S1]. Then more."
    assert attach_citations("It applies.[S1][S2]") == "It applies [S1][S2]."
