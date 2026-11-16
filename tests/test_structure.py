import pytest

from reglens.ingest import layout, pdfread, structure
from reglens.ingest.layout import PageLines, RawNote
from reglens.ingest.models import FOOTNOTE, HEADING, PARAGRAPH, OutlineEntry
from reglens.ingest.structure import build_blocks, ends_sentence
from fixtures import regulation_pages, statement_pages
from helpers import bold, make_multipage_pdf, typeset

BODY = "This sentence is long enough to look like a real wrapped line of body text in a document page."
FILLER = [BODY, BODY.replace("sentence", "statement"), BODY.replace("document", "report"), "A final sentence ends the paragraph here."]


def pages_of(*pages: list[str]) -> list[PageLines]:
    return [PageLines(number, list(lines)) for number, lines in enumerate(pages, start=1)]


def blocks_of(*pages: list[str]):
    return build_blocks(pages_of(*pages))


def pipeline(raw_pages: list[list[str]], outline=None):
    """Layout and structure together, as the ingestion runs them."""
    laid_out, _ = layout.prepare_pages(*typeset(raw_pages))
    return build_blocks(laid_out, outline)


def headings(blocks):
    return [(b.level, b.text) for b in blocks if b.kind == HEADING]


def texts(blocks, kind=PARAGRAPH):
    return [b.text for b in blocks if b.kind == kind]


# ------------------------------------------------------------------------------------ ends_sentence


@pytest.mark.parametrize(
    "line", ["It ends here.", "Is it so?", "Stop!", "as follows:", "one;", "He said “yes.”", "(see above.)", "It ends. [12]", "It ends.” [3] [4]"]
)
def test_lines_that_end_a_sentence(line):
    assert ends_sentence(line)


@pytest.mark.parametrize("line", ["and then", "the Commission,", "in the year 2016", "Article 5"])
def test_lines_that_do_not_end_a_sentence(line):
    assert not ends_sentence(line)


# -------------------------------------------------------------------------------- headings: numbered


def test_numbered_headings_get_nested_levels_and_body_text_stays_paragraphs():
    blocks, report = blocks_of(
        ["1. Introduction", *FILLER, "2. Scope of the guidance", *FILLER, "2.1 Who is affected", *FILLER]
    )
    assert headings(blocks) == [(1, "1. Introduction"), (1, "2. Scope of the guidance"), (2, "2.1 Who is affected")]
    assert len(texts(blocks)) == 3
    assert report.headings == 3


def test_a_number_at_the_start_of_a_sentence_is_not_a_heading():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "10 percent of respondents said otherwise in the survey and the text goes on.", *FILLER])
    assert headings(blocks) == [(1, "1. Introduction")]


def test_a_numbered_lead_in_with_a_modal_verb_or_colon_is_not_a_heading():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "2. Personal data shall be:", "(a) processed lawfully;", "(b) collected for purposes."])
    assert headings(blocks) == [(1, "1. Introduction")]


def test_a_numbered_sentence_without_punctuation_is_not_a_heading():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "3. The controller shall keep records", "Each record must contain the following data and be kept for five years in total.", *FILLER])
    assert headings(blocks) == [(1, "1. Introduction")]


def test_a_numbered_lead_in_that_ends_with_a_colon_is_not_a_heading():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "2. Types of data processed:", "(a) names and addresses;", "(b) bank details."])
    assert headings(blocks) == [(1, "1. Introduction")]


def test_a_date_is_not_mistaken_for_a_heading():
    blocks, _ = blocks_of(["TITLE OF THE STATEMENT", "July 1, 2026", *FILLER])
    assert headings(blocks) == [(1, "TITLE OF THE STATEMENT")]


def test_roman_numeral_headings():
    blocks, _ = blocks_of(["I. Summary", *FILLER, "II. Background", *FILLER, "IV. A Longer Title of the Fourth Part", *FILLER])
    assert [text for _level, text in headings(blocks)] == ["I. Summary", "II. Background", "IV. A Longer Title of the Fourth Part"]


def test_a_numbered_heading_may_be_followed_by_the_first_item_of_a_list():
    for item in ("a) AI actors should respect the rule of law.", "(a) AI actors should respect the rule of law.", "(i) first the rules"):
        blocks, _ = blocks_of(["1. Introduction", *FILLER, "1.4. Robustness, security and safety", item, *FILLER[1:]])
        assert [text for _level, text in headings(blocks)] == ["1. Introduction", "1.4. Robustness, security and safety"], item
        assert any(paragraph.startswith(item) for paragraph in texts(blocks)), item  # the item starts a paragraph


def test_a_numbered_line_that_goes_on_in_lower_case_is_a_wrapped_sentence_not_a_heading():
    for continuation in ("its main steps follow", "ands the like", "a lowercase line that is no list item"):
        blocks, _ = blocks_of(["1. Introduction", *FILLER, "4. Overview of the Process and", continuation, *FILLER])
        assert [text for _level, text in headings(blocks)] == ["1. Introduction"], continuation


# ------------------------------------------------------------------------------ headings: keywords


def test_keyword_headings_with_the_title_on_the_same_or_the_next_line():
    blocks, _ = blocks_of(
        ["Article 5: Lawfulness of processing", *FILLER, "Section 7", "Transparency", *FILLER, "Annex II", *FILLER]
    )
    assert headings(blocks) == [
        (3, "Article 5 – Lawfulness of processing"),
        (2, "Section 7 – Transparency"),
        (1, "Annex II"),
    ]


UNFINISHED = BODY.rstrip(".")  # a full-width line whose sentence goes on in the next line


def test_a_line_that_starts_with_a_label_inside_a_wrapped_sentence_is_not_a_heading():
    lines = ["1. Introduction", *FILLER, UNFINISHED, "Annex I. In certain cases, it is possible for such legislation to lay down rules", *FILLER]
    assert [text for _level, text in headings(blocks_of(lines)[0])] == ["1. Introduction"]
    lines = ["1. Introduction", *FILLER, UNFINISHED, "Article 5 – This means that the premium subscribers get a better service", *FILLER]
    assert [text for _level, text in headings(blocks_of(lines)[0])] == ["1. Introduction"]


def test_the_same_line_is_a_heading_when_a_paragraph_ends_before_it():
    lines = ["1. Introduction", *FILLER, "Annex I. List of examples", *FILLER]
    assert [text for _level, text in headings(blocks_of(lines)[0])] == ["1. Introduction", "Annex I – List of examples"]
    lines = ["1. Introduction", *FILLER, "a short line", "Annex I. List of examples", *FILLER]  # a short line ends a block too
    assert "Annex I – List of examples" in [text for _level, text in headings(blocks_of(lines)[0])]


def test_a_label_in_heading_type_is_a_heading_even_after_an_unfinished_line():
    for item in (("Annex I. List of examples", 10.0, True), ("Annex I. List of examples", 12.0, False)):
        page = typed_page(*FILLER, UNFINISHED, item, *FILLER)
        blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
        assert "Annex I – List of examples" in [text for _level, text in headings(blocks)], item
    page = typed_page(*FILLER, UNFINISHED, "Annex I. List of examples", *FILLER)  # in running type it is no heading
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks) == []


def test_a_bare_label_and_title_inside_a_wrapped_sentence_are_no_heading_unless_typeset_as_one():
    lines = ["1. Introduction", *FILLER, UNFINISHED, "Article 5", "Prohibited AI practices", *FILLER]
    assert [text for _level, text in headings(blocks_of(lines)[0])] == ["1. Introduction"]
    page = typed_page(*FILLER, UNFINISHED, ("Article 5", 12.0, True), ("Prohibited AI practices", 12.0, True), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)][:1] == ["Article 5 – Prohibited AI practices"]


def test_the_adoption_formula_ends_the_recitals_and_opens_the_enacting_terms():
    for formula in ("HAVE ADOPTED THIS REGULATION:", "HAS ADOPTED THIS DIRECTIVE:", "HAS ADOPTED THIS DECISION"):
        blocks, _ = blocks_of(["Whereas:", *FILLER, formula, "Article 1", "Subject matter", *FILLER])
        found = headings(blocks)
        assert [text for _level, text in found] == ["Recitals", "Enacting terms", "Article 1 – Subject matter"], formula
        assert found[0][0] == found[1][0] < found[2][0]  # as high in the hierarchy as the recitals
        assert not any(formula in text for text in texts(blocks))  # the formula became the heading


def test_other_adoption_formulas_and_formulas_inside_a_sentence_are_no_enacting_terms():
    lines = ["Whereas:", *FILLER, "HAS ADOPTED THE FOLLOWING GUIDELINES", *FILLER, f"{BODY} HAVE ADOPTED THIS REGULATION:", *FILLER]
    assert "Enacting terms" not in [text for _level, text in headings(blocks_of(lines)[0])]
    lines = ["Whereas:", *FILLER, "HAVE ADOPTED THIS REGULATION: and more words follow on the same line", *FILLER]
    assert "Enacting terms" not in [text for _level, text in headings(blocks_of(lines)[0])]


AMENDING = [
    "Article 1", "Amendments to Regulation (EU) 2024/1689", *FILLER,
    "(32) the following articles are inserted:",
    "‘Article 75a", "Supervision", *FILLER, "The Member State’s authority shall be informed.", "Section 2", *FILLER,
    "Article 75b", "Commitments", *FILLER,
    "Article 75c", "Fines", *FILLER, "The Commission shall adopt the necessary measures.’;",
    "(33) Article 99 is amended as follows:", *FILLER,
    "Article 2", "Entry into force", *FILLER,
]


def test_articles_inserted_into_another_act_are_part_of_the_quotation_not_headings():
    blocks, _ = blocks_of(AMENDING)
    assert [text for _level, text in headings(blocks)] == ["Article 1 – Amendments to Regulation (EU) 2024/1689", "Article 2 – Entry into force"]
    text = " ".join(texts(blocks))
    for word in ("Article 75a", "Supervision", "Section 2", "Article 75b", "Commitments", "Article 75c", "Fines"):
        assert word in text, word  # nothing is lost
    assert "(33) Article 99 is amended" in text


def test_a_quotation_without_a_closing_mark_ends_at_the_next_article_of_the_act():
    lines = [line for line in AMENDING if "adopt the necessary measures" not in line]
    blocks, _ = blocks_of(lines)
    assert [text for _level, text in headings(blocks)] == ["Article 1 – Amendments to Regulation (EU) 2024/1689", "Article 2 – Entry into force"]


def test_after_the_closing_mark_headings_are_recognised_again():
    lines = [*AMENDING[: AMENDING.index("Article 2")], "Article 7", "Final provisions", *FILLER]
    blocks, _ = blocks_of(lines)
    assert [text for _level, text in headings(blocks)] == ["Article 1 – Amendments to Regulation (EU) 2024/1689", "Article 7 – Final provisions"]


def test_a_quotation_that_opens_and_closes_on_one_line_changes_nothing():
    lines = ["Article 1", "Amendments", *FILLER, "‘Article 5 is deleted’;", *FILLER, "Article 7", "Final provisions", *FILLER]
    blocks, _ = blocks_of(lines)
    assert [text for _level, text in headings(blocks)] == ["Article 1 – Amendments", "Article 7 – Final provisions"]


@pytest.mark.parametrize(
    "opening",
    [
        "‘(g) measures to support innovation in small companies,",  # a quotation, but not of a whole article
        "“Article 75a: Supervision of ‘draft’ acts",  # a double quotation mark does not open an insertion
        "Article 75a: Supervision of ‘draft’ acts",  # nor does a quotation mark in the middle of a line
        "(‘Article 75a: Supervision",  # nor one that does not start the line
    ],
)
def test_a_quotation_that_is_not_a_quoted_division_does_not_hide_headings(opening):
    lines = ["Article 1", "Amendments", *FILLER, opening, *FILLER, "Article 7", "Final provisions", *FILLER]
    blocks, _ = blocks_of(lines)
    assert [text for _level, text in headings(blocks)][-1] == "Article 7 – Final provisions", opening


def test_a_space_after_the_opening_mark_does_not_matter():
    lines = [line.replace("‘Article 75a", "‘ Article 75a") for line in AMENDING]
    blocks, _ = blocks_of(lines)
    assert [text for _level, text in headings(blocks)] == ["Article 1 – Amendments to Regulation (EU) 2024/1689", "Article 2 – Entry into force"]


def test_the_last_line_of_a_quotation_is_still_part_of_it():
    lines = ["Article 1", "Amendments", *FILLER, "‘Article 75a", "Supervision", *FILLER, "Closing Words Of The Quotation’", *FILLER, "Article 7", "Final provisions", *FILLER]
    found = [text for _level, text in headings(blocks_of(lines)[0])]
    assert found == ["Article 1 – Amendments", "Article 7 – Final provisions"]


def test_article_numbers_are_read_only_from_plain_article_labels():
    for text, number in (("Article 75", 75), ("Article 5 – Subject matter", 5), ("ARTICLE 12: Scope", 12), ("Article 75a", None), ("Annex 2", None), ("Section 2", None), ("5. Introduction", None)):
        assert structure._article_number(text) == number, text


def test_a_quotation_that_seems_never_to_end_stops_hiding_headings_after_twelve_pages():
    page = lambda n: [f"{BODY} {n}", BODY.replace("sentence", "statement"), "A last line ends here."]  # noqa: E731
    pages = [["Article 1", "Amendments", *FILLER, "‘Article 75a", "Supervision", *FILLER]]
    pages += [page(n) for n in range(2, 2 + structure.MAX_QUOTE_PAGES)]
    pages += [["Article 90", "Late heading", *FILLER]]
    headings_found = [text for _level, text in headings(blocks_of(*pages)[0])]
    assert headings_found == ["Article 1 – Amendments", "Article 90 – Late heading"]
    shorter = pages[: structure.MAX_QUOTE_PAGES] + [["Article 90", "Late heading", *FILLER]]  # one page less: still quoted
    assert [text for _level, text in headings(blocks_of(*shorter)[0])] == ["Article 1 – Amendments"]


def test_a_reference_in_the_middle_of_a_sentence_is_not_a_heading():
    blocks, _ = blocks_of(["as set out in", "Article 5", "of this Regulation and also more words so that this line is long enough to be body text."])
    assert headings(blocks) == []


def test_a_bare_label_inside_wrapped_text_is_not_a_heading_even_when_the_previous_line_is_short():
    blocks, _ = blocks_of([*FILLER, "as set out in", "Article 5", "of this Regulation and also more words so that this line is long enough to be body text."])
    assert headings(blocks) == []


def test_whereas_opens_the_recitals():
    blocks, _ = blocks_of(["Whereas:", "(1)", *FILLER])
    assert headings(blocks) == [(1, "Recitals")]
    assert texts(blocks)[0].startswith("(1) This sentence")


def test_section_sign_and_code_section_headings():
    blocks, _ = blocks_of(
        [
            "§ 7001. Definitions.",
            *FILLER,
            "§ 7002 Restrictions on collection",
            *FILLER,
            "1798.100. A business that controls the collection of personal information shall inform the consumer.",
            *FILLER,
        ]
    )
    assert [text for _level, text in headings(blocks)] == [
        "§ 7001 – Definitions",
        "§ 7002 – Restrictions on collection",
        "Section 1798.100",
    ]
    # what follows the code section number on the same line is the first sentence of its text
    assert texts(blocks)[2].startswith("A business that controls")


def test_all_caps_lines_merge_into_one_heading_but_a_lead_in_with_a_colon_is_text():
    blocks, _ = blocks_of(
        ["PART ONE", "GENERAL OBLIGATIONS", *FILLER, "KEY DEFINITIONS:", "DEFINITIONS AND SCOPE", *FILLER]
    )
    assert [text for _level, text in headings(blocks)] == ["PART ONE GENERAL OBLIGATIONS", "DEFINITIONS AND SCOPE"]
    assert "KEY DEFINITIONS:" in texts(blocks)


def test_all_caps_merge_stops_at_a_keyword_heading():
    blocks, _ = blocks_of(["THE PRESIDENT OF THE COMMISSION", "CHAPTER I", "General provisions", *FILLER])
    assert [text for _level, text in headings(blocks)] == ["THE PRESIDENT OF THE COMMISSION", "Chapter I – General provisions"]


def test_title_case_line_is_a_heading_but_not_when_text_continues_in_lower_case():
    blocks, _ = blocks_of(["Executive Summary", *FILLER, "Some Words In Title Case", "continues in lower case here and goes on", *FILLER])
    assert [text for _level, text in headings(blocks)] == ["Executive Summary"]


def test_the_safety_valve_switches_off_weak_rules_when_everything_looks_like_a_heading():
    glossary = [f"{word} {other}" for word in ("Alpha", "Delta", "Omega", "Sigma") for other in ("Beta", "Gamma", "Zeta", "Theta", "Kappa") for _ in range(3)]
    blocks, report = blocks_of(glossary)
    assert headings(blocks) == []
    assert report.rule_level < 2


# ------------------------------------------------------------------------------ legal act mode


def test_legal_act_mode_ignores_numbered_paragraphs_and_title_case_lines():
    articles = []
    for number in range(1, 5):
        articles += [f"Article {number}", f"Title of article {number}", f"1. {BODY}", "2. Processing shall be lawful where:", "(a) the data subject consents;"]
    blocks, report = blocks_of(articles + ["Done at Brussels.", "For the Council", "The President"])
    assert report.legal_act
    assert [text for _level, text in headings(blocks)] == [f"Article {n} – Title of article {n}" for n in range(1, 5)]
    assert "2. Processing shall be lawful where:" in texts(blocks)


def test_legal_act_mode_turns_short_numbered_paragraphs_into_text():
    articles = []
    for number in range(1, 4):
        articles += [f"Article {number}", f"Title of article {number}", BODY, "2. Scope of application", BODY]
    blocks, report = blocks_of(articles)
    assert report.legal_act and len(headings(blocks)) == 3
    assert sum("2. Scope of application" in t for t in texts(blocks)) == 3


def test_without_the_articles_the_same_numbered_line_is_a_heading():
    blocks, report = blocks_of([BODY, "2. Scope of application", BODY])
    assert not report.legal_act and (1, "2. Scope of application") in headings(blocks)


def test_without_enough_articles_numbered_headings_stay_headings():
    blocks, report = blocks_of(["Article 1", "First", *FILLER, "1. Introduction", *FILLER, "2. Scope", *FILLER])
    assert not report.legal_act
    assert (1, "1. Introduction") in headings(blocks) or any(text == "1. Introduction" for _level, text in headings(blocks))


# --------------------------------------------------------------------------------- paragraphs


def test_wrapped_lines_become_one_paragraph_and_blank_lines_separate_paragraphs():
    first = ["The controller shall implement measures to ensure that the processing of data is", "performed in accordance with this Regulation and with the other applicable rules."]
    second = ["Member States shall provide the supervisory authority with the human, technical", "and financial resources that are necessary for its tasks."]
    blocks, _ = blocks_of([*first, "", *second])
    assert texts(blocks) == [" ".join(first), " ".join(second)]


def test_a_short_line_followed_by_a_capital_starts_a_new_paragraph():
    long_a = "The controller shall implement measures to ensure that the processing of personal data is"
    blocks, _ = blocks_of([long_a, long_a, "It ends here.", long_a + " Another paragraph.", long_a])
    assert len(texts(blocks)) == 2


def test_list_markers_start_new_paragraphs_and_lone_markers_join_the_next_line():
    blocks, _ = blocks_of(["Processing is lawful where at least one of the following applies:", "(a)", "the data subject has consented to the processing;", "(b) processing is necessary for a contract;", "• a bullet item that ends here."])
    assert texts(blocks) == [
        "Processing is lawful where at least one of the following applies:",
        "(a) the data subject has consented to the processing;",
        "(b) processing is necessary for a contract;",
        "• a bullet item that ends here.",
    ]


def test_hyphenated_line_ends_are_joined_using_the_words_of_the_document():
    lines = [
        "The supervisory authority shall coop-",
        "erate with the other authorities and the organi-",
        "sation shall keep records. The non-",
        "binding opinion is public. A second non-binding opinion exists and organisation records are kept.",
    ]
    text = texts(blocks_of(lines)[0])[0]
    assert "cooperate with" in text
    assert "organisation shall" in text
    assert "non-binding opinion" in text


def test_spacing_artefacts_inside_a_paragraph_are_repaired():
    blocks, _ = blocks_of(["Of course , the Commission applies the built -in rules ( 1 ) to every case ."])
    assert texts(blocks) == ["Of course, the Commission applies the built-in rules (1) to every case."]


def test_a_paragraph_that_continues_on_the_next_page_keeps_both_page_numbers():
    page_1 = ["The controller shall implement appropriate measures to ensure that the processing of data is", "performed in accordance with this Regulation and, where necessary, those measures are"]
    page_2 = ["reviewed and updated. The next sentence follows on the second page of the document."]
    blocks, _ = blocks_of(page_1, page_2)
    (paragraph,) = [b for b in blocks if b.kind == PARAGRAPH]
    assert (paragraph.page, paragraph.page_end) == (1, 2)
    assert "are reviewed and updated" in paragraph.text


def test_footnotes_follow_the_paragraph_in_progress_when_the_page_changes():
    page_1 = PageLines(1, ["A sentence that runs over two pages and does not end on the first one because it is", "long and continues"], [RawNote(1, ["See the source."], 1)])
    page_2 = PageLines(2, ["on the second page until its end."], [RawNote(2, ["Another source."], 2)])
    blocks, report = build_blocks([page_1, page_2])
    assert [b.kind for b in blocks] == [PARAGRAPH, FOOTNOTE, FOOTNOTE]
    assert texts(blocks, FOOTNOTE) == ["[1] See the source.", "[2] Another source."]
    assert report.footnotes == 2


def test_a_note_without_a_number_is_kept_without_a_label():
    page = PageLines(1, [BODY], [RawNote(None, ["unnumbered text"], 1)])
    blocks, _ = build_blocks([page])
    assert texts(blocks, FOOTNOTE) == ["unnumbered text"]


def test_an_empty_document_gives_no_blocks():
    blocks, report = build_blocks([])
    assert blocks == [] and report.headings == 0


# --------------------------------------------------------------------------------- fixtures


def test_official_journal_fixture_structure():
    blocks, report = pipeline(regulation_pages())
    assert report.legal_act
    assert headings(blocks) == [
        (1, "REGULATION (EU) 2099/1 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL"),
        (2, "Recitals"),
        (2, "Enacting terms"),
        (2, "Chapter I – General provisions"),
        (3, "Article 1 – Subject-matter and objectives"),
        (3, "Article 2 – Material scope"),
        (3, "Article 3 – Definitions"),
        (3, "Article 4 – Principles relating to processing of personal data"),
        (3, "Article 5 – Lawfulness of processing"),
        (3, "Article 6 – Conditions for consent"),
        (2, "Chapter II – Rights of the data subject"),
        (3, "Article 7 – Transparent information"),
        (3, "Article 8 – Entry into force"),
        (2, "Annex I – List of examples"),
    ]
    body = texts(blocks)
    assert any(t.startswith("(1) The protection of natural persons") for t in body)  # recital number joined to its text
    assert "2. Personal data shall be:" in body
    assert any(t.startswith("(a) processed lawfully") for t in body)
    assert any(t.startswith("(1) 'personal data' means") for t in body)
    assert "For the European Parliament" in body and "The President" in body  # signatures are not headings
    assert not any("Official Journal of the European Union L" in t for t in body)


def test_official_journal_fixture_hyphenation_and_footnotes():
    blocks, _ = pipeline(regulation_pages())
    joined = " ".join(texts(blocks))
    assert "shall cooperate with the other authorities. An organisation shall record every breach." in joined
    assert "The non-binding opinion is public" in joined
    notes = texts(blocks, FOOTNOTE)
    assert [n.split("]")[0] for n in notes] == [f"[{k}" for k in range(1, 13)]
    assert notes[9].endswith("on the protection of natural persons, as amended.")


def test_official_journal_fixture_page_ranges():
    blocks, _ = pipeline(regulation_pages())
    article_4 = next(i for i, b in enumerate(blocks) if b.text.startswith("Article 4"))
    assert blocks[article_4].page == 4
    paragraphs_of_article_4 = []
    for block in blocks[article_4 + 1 :]:
        if block.kind == HEADING:
            break
        if block.kind == PARAGRAPH:
            paragraphs_of_article_4.append(block)
    assert {b.page for b in paragraphs_of_article_4} == {4, 5}


def test_word_export_statement_fixture_structure():
    blocks, report = pipeline(statement_pages())
    assert headings(blocks) == [
        (1, "PROPOSED POLICY STATEMENT CONCERNING THE ACCURACY OF AI SYSTEMS"),
        (2, "I. Summary"),
        (2, "II. Background"),
        (2, "III. The Commission's Authority under Section 5"),
    ]
    assert report.rule_level == 2
    body = " ".join(texts(blocks))
    assert "[1] As companies" in body  # the "0F" + "1" artefact became a marker
    assert "0F" not in body
    assert "truthful output, but" in body  # a space before a comma is repaired
    notes = texts(blocks, FOOTNOTE)
    assert len(notes) == 6
    assert notes[0].endswith("a-long-address-that-continues-on-the-next-line.pdf.")
    assert notes[4].endswith("continued text of footnote 5 that started on the page before.")


def test_the_statement_date_line_is_text_not_a_heading():
    blocks, _ = pipeline(statement_pages())
    assert "July 1, 2099" in texts(blocks)


# --------------------------------------------------------------------------------------- bookmarks


def _bookmarked_pdf(tmp_path):
    names = ["Overview of the duties", "Duties of providers", "Duties of deployers", "Penalties and fines", "Transitional rules"]
    body = [BODY, BODY, "A last sentence ends this paragraph on the page."]
    pages = [[f"{name}", *body] for name in names]
    outline = [(name, 1 if index in (0, 3) else 2, index) for index, name in enumerate(names)]
    path = tmp_path / "bookmarked.pdf"
    path.write_bytes(make_multipage_pdf(pages, outline))
    return path, names


def test_pdf_bookmarks_supply_headings_that_patterns_cannot_see(tmp_path):
    path, names = _bookmarked_pdf(tmp_path)
    extraction = pdfread.read_pdf(path)
    assert [e.title for e in extraction.outline] == names
    laid_out, _ = layout.prepare_pages(extraction.pages)

    without, report_without = build_blocks(laid_out)
    assert headings(without) == [] and not report_without.outline_used

    blocks, report = build_blocks(laid_out, extraction.outline)
    assert report.outline_used and report.outline_matches == 5
    assert headings(blocks) == [(1, names[0]), (2, names[1]), (2, names[2]), (1, names[3]), (2, names[4])]


def test_bookmarks_are_ignored_when_too_few_match_the_text():
    pages = pages_of([BODY, BODY, "Some body text on the first page that ends here."])
    outline = [OutlineEntry(f"Chapter that is not in the text {n}", 1, 1) for n in range(6)]
    blocks, report = build_blocks(pages, outline)
    assert not report.outline_used and headings(blocks) == []


def test_use_outline_false_ignores_bookmarks():
    names = [f"Duty number {word}" for word in ("one", "two", "three", "four", "five")]
    pages = pages_of(*[[name, BODY, BODY, "A final sentence ends this paragraph."] for name in names])
    outline = [OutlineEntry(name, 1, index + 1) for index, name in enumerate(names)]
    assert build_blocks(pages, outline)[1].outline_used
    assert not build_blocks(pages, outline, use_outline=False)[1].outline_used


# --------------------------------------------------------------------------------- invariants


def test_every_input_word_survives_structure_detection():
    raw_pages = regulation_pages() + statement_pages()
    laid_out, _ = layout.prepare_pages(["\n".join(lines) for lines in raw_pages])
    blocks, _ = build_blocks(laid_out)
    output = " ".join(b.text for b in blocks)
    for page in laid_out:
        for line in page.lines:
            for word in line.split():
                core = word.strip("(),.;:'\"’“”[]-")
                if len(core) >= 6 and core.isalpha() and core not in ("Whereas", "ADOPTED"):  # "Whereas:" and "HAVE ADOPTED THIS REGULATION:" are renamed
                    assert core.lower() in output.lower() or core.lower().replace("-", "") in output.lower(), core


def test_scheme_levels_are_compressed_to_the_schemes_that_are_used():
    blocks, _ = blocks_of(["Article 1", "First rule", *FILLER, "Article 2", "Second rule", *FILLER, "Article 3", "Third rule", *FILLER])
    assert {b.level for b in blocks if b.kind == HEADING} == {1}


def test_structure_module_constants_are_sane():
    assert structure.MAX_HEADING_DENSITY > 0
    assert structure.MIN_OUTLINE_MATCHES >= 3


# ------------------------------------------------------------------- headings must be set apart (typography)


from reglens.ingest.models import TextLine  # noqa: E402
from reglens.ingest.structure import _body_style  # noqa: E402


def typed_page(*items) -> list[TextLine]:
    """A page of lines; each item is a string (running text, 10 pt) or a ``(text, size, bold)`` tuple."""
    return [TextLine(item, 10.0) if isinstance(item, str) else TextLine(item[0], item[1], item[2]) for item in items]


def table_page(cells_style: tuple[float, bool]) -> list[TextLine]:
    """Running text, then a block of Title Case cells (a table column) that wrap over several lines."""
    cells = ["Information Integrity", "Value Chain and Component", "Integration", "Harmful Bias and Homogenization"]
    return typed_page(*FILLER, *[(cell, *cells_style) for cell in cells], *FILLER)


def test_title_case_table_cells_in_running_type_are_not_headings():
    blocks, report = build_blocks([PageLines(1, table_page((10.0, False))), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks) == []
    assert "Information Integrity" in " ".join(texts(blocks))  # the cells stay in the text


def test_the_same_lines_in_bold_or_larger_type_are_headings():
    for style in ((10.0, True), (12.0, False)):
        blocks, _ = build_blocks([PageLines(1, table_page(style)), PageLines(2, typed_page(*FILLER))])
        assert headings(blocks)[0][1].startswith("Information Integrity"), style


def wrapped(*lines, size=16.0, bold_type=True):
    """A page with running text around a block of lines in one heading type."""
    return typed_page(*FILLER, *[(line, size, bold_type) for line in lines], *FILLER)


def test_a_heading_that_wraps_over_several_lines_in_the_same_type_is_one_heading():
    page = wrapped("Who is", "Covered by the", "Privacy Rule")
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["Who is Covered by the Privacy Rule"]
    assert len(texts(blocks)) == 3  # the running text before and after stays what it was


def test_lines_in_another_size_or_boldness_are_not_part_of_the_heading():
    for second in ((14.0, True), (16.0, False), (10.0, False)):
        page = typed_page(*FILLER, ("Who is", 16.0, True), ("Covered by the", *second), *FILLER)
        blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
        assert headings(blocks)[0][1] == "Who is", second


def test_sizes_that_differ_by_a_rounding_error_are_the_same_type():
    page = typed_page(*FILLER, ("Who is", 16.0, True), ("Covered by the", 16.1, True), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["Who is Covered by the"]


def test_a_blank_line_or_a_page_break_ends_the_heading():
    page = [*typed_page(*FILLER, ("Who is", 16.0, True)), TextLine(""), TextLine("Covered by the", 16.0, True), *typed_page(*FILLER)]
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)][:2] == ["Who is", "Covered by the"]
    first = typed_page(*FILLER, ("Who is", 16.0, True))
    second = typed_page(("Covered by the", 16.0, True), *FILLER)
    blocks, _ = build_blocks([PageLines(1, first), PageLines(2, second)])
    assert [text for _level, text in headings(blocks)][:2] == ["Who is", "Covered by the"]


def test_a_heading_that_ends_like_a_sentence_is_complete():
    blocks, _ = build_blocks([PageLines(1, wrapped("Scope and Aims:", "Another Heading")), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)][:2] == ["Scope and Aims:", "Another Heading"]


def test_a_numbered_or_labelled_line_is_never_swallowed_by_the_heading_above():
    for following in ("2.1 Subheading Of It", "Article 5", "(a) a list item that follows", "III. Roman Heading"):
        blocks, _ = build_blocks([PageLines(1, wrapped("Main Heading", following)), PageLines(2, typed_page(*FILLER))])
        assert headings(blocks)[0][1] == "Main Heading", following


def test_a_wrapped_heading_has_at_most_four_lines():
    blocks, _ = build_blocks([PageLines(1, wrapped(*["Heading One", "Heading Two", "Heading Three", "Heading Four", "Heading Five"])), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)][0] == "Heading One Heading Two Heading Three Heading Four"


def test_a_long_heading_is_not_extended_beyond_the_heading_length_limit():
    long_lines = [
        "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda",
        "Sigma Tau Upsilon Phi Chi Psi Omega Alpha Beta Gamma Delta Epsilon",
        "Zeta Eta Theta Iota Kappa Lambda Sigma Tau Upsilon Phi Chi Psi",
    ]
    blocks, _ = build_blocks([PageLines(1, wrapped(*long_lines, "Another Line")), PageLines(2, typed_page(*FILLER))])
    assert "Another Line" not in headings(blocks)[0][1]
    assert headings(blocks)[0][1].startswith("Alpha Beta") and "Zeta Eta" in headings(blocks)[0][1]


def test_a_line_without_known_typography_is_not_joined_to_a_typeset_heading_and_the_other_way_round():
    for page in (
        [*typed_page(*FILLER, ("Who is", 16.0, True)), TextLine("Covered by the"), *typed_page(*FILLER)],
        [*typed_page(*FILLER), TextLine("Who is"), *typed_page(("Covered by the", 16.0, True), *FILLER)],
    ):
        blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
        assert not any(text.startswith("Who is ") for _level, text in headings(blocks))


def test_a_continuation_must_be_a_short_line_not_a_full_width_sentence_in_heading_type():
    sentence = "Every Word Of This Long Line Is Capitalised And It Runs Almost As Wide As The Running Text Does"
    blocks, _ = build_blocks([PageLines(1, wrapped("Who is", sentence)), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks)[0][1] == "Who is"


OECD_TITLE = "1.2. Respect for the rule of law, human rights and democratic values, including fairness and"


def test_a_numbered_heading_that_wraps_in_bold_type_is_one_heading():
    page = typed_page(*FILLER, (OECD_TITLE, 10.0, True), ("privacy", 10.0, True), "a) AI actors should respect the rule of law.", *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == [f"{OECD_TITLE} privacy"]
    assert texts(blocks)[1].startswith("a) AI actors")  # the list item that follows is body text


def test_a_numbered_heading_that_wraps_in_larger_type_is_one_heading():
    page = typed_page(*FILLER, ("2.3 Shaping an enabling interoperable governance and", 12.0, False), ("policy environment for AI", 12.0, False), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["2.3 Shaping an enabling interoperable governance and policy environment for AI"]


def test_numbered_lines_in_running_type_are_never_joined():
    page = typed_page(*FILLER, (OECD_TITLE, 10.0, False), ("privacy", 10.0, False), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks) == []  # the wrapped sentence goes on in lower case: no heading, as before


def test_a_numbered_heading_is_not_extended_by_lines_that_make_it_too_long():
    page = typed_page(
        *FILLER, ("1.3 Short Title", 10.0, True),
        ("Bold Words That Go On For A Very Long Time Indeed Until The Heading Would Be Silly Long", 10.0, True),
        *FILLER,
    )
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks)[0][1] == "1.3 Short Title"


def test_a_full_width_numbered_line_in_running_type_is_a_paragraph_not_a_heading():
    full = "2.2 Every Word Of This Line Is Capitalised And It Runs As Wide As The Running Text Does In Print Here"
    blocks, _ = blocks_of(["1. Introduction", *FILLER, full, *FILLER])
    assert [text for _level, text in headings(blocks)] == ["1. Introduction"]


def test_a_numbered_heading_takes_in_every_line_of_its_type():
    page = typed_page(*FILLER, ("2.3 Shaping an Enabling", 10.0, True), ("Environment", 10.0, True), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["2.3 Shaping an Enabling Environment"]


def test_a_long_numbered_heading_over_three_lines_is_one_heading():
    lines = [
        "3 Implementing Data Protection Principles In The Processing",
        "Of Personal Data Using Data Protection By Design And",
        "By Default",
    ]
    page = typed_page(*FILLER, *[(line, 10.0, True) for line in lines], *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == [" ".join(lines)]
    assert len(" ".join(lines).split()) > structure.MAX_NUMBERED_WORDS


SHORT_WORDS = "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj"  # ten words, only 29 characters


def test_a_wrapped_numbered_heading_is_not_longer_than_the_word_limit_for_wrapped_headings():
    page = typed_page(*FILLER, ("4 " + SHORT_WORDS, 10.0, True), (SHORT_WORDS, 10.0, True), (SHORT_WORDS, 10.0, True), *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    found = [text for _level, text in headings(blocks)]
    # three lines would be 30 words (over the limit of 25), two lines are 20: the third line is left over
    assert found[:2] == [f"4 {SHORT_WORDS} {SHORT_WORDS}", SHORT_WORDS]
    assert structure.MAX_NUMBERED_WORDS < len(found[0].split()) - 1 <= structure.MAX_WRAPPED_NUMBERED_WORDS


def test_a_one_line_numbered_heading_is_not_longer_than_sixteen_words():
    twenty = " ".join(["Aa Bb Cc Dd Ee"] * 4)
    blocks, _ = blocks_of(["1. Introduction", *FILLER, f"2.2 {twenty}", *FILLER])
    assert [text for _level, text in headings(blocks)] == ["1. Introduction"]


def test_a_numbered_line_that_is_as_wide_as_the_running_text_is_not_a_heading_in_running_type():
    wide = "2.2 Telecommunications Infrastructure Interoperability Standardisation Recommendations Implementation Considerations"
    assert len(wide) > 0.9 * len(BODY) and len(wide.split()) < structure.MAX_NUMBERED_WORDS
    blocks, _ = blocks_of(["1. Introduction", *FILLER, wide, *FILLER])
    assert [text for _level, text in headings(blocks)] == ["1. Introduction"]


def test_a_numbered_line_at_the_very_end_has_nothing_to_head():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "2. Final words"])
    assert [text for _level, text in headings(blocks)] == ["1. Introduction"]


def test_numbered_lines_are_not_joined_when_the_type_of_the_running_text_is_unknown():
    plain = [*FILLER, TextLine(OECD_TITLE, 10.0, True), TextLine("privacy", 10.0, True), *FILLER]  # most lines have no type
    blocks, _ = build_blocks([PageLines(1, plain), PageLines(2, list(FILLER))])
    assert headings(blocks) == []


# ------------------------------------------------------------------------- pieces of a tag or link


@pytest.mark.parametrize("line", ["[NIS", "TAML.016]", "[NISTAML.016", "NISTAML.016] [BACK"])
def test_a_line_with_a_cut_bracket_is_not_a_heading(line):
    for style in ((12.0, True), (10.0, True)):
        page = typed_page(*FILLER, (line, *style), *FILLER)
        blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
        assert headings(blocks) == [], (line, style)
        assert line in " ".join(texts(blocks))  # nothing is lost: the piece stays in the text


def test_a_title_case_line_with_a_cut_bracket_is_not_a_heading_but_a_closed_one_is():
    for line, expected in (("Availability Attacks [NIS", []), ("Availability Attacks [Back To Index]", ["Availability Attacks [Back To Index]"])):
        page = typed_page(*FILLER, (line, 12.0, True), *FILLER)
        blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
        assert [text for _level, text in headings(blocks)] == expected, line


def test_a_numbered_heading_does_not_take_in_the_piece_of_a_tag_below_it():
    page = typed_page(*FILLER, ("2.3.1. Availability Poisoning", 10.0, True), ("[NIS", 10.0, True), "TAML.016] [Back to Index] Attackers can manipulate resources.", *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["2.3.1. Availability Poisoning"]
    assert "[NIS" in texts(blocks)  # the piece stays in the text, as a paragraph of its own


def test_a_bare_article_followed_by_a_list_item_is_not_a_heading():
    blocks, _ = blocks_of(["1. Introduction", *FILLER, "Article 5", "a) the first thing", *FILLER])
    assert [text for _level, text in headings(blocks)] == ["1. Introduction"]


@pytest.mark.parametrize("following", ["B. Alpha Heading Here", "§ 5 Section Title Here", "1798.100. Code Section Title"])
def test_every_kind_of_numbered_heading_starts_a_heading_of_its_own(following):
    blocks, _ = build_blocks([PageLines(1, wrapped("Main Heading", following)), PageLines(2, typed_page(*FILLER))])
    assert headings(blocks)[0][1] == "Main Heading"


def test_without_readable_typography_lines_are_not_joined():
    page = [*FILLER, "Who is", "Covered by the", *FILLER]
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, list(FILLER))])
    assert [text for _level, text in headings(blocks)][:2] == ["Who is", "Covered by the"]


def test_all_caps_lines_must_be_set_apart_too():
    page = typed_page(*FILLER, "TABLE OF RESULTS", *FILLER)
    assert headings(build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])[0]) == []
    page = typed_page(*FILLER, ("TABLE OF RESULTS", 10.0, True), *FILLER)
    assert headings(build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])[0])[0][1] == "TABLE OF RESULTS"


def test_labelled_headings_need_no_typography():
    page = typed_page(*FILLER, "Article 5 – Prohibited practices", *FILLER, "2.1 Scope of the guidelines", *FILLER)
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, typed_page(*FILLER))])
    assert [text for _level, text in headings(blocks)] == ["Article 5 – Prohibited practices", "2.1 Scope of the guidelines"]


def test_when_most_of_the_text_is_bold_bold_does_not_make_a_heading():
    page = [TextLine(text, 10.0, True) for text in [*FILLER, "Information Integrity", *FILLER]]
    blocks, _ = build_blocks([PageLines(1, page), PageLines(2, [TextLine(t, 10.0, True) for t in FILLER])])
    assert headings(blocks) == []


def test_unknown_typography_leaves_the_text_rules_alone():
    plain = [*FILLER, "Information Integrity", *FILLER]
    blocks, _ = build_blocks([PageLines(1, plain), PageLines(2, list(FILLER))])
    assert [text for _level, text in headings(blocks)] == ["Information Integrity"]


@pytest.mark.parametrize("line", ["Information Integrity", "TABLE OF RESULTS"])
def test_title_case_lines_on_a_page_without_readable_fonts_are_not_headings_in_a_typeset_document(line):
    """Some pypdf versions give a few pages no typography. Table cells and figure labels look like headings in the text
    alone, so those pages must not add headings to a document whose other pages show what a heading looks like."""
    readable = typed_page(*FILLER, *FILLER, *FILLER)
    unreadable = [*FILLER, line, "Value Chain and Component", *FILLER]  # plain strings: nothing is known about their type
    blocks, _ = build_blocks([PageLines(1, readable), PageLines(2, unreadable)])
    assert headings(blocks) == []
    assert line in " ".join(texts(blocks))


def test_body_style_is_the_commonest_size_and_needs_most_text_to_be_typeset():
    stream = [structure.Line("a" * 100, 1, size=10.0), structure.Line("b" * 20, 1, size=8.0), structure.Line("c" * 10, 1, size=14.0, bold=True)]
    assert _body_style(stream) == (10.0, False)
    assert _body_style([structure.Line("a" * 100, 1), structure.Line("b" * 10, 1, size=10.0)]) == (None, False)
    assert _body_style([]) == (None, False)


def test_body_style_takes_the_larger_of_two_sizes_that_both_carry_much_of_the_text():
    stream = [structure.Line("a" * 54, 1, size=10.0), structure.Line("b" * 45, 1, size=12.0), structure.Line("c" * 3, 1, size=28.0, bold=True)]
    assert _body_style(stream) == (12.0, False)


def test_numbered_paragraphs_without_a_full_stop_start_a_new_paragraph_even_after_a_full_line():
    full = "x" * 95  # a line that reaches the right margin: no other sign of a paragraph end
    lines = [f"1 Each Party shall adopt measures {full}", f"{full} and report on them.", f"2 Each Party shall also report {full}", "and keep records."]
    blocks, _ = blocks_of(lines + FILLER)
    assert [t[:2] for t in texts(blocks)][:2] == ["1 ", "2 "]


def test_a_line_that_merely_starts_with_a_number_does_not_break_a_paragraph():
    lines = [f"1 Each Party shall adopt measures {'x' * 95}", f"2 000 euros is the amount {'y' * 95}", "and nothing more."]
    blocks, _ = blocks_of(lines + FILLER)
    first = texts(blocks)[0]
    assert "2 000 euros is the amount" in first and first.endswith("and nothing more.")
    assert len(texts(blocks)) == 2  # this paragraph and the filler paragraph after it


def test_a_wrapped_line_that_starts_with_some_other_number_stays_in_its_paragraph():
    lines = [f"1 The Convention applies to {'x' * 75}", f"27 Member States and {'y' * 85}", "to all other Parties."]
    blocks, _ = blocks_of(lines + FILLER)
    first = texts(blocks)[0]
    assert "27 Member States" in first and first.endswith("to all other Parties.")
