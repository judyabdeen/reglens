from collections import Counter

import pytest

from reglens.ingest import layout
from reglens.ingest.layout import (
    LayoutReport,
    body_font_size,
    find_margin_keys,
    furniture_regex,
    margin_key,
    pick_body_size,
    prepare_pages,
    remove_inline_furniture,
    split_lines,
    strip_margins,
)
from reglens.ingest.models import TextLine
from fixtures import official_journal_pages, regulation_pages, statement_pages
from helpers import bold, small, typeset


def raw(pages: list[list[str]]) -> list[str]:
    return ["\n".join(lines) for lines in pages]


def prepare(pages: list[list[str]]):
    """prepare_pages on fixture pages, with the typography of their lines."""
    return prepare_pages(*typeset(pages))


_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()


def body_lines(page: int, count: int = 10) -> list[str]:
    """Lines that differ on every page and every row, even after digits are ignored."""
    return [f"{_WORDS[(page + row) % 26]} {_WORDS[(page * 3 + row * 5) % 26]} {_WORDS[(page * 7 + row) % 26]} text" for row in range(count)]


# ------------------------------------------------------------------------------------- split_lines


def test_split_lines_collapses_spaces_keeps_small_indents_and_trims_blank_edges():
    page = "\n\n   indented  line   with   gaps \nplain\n\n\n"
    assert split_lines(page) == ["   indented line with gaps", "plain"]


def test_split_lines_caps_the_indent_at_eight_spaces():
    assert split_lines(" " * 30 + "deep")[0] == " " * 8 + "deep"


def test_split_lines_repairs_word_export_footnote_markers_across_a_line_break():
    lines = split_lines("detriment.”17F\n18 This test, which consists\nof three elements.")
    assert lines[0].startswith("detriment.” [18]")
    assert "17F" not in " ".join(lines)


def test_split_lines_counts_unreadable_characters():
    stats: Counter = Counter()
    split_lines("a (cid:7) b�", stats)
    assert stats["cid_refs"] == 1
    assert stats["replacement_chars"] == 1


# ------------------------------------------------------------------------------------ margin lines


def test_margin_key_ignores_digits_and_punctuation():
    assert margin_key("L 119/32 EN Official Journal") == margin_key("L 119/33  EN  Official Journal")
    assert margin_key("4.5.2016 EN Official Journal L 119/2") != margin_key("L 119/32 EN Official Journal")


def test_a_header_that_repeats_on_enough_pages_is_found():
    pages = [[f"Running title {n}", *body_lines(n)] for n in range(1, 6)]
    assert find_margin_keys(pages) == {"running title #"}


def test_too_few_pages_or_too_few_repeats_find_nothing():
    two = [["Running title", *body_lines(1)], ["Running title", *body_lines(2)]]
    assert find_margin_keys(two) == set()
    # A line on 3 of 40 pages is not furniture (it needs 10% of the pages).
    many = [["Header", *body_lines(n)] if n < 3 else [_WORDS[n % 26] + f" unique {_WORDS[(n * 3) % 26]} title", *body_lines(n)] for n in range(40)]
    assert "header" not in find_margin_keys(many)


def test_structural_and_long_lines_are_never_margin_lines():
    pages = [["Article 5", "x" * 130, *body_lines(n)] for n in range(6)]
    assert find_margin_keys(pages) == set()


def test_strip_margins_removes_headers_page_numbers_and_toc_lines_and_counts_them():
    report = LayoutReport()
    keys = {"running title #"}
    lines = [
        "Running title 7",
        "Contents ........ 12",
        "A sentence in the body.",
        "The year 2016 is in the middle",
        "42",
        "Closing sentence.",
        "7",
    ]
    kept = strip_margins(lines, keys, report)
    assert kept == ["A sentence in the body.", "The year 2016 is in the middle", "42", "Closing sentence."]
    assert report.margin_lines_removed == Counter({"running title #": 1})
    assert report.margin_examples == {"running title #": "Running title 7"}
    assert report.page_numbers_removed == 1  # only the last line "7"; a number in the middle is body text
    assert report.toc_lines_removed == 1


def test_a_bare_number_in_the_middle_of_a_page_is_kept():
    kept = strip_margins(["first", "second", "2016", "third", "fourth"], set(), LayoutReport())
    assert "2016" in kept


# --------------------------------------------------------------------------------------- footnotes


def test_footnotes_of_the_official_journal_fixture_are_separated_in_order():
    pages, report = prepare(regulation_pages())
    numbers = [note.number for page in pages for note in page.notes]
    assert numbers == list(range(1, 13))
    assert report.footnotes == 12 and report.footnote_pages == 8
    all_body = "\n".join("\n".join(page.lines) for page in pages)
    assert "Position of the European Parliament" not in all_body
    assert "OJ C 229" not in all_body


def test_running_headers_of_both_page_parities_are_removed():
    pages, report = prepare(regulation_pages())
    joined = "\n".join("\n".join(page.lines) for page in pages)
    assert "L 119/" not in joined and "4.5.2099" not in joined
    assert "Official Journal of the European Union." in joined  # the same words inside a sentence stay
    assert sum(report.margin_lines_removed.values()) == 10


def test_a_list_of_definitions_is_not_mistaken_for_footnotes():
    pages, _ = prepare(regulation_pages())
    page3 = pages[2]
    assert any(line.startswith("(1) 'personal data'") for line in page3.lines)
    assert any(line.startswith("(2) 'processing'") for line in page3.lines)
    assert [note.number for note in page3.notes] == [5, 6]


def test_recital_numbers_on_their_own_line_are_not_footnotes():
    pages, _ = prepare(regulation_pages())
    assert "(1)" in pages[0].lines and "(2)" in pages[0].lines
    assert [note.number for note in pages[0].notes] == [1, 2]


def test_a_footnote_that_continues_on_the_next_page_is_completed():
    pages, _ = prepare(regulation_pages())
    note_10 = pages[6].notes[0]
    assert note_10.number == 10
    assert note_10.lines[0].endswith("on the")
    assert note_10.lines[-1] == "protection of natural persons, as amended."
    assert [n.number for n in pages[7].notes] == [11]  # the continuation line did not become a note


def test_word_export_statement_footnotes_and_continuation():
    pages, report = prepare(statement_pages())
    assert [n.number for page in pages for n in page.notes] == [1, 2, 3, 4, 5, 6]
    note_5 = pages[2].notes[1]
    assert note_5.lines[-1] == "continued text of footnote 5 that started on the page before."
    assert "continued text" not in "\n".join(pages[3].lines)
    # the page numbers "1".."4" at the top of each page are page furniture
    assert all(not page.lines[0].strip().isdigit() for page in pages)
    assert sum(report.margin_lines_removed.values()) + report.page_numbers_removed == 4


def test_without_type_sizes_nothing_is_a_footnote():
    """Numbered lines at the foot of a page look the same as footnotes; only their type tells them apart."""
    pages, report = prepare_pages(raw(regulation_pages()))
    assert report.footnotes == 0 and report.body_size is None
    assert any("OJ C 229" in line for line in pages[0].lines)  # the note text stays in the body, in order


def test_numbered_lines_in_running_text_are_not_footnotes_however_they_are_numbered():
    """Treaty paragraphs "1 ... 2 ..." and statute list items "(1) ... (2) ..." at the end of pages keep their place."""
    pages = []
    for page in range(1, 9):
        body = body_lines(page, 6)
        items = [f"({row}) {body_lines(page + 3, 6)[row - 1]}" for row in range(1, 4)]
        paragraphs = [f"{row} {body_lines(page + 5, 6)[row - 1]}" for row in range(1, 3)]
        pages.append([*body, *(items if page % 2 else paragraphs)])
    laid_out, report = prepare(pages)
    assert report.footnotes == 0 and report.footnote_pages == 0
    assert sum(len(page.lines) for page in laid_out) == sum(len(lines) for lines in pages)  # nothing moved or lost


def test_small_print_without_a_note_number_stays_in_the_body():
    pages = [[*body_lines(n, 8), small(f"Source: the authority's {_WORDS[(n * 3) % 26]} statistics for the year")] for n in range(1, 7)]
    laid_out, report = prepare(pages)
    assert report.footnotes == 0
    assert all(any("statistics for the year" in line for line in page.lines) for page in laid_out)


def test_a_page_set_entirely_in_small_print_is_not_a_footnote_zone():
    pages = [body_lines(n, 8) for n in range(1, 6)]
    pages.append([small(f"{row} {line}") for row, line in enumerate(body_lines(9, 8), start=1)])
    laid_out, report = prepare(pages)
    assert report.footnotes == 0


def test_a_glued_note_number_is_recognised_in_small_print():
    pages = []
    for page in range(1, 7):
        pages.append([*body_lines(page, 8), "", small(f"{page}Regulation (EU) 2016/{600 + page} of the {_WORDS[(page * 3) % 26]} Parliament.")])
    laid_out, report = prepare(pages)
    assert report.footnotes == 6
    assert laid_out[2].notes[0].lines == [f"Regulation (EU) 2016/603 of the {_WORDS[9]} Parliament."]


def test_a_note_continuation_is_ignored_when_the_previous_page_had_no_notes():
    pages = [body_lines(n, 8) for n in range(1, 4)]
    pages.append([*body_lines(4, 8), small("lowercase tail of a note nobody started"), small("7 A real note.")])
    pages += [[*body_lines(n, 8), small(f"{n + 4} A note about {_WORDS[(n * 3) % 26]} and {_WORDS[(n * 5) % 26]}.")] for n in range(5, 9)]
    laid_out, _ = prepare(pages)
    assert any("lowercase tail" in line for line in laid_out[3].lines)  # stays in the body
    assert [n.number for n in laid_out[3].notes] == [7]


def test_small_print_above_the_notes_stays_in_the_body_when_the_previous_note_was_finished():
    pages = [
        [*body_lines(n, 8), small(f"Source: the {_WORDS[n]} office statistics"), small(f"{n} A finished note about {_WORDS[(n * 3) % 26]} and {_WORDS[(n * 5) % 26]}.")]
        for n in range(1, 7)
    ]
    laid_out, report = prepare(pages)
    assert report.footnotes == 6
    assert all(any(line.startswith("Source: the") for line in page.lines) for page in laid_out)
    assert not any("Source" in " ".join(note.lines) for page in laid_out for note in page.notes)


def test_a_note_does_not_continue_over_a_page_without_notes():
    def note_page(n: int, number: int):
        return [*body_lines(n, 8), small(f"{number} A note about {_WORDS[(n * 3) % 26]} and {_WORDS[(n * 5) % 26]}.")]

    pages = [note_page(n, n) for n in range(1, 4)]
    pages.append(body_lines(4, 8))
    pages.append([*body_lines(5, 8), small("lowercase tail of nobody's note"), small("4 A note about victor and whiskey.")])
    pages += [note_page(n, n - 1) for n in range(6, 8)]
    laid_out, _ = prepare(pages)
    assert any("lowercase tail" in line for line in laid_out[4].lines)  # stays in the body of its page
    assert not any("lowercase tail" in " ".join(note.lines) for page in laid_out for note in page.notes)


def test_a_wrapped_note_line_that_starts_with_a_number_stays_in_its_note():
    pages = [
        [*body_lines(n, 8), small(f"{n} Regulation of the {_WORDS[(n * 3) % 26]} Parliament and of the Council of"), small(f"27 April 2016 on the protection of {_WORDS[(n * 5) % 26]} persons.")]
        for n in range(1, 7)
    ]
    laid_out, _ = prepare(pages)
    assert [len(page.notes) for page in laid_out] == [1] * 6
    assert laid_out[0].notes[0].lines[-1].startswith("27 April 2016")


@pytest.mark.parametrize(
    "body, size, expected",
    [
        (12.0, 11.0, True),
        (12.0, 8.0, True),
        (12.0, 11.3, False),  # more than 93% of the body size
        (12.0, 12.0, False),
        (6.0, 5.5, False),  # a smaller ratio, but under 0.6 pt smaller: rounding noise
        (12.0, None, False),  # unknown
    ],
)
def test_small_print_is_clearly_smaller_than_the_running_text(body, size, expected):
    assert layout._is_small(TextLine("1 A note.", size), body) is expected


@pytest.mark.parametrize(
    "spans, pages_total, expected",
    [
        ([], 10, False),
        ([(1, 3)], 3, True),  # a short document may have notes on one page
        ([(1, 3)], 40, False),  # a long one does not
        ([(5, 5)], 3, False),  # a lone note that does not start at 1
        ([(5, 9)], 3, True),  # but a run of notes may
        ([(1, 2), (3, 4), (5, 6)], 20, True),  # the numbers continue
        ([(1, 2), (3, 4), (1, 2)], 20, True),  # or start again at 1 on each page
        ([(1, 2), (40, 41), (7, 8)], 20, False),
        ([(1, 2), (10, 12), (3, 4)], 20, False),
    ],
)
def test_notes_look_real_only_when_their_numbers_continue(spans, pages_total, expected):
    assert layout._notes_look_real(spans, pages_total) is expected


def _note(*lines):
    return layout.RawNote(1, list(lines), 1)


@pytest.mark.parametrize(
    "previous, line, expected",
    [
        (None, "lowercase tail", False),
        (_note(), "lowercase tail", False),
        (_note("See the opinion of the Board on"), "Data protection.", True),  # the note is unfinished
        (_note("See the opinion."), "and the annexes thereto.", True),  # a lower-case start continues a sentence
        (_note("See the opinion."), "Source: the authority", False),  # finished note, new sentence: not a continuation
    ],
)
def test_small_print_without_a_number_continues_the_previous_note_only_if_it_reads_like_a_continuation(previous, line, expected):
    assert layout._continues_previous(previous, TextLine(line, 8.0)) is expected


def test_a_repeated_short_footnote_is_not_page_furniture():
    pages = []
    for page in range(1, 13):
        first = 3 * page - 2  # footnote numbers grow three times faster than the page numbers
        pages.append([*body_lines(page, 12), "", small(f"{first} See the source for {_WORDS[page]}."), small(f"{first + 1} Id."), small(f"{first + 2} Id.")])
    laid_out, report = prepare(pages)
    assert report.footnotes == 36
    assert all(page.notes[-1].lines == ["Id."] for page in laid_out)


def test_a_header_with_a_page_counter_is_still_furniture():
    pages = [[f"Guidelines 05/2020 - page {n} of 80", *body_lines(n)] for n in range(10, 22)]
    laid_out, report = prepare_pages(raw(pages))
    assert not any("Guidelines 05/2020" in line for page in laid_out for line in page.lines)
    assert sum(report.margin_lines_removed.values()) == 12


# ------------------------------------------------------------------------- headers inside the page text


def test_a_header_stored_inside_the_page_text_is_cut_out_wherever_it_is():
    pages, report = prepare(official_journal_pages())
    joined = "\n".join("\n".join(page.lines) for page in pages)
    assert "Official Journal" not in joined and "L 119/" not in joined and "4.5.2099" not in joined
    assert "and more words" in joined  # the text around the header stays
    assert sum(report.margin_lines_removed.values()) == 12
    assert [note.number for page in pages for note in page.notes] == [1] * 12


def test_text_next_to_a_cut_out_header_is_joined_without_gaps():
    pages, _ = prepare(official_journal_pages())
    line = next(line for line in pages[3].lines if "and more words" in line)
    assert "  " not in line and line.endswith("and more words")


def test_furniture_regex_needs_a_number_and_enough_words():
    assert furniture_regex("Page 3 of 80") is None  # too short and generic
    assert furniture_regex("Guidelines on consent under the Regulation") is None  # no number: could be a phrase of the text
    pattern = furniture_regex("4.5.2016 L 119/2 Official Journal of the European Union EN")
    assert pattern.search("end of a sentence 12.12.1999   L119/31 Official Journal of the European Union EN")
    assert not pattern.search("published in the Official Journal of the European Union")
    assert not pattern.search("v4.5.2016 L 119/2 Official Journal of the European Union EN")  # not inside a longer word


def test_cutting_out_a_header_keeps_the_typography_of_the_line():
    line = TextLine("some text 4.5.2016 L 119/2 Official Journal of the European Union EN", 9.5, True)
    pattern = furniture_regex("4.5.2016 L 119/2 Official Journal of the European Union EN")
    (kept,) = remove_inline_furniture([line], [("key", pattern)], LayoutReport())
    assert kept == "some text" and kept.size == 9.5 and kept.bold is True


# ---------------------------------------------------------------------------------- body size and styles


def test_body_font_size_is_the_size_with_most_characters():
    pages = [[TextLine("a" * 100, 10.0), TextLine("b" * 30, 8.0), TextLine("c" * 12, 14.0)]]
    assert body_font_size(pages) == 10.0


@pytest.mark.parametrize(
    "characters_by_size, expected",
    [
        ({10.0: 80, 8.0: 15, 14.0: 5}, 10.0),  # an ordinary document: the commonest size
        ({10.0: 54, 12.0: 45, 8.0: 1}, 12.0),  # a short statement whose long footnotes outweigh the text above them
        ({10.0: 51, 11.0: 47, 28.0: 2}, 11.0),  # prose in 11 pt next to a lot of table text in 10 pt
        ({10.0: 60, 16.0: 40}, 10.0),  # a size more than half as large again is a heading or a poster, not the text
        ({10.0: 60, 15.0: 40}, 15.0),  # exactly half as large again still counts
        ({10.0: 50, 12.0: 50}, 12.0),  # a tie goes to the larger size
        ({10.0: 21, 11.0: 20, 12.0: 20, 13.0: 20, 14.0: 19}, 10.0),  # no size is a real share: the commonest
        ({10.0: 70, 12.0: 29, 14.0: 1}, 10.0),  # 29 % is not a real share
        ({10.0: 70, 12.0: 30}, 12.0),  # 30 % is
        ({12.0: 70, 10.0: 30}, 12.0),  # the commonest is also the larger
        ({9.5: 100}, 9.5),
    ],
)
def test_pick_body_size(characters_by_size, expected):
    assert pick_body_size(Counter(characters_by_size)) == expected


def test_body_font_size_uses_the_same_rule():
    pages = [[TextLine("a" * 54, 10.0), TextLine("b" * 45, 12.0), TextLine("c" * 3, 8.0)]]
    assert body_font_size(pages) == 12.0


def test_body_font_size_is_unknown_without_sizes_for_most_of_the_text():
    assert body_font_size([["plain " * 20, TextLine("x" * 20, 10.0)]]) is None
    assert body_font_size([["plain text"]]) is None


def test_split_lines_attaches_the_typography_when_it_matches_the_lines():
    lines = split_lines("one\n\ntwo", styles=[(10.0, False), None, (8.5, True)])
    assert lines == ["one", "", "two"]
    assert (lines[0].size, lines[0].bold, lines[2].size, lines[2].bold) == (10.0, False, 8.5, True)
    assert not hasattr(lines[1], "size")


def test_split_lines_drops_typography_that_does_not_match_the_lines():
    lines = split_lines("one\ntwo\nthree", styles=[(10.0, False)])
    assert lines == ["one", "two", "three"] and not hasattr(lines[0], "size")


def test_typography_survives_the_repair_of_a_marker_that_spans_a_line_break():
    page = "detriment.”17F\n18 This test, which consists\nof three elements."
    lines = split_lines(page, styles=[(10.0, False), (10.0, False), (10.0, False)])
    assert len(lines) == 3 and all(line.size == 10.0 for line in lines)
    assert lines[0].endswith("[18]")


@pytest.mark.parametrize(
    "seen, expected",
    [
        ([(0, ()), (3, ())], True),  # no numbers at all
        ([(0, (1, 80)), (1, (2, 80)), (2, (3, 80))], True),  # a page counter and a constant
        ([(5, (6,)), (7, (8,)), (9, (10,))], True),  # every other page: the offset stays the same
        ([(0, (7,)), (1, (7,)), (2, (7,))], True),  # constant
        ([(0, (1,)), (1, (4,)), (2, (7,))], False),  # grows faster than the pages
        ([(0, (1,)), (1, (2, 3))], False),  # a different count of numbers
    ],
)
def test_numbers_move_with_the_page(seen, expected):
    assert layout._numbers_move_with_the_page(seen) is expected


def test_numbering_that_does_not_continue_from_page_to_page_is_not_footnotes():
    pages = []
    for n, first in enumerate([7, 91, 3, 55, 20], start=1):
        pages.append([f"Alpha {n}", "Beta body text line", "Gamma body text line", "Delta body text line", "", small(f"{first} note"), small(f"{first + 1} note")])
    _, report = prepare(pages)
    assert report.footnotes == 0


def test_pages_with_hardly_any_text_are_counted_as_empty():
    pages = [["Some real text " * 10, "more " * 20], ["", "  "], ["x"]]
    _, report = prepare_pages(raw(pages))
    assert report.empty_pages == 2 and report.empty_page_numbers == [2, 3]
    assert report.pages == 3


def test_prepare_pages_never_loses_body_text_other_than_furniture():
    original = regulation_pages()
    pages, _ = prepare(original)
    body_words = " ".join(" ".join(page.lines) for page in pages).split()
    for needle in ("fundamental", "Subject-matter", "twentieth", "supervisory", "Annex" if False else "ANNEX"):
        assert needle in body_words or any(needle in word for word in body_words), needle


def test_module_constants_have_sane_values():
    assert layout.MARGIN_MIN_PAGES >= 2
    assert 0 < layout.MAX_NOTE_FRACTION <= 1
