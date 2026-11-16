"""The PDF reader: text, typography of every line, and raised digits (footnote marks and exponents)."""

import io

import pytest

from reglens.ingest import pdfread
from reglens.ingest.pdfread import ExtractionError, read_page, read_pdf
from helpers import make_multipage_pdf, make_pdf_from_stream, make_pdf_with_form, small, bold


def page_of(commands: list[str]):
    from pypdf import PdfReader

    return PdfReader(io.BytesIO(make_pdf_from_stream(commands))).pages[0]


def lines_and_styles(commands: list[str]):
    text, styles = read_page(page_of(commands))
    assert styles is not None
    return text.split("\n"), styles


def run(text: str, size: float = 11) -> str:
    """Running text in the page's main size, continuing on the same line."""
    return f"/F1 {size} Tf ({text}) Tj"


def raised(text: str, size: float = 7, rise: float = 4) -> str:
    """Text raised above the line by the PDF's text-rise operator, in a smaller size (a footnote mark, an exponent)."""
    return f"/F1 {size} Tf {rise} Ts ({text}) Tj 0 Ts"


def line(*pieces: str, y: int = 700) -> str:
    return f"BT 1 0 0 1 40 {y} Tm " + " ".join(pieces) + " ET"


# ---------------------------------------------------------------------------------------- typography


def test_every_line_gets_its_dominant_size_and_boldness():
    lines, styles = lines_and_styles([
        "BT /F2 14 Tf 1 0 0 1 40 700 Tm (A Bold Heading) Tj ET",
        "BT /F1 10 Tf 1 0 0 1 40 680 Tm (Running text of the page.) Tj ET",
        "BT /F1 8 Tf 1 0 0 1 40 100 Tm (1 A footnote.) Tj ET",
    ])
    assert lines == ["A Bold Heading", "Running text of the page.", "1 A footnote."]
    assert styles == [(14.0, True), (10.0, False), (8.0, False)]


def test_the_size_comes_from_the_text_matrix_and_the_page_matrix_not_only_from_the_font_size():
    _, styles = lines_and_styles([
        "BT /F1 1 Tf 10 0 0 10 40 700 Tm (Scaled by the text matrix) Tj ET",
        "q 0.5 0 0 0.5 0 0 cm BT /F1 20 Tf 1 0 0 1 80 1300 Tm (Scaled by the page matrix) Tj ET Q",
    ])
    assert styles == [(10.0, False), (10.0, False)]


def test_a_line_takes_the_size_of_most_of_its_characters():
    _, styles = lines_and_styles([
        "BT /F1 10 Tf 1 0 0 1 40 700 Tm (Mostly running text with ) Tj /F1 6 Tf (tiny) Tj ET",
    ])
    assert styles == [(10.0, False)]


def test_the_dominant_size_is_decided_by_characters_not_by_the_number_of_pieces():
    _, styles = lines_and_styles([line(run("A long stretch of running text in the main size", 11), run(" a", 8), run(" b", 8), run(" c", 8), run(" d", 8))])
    assert styles == [(11.0, False)]


def test_empty_lines_have_no_style():
    text, styles = read_page(page_of(["BT /F1 10 Tf 1 0 0 1 40 700 Tm () Tj ET", "BT /F1 10 Tf 1 0 0 1 40 680 Tm (x) Tj ET"]))
    assert len(text.split("\n")) == len(styles)


# ------------------------------------------------------------------------------------- raised digits


def test_an_exponent_that_plain_extraction_flattens_is_written_with_a_caret():
    """10^25 floating point operations is the threshold of the AI Act; "1025" would be a different number."""
    lines, _ = lines_and_styles([line(run("is greater than 10"), raised("25"), run(" floating point operations."))])
    assert lines == ["is greater than 10^25 floating point operations."]


def test_plain_extraction_really_does_flatten_the_exponent():
    """Documents why the reader is needed at all: pypdf alone returns 1025."""
    page = page_of([line(run("is greater than 10"), raised("25"), run(" floating point operations."))])
    assert "1025 floating" in page.extract_text()


def test_an_exponent_drawn_digit_by_digit_is_one_number():
    """The EC guidelines draw 10^24 as "10", "2", "4": the second digit must not become a footnote mark."""
    lines, _ = lines_and_styles([line(run("using 10"), raised("2"), raised("4"), run(" FLOP."))])
    assert lines == ["using 10^24 FLOP."]


def test_the_space_that_follows_a_raised_number_is_kept():
    lines, _ = lines_and_styles([line(run("greater than 10"), raised("24 "), run("FLOP"))])
    assert lines == ["greater than 10^24 FLOP"]


def test_a_footnote_mark_drawn_digit_by_digit_is_one_mark():
    lines, _ = lines_and_styles([line(run("as set out in the opinion"), raised("1"), raised("2"), run(". Next"))])
    assert lines == ["as set out in the opinion [12]. Next"]


def piece(text, size=11.0, baseline=700.0):
    return (text, size, False, baseline)


def compose(*pieces):
    return pdfread._compose_line(list(pieces))[0]


def test_raised_numbers_that_are_not_directly_adjacent_are_separate_marks():
    assert compose(piece("text"), piece("2", 7, 704), piece(" 4", 7, 704)) == "text [2] [4]"
    assert compose(piece("text"), piece("2 ", 7, 704), piece("4", 7, 704)) == "text [2] [4]"
    assert compose(piece("text"), piece("2", 7, 704), piece(" ", 7, 704), piece("4", 7, 704)) == "text [2] [4]"
    assert compose(piece("text"), piece("1,2", 7, 704), piece("3", 7, 704)) == "text [1,2] [3]"
    assert compose(piece("text"), piece("2", 7, 704), piece("4,5", 7, 704)) == "text [2] [4,5]"


def test_a_raised_number_at_the_end_of_the_line_is_written():
    assert compose(piece("text"), piece("2", 7, 704), piece("4", 7, 704)) == "text [24]"
    assert compose(piece("greater than 10"), piece("2", 7, 704), piece("5", 7, 704)) == "greater than 10^25"


def test_a_raised_number_after_the_space_pypdf_inserted_is_still_directly_after_the_text():
    assert compose(piece("greater than 10 "), piece("25", 7, 704), piece(" FLOP")) == "greater than 10^25 FLOP"


def test_a_footnote_mark_after_a_number_is_kept_apart_from_it():
    lines, _ = lines_and_styles([line(run("see WP 218"), raised("6"), run("; and more"))])
    assert lines == ["see WP 218 [6]; and more"]


def test_a_mark_after_article_10_is_a_footnote_not_an_exponent():
    lines, _ = lines_and_styles([line(run("as set out in Article 10"), raised("3"), run(" of the Regulation"))])
    assert lines == ["as set out in Article 10 [3] of the Regulation"]


def test_a_mark_after_a_word_and_a_mark_with_several_numbers():
    lines, _ = lines_and_styles([line(run("international standards"), raised("9"), run(". See also the opinion"), raised("1,2"), run("."))])
    assert lines == ["international standards [9]. See also the opinion [1,2]."]


def test_a_smaller_digit_between_brackets_is_part_of_the_numbering_not_a_footnote_mark():
    """Official Journal texts set the "1" of "(1)" smaller than the brackets, both in the notes and in the text."""
    lines, _ = lines_and_styles([
        line(run("small and medium-sized enterprises should draw from Commission Recommendation 2003/361/EC \\("), run("1", 7), run("\\)."), y=700),
        line(run("\\(", 8), run("1", 6), run("\\) Commission Recommendation of 6 May 2003 concerning the definition of micro enterprises", 8), y=100),
    ])
    assert lines == [
        "small and medium-sized enterprises should draw from Commission Recommendation 2003/361/EC (1).",
        "(1) Commission Recommendation of 6 May 2003 concerning the definition of micro enterprises",
    ]


def test_a_mark_after_a_closing_bracket_is_still_a_mark():
    lines, _ = lines_and_styles([line(run("as defined in the Directive (EU) 2016/680\\)"), raised("4"), run(" and"))])
    assert lines == ["as defined in the Directive (EU) 2016/680) [4] and"]


def test_a_note_number_at_the_start_of_a_line_is_not_turned_into_a_mark():
    lines, _ = lines_and_styles([
        line(run("1", 8), run(" Regulation (EU) 2016/679.", 8), y=100),
        line(raised("2", 6, 0), run(" Regulation 2016/680.", 8), y=90),
    ])
    assert [" ".join(text.split()) for text in lines] == ["1 Regulation (EU) 2016/679.", "2 Regulation 2016/680."]


def test_digits_that_are_not_smaller_than_their_neighbours_stay_digits():
    lines, _ = lines_and_styles([line(run("version 10"), run("25"))])
    assert lines == ["version 1025"]


def test_text_that_is_not_just_digits_is_never_a_mark():
    lines, _ = lines_and_styles([line(run("the model"), raised("v2"), run(" and"), raised("abc"))])
    assert lines == ["the modelv2 andabc"]


def test_a_lowered_digit_is_not_a_footnote_mark():
    """A digit that sits below the baseline (CO2) is a subscript. Needs positions: the PDF moves the text matrix."""
    lines, _ = lines_and_styles([
        line(run("CO"), "/F1 7 Tf 1 0 0 1 52 696 Tm (2) Tj", "/F1 11 Tf 1 0 0 1 58 700 Tm ( emissions) Tj")
    ])
    assert "[2]" not in lines[0] and lines[0].replace(" ", "") == "CO2emissions"


@pytest.mark.parametrize(
    "base_font, expected",
    [
        ("/Arial-BoldMT", True),
        ("/ABCDEF+Calibri,Bold", True),
        ("/Frutiger-Black", True),
        ("/Gotham-Heavy", True),
        ("/Source-Sans-DemiBold", True),
        ("/Arial-ItalicMT", False),
        ("/TimesNewRomanPSMT", False),
        ("", False),
    ],
)
def test_bold_is_read_from_the_name_of_the_font(base_font, expected):
    assert pdfread._is_bold({"/BaseFont": base_font}) is expected


def test_a_font_without_a_readable_name_is_not_bold():
    assert pdfread._is_bold(None) is False and pdfread._is_bold({}) is False


def test_a_line_is_bold_when_at_least_half_of_its_characters_are():
    _, styles = lines_and_styles([
        "BT /F2 11 Tf 1 0 0 1 40 700 Tm (Bold words) Tj /F1 11 Tf (plain) Tj ET",
        "BT /F2 11 Tf 1 0 0 1 40 680 Tm (Bold) Tj /F1 11 Tf ( and a long plain tail) Tj ET",
    ])
    assert [bold_flag for _size, bold_flag in styles] == [True, False]


# ------------------------------------------------------------------------- text drawn at another scale

# An HHS-style line: the text matrix is rescaled for the raised digit and back again, inside one run of text.
NOTE_AT_ANOTHER_SCALE = (
    "BT /F1 1 Tf 11.04 0 0 11.04 100 700 Tm [(was published December 28, 2000.)] TJ "
    "6.96 0 0 6.96 480 705 Tm (2) Tj 11.04 0 0 11.04 485 700 Tm (    ) Tj ET"
)


def squeezed(text: str) -> str:
    return " ".join(text.split())


def separation(commands: list[str]) -> bool:
    page = page_of(commands)
    original, separated = pdfread._separate_scale_changes(page)
    assert original is not None
    return separated


def test_a_footnote_mark_drawn_by_rescaling_the_text_matrix_is_found():
    lines, styles = lines_and_styles([NOTE_AT_ANOTHER_SCALE])
    assert squeezed(lines[0]) == "was published December 28, 2000. [2]"
    assert styles == [(11.0, False)]


def test_an_exponent_drawn_by_rescaling_the_text_matrix_is_written_with_a_caret():
    lines, _ = lines_and_styles([
        "BT /F1 1 Tf 11 0 0 11 100 700 Tm (more than 10) Tj 7 0 0 7 160 705 Tm (25) Tj 11 0 0 11 175 700 Tm ( FLOPs) Tj ET",
    ])
    assert squeezed(lines[0]) == "more than 10^25 FLOPs"


def test_the_mark_is_set_directly_after_the_word_whatever_space_pypdf_put_between():
    lines, _ = lines_and_styles([NOTE_AT_ANOTHER_SCALE])
    assert "2000. [2]" in lines[0] and "2000.  [2]" not in lines[0]


def test_a_change_of_scale_is_separated_and_text_at_one_scale_is_not():
    assert separation([NOTE_AT_ANOTHER_SCALE]) is True
    assert separation(["BT /F1 1 Tf 11 0 0 11 100 700 Tm (Executive Summary) Tj 11 0 0 11 400 700 Tm (1) Tj ET"]) is False


@pytest.mark.parametrize("second_scale, expected", [(11.2, False), (10.8, False), (12.0, True), (9.0, True)])
def test_scales_within_three_percent_are_the_same_scale(second_scale, expected):
    assert separation([f"BT /F1 1 Tf 11 0 0 11 100 700 Tm (First) Tj {second_scale} 0 0 {second_scale} 300 700 Tm (Second) Tj ET"]) is expected


def test_the_scale_is_the_vertical_one_which_is_what_sets_the_size_of_the_letters():
    assert separation(["BT /F1 1 Tf 11 0 0 11 100 700 Tm (First) Tj 11 0 0 7 300 700 Tm (Squeezed) Tj ET"]) is True
    assert separation(["BT /F1 1 Tf 11 0 0 11 100 700 Tm (First) Tj 7 0 0 11 300 700 Tm (Narrow) Tj ET"]) is False


def test_the_tolerance_is_measured_against_the_larger_scale():
    both = lambda second: separation([f"BT /F1 1 Tf 100 0 0 100 100 700 Tm (First) Tj {second} 0 0 {second} 300 700 Tm (Second) Tj ET"])
    assert both(97.05) is False and both(96.5) is True and both(103.0) is False and both(103.5) is True


def test_a_new_text_object_starts_again_at_the_unit_scale():
    commands = ["BT /F1 1 Tf 11 0 0 11 100 700 Tm (First) Tj ET", "BT 11 0 0 11 100 600 Tm (Second) Tj ET"]
    assert separation(commands) is False  # the second object has the same scale as the first
    assert separation(["BT /F1 11 Tf 100 700 Td (First) Tj ET", "BT 100 650 Td (Second) Tj ET"]) is False  # both at the unit scale
    assert separation(["BT /F1 11 Tf 11 0 0 11 100 700 Tm (First) Tj ET", "BT 100 650 Td (Second) Tj ET"]) is True


def test_a_change_of_font_or_size_needs_no_separation_because_pypdf_hands_over_its_text_there_anyway():
    assert separation(["BT /F1 11 Tf 100 700 Td (First) Tj /F1 7 Tf 6 Ts (2) Tj ET"]) is False
    assert separation(["BT /F1 11 Tf 1 0 0 1 100 700 Tm (First) Tj /F2 11 Tf 7 0 0 7 300 700 Tm (Second) Tj ET"]) is False


def test_the_font_that_applies_after_a_restored_graphics_state_is_known():
    commands = [
        "BT /F1 1 Tf 11 0 0 11 100 700 Tm (First) Tj ET",
        "q /F1 20 Tf Q",  # a change of size that is undone before any text is shown with it
        "BT 7 0 0 7 100 600 Tm (Third) Tj ET",  # still /F1 1, now at scale 7 against the 11 of the first text
    ]
    assert separation(commands) is True


def test_reading_leaves_the_page_as_it_was():
    page = page_of([NOTE_AT_ANOTHER_SCALE])
    contents = page.get("/Contents")
    assert read_page(page)[1] is not None
    assert page.get("/Contents") is contents


def test_a_page_whose_text_would_change_by_the_separation_is_read_as_it_is(monkeypatch):
    page = page_of([NOTE_AT_ANOTHER_SCALE])
    as_it_is = squeezed(page.extract_text())
    real = page.extract_text
    contents = page.get("/Contents")

    def different_when_separated(*args, **kwargs):
        text = real(*args, **kwargs)
        return text if page.get("/Contents") is contents else text + " (changed by the separation)"

    monkeypatch.setattr(page, "extract_text", different_when_separated)
    text, styles = read_page(page)
    assert squeezed(text) == as_it_is and "[2]" not in text and styles == [(11.0, False)]


def test_a_page_that_cannot_be_parsed_for_the_separation_is_read_as_it_is(monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("cannot parse")

    monkeypatch.setattr("pypdf.generic.ContentStream", refuse)
    text, styles = read_page(page_of([NOTE_AT_ANOTHER_SCALE]))
    assert squeezed(text) == "was published December 28, 2000. 2" and styles == [(11.0, False)]


def test_a_page_without_content_is_empty():
    from pypdf import PdfReader

    page = PdfReader(io.BytesIO(make_pdf_from_stream([""]))).pages[0]
    del page["/Contents"]
    assert read_page(page) == ("", None)


def test_a_page_that_fails_only_with_its_parsed_contents_is_read_as_it_is(monkeypatch):
    page = page_of(["BT /F1 12 Tf 1 0 0 1 100 700 Tm (A line of text at one scale.) Tj ET"])
    real = page.extract_text
    contents = page.get("/Contents")

    def failing_with_parsed_contents(*args, **kwargs):
        if page.get("/Contents") is not contents:
            raise RuntimeError("the parsed page broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(page, "extract_text", failing_with_parsed_contents)
    text, styles = read_page(page)
    assert text == "A line of text at one scale." and styles == [(12.0, False)]


def test_the_separated_reading_is_given_up_when_only_it_fails(monkeypatch):
    page = page_of([NOTE_AT_ANOTHER_SCALE])
    real = page.extract_text
    contents = page.get("/Contents")

    def failing_when_separated(*args, **kwargs):
        if page.get("/Contents") is not contents:
            raise RuntimeError("the separated page broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(page, "extract_text", failing_when_separated)
    text, styles = read_page(page)
    assert squeezed(text) == "was published December 28, 2000. 2" and styles == [(11.0, False)]


# ------------------------------------------------------------------ sizes checked against the operators

UNIT = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
SCALED = [11.0, 0.0, 0.0, 11.0, 100.0, 700.0]


def test_a_piece_that_arrives_without_its_matrices_takes_the_size_of_the_operator_that_drew_it():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, SCALED)
    reader.visit(" and the rest of the sentence", UNIT, UNIT, {}, 1.0)  # pypdf lost this piece's matrices
    assert reader.pieces == [(" and the rest of the sentence", 11.0, False, 700.0)]


def test_a_piece_that_describes_itself_correctly_keeps_its_size():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, SCALED)
    reader.visit("text", UNIT, SCALED, {}, 1.0)
    assert reader.pieces[0][1:] == (11.0, False, 700.0)


def test_a_piece_in_unit_matrices_takes_the_baseline_of_the_operator_too():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 8.04], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, [1.0, 0.0, 0.0, 1.0, 100.0, 397.73])
    reader.visit(" 2", UNIT, UNIT, {}, 8.04)  # a raised digit whose own matrices were lost; its size is right
    assert reader.pieces[0][1:] == (8.04, False, 397.73)


def test_unit_matrices_are_believed_when_the_operator_agrees():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 12.0], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, UNIT)
    reader.visit("plain text at the font size", UNIT, UNIT, {}, 12.0)
    assert reader.pieces[0][1] == 12.0


def test_without_an_operator_to_check_against_the_piece_is_taken_as_it_is():
    reader = pdfread._PageReader()
    reader.visit("text", UNIT, UNIT, {}, 1.0)
    assert reader.pieces[0][1] == 1.0


def test_an_operator_of_another_font_size_is_not_taken_for_the_one_that_drew_the_piece():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, SCALED)
    reader.visit("text of another font size", UNIT, UNIT, {}, 8.0)
    assert reader.pieces[0][1] == 8.0


def test_a_second_piece_of_the_same_operator_is_checked_against_it_too():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, SCALED)
    reader.visit("first part", UNIT, SCALED, {}, 1.0)
    reader.visit(" second part", UNIT, UNIT, {}, 1.0)
    assert [piece[1] for piece in reader.pieces] == [11.0, 11.0]


def test_the_font_size_of_a_restored_graphics_state_is_known_to_the_operators():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    reader.before(b"q", [], UNIT, UNIT)
    reader.before(b"Tf", ["/F1", 20.0], UNIT, UNIT)
    reader.before(b"Q", [], UNIT, UNIT)
    reader.before(b"Tj", [], UNIT, SCALED)  # drawn at font size 1 again
    reader.visit("text", UNIT, UNIT, {}, 1.0)
    assert reader.pieces[0][1] == 11.0


def test_operators_waiting_for_their_text_are_bounded():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", 1.0], UNIT, UNIT)
    for _ in range(pdfread.MAX_PENDING * 3):
        reader.before(b"Tj", [], UNIT, SCALED)
    assert len(reader._pending) <= pdfread.MAX_PENDING


def test_an_operator_that_cannot_be_read_does_not_stop_the_report():
    reader = pdfread._PageReader()
    reader.before(b"Tf", ["/F1", "not a number"], UNIT, UNIT)
    reader.before(b"Tj", [], "not a matrix", None)
    reader.visit("text", UNIT, SCALED, {}, 1.0)
    assert reader.pieces[0][0] == "text"


# ------------------------------------------------------------------------------------- text drawn by forms


def page_with_footer_form(scale: float = 1.0):
    """A page with a line of body text and a footer drawn by a form (``scale`` is the size of the form on the page)."""
    from pypdf import PdfReader

    pdf = make_pdf_with_form(
        ["BT /F1 10 Tf 1 0 0 1 40 700 Tm (Body text of the page.) Tj ET", f"q {scale} 0 0 {scale} 0 0 cm /Fm1 Do Q"],
        ["BT /F1 10 Tf 1 0 0 1 100 100 Tm (OECD/LEGAL/0449) Tj ET"],
    )
    return PdfReader(io.BytesIO(pdf)).pages[0]


def test_text_drawn_by_a_form_gets_its_typography_whichever_pypdf_is_installed():
    """The OECD recommendation puts its header in a form. pypdf 6.17 reports such text twice, newer versions once;
    the page must come out the same either way (this test once asserted the count and failed on pypdf 6.21)."""
    page = page_with_footer_form()
    reader = pdfread._PageReader()
    page.extract_text(visitor_text=reader.visit, visitor_operand_before=reader.before)
    assert [piece[0] for piece in reader.pieces].count("OECD/LEGAL/0449") in (1, 2)
    text, styles = read_page(page_with_footer_form())
    assert text == "Body text of the page.\nOECD/LEGAL/0449"
    assert styles == [(10.0, False), (10.0, False)]


@pytest.mark.parametrize("reports", [1, 2])
def test_text_that_pypdf_reports_once_or_twice_still_gets_its_typography(monkeypatch, reports):
    """Report the form's text once or twice whatever the installed pypdf does. Twice is what pypdf 6.17 does: once with
    the matrices of the form (10 pt) and once more with those of the page (3 pt). The repeat is dropped, the first kept."""
    page = page_with_footer_form(scale=0.3)
    real = page.extract_text
    reported = []

    def controlled(*args, visitor_text=None, **kwargs):
        if visitor_text is None:
            return real(*args, **kwargs)

        def visit(text, cm, tm, font_dict, font_size):
            if "OECD" not in text:
                return visitor_text(text, cm, tm, font_dict, font_size)
            if not reported:  # one controlled set of reports, whatever the installed pypdf says about this header
                reported.append(text)
                at = [1.0, 0.0, 0.0, 1.0, 100.0, 100.0]
                visitor_text(text, UNIT, at, font_dict, font_size)
                if reports == 2:
                    visitor_text(text, [0.3, 0.0, 0.0, 0.3, 0.0, 0.0], at, font_dict, font_size)

        return real(*args, visitor_text=visit, **kwargs)

    monkeypatch.setattr(page, "extract_text", controlled)
    reader = pdfread._PageReader()
    page.extract_text(visitor_text=reader.visit, visitor_operand_before=reader.before)
    assert [piece[1] for piece in reader.pieces if "OECD" in piece[0]] == [10.0, 3.0][:reports]
    reported.clear()
    text, styles = read_page(page)
    assert text == "Body text of the page.\nOECD/LEGAL/0449"
    assert styles == [(10.0, False), (10.0, False)]


def test_two_identical_pieces_in_a_row_are_kept_when_the_page_really_says_so():
    lines, styles = lines_and_styles([line(run("ha"), run("ha"))])
    assert lines == ["haha"] and styles == [(11.0, False)]


def test_repeated_pieces_are_dropped_only_when_that_makes_them_add_up_to_the_text():
    a, b, newline = ("alpha", 10.0, False, 1.0), ("beta", 10.0, False, 2.0), ("\n", 10.0, False, 3.0)
    assert pdfread._without_repeats([a, a, newline, b, b], "alpha\nbeta") == [a, newline, b]
    assert pdfread._without_repeats([a, a, a], "alpha") == [a]  # three times
    assert pdfread._without_repeats([a, a, b], "alpha beta") is None  # the text itself was different
    assert pdfread._without_repeats([a, newline, newline, b], "alpha\n\nbeta") == [a, newline, newline, b]  # line breaks are not repeats
    assert pdfread._without_repeats([a, b], "alphabeta") == [a, b]  # nothing to drop
    assert pdfread._without_repeats([a, b, a], "alphabetaalpha") == [a, b, a]  # the first piece is not a repeat of the last
    inside_the_form, from_the_page = ("alpha", 10.0, False, 100.0), ("alpha", 3.0, False, 210.0)
    assert pdfread._without_repeats([inside_the_form, from_the_page], "alpha") == [inside_the_form]  # the first is the reliable one


# ------------------------------------------------------------------------------- fallbacks and files


def test_read_page_gives_no_typography_when_the_pieces_do_not_add_up_to_the_page_text(monkeypatch):
    page = page_of(["BT /F1 10 Tf 1 0 0 1 40 700 Tm (Some text.) Tj ET"])
    real = page.extract_text

    def odd(visitor_text=None, **kwargs):
        text = real(visitor_text=visitor_text, **kwargs)
        return text + " (text the visitor never saw)"

    monkeypatch.setattr(page, "extract_text", odd)
    text, styles = read_page(page)
    assert styles is None and text.endswith("never saw)")


def test_a_page_whose_font_report_fails_still_gives_its_text(monkeypatch):
    page = page_of(["BT /F1 10 Tf 1 0 0 1 40 700 Tm (Some text on the page.) Tj ET"])
    real = page.extract_text

    def refuse_the_report(*args, visitor_text=None, **kwargs):
        if visitor_text is not None:
            raise RuntimeError("the font report broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(page, "extract_text", refuse_the_report)
    text, styles = read_page(page)
    assert text == "Some text on the page." and styles is None


def test_a_page_that_cannot_be_read_at_all_is_still_an_error(monkeypatch):
    page = page_of(["BT /F1 10 Tf 1 0 0 1 40 700 Tm (Some text.) Tj ET"])

    def broken(*args, **kwargs):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(page, "extract_text", broken)
    with pytest.raises(RuntimeError):
        read_page(page)


def test_read_pdf_returns_text_and_styles_for_every_page(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_multipage_pdf([["First page.", small("1 A note.")], ["Second page."]]))
    result = read_pdf(path)
    assert result.pages == ["First page.\n1 A note.", "Second page."]
    assert result.styles == [[(10.0, False), (8.0, False)], [(10.0, False)]]


def test_read_pdf_marks_bold_lines(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_multipage_pdf([[bold("A bold heading", 12), "Text."]]))
    assert read_pdf(path).styles == [[(12.0, True), (10.0, False)]]


def test_read_pdf_still_reports_unreadable_files(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nnot a pdf " * 100)
    with pytest.raises(ExtractionError):
        read_pdf(path)


def test_module_constants():
    assert 0.3 < pdfread.RAISED_RATIO < 0.9
