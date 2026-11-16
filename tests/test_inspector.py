"""The PDF inspection tool: page ranges, finding a file, and what it prints about a document."""

import pytest

from reglens.config import PROJECT_ROOT
from reglens.ingest import inspector
from fixtures import official_journal_pages
from helpers import bold, make_multipage_pdf, small

MANIFEST = PROJECT_ROOT / "data" / "sources.csv"


def run(*args, raw_dir=None):
    lines: list[str] = []
    extra = ["--raw-dir", str(raw_dir)] if raw_dir else []
    code = inspector.main([*args, "--manifest", str(MANIFEST), *extra], out=lines.append)
    return code, "\n".join(lines)


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "journal.pdf"
    path.write_bytes(make_multipage_pdf(official_journal_pages()))
    return path


# ------------------------------------------------------------------------------------------ pages


@pytest.mark.parametrize(
    "spec, total, expected",
    [("5", 12, [5]), ("3-6", 12, [3, 4, 5, 6]), ("10-", 12, [10, 11, 12]), (" 2 - 4 ", 12, [2, 3, 4]), ("3-99", 5, [3, 4, 5]), ("1", 1, [1])],
)
def test_page_ranges(spec, total, expected):
    assert inspector.parse_pages(spec, total) == expected


@pytest.mark.parametrize("spec", ["", "abc", "0", "6-3", "-4", "1,2", "99"])
def test_page_ranges_that_make_no_sense_are_explained(spec):
    with pytest.raises(ValueError):
        inspector.parse_pages(spec, 12)


# ------------------------------------------------------------------------------------------ files


def test_a_path_is_used_as_it_is(journal):
    assert inspector.find_file(str(journal), journal.parent, MANIFEST) == journal


def test_a_document_id_is_looked_up_in_the_raw_folder(tmp_path):
    (tmp_path / "eu-gdpr.pdf").write_bytes(b"%PDF-1.4")
    assert inspector.find_file("eu-gdpr", tmp_path, MANIFEST) == tmp_path / "eu-gdpr.pdf"


def test_an_id_whose_file_was_not_downloaded_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_corpus"):
        inspector.find_file("eu-gdpr", tmp_path, MANIFEST)


def test_an_unknown_name_suggests_similar_ids(tmp_path):
    with pytest.raises(FileNotFoundError, match="Did you mean: .*eu-gdpr"):
        inspector.find_file("gdpr", tmp_path, MANIFEST)


# ------------------------------------------------------------------------------------------ output


def test_the_summary_gives_the_sizes_of_the_document_and_what_was_found(journal):
    code, text = run(str(journal), "--summary-only")
    assert code == 0
    assert "pages: 12   with readable font sizes: 12" in text
    assert "running text: 10.0 pt" in text and "footnotes found: 12 on 12 pages" in text
    assert "8.0 pt" in text and "10.0 pt" in text


def test_every_line_of_a_page_is_listed_with_its_type_and_tag(journal):
    code, text = run(str(journal), "--pages", "3")
    assert code == 0
    foot = next(line for line in text.splitlines() if "OJ C 3, 31.7.2099" in line)
    assert " 8.0 " in foot and "small" in foot and "number" in foot
    body_line = next(line for line in text.splitlines() if line.strip().startswith("1 ") and "text" in line)
    assert "10.0" in body_line and "small" not in body_line


def test_large_and_bold_lines_are_tagged(tmp_path):
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
    pages = [
        [bold(f"A Heading {words[page]}", 16), *[f"Line {words[(page + n) % 12]} of the running text {words[(page * 5 + n) % 12]}." for n in range(10)], bold(f"Bold in body size {words[page]}")]
        for page in range(3)
    ]
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_multipage_pdf(pages))
    _, text = run(str(path), "--pages", "1")
    heading = next(line for line in text.splitlines() if "A Heading alpha" in line)
    assert "LARGE" in heading and "bold" in heading and " B " in heading
    assert "bold" in next(line for line in text.splitlines() if "Bold in body size alpha" in line)


def test_find_lists_the_pages_with_a_snippet(journal):
    _, text = run(str(journal), "--summary-only", "--find", "OJ C 3,")
    assert "page    3:" in text and "OJ C 3, 31.7.2099" in text
    _, none = run(str(journal), "--summary-only", "--find", "no such words anywhere")
    assert "0 page(s)" in none


def test_find_ignores_case_and_line_breaks(journal):
    _, text = run(str(journal), "--summary-only", "--find", "official   JOURNAL of")
    assert "0 page(s)" not in text


def test_blocks_shows_what_the_ingestion_makes_of_the_page(journal):
    _, text = run(str(journal), "--pages", "3", "--blocks")
    assert "blocks the ingestion makes of these pages" in text
    assert any(line.lstrip().startswith("[footnote") and "OJ C 3, 31.7.2099" in line for line in text.splitlines())


def test_unreadable_font_sizes_are_reported_instead_of_crashing(tmp_path):
    from helpers import make_pdf

    path = tmp_path / "plain.pdf"
    path.write_bytes(make_pdf(["Some text on a page that is long enough to be a line."]))
    code, text = run(str(path), "--pages", "1")
    assert code == 0 and "pages: 1" in text


# ------------------------------------------------------------------------------------------ problems


def test_a_web_page_is_refused_politely(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<html></html>")
    code, text = run(str(page))
    assert code == 2 and "not a PDF" in text


def test_a_page_range_outside_the_document_is_refused(journal):
    code, text = run(str(journal), "--pages", "99")
    assert code == 2 and "only 12 page(s)" in text


def test_an_unknown_name_is_refused(tmp_path):
    code, text = run("definitely-not-a-document", raw_dir=tmp_path)
    assert code == 2 and "neither a file nor a document id" in text


def test_a_damaged_pdf_is_reported(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a real pdf at all " * 20)
    code, text = run(str(path))
    assert code == 1 and "could not be read" in text


def test_characters_a_console_cannot_show_are_written_as_codes():
    assert inspector._visible("ab\x07c – “quoted”") == "a<U+F0B7>b<U+0007>c – “quoted”"
    assert inspector._short("x" * 300).endswith("...") and len(inspector._short("x" * 300)) == inspector.MAX_TEXT
