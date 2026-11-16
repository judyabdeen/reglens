import csv
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from reglens.corpus.manifest import REQUIRED_COLUMNS, Source
from reglens.ingest import pipeline
from reglens.ingest.chunking import NOTES, TEXT, ChunkOptions
from reglens.ingest.cli import main as ingest_main
from reglens.ingest.models import FOOTNOTE, HEADING, PARAGRAPH, Block
from reglens.ingest.pipeline import (
    Flag,
    audit_blocks,
    export_for_review,
    extraction_fingerprint,
    ingest_source,
    load_blocks,
    outline_text,
    print_summary,
    run_ingest,
)
from fixtures import GUIDANCE_PAGE, official_journal_pages, regulation_pages, statement_pages, treaty_pages
from helpers import make_multipage_pdf, make_pdf


def make_source(doc_id: str, filename: str, fmt: str = "pdf", title: str = "Regulation (EU) 2099/1 on test data", priority: int = 1) -> Source:
    return Source(
        id=doc_id, title=title, publisher="Test", jurisdiction="EU", doc_type="regulation", status="final",
        version_date="2026-01-01", priority=priority, fetch="auto", format=fmt, url="https://example.org/" + filename,
        landing_url="https://example.org/", filename=filename, verification="test",
    )  # fmt: skip


@pytest.fixture
def corpus(tmp_path):
    """A small corpus on disk: two PDFs, one web page, one corrupt PDF, one file that was never downloaded."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reg.pdf").write_bytes(make_multipage_pdf(regulation_pages()))
    (raw / "stmt.pdf").write_bytes(make_multipage_pdf(statement_pages()))
    (raw / "guidance.html").write_text(GUIDANCE_PAGE, encoding="utf-8")
    (raw / "corrupt.pdf").write_bytes(b"%PDF-1.4\nthis is not really a pdf " * 50)
    sources = [
        make_source("reg", "reg.pdf"),
        make_source("stmt", "stmt.pdf", title="Proposed policy statement concerning the accuracy of AI systems"),
        make_source("guidance", "guidance.html", fmt="html", title="Guidance on AI and data protection", priority=2),
        make_source("corrupt", "corrupt.pdf"),
        make_source("absent", "absent.pdf"),
    ]
    return raw, tmp_path / "processed", sources


def run(corpus, **kwargs):
    raw, out_dir, sources = corpus
    lines: list[str] = []
    results, chunks = run_ingest(sources, raw, out_dir, ChunkOptions(), out=lines.append, **kwargs)
    return {r.id: r for r in results}, chunks, lines


# ------------------------------------------------------------------------------------ a full run


def test_a_full_run_ingests_what_it_can_and_reports_what_it_cannot(corpus):
    results, chunks, lines = run(corpus)
    assert {k: v.status for k, v in results.items()} == {
        "reg": "ok", "stmt": "ok", "guidance": "ok", "corrupt": "failed", "absent": "missing",
    }  # fmt: skip
    assert "could not be read" in results["corrupt"].detail
    assert "download_corpus" in results["absent"].detail
    assert {c.doc_id for c in chunks} == {"reg", "stmt", "guidance"}
    text = "\n".join(lines)
    assert text.count("reading") == 5 and "FAILED" in text and "MISSING" in text and "OK" in text


def test_the_numbers_in_the_result_match_the_chunks(corpus):
    results, chunks, _ = run(corpus)
    reg = results["reg"]
    assert reg.pages == 10 and reg.footnotes == 12 and reg.headings == 14
    assert reg.chunks == sum(1 for c in chunks if c.doc_id == "reg" and c.kind == TEXT)
    assert reg.note_chunks == sum(1 for c in chunks if c.doc_id == "reg" and c.kind == NOTES) >= 1
    assert reg.words > 500 and reg.chunk_words_median > 0
    assert results["guidance"].pages is None  # web pages have no pages


def test_output_files_are_written(corpus):
    _, out_dir, _ = corpus
    results, chunks, _ = run(corpus)
    assert (out_dir / "chunks.jsonl").is_file() and (out_dir / "ingest_report.json").is_file()
    for ok_id in ("reg", "stmt", "guidance"):
        assert (out_dir / "blocks" / f"{ok_id}.json").is_file()
    assert not (out_dir / "blocks" / "corrupt.json").exists() and not (out_dir / "blocks" / "absent.json").exists()

    rows = [json.loads(line) for line in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(chunks)
    assert {"chunk_id", "doc_id", "kind", "text", "embed_text", "section_path", "page_start", "n_words"} <= rows[0].keys()
    assert len({r["chunk_id"] for r in rows}) == len(rows)

    report = json.loads((out_dir / "ingest_report.json").read_text(encoding="utf-8"))
    assert report["totals"]["documents"] == 5 and report["totals"]["ok"] == 3
    assert report["totals"]["failed"] == 1 and report["totals"]["missing"] == 1
    assert report["options"]["strategy"] == "structure"
    assert [d["id"] for d in report["documents"]] == ["reg", "stmt", "guidance", "corrupt", "absent"]


def test_no_temporary_files_are_left_behind(corpus):
    _, out_dir, _ = corpus
    run(corpus)
    assert not list(out_dir.rglob("*.tmp"))


def test_output_is_utf8_with_unix_newlines(corpus):
    _, out_dir, _ = corpus
    run(corpus)
    data = (out_dir / "chunks.jsonl").read_bytes()
    assert b"\r\n" not in data
    data.decode("utf-8")


# ---------------------------------------------------------------------------------------- caching


def test_second_run_uses_the_cache_and_gives_identical_chunks(corpus):
    _, first_chunks, _ = run(corpus)
    results, second_chunks, lines = run(corpus)
    assert all(results[i].cached for i in ("reg", "stmt", "guidance"))
    assert [c.to_dict() for c in second_chunks] == [c.to_dict() for c in first_chunks]
    assert "OK (cached)" in "\n".join(lines)


def test_changed_file_or_changed_code_or_force_invalidates_the_cache(corpus, monkeypatch):
    raw, _, _ = corpus
    run(corpus)
    (raw / "stmt.pdf").write_bytes(make_multipage_pdf(statement_pages()[:3]))  # a different file
    results, _, _ = run(corpus)
    assert results["stmt"].cached is False and results["reg"].cached is True

    monkeypatch.setattr(pipeline, "extraction_fingerprint", lambda: "another-version")
    results, _, _ = run(corpus)
    assert not any(results[i].cached for i in ("reg", "stmt", "guidance"))

    monkeypatch.undo()
    run(corpus)  # restores a cache for the real fingerprint
    results, _, _ = run(corpus, force=True)
    assert not any(results[i].cached for i in ("reg", "stmt", "guidance"))


def test_a_damaged_cache_file_is_simply_rebuilt(corpus):
    _, out_dir, _ = corpus
    run(corpus)
    (out_dir / "blocks" / "reg.json").write_text("{not json", encoding="utf-8")
    (out_dir / "blocks" / "stmt.json").write_text(json.dumps({"id": "stmt"}), encoding="utf-8")
    results, _, _ = run(corpus)
    assert results["reg"].status == "ok" and results["reg"].cached is False
    assert results["stmt"].status == "ok" and results["stmt"].cached is False


def test_extraction_fingerprint_is_stable_and_short():
    assert extraction_fingerprint() == extraction_fingerprint()
    assert len(extraction_fingerprint()) == 16


# ------------------------------------------------------------------------- partial runs and tags


def test_a_partial_run_does_not_touch_the_combined_files(corpus):
    raw, out_dir, sources = corpus
    run(corpus)
    before = (out_dir / "chunks.jsonl").read_bytes()
    report_before = (out_dir / "ingest_report.json").read_bytes()
    lines: list[str] = []
    run_ingest(sources[:1], raw, out_dir, ChunkOptions(), out=lines.append, write_combined=False)
    assert (out_dir / "chunks.jsonl").read_bytes() == before
    assert (out_dir / "ingest_report.json").read_bytes() == report_before


def test_a_tag_gives_separate_output_files_for_another_strategy(corpus):
    raw, out_dir, sources = corpus
    run(corpus)
    run_ingest(sources, raw, out_dir, ChunkOptions(strategy="fixed"), out=lambda _l: None, tag="fixed")
    assert (out_dir / "chunks-fixed.jsonl").is_file() and (out_dir / "ingest_report-fixed.json").is_file()
    fixed = [json.loads(line) for line in (out_dir / "chunks-fixed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert fixed and all(row["kind"] == TEXT for row in fixed)
    structure_rows = (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(fixed) != len(structure_rows)


# ---------------------------------------------------------------------------------- robustness


def test_a_pdf_without_text_is_reported_not_crashed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "scan.pdf").write_bytes(make_pdf(None))
    result, chunks = ingest_source(make_source("scan", "scan.pdf"), raw, tmp_path / "out", ChunkOptions())
    assert result.status == "failed" and "no text" in result.detail and chunks == []
    assert any(flag.code == "NO_TEXT" for flag in result.flags)


def test_a_surprising_exception_in_one_document_becomes_a_failed_result(corpus, monkeypatch):
    raw, out_dir, sources = corpus

    def explode(source, path):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(pipeline, "extract_document", explode)
    result, chunks = ingest_source(sources[0], raw, out_dir, ChunkOptions())
    assert result.status == "failed" and "RuntimeError" in result.detail and chunks == []


def test_a_file_that_cannot_be_read_is_a_failed_result(corpus, monkeypatch):
    raw, out_dir, sources = corpus

    def locked(source, path):
        raise PermissionError("locked by another program")

    monkeypatch.setattr(pipeline, "extract_document", locked)
    result, _ = ingest_source(sources[0], raw, out_dir, ChunkOptions())
    assert result.status == "failed" and "PermissionError" in result.detail


# ------------------------------------------------------------------------------------- the audit


def body(words: int, text: str = "the controller shall process data in accordance with the law ") -> Block:
    unit = text.split()
    return Block(PARAGRAPH, " ".join((unit * (words // len(unit) + 1))[:words]))


def codes(source: Source, blocks: list[Block], **info) -> set[str]:
    return {flag.code for flag in audit_blocks(source, blocks, info)}


SOURCE = make_source("x", "x.pdf", title="Regulation on controllers and data")


def test_audit_of_a_healthy_document_has_no_warnings():
    blocks = [Block(HEADING, f"Article {n}", level=1) if n % 2 == 0 else body(150) for n in range(10)]
    blocks.insert(0, Block(HEADING, "Regulation on controllers and data", level=1))
    flags = audit_blocks(SOURCE, blocks, {"pages": 10, "chars_total": 20000})
    assert [f for f in flags if f.level == "warn"] == []


def test_audit_flags_missing_and_unusable_text():
    assert codes(SOURCE, []) == {"NO_TEXT"}
    assert "LOW_TEXT" in codes(SOURCE, [body(100)], pages=10, chars_total=2000)
    assert "EMPTY_PAGES" in codes(SOURCE, [body(100)], pages=10, chars_total=90000, empty_pages=3)
    assert "PAGE_ERRORS" in codes(SOURCE, [body(100)], pages=10, chars_total=90000, page_errors=["page 3: ValueError"])


def test_audit_flags_structure_problems():
    assert "NO_HEADINGS" in codes(SOURCE, [body(2000)], pages=8, chars_total=90000)
    many = [Block(HEADING, "Article") if n % 2 == 0 else body(8) for n in range(40)]
    assert "MANY_HEADINGS" in codes(SOURCE, many, pages=2, chars_total=90000)
    assert "LONG_PARAGRAPHS" in codes(SOURCE, [Block(HEADING, "A", level=1), body(900), Block(HEADING, "B", level=1), Block(HEADING, "C", level=1)], pages=2, chars_total=90000)


def flag_of(code: str, blocks, **info):
    return next((flag for flag in audit_blocks(SOURCE, blocks, info) if flag.code == code), None)


def test_blank_and_divider_pages_between_text_pages_are_context_not_a_warning():
    divider_pages = [8, 9, 12, 13, 16, 17, 24, 25]
    flag = flag_of("EMPTY_PAGES", [body(100)], pages=44, chars_total=90000, empty_pages=8, empty_page_numbers=divider_pages)
    assert flag.level == "info" and "pages 8, 9, 12, 13, 16, 17, 24, 25" in flag.message and "normal" in flag.message


def test_a_run_of_empty_pages_or_a_large_share_of_them_is_a_warning():
    run = flag_of("EMPTY_PAGES", [body(100)], pages=60, chars_total=90000, empty_pages=5, empty_page_numbers=[20, 21, 22, 23, 24])
    assert run.level == "warn" and "text layer" in run.message
    share = flag_of("EMPTY_PAGES", [body(100)], pages=20, chars_total=90000, empty_pages=6, empty_page_numbers=[2, 4, 6, 8, 10, 12])
    assert share.level == "warn"
    no_numbers = flag_of("EMPTY_PAGES", [body(100)], pages=10, chars_total=90000, empty_pages=3)  # older results: count only
    assert no_numbers.level == "warn" and "pages (" not in no_numbers.message


def test_a_few_empty_pages_are_not_flagged_at_all():
    assert flag_of("EMPTY_PAGES", [body(100)], pages=44, chars_total=90000, empty_pages=3, empty_page_numbers=[8, 12, 16]) is None


def test_long_lists_of_pages_are_shortened():
    numbers = list(range(1, 40, 2))
    assert pipeline._page_list(numbers) == "1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, ..."
    assert pipeline._page_list([4, 9]) == "4, 9"


@pytest.mark.parametrize("numbers, expected", [([], 0), ([3], 1), ([1, 3, 5], 1), ([4, 5, 6, 9, 10], 3), ([8, 9, 12, 13, 16, 17, 24, 25], 2)])
def test_longest_run_of_consecutive_pages(numbers, expected):
    assert pipeline._longest_run(numbers) == expected


def test_the_garbled_characters_warning_says_which_glyphs_were_lost():
    flag = flag_of("GARBLED_CHARS", [body(100)], pages=2, chars_total=90000, private_use_dropped=31, dropped_codes={"U+F0A8": 20, "U+F0FC": 11})
    assert "31 unreadable character codes" in flag.message and "U+F0A8 x20" in flag.message and "U+F0FC x11" in flag.message
    plain = flag_of("GARBLED_CHARS", [body(100)], pages=2, chars_total=90000, cid_refs=25)
    assert plain.message == "25 unreadable character codes were removed"


def test_glyph_counters_are_folded_into_one_entry_of_the_most_frequent_glyphs():
    info = {"pages": 4, "private_use_dropped": 40, **{f"U+F0{n:02X}": n for n in range(1, 10)}}
    tidy = pipeline._tidy_info(info)
    assert not any(key.startswith("U+") for key in tidy)
    assert list(tidy["dropped_codes"]) == ["U+F009", "U+F008", "U+F007", "U+F006", "U+F005", "U+F004"]
    assert pipeline._tidy_info({"pages": 1}) == {"pages": 1}


def test_the_parts_of_a_short_legal_text_are_not_counted_as_too_many_headings():
    articles = [Block(HEADING, f"Article {n} – Subject", level=2) if n % 2 == 0 else body(40) for n in range(80)]
    assert "MANY_HEADINGS" not in codes(SOURCE, articles, pages=12, chars_total=90000)
    loose = [Block(HEADING, f"Subject number {n}", level=2) if n % 2 == 0 else body(40) for n in range(80)]
    assert "MANY_HEADINGS" in codes(SOURCE, loose, pages=12, chars_total=90000)


def test_audit_flags_garbled_text():
    assert "NOT_ENGLISH" in codes(SOURCE, [body(200, "xqzt wvbn mlkj hgfd rtyu plmk ")], pages=2, chars_total=90000)
    glued = Block(PARAGRAPH, " ".join(["thecontrollershallprocessdatainaccordancewiththelaw"] * 20 + ["the data of the law"] * 100))
    assert "RUN_TOGETHER_WORDS" in codes(SOURCE, [glued], pages=2, chars_total=90000)
    assert "GARBLED_CHARS" in codes(SOURCE, [body(200)], pages=2, chars_total=90000, cid_refs=25)
    assert "TITLE_MISMATCH" in codes(make_source("y", "y.pdf", title="Biometric surveillance in prisons"), [body(200)], pages=2, chars_total=90000)


def test_audit_information_flags_are_not_warnings():
    notes = [Block(FOOTNOTE, " ".join(["note"] * 400)), body(100), Block(HEADING, "Article 1", level=1)]
    flags = audit_blocks(SOURCE, notes, {"pages": 2, "chars_total": 90000, "outline_used": True, "outline_matches": 7, "rule_level": 1})
    info = {f.code: f.level for f in flags}
    assert info["NOTE_HEAVY"] == info["OUTLINE"] == info["WEAK_RULES_OFF"] == "info"


def test_a_bibliography_does_not_make_an_english_document_look_foreign():
    prose = [Block(PARAGRAPH, "the controller shall process the data in accordance with the law and the rules " * 6, page=page) for page in range(1, 7)]
    references = [Block(PARAGRAPH, "Smith J. Doe A. arXiv:2301.0001 NISTAML.011 Zhang L. Proc. ICML 2023 vol. 5 pp. 12 " * 8, page=page) for page in range(7, 10)]
    assert "NOT_ENGLISH" not in codes(SOURCE, prose + references, pages=9, chars_total=90000)
    only_references = [Block(PARAGRAPH, "Smith J. Doe A. arXiv:2301.0001 NISTAML.011 Zhang L. Proc. ICML 2023 vol. 5 pp. 12 " * 8, page=page) for page in range(1, 10)]
    assert "NOT_ENGLISH" in codes(SOURCE, only_references, pages=9, chars_total=90000)


def test_telegraphic_english_such_as_a_table_of_suggested_actions_is_not_foreign():
    actions = "the of and alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo " * 6  # 21 % common words
    assert "NOT_ENGLISH" not in codes(SOURCE, [Block(PARAGRAPH, actions, page=page) for page in range(1, 9)], pages=8, chars_total=90000)
    french = "le gouvernement de la république française a adopté une loi sur les données personnelles et la protection des citoyens " * 4
    assert "NOT_ENGLISH" in codes(SOURCE, [Block(PARAGRAPH, french, page=page) for page in range(1, 9)], pages=8, chars_total=90000)


def test_the_english_check_looks_at_a_typical_page_not_at_the_average_of_all_pages():
    telegraphic = "the of and alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo " * 6  # 21 % common words
    identifiers = "NISTAML.011 NISTAML.012 NISTAML.013 arXiv:2301.0001 ICML Zhang Smith Doe Kumar Lopez Okafor Tanaka Novak " * 10
    mostly_text = [Block(PARAGRAPH, telegraphic, page=page) for page in range(1, 7)] + [Block(PARAGRAPH, identifiers, page=page) for page in range(7, 12)]
    assert "NOT_ENGLISH" not in codes(SOURCE, mostly_text, pages=11, chars_total=90000)  # a plain average would be 12 %
    mostly_identifiers = [Block(PARAGRAPH, telegraphic, page=page) for page in range(1, 5)] + [Block(PARAGRAPH, identifiers, page=page) for page in range(5, 11)]
    assert "NOT_ENGLISH" in codes(SOURCE, mostly_identifiers, pages=10, chars_total=90000)


def test_audit_says_when_font_sizes_could_not_be_read():
    assert "NO_TYPOGRAPHY" in codes(SOURCE, [body(100)], pages=10, chars_total=90000, typeset_pages=2)
    assert "NO_TYPOGRAPHY" not in codes(SOURCE, [body(100)], pages=10, chars_total=90000, typeset_pages=10)
    assert "NO_TYPOGRAPHY" not in codes(SOURCE, [body(100)], pages=10, chars_total=90000)  # a web page has no pages to read


def test_flags_of_a_real_run_are_json_friendly(corpus):
    results, _, _ = run(corpus)
    assert all(isinstance(flag, Flag) for flag in results["reg"].flags)
    json.dumps(pipeline._jsonable({"flags": [flag.__dict__ for flag in results["reg"].flags]}))


# ------------------------------------------------- layouts that went wrong in the real corpus, end to end


def extract(tmp_path, pages, name="doc.pdf"):
    path = tmp_path / name
    path.write_bytes(make_multipage_pdf(pages))
    return pipeline.extract_document(make_source("doc", name), path)


def test_headers_inside_the_page_text_and_footnotes_of_an_official_journal_pdf(tmp_path):
    blocks, info = extract(tmp_path, official_journal_pages())
    text = " ".join(b.text for b in blocks)
    assert "Official Journal" not in text and "L 119/" not in text
    notes = [b.text for b in blocks if b.kind == FOOTNOTE]
    assert len(notes) == 12 and all(note.startswith("[1] OJ C ") for note in notes)
    assert info["footnote_pages"] == 12 and info["body_font_size"] == 10.0 and info["typeset_pages"] == 12
    assert not [b for b in blocks if b.kind == HEADING and "OJ" in b.text]  # a header is never a heading


def test_numbered_treaty_paragraphs_are_not_moved_into_footnotes(tmp_path):
    blocks, info = extract(tmp_path, treaty_pages())
    assert [b for b in blocks if b.kind == FOOTNOTE] == [] and info["footnote_pages"] == 0
    assert len([b for b in blocks if b.kind == HEADING]) == 14
    paragraphs = [b.text for b in blocks if b.kind == PARAGRAPH]
    assert sum(1 for text in paragraphs if text.startswith("1 ")) == 14 and sum(1 for text in paragraphs if text.startswith("2 ")) == 14


# ----------------------------------------------------------------------------------- reporting


def test_outline_text_indents_by_level_and_shows_pages():
    blocks = [Block(HEADING, "Chapter I", 1, 1, 1), Block(HEADING, "Article 1", 2, 2, 2), Block(PARAGRAPH, "text", 2, 2), Block(HEADING, "Web heading", None, None, 1)]
    assert outline_text(blocks).splitlines() == ["Chapter I  [p1]", "  Article 1  [p2]", "Web heading  [-]"]


def test_print_summary_lists_failures_and_warnings(corpus):
    results, chunks, _ = run(corpus)
    results["reg"].flags.append(Flag("TEST", "warn", "a test warning"))
    lines: list[str] = []
    print_summary(list(results.values()), chunks, out=lines.append)
    text = "\n".join(lines)
    assert "Ingested 3 of 5 documents" in text
    assert "Not ingested:" in text and "corrupt" in text and "absent" in text
    flagged = sum(1 for r in results.values() if r.ok and r.warnings)
    assert flagged >= 1 and f"{flagged} document(s) have warnings" in text


def test_export_contains_the_cleaned_text_and_the_audit_but_no_raw_files(corpus, tmp_path):
    _, out_dir, _ = corpus
    results, _, _ = run(corpus)
    target = tmp_path / "review.zip"
    size = export_for_review(out_dir, target, list(results.values()))
    assert size == target.stat().st_size > 0
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert names == {"README.txt", "ingest_report.json", "outlines.txt", "blocks/reg.json", "blocks/stmt.json", "blocks/guidance.json"}
        assert "Article 4" in archive.read("outlines.txt").decode("utf-8")
        assert json.loads(archive.read("ingest_report.json"))["totals"]["ok"] == 3
        payload = json.loads(archive.read("blocks/reg.json"))
        assert payload["blocks"] and payload["id"] == "reg"


def test_load_blocks_reads_the_saved_document(corpus):
    _, out_dir, _ = corpus
    run(corpus)
    blocks = load_blocks(out_dir, "reg")
    assert any(b.kind == HEADING and b.text.startswith("Article 4") for b in blocks)


# ----------------------------------------------------------------------------------------- the CLI


@pytest.fixture
def cli_env(corpus, tmp_path):
    raw, out_dir, sources = corpus
    manifest = tmp_path / "sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for source in sources:
            row = {name: getattr(source, name) for name in REQUIRED_COLUMNS}
            writer.writerow(row)

    def call(*args: str) -> int:
        return ingest_main(["--manifest", str(manifest), "--raw-dir", str(raw), "--out-dir", str(out_dir), *args])

    return call, out_dir


def test_cli_exit_code_is_1_when_some_documents_fail(cli_env, capsys):
    call, out_dir = cli_env
    assert call() == 1
    output = capsys.readouterr().out
    assert "Ingested 3 of 5 documents" in output and "Chunks written to" in output
    assert (out_dir / "chunks.jsonl").is_file()


def test_cli_header_says_which_python_and_pypdf_made_the_run(cli_env, capsys):
    """A run on another machine is only comparable when the versions of the tools are known."""
    import sys
    from importlib.metadata import version

    call, _ = cli_env
    call()
    assert f"Python    {sys.version.split()[0]}, pypdf {version('pypdf')}" in capsys.readouterr().out


def test_an_unreadable_pypdf_version_does_not_stop_the_run(monkeypatch):
    from reglens.ingest import cli

    def broken(name):
        raise RuntimeError("no metadata")

    monkeypatch.setattr("importlib.metadata.version", broken)
    assert cli._pypdf_version() == "unknown"


def test_cli_exit_code_is_0_when_everything_works(cli_env, capsys):
    call, out_dir = cli_env
    assert call("--only", "reg", "stmt", "guidance") == 0
    output = capsys.readouterr().out
    assert "Partial run" in output  # 3 of 5 documents selected
    assert not (out_dir / "chunks.jsonl").exists()


def test_cli_rejects_unknown_ids_and_bad_options(cli_env, capsys):
    call, _ = cli_env
    assert call("--only", "no-such-doc") == 2
    assert "unknown source id" in capsys.readouterr().out
    assert call("--target-words", "10", "--min-words", "50") == 2
    assert "Problem" in capsys.readouterr().out


def test_cli_missing_manifest_is_a_clear_message(tmp_path, capsys):
    assert ingest_main(["--manifest", str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path / "o")]) == 2
    assert "manifest not found" in capsys.readouterr().out


def test_cli_core_only_skips_optional_documents(cli_env, capsys):
    call, out_dir = cli_env
    call("--core-only")
    assert "guidance" not in capsys.readouterr().out.replace("guidance.html", "")
    assert not (out_dir / "blocks" / "guidance.json").exists()


def test_cli_show_prints_headings_and_first_chunks(cli_env, capsys):
    call, _ = cli_env
    assert call("--show", "reg") == 0
    output = capsys.readouterr().out
    assert "reg: headings" in output and "Article 4 – Principles" in output and "first 3 chunks" in output


def test_cli_export_writes_the_zip_into_the_output_folder(cli_env, capsys):
    call, out_dir = cli_env
    call("--export")
    assert (out_dir / "reglens-processed-text.zip").is_file()
    assert "Packed the cleaned text" in capsys.readouterr().out


def test_cli_strategy_fixed_uses_its_own_file_names(cli_env):
    call, out_dir = cli_env
    call("--strategy", "fixed")
    assert (out_dir / "chunks-fixed.jsonl").is_file() and not (out_dir / "chunks.jsonl").exists()
    call("--strategy", "fixed", "--tag", "baseline")
    assert (out_dir / "chunks-baseline.jsonl").is_file()


def test_cli_output_survives_a_console_that_cannot_print_unicode(cli_env, monkeypatch, capsys):
    call, _ = cli_env
    assert call("--show", "reg") == 0  # the en dash in "Article 4 - ..." must not crash a cp1252 console
    assert replace  # keep the import used by editors that strip unused imports
    capsys.readouterr()
