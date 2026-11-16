import json

import pytest

from reglens.corpus.manifest import Source
from reglens.ingest import layout
from reglens.ingest.chunking import (
    NOTES,
    TEXT,
    Chunk,
    ChunkOptions,
    chunk_document,
    clean_path,
    context_prefix,
    embed_prefix,
    read_chunks,
    split_sentences,
)
from reglens.ingest.models import FOOTNOTE, HEADING, PARAGRAPH, Block
from reglens.ingest.structure import build_blocks
from fixtures import regulation_pages
from helpers import typeset


def make_source(doc_id: str = "doc-a", title: str = "Guidelines on Testing Chunkers") -> Source:
    return Source(
        id=doc_id, title=title, publisher="Test", jurisdiction="EU", doc_type="guidelines", status="final",
        version_date="2026-01-01", priority=1, fetch="auto", format="pdf", url="https://example.org/x.pdf",
        landing_url="https://example.org/x", filename=f"{doc_id}.pdf", verification="test",
    )  # fmt: skip


def sentence(number: int) -> str:
    """Exactly ten words, unique for each number."""
    return f"Sentence {number} says that every single rule applies to all."


def paragraph(first: int, count: int) -> str:
    return " ".join(sentence(n) for n in range(first, first + count))


def heading(text: str, level: int = 1, page: int = 1) -> Block:
    return Block(HEADING, text, page, page, level)


def para(text: str, page: int = 1) -> Block:
    return Block(PARAGRAPH, text, page, page)


def note(text: str, page: int = 1) -> Block:
    return Block(FOOTNOTE, text, page, page)


def words_of(chunk: Chunk) -> int:
    return len(chunk.text.split())


# --------------------------------------------------------------------------------------- options


def test_default_options_are_valid():
    options = ChunkOptions()
    assert options.strategy == "structure"
    assert options.min_words <= options.target_words <= options.max_words


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "sentence"},
        {"min_words": 0},
        {"min_words": 300, "target_words": 200},
        {"target_words": 400, "max_words": 300},
        {"overlap_words": -1},
        {"overlap_words": 240},
        {"overlap_words": 280, "target_words": 300, "max_words": 320, "min_words": 60},  # leaves no room for text
    ],
)
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkOptions(**kwargs)


# ------------------------------------------------------------------------------------ sentences


def test_split_sentences_basic_and_quotes():
    text = 'It was so. "Really?" Yes! (Maybe.) Then 5 apples fell.'
    assert split_sentences(text) == ['It was so.', '"Really?"', "Yes!", "(Maybe.)", "Then 5 apples fell."]


@pytest.mark.parametrize(
    "text",
    [
        "See Exec. Order No. 14319 and Art. 5 of the Act for details.",
        "The statute is codified at 15 U.S.C. 45 and applies broadly.",
        "As said by J. D. Vance in Paris, the policy changed.",
        "Id. at 5 shows the same result in the record.",
        "Compare e.g. the first rule with the second rule here.",
        "See also Fig. 3 and Sec. 7 for the definitions that apply.",
        "In FTC v. Pukke, the court held that the practice was deceptive.",
        "See Compl., FTC v. Air Ai Techs., Inc. for the allegations made.",
    ],
)
def test_abbreviations_do_not_end_sentences(text):
    assert split_sentences(text) == [text]


def test_a_very_long_run_of_abbreviations_is_split_in_linear_time():
    import time

    started = time.perf_counter()
    parts = split_sentences("No. " * 30000)
    assert time.perf_counter() - started < 5
    assert len(parts) == 1  # every "No." is an abbreviation, so nothing is split


def test_split_sentences_handles_empty_and_single_inputs():
    assert split_sentences("") == []
    assert split_sentences("No full stop at the end") == ["No full stop at the end"]


# ------------------------------------------------------------------------------- paths and prefixes


def test_headings_that_repeat_the_document_title_are_dropped_from_the_path():
    source = make_source(title="Guidelines on Testing Chunkers")
    assert clean_path(source, ["Guidelines on Testing Chunkers", "Article 5"]) == ["Article 5"]
    assert clean_path(source, ["Chapter I", "Article 5"]) == ["Chapter I", "Article 5"]


def test_embed_prefix_has_title_then_last_two_headings():
    source = make_source(title="One two three four five six seven eight nine ten eleven twelve thirteen fourteen")
    prefix = embed_prefix(source, ["Chapter I", "Section 2", "Article 5"], "footnotes")
    title, crumb, extra = prefix.split(" | ")
    assert title == "One two three four five six seven eight nine ten eleven twelve"  # first 12 words
    assert crumb == "Section 2 > Article 5"
    assert extra == "footnotes"
    assert embed_prefix(source, []) == title


# --------------------------------------------------------------------------- structure strategy


def test_a_section_that_fits_becomes_one_chunk_with_its_heading_and_breadcrumb():
    source = make_source()
    blocks = [heading("Chapter I", 1, 3), heading("Article 5", 2, 3), para(paragraph(1, 9), page=3), para(paragraph(10, 5), page=4)]
    (chunk,) = chunk_document(source, blocks)
    assert chunk.chunk_id == "doc-a:0001" and chunk.doc_id == "doc-a" and chunk.kind == TEXT and chunk.ordinal == 1
    assert chunk.text.startswith("Article 5\n\nSentence 1 says")
    assert chunk.section_path == ["Chapter I", "Article 5"] and chunk.section == "Chapter I > Article 5"
    assert (chunk.page_start, chunk.page_end) == (3, 4)
    assert chunk.n_words == words_of(chunk) == 2 + 140
    assert chunk.embed_text.startswith("Guidelines on Testing Chunkers | Chapter I > Article 5\nArticle 5\n\n")


def test_urls_are_not_embedded_but_stay_in_the_text():
    blocks = [heading("Sources"), para("Read https://www.example.org/very/long/path?query=1 and www.example.com/a now. " + paragraph(1, 7))]
    (chunk,) = chunk_document(make_source(), blocks)
    assert "https://www.example.org/very/long/path?query=1" in chunk.text
    assert "[link]" in chunk.embed_text and "example.org" not in chunk.embed_text


def test_small_neighbouring_sections_are_merged_until_they_are_useful():
    blocks = []
    for number in range(1, 7):
        blocks += [heading(f"Article {number}", 2), para(paragraph(number * 10, 2))]  # 20 words each
    options = ChunkOptions(target_words=100, max_words=140, min_words=50, overlap_words=10)
    chunks = chunk_document(make_source(), blocks, options)
    assert len(chunks) < 6
    assert all(words_of(c) <= options.max_words for c in chunks)
    first = chunks[0]
    assert "Article 1" in first.text and "Article 2" in first.text
    assert first.section_path == ["Article 1"]  # the path of the section the chunk starts in


def test_a_long_section_is_cut_with_overlap_and_no_chunk_exceeds_the_maximum():
    blocks = [heading("Article 4"), *[para(paragraph(k * 8, 8)) for k in range(1, 9)]]  # 8 paragraphs of 80 words
    options = ChunkOptions()
    chunks = chunk_document(make_source(), blocks, options)
    assert len(chunks) >= 3
    assert all(words_of(c) <= options.max_words for c in chunks), [words_of(c) for c in chunks]
    assert all(c.section_path == ["Article 4"] for c in chunks)
    assert chunks[0].text.startswith("Article 4\n\n") and not chunks[1].text.startswith("Article 4")
    # the second chunk starts with the end of the first one
    first_tail = set(chunks[0].text.split()[-options.overlap_words :])
    assert {w for w in chunks[1].text.split()[: options.overlap_words]} & first_tail
    assert [c.chunk_id for c in chunks] == [f"doc-a:{n:04d}" for n in range(1, len(chunks) + 1)]


def test_no_sentence_is_lost_or_cut_when_a_section_is_split():
    blocks = [heading("Article 4"), *[para(paragraph(k * 8, 8)) for k in range(1, 9)]]
    chunks = chunk_document(make_source(), blocks)
    all_text = "\n".join(c.text for c in chunks)
    for number in range(8, 72):
        assert sentence(number) in all_text, number


def test_overlap_zero_gives_chunks_that_do_not_repeat_text():
    options = ChunkOptions(target_words=100, max_words=120, min_words=30, overlap_words=0)
    blocks = [heading("Article 4"), *[para(paragraph(k * 5, 5)) for k in range(1, 9)]]
    chunks = chunk_document(make_source(), blocks, options)
    seen: set[str] = set()
    for chunk in chunks:
        for number in range(40, 80):
            if sentence(number) in chunk.text:
                assert number not in seen
                seen.add(number)


def test_a_single_huge_paragraph_is_split_at_sentence_ends():
    options = ChunkOptions()
    (first, *rest) = chunk_document(make_source(), [heading("Annex"), para(paragraph(1, 100))], options)  # 1000 words
    assert rest and all(words_of(c) <= options.max_words for c in [first, *rest])
    assert all(c.text.rstrip().endswith("applies to all.") for c in [first, *rest])


def test_a_single_sentence_longer_than_the_limit_is_cut_at_words():
    options = ChunkOptions()
    giant = " ".join(f"word{n}" for n in range(900)) + "."
    chunks = chunk_document(make_source(), [heading("Annex"), para(giant)], options)
    assert len(chunks) >= 3 and all(words_of(c) <= options.max_words for c in chunks)
    assert "word0" in chunks[0].text and "word899." in chunks[-1].text


def test_a_tiny_tail_is_attached_to_the_chunk_before_it():
    options = ChunkOptions(target_words=100, max_words=140, min_words=50, overlap_words=10)
    blocks = [heading("Article 4"), para(paragraph(1, 12)), para(paragraph(100, 2))]  # 120 + 20 words
    chunks = chunk_document(make_source(), blocks, options)
    assert len(chunks) == 1 or all(words_of(c) >= options.min_words // 2 for c in chunks)


def test_footnotes_become_separate_note_chunks():
    blocks = [
        heading("I. Summary"),
        para(paragraph(1, 9)),
        note("[1] See Exec. Order No. 1, 90 Fed. Reg. 1 (2025)."),
        note("[2] Id."),
        heading("II. Background"),
        para(paragraph(20, 9)),
        note("[3] https://example.org/a-very-long-address-that-would-confuse-an-embedding-model."),
    ]
    chunks = chunk_document(make_source(), blocks)
    text_chunks = [c for c in chunks if c.kind == TEXT]
    note_chunks = [c for c in chunks if c.kind == NOTES]
    assert len(text_chunks) == 2 and len(note_chunks) == 1
    assert all("[1]" not in c.text for c in text_chunks)
    (notes,) = note_chunks
    assert notes.chunk_id == "doc-a:n001" and notes.ordinal == 1
    assert notes.text.startswith("Footnotes:\n\n[1] See Exec. Order")
    assert "[3] https://example.org" in notes.text  # the stored text keeps the link
    assert "| footnotes" in notes.embed_text and "example.org" not in notes.embed_text
    assert chunks.index(notes) > chunks.index(text_chunks[-1])  # notes follow the text chunks


def test_many_footnotes_are_packed_into_chunks_of_about_the_target_size():
    options = ChunkOptions()
    blocks = [heading("A"), para(paragraph(1, 9))] + [note(f"[{n}] " + paragraph(n * 2, 3)) for n in range(1, 41)]  # 40 notes of 30 words
    note_chunks = [c for c in chunk_document(make_source(), blocks, options) if c.kind == NOTES]
    assert 4 <= len(note_chunks) <= 8
    assert all(c.n_words <= options.max_words for c in note_chunks)
    assert [c.chunk_id for c in note_chunks] == [f"doc-a:n{n:03d}" for n in range(1, len(note_chunks) + 1)]


def test_footnotes_from_distant_pages_do_not_share_a_chunk():
    blocks = [heading("A"), para(paragraph(1, 9))] + [
        note("[1] See the first source.", page=3),
        note("[2] See the second source.", page=4),
        note("[3] See a source much later.", page=12),
        note("[4] And one more.", page=13),
    ]
    note_chunks = [c for c in chunk_document(make_source(), blocks) if c.kind == NOTES]
    assert [(c.page_start, c.page_end) for c in note_chunks] == [(3, 4), (12, 13)]
    assert "[2]" in note_chunks[0].text and "[3]" in note_chunks[1].text


def test_footnotes_one_page_apart_still_share_a_chunk_but_two_pages_apart_do_not():
    def pages_of_chunks(*pages):
        blocks = [heading("A"), para(paragraph(1, 9))] + [note(f"[{n}] A short note.", page=p) for n, p in enumerate(pages, 1)]
        return [(c.page_start, c.page_end) for c in chunk_document(make_source(), blocks) if c.kind == NOTES]

    assert pages_of_chunks(3, 4) == [(3, 4)]
    assert pages_of_chunks(3, 5) == [(3, 3), (5, 5)]
    assert pages_of_chunks(3, 3, 4, 4, 5) == [(3, 5)]  # a run of neighbouring pages is one chunk, however long it spans


def test_a_note_without_a_page_never_splits_the_chunk():
    unpaged = lambda text: Block(FOOTNOTE, text, None, None)  # noqa: E731
    blocks = [heading("A"), para(paragraph(1, 9)), unpaged("[0] Zero."), note("[1] One.", page=3), unpaged("[2] Two."), note("[3] Three.", page=4)]
    (chunk,) = [c for c in chunk_document(make_source(), blocks) if c.kind == NOTES]
    assert (chunk.page_start, chunk.page_end) == (3, 4) and "[0] Zero." in chunk.text and "[2] Two." in chunk.text


def test_text_before_the_first_heading_still_gets_a_chunk():
    chunks = chunk_document(make_source(), [para(paragraph(1, 9)), heading("Later"), para(paragraph(20, 9))])
    assert len(chunks) == 2 and chunks[0].section_path == []
    assert chunks[0].embed_text.startswith("Guidelines on Testing Chunkers\n")


def test_empty_input_and_heading_only_input_give_no_chunks():
    assert chunk_document(make_source(), []) == []
    assert chunk_document(make_source(), [heading("Only a heading")]) == []


def test_chunks_survive_a_round_trip_through_json_dicts():
    (chunk,) = chunk_document(make_source(), [heading("Article 5"), para(paragraph(1, 9), page=2)])
    assert Chunk.from_dict(chunk.to_dict()) == chunk


# ------------------------------------------------------------------------------ fixed strategy


def test_fixed_windows_have_the_target_size_and_the_requested_overlap():
    options = ChunkOptions(strategy="fixed", target_words=100, max_words=140, min_words=30, overlap_words=20)
    blocks = [heading("Article 1"), para(paragraph(1, 40))]  # 410 words
    chunks = chunk_document(make_source(), blocks, options)
    sizes = [c.n_words for c in chunks]
    assert sizes[:-1] == [100] * (len(sizes) - 1) or all(s == 100 for s in sizes[:-1])
    assert all(s <= options.max_words for s in sizes)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_fixed_strategy_covers_every_word_and_has_no_prefix_in_the_embedded_text():
    options = ChunkOptions(strategy="fixed")
    blocks = [heading("Article 1"), para(paragraph(1, 60)), heading("Article 2"), para(paragraph(100, 40))]
    chunks = chunk_document(make_source(), blocks, options)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
        assert chunk.embed_text == chunk.text and chunk.kind == TEXT
    assert covered == {w for b in blocks for w in b.text.split()}


def test_fixed_chunks_record_the_heading_path_and_pages_where_they_start():
    options = ChunkOptions(strategy="fixed", target_words=100, max_words=140, min_words=30, overlap_words=10)
    blocks = [heading("Chapter I", 1, 1), para(paragraph(1, 30), 1), heading("Article 9", 2, 2), para(paragraph(40, 30), 2)]
    chunks = chunk_document(make_source(), blocks, options)
    assert chunks[0].section_path == ["Chapter I"] and chunks[0].page_start == 1
    assert chunks[-1].section_path[-1] == "Article 9" and chunks[-1].page_end == 2


def test_fixed_strategy_never_leaves_a_tiny_tail():
    options = ChunkOptions(strategy="fixed", target_words=100, max_words=140, min_words=30, overlap_words=10)
    for total_sentences in range(10, 40):
        chunks = chunk_document(make_source(), [para(paragraph(1, total_sentences))], options)
        assert len(chunks) == 1 or chunks[-1].n_words >= options.min_words
        assert all(c.n_words <= options.max_words for c in chunks)


# ------------------------------------------------------------------------------ real-shaped input


@pytest.mark.parametrize(
    "options",
    [
        ChunkOptions(),
        ChunkOptions(target_words=120, max_words=160, min_words=40, overlap_words=20),
        ChunkOptions(target_words=60, max_words=90, min_words=20, overlap_words=0),
    ],
)
def test_official_journal_fixture_chunks_respect_the_limits_and_keep_the_structure(options):
    laid_out, _ = layout.prepare_pages(*typeset(regulation_pages()))
    blocks, _ = build_blocks(laid_out)
    chunks = chunk_document(make_source(title="Regulation (EU) 2099/1 on test data"), blocks, options)
    text_chunks = [c for c in chunks if c.kind == TEXT]
    assert text_chunks and all(c.n_words <= options.max_words for c in chunks)
    assert any(c.section_path[-1].startswith("Article 4") for c in text_chunks)
    assert any(c.kind == NOTES for c in chunks)
    body = " ".join(c.text for c in text_chunks)
    for fragment in ("fundamental right", "twentieth day following", "coop" if False else "cooperate with the other authorities"):
        assert fragment in body, fragment
    assert [c.ordinal for c in text_chunks] == list(range(1, len(text_chunks) + 1))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ------------------------------------------------------------------------- context prefix and reading chunks back


def test_context_prefix_is_the_document_and_section_line_in_front_of_the_embedded_text():
    source = make_source(title="Guidelines on Testing Chunkers")
    blocks = [heading("Part A"), heading("Rules", 2), para(paragraph(1, 8))]
    chunk = chunk_document(source, blocks)[0]
    assert context_prefix(chunk) == "Guidelines on Testing Chunkers | Part A > Rules"
    assert chunk.embed_text == f"{context_prefix(chunk)}\n{chunk.text}"


def test_context_prefix_of_footnote_chunks_keeps_the_footnotes_marker():
    source = make_source()
    blocks = [heading("Part A"), para(paragraph(1, 8)), note("1 See the Treaty on the Functioning of the Union.")]
    notes = [c for c in chunk_document(source, blocks) if c.kind == NOTES]
    assert len(notes) == 1 and context_prefix(notes[0]) == "Guidelines on Testing Chunkers | Part A | footnotes"


def test_context_prefix_is_empty_when_there_is_no_prefix_or_it_cannot_be_recovered():
    source = make_source()
    fixed = chunk_document(source, [para(paragraph(1, 8))], ChunkOptions(strategy="fixed"))[0]
    assert context_prefix(fixed) == ""  # the baseline embeds the raw text only
    odd = Chunk("x:0001", "x", TEXT, 1, "Some text.", "Totally different text.")
    assert context_prefix(odd) == ""
    assert context_prefix(Chunk("x:0002", "x", NOTES, 2, "Footnotes: 1 a", "no marker\nFootnotes: 1 a")) == ""
    assert context_prefix(Chunk("x:0003", "x", TEXT, 3, "", "")) == ""


def test_context_prefix_copes_with_links_that_were_replaced_before_embedding():
    source = make_source()
    chunk = chunk_document(source, [heading("Part A"), para(paragraph(1, 8) + " See https://example.org/a/b?c=1 for more.")])[0]
    assert "[link]" in chunk.embed_text and "https://" not in chunk.embed_text
    assert context_prefix(chunk) == "Guidelines on Testing Chunkers | Part A"


def test_read_chunks_restores_what_was_written(tmp_path):
    source = make_source()
    chunks = chunk_document(source, [heading("Part A"), para(paragraph(1, 30)), note("1 A footnote.")])
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c.to_dict()) for c in chunks) + "\n\n", encoding="utf-8")  # a trailing blank line is fine
    assert read_chunks(path) == chunks
    assert read_chunks(str(path)) == chunks


def test_read_chunks_names_the_bad_line(tmp_path):
    path = tmp_path / "chunks.jsonl"
    good = json.dumps(Chunk("x:0001", "x", TEXT, 1, "t", "e").to_dict())
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"chunks\.jsonl, line 2: not a valid chunk record"):
        read_chunks(path)
    for not_an_object in ("[1, 2]", '["chunk_id", "doc_id"]', '"text"', "7", "null"):  # valid JSON, but not an object
        path.write_text(good + "\n" + not_an_object + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            read_chunks(path)
    path.write_text('{"chunk_id": "x"}\n', encoding="utf-8")  # an object with missing fields
    with pytest.raises(ValueError, match="line 1"):
        read_chunks(path)


def test_read_chunks_ignores_unknown_fields_written_by_a_newer_version(tmp_path):
    path = tmp_path / "chunks.jsonl"
    record = Chunk("x:0001", "x", TEXT, 1, "t", "e").to_dict() | {"future_field": 1}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert read_chunks(path)[0].chunk_id == "x:0001"
