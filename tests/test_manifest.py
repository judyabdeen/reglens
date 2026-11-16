import csv
import re
from pathlib import Path

import pytest

from reglens.config import PROJECT_ROOT
from reglens.corpus.manifest import REQUIRED_COLUMNS, ManifestError, load_manifest

MANIFEST = PROJECT_ROOT / "data" / "sources.csv"

VALID_ROW = {
    "id": "example-doc",
    "title": "An example document",
    "publisher": "Example Publisher",
    "jurisdiction": "EU",
    "doc_type": "guidelines",
    "status": "final",
    "version_date": "2026-01-01",
    "priority": "1",
    "fetch": "auto",
    "format": "pdf",
    "url": "https://example.org/doc.pdf",
    "landing_url": "https://example.org/doc",
    "filename": "example-doc.pdf",
    "verification": "Fetched OK",
    "notes": "",
}


def write_manifest(tmp_path: Path, rows: list[dict], columns=REQUIRED_COLUMNS) -> Path:
    path = tmp_path / "sources.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# ------------------------------------------------------------- the real manifest


def test_real_manifest_is_valid_and_within_assignment_range():
    sources = load_manifest(MANIFEST)
    assert 20 <= len(sources) <= 40, "the assignment asks for 20 to 40 documents"
    assert sum(s.is_core for s in sources) >= 20


def test_real_manifest_has_the_pieces_the_demo_depends_on():
    ids = {s.id for s in load_manifest(MANIFEST)}
    # The AI Act and its amending regulation must both be present for the stale-knowledge demo.
    assert {"eu-ai-act", "eu-ai-omnibus", "eu-gdpr"} <= ids


def test_draft_and_proposed_documents_are_flagged_in_their_notes():
    for source in load_manifest(MANIFEST):
        if source.status in {"draft", "proposed", "consultation"}:
            assert source.notes, f"{source.id} is {source.status} and needs a caution in notes"


def test_every_manual_source_is_documented_in_manual_downloads():
    text = (PROJECT_ROOT / "data" / "MANUAL_DOWNLOADS.md").read_text(encoding="utf-8")
    manual = [s for s in load_manifest(MANIFEST) if s.is_manual]
    assert manual, "expected at least the three EUR-Lex documents to be manual"
    for source in manual:
        assert source.filename in text, f"{source.filename} missing from MANUAL_DOWNLOADS.md"
        assert source.url in text, f"{source.url} missing from MANUAL_DOWNLOADS.md"


def test_readme_states_the_true_corpus_size():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*Corpus:\*\*\s+(\d+)\s+documents", readme)
    assert match, "README must contain '**Corpus:** N documents'"
    assert int(match.group(1)) == len(load_manifest(MANIFEST))


def test_manifest_file_is_plain_ascii():
    # ASCII keeps the file safe from Windows encoding surprises when opened in Excel or Notepad.
    MANIFEST.read_bytes().decode("ascii")


# -------------------------------------------------------------- validation rules


def test_valid_row_loads(tmp_path):
    sources = load_manifest(write_manifest(tmp_path, [VALID_ROW]))
    assert sources[0].id == "example-doc"
    assert sources[0].priority == 1 and sources[0].is_core and not sources[0].is_manual


def test_missing_file(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.csv")


def test_missing_column(tmp_path):
    columns = [c for c in REQUIRED_COLUMNS if c != "status"]
    with pytest.raises(ManifestError, match="missing column.*status"):
        load_manifest(write_manifest(tmp_path, [VALID_ROW], columns))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"status": "approved"}, "status="),
        ({"fetch": "sometimes"}, "fetch="),
        ({"format": "docx"}, "format="),
        ({"jurisdiction": "Mars"}, "jurisdiction="),
        ({"doc_type": "poem"}, "doc_type="),
        ({"priority": "3"}, "priority="),
        ({"priority": "high"}, "priority="),
        ({"url": "http://example.org/doc.pdf"}, "url must start with https"),
        ({"landing_url": "ftp://example.org"}, "landing_url must start with https"),
        ({"id": "Bad_ID"}, "lowercase"),
        ({"filename": "other.pdf"}, "filename must be"),
        ({"title": ""}, "title is empty"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, change, expected):
    row = {**VALID_ROW, **change}
    with pytest.raises(ManifestError, match=re.escape(expected)):
        load_manifest(write_manifest(tmp_path, [row]))


def test_duplicate_ids_and_urls_are_rejected(tmp_path):
    with pytest.raises(ManifestError) as info:
        load_manifest(write_manifest(tmp_path, [VALID_ROW, VALID_ROW]))
    message = str(info.value)
    assert "duplicate id" in message and "url already used" in message and "filename already used" in message


def test_all_problems_are_reported_together(tmp_path):
    rows = [{**VALID_ROW, "status": "nope"}, {**VALID_ROW, "id": "second", "filename": "second.pdf", "url": "http://x.org/a"}]
    with pytest.raises(ManifestError) as info:
        load_manifest(write_manifest(tmp_path, rows))
    assert "2 problem" in str(info.value)


def test_unquoted_comma_is_reported_clearly(tmp_path):
    path = tmp_path / "sources.csv"
    header = ",".join(REQUIRED_COLUMNS)
    row = "example-doc,Title, with comma,Pub,EU,guidelines,final,2026,1,auto,pdf,https://a.org/x,https://a.org/y,example-doc.pdf,ok,"
    path.write_text(header + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="too many fields"):
        load_manifest(path)


def test_byte_order_mark_is_tolerated(tmp_path):
    path = write_manifest(tmp_path, [VALID_ROW])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_manifest(path)[0].id == "example-doc"
