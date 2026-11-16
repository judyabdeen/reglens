"""Downloader tests. A local HTTP server imitates the situations real publishers create."""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from helpers import GOOD_HTML, SAMPLE_LINES, WAF_HTML, make_pdf
from reglens.config import PROJECT_ROOT
from reglens.corpus import download
from reglens.corpus.download import (
    Outcome,
    check_existing,
    download_one,
    find_unrecognized,
    main,
    run,
    select_sources,
    summarize,
    validate_html,
    validate_pdf,
)
from reglens.corpus.manifest import Source

GOOD_PDF = make_pdf(SAMPLE_LINES)
BLANK_PDF = make_pdf(None)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, code, body=b"", ctype="application/octet-stream", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server naming)
        path = self.path
        self.server.hits[path] += 1
        hits = self.server.hits[path]
        if path == "/good.pdf":
            self._send(200, GOOD_PDF, "application/pdf")
        elif path == "/good.html":
            self._send(200, GOOD_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/waf.pdf":
            self._send(202, b"<html>challenge</html>", "text/html", {"x-amzn-waf-action": "challenge"})
        elif path == "/waf.html":
            self._send(200, WAF_HTML.encode(), "text/html")
        elif path == "/missing.pdf":
            self._send(404, b"not here", "text/plain")
        elif path == "/forbidden.pdf":
            self._send(403, b"no", "text/plain")
        elif path == "/flaky.pdf":
            if hits == 1:
                self._send(500, b"oops", "text/plain")
            else:
                self._send(200, GOOD_PDF, "application/pdf")
        elif path == "/busy.pdf":
            self._send(503, b"busy", "text/plain")
        elif path == "/html-as-pdf":
            self._send(200, GOOD_HTML.encode(), "text/html")
        elif path == "/tiny.pdf":
            self._send(200, b"%PDF-1.4 tiny", "application/pdf")
        elif path == "/blank.pdf":
            self._send(200, BLANK_PDF, "application/pdf")
        elif path == "/not-a-pdf.pdf":
            self._send(200, b"x" * 8000, "application/pdf")
        elif path == "/redirect.pdf":
            self._send(302, b"", "text/plain", {"Location": "/good.pdf"})
        elif path == "/cut.pdf":
            # Promise 50 000 bytes, send 100, hang up.
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", "50000")
            self.end_headers()
            self.wfile.write(GOOD_PDF[:100])
            self.wfile.flush()
            self.close_connection = True
        else:
            self._send(404, b"unknown route", "text/plain")


class QuietServer(ThreadingHTTPServer):
    """A client that hangs up early (as the downloader does when it refuses a response) is expected here."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


@pytest.fixture()
def server():
    httpd = QuietServer(("127.0.0.1", 0), Handler)
    httpd.hits = Counter()
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def session():
    s = requests.Session()
    s.trust_env = False  # never route the loopback test server through a proxy
    return s


def base(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def make_source(source_id, url, fmt="pdf", fetch="auto", priority=1) -> Source:
    return Source(
        id=source_id,
        title=source_id,
        publisher="Test",
        jurisdiction="EU",
        doc_type="guidelines",
        status="final",
        version_date="2026",
        priority=priority,
        fetch=fetch,
        format=fmt,
        url=url,
        landing_url="https://example.org/landing",
        filename=f"{source_id}.{fmt}",
        verification="test",
        notes="",
    )


def fast(**overrides):
    return {"retries": 2, "backoff": 0.0, "sleep": lambda _s: None, "timeout": (3.0, 3.0), **overrides}


# ------------------------------------------------------------------ happy paths


def test_downloads_and_validates_a_pdf(server, session, tmp_path):
    src = make_source("good", base(server) + "/good.pdf")
    outcome = download_one(src, tmp_path, session, **fast())
    assert outcome.status == "downloaded" and outcome.usable
    assert (tmp_path / "good.pdf").read_bytes() == GOOD_PDF
    assert outcome.size_bytes == len(GOOD_PDF)
    assert len(outcome.sha256) == 64
    assert not list(tmp_path.glob("*.part"))


def test_downloads_an_html_page(server, session, tmp_path):
    outcome = download_one(make_source("page", base(server) + "/good.html", "html"), tmp_path, session, **fast())
    assert outcome.status == "downloaded"
    assert (tmp_path / "page.html").exists()


def test_second_run_makes_no_request(server, session, tmp_path):
    src = make_source("good", base(server) + "/good.pdf")
    download_one(src, tmp_path, session, **fast())
    again = download_one(src, tmp_path, session, **fast())
    assert again.status == "present" and again.attempts == 0
    assert server.hits["/good.pdf"] == 1


def test_force_downloads_again(server, session, tmp_path):
    src = make_source("good", base(server) + "/good.pdf")
    download_one(src, tmp_path, session, **fast())
    again = download_one(src, tmp_path, session, force=True, **fast())
    assert again.status == "downloaded"
    assert server.hits["/good.pdf"] == 2


def test_redirects_are_followed(server, session, tmp_path):
    outcome = download_one(make_source("r", base(server) + "/redirect.pdf"), tmp_path, session, **fast())
    assert outcome.status == "downloaded"
    assert outcome.final_url.endswith("/good.pdf")


def test_flaky_server_succeeds_after_retry(server, session, tmp_path):
    outcome = download_one(make_source("f", base(server) + "/flaky.pdf"), tmp_path, session, **fast())
    assert outcome.status == "downloaded" and outcome.attempts == 2


def test_invalid_existing_file_is_replaced(server, session, tmp_path):
    (tmp_path / "good.pdf").write_bytes(b"garbage")
    outcome = download_one(make_source("good", base(server) + "/good.pdf"), tmp_path, session, **fast())
    assert outcome.status == "downloaded"
    assert (tmp_path / "good.pdf").read_bytes() == GOOD_PDF


# ---------------------------------------------------------------- failure paths


def test_bot_challenge_is_reported_and_not_retried(server, session, tmp_path):
    outcome = download_one(make_source("w", base(server) + "/waf.pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed"
    assert "bot-detection" in outcome.detail and "by hand" in outcome.detail
    assert server.hits["/waf.pdf"] == 1
    assert not (tmp_path / "w.pdf").exists()


def test_challenge_page_served_with_200_is_rejected(server, session, tmp_path):
    outcome = download_one(make_source("w", base(server) + "/waf.html", "html"), tmp_path, session, **fast())
    assert outcome.status == "failed"
    assert not (tmp_path / "w.html").exists() and not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    ("route", "expected", "hits"),
    [("/missing.pdf", "not found", 1), ("/forbidden.pdf", "access denied", 1)],
)
def test_client_errors_fail_fast(server, session, tmp_path, route, expected, hits):
    outcome = download_one(make_source("x", base(server) + route), tmp_path, session, **fast())
    assert outcome.status == "failed" and expected in outcome.detail
    assert server.hits[route] == hits


def test_persistent_server_error_gives_up_after_retries(server, session, tmp_path):
    outcome = download_one(make_source("b", base(server) + "/busy.pdf"), tmp_path, session, **fast(retries=2))
    assert outcome.status == "failed" and "gave up after 3 attempt" in outcome.detail
    assert server.hits["/busy.pdf"] == 3


def test_web_page_served_instead_of_pdf_is_rejected(server, session, tmp_path):
    outcome = download_one(make_source("h", base(server) + "/html-as-pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed" and "web page instead of a PDF" in outcome.detail


def test_tiny_file_is_rejected(server, session, tmp_path):
    outcome = download_one(make_source("t", base(server) + "/tiny.pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed" and "too small" in outcome.detail


def test_file_without_pdf_header_is_rejected(server, session, tmp_path):
    outcome = download_one(make_source("n", base(server) + "/not-a-pdf.pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed" and "not a PDF" in outcome.detail


def test_pdf_without_text_is_rejected(server, session, tmp_path):
    pytest.importorskip("pypdf")
    outcome = download_one(make_source("b", base(server) + "/blank.pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed" and "no extractable text" in outcome.detail


def test_cut_off_transfer_is_retried_then_reported(server, session, tmp_path):
    outcome = download_one(make_source("c", base(server) + "/cut.pdf"), tmp_path, session, **fast(retries=1))
    assert outcome.status == "failed" and "cut off" in outcome.detail
    assert server.hits["/cut.pdf"] == 2
    assert not list(tmp_path.glob("*.part"))


def test_unreachable_host_is_reported_without_crashing(session, tmp_path):
    # Port 9 on loopback is almost certainly closed; the connection is refused immediately.
    outcome = download_one(make_source("u", "http://127.0.0.1:9/x.pdf"), tmp_path, session, **fast(retries=1))
    assert outcome.status == "failed" and "gave up after 2 attempt" in outcome.detail


def test_proxy_credentials_never_leak_into_error_text():
    exc = requests.exceptions.ProxyError("cannot reach http://user:secret@proxy.example:8080")
    assert "secret" not in download._short(exc)


def test_tls_certificate_problem_fails_fast_with_a_helpful_message(tmp_path):
    class InterceptedSession:
        calls = 0

        def get(self, *_args, **_kwargs):
            type(self).calls += 1
            raise requests.exceptions.SSLError("certificate verify failed: self-signed certificate in chain")

    outcome = download_one(
        make_source("tls", "https://example.org/a.pdf"), tmp_path, InterceptedSession(), **fast(retries=3)
    )
    assert outcome.status == "failed"
    assert "TLS certificate" in outcome.detail and "save the file by hand" in outcome.detail
    assert InterceptedSession.calls == 1  # retrying cannot fix a certificate problem
    assert not list(tmp_path.glob("*.part"))


def test_save_failure_is_reported_not_raised(server, session, tmp_path, monkeypatch):
    # On Windows, replacing a file that is open in a PDF viewer raises PermissionError.
    real_replace = download.os.replace

    def locked(src, dst):
        if str(src).endswith(".part"):
            raise PermissionError("file in use")
        return real_replace(src, dst)

    monkeypatch.setattr(download.os, "replace", locked)
    outcome = download_one(make_source("good", base(server) + "/good.pdf"), tmp_path, session, **fast())
    assert outcome.status == "failed" and "could not save over good.pdf" in outcome.detail
    assert not list(tmp_path.glob("*.part"))


# --------------------------------------------------------------- manual sources


def test_manual_source_missing_then_present(tmp_path):
    src = make_source("eu-law", "https://example.org/law", fetch="manual")
    assert check_existing(src, tmp_path).status == "missing_manual"
    (tmp_path / "eu-law.pdf").write_bytes(GOOD_PDF)
    assert check_existing(src, tmp_path).status == "present_manual"


def test_manual_file_that_is_actually_html_is_flagged(tmp_path):
    src = make_source("eu-law", "https://example.org/law", fetch="manual")
    (tmp_path / "eu-law.pdf").write_bytes(WAF_HTML.encode())
    outcome = check_existing(src, tmp_path)
    assert outcome.status == "failed" and "not usable" in outcome.detail


# ------------------------------------------------------------------ validators


def test_validate_pdf_accepts_a_good_pdf(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(GOOD_PDF)
    assert validate_pdf(path) is None


def test_unreadable_file_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "locked.pdf").write_bytes(GOOD_PDF)

    def locked(*_args, **_kwargs):
        raise PermissionError("locked by another program")

    monkeypatch.setattr(download, "validate_pdf", locked)
    reason = download.validate_file(tmp_path / "locked.pdf", "pdf")
    assert reason and "cannot be read" in reason
    outcome = check_existing(make_source("locked", "https://example.org/law", fetch="manual"), tmp_path)
    assert outcome.status == "failed" and "not usable" in outcome.detail


def make_encrypted_pdf(path):
    """An AES-256 PDF that opens with an empty password, like official documents with editing locked."""
    pytest.importorskip("cryptography")
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(BytesIO(GOOD_PDF)))
    writer.encrypt("", owner_password="owner-only", algorithm="AES-256")
    with path.open("wb") as handle:
        writer.write(handle)
    assert PdfReader(str(path)).is_encrypted  # guard against a test that never reaches the encrypted path
    return path


def test_aes_encrypted_pdf_with_an_empty_password_is_accepted(tmp_path, monkeypatch):
    path = make_encrypted_pdf(tmp_path / "locked.pdf")
    monkeypatch.setattr(download, "MIN_PDF_BYTES", 100)  # re-saving drops the padding the helper adds
    assert validate_pdf(path) is None


def test_missing_cryptography_is_explained_using_the_real_pypdf_error(tmp_path):
    # A separate process with the package blocked, so the error is pypdf's own and its name cannot drift unnoticed.
    path = make_encrypted_pdf(tmp_path / "locked.pdf")
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "sys.modules['cryptography'] = None\n"  # makes `import cryptography` raise ImportError
        "from pathlib import Path\n"
        "from reglens.corpus import download\n"
        "download.MIN_PDF_BYTES = 100\n"
        "print(download.validate_pdf(Path(sys.argv[2])))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(PROJECT_ROOT / "src"), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "DependencyError" in result.stdout and "pip install cryptography" in result.stdout


@pytest.mark.parametrize(
    ("error_name", "fix"),
    [("DependencyError", "pip install cryptography"), ("DeprecationError", 'pip install "pypdf<7"')],
)
def test_known_pypdf_failures_come_with_the_fix(tmp_path, monkeypatch, error_name, fix):
    import pypdf

    (tmp_path / "doc.pdf").write_bytes(GOOD_PDF)
    error_class = type(error_name, (Exception,), {})  # the validator matches on the name

    class BrokenReader:
        def __init__(self, *_args, **_kwargs):
            raise error_class("simulated")

    monkeypatch.setattr(pypdf, "PdfReader", BrokenReader)
    reason = validate_pdf(tmp_path / "doc.pdf")
    assert error_name in reason and fix in reason


def test_unknown_pypdf_failures_get_no_invented_fix(tmp_path, monkeypatch):
    import pypdf

    (tmp_path / "doc.pdf").write_bytes(GOOD_PDF)

    class BrokenReader:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("simulated")

    monkeypatch.setattr(pypdf, "PdfReader", BrokenReader)
    assert validate_pdf(tmp_path / "doc.pdf") == "PDF could not be parsed (RuntimeError)"


def test_font_encoding_errors_from_real_pypdf_do_not_reach_the_console(tmp_path, caplog):
    # The real thing: a font declaring /Encoding /SymbolSetEncoding flooded the console during the first download.
    from pypdf import PdfReader

    path = tmp_path / "symbol-font.pdf"
    path.write_bytes(make_pdf(SAMPLE_LINES, font_extra="/Encoding /SymbolSetEncoding"))
    with caplog.at_level(logging.DEBUG):
        PdfReader(str(path)).pages[0].extract_text()  # control: this is the noise users saw
    if "not implemented yet" not in caplog.text:
        pytest.skip("this pypdf version does not complain about the crafted font")
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert validate_pdf(path) is None
    assert "not implemented yet" not in caplog.text


def test_pdf_validation_runs_inside_the_quiet_zone(tmp_path, monkeypatch):
    entered = []

    @contextlib.contextmanager
    def recorder():
        entered.append(True)
        yield

    monkeypatch.setattr(download, "quiet_pypdf", recorder)
    (tmp_path / "a.pdf").write_bytes(GOOD_PDF)
    assert validate_pdf(tmp_path / "a.pdf") is None
    assert entered == [True]


def test_validate_html_accepts_real_page_and_rejects_error_titles(tmp_path):
    good = tmp_path / "a.html"
    good.write_text(GOOD_HTML, encoding="utf-8")
    assert validate_html(good) is None
    bad = tmp_path / "b.html"
    bad.write_text(GOOD_HTML.replace("Guidance on testing downloaders", "403 Forbidden"), encoding="utf-8")
    assert "error or challenge page" in validate_html(bad)


# ----------------------------------------------------------------- orchestration


def test_select_sources_filters_and_validates_ids():
    sources = [make_source("a", "https://x.org/a"), make_source("b", "https://x.org/b", priority=2)]
    assert [s.id for s in select_sources(sources, only=["b"])] == ["b"]
    assert [s.id for s in select_sources(sources, core_only=True)] == ["a"]
    with pytest.raises(ValueError, match="unknown source id"):
        select_sources(sources, only=["zzz"])


def test_run_mixes_downloads_manual_and_failures_and_summarizes(server, session, tmp_path):
    sources = [
        make_source("ok-pdf", base(server) + "/good.pdf"),
        make_source("ok-page", base(server) + "/good.html", "html", priority=2),
        make_source("blocked", base(server) + "/waf.pdf"),
        make_source("by-hand", "https://example.org/law", fetch="manual"),
    ]
    lines: list[str] = []
    outcomes = run(sources, tmp_path, session=session, delay=0.0, out=lines.append, **fast())
    assert [o.status for o in outcomes] == ["downloaded", "downloaded", "failed", "missing_manual"]
    assert len(lines) == 4 and "MANUAL" in lines[3] and "FAILED" in lines[2]
    summary = summarize(sources, outcomes, tmp_path)
    assert (summary.total, summary.usable, summary.core_usable, summary.optional_usable) == (4, 2, 1, 1)
    assert set(summary.missing_core) == {"blocked", "by-hand"}
    assert not summary.meets_minimum  # fewer than 20 documents


def test_summary_rechecks_disk_for_sources_not_in_this_run(server, session, tmp_path):
    a = make_source("a", base(server) + "/good.pdf")
    b = make_source("b", base(server) + "/good.pdf?second")  # a different file name on disk
    download_one(a, tmp_path, session, **fast())
    outcomes = [Outcome("b", "failed", "x")]
    summary = summarize([a, b], outcomes, tmp_path)
    assert summary.usable == 1 and summary.missing_core == ("b",)


def test_summary_separates_untried_downloads_from_by_hand_files(tmp_path):
    auto = make_source("auto-one", "https://example.org/a.pdf")
    hand = make_source("hand-one", "https://example.org/law", fetch="manual")
    sources = [auto, hand]
    summary = summarize(sources, [], tmp_path)

    before_download: list[str] = []
    download.print_summary(summary, sources, tmp_path, [], out=before_download.append, attempted=False)
    text = "\n".join(before_download)
    assert "hand-one  [core]" in text and "auto-one  [core]" not in text
    assert "1 file(s) have not been downloaded yet" in text

    after_download: list[str] = []
    download.print_summary(summary, sources, tmp_path, [], out=after_download.append, attempted=True)
    text = "\n".join(after_download)
    assert "hand-one  [core]" in text and "auto-one  [core]" in text  # a download that failed is now a by-hand job
    assert "have not been downloaded yet" not in text


def test_find_unrecognized_suggests_renames(tmp_path):
    sources = [make_source("eu-gdpr", "https://x.org/g"), make_source("eu-ai-act", "https://x.org/a")]
    (tmp_path / "eu-gdpr.pdf").write_bytes(b"ok")  # recognised
    (tmp_path / "eu-gdpr (1).pdf").write_bytes(b"dup")  # Windows duplicate
    (tmp_path / "random-notes.txt").write_bytes(b"x")
    (tmp_path / ".gitkeep").write_bytes(b"")
    (tmp_path / "eu-ai-act.pdf.part").write_bytes(b"half")
    assert find_unrecognized(sources, tmp_path) == [("eu-gdpr (1).pdf", "eu-gdpr.pdf"), ("random-notes.txt", None)]


# --------------------------------------------------------------------- the CLI


def write_manifest(tmp_path, rows):
    import csv

    from reglens.corpus.manifest import REQUIRED_COLUMNS

    path = tmp_path / "sources.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for s in rows:
            writer.writerow({**{c: getattr(s, c) for c in REQUIRED_COLUMNS}, "priority": s.priority})
    return path


def test_cli_status_mode_makes_no_network_calls(tmp_path, capsys, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("network used in --status mode")

    monkeypatch.setattr(requests.Session, "get", boom)
    manifest = write_manifest(
        tmp_path,
        [
            make_source("auto-one", "https://example.org/a.pdf"),
            make_source("hand-one", "https://example.org/law", fetch="manual"),
        ],
    )
    raw = tmp_path / "raw"
    code = main(["--manifest", str(manifest), "--raw-dir", str(raw), "--status"])
    out = capsys.readouterr().out
    assert code == 1  # below the 20-document minimum
    assert "MISSING" in out and "MANUAL" in out and "Assignment minimum of 20 documents: NOT MET" in out
    # A manual file is a by-hand job; an automatic one has simply not been tried yet.
    assert "hand-one  [core]" in out and "open  https://example.org/landing" in out
    assert "auto-one  [core]" not in out
    assert "1 file(s) have not been downloaded yet" in out and "python scripts/download_corpus.py" in out


def test_cli_full_run_writes_report_and_merges_on_rerun(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(download, "make_session", lambda: _no_proxy_session())
    rows = [make_source("one", base(server) + "/good.pdf"), make_source("two", base(server) + "/good.html", "html")]
    # The real loader (correctly) insists on https://; the local test server is plain http, so hand the
    # CLI the sources directly. Manifest validation has its own tests in test_manifest.py.
    monkeypatch.setattr(download, "load_manifest", lambda _path: rows)
    manifest = tmp_path / "unused.csv"
    raw, report = tmp_path / "raw", tmp_path / "report.json"
    args = ["--manifest", str(manifest), "--raw-dir", str(raw), "--report", str(report), "--delay", "0", "--retries", "0"]
    main(args)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert {e["id"] for e in data["sources"]} == {"one", "two"}
    assert data["summary"]["usable"] == 2
    main(args + ["--only", "one"])  # partial re-run keeps the other entry
    data = json.loads(report.read_text(encoding="utf-8"))
    assert {e["id"] for e in data["sources"]} == {"one", "two"}
    assert "Usable documents: 2 of 2" in capsys.readouterr().out


def test_cli_rejects_bad_manifest_and_unknown_ids(tmp_path, capsys):
    bad = tmp_path / "sources.csv"
    bad.write_text("id,title\nx,y\n", encoding="utf-8")
    assert main(["--manifest", str(bad)]) == 2
    assert "missing column" in capsys.readouterr().out
    good = write_manifest(tmp_path, [make_source("a", "https://example.org/a.pdf")])
    assert main(["--manifest", str(good), "--only", "nope", "--status"]) == 2
    assert "unknown source id" in capsys.readouterr().out


def _no_proxy_session():
    s = requests.Session()
    s.trust_env = False
    return s
