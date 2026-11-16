"""The HTTP layer, tested against a real server on a free local port (Starlette + uvicorn) with the ``requests`` library."""

from __future__ import annotations

import http.client
import json
import logging

import pytest
import requests
from webfixtures import BREACH, SECRET, LiveServer, make_service, scripted

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from reglens.web.app import MAX_BODY, SECURITY_HEADERS, STATIC_DIR, create_app, hostname  # noqa: E402
from reglens.web.service import Busy, NotFound, NotReady, RequestError, Unavailable  # noqa: E402


@pytest.fixture()
def static_dir(tmp_path):
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "index.html").write_text("<!doctype html><title>RegLens</title><script src=\"/static/app.js\"></script>", encoding="utf-8")
    (folder / "app.js").write_text("console.log('hi')", encoding="utf-8")
    (folder / "app.css").write_text("body{margin:0}", encoding="utf-8")
    (folder / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (folder / "secret.py").write_text("PASSWORD = 1", encoding="utf-8")
    (folder / ".hidden.css").write_text("x{}", encoding="utf-8")
    return folder


@pytest.fixture()
def live(tmp_path, static_dir):
    llm = scripted()
    service = make_service(tmp_path, llm)
    with LiveServer(create_app(service, static_dir=static_dir)) as server:
        server.llm, server.service = llm, service
        yield server


def ask(live, payload=None, **kwargs):
    return requests.post(f"{live.base}/api/ask", json={"question": BREACH} if payload is None else payload, timeout=20, **kwargs)


def error_of(response):
    body = response.json()
    assert list(body) == ["error"] and {"kind", "message"} <= set(body["error"])
    return body["error"]


# ----------------------------------------------------------------------------------------------- the page and its files


def test_the_page_and_its_files_are_served_with_hardening_headers(live):
    page = requests.get(f"{live.base}/", timeout=10)
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html") and "RegLens" in page.text
    for name, value in SECURITY_HEADERS:
        assert page.headers[name] == value
    assert "script-src 'self'" in page.headers["content-security-policy"] and "default-src 'none'" in page.headers["content-security-policy"]
    assert requests.get(f"{live.base}/static/app.js", timeout=10).headers["content-type"].startswith("text/javascript")
    assert requests.get(f"{live.base}/static/app.css", timeout=10).headers["content-type"].startswith("text/css")
    assert requests.get(f"{live.base}/static/favicon.svg", timeout=10).headers["content-type"] == "image/svg+xml"


def test_a_file_that_has_not_changed_is_not_sent_again(live):
    first = requests.get(f"{live.base}/static/app.js", timeout=10)
    again = requests.get(f"{live.base}/static/app.js", headers={"If-None-Match": first.headers["etag"]}, timeout=10)
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == first.headers["etag"]
    assert requests.get(f"{live.base}/static/app.js", headers={"If-None-Match": '"other"'}, timeout=10).status_code == 200


def raw_get(live, path):
    """GET without any tidying of the path (the requests library would resolve ".." before sending it)."""
    connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", "replace")
    finally:
        connection.close()


@pytest.mark.parametrize(
    "path",
    ["/static/secret.py", "/static/.hidden.css", "/static/missing.css", "/static/..", "/static/%2e%2e", "/static/..%2fapp.js", "/static/../app.js", "/static/app.js%00.css", "/static/sub/app.js", "/static//etc/passwd", "/static/%2e%2e%2f%2e%2e%2fetc%2fpasswd"],
)
def test_only_the_files_of_the_static_folder_can_be_fetched(live, path):
    status, body = raw_get(live, path)
    assert status == 404 and json.loads(body)["error"]["kind"] == "not_found"
    assert "PASSWORD" not in body and "root:" not in body


def test_the_favicon_request_is_answered_quietly(live):
    assert requests.get(f"{live.base}/favicon.ico", timeout=10).status_code == 204


def test_unknown_addresses_and_wrong_methods_get_json_errors(live):
    missing = requests.get(f"{live.base}/api/nope", timeout=10)
    assert missing.status_code == 404 and error_of(missing)["kind"] == "not_found"
    wrong = requests.get(f"{live.base}/api/ask", timeout=10)
    assert wrong.status_code == 405 and error_of(wrong)["kind"] == "method_not_allowed" and "POST" in wrong.headers["allow"]
    assert requests.post(f"{live.base}/api/status", json={}, timeout=10).status_code == 405
    assert requests.delete(f"{live.base}/", timeout=10).status_code == 405


# ----------------------------------------------------------------------------------------------- the API


def test_status_and_documents_and_passages(live):
    status = requests.get(f"{live.base}/api/status", timeout=10)
    assert status.status_code == 200 and status.headers["cache-control"] == "no-store"
    body = status.json()
    assert body["ready"] is True and body["index"]["documents"] == 6 and SECRET not in status.text
    documents = requests.get(f"{live.base}/api/documents", timeout=10).json()["documents"]
    assert len(documents) == 6 and {"doc_id", "title", "jurisdiction", "status", "passages"} <= set(documents[0])
    passage = requests.get(f"{live.base}/api/passages/gdpr:0002", timeout=10).json()
    assert passage["passage"]["chunk_id"] == "gdpr:0002" and passage["before"]["chunk_id"] == "gdpr:0001"
    missing = requests.get(f"{live.base}/api/passages/nope:0001", timeout=10)
    assert missing.status_code == 404 and error_of(missing)["kind"] == "not_found"


def test_a_question_is_answered_with_sources_and_checks(live):
    response = ask(live)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["mode"] == "rag" and body["answer"]["text"] == "Notify within 72 hours [S1]." and body["answer"]["cited"] == ["S1"]
    assert body["answer"]["sources"][0]["chunk_id"] == "gdpr:0001" and body["baseline"] is None
    assert SECRET not in response.text


def test_the_comparison_mode_returns_both_answers(live):
    body = ask(live, {"question": BREACH, "mode": "compare"}).json()
    assert body["answer"]["mode"] == "rag" and body["baseline"]["mode"] == "llm_only" and body["baseline"]["text"] == "Within 72 hours."


@pytest.mark.parametrize(
    "payload, field",
    [({"question": ""}, "question"), ({"question": "q", "mode": "x"}, "mode"), ({"question": "q", "options": {"top_k": 99}}, "options.top_k"), ({"question": "q", "filters": {"jurisdiction": "Mars"}}, "filters.jurisdiction")],
)
def test_a_bad_request_gets_a_400_that_names_the_field(live, payload, field):
    response = ask(live, payload)
    assert response.status_code == 400
    assert error_of(response)["kind"] == "bad_request" and error_of(response)["field"] == field
    assert live.llm.requests == []


def test_a_body_that_is_not_json_is_refused(live):
    response = requests.post(f"{live.base}/api/ask", data=b"{not json", headers={"Content-Type": "application/json"}, timeout=10)
    assert response.status_code == 400 and "not valid JSON" in error_of(response)["message"]
    binary = requests.post(f"{live.base}/api/ask", data=b"\xff\xfe\x00", headers={"Content-Type": "application/json"}, timeout=10)
    assert binary.status_code == 400
    assert requests.post(f"{live.base}/api/ask", data=b"[1, 2]", headers={"Content-Type": "application/json"}, timeout=10).status_code == 400  # valid JSON, wrong shape


def test_only_json_is_accepted_for_questions(live):
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        response = requests.post(f"{live.base}/api/ask", data=json.dumps({"question": "q"}), headers={"Content-Type": content_type}, timeout=10)
        assert response.status_code == 415 and error_of(response)["kind"] == "unsupported_media_type"
    assert requests.post(f"{live.base}/api/ask", data=json.dumps({"question": "q"}), timeout=10).status_code == 415  # no content type at all
    ok = requests.post(f"{live.base}/api/ask", data=json.dumps({"question": BREACH}), headers={"Content-Type": "Application/JSON; charset=utf-8"}, timeout=20)
    assert ok.status_code == 200


def test_a_large_body_is_refused_whether_or_not_its_size_is_announced(live):
    big = {"question": "q", "history": [{"question": "a" * 5000, "answer": "b" * 5000}] * 8}
    assert len(json.dumps(big)) > MAX_BODY
    announced = requests.post(f"{live.base}/api/ask", json=big, timeout=10)
    assert announced.status_code == 413 and error_of(announced)["kind"] == "too_large"

    def chunks():
        for _ in range(20):
            yield b" " * 4096  # no Content-Length: sent in chunks

    streamed = requests.post(f"{live.base}/api/ask", data=chunks(), headers={"Content-Type": "application/json"}, timeout=10)
    assert streamed.status_code == 413


# ----------------------------------------------------------------------------------------------- local-server security


def test_requests_for_a_foreign_host_name_are_refused(live):
    for host in ("evil.example", "evil.example:8000", "127.0.0.1.evil.example", "localhost.evil.example:80", ""):
        response = requests.get(f"{live.base}/api/status", headers={"Host": host}, timeout=10)
        assert response.status_code == 403, host
        assert error_of(response)["kind"] == "forbidden"
        assert "Content-Security-Policy" in response.headers
    for host in ("localhost", "localhost:1234", "127.0.0.1", "LOCALHOST:8000", "[::1]:8000"):
        assert requests.get(f"{live.base}/api/status", headers={"Host": host}, timeout=10).status_code == 200, host


def test_a_question_from_another_web_site_is_refused_before_it_costs_anything(live):
    for origin in ("http://evil.example", "https://evil.example:8000", "http://localhost:1", "null"):
        response = ask(live, headers={"Origin": origin})
        assert response.status_code == 403 and error_of(response)["kind"] == "forbidden", origin
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert live.llm.requests == []
    assert ask(live, headers={"Origin": live.base}).status_code == 200  # the page itself
    assert ask(live).status_code == 200  # no Origin header at all: curl, scripts, the tests


def test_other_host_names_can_be_allowed_on_purpose(tmp_path, static_dir):
    service = make_service(tmp_path, scripted())
    with LiveServer(create_app(service, allowed_hosts=["demo.local"], static_dir=static_dir)) as server:
        assert requests.get(f"{server.base}/api/status", headers={"Host": "demo.local:8000"}, timeout=10).status_code == 200
        assert requests.get(f"{server.base}/api/status", headers={"Host": "localhost"}, timeout=10).status_code == 403


@pytest.mark.parametrize(
    "header, expected",
    [("localhost", "localhost"), ("LocalHost:8000", "localhost"), ("127.0.0.1:80", "127.0.0.1"), ("[::1]:8000", "[::1]"), ("[::1]", "[::1]"), ("[::1", "[::1"), ("a:b:c", "a:b:c"), ("", ""), (" example.org : ", "example.org")],
)
def test_the_host_name_is_read_without_its_port(header, expected):
    assert hostname(header) == expected


# ----------------------------------------------------------------------------------------------- failures


class StubService:
    """A service whose methods fail in a chosen way."""

    def __init__(self, error):
        self.error = error

    def _fail(self, *args):
        raise self.error

    status = documents = passage = ask = _fail


@pytest.mark.parametrize(
    "error, status, kind",
    [
        (RequestError("question is empty", field="question"), 400, "bad_request"),
        (NotFound("no such passage"), 404, "not_found"),
        (Busy("Another question is still being answered."), 429, "busy"),
        (Unavailable("A search component failed: x"), 503, "unavailable"),
        (NotReady("There is no index yet."), 503, "not_ready"),
    ],
)
def test_every_service_error_becomes_its_own_status(error, status, kind):
    with LiveServer(create_app(StubService(error))) as server:
        for response in (requests.get(f"{server.base}/api/documents", timeout=10), requests.post(f"{server.base}/api/ask", json={"question": "q"}, timeout=10)):
            assert response.status_code == status
            assert error_of(response)["kind"] == kind and error_of(response)["message"] == str(error)
        if kind == "not_ready":
            assert "build_index" in error_of(response)["fix"]


def test_an_unexpected_bug_gives_a_safe_message_and_a_log_entry(caplog):
    with LiveServer(create_app(StubService(RuntimeError("password=hunter2 in /home/secret"))) ) as server:
        with caplog.at_level(logging.ERROR, logger="reglens.web"):
            response = requests.post(f"{server.base}/api/ask", json={"question": "q"}, timeout=10)
    assert response.status_code == 500 and error_of(response)["kind"] == "internal"
    assert "hunter2" not in response.text and "/home/secret" not in response.text
    assert "unexpected error in POST /api/ask" in caplog.text and "hunter2" in caplog.text  # the traceback is for the person running the server


def test_a_missing_index_is_a_normal_answer_for_the_status_page(tmp_path, static_dir):
    from reglens.web.service import WebService
    from webfixtures import settings_for

    service = WebService(settings_for(tmp_path), retry_seconds=0)  # no index was ever built here
    with LiveServer(create_app(service, static_dir=static_dir)) as server:
        status = requests.get(f"{server.base}/api/status", timeout=10)
        assert status.status_code == 200 and status.json()["ready"] is False and "build_index" in status.json()["fix"]
        refused = requests.post(f"{server.base}/api/ask", json={"question": "q"}, timeout=10)
        assert refused.status_code == 503 and error_of(refused)["kind"] == "not_ready"


def test_the_real_static_folder_has_the_files_the_server_expects():
    assert STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file()
