import subprocess
from types import SimpleNamespace

import requests

from reglens import diagnostics
from reglens.config import load_settings
from reglens.diagnostics import (
    Check,
    GeminiAccessError,
    check_disk,
    check_env_file,
    check_gemini,
    check_git,
    check_package,
    check_python,
    check_reachability,
    list_gemini_models,
    main,
    render,
    run_checks,
)

KEY = "SECRET-KEY-123456"


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def models_payload(*names, methods=("generateContent",)):
    return {"models": [{"name": f"models/{n}", "supportedGenerationMethods": list(methods)} for n in names]}


def settings(tmp_path, **env):
    return load_settings(root=tmp_path, environ={"GEMINI_API_KEY": KEY, **env})


# ------------------------------------------------------------------ local checks


def test_python_version_levels():
    assert check_python((3, 13, 1)).status == "OK"
    assert check_python((3, 9, 7)).status == "FAIL"
    assert check_python((3, 14, 0)).status == "WARN"


def test_missing_package_is_a_failure():
    assert check_package("definitely-not-a-real-package-xyz").status == "FAIL"
    assert check_package("requests").status == "OK"


def test_env_file_hints(tmp_path):
    assert check_env_file(load_settings(root=tmp_path, environ={})).status == "WARN"
    (tmp_path / ".env.txt").write_text("x", encoding="utf-8")
    hint = check_env_file(load_settings(root=tmp_path, environ={}))
    assert hint.status == "WARN" and "rename" in hint.detail
    (tmp_path / ".env").write_text("x=1", encoding="utf-8")
    assert check_env_file(load_settings(root=tmp_path, environ={})).status == "OK"


def test_disk_space_thresholds(tmp_path):
    gb = 1024**3
    assert check_disk(tmp_path, usage=lambda _p: SimpleNamespace(free=50 * gb)).status == "OK"
    assert check_disk(tmp_path, usage=lambda _p: SimpleNamespace(free=2 * gb)).status == "WARN"


def test_git_missing_and_present():
    def missing(*_a, **_k):
        raise FileNotFoundError

    assert check_git(missing).status == "WARN"
    ok = lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="git version 2.43.0\n")  # noqa: E731
    result = check_git(ok)
    assert result.status == "OK" and result.detail == "2.43.0"
    broken = lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="")  # noqa: E731
    assert check_git(broken).status == "WARN"
    timeout = subprocess.TimeoutExpired("git", 10)

    def hangs(*_a, **_k):
        raise timeout

    assert check_git(hangs).status == "WARN"


# ------------------------------------------------------------------- network


def test_reachability_reports_each_host():
    def head(url, **_k):
        if "hhs.gov" in url:
            raise requests.ConnectionError("blocked")
        return SimpleNamespace(status_code=200)

    results = check_reachability(head=head)
    assert [c.status for c in results] == ["OK", "OK", "WARN"]
    assert "firewall" in results[2].detail


def test_list_models_follows_pages_and_strips_prefix_and_hides_key_from_url():
    calls = []

    def get(url, headers=None, params=None, timeout=None):
        calls.append((url, headers, params))
        if not params.get("pageToken"):
            return FakeResponse(200, {**models_payload("gemini-3.5-flash"), "nextPageToken": "p2"})
        return FakeResponse(200, models_payload("gemini-3.1-flash-lite", methods=("embedContent",)))

    models = list_gemini_models(KEY, get=get)
    assert [(m.name, m.methods) for m in models] == [
        ("gemini-3.5-flash", ("generateContent",)),
        ("gemini-3.1-flash-lite", ("embedContent",)),
    ]
    assert len(calls) == 2
    for url, headers, params in calls:
        assert KEY not in url and KEY not in str(params)
        assert headers == {"x-goog-api-key": KEY}


def test_http_errors_are_translated_and_never_echo_the_key():
    def make_get(status, message=""):
        body = {"error": {"message": message}} if message else None
        return lambda *_a, **_k: FakeResponse(status, body)

    for status, kind in ((400, "invalid_key"), (401, "invalid_key"), (403, "forbidden"), (429, "rate_limited"), (500, "http")):
        try:
            list_gemini_models(KEY, get=make_get(status, f"bad key {KEY} supplied"))
        except GeminiAccessError as exc:
            assert exc.kind == kind
            assert KEY not in exc.message
        else:
            raise AssertionError("expected GeminiAccessError")


def test_network_failures_are_translated():
    def timeout(*_a, **_k):
        raise requests.Timeout("slow")

    def refused(*_a, **_k):
        raise requests.ConnectionError("no route")

    for getter in (timeout, refused):
        try:
            list_gemini_models(KEY, get=getter)
        except GeminiAccessError as exc:
            assert exc.kind == "network"
        else:
            raise AssertionError("expected GeminiAccessError")


def test_non_json_response_is_handled():
    try:
        list_gemini_models(KEY, get=lambda *_a, **_k: FakeResponse(200, None))
    except GeminiAccessError as exc:
        assert "not JSON" in exc.message
    else:
        raise AssertionError("expected GeminiAccessError")


def test_gemini_check_without_key(tmp_path):
    result = check_gemini(load_settings(root=tmp_path, environ={}), get=lambda *_a, **_k: 1 / 0)
    assert len(result) == 1 and result[0].status == "WARN" and "aistudio.google.com/apikey" in result[0].detail


def test_gemini_check_happy_path_and_missing_models(tmp_path):
    names = ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-embedding-2")

    def ok(*_a, **_k):
        return FakeResponse(200, models_payload(*names))

    checks = check_gemini(settings(tmp_path), get=ok)
    assert [c.status for c in checks] == ["OK", "OK", "OK", "OK"]

    checks = check_gemini(settings(tmp_path, GEMINI_MODEL="gemini-9-ultra"), get=ok)
    primary = next(c for c in checks if c.name == "Gemini primary model")
    assert primary.status == "FAIL" and "gemini-9-ultra" in primary.detail

    checks = check_gemini(settings(tmp_path, GEMINI_FALLBACK_MODEL="gemini-1-tiny"), get=ok)
    assert next(c for c in checks if c.name == "Gemini fallback model").status == "WARN"


def test_gemini_check_failure_message_has_no_key(tmp_path):
    def bad(*_a, **_k):
        return FakeResponse(400, {"error": {"message": f"API key not valid: {KEY}"}})

    checks = check_gemini(settings(tmp_path), get=bad)
    assert checks[0].status == "FAIL" and KEY not in checks[0].detail


# --------------------------------------------------------- orchestration and output


def test_offline_mode_uses_no_network(tmp_path):
    def boom(*_a, **_k):
        raise AssertionError("network used in --offline mode")

    checks = run_checks(settings(tmp_path), online=False, get=boom, head=boom)
    assert any(c.name == "Network checks" and c.status == "SKIP" for c in checks)


def test_rendered_report_never_contains_the_key_or_paths(tmp_path):
    def get(*_a, **_k):
        return FakeResponse(200, models_payload("gemini-3.5-flash", "gemini-3.1-flash-lite"))

    head = lambda *_a, **_k: SimpleNamespace(status_code=200)  # noqa: E731
    text = render(run_checks(settings(tmp_path), get=get, head=head))
    assert KEY not in text
    assert str(tmp_path) not in text
    assert "--- end ---" in text and "0 problem(s)" in text


def test_main_returns_failure_code_for_bad_key(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text(f"GEMINI_API_KEY={KEY}\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(diagnostics.requests, "get", lambda *_a, **_k: FakeResponse(400, {"error": {"message": "API key not valid"}}))
    monkeypatch.setattr(diagnostics.requests, "head", lambda *_a, **_k: SimpleNamespace(status_code=200))
    code = main(["--env-file", str(env)])
    out = capsys.readouterr().out
    assert code == 1 and "[FAIL] Gemini API key" in out and KEY not in out


def test_main_offline_succeeds_when_basics_are_fine(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    code = main(["--offline", "--env-file", str(tmp_path / ".env")])
    assert code == 0
    assert "Network checks" in capsys.readouterr().out


def test_check_dataclass_is_immutable():
    c = Check("x", "OK", "y")
    try:
        c.status = "FAIL"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Check should be frozen")
