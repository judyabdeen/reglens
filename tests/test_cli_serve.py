from __future__ import annotations

import socket
import sys

import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from reglens.cli import serve as serve_cli  # noqa: E402
from reglens.config import ConfigError, load_settings  # noqa: E402


class StubService:
    """Stands in for WebService: reports a ready index (or a problem) and records the calls the command makes."""

    def __init__(self, settings, runtime_factory=None, ready=True, llm=True, warnings=()):
        self.settings, self.factory, self.ready, self.llm, self.warnings = settings, runtime_factory, ready, llm, warnings
        self.warmed = 0

    def status(self):
        if not self.ready:
            return {"ready": False, "problem": "There is no index yet. Build it first.", "fix": "python scripts/build_index.py"}
        llm = {"available": True, "provider": "gemini", "model": "gemini-test"} if self.llm else {"available": False, "problem": "GEMINI_API_KEY is empty"}
        index = {"passages": 3813, "documents": 34, "embedding_model": "bge-small", "vector_store": "chroma"}
        return {"ready": True, "index": index, "llm": llm, "warnings": list(self.warnings)}

    def warm_up(self):
        self.warmed += 1


class Run:
    """Runs the command with every outside effect replaced, and keeps what happened."""

    def __init__(self, tmp_path, *argv, ready=True, llm=True, warnings=(), free=True, serve_effect=None, loader=None):
        self.lines, self.opened, self.served, self.service = [], [], [], None

        def make_service(settings, runtime_factory=None):
            self.service = StubService(settings, runtime_factory, ready, llm, warnings)
            return self.service

        def serve(app, host, port, on_started=None):
            self.served.append((app, host, port))
            if on_started:
                on_started()
            if serve_effect:
                raise serve_effect

        ticks = iter(range(0, 1000, 2))
        self.code = serve_cli.main(
            list(argv),
            settings_loader=loader or (lambda env_file=None: load_settings(root=tmp_path, environ={"GEMINI_API_KEY": ""})),
            make_service=make_service, serve=serve, free=lambda host, port: free, open_page=self.opened.append,
            write=self.lines.append, clock=lambda: next(ticks),
        )  # fmt: skip
        self.out = "\n".join(self.lines)


def test_it_starts_the_server_and_opens_the_page(tmp_path):
    run = Run(tmp_path)
    assert run.code == 0, run.out
    assert "Index     3,813 passages from 34 documents (bge-small, chroma)" in run.out and "Model     gemini/gemini-test" in run.out
    assert "ready in 2.0 s" in run.out and run.service.warmed == 1
    assert "RegLens is running at http://localhost:8000" in run.out and run.opened == ["http://localhost:8000"]
    assert run.served[0][1:] == ("127.0.0.1", 8000) and "RegLens stopped." in run.out


def test_port_host_and_browser_options(tmp_path):
    run = Run(tmp_path, "--port", "8123", "--no-browser", "--no-warmup")
    assert run.served[0][1:] == ("127.0.0.1", 8123) and run.opened == [] and run.service.warmed == 0
    assert "http://localhost:8123" in run.out and "Loading the embedding model" not in run.out


def test_the_backend_option_reaches_the_runtime_factory(tmp_path):
    run = Run(tmp_path, "--backend", "numpy", "--no-browser")
    assert callable(run.service.factory) and run.service.factory.__name__ == "<lambda>"
    assert Run(tmp_path, "--no-browser").service.factory.__name__ == "build_runtime"


def test_a_missing_index_still_starts_the_page_that_explains_it(tmp_path):
    run = Run(tmp_path, ready=False)
    assert run.code == 0 and "Index     not ready: There is no index yet." in run.out
    assert "Run this in another window: python scripts/build_index.py" in run.out and "no restart needed" in run.out
    assert run.service.warmed == 0 and run.served  # nothing to warm up, but the server runs so the page can show the problem


def test_a_missing_language_model_is_said_plainly(tmp_path):
    run = Run(tmp_path, llm=False)
    assert "Model     none - GEMINI_API_KEY is empty; answers will list the matching passages only" in run.out


def test_warnings_about_the_index_are_shown(tmp_path):
    assert "Warning   The index is older than the chunks." in Run(tmp_path, warnings=["The index is older than the chunks."]).out


def test_a_port_that_is_taken_is_explained_before_anything_starts(tmp_path):
    run = Run(tmp_path, "--port", "8000", free=False)
    assert run.code == 1 and run.served == [] and "port 8000 is already in use" in run.out and "--port 8001" in run.out


def test_a_host_that_other_computers_can_reach_is_called_out(tmp_path):
    run = Run(tmp_path, "--host", "0.0.0.0", "--allow-host", "lab-pc")
    assert "Warning   0.0.0.0 is reachable from other computers" in run.out
    assert "http://0.0.0.0:8000" in run.out and run.served[0][1] == "0.0.0.0"
    assert "Warning" not in Run(tmp_path, "--host", "localhost").out


def test_the_host_names_the_server_answers_to(tmp_path):
    def allowed(*argv):
        app = Run(tmp_path, *argv).served[0][0]
        return set(app.user_middleware[0].kwargs["allowed_hosts"])

    assert allowed() == {"localhost", "127.0.0.1", "[::1]"}
    assert allowed("--host", "192.168.1.20") == {"localhost", "127.0.0.1", "[::1]", "192.168.1.20"}
    assert allowed("--host", "0.0.0.0", "--allow-host", "lab-pc") == {"localhost", "127.0.0.1", "[::1]", "lab-pc"}


@pytest.mark.parametrize("port", ["0", "70000", "-5"])
def test_an_impossible_port_is_refused(tmp_path, port):
    run = Run(tmp_path, "--port", port)
    assert run.code == 2 and "--port must be between 1 and 65535" in run.out and run.served == []


def test_a_bad_configuration_is_reported_in_plain_words(tmp_path):
    def broken(env_file=None):
        raise ConfigError("EMBEDDING_MODEL must not be empty")

    run = Run(tmp_path, loader=broken)
    assert run.code == 2 and run.out == "Problem: EMBEDDING_MODEL must not be empty"


def test_missing_web_libraries_are_explained(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # makes "import uvicorn" fail
    run = Run(tmp_path)
    assert run.code == 1 and "pip install -r requirements.txt" in run.out and run.served == []


def test_ctrl_c_stops_it_cleanly_and_a_failing_server_is_reported(tmp_path):
    assert Run(tmp_path, serve_effect=KeyboardInterrupt()).code == 0
    failed = Run(tmp_path, serve_effect=OSError("address already in use"))
    assert failed.code == 1 and "the server could not start (address already in use)" in failed.out


def test_port_is_free_sees_a_port_that_is_in_use():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        assert serve_cli.port_is_free("127.0.0.1", port) is False
    assert serve_cli.port_is_free("127.0.0.1", port) is True


def test_uvicorn_runner_reports_when_the_server_is_up_and_stops_when_asked():
    import threading

    import requests

    from reglens.web.app import create_app

    started, servers = threading.Event(), []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    thread = threading.Thread(target=serve_cli.serve_with_uvicorn, args=(create_app(StubService(None)), "127.0.0.1", port, started.set, servers.append), daemon=True)
    thread.start()
    try:
        assert started.wait(10)
        assert requests.get(f"http://127.0.0.1:{port}/api/status", timeout=10).status_code == 200
    finally:
        servers[0].should_exit = True
        thread.join(10)
    assert not thread.is_alive() and serve_cli.port_is_free("127.0.0.1", port)
