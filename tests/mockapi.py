"""A tiny scriptable HTTP server for testing the model clients with the real ``requests`` library.

Usage::

    with MockAPI() as api:
        api.reply(200, {"ok": True})
        api.reply(429, {"error": {"message": "slow down"}}, headers={"Retry-After": "7"})
        ... call code that posts to api.url ...
        assert api.requests[0].json == {...}
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Recorded:
    path: str
    headers: dict[str, str]
    raw: bytes
    json: object = None


@dataclass
class Reply:
    status: int = 200
    body: object = None  # dict/list -> JSON; str/bytes -> sent as they are
    headers: dict[str, str] = field(default_factory=dict)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # keep the test output clean
        return

    def do_POST(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        server: MockAPI = self.server.owner  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = None
        server.requests.append(Recorded(self.path, {k.lower(): v for k, v in self.headers.items()}, raw, parsed))
        with server.lock:
            reply = server.script.pop(0) if server.script else Reply(500, {"error": {"message": "no reply scripted"}})
        body = reply.body
        if isinstance(body, (dict, list)):
            payload, ctype = json.dumps(body).encode("utf-8"), "application/json"
        elif isinstance(body, str):
            payload, ctype = body.encode("utf-8"), "text/html"
        else:
            payload, ctype = (body or b""), "application/octet-stream"
        self.send_response(reply.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)


class MockAPI:
    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.script: list[Reply] = []
        self.lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)  # the default 0.5 s makes every shutdown slow

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def reply(self, status: int = 200, body: object = None, headers: dict[str, str] | None = None, times: int = 1) -> None:
        with self.lock:
            self.script.extend(Reply(status, body, dict(headers or {})) for _ in range(times))

    def __enter__(self) -> MockAPI:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
