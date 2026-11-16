"""The HTTP layer: routes, the security checks for a local server, and the static page.

It uses Starlette (the small ASGI toolkit that FastAPI is built on) and contains no decisions of its own: every route
hands its work to ``WebService`` and turns the outcome into JSON. Errors always have the shape
``{"error": {"kind": ..., "message": ..., "field": ...}}`` and never contain a stack trace.

Security, because this server runs on a personal computer next to a private API key:

* it answers only requests whose ``Host`` is a local name (this blocks DNS-rebinding attacks);
* a request that changes anything (POST) must carry JSON, and if the browser names an ``Origin`` it must be this server
  (so another web page cannot make your browser spend your quota);
* a strict Content-Security-Policy (no inline scripts, no outside resources) and the usual hardening headers;
* request bodies are capped at 64 KiB, and static files are served from a fixed folder by a safe-name rule.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from reglens.web.service import FIX_HINT, NotReady, RequestError, ServiceError, WebService

log = logging.getLogger("reglens.web")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_BODY = 64 * 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS = (
    ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
)


class TooLarge(ServiceError):
    status = 413
    kind = "too_large"


class NotJson(ServiceError):
    status = 415
    kind = "unsupported_media_type"


# ------------------------------------------------------------------------------------------------------ security


def hostname(host_header: str) -> str:
    """``localhost:8000`` -> ``localhost``; ``[::1]:8000`` -> ``[::1]``."""
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.rsplit(":", 1)[0].strip() if host.count(":") == 1 else host


class SecurityMiddleware:
    """Pure ASGI middleware: refuse foreign hosts and cross-site writes, and add the hardening headers to every answer."""

    def __init__(self, app, allowed_hosts: Iterable[str] = LOCAL_HOSTS) -> None:
        self.app = app
        self.allowed = frozenset(name.lower() for name in allowed_hosts)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_hardened(message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    response_headers.setdefault(name, value)
            await send(message)

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        host = headers.get("host", "")
        if hostname(host) not in self.allowed:
            await self._refuse(scope, receive, send_hardened, "This address is not allowed. Open RegLens at http://localhost:8000 instead.")
            return
        origin = headers.get("origin")
        if scope["method"] not in ("GET", "HEAD", "OPTIONS") and origin is not None and urlsplit(origin).netloc.lower() != host.lower():
            await self._refuse(scope, receive, send_hardened, "Requests from other web sites are not allowed.")
            return
        await self.app(scope, receive, send_hardened)

    @staticmethod
    async def _refuse(scope, receive, send, message: str) -> None:
        body = JSONResponse({"error": {"kind": "forbidden", "message": message}}, status_code=403, headers={"Cache-Control": "no-store"})
        await body(scope, receive, send)


# ------------------------------------------------------------------------------------------------------ responses


def json_response(data: object, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store"})


def error_response(status: int, kind: str, message: str, *, field: str | None = None, **extra: object) -> JSONResponse:
    error: dict[str, object] = {"kind": kind, "message": message}
    if field:
        error["field"] = field
    error.update(extra)
    return json_response({"error": error}, status)


def guarded(handler: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
    """Turn the service's errors into JSON; anything unexpected becomes a plain 500 and is logged, not shown."""

    async def run(request: Request) -> Response:
        try:
            return await handler(request)
        except NotReady as exc:
            return error_response(exc.status, exc.kind, str(exc), fix=FIX_HINT)
        except ServiceError as exc:
            return error_response(exc.status, exc.kind, str(exc), field=exc.field)
        except Exception:  # noqa: BLE001 - the visitor gets a safe message; the traceback goes to the server window
            log.exception("unexpected error in %s %s", request.method, request.url.path)
            return error_response(500, "internal", "Something went wrong inside RegLens. The server window shows the details.")

    return run


async def read_json(request: Request) -> object:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise NotJson("Send the request as JSON (Content-Type: application/json).")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY:
        raise TooLarge(f"The request is larger than {MAX_BODY // 1024} KiB.")
    parts: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise TooLarge(f"The request is larger than {MAX_BODY // 1024} KiB.")
        parts.append(chunk)
    try:
        return json.loads(b"".join(parts).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RequestError("The request body is not valid JSON.") from exc


def file_response(request: Request, directory: Path, name: str) -> Response:
    """Serve one file of ``directory`` (never anything outside it); a cache check keeps repeat visits cheap."""
    path = directory / name
    safe = SAFE_NAME.fullmatch(name) and path.suffix in STATIC_TYPES and path.is_file() and path.resolve().parent == directory.resolve()
    if not safe:
        return error_response(404, "not_found", "There is no such file.")
    body = path.read_bytes()
    headers = {"ETag": f'"{hashlib.sha256(body).hexdigest()[:20]}"', "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type=STATIC_TYPES[path.suffix], headers=headers)


# ------------------------------------------------------------------------------------------------------ the app


def create_app(service: WebService, *, allowed_hosts: Iterable[str] = LOCAL_HOSTS, static_dir: Path = STATIC_DIR) -> Starlette:
    async def index(request: Request) -> Response:
        return file_response(request, static_dir, "index.html")

    async def static(request: Request) -> Response:
        return file_response(request, static_dir, request.path_params["name"])

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    @guarded
    async def status(request: Request) -> Response:
        return json_response(await run_in_threadpool(service.status))

    @guarded
    async def documents(request: Request) -> Response:
        return json_response(await run_in_threadpool(service.documents))

    @guarded
    async def passage(request: Request) -> Response:
        return json_response(await run_in_threadpool(service.passage, request.path_params["chunk_id"]))

    @guarded
    async def ask(request: Request) -> Response:
        payload = await read_json(request)
        return json_response(await run_in_threadpool(service.ask, payload))

    async def not_found(request: Request, exc: Exception) -> Response:
        return error_response(404, "not_found", "There is no such page.")

    async def not_allowed(request: Request, exc: Exception) -> Response:
        response = error_response(405, "method_not_allowed", "This address does not accept that kind of request.")
        response.headers.update(getattr(exc, "headers", None) or {})
        return response

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/static/{name}", static, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        Route("/api/status", status, methods=["GET"]),
        Route("/api/documents", documents, methods=["GET"]),
        Route("/api/passages/{chunk_id}", passage, methods=["GET"]),
        Route("/api/ask", ask, methods=["POST"]),
    ]
    return Starlette(
        routes=routes,
        middleware=[Middleware(SecurityMiddleware, allowed_hosts=tuple(allowed_hosts))],
        exception_handlers={404: not_found, 405: not_allowed},
    )
