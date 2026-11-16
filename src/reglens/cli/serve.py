"""Command line:  python scripts/serve.py  -  start the web interface on this computer."""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

from reglens.config import ConfigError, Settings, load_settings
from reglens.console import make_console_safe, short_duration
from reglens.runtime import build_runtime
from reglens.web.service import FIX_HINT, WebService

LOCAL =("127.0.0.1", "localhost", "::1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="serve", description="Start the RegLens web interface (it runs on this computer only).")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    parser.add_argument("--backend", choices=("chroma", "numpy"), help="vector store to search (default: VECTORSTORE from .env)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    parser.add_argument("--no-warmup", action="store_true", help="do not load the embedding model before starting (the first question is slower)")
    parser.add_argument("--allow-host", action="append", default=[], metavar="NAME", help="also answer requests addressed to this host name (needed when --host is not local)")
    parser.add_argument("--env-file", type=Path, help="use this .env file")
    return parser


def port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def serve_with_uvicorn(app, host: str, port: int, on_started: Callable[[], None] | None = None, on_server: Callable[[object], None] | None = None) -> None:
    """Run the app until Ctrl+C. ``on_started`` is called once the port is open; ``on_server`` receives the server object (tests stop it)."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False, lifespan="off", ws="none")
    server = uvicorn.Server(config)
    if on_server is not None:
        on_server(server)
    if on_started is not None:

        def announce() -> None:
            while not server.started and not server.should_exit:
                time.sleep(0.05)
            if server.started:
                on_started()

        threading.Thread(target=announce, daemon=True).start()
    server.run()


def _log_to_console() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    for name in ("reglens.web",):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader: Callable[..., Settings] = load_settings,
    make_service: Callable[..., WebService] = WebService,
    serve: Callable[..., None] = serve_with_uvicorn,
    free: Callable[[str, int], bool] = port_is_free,
    open_page: Callable[[str], object] = webbrowser.open,
    write: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    make_console_safe()
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        write("Problem: --port must be between 1 and 65535")
        return 2
    try:
        settings = settings_loader(env_file=args.env_file)
    except ConfigError as exc:
        write(f"Problem: {exc}")
        return 2
    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        write("Problem: the web server libraries are not installed - run: pip install -r requirements.txt")
        return 1

    from reglens.web.app import LOCAL_HOSTS, create_app

    write("RegLens web interface")
    runtime_factory = (lambda s: build_runtime(s, backend=args.backend)) if args.backend else build_runtime
    service = make_service(settings, runtime_factory=runtime_factory)
    status = service.status()  # opens the index now, so that problems show up here as well as on the page
    if status["ready"]:
        index, llm = status["index"], status["llm"]
        write(f"Index     {index['passages']:,} passages from {index['documents']} documents ({index['embedding_model']}, {index['vector_store']})")
        write(f"Model     {llm['provider']}/{llm['model']}" if llm["available"] else f"Model     none - {llm['problem']}; answers will list the matching passages only")
        for warning in status["warnings"]:
            write(f"Warning   {warning}")
        if not args.no_warmup:
            write("Loading the embedding model (the first run downloads about 130 MB) ...")
            mark = clock()
            service.warm_up()
            write(f"          ready in {short_duration(clock() - mark)}")
    else:
        write(f"Index     not ready: {status['problem']}")
        write(f"          Run this in another window: {status.get('fix', FIX_HINT)}")
        write("          The page explains this too, and notices when the index is ready - no restart needed.")

    if args.host not in LOCAL:
        write(f"Warning   {args.host} is reachable from other computers: anyone who can open it can spend your language-model quota.")
    if not free(args.host, args.port):
        write(f"Problem: port {args.port} is already in use. Is RegLens already running? Open http://localhost:{args.port}, or choose another port: python scripts/serve.py --port {args.port + 1}")
        return 1

    shown = "localhost" if args.host in LOCAL else args.host
    url = f"http://{shown}:{args.port}"
    app = create_app(service, allowed_hosts=(*LOCAL_HOSTS, *args.allow_host, *([] if args.host in LOCAL or args.host == "0.0.0.0" else [args.host])))
    _log_to_console()

    def started() -> None:
        write(f"\nRegLens is running at {url}   (press Ctrl+C to stop)\n")
        if not args.no_browser:
            open_page(url)

    try:
        serve(app, args.host, args.port, started)
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        write(f"Problem: the server could not start ({exc})")
        return 1
    write("RegLens stopped.")
    return 0
