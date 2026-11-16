"""Environment and credential checks behind ``python scripts/check_setup.py``.

Nothing here prints or stores an API key. The only network calls are:
  * HEAD requests to a few publisher websites (can this machine reach them?), and
  * one read-only "list models" request to the Gemini API, which also proves the key works.
Output is plain ASCII so it prints correctly in any Windows console.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from reglens.config import ConfigError, Settings, load_settings

MIN_PYTHON = (3, 10)
REQUIRED_PACKAGES = ("requests", "python-dotenv", "pypdf", "cryptography", "beautifulsoup4")
MIN_FREE_GB = 5.0
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
REACHABILITY_URLS = (
    "https://nvlpubs.nist.gov/",
    "https://www.edpb.europa.eu/",
    "https://www.hhs.gov/",
)

_TAGS = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # OK | WARN | FAIL | SKIP
    detail: str = ""


@dataclass(frozen=True)
class GeminiModel:
    name: str  # without the "models/" prefix
    methods: tuple[str, ...]


class GeminiAccessError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind  # invalid_key | forbidden | rate_limited | network | http
        self.message = message


# ------------------------------------------------------------------------------ local checks


def check_python(version_info: Sequence[int] | None = None) -> Check:
    v = tuple(version_info) if version_info is not None else tuple(sys.version_info[:3])
    text = ".".join(str(part) for part in v[:3])
    if (v[0], v[1]) < MIN_PYTHON:
        return Check("Python", "FAIL", f"{text} - RegLens needs Python 3.10 or newer")
    if (v[0], v[1]) >= (3, 14):
        return Check(
            "Python",
            "WARN",
            f"{text} - fine for now, but some ML packages used later may lack builds for this version; "
            "if an install fails, use Python 3.11 or 3.12",
        )
    return Check("Python", "OK", text)


def check_package(dist: str) -> Check:
    try:
        return Check(dist, "OK", metadata.version(dist))
    except metadata.PackageNotFoundError:
        return Check(dist, "FAIL", "not installed - run: pip install -r requirements.txt")


def check_env_file(settings: Settings) -> Check:
    if settings.env_file_found:
        return Check(".env file", "OK", "found")
    if (settings.root / ".env.txt").is_file():
        return Check(".env file", "WARN", "found .env.txt - Windows added .txt; rename it to exactly .env")
    return Check(".env file", "WARN", "not found - copy .env.example to .env, then add your API key")


def check_disk(path: Path, min_gb: float = MIN_FREE_GB, usage: Callable = shutil.disk_usage) -> Check:
    try:
        free_gb = usage(path).free / (1024**3)
    except OSError as exc:
        return Check("Disk space", "WARN", f"could not check ({type(exc).__name__})")
    if free_gb < min_gb:
        return Check("Disk space", "WARN", f"{free_gb:.1f} GB free - plan for about {min_gb:.0f} GB (models, index, PDFs)")
    return Check("Disk space", "OK", f"{free_gb:.0f} GB free")


def check_git(run: Callable = subprocess.run) -> Check:
    try:
        result = run(["git", "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return Check("git", "WARN", "not found - needed for the GitHub step; install from https://git-scm.com/downloads")
    version = (result.stdout or "").strip()
    if result.returncode != 0 or not version:
        return Check("git", "WARN", "git did not run correctly - reinstall from https://git-scm.com/downloads")
    return Check("git", "OK", version.removeprefix("git version "))


# ------------------------------------------------------------------------------ network checks


def check_reachability(
    urls: Sequence[str] = REACHABILITY_URLS, *, head: Callable = requests.head, timeout: float = 10.0
) -> list[Check]:
    checks = []
    for url in urls:
        host = urlparse(url).netloc
        try:
            response = head(url, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            checks.append(
                Check(host, "WARN", f"not reachable ({type(exc).__name__}) - check internet, VPN, firewall or proxy")
            )
        else:
            checks.append(Check(host, "OK", f"reachable (HTTP {response.status_code})"))
    return checks


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _describe_http_error(response: requests.Response, api_key: str) -> tuple[str, str]:
    code = response.status_code
    message = ""
    with contextlib.suppress(Exception):
        message = str(response.json().get("error", {}).get("message", ""))
    message = _scrub(" ".join(message.split()), api_key)[:160]
    if code in (400, 401):
        return "invalid_key", f"key rejected (HTTP {code}): {message or 'check that you copied the whole key'}"
    if code == 403:
        return "forbidden", f"access denied (HTTP 403): {message or 'the key may be restricted, or the API is unavailable in your region'}"
    if code == 429:
        return "rate_limited", "rate limited (HTTP 429) - wait a minute and run the check again"
    return "http", f"unexpected response (HTTP {code}) {message}".strip()


def list_gemini_models(
    api_key: str, *, get: Callable = requests.get, timeout: float = 15.0, max_pages: int = 5
) -> list[GeminiModel]:
    """Ask the Gemini API which models this key can see. The key goes in a header, never in the URL."""
    models: list[GeminiModel] = []
    page_token: str | None = None
    for _ in range(max_pages):
        params: dict[str, str | int] = {"pageSize": 200}
        if page_token:
            params["pageToken"] = page_token
        try:
            response = get(GEMINI_MODELS_URL, headers={"x-goog-api-key": api_key}, params=params, timeout=timeout)
        except requests.Timeout as exc:
            raise GeminiAccessError("network", "timed out reaching generativelanguage.googleapis.com") from exc
        except requests.RequestException as exc:
            raise GeminiAccessError(
                "network",
                f"could not reach generativelanguage.googleapis.com ({type(exc).__name__}) - check internet, VPN, firewall or proxy",
            ) from exc
        if response.status_code != 200:
            raise GeminiAccessError(*_describe_http_error(response, api_key))
        try:
            data = response.json()
        except ValueError as exc:
            raise GeminiAccessError("http", "the API answered with something that is not JSON") from exc
        for item in data.get("models", []):
            name = str(item.get("name", "")).removeprefix("models/")
            if name:
                models.append(GeminiModel(name, tuple(item.get("supportedGenerationMethods") or ())))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return models


def check_gemini(settings: Settings, *, get: Callable = requests.get) -> list[Check]:
    if not settings.gemini_api_key:
        return [
            Check(
                "Gemini API key",
                "WARN",
                "GEMINI_API_KEY is empty - create a free key at https://aistudio.google.com/apikey and put it in .env",
            )
        ]
    try:
        models = list_gemini_models(settings.gemini_api_key, get=get)
    except GeminiAccessError as exc:
        return [Check("Gemini API key", "FAIL", _scrub(exc.message, settings.gemini_api_key))]

    usable = {m.name for m in models if "generateContent" in m.methods}
    checks = [Check("Gemini API key", "OK", f"accepted; {len(models)} models visible to this key")]
    wanted = [("Gemini primary model", settings.gemini_model, "FAIL")]
    for number, name in enumerate(settings.gemini_fallbacks, start=1):
        wanted.append(("Gemini fallback model" + (f" {number}" if number > 1 else ""), name, "WARN"))
    for label, name, missing_status in wanted:
        if name in usable:
            checks.append(Check(label, "OK", name))
        else:
            checks.append(
                Check(
                    label,
                    missing_status,
                    f"{name!r} is not offered to this key - choose one from the list below and set it in .env",
                )
            )
    listing = sorted(n for n in usable if n.startswith("gemini-"))
    checks.append(Check("Gemini models you can use", "OK", ", ".join(listing[:20]) or "(none listed)"))
    return checks


# ------------------------------------------------------------------------------ orchestration


def run_checks(
    settings: Settings,
    *,
    online: bool = True,
    get: Callable = requests.get,
    head: Callable = requests.head,
) -> list[Check]:
    checks = [check_python()]
    checks.extend(check_package(p) for p in REQUIRED_PACKAGES)
    checks.append(check_env_file(settings))
    checks.append(check_disk(settings.root))
    checks.append(check_git())
    if online:
        checks.extend(check_reachability(head=head))
        checks.extend(check_gemini(settings, get=get))
    else:
        checks.append(Check("Network checks", "SKIP", "--offline given"))
    return checks


def render(checks: Sequence[Check], title: str = "RegLens setup check") -> str:
    lines = [title, "=" * len(title), ""]
    lines.extend(f"{_TAGS[c.status]} {c.name}: {c.detail}".rstrip(": ").rstrip() for c in checks)
    problems = sum(c.status == "FAIL" for c in checks)
    warnings = sum(c.status == "WARN" for c in checks)
    lines += ["", f"{problems} problem(s), {warnings} warning(s).", ""]
    lines.append("--- copy everything between these lines and send it to me (it contains no secrets) ---")
    lines.append(f"python={platform.python_version()} os={platform.system()} {platform.release()} machine={platform.machine()}")
    lines.extend(f"{c.status}: {c.name}: {c.detail}" for c in checks)
    lines.append("--- end ---")
    return "\n".join(lines)


def _make_console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_setup", description="Check that this machine is ready to run RegLens.")
    parser.add_argument("--offline", action="store_true", help="skip every check that needs the internet")
    parser.add_argument("--env-file", type=Path, help="use this .env file instead of the one in the project folder")
    args = parser.parse_args(argv)
    _make_console_safe()
    try:
        settings = load_settings(env_file=args.env_file)
    except ConfigError as exc:
        print(f"[FAIL] .env: {exc}")
        return 1
    checks = run_checks(settings, online=not args.offline)
    print(render(checks))
    return 1 if any(c.status == "FAIL" for c in checks) else 0
