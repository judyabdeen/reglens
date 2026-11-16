"""Download and validate the corpus listed in ``data/sources.csv``.

Design notes
------------
* One document at a time with a pause between requests - polite to publisher servers.
* Each download is streamed to ``<name>.part``, validated, then renamed. A half-finished
  or wrong file (an HTML error page saved as .pdf, a scanned PDF, a bot-challenge page)
  never looks like a good document.
* No attempt is made to get around bot protection. If a site asks for a browser
  challenge, the script says so and the file is saved by hand
  (see ``data/MANUAL_DOWNLOADS.md``).
* Output is plain ASCII so it prints correctly in any Windows console.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from reglens.config import PROJECT_ROOT
from reglens.corpus.manifest import ManifestError, Source, load_manifest
from reglens.pdfutil import describe_pdf_error, quiet_pypdf

USER_AGENT = "Mozilla/5.0 (compatible; RegLens-Corpus-Downloader/0.1; educational use)"
MIN_DOCS = 20  # the assignment asks for 20 to 40 documents
MAX_BYTES = 150 * 1024 * 1024
MIN_PDF_BYTES = 5_000
MIN_HTML_BYTES = 3_000
MIN_HTML_TEXT_CHARS = 1_000
PDF_TEXT_PROBE_PAGES = 8
PDF_MIN_TEXT_CHARS = 200

CHALLENGE_MARKERS = (
    "awswaf",
    "aws-waf",
    "cf-chl",
    "captcha",
    "just a moment",
    "enable javascript",
    "attention required",
    "access denied",
    "request unsuccessful",
)
BAD_TITLE_PARTS = (
    "access denied",
    "just a moment",
    "attention required",
    "403 forbidden",
    "404 not found",
    "page not found",
    "robot check",
    "request unsuccessful",
)

USABLE = frozenset({"downloaded", "present", "present_manual"})
_TAGS = {
    "downloaded": "OK",
    "present": "PRESENT",
    "present_manual": "PRESENT",
    "failed": "FAILED",
    "missing": "MISSING",
    "missing_manual": "MANUAL",
}


# --------------------------------------------------------------------------- results


@dataclass
class Outcome:
    id: str
    status: str  # downloaded | present | present_manual | failed | missing | missing_manual
    detail: str = ""
    http_status: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    final_url: str | None = None
    attempts: int = 0  # 0 means no network request was made
    at: str = ""

    @property
    def usable(self) -> bool:
        return self.status in USABLE


@dataclass(frozen=True)
class Summary:
    total: int
    usable: int
    core_total: int
    core_usable: int
    optional_total: int
    optional_usable: int
    missing_core: tuple[str, ...]
    missing_optional: tuple[str, ...]

    @property
    def meets_minimum(self) -> bool:
        return self.usable >= MIN_DOCS


# ------------------------------------------------------------------------ small helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_CREDENTIALS_IN_URL = re.compile(r"(?<=://)[^/\s@]+@")


def _short(exc: BaseException, limit: int = 140) -> str:
    """One-line error text that never contains credentials from a proxy URL."""
    text = _CREDENTIALS_IN_URL.sub("***@", f"{type(exc).__name__}: {exc}")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{max(1, n // 1024)} KB"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
    return session


# --------------------------------------------------------------------------- validation


def _visible_text(html_lower: str) -> str:
    stripped = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html_lower)
    stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
    return " ".join(stripped.split())


def validate_pdf(path: Path, *, deep: bool = True) -> str | None:
    """Return None if ``path`` looks like a usable PDF, otherwise a short reason."""
    size = path.stat().st_size
    if size < MIN_PDF_BYTES:
        return f"file is only {size} bytes - too small to be the document"
    with path.open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF-" not in head:
        return "not a PDF (no %PDF header) - probably an error, login or consent page"
    if not deep:
        return None
    try:
        from pypdf import PdfReader
    except ImportError:  # pypdf is in requirements.txt; skip the deep check if it is missing
        return None
    try:
        with quiet_pypdf():
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                return "PDF is password-protected"
            pages = len(reader.pages)
            if pages == 0:
                return "PDF has no pages"
            text = ""
            for index in range(min(pages, PDF_TEXT_PROBE_PAGES)):
                text += reader.pages[index].extract_text() or ""
                if len(text.strip()) >= PDF_MIN_TEXT_CHARS:
                    break
    except Exception as exc:  # pypdf raises many different exception types on damaged files
        return describe_pdf_error(exc)
    if len(text.strip()) < PDF_MIN_TEXT_CHARS:
        return "PDF has no extractable text in its first pages (scanned image?)"
    return None


def validate_html(path: Path) -> str | None:
    """Return None if ``path`` looks like a real, readable web page, otherwise a short reason."""
    size = path.stat().st_size
    if size < MIN_HTML_BYTES:
        return f"page is only {size} bytes - too small to be the document"
    text = path.read_bytes().decode("utf-8", errors="replace")
    lower = text.lower()
    if "<html" not in lower and "<body" not in lower:
        return "not an HTML page"
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    title = " ".join(title_match.group(1).split()) if title_match else ""
    if any(part in title.lower() for part in BAD_TITLE_PARTS):
        return f"error or challenge page (title: {title[:60]!r})"
    visible = _visible_text(lower)
    if len(visible) < MIN_HTML_TEXT_CHARS:
        if any(marker in lower for marker in CHALLENGE_MARKERS):
            return "bot-detection or challenge page - save the page by hand"
        return "page has almost no text (rendered by JavaScript, or an error page)"
    return None


def validate_file(path: Path, fmt: str, *, deep: bool = True) -> str | None:
    try:
        return validate_pdf(path, deep=deep) if fmt == "pdf" else validate_html(path)
    except OSError as exc:
        return f"file cannot be read ({type(exc).__name__}) - close it if another program has it open"


# ---------------------------------------------------------------------------- downloading


class _TooLarge(Exception):
    pass


class _Incomplete(Exception):
    pass


def _classify(resp: requests.Response) -> tuple[str, str]:
    """Return ("ok" | "retry" | "fatal", reason)."""
    code = resp.status_code
    waf = resp.headers.get("x-amzn-waf-action", "").lower()
    if code == 202 or waf in ("challenge", "captcha"):
        return "fatal", (
            f"the site answered with a bot-detection challenge (HTTP {code}); "
            "it needs a real browser - save the file by hand"
        )
    if code == 200:
        return "ok", ""
    if code == 429 or 500 <= code < 600:
        return "retry", f"server busy or failing (HTTP {code})"
    if code == 403:
        return "fatal", "access denied (HTTP 403) - the site blocks scripts; save the file by hand"
    if code in (404, 410):
        return "fatal", f"not found (HTTP {code}) - the publisher may have moved the file"
    return "fatal", f"unexpected HTTP {code}"


def _content_type_problem(content_type: str, fmt: str) -> str | None:
    if fmt == "pdf" and "text/html" in content_type:
        return "the server returned a web page instead of a PDF (consent wall, login or error page)"
    if fmt == "html" and "application/pdf" in content_type:
        return "expected a web page but the server returned a PDF - fix the format column in sources.csv"
    return None


def _wait_seconds(backoff: float, attempt: int, resp: requests.Response | None = None) -> float:
    delay = backoff * (2 ** (attempt - 1))
    if resp is not None:
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.isdigit():
            delay = max(delay, float(retry_after))
    return min(delay, 30.0)


def _stream_to_file(resp: requests.Response, part: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with part.open("wb") as handle:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_BYTES:
                raise _TooLarge
            handle.write(chunk)
            digest.update(chunk)
    declared = resp.headers.get("Content-Length", "")
    if declared.isdigit() and not resp.headers.get("Content-Encoding") and total < int(declared):
        raise _Incomplete(f"received {total} of {declared} bytes")
    return total, digest.hexdigest()


def _failed(source: Source, reason: str, http_status: int | None, attempts: int, final_url: str | None = None) -> Outcome:
    return Outcome(source.id, "failed", reason, http_status, None, None, final_url, attempts, _utc_now())


def download_one(
    source: Source,
    raw_dir: Path,
    session: requests.Session,
    *,
    force: bool = False,
    retries: int = 2,
    timeout: tuple[float, float] = (10.0, 60.0),
    backoff: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Outcome:
    """Fetch one auto-download source into ``raw_dir`` and validate it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / source.filename
    part = dest.with_name(dest.name + ".part")
    _unlink(part)  # leftover from an interrupted earlier run

    if dest.exists() and not force:
        if validate_file(dest, source.format) is None:
            return Outcome(
                source.id, "present", "already downloaded", None, dest.stat().st_size, _sha256_file(dest), None, 0, _utc_now()
            )
        # A bad file is replaced rather than trusted.

    last_error = "no attempt made"
    status_code: int | None = None
    attempts = 0
    for attempt in range(1, retries + 2):
        attempts = attempt
        try:
            resp = session.get(source.url, stream=True, timeout=timeout, allow_redirects=True)
        except requests.exceptions.SSLError:
            # Retrying cannot fix a certificate problem. On university and company networks this usually
            # means the network is inspecting HTTPS traffic with its own certificate.
            return _failed(
                source,
                "TLS certificate problem - your network may be intercepting HTTPS (common on university or "
                "company networks); try another network, or save the file by hand",
                None,
                attempts,
            )
        except requests.RequestException as exc:
            last_error = _short(exc)
            if attempt <= retries:
                sleep(_wait_seconds(backoff, attempt))
                continue
            break

        with contextlib.closing(resp):
            status_code = resp.status_code
            final_url = resp.url
            kind, why = _classify(resp)
            if kind == "fatal":
                return _failed(source, why, status_code, attempts, final_url)
            if kind == "retry":
                last_error = why
                if attempt <= retries:
                    sleep(_wait_seconds(backoff, attempt, resp))
                    continue
                break
            problem = _content_type_problem(resp.headers.get("Content-Type", "").lower(), source.format)
            if problem:
                return _failed(source, problem, status_code, attempts, final_url)
            try:
                size, digest = _stream_to_file(resp, part)
            except _TooLarge:
                _unlink(part)
                return _failed(
                    source, f"larger than {MAX_BYTES // (1024 * 1024)} MB - refusing to save it", status_code, attempts, final_url
                )
            except (requests.RequestException, _Incomplete, OSError) as exc:
                _unlink(part)
                last_error = f"download was cut off ({_short(exc)})"
                if attempt <= retries:
                    sleep(_wait_seconds(backoff, attempt))
                    continue
                break

        reason = validate_file(part, source.format)
        if reason:
            _unlink(part)
            return _failed(source, reason, status_code, attempts, final_url)
        try:
            os.replace(part, dest)
        except OSError as exc:  # on Windows this happens when the old file is open in a PDF viewer
            _unlink(part)
            return _failed(
                source,
                f"could not save over {dest.name} ({type(exc).__name__}) - close it if another program has it open",
                status_code,
                attempts,
                final_url,
            )
        return Outcome(source.id, "downloaded", _fmt_bytes(size), status_code, size, digest, final_url, attempts, _utc_now())

    _unlink(part)
    return _failed(source, f"gave up after {attempts} attempt(s): {last_error}", status_code, attempts)


def check_existing(source: Source, raw_dir: Path) -> Outcome:
    """Inspect ``raw_dir`` without touching the network."""
    path = raw_dir / source.filename
    if not path.exists():
        return Outcome(source.id, "missing_manual" if source.is_manual else "missing", "file not found", at=_utc_now())
    reason = validate_file(path, source.format)
    if reason:
        return Outcome(source.id, "failed", f"file exists but is not usable: {reason}", at=_utc_now())
    return Outcome(
        source.id,
        "present_manual" if source.is_manual else "present",
        "present and valid",
        None,
        path.stat().st_size,
        _sha256_file(path),
        None,
        0,
        _utc_now(),
    )


# ------------------------------------------------------------------------------ orchestration


def select_sources(sources: Sequence[Source], only: Sequence[str] | None = None, core_only: bool = False) -> list[Source]:
    chosen = list(sources)
    if only:
        known = {s.id for s in sources}
        unknown = [i for i in only if i not in known]
        if unknown:
            raise ValueError(f"unknown source id(s): {', '.join(unknown)} (see the id column in data/sources.csv)")
        chosen = [s for s in chosen if s.id in set(only)]
    if core_only:
        chosen = [s for s in chosen if s.is_core]
    return chosen


def _line(index: int, total: int, source: Source, outcome: Outcome) -> str:
    tag = _TAGS.get(outcome.status, outcome.status.upper())
    detail = outcome.detail
    if outcome.status == "missing_manual":
        detail = f"save by hand as {source.filename}"
    return f"[{index:>2}/{total}] {tag:<8} {source.id:<50} {detail}"


def run(
    sources: Sequence[Source],
    raw_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    delay: float = 1.0,
    retries: int = 2,
    timeout: tuple[float, float] = (10.0, 60.0),
    backoff: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] = print,
) -> list[Outcome]:
    """Download every auto source (and check every manual one). Returns one Outcome per source."""
    session = session or make_session()
    outcomes: list[Outcome] = []
    previous_used_network = False
    for index, source in enumerate(sources, start=1):
        if source.is_manual:
            outcome = check_existing(source, raw_dir)
        else:
            if previous_used_network and delay > 0:
                sleep(delay)
            outcome = download_one(
                source, raw_dir, session, force=force, retries=retries, timeout=timeout, backoff=backoff, sleep=sleep
            )
            previous_used_network = outcome.attempts > 0
        outcomes.append(outcome)
        out(_line(index, len(sources), source, outcome))
    return outcomes


def status_only(sources: Sequence[Source], raw_dir: Path, out: Callable[[str], None] = print) -> list[Outcome]:
    outcomes = []
    for index, source in enumerate(sources, start=1):
        outcome = check_existing(source, raw_dir)
        outcomes.append(outcome)
        out(_line(index, len(sources), source, outcome))
    return outcomes


def summarize(all_sources: Sequence[Source], outcomes: Sequence[Outcome], raw_dir: Path) -> Summary:
    """Count usable documents across the whole manifest (re-checking disk for sources not in ``outcomes``)."""
    known = {o.id: o for o in outcomes}
    usable_ids: set[str] = set()
    for source in all_sources:
        outcome = known.get(source.id) or check_existing(source, raw_dir)
        if outcome.usable:
            usable_ids.add(source.id)
    core = [s for s in all_sources if s.is_core]
    optional = [s for s in all_sources if not s.is_core]
    return Summary(
        total=len(all_sources),
        usable=len(usable_ids),
        core_total=len(core),
        core_usable=sum(s.id in usable_ids for s in core),
        optional_total=len(optional),
        optional_usable=sum(s.id in usable_ids for s in optional),
        missing_core=tuple(s.id for s in core if s.id not in usable_ids),
        missing_optional=tuple(s.id for s in optional if s.id not in usable_ids),
    )


def find_unrecognized(sources: Sequence[Source], raw_dir: Path) -> list[tuple[str, str | None]]:
    """Files in ``raw_dir`` that are not in the manifest, with a rename hint when the name looks close."""
    if not raw_dir.is_dir():
        return []
    expected = {s.filename for s in sources}
    found = []
    for path in sorted(raw_dir.iterdir()):
        name = path.name
        if not path.is_file() or name.startswith(".") or name.endswith(".part") or name in expected:
            continue
        hint = next(
            (s.filename for s in sorted(sources, key=lambda s: -len(s.id)) if name.lower().startswith(s.id)), None
        )
        found.append((name, hint))
    return found


def write_report(path: Path, sources: Sequence[Source], outcomes: Sequence[Outcome], summary: Summary) -> None:
    """Write a provenance log (what was fetched, when, from where, with which hash). Merges with an older report."""
    previous: dict[str, dict] = {}
    if path.is_file():
        try:
            previous = {entry["id"]: entry for entry in json.loads(path.read_text(encoding="utf-8"))["sources"]}
        except (OSError, ValueError, KeyError, TypeError):
            previous = {}
    for outcome in outcomes:
        previous[outcome.id] = asdict(outcome)
    report = {
        "generated_at": _utc_now(),
        "python": sys.version.split()[0],
        "manifest": "data/sources.csv",
        "summary": {
            "total": summary.total,
            "usable": summary.usable,
            "core_usable": summary.core_usable,
            "core_total": summary.core_total,
            "meets_minimum_of_20": summary.meets_minimum,
        },
        "sources": [previous[s.id] for s in sources if s.id in previous],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_summary(
    summary: Summary,
    sources: Sequence[Source],
    raw_dir: Path,
    unrecognized: Sequence[tuple[str, str | None]],
    out: Callable[[str], None] = print,
    attempted: bool = True,
) -> None:
    """Print the totals and what is left to do.

    ``attempted`` is False in --status mode: an auto-download file that is simply absent has not been
    tried yet, so it is not described as a by-hand job.
    """
    by_id = {s.id: s for s in sources}
    out("")
    out(
        f"Usable documents: {summary.usable} of {summary.total} "
        f"(core {summary.core_usable}/{summary.core_total}, optional {summary.optional_usable}/{summary.optional_total})"
    )
    verdict = "MET" if summary.meets_minimum else "NOT MET"
    out(f"Assignment minimum of {MIN_DOCS} documents: {verdict}")

    missing = [by_id[i] for i in list(summary.missing_core) + list(summary.missing_optional)]
    not_tried = [] if attempted else [s for s in missing if not s.is_manual and not (raw_dir / s.filename).exists()]
    by_hand = [s for s in missing if s not in not_tried]
    if by_hand:
        out("")
        out("Still to do by hand (see data/MANUAL_DOWNLOADS.md):")
        for source in by_hand:
            kind = "core" if source.is_core else "optional"
            out(f"  {source.id}  [{kind}]")
            out(f"      open  {source.landing_url}")
            out(f"      save  {_display_path(raw_dir / source.filename)}")
    if not_tried:
        out("")
        out(f"{len(not_tried)} file(s) have not been downloaded yet. Run:  python scripts/download_corpus.py")
        out("If a file keeps failing, open its landing_url from data/sources.csv in your browser and save it by hand.")
    if unrecognized:
        out("")
        out("Files in data/raw/ that are not in the manifest:")
        for name, hint in unrecognized:
            out(f"  {name}" + (f"   -> rename to {hint}?" if hint else ""))


# ---------------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_corpus",
        description="Download and validate the RegLens corpus listed in data/sources.csv.",
    )
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "sources.csv"), help="path to sources.csv")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="where files are saved")
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these source ids")
    parser.add_argument("--core-only", action="store_true", help="skip optional (priority 2) documents")
    parser.add_argument("--force", action="store_true", help="download again even if a valid file exists")
    parser.add_argument("--status", action="store_true", help="no network: report what is present, missing or invalid")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to pause between downloads (default 1)")
    parser.add_argument("--retries", type=int, default=2, help="retries per file after a network error (default 2)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for data (default 60)")
    parser.add_argument(
        "--report", default=str(PROJECT_ROOT / "data" / "download_report.json"), help="where to write the provenance log"
    )
    return parser


def _make_console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _make_console_safe()
    args = _parser().parse_args(argv)
    manifest_path, raw_dir = Path(args.manifest), Path(args.raw_dir)

    try:
        sources = load_manifest(manifest_path)
        selected = select_sources(sources, args.only, args.core_only)
    except (ManifestError, ValueError) as exc:
        print(f"Problem: {exc}")
        return 2

    core = sum(s.is_core for s in sources)
    manual = sum(s.is_manual for s in sources)
    print("RegLens corpus downloader")
    print(f"Manifest: {_display_path(manifest_path)}  ({len(sources)} sources: {core} core, {len(sources) - core} optional; {manual} manual)")
    print(f"Target:   {_display_path(raw_dir)}")
    if not selected:
        print("Nothing selected.")
        return 2
    print("")

    if args.status:
        outcomes = status_only(selected, raw_dir)
    else:
        outcomes = run(
            selected,
            raw_dir,
            force=args.force,
            delay=max(0.0, args.delay),
            retries=max(0, args.retries),
            timeout=(10.0, max(5.0, args.timeout)),
        )

    summary = summarize(sources, outcomes, raw_dir)
    if not args.status:
        write_report(Path(args.report), sources, outcomes, summary)
    print_summary(summary, sources, raw_dir, find_unrecognized(sources, raw_dir), attempted=not args.status)
    if not args.status:
        print(f"\nProvenance log: {_display_path(Path(args.report))}")

    if args.only:
        return 0 if all(o.usable for o in outcomes) else 1
    return 0 if summary.meets_minimum else 1
