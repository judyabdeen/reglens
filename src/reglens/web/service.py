"""What the web page asks of RegLens, as plain Python with no web framework.

``web/app.py`` only moves bytes between the browser and this module. Every decision that matters is made here and is
tested by calling it directly: which requests are valid, what happens while the index has not been built yet, how the
comparison with the model alone is made, and which errors a visitor may see (never a stack trace, never a key).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from reglens import __version__
from reglens.config import ConfigError, Settings
from reglens.embeddings import EmbeddingError
from reglens.generation.answer import RAG, Answer
from reglens.generation.prompts import page_text, status_text
from reglens.indexing import IndexingError, IndexNotReady
from reglens.pipeline import MAX_QUESTION_CHARS, AskOptions
from reglens.retrieval import MODES, RetrievalError, RetrievalOptions, TransformOptions, Turn
from reglens.runtime import Runtime, build_runtime, describe
from reglens.vectorstore import VectorStoreError
from reglens.vectorstore.base import Record

log = logging.getLogger("reglens.web")

ASK_MODES = ("rag", "compare", "passages", "llm_only")  # documents + model | both, side by side | documents only | model only
FILTER_KEYS = ("jurisdiction", "status", "doc_type", "doc_id")
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 6000
MAX_FILTER_VALUES = 60
TOP_K_RANGE = (1, 12)
MULTI_QUERY_RANGE = (0, 3)
FIX_HINT = "python scripts/build_index.py"
_FIX_AT_THE_END = re.compile(r"\s*:?\s*python scripts/build_index\.py\.?\s*$")

# The settings of the page and their defaults; the page reads them from /api/status so both always agree.
DEFAULT_OPTIONS = {
    "search": "hybrid",
    "top_k": 8,
    "rerank": False,
    "glossary": True,
    "condense": True,
    "multi_query": 0,
    "hyde": False,
    "include_notes": True,
}

# What to tell a visitor when the language model could not be used, by the kind of failure.
HINTS = {
    "unavailable": "No language model is set up, so only the matching passages are shown. Add your API key to the .env file (see .env.example) and restart RegLens.",
    "auth": "The provider rejected the API key. Check it in the .env file (python scripts/check_setup.py tests it) and restart RegLens.",
    "quota": "The free quota of this model is used up. Try again later, or set another GEMINI_MODEL in the .env file.",
    "rate_limited": "Too many requests per minute. Wait a minute and ask again.",
    "overloaded": "The provider is busy right now. Ask again in a moment.",
    "server": "The provider had an internal error. Ask again in a moment.",
    "timeout": "The provider did not answer in time. Ask again, or check the internet connection.",
    "network": "The provider could not be reached. Check the internet connection, VPN or firewall.",
    "forbidden": "This key may not use this model. Set another GEMINI_MODEL in the .env file.",
    "not_found": "The model name is unknown or retired. Set a current model in the .env file (python scripts/check_setup.py lists yours).",
    "empty": "The model returned no text. Ask again, or set GEMINI_THINKING_LEVEL=minimal in the .env file.",
    "blocked": "The provider's safety filter refused this request.",
    "bad_request": "The provider refused the request as malformed. This is a bug in RegLens: the message in the server window says why.",
    "unknown": "The language model failed in an unexpected way. Ask again; if it keeps happening, read the message in the server window.",
}


def tidy_problem(message: object) -> str:
    """A start-up problem as a sentence for the page.

    The messages of the index code are written for the command line and end with the command that fixes the problem.
    The page shows that command in a box of its own (``FIX_HINT``), so it is cut from the end of the sentence, which then
    starts with a capital letter and ends with a full stop.
    """
    text = _FIX_AT_THE_END.sub("", " ".join(str(message).split())).strip()
    if not text:
        return "RegLens could not open its library."
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


# ------------------------------------------------------------------------------------------------------ errors


class ServiceError(Exception):
    """A problem a visitor may be told about. ``status`` is the HTTP status the web layer uses."""

    status = 500
    kind = "internal"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class RequestError(ServiceError):
    status = 400
    kind = "bad_request"


class NotFound(ServiceError):
    status = 404
    kind = "not_found"


class Busy(ServiceError):
    status = 429
    kind = "busy"


class NotReady(ServiceError):
    """The index has not been built (or cannot be opened) yet."""

    status = 503
    kind = "not_ready"


class Unavailable(ServiceError):
    """A local model or the database failed while a question was being answered."""

    status = 503
    kind = "unavailable"


# ------------------------------------------------------------------------------------------------------ requests


@dataclass(frozen=True)
class AskRequest:
    question: str
    mode: str
    history: tuple[Turn, ...]
    options: AskOptions


def _squash(value: object, field: str, *, limit: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RequestError(f"{field} must be text", field=field)
    text = " ".join(value.split())
    if required and not text:
        raise RequestError(f"{field} is empty", field=field)
    if len(text) > limit:
        raise RequestError(f"{field} is too long ({len(text):,} characters; the limit is {limit:,})", field=field)
    return text


def _history(raw: object) -> tuple[Turn, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RequestError("history must be a list of {question, answer} objects", field="history")
    turns: list[Turn] = []
    for position, item in enumerate(raw[-MAX_HISTORY_TURNS:]):
        if not isinstance(item, dict):
            raise RequestError(f"history entry {position + 1} must be an object with a question and an answer", field="history")
        question = _squash(item.get("question"), "history question", limit=MAX_HISTORY_CHARS, required=False)
        answer = _squash(item.get("answer"), "history answer", limit=MAX_HISTORY_CHARS, required=False)
        if question and answer:
            turns.append(Turn(question, answer))
    return tuple(turns)


def _integer(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RequestError(f"{name} must be a whole number from {low} to {high}", field=f"options.{name}")
    return value


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RequestError(f"{name} must be true or false", field=f"options.{name}")
    return value


def _known_values(catalog: Sequence[Mapping]) -> dict[str, set[str]]:
    return {key: {str(entry[key]) for entry in catalog if entry.get(key)} for key in FILTER_KEYS}


def _filters(raw: object, catalog: Sequence[Mapping] | None) -> dict[str, str | list[str]]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise RequestError("filters must be an object such as {\"jurisdiction\": [\"EU\"]}", field="filters")
    known = _known_values(catalog) if catalog is not None else None
    where: dict[str, str | list[str]] = {}
    for key, value in raw.items():
        if key not in FILTER_KEYS:
            raise RequestError(f"unknown filter {key!r}; use one of {', '.join(FILTER_KEYS)}", field="filters")
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise RequestError(f"filter {key} must be a text or a list of texts", field=f"filters.{key}")
        values = list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if not values:
            continue
        if len(values) > MAX_FILTER_VALUES:
            raise RequestError(f"filter {key} lists too many values (the limit is {MAX_FILTER_VALUES})", field=f"filters.{key}")
        if known is not None:
            unknown = [v for v in values if v not in known[key]]
            if unknown:
                choices = f" Choose from: {', '.join(sorted(known[key]))}." if len(known[key]) <= 12 else ""
                raise RequestError(f"No indexed document has {key} {unknown[0]!r}.{choices}", field=f"filters.{key}")
        where[key] = values[0] if len(values) == 1 else values
    return where


def parse_ask_request(payload: object, catalog: Sequence[Mapping] | None = None) -> AskRequest:
    """Check a request from the page and turn it into the settings the pipeline understands.

    Everything is checked strictly (types, ranges, unknown keys) so that a typo in a request is reported instead of being
    silently ignored. ``catalog`` (the documents of the index) lets the filter values be checked too.
    """
    if not isinstance(payload, dict):
        raise RequestError("The request must be a JSON object")
    unknown = sorted(set(payload) - {"question", "mode", "history", "options", "filters"})
    if unknown:
        raise RequestError(f"unknown field {unknown[0]!r}", field=unknown[0])
    question = _squash(payload.get("question"), "question", limit=MAX_QUESTION_CHARS)
    mode = payload.get("mode", "rag")
    if mode not in ASK_MODES:
        raise RequestError(f"mode must be one of {', '.join(ASK_MODES)}", field="mode")
    history = _history(payload.get("history"))

    raw = payload.get("options")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RequestError("options must be an object", field="options")
    extra = sorted(set(raw) - set(DEFAULT_OPTIONS))
    if extra:
        raise RequestError(f"unknown setting {extra[0]!r}; use one of {', '.join(DEFAULT_OPTIONS)}", field=f"options.{extra[0]}")
    chosen = {**DEFAULT_OPTIONS, **raw}
    search = chosen["search"]
    if search not in MODES:
        raise RequestError(f"search must be one of {', '.join(MODES)}", field="options.search")
    top_k = _integer(chosen["top_k"], "top_k", *TOP_K_RANGE)
    multi_query = _integer(chosen["multi_query"], "multi_query", *MULTI_QUERY_RANGE)
    where = _filters(payload.get("filters"), catalog)

    options = AskOptions(
        retrieval=RetrievalOptions(
            mode=search,
            top_k=top_k,
            candidates=max(40, top_k),
            rerank=_flag(chosen["rerank"], "rerank"),
            rerank_pool=max(20, top_k),
            where=where or None,
            include_notes=_flag(chosen["include_notes"], "include_notes"),
        ),
        transform=TransformOptions(
            glossary=_flag(chosen["glossary"], "glossary"),
            condense=_flag(chosen["condense"], "condense"),
            multi_query=multi_query,
            hyde=_flag(chosen["hyde"], "hyde"),
        ),
        use_llm=mode in ("rag", "compare"),
    )
    return AskRequest(question=question, mode=mode, history=history, options=options)


# ------------------------------------------------------------------------------------------------------ output


def json_safe(value: object) -> object:
    """Plain lists, dictionaries and numbers only: tuples become lists and NaN or infinity become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _answer_payload(answer: Answer | None) -> dict | None:
    if answer is None:
        return None
    data = json_safe(answer.to_dict())
    data["hint"] = HINTS.get(answer.error_kind or "") if answer.error_kind else None
    return data


def _passage(record: Record | None) -> dict | None:
    if record is None:
        return None
    meta = record.metadata
    return {
        "chunk_id": record.id,
        "doc_id": str(meta.get("doc_id", "")),
        "title": str(meta.get("title", meta.get("doc_id", record.id))),
        "jurisdiction": str(meta.get("jurisdiction", "")),
        "status": str(meta.get("status", "")),
        "status_text": status_text(str(meta.get("status", "")) or None),
        "section": str(meta.get("section", "")),
        "pages": page_text(meta.get("page_start"), meta.get("page_end")),
        "kind": str(meta.get("kind", "text")),
        "url": str(meta.get("url", "")),
        "text": record.text,
    }


# ------------------------------------------------------------------------------------------------------ the service


class WebService:
    """The index, the models and the pipeline behind the page. Thread-safe: the HTTP layer calls it from worker threads."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: Callable[[Settings], Runtime] = build_runtime,
        max_concurrent: int = 2,
        retry_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._factory = runtime_factory
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._runtime: Runtime | None = None
        self._documents: list[dict] = []
        self._problem: str | None = None
        self._failed_at = 0.0

    # ---- the runtime is built on first use, and again after a failure (so building the index while the page is open works)
    def runtime(self) -> Runtime:
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            now = self._clock()
            if self._problem is not None and now - self._failed_at < self._retry_seconds:
                raise NotReady(self._problem)
            try:
                runtime = self._factory(self._settings)
                documents = runtime.retriever.catalog()
            except (IndexNotReady, IndexingError, ConfigError, VectorStoreError, EmbeddingError, OSError) as exc:
                self._problem, self._failed_at = tidy_problem(exc), now
                raise NotReady(self._problem) from exc
            self._runtime, self._documents, self._problem = runtime, documents, None
            return runtime

    def warm_up(self) -> None:
        """Load the embedding model now rather than during the first question. Never raises."""
        try:
            runtime = self.runtime()
            runtime.retriever.retrieve("warm up", RetrievalOptions(mode="dense", top_k=1))
        except Exception as exc:  # noqa: BLE001 - a failed warm-up only means the first question is slower or reports the problem
            log.warning("warm-up failed: %s: %s", type(exc).__name__, exc)

    # ---- read-only pages
    def status(self) -> dict:
        base: dict = {
            "app": "RegLens",
            "version": __version__,
            "defaults": dict(DEFAULT_OPTIONS),
            "limits": {
                "question_chars": MAX_QUESTION_CHARS,
                "history_turns": MAX_HISTORY_TURNS,
                "top_k": list(TOP_K_RANGE),
                "multi_query": list(MULTI_QUERY_RANGE),
            },
        }
        try:
            runtime = self.runtime()
        except NotReady as exc:
            return {**base, "ready": False, "problem": str(exc), "fix": FIX_HINT}
        return {
            **base,
            "ready": True,
            "index": dict(describe(runtime)),
            "llm": {
                "available": runtime.has_llm,
                "provider": runtime.llm_provider,
                "model": runtime.llm_model,
                "problem": runtime.llm_problem,
            },
            "warnings": list(runtime.warnings),
        }

    def documents(self) -> dict:
        self.runtime()
        return {"documents": [dict(entry) for entry in self._documents]}

    def passage(self, chunk_id: object) -> dict:
        if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > 200:
            raise RequestError("The passage id is missing or too long", field="chunk_id")
        retriever = self.runtime().retriever
        record = retriever.record(chunk_id)
        if record is None:
            raise NotFound(f"There is no passage {chunk_id!r} in the index.")
        before, after = retriever.neighbours(chunk_id)
        return {"passage": _passage(record), "before": _passage(before), "after": _passage(after)}

    # ---- questions
    def ask(self, payload: object) -> dict:
        runtime = self.runtime()
        request = parse_ask_request(payload, self._documents)
        if not self._slots.acquire(blocking=False):
            raise Busy("Another question is still being answered. Wait for it to finish, then ask again.")
        started = time.perf_counter()
        try:
            pipeline = runtime.pipeline
            answer = baseline = None
            if request.mode == "llm_only":
                baseline = pipeline.ask_llm_only(request.question, request.options.max_output_tokens)
            else:
                answer = pipeline.ask(request.question, request.history, request.options)
                if request.mode == "compare" and answer.mode == RAG:  # not after a failed model call: the model alone would fail too
                    # the standalone question, so a follow-up such as "and in the US?" means the same to both sides
                    baseline = pipeline.ask_llm_only(answer.standalone_question, request.options.max_output_tokens)
        except (EmbeddingError, VectorStoreError, RetrievalError) as exc:
            raise Unavailable(f"A search component failed: {exc}") from exc
        finally:
            self._slots.release()
        elapsed = time.perf_counter() - started
        log.info(
            "answered mode=%s sources=%d cited=%d in %.1fs",
            request.mode, len(answer.sources) if answer else 0, len(answer.cited) if answer else 0, elapsed,
        )  # the question itself is never logged
        return {"mode": request.mode, "answer": _answer_payload(answer), "baseline": _answer_payload(baseline), "elapsed": round(elapsed, 3)}
