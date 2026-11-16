"""What every language-model client has in common: the request, the response and the ways a call can fail.

RegLens talks to the model providers with plain HTTP (``requests``), not with their SDKs. The three REST APIs are
small, and writing the calls by hand keeps every step visible: what is sent, what comes back, and how each kind of
failure is handled.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Kinds of failure. The retry logic in ``resilient.py`` decides what to do from the kind alone.
AUTH = "auth"  # the key is missing or wrong: no other model will work either
FORBIDDEN = "forbidden"  # this model is not allowed for this key: another model may work
QUOTA = "quota"  # the daily or billing quota of this model is used up: waiting will not help
RATE_LIMITED = "rate_limited"  # too many requests per minute: wait and retry
OVERLOADED = "overloaded"  # the provider is busy (HTTP 503 / 529): wait and retry
SERVER = "server"  # HTTP 500 / 502 / 504: retry
TIMEOUT = "timeout"
NETWORK = "network"
BAD_REQUEST = "bad_request"  # HTTP 400: retrying the same request cannot help
NOT_FOUND = "not_found"  # unknown or retired model name
BLOCKED = "blocked"  # the provider's safety filter refused the prompt or the answer
EMPTY = "empty"  # the call succeeded but returned no text (for example the output budget went into thinking)
UNKNOWN = "unknown"

RETRYABLE_KINDS = frozenset({RATE_LIMITED, OVERLOADED, SERVER, TIMEOUT, NETWORK})


class LLMError(Exception):
    """A failed model call. ``kind`` is one of the constants above; ``retry_after`` is the provider's hint in seconds."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after
        self.provider = provider
        self.model = model
        self.tried: list[tuple[str, str]] = []  # (model, kind) of every model that failed before this error

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def __str__(self) -> str:
        where = f"{self.provider}/{self.model}: " if self.provider and self.model else ""
        return f"{where}{self.message}"


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', not {self.role!r}")


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: tuple[Message, ...]
    max_output_tokens: int = 2048
    # None leaves the provider's default in place. Google recommends not changing it for Gemini 3 models, and some
    # OpenAI and Anthropic models reject any other value.
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a request needs at least one message")
        if self.messages[0].role != "user":
            raise ValueError("the first message must come from the user")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    @classmethod
    def single(cls, system: str, user: str, **options) -> LLMRequest:
        """The common case: one system prompt and one user message."""
        return cls(system=system, messages=(Message("user", user),), **options)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    output_tokens: int | None = None  # including any thinking tokens the provider bills as output

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.output_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    truncated: bool = False  # the answer was cut off by the output limit
    attempts: int = 1  # HTTP calls made, over all models tried
    fell_back_from: str | None = None  # the model that failed first, when another one answered


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------------------- helpers shared by clients


def scrub(text: str, *secrets: str | None) -> str:
    """Remove API keys from a message that is about to be shown or logged."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


_SECONDS = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*$")


def parse_seconds(value: object) -> float | None:
    """Read "34s", "0.5s", "12" or 12 as a number of seconds. Anything else (a date, junk) gives None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        match = _SECONDS.match(value)
        if match:
            return float(match.group(1))
    return None


@dataclass(frozen=True)
class OptionalSetting:
    """A request field that newer or older models may refuse.

    When the API answers HTTP 400 and its message contains one of ``keywords``, the field is removed from the
    request and the call is repeated once without it.
    """

    name: str
    keywords: tuple[str, ...]
    remove: Callable[[dict], None]
