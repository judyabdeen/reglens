"""The HTTP plumbing shared by the three provider clients.

A provider client only says how to build its request body, how to read its answer and how to recognise its own error
messages. Sending the request, turning network problems and HTTP error codes into ``LLMError`` and removing a
request field the model refuses all happen here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import requests

from reglens.llm import base
from reglens.llm.base import LLMError, LLMRequest, LLMResponse, OptionalSetting

DEFAULT_TIMEOUT = (10.0, 120.0)  # seconds to connect, seconds to wait for the answer


def kind_for_status(status: int) -> str:
    """The failure kind that an HTTP status code means when the body says nothing more."""
    if status == 401:
        return base.AUTH
    if status == 403:
        return base.FORBIDDEN
    if status == 404:
        return base.NOT_FOUND
    if status == 408:
        return base.TIMEOUT
    if status == 429:
        return base.RATE_LIMITED
    if status in (503, 529):
        return base.OVERLOADED
    if status >= 500:
        return base.SERVER
    if status in (400, 413, 422):
        return base.BAD_REQUEST
    return base.UNKNOWN


def error_message(data: object, status: int) -> str:
    """The provider's own explanation (``{"error": {"message": ...}}`` in all three APIs), or the status code."""
    if isinstance(data, Mapping):
        error = data.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            return " ".join(str(error["message"]).split())[:400]
        if isinstance(error, str) and error:
            return " ".join(error.split())[:400]
        if data.get("message"):
            return " ".join(str(data["message"]).split())[:400]
    return f"HTTP {status}"


class RestClient(ABC):
    """Base class of the provider clients."""

    provider: str = ""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        post: Callable[..., requests.Response] = requests.post,
    ) -> None:
        if not api_key:
            raise LLMError(base.AUTH, f"no API key for {self.provider}", provider=self.provider, model=model)
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._post = post

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.model!r})"

    # ---- provider-specific parts
    @abstractmethod
    def _url(self) -> str: ...

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _build_body(self, request: LLMRequest) -> dict: ...

    @abstractmethod
    def _parse(self, data: Mapping, latency_s: float) -> LLMResponse: ...

    def _classify(self, status: int, data: object, headers: Mapping[str, str]) -> LLMError:
        """Turn an HTTP error into an LLMError. Providers refine this where their error bodies say more."""
        return LLMError(
            kind_for_status(status),
            base.scrub(error_message(data, status), self._api_key),
            status=status,
            retry_after=base.parse_seconds(headers.get("retry-after")),
        )

    def _optional_settings(self) -> tuple[OptionalSetting, ...]:
        return ()

    # ---- the call
    def generate(self, request: LLMRequest) -> LLMResponse:
        body = self._build_body(request)
        dropped: set[str] = set()
        while True:
            started = time.monotonic()
            try:
                data = self._send(body)
            except LLMError as exc:
                exc.provider, exc.model = exc.provider or self.provider, exc.model or self.model
                option = self._refused_option(exc, dropped)
                if option is None:
                    raise
                option.remove(body)
                dropped.add(option.name)
                continue
            try:
                return self._parse(data, time.monotonic() - started)
            except LLMError as exc:
                exc.provider, exc.model = exc.provider or self.provider, exc.model or self.model
                raise

    def _refused_option(self, error: LLMError, dropped: set[str]) -> OptionalSetting | None:
        if error.kind != base.BAD_REQUEST:
            return None
        message = error.message.lower()
        for option in self._optional_settings():
            if option.name not in dropped and any(word in message for word in option.keywords):
                return option
        return None

    def _send(self, body: dict) -> Mapping:
        try:
            response = self._post(self._url(), headers=self._headers(), json=body, timeout=self._timeout)
        except requests.Timeout as exc:
            raise LLMError(base.TIMEOUT, f"no answer within {self._timeout[1]:.0f} seconds") from exc
        except requests.RequestException as exc:
            text = base.scrub(f"could not reach the {self.provider} API ({type(exc).__name__})", self._api_key)
            raise LLMError(base.NETWORK, text + " - check internet, VPN, firewall or proxy") from exc
        try:
            data = response.json()
        except ValueError:
            data = None
        status = response.status_code
        if status != 200:
            raise self._classify(status, data, {k.lower(): v for k, v in (response.headers or {}).items()})
        if not isinstance(data, Mapping):
            raise LLMError(base.SERVER, f"the {self.provider} API answered with something that is not JSON", status=status)
        return data
