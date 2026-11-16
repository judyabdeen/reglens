"""Anthropic Claude through the Messages REST endpoint (``POST /v1/messages``).

Follows Anthropic's public API reference. Selectable with ``LLM_PROVIDER=anthropic``; the default model is
``claude-haiku-5-5``. Tested against a local mock server; the live verification of RegLens was done with Gemini.
"""

from __future__ import annotations

from collections.abc import Mapping

from reglens.llm import base
from reglens.llm.base import LLMError, LLMRequest, LLMResponse, OptionalSetting, Usage
from reglens.llm.rest import RestClient

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(RestClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, *, base_url: str = ANTHROPIC_BASE_URL, **options) -> None:
        super().__init__(model, api_key, base_url=base_url, **options)

    def _url(self) -> str:
        return f"{self._base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_body(self, request: LLMRequest) -> dict:
        body: dict = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            body["system"] = request.system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        return body

    def _optional_settings(self) -> tuple[OptionalSetting, ...]:
        return (OptionalSetting("temperature", ("temperature",), lambda body: body.pop("temperature", None)),)

    def _classify(self, status: int, data: object, headers: Mapping[str, str]) -> LLMError:
        error = super()._classify(status, data, headers)
        detail = data.get("error") if isinstance(data, Mapping) else None
        kind = str(detail.get("type", "")) if isinstance(detail, Mapping) else ""
        mapped = {
            "authentication_error": base.AUTH,
            "permission_error": base.FORBIDDEN,
            "not_found_error": base.NOT_FOUND,
            "rate_limit_error": base.RATE_LIMITED,
            "overloaded_error": base.OVERLOADED,
            "invalid_request_error": base.BAD_REQUEST,
            "api_error": base.SERVER,
        }.get(kind)
        if mapped and mapped != error.kind:
            return LLMError(mapped, error.message, status=status, retry_after=error.retry_after)
        return error

    def _parse(self, data: Mapping, latency_s: float) -> LLMResponse:
        blocks = data.get("content") or []
        text = "".join(
            block["text"]
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ).strip()
        stop = data.get("stop_reason")
        if not text:
            if stop == "refusal":
                raise LLMError(base.BLOCKED, "Claude declined to answer")
            raise LLMError(base.EMPTY, f"Claude returned an empty answer (stop_reason={stop})")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            finish_reason=stop,
            usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
            latency_s=latency_s,
            truncated=stop == "max_tokens",
        )
