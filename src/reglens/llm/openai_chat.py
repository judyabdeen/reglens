"""OpenAI through the Chat Completions REST endpoint (``POST /v1/chat/completions``).

This client follows OpenAI's public API reference. It is selectable with ``LLM_PROVIDER=openai`` and needs
``OPENAI_MODEL`` (OpenAI model names change too often for RegLens to guess one). It is tested against a local mock
server; the live verification of RegLens was done with Gemini.
"""

from __future__ import annotations

from collections.abc import Mapping

from reglens.llm import base
from reglens.llm.base import LLMError, LLMRequest, LLMResponse, OptionalSetting, Usage
from reglens.llm.rest import RestClient

OPENAI_BASE_URL = "https://api.openai.com"


class OpenAIClient(RestClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str, *, base_url: str = OPENAI_BASE_URL, **options) -> None:
        super().__init__(model, api_key, base_url=base_url, **options)

    def _url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _build_body(self, request: LLMRequest) -> dict:
        messages = [{"role": "system", "content": request.system}] if request.system else []
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        body: dict = {"model": self.model, "messages": messages, "max_completion_tokens": request.max_output_tokens}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        return body

    def _optional_settings(self) -> tuple[OptionalSetting, ...]:
        # Reasoning models accept only the default temperature and answer HTTP 400 otherwise.
        return (OptionalSetting("temperature", ("temperature",), lambda body: body.pop("temperature", None)),)

    def _classify(self, status: int, data: object, headers: Mapping[str, str]) -> LLMError:
        error = super()._classify(status, data, headers)
        detail = data.get("error") if isinstance(data, Mapping) else None
        code = str(detail.get("code") or detail.get("type") or "") if isinstance(detail, Mapping) else ""
        if status == 429 and ("insufficient_quota" in code or "billing" in error.message.lower()):
            return LLMError(base.QUOTA, error.message, status=status)
        return error

    def _parse(self, data: Mapping, latency_s: float) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            raise LLMError(base.EMPTY, "OpenAI returned no answer")
        choice = choices[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            if message.get("refusal") or finish == "content_filter":
                raise LLMError(base.BLOCKED, f"OpenAI declined to answer ({message.get('refusal') or finish})")
            raise LLMError(base.EMPTY, f"OpenAI returned an empty answer (finish_reason={finish})")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            finish_reason=finish,
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_s=latency_s,
            truncated=finish == "length",
        )
