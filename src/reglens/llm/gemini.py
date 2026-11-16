"""Google Gemini through the ``generateContent`` REST endpoint (the default provider of RegLens).

Facts this client is built on (Google's Gemini API documentation, checked 2026-11-15):
  * the key travels in the ``x-goog-api-key`` header, never in the URL;
  * Gemini 3 models think before answering and the thinking tokens count against ``maxOutputTokens``, so a small
    limit can produce an empty answer; ``thinkingConfig.thinkingLevel`` ("minimal" | "low" | "medium" | "high")
    controls how much;
  * Google advises leaving ``temperature``, ``top_p`` and ``top_k`` at their defaults for Gemini 3 models.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote

from reglens.llm import base
from reglens.llm.base import LLMError, LLMRequest, LLMResponse, OptionalSetting, Usage
from reglens.llm.rest import RestClient

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
# Finish reasons that mean the provider stopped the text for policy reasons.
_STOPPED_BY_POLICY = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"})


class GeminiClient(RestClient):
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        thinking_level: str | None = None,
        base_url: str = GEMINI_BASE_URL,
        **options,
    ) -> None:
        super().__init__(model.removeprefix("models/"), api_key, base_url=base_url, **options)
        self.thinking_level = thinking_level

    def _url(self) -> str:
        return f"{self._base_url}/v1beta/models/{quote(self.model, safe='-._')}:generateContent"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def _build_body(self, request: LLMRequest) -> dict:
        config: dict = {"maxOutputTokens": request.max_output_tokens}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        body: dict = {
            "contents": [
                {"role": "user" if message.role == "user" else "model", "parts": [{"text": message.content}]}
                for message in request.messages
            ],
            "generationConfig": config,
        }
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        return body

    def _optional_settings(self) -> tuple[OptionalSetting, ...]:
        # A model that does not know a thinking level answers HTTP 400 and names the field: repeat without it.
        return (
            OptionalSetting("thinkingConfig", ("thinking",), lambda body: body["generationConfig"].pop("thinkingConfig", None)),
            OptionalSetting("temperature", ("temperature",), lambda body: body["generationConfig"].pop("temperature", None)),
        )

    def _classify(self, status: int, data: object, headers: Mapping[str, str]) -> LLMError:
        error = super()._classify(status, data, headers)
        body = data.get("error") if isinstance(data, Mapping) else None
        details = [d for d in (body.get("details") or []) if isinstance(d, Mapping)] if isinstance(body, Mapping) else []
        reasons = {str(d.get("reason", "")) for d in details}
        if "API_KEY_INVALID" in reasons or "api key not valid" in error.message.lower():
            return LLMError(base.AUTH, error.message, status=status)
        if status == 429 or (isinstance(body, Mapping) and body.get("status") == "RESOURCE_EXHAUSTED"):
            delays = [
                base.parse_seconds(d.get("retryDelay")) for d in details if "RetryInfo" in str(d.get("@type", ""))
            ]
            quota_ids = [
                str(v.get("quotaId", ""))
                for d in details
                if "QuotaFailure" in str(d.get("@type", ""))
                for v in (d.get("violations") or [])
                if isinstance(v, Mapping)
            ]
            daily = any("perday" in q.lower().replace("_", "").replace("-", "") for q in quota_ids) or "per day" in error.message.lower()
            retry_after = next((d for d in delays if d is not None), error.retry_after)
            return LLMError(base.QUOTA if daily else base.RATE_LIMITED, error.message, status=status, retry_after=retry_after)
        return error

    def _parse(self, data: Mapping, latency_s: float) -> LLMResponse:
        feedback = data.get("promptFeedback") or {}
        if isinstance(feedback, Mapping) and feedback.get("blockReason"):
            raise LLMError(base.BLOCKED, f"Gemini refused the prompt ({feedback['blockReason']})")
        candidates = data.get("candidates") or []
        if not candidates or not isinstance(candidates[0], Mapping):
            raise LLMError(base.EMPTY, "Gemini returned no answer")
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        # Thought summaries (only sent when requested) are marked "thought": true and are not part of the answer.
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, Mapping) and isinstance(part.get("text"), str) and not part.get("thought")
        ).strip()
        if not text:
            if finish == "MAX_TOKENS":
                raise LLMError(
                    base.EMPTY,
                    "Gemini used the whole output limit for thinking and wrote no answer; "
                    "lower GEMINI_THINKING_LEVEL or raise the limit",
                )
            if finish in _STOPPED_BY_POLICY:
                raise LLMError(base.BLOCKED, f"Gemini stopped without an answer ({finish})")
            raise LLMError(base.EMPTY, f"Gemini returned an empty answer (finishReason={finish})")
        meta = data.get("usageMetadata") or {}
        output = (meta.get("candidatesTokenCount") or 0) + (meta.get("thoughtsTokenCount") or 0)
        usage = Usage(
            prompt_tokens=meta.get("promptTokenCount"),
            output_tokens=output if (meta.get("candidatesTokenCount") is not None or output) else None,
        )
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            finish_reason=finish,
            usage=usage,
            latency_s=latency_s,
            truncated=finish == "MAX_TOKENS" or finish in _STOPPED_BY_POLICY,
        )
