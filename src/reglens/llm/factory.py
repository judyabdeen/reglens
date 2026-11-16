"""Build the model client that the settings ask for."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import requests

from reglens.config import ConfigError, Settings
from reglens.llm import base
from reglens.llm.anthropic_messages import AnthropicClient
from reglens.llm.base import LLMError
from reglens.llm.gemini import GeminiClient
from reglens.llm.openai_chat import OpenAIClient
from reglens.llm.resilient import ResilientClient

KEY_NAMES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def make_llm(
    settings: Settings,
    provider: str | None = None,
    *,
    model: str | None = None,
    min_interval: float = 0.0,
    post: Callable[..., requests.Response] = requests.post,
    base_urls: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ResilientClient:
    """Return a client for ``provider`` (default: ``LLM_PROVIDER``). ``model`` overrides the configured model.

    Gemini gets a chain: the primary model followed by the fallback models of ``GEMINI_FALLBACK_MODEL`` (free-tier
    limits are per model, so the next model often still has room). Raises ``LLMError(auth)`` when the key is empty
    and ``ConfigError`` when OpenAI is chosen without ``OPENAI_MODEL``.
    """
    provider = (provider or settings.llm_provider).lower()
    key = settings.key_for(provider)  # raises ConfigError for an unknown provider
    if not key:
        raise LLMError(
            base.AUTH,
            f"{KEY_NAMES[provider]} is empty - add your key to the .env file (see .env.example)",
            provider=provider,
        )
    urls = dict(base_urls or {})
    common = {"post": post}
    if provider == "gemini":
        names = [model or settings.gemini_model]
        if not model:
            names += [name for name in settings.gemini_fallbacks if name not in names]
        extra = {"base_url": urls["gemini"]} if "gemini" in urls else {}
        clients = [
            GeminiClient(name, key, thinking_level=settings.gemini_thinking_level, **common, **extra) for name in names
        ]
    elif provider == "openai":
        chosen = model or settings.openai_model
        if not chosen:
            raise ConfigError("OPENAI_MODEL is empty - set the model name in .env (python scripts/check_stack.py --llm lists yours)")
        extra = {"base_url": urls["openai"]} if "openai" in urls else {}
        clients = [OpenAIClient(chosen, key, **common, **extra)]
    else:
        extra = {"base_url": urls["anthropic"]} if "anthropic" in urls else {}
        clients = [AnthropicClient(model or settings.anthropic_model, key, **common, **extra)]
    return ResilientClient(clients, min_interval=min_interval, sleep=sleep)
