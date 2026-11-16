"""Make a model call dependable: pace it, retry transient failures and fall back to another model.

Free API tiers rate-limit by the minute and by the day, and providers have busy moments. This wrapper turns those
into short waits or a switch to the next model of the chain, and raises one clear ``LLMError`` only when every model
has failed.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

from reglens.llm import base
from reglens.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse


class ResilientClient:
    """A chain of clients (usually one provider, several models) behaving like one client.

    * calls are spaced at least ``min_interval`` seconds apart (useful for free-tier limits and for the evaluation);
    * a transient failure (rate limit, busy server, timeout, network) is retried up to ``max_attempts`` times with
      exponential backoff, or after the delay the provider asked for;
    * a provider that asks for a wait longer than ``max_delay`` is not waited for: the next model is tried instead;
    * every other failure (bad request, quota used up, blocked, empty answer) moves on to the next model at once;
    * an authentication failure is raised immediately, because every model uses the same key.
    """

    def __init__(
        self,
        clients: Sequence[LLMClient],
        *,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        min_interval: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not clients:
            raise ValueError("at least one client is needed")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._clients = tuple(clients)
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._lock = threading.Lock()
        self._last_start: float | None = None

    @property
    def provider(self) -> str:
        return self._clients[0].provider

    @property
    def model(self) -> str:
        return self._clients[0].model

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(client.model for client in self._clients)

    def _pace(self) -> None:
        with self._lock:
            if self._min_interval > 0 and self._last_start is not None:
                wait = self._min_interval - (self._clock() - self._last_start)
                if wait > 0:
                    self._sleep(wait)
            self._last_start = self._clock()

    def _delay(self, error: LLMError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None if the wait is not worth it."""
        if error.retry_after is not None:
            return error.retry_after + 0.5 if error.retry_after <= self._max_delay else None
        delay = min(self._max_delay, self._base_delay * 2 ** (attempt - 1))
        return delay * (0.75 + 0.5 * self._jitter())  # +-25 % so that parallel callers do not retry together

    def generate(self, request: LLMRequest) -> LLMResponse:
        tried: list[tuple[str, str]] = []
        attempts = 0
        last: LLMError | None = None
        for client in self._clients:
            for attempt in range(1, self._max_attempts + 1):
                self._pace()
                attempts += 1
                try:
                    response = client.generate(request)
                except LLMError as error:
                    last = error
                    if error.kind == base.AUTH:
                        error.tried = [*tried, (client.model, error.kind)]
                        raise
                    delay = self._delay(error, attempt) if error.retryable and attempt < self._max_attempts else None
                    if delay is None:
                        break
                    self._sleep(delay)
                else:
                    return replace(response, attempts=attempts, fell_back_from=tried[0][0] if tried else None)
            assert last is not None  # the loop above only ends through a failure
            tried.append((client.model, last.kind))
        assert last is not None
        summary = ", ".join(f"{model}: {kind}" for model, kind in tried)
        failure = LLMError(
            last.kind,
            f"{last.message} (models tried: {summary})" if len(tried) > 1 else last.message,
            status=last.status,
            retry_after=last.retry_after,
            provider=last.provider,
            model=last.model,
        )
        failure.tried = tried
        raise failure from last
