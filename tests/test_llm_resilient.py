"""Retry, back-off, pacing and fallback between models - tested with scripted fake clients and a fake clock."""

from __future__ import annotations

import pytest

from reglens.config import ConfigError, load_settings
from reglens.llm import base
from reglens.llm.base import LLMError, LLMRequest, LLMResponse
from reglens.llm.factory import make_llm
from reglens.llm.resilient import ResilientClient
from mockapi import MockAPI

REQUEST = LLMRequest.single("system", "question")


class Scripted:
    """A fake client: each call returns or raises the next item of its script."""

    provider = "fake"

    def __init__(self, model, *script):
        self.model = model
        self.script = list(script)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, provider=self.provider, model=self.model)


class Clock:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def make(clients, clock=None, **options):
    clock = clock or Clock()
    options.setdefault("jitter", lambda: 0.5)  # jitter factor exactly 1.0
    return ResilientClient(clients, sleep=clock.sleep, clock=clock, **options), clock


def err(kind, **kw):
    return LLMError(kind, f"{kind} happened", provider="fake", model="m", **kw)


def test_first_success_is_returned_untouched():
    client, clock = make([Scripted("a", "hello")])
    response = client.generate(REQUEST)
    assert (response.text, response.model, response.attempts, response.fell_back_from) == ("hello", "a", 1, None)
    assert clock.slept == []


def test_transient_failures_are_retried_with_exponential_backoff():
    a = Scripted("a", err(base.OVERLOADED), err(base.SERVER), "finally")
    client, clock = make([a], max_attempts=3, base_delay=2.0)
    response = client.generate(REQUEST)
    assert response.text == "finally" and response.attempts == 3 and a.calls == 3
    assert clock.slept == [2.0, 4.0]


def test_backoff_is_capped():
    a = Scripted("a", err(base.SERVER), err(base.SERVER), err(base.SERVER), "ok")
    client, clock = make([a], max_attempts=4, base_delay=10.0, max_delay=15.0)
    client.generate(REQUEST)
    assert clock.slept == [10.0, 15.0, 15.0]


def test_provider_retry_hint_is_honoured():
    a = Scripted("a", err(base.RATE_LIMITED, retry_after=7.0), "ok")
    client, clock = make([a])
    client.generate(REQUEST)
    assert clock.slept == [7.5]


def test_a_wait_longer_than_the_limit_moves_on_to_the_next_model():
    a = Scripted("a", err(base.RATE_LIMITED, retry_after=600.0))
    b = Scripted("b", "from b")
    client, clock = make([a, b], max_delay=30.0)
    response = client.generate(REQUEST)
    assert response.text == "from b" and response.model == "b" and response.fell_back_from == "a"
    assert clock.slept == [] and a.calls == 1


def test_models_are_tried_in_order_after_the_attempts_are_used_up():
    a = Scripted("a", err(base.OVERLOADED), err(base.OVERLOADED))
    b = Scripted("b", "ok")
    client, clock = make([a, b], max_attempts=2)
    response = client.generate(REQUEST)
    assert response.model == "b" and response.fell_back_from == "a" and response.attempts == 3
    assert len(clock.slept) == 1


@pytest.mark.parametrize("kind", [base.BAD_REQUEST, base.QUOTA, base.BLOCKED, base.EMPTY, base.NOT_FOUND, base.FORBIDDEN, base.UNKNOWN])
def test_permanent_failures_are_not_retried_but_the_next_model_is_tried(kind):
    a = Scripted("a", err(kind))
    b = Scripted("b", "ok")
    client, clock = make([a, b])
    assert client.generate(REQUEST).model == "b"
    assert a.calls == 1 and clock.slept == []


def test_an_auth_error_stops_everything_at_once():
    a = Scripted("a", err(base.AUTH))
    b = Scripted("b", "never used")
    client, _ = make([a, b])
    with pytest.raises(LLMError) as caught:
        client.generate(REQUEST)
    assert caught.value.kind == base.AUTH and b.calls == 0
    assert caught.value.tried == [("a", base.AUTH)]


def test_when_every_model_fails_the_error_lists_them_all():
    a = Scripted("a", err(base.QUOTA))
    b = Scripted("b", err(base.RATE_LIMITED, retry_after=999.0))
    client, _ = make([a, b])
    with pytest.raises(LLMError) as caught:
        client.generate(REQUEST)
    error = caught.value
    assert error.kind == base.RATE_LIMITED
    assert error.tried == [("a", base.QUOTA), ("b", base.RATE_LIMITED)]
    assert "a: quota" in error.message and "b: rate_limited" in error.message


def test_a_single_model_error_keeps_its_own_message():
    client, _ = make([Scripted("a", err(base.BAD_REQUEST))])
    with pytest.raises(LLMError) as caught:
        client.generate(REQUEST)
    assert caught.value.message == "bad_request happened"  # no "models tried" noise for a single model


def test_max_attempts_one_means_no_retry():
    a = Scripted("a", err(base.SERVER))
    client, clock = make([a], max_attempts=1)
    with pytest.raises(LLMError):
        client.generate(REQUEST)
    assert a.calls == 1 and clock.slept == []


def test_calls_are_spaced_by_min_interval():
    a = Scripted("a", "1", "2", "3")
    client, clock = make([a], min_interval=5.0)
    client.generate(REQUEST)
    assert clock.slept == []  # the first call never waits
    clock.now += 2.0
    client.generate(REQUEST)
    assert clock.slept == [3.0]
    clock.now += 10.0
    client.generate(REQUEST)
    assert clock.slept == [3.0]  # already more than 5 s since the last call


def test_retries_are_paced_too():
    a = Scripted("a", err(base.SERVER), "ok")
    client, clock = make([a], min_interval=5.0, base_delay=1.0)
    client.generate(REQUEST)
    assert clock.slept[0] == 1.0  # the back-off
    assert sum(clock.slept) >= 5.0  # and the pacing on top of it, so two calls are never closer than 5 s


def test_jitter_stays_within_a_quarter():
    delays = []
    for jitter in (0.0, 1.0):
        a = Scripted("a", err(base.SERVER), "ok")
        client, clock = make([a], base_delay=8.0, jitter=lambda j=jitter: j)
        client.generate(REQUEST)
        delays.append(clock.slept[0])
    assert delays == [6.0, 10.0]


def test_provider_and_model_come_from_the_first_client():
    client, _ = make([Scripted("first", "x"), Scripted("second", "y")])
    assert (client.provider, client.model, client.models) == ("fake", "first", ("first", "second"))


def test_bad_construction():
    with pytest.raises(ValueError):
        ResilientClient([])
    with pytest.raises(ValueError):
        ResilientClient([Scripted("a")], max_attempts=0)


def test_concurrent_callers_are_all_served():
    import threading

    a = Scripted("a", *[f"r{i}" for i in range(8)])
    client = ResilientClient([a], sleep=lambda s: None)
    results, lock = [], threading.Lock()

    def work():
        text = client.generate(REQUEST).text
        with lock:
            results.append(text)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == sorted(f"r{i}" for i in range(8))


# ----------------------------------------------------------------------------------------------- factory


def settings(tmp_path, **env):
    return load_settings(root=tmp_path, environ={"GEMINI_API_KEY": "G-KEY", **env})


def test_gemini_chain_is_primary_then_fallbacks_without_duplicates(tmp_path):
    s = settings(tmp_path, GEMINI_MODEL="m1", GEMINI_FALLBACK_MODEL="m2,m1,m3")
    client = make_llm(s)
    assert client.models == ("m1", "m2", "m3")
    assert all(c.thinking_level == "low" for c in client._clients)


def test_model_override_gives_a_single_client(tmp_path):
    client = make_llm(settings(tmp_path), model="special")
    assert client.models == ("special",)


def test_missing_key_is_reported_with_the_variable_name(tmp_path):
    s = load_settings(root=tmp_path, environ={})
    with pytest.raises(LLMError) as caught:
        make_llm(s)
    assert caught.value.kind == base.AUTH and "GEMINI_API_KEY" in caught.value.message
    with pytest.raises(LLMError) as other:
        make_llm(s, "anthropic")
    assert "ANTHROPIC_API_KEY" in other.value.message


def test_openai_needs_a_model_name(tmp_path):
    s = load_settings(root=tmp_path, environ={"OPENAI_API_KEY": "O-KEY"})
    with pytest.raises(ConfigError, match="OPENAI_MODEL"):
        make_llm(s, "openai")
    s = load_settings(root=tmp_path, environ={"OPENAI_API_KEY": "O-KEY", "OPENAI_MODEL": "x"})
    assert make_llm(s, "openai").models == ("x",)


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        make_llm(settings(tmp_path), "skynet")


def test_the_configured_provider_is_used_by_default(tmp_path):
    s = load_settings(root=tmp_path, environ={"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "A-KEY"})
    client = make_llm(s)
    assert (client.provider, client.model) == ("anthropic", "claude-haiku-5-5")


def test_end_to_end_fallback_over_http_with_the_real_clients(tmp_path):
    ok = {"candidates": [{"content": {"parts": [{"text": "from the second model"}]}, "finishReason": "STOP"}]}
    quota = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]}}
    with MockAPI() as api:
        api.reply(429, quota)
        api.reply(200, ok)
        s = settings(tmp_path, GEMINI_MODEL="first", GEMINI_FALLBACK_MODEL="second")
        client = make_llm(s, base_urls={"gemini": api.url}, sleep=lambda s_: None)
        response = client.generate(REQUEST)
    assert response.text == "from the second model" and response.model == "second" and response.fell_back_from == "first"
    assert [r.path for r in api.requests] == ["/v1beta/models/first:generateContent", "/v1beta/models/second:generateContent"]


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_no_pause_when_the_minimum_interval_has_exactly_passed():
    a = Scripted("a", "one", "two")
    client, clock = make([a], min_interval=5.0)
    client.generate(REQUEST)
    clock.now += 5.0  # exactly the interval since the last call started
    client.generate(REQUEST)
    assert clock.slept == []


def test_a_wait_exactly_as_long_as_the_limit_is_still_waited_for():
    a = Scripted("a", err(base.RATE_LIMITED, retry_after=30.0), "after the wait")
    b = Scripted("b", "must not be used")
    client, clock = make([a, b], max_delay=30.0)
    response = client.generate(REQUEST)
    assert response.text == "after the wait" and response.model == "a" and clock.slept == [30.5] and b.calls == 0


def test_a_wait_just_over_the_limit_moves_on_to_the_next_model():
    a = Scripted("a", err(base.RATE_LIMITED, retry_after=30.1))
    b = Scripted("b", "answered")
    client, clock = make([a, b], max_delay=30.0)
    response = client.generate(REQUEST)
    assert response.model == "b" and response.fell_back_from == "a" and clock.slept == []


def test_the_final_error_remembers_the_error_that_caused_it():
    original = err(base.SERVER)
    client, _ = make([Scripted("a", original)], max_attempts=1)
    with pytest.raises(LLMError) as raised:
        client.generate(REQUEST)
    assert raised.value.__cause__ is original and raised.value is not original
