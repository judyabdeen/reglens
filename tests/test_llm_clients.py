"""The three provider clients against a local mock server that answers the way the real APIs do."""

from __future__ import annotations

import pytest

from reglens.llm import base
from reglens.llm.anthropic_messages import AnthropicClient
from reglens.llm.base import LLMError, LLMRequest, Message
from reglens.llm.gemini import GeminiClient
from reglens.llm.openai_chat import OpenAIClient
from mockapi import MockAPI

KEY = "KEY-1234567890"


@pytest.fixture()
def api():
    with MockAPI() as server:
        yield server


def gemini_ok(text="The answer [S1].", finish="STOP", **extra):
    candidate = {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
    return {
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 20, "totalTokenCount": 35},
        **extra,
    }


def gemini(api, **options):
    return GeminiClient("gemini-3.5-flash", KEY, base_url=api.url, thinking_level="low", **options)


REQUEST = LLMRequest.single("You are careful.", "What is the deadline?", max_output_tokens=500)


# ----------------------------------------------------------------------------------------------- request shape


def test_request_validation():
    with pytest.raises(ValueError):
        LLMRequest(system="", messages=())
    with pytest.raises(ValueError):
        LLMRequest(system="", messages=(Message("assistant", "hi"),))
    with pytest.raises(ValueError):
        LLMRequest.single("", "hi", max_output_tokens=0)
    with pytest.raises(ValueError):
        Message("system", "no")


def test_usage_total():
    assert base.Usage(3, 4).total_tokens == 7
    assert base.Usage().total_tokens is None
    assert base.Usage(prompt_tokens=3).total_tokens == 3


@pytest.mark.parametrize("value, expected", [("34s", 34.0), ("0.5s", 0.5), ("12", 12.0), (7, 7.0), ("soon", None), (-1, None), (True, None), (None, None)])
def test_parse_seconds(value, expected):
    assert base.parse_seconds(value) == expected


def test_scrub_removes_every_secret():
    assert base.scrub("a KEY1 b KEY2", "KEY1", None, "KEY2", "") == "a *** b ***"


# ----------------------------------------------------------------------------------------------- Gemini


def test_gemini_request_and_response(api):
    api.reply(200, gemini_ok())
    response = gemini(api).generate(REQUEST)
    sent = api.requests[0]
    assert sent.path == "/v1beta/models/gemini-3.5-flash:generateContent"
    assert sent.headers["x-goog-api-key"] == KEY
    assert KEY not in sent.path and KEY.encode() not in sent.raw  # the key travels only in the header
    assert sent.json == {
        "contents": [{"role": "user", "parts": [{"text": "What is the deadline?"}]}],
        "generationConfig": {"maxOutputTokens": 500, "thinkingConfig": {"thinkingLevel": "low"}},
        "systemInstruction": {"parts": [{"text": "You are careful."}]},
    }
    assert response.text == "The answer [S1]."
    assert (response.provider, response.model, response.finish_reason) == ("gemini", "gemini-3.5-flash", "STOP")
    assert response.usage.prompt_tokens == 10 and response.usage.output_tokens == 25  # thinking tokens are billed as output
    assert response.truncated is False and response.latency_s >= 0


def test_gemini_sends_no_temperature_unless_asked_and_no_thinking_when_unset(api):
    api.reply(200, gemini_ok())
    GeminiClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert api.requests[0].json["generationConfig"] == {"maxOutputTokens": 500}
    api.reply(200, gemini_ok())
    GeminiClient("m", KEY, base_url=api.url).generate(LLMRequest.single("", "q", temperature=0.2))
    assert api.requests[1].json["generationConfig"]["temperature"] == 0.2
    assert "systemInstruction" not in api.requests[1].json  # an empty system prompt is not sent


def test_gemini_conversation_roles(api):
    api.reply(200, gemini_ok())
    request = LLMRequest(system="s", messages=(Message("user", "a"), Message("assistant", "b"), Message("user", "c")))
    gemini(api).generate(request)
    assert [c["role"] for c in api.requests[0].json["contents"]] == ["user", "model", "user"]


def test_gemini_model_prefix_and_odd_characters_are_handled(api):
    api.reply(200, gemini_ok())
    GeminiClient("models/gemini-3.1-flash-lite", KEY, base_url=api.url).generate(REQUEST)
    assert api.requests[0].path == "/v1beta/models/gemini-3.1-flash-lite:generateContent"


def test_gemini_thought_summaries_are_not_part_of_the_answer(api):
    parts = [{"text": "thinking about it", "thought": True}, {"text": "Real "}, {"text": "answer."}]
    api.reply(200, {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]})
    assert gemini(api).generate(REQUEST).text == "Real answer."


def test_gemini_retries_without_thinking_when_the_model_refuses_it(api):
    api.reply(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Thinking level 'low' is not supported for this model."}})
    api.reply(200, gemini_ok())
    response = gemini(api).generate(REQUEST)
    assert response.text
    assert len(api.requests) == 2
    assert "thinkingConfig" in api.requests[0].json["generationConfig"]
    assert "thinkingConfig" not in api.requests[1].json["generationConfig"]


def test_gemini_drops_each_refused_setting_only_once(api):
    refusal = {"error": {"status": "INVALID_ARGUMENT", "message": "Thinking is not supported"}}
    api.reply(400, refusal, times=3)
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.BAD_REQUEST
    assert len(api.requests) == 2  # the second request had nothing left to drop, so the error is raised


def test_gemini_drops_temperature_when_refused(api):
    api.reply(400, {"error": {"message": "temperature is not supported for this model"}})
    api.reply(200, gemini_ok())
    gemini(api).generate(LLMRequest.single("", "q", temperature=0.1))
    assert "temperature" not in api.requests[1].json["generationConfig"]


def test_gemini_invalid_key_is_an_auth_error_without_the_key(api):
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": f"API key not valid ({KEY}). Please pass a valid API key.", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}}
    api.reply(400, body)
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.AUTH
    assert KEY not in caught.value.message and KEY not in str(caught.value)
    assert (caught.value.provider, caught.value.model) == ("gemini", "gemini-3.5-flash")


def test_gemini_rate_limit_carries_the_retry_delay(api):
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for metric ...", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
    ]}}
    api.reply(429, body)
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.RATE_LIMITED and caught.value.retry_after == 34.0 and caught.value.retryable


def test_gemini_daily_quota_is_not_retryable(api):
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
    ]}}
    api.reply(429, body)
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.QUOTA and not caught.value.retryable


def test_gemini_blocked_prompt(api):
    api.reply(200, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.BLOCKED and "SAFETY" in caught.value.message


def test_gemini_no_candidates_is_empty(api):
    api.reply(200, {"candidates": []})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.EMPTY


def test_gemini_thinking_that_eats_the_output_budget_gives_a_helpful_error(api):
    api.reply(200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.EMPTY and "GEMINI_THINKING_LEVEL" in caught.value.message


def test_gemini_partial_answers_are_marked_truncated(api):
    api.reply(200, gemini_ok("Half an ans", finish="MAX_TOKENS"))
    assert gemini(api).generate(REQUEST).truncated is True
    api.reply(200, gemini_ok("Quoted text", finish="RECITATION"))
    assert gemini(api).generate(REQUEST).truncated is True


def test_gemini_policy_stop_without_text_is_blocked(api):
    api.reply(200, {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.BLOCKED


@pytest.mark.parametrize("status, kind", [(401, base.AUTH), (403, base.FORBIDDEN), (404, base.NOT_FOUND), (500, base.SERVER), (503, base.OVERLOADED), (418, base.UNKNOWN)])
def test_gemini_status_codes(api, status, kind):
    api.reply(status, {"error": {"message": "x"}})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == kind and caught.value.status == status


def test_gemini_html_error_page_is_still_classified(api):
    api.reply(502, "<html>Bad gateway</html>")
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.SERVER and "502" in caught.value.message


def test_gemini_success_that_is_not_json_is_a_server_error(api):
    api.reply(200, "<html>captive portal</html>")
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.SERVER and "not JSON" in caught.value.message


def test_unreachable_server_is_a_retryable_network_error():
    # Linux refuses the connection at once (network); Windows keeps retrying a refused connection for about two seconds,
    # so the one-second limit of this test fires first (timeout). Both are retryable and neither leaks the key.
    client = GeminiClient("m", KEY, base_url="http://127.0.0.1:9", timeout=(1.0, 1.0))
    with pytest.raises(LLMError) as caught:
        client.generate(REQUEST)
    assert caught.value.kind in (base.NETWORK, base.TIMEOUT) and caught.value.retryable
    assert KEY not in caught.value.message


def test_slow_server_is_a_timeout(api):
    import threading
    import time

    gate = threading.Event()
    original = api._server.RequestHandlerClass.do_POST

    def slow(self):
        gate.wait(3)
        original(self)

    api._server.RequestHandlerClass.do_POST = slow
    try:
        with pytest.raises(LLMError) as caught:
            GeminiClient("m", KEY, base_url=api.url, timeout=(1.0, 0.3)).generate(REQUEST)
        assert caught.value.kind == base.TIMEOUT
    finally:
        gate.set()
        time.sleep(0.05)
        api._server.RequestHandlerClass.do_POST = original


def test_a_client_needs_a_key_and_never_shows_it():
    with pytest.raises(LLMError) as caught:
        GeminiClient("m", "")
    assert caught.value.kind == base.AUTH
    assert KEY not in repr(GeminiClient("m", KEY))


# ----------------------------------------------------------------------------------------------- OpenAI


def openai_ok(text="Answer [S1]", finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": {"prompt_tokens": 11, "completion_tokens": 6}}


def test_openai_request_and_response(api):
    api.reply(200, openai_ok())
    client = OpenAIClient("some-model", KEY, base_url=api.url)
    response = client.generate(REQUEST)
    sent = api.requests[0]
    assert sent.path == "/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert sent.json == {
        "model": "some-model",
        "messages": [{"role": "system", "content": "You are careful."}, {"role": "user", "content": "What is the deadline?"}],
        "max_completion_tokens": 500,
    }
    assert response.text == "Answer [S1]" and response.usage.prompt_tokens == 11 and response.usage.output_tokens == 6
    assert response.provider == "openai" and not response.truncated


def test_openai_drops_a_refused_temperature(api):
    api.reply(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.1 with this model.", "type": "invalid_request_error"}})
    api.reply(200, openai_ok())
    OpenAIClient("m", KEY, base_url=api.url).generate(LLMRequest.single("", "q", temperature=0.1))
    assert "temperature" in api.requests[0].json and "temperature" not in api.requests[1].json


def test_openai_quota_and_rate_limit_are_told_apart(api):
    api.reply(429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}})
    api.reply(429, {"error": {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}}, headers={"Retry-After": "7"})
    client = OpenAIClient("m", KEY, base_url=api.url)
    with pytest.raises(LLMError) as first:
        client.generate(REQUEST)
    with pytest.raises(LLMError) as second:
        client.generate(REQUEST)
    assert first.value.kind == base.QUOTA
    assert second.value.kind == base.RATE_LIMITED and second.value.retry_after == 7.0


def test_openai_refusal_and_truncation(api):
    api.reply(200, {"choices": [{"message": {"content": None, "refusal": "I can't help with that."}, "finish_reason": "stop"}]})
    with pytest.raises(LLMError) as caught:
        OpenAIClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert caught.value.kind == base.BLOCKED
    api.reply(200, openai_ok("cut off", finish="length"))
    assert OpenAIClient("m", KEY, base_url=api.url).generate(REQUEST).truncated is True


def test_openai_empty_choices(api):
    api.reply(200, {"choices": []})
    with pytest.raises(LLMError) as caught:
        OpenAIClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert caught.value.kind == base.EMPTY


# ----------------------------------------------------------------------------------------------- Anthropic


def anthropic_ok(text="Answer [S1]", stop="end_turn"):
    return {"type": "message", "role": "assistant", "content": [{"type": "text", "text": text}], "stop_reason": stop, "usage": {"input_tokens": 12, "output_tokens": 7}}


def test_anthropic_request_and_response(api):
    api.reply(200, anthropic_ok())
    response = AnthropicClient("claude-haiku-5-5", KEY, base_url=api.url).generate(REQUEST)
    sent = api.requests[0]
    assert sent.path == "/v1/messages"
    assert sent.headers["x-api-key"] == KEY and sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.json == {
        "model": "claude-haiku-5-5",
        "max_tokens": 500,
        "system": "You are careful.",
        "messages": [{"role": "user", "content": "What is the deadline?"}],
    }
    assert response.text == "Answer [S1]" and response.usage.prompt_tokens == 12 and response.usage.output_tokens == 7
    assert response.provider == "anthropic" and not response.truncated


def test_anthropic_joins_text_blocks_and_ignores_others(api):
    blocks = [{"type": "text", "text": "One. "}, {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Two."}]
    api.reply(200, {"content": blocks, "stop_reason": "end_turn"})
    assert AnthropicClient("m", KEY, base_url=api.url).generate(REQUEST).text == "One. Two."


@pytest.mark.parametrize(
    "status, error_type, kind",
    [(401, "authentication_error", base.AUTH), (403, "permission_error", base.FORBIDDEN), (404, "not_found_error", base.NOT_FOUND),
     (429, "rate_limit_error", base.RATE_LIMITED), (529, "overloaded_error", base.OVERLOADED), (400, "invalid_request_error", base.BAD_REQUEST), (500, "api_error", base.SERVER)],
)
def test_anthropic_error_types(api, status, error_type, kind):
    api.reply(status, {"type": "error", "error": {"type": error_type, "message": "nope"}}, headers={"retry-after": "3"})
    with pytest.raises(LLMError) as caught:
        AnthropicClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert caught.value.kind == kind
    assert caught.value.retry_after == 3.0


def test_anthropic_truncation_refusal_and_empty(api):
    api.reply(200, anthropic_ok("cut", stop="max_tokens"))
    assert AnthropicClient("m", KEY, base_url=api.url).generate(REQUEST).truncated is True
    api.reply(200, {"content": [], "stop_reason": "refusal"})
    with pytest.raises(LLMError) as blocked:
        AnthropicClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert blocked.value.kind == base.BLOCKED
    api.reply(200, {"content": [], "stop_reason": "end_turn"})
    with pytest.raises(LLMError) as empty:
        AnthropicClient("m", KEY, base_url=api.url).generate(REQUEST)
    assert empty.value.kind == base.EMPTY


def test_anthropic_drops_a_refused_temperature(api):
    api.reply(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` may only be set to 1 when thinking is enabled"}})
    api.reply(200, anthropic_ok())
    AnthropicClient("m", KEY, base_url=api.url).generate(LLMRequest.single("", "q", temperature=0.0))
    assert "temperature" not in api.requests[1].json


# ------------------------------------------------------------------ boundaries found by mutation testing


def test_gemini_model_names_are_url_encoded(api):
    api.reply(200, gemini_ok())
    GeminiClient("odd model/name?x=1", KEY, base_url=api.url).generate(REQUEST)
    assert api.requests[0].path == "/v1beta/models/odd%20model%2Fname%3Fx%3D1:generateContent"


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Request contains an invalid argument.", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}},
        {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}},
    ],
    ids=["reason only", "message only"],
)
def test_gemini_key_problem_is_recognised_from_the_reason_alone_or_from_the_message_alone(api, body):
    api.reply(400, body)
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.AUTH


def test_gemini_quota_is_recognised_from_the_status_in_the_body_even_without_http_429(api):
    api.reply(400, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Too many requests per minute."}})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.RATE_LIMITED and caught.value.retryable


def test_gemini_daily_limit_is_recognised_from_the_message_when_there_are_no_details(api):
    api.reply(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "You exceeded the free-tier limit of 500 requests per day."}})
    with pytest.raises(LLMError) as caught:
        gemini(api).generate(REQUEST)
    assert caught.value.kind == base.QUOTA and not caught.value.retryable


def test_gemini_answer_is_trimmed(api):
    api.reply(200, gemini_ok("  \n The answer [S1].  \n"))
    assert gemini(api).generate(REQUEST).text == "The answer [S1]."


def test_gemini_without_usage_information_reports_no_token_counts(api):
    api.reply(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})
    usage = gemini(api).generate(REQUEST).usage
    assert usage.prompt_tokens is None and usage.output_tokens is None and usage.total_tokens is None


def test_gemini_reports_how_long_the_call_took():
    import time

    class Reply:
        status_code, headers = 200, {}

        def json(self):
            return gemini_ok()

    def slow_post(url, **kwargs):
        time.sleep(0.03)
        return Reply()

    response = GeminiClient("m", KEY, base_url="http://unused.invalid", post=slow_post).generate(REQUEST)
    assert 0.03 <= response.latency_s < 2.0
