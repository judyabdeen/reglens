"""The page in a real browser (Chromium, driven by Playwright), against the real server with an invented corpus.

These tests need ``pip install playwright`` and ``playwright install chromium``; without them they are skipped.
They check what no other test can: that the scripts run under the server's Content-Security-Policy without a single
console error, and that a person can ask, read the sources, stop a slow answer, change the scope and save the result.
The model is scripted, so nothing here costs quota and no answer depends on a provider.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from ragfixtures import FakeLLM
from webfixtures import BREACH, LiveServer, build_the_index, make_service, scripted, settings_for

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from reglens.embeddings import HashingEmbedder  # noqa: E402
from reglens.generation.prompts import SYSTEM_PROMPT  # noqa: E402
from reglens.runtime import build_runtime  # noqa: E402
from reglens.web.app import create_app  # noqa: E402
from reglens.web.service import WebService  # noqa: E402

pytestmark = pytest.mark.browser

PAGE_TIMEOUT_MS = 15_000


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - the browser program itself is missing
            pytest.skip(f"Chromium is not available to Playwright ({type(exc).__name__}); install it with: playwright install chromium")
        yield chromium
        chromium.close()


@contextmanager
def running(browser, service, *, width=1280, height=900):
    """The real server on a free port, and a browser page opened on it with everything it sends and logs recorded."""
    with LiveServer(create_app(service)) as server:
        context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce", accept_downloads=True)
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        problems, asked = [], []
        page.on("console", lambda message: problems.append(f"{message.type}: {message.text}") if message.type in ("error", "warning") else None)
        page.on("pageerror", lambda error: problems.append(f"page error: {error}"))
        page.on("request", lambda request: asked.append(request.post_data_json) if request.url.endswith("/api/ask") else None)
        page.goto(f"{server.base}/")
        try:
            yield SimpleNamespace(page=page, base=server.base, asked=asked, service=service)
            assert problems == [], f"the page logged problems: {problems}"  # the Content-Security-Policy shows up here too
        finally:
            context.close()


def library(tmp_path, llm="scripted"):
    llm = scripted() if llm == "scripted" else llm
    return make_service(tmp_path, llm)


def ask(page, question):
    page.fill("#question", question)
    page.keyboard.press("Enter")


def text(page, selector):
    return " ".join(page.locator(selector).first.inner_text().split())


# ---------------------------------------------------------------------------------------------- the first screen


def test_the_first_screen_explains_the_library_and_offers_questions(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        page.wait_for_selector(".welcome")
        assert text(page, "#bar-status") == "6 documents, 10 passages, fake-1"
        assert page.locator(".example").count() == 5
        assert page.locator("#ask").is_disabled() and page.locator("#new-chat").is_disabled()
        page.keyboard.press("Tab")  # the first stop of the page is the skip link, which leads to the question box
        assert page.evaluate("document.activeElement.className") == "skip"
        page.keyboard.press("Enter")
        assert page.evaluate("document.activeElement.id") == "question"
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("/")  # the usual shortcut for "search"
        assert page.evaluate("document.activeElement.id") == "question"


# ---------------------------------------------------------------------------------------------- asking


def test_a_question_gets_a_cited_answer_and_its_source_is_one_click_away(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .prose")
        assert app.asked[0]["mode"] == "rag" and app.asked[0]["history"] == []
        assert "72 hours" in text(page, ".turn .prose")
        assert page.locator(".turn .prose .cite").count() >= 1
        assert page.locator(".turn .margin .note").count() >= 1
        assert "General Data Protection Regulation" in text(page, ".turn .margin .note")
        assert "Automatic checks found no problem" in text(page, ".turn .checks")

        page.locator(".turn .prose .cite").first.click()
        assert page.evaluate("document.activeElement.classList.contains('note')")

        opener = page.locator(".turn .margin .note").first.get_by_role("button", name=re.compile("Read the full passage"))
        opener.click()
        page.wait_for_selector("#reader[open] .reader-text")
        assert "Article 33" in text(page, "#reader") and page.evaluate("document.getElementById('reader').contains(document.activeElement)")
        page.keyboard.press("Escape")
        assert page.evaluate("document.activeElement.textContent.includes('Read the full passage')")


def test_a_follow_up_question_carries_the_earlier_exchange_without_its_citation_marks(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .prose")
        ask(page, "And in the US?")
        expect(page.locator(".turn")).to_have_count(2)  # the checks below wait for the page; a script string would need eval, which the policy forbids
        expect(page.locator(".turn .working")).to_have_count(0)
        history = app.asked[1]["history"]
        assert len(history) == 1 and history[0]["question"] == BREACH
        assert "72 hours" in history[0]["answer"] and "[S1]" not in history[0]["answer"]


def test_compare_shows_the_model_alone_beside_the_documents_and_lists_the_figures(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        page.click('.mode:has(input[value="compare"])')
        ask(page, BREACH)
        page.wait_for_selector(".turn .baseline .prose")
        assert app.asked[0]["mode"] == "compare"
        assert "Within 72 hours" in text(page, ".turn .baseline")
        assert "72" in text(page, ".turn .figures")


def test_passages_mode_does_not_ask_the_model(browser, tmp_path):
    llm = scripted()
    with running(browser, make_service(tmp_path, llm)) as app:
        page = app.page
        page.click('.mode:has(input[value="passages"])')
        ask(page, BREACH)
        page.wait_for_selector(".turn .passage-list .note")
        assert llm.requests == [] and page.locator(".turn .problem").count() == 0


def test_enter_asks_and_shift_enter_starts_a_new_line(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        page.fill("#question", "first line")
        page.keyboard.press("Shift+Enter")
        page.keyboard.type("second line")
        assert page.input_value("#question") == "first line\nsecond line" and app.asked == []
        page.keyboard.press("Enter")
        page.wait_for_selector(".turn")
        assert app.asked[0]["question"] == "first line\nsecond line"


# ---------------------------------------------------------------------------------------------- when things go wrong


def test_stop_cancels_a_slow_answer_and_asking_again_works(browser, tmp_path):
    def slow(request):
        if request.system == SYSTEM_PROMPT:
            time.sleep(2.5)
            return "Notify within 72 hours [S1]."
        return "How quickly must a data breach be reported?"

    with running(browser, make_service(tmp_path, FakeLLM(slow))) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .working")
        assert text(page, "#ask") == "Stop"
        page.click("#ask")
        page.wait_for_selector(".turn .problem")
        assert "You stopped this question" in text(page, ".turn .problem") and text(page, "#ask") == "Ask"
        page.click(".turn .problem .btn")  # "Ask again"
        page.wait_for_selector(".turn .prose", timeout=20_000)
        assert "72 hours" in text(page, ".turn .prose")


def test_new_conversation_during_a_running_question_leaves_no_stopped_turn_behind(browser, tmp_path):
    def slow(request):
        if request.system == SYSTEM_PROMPT:
            time.sleep(1.5)
        return "Notify within 72 hours [S1]."

    with running(browser, make_service(tmp_path, FakeLLM(slow))) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .working")
        page.click("#new-chat")  # the first click asks for confirmation
        page.click("#new-chat")
        page.wait_for_selector(".welcome")
        page.wait_for_timeout(600)
        assert page.locator(".turn").count() == 0 and text(page, "#ask") == "Ask"


def test_without_an_api_key_the_page_offers_the_matching_passages(browser, tmp_path):
    with running(browser, make_service(tmp_path, None)) as app:
        page = app.page
        page.wait_for_selector(".welcome")
        assert "No language model is connected" in text(page, ".notice")
        disabled = page.eval_on_selector_all("#modes input", "inputs => Object.fromEntries(inputs.map(i => [i.value, i.disabled]))")
        assert disabled == {"rag": True, "compare": True, "passages": False, "llm_only": True}
        assert page.locator('#modes input[value="passages"]').is_checked()
        ask(page, BREACH)
        page.wait_for_selector(".turn .passage-list .note")
        assert "General Data Protection Regulation" in text(page, ".turn .passage-list")


def test_a_failing_model_still_shows_the_passages_with_a_plain_explanation(browser, tmp_path):
    from reglens.llm.base import LLMError

    def refuse(request):
        if request.system == SYSTEM_PROMPT:
            raise LLMError("quota", "quota exhausted")
        return "How quickly must a data breach be reported?"

    with running(browser, make_service(tmp_path, FakeLLM(refuse))) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .problem-soft")
        assert text(page, ".turn .problem-soft .problem-title") == "The language model was not used"
        assert "quota" in text(page, ".turn .problem-soft").lower()
        assert page.locator(".turn .passage-list .note").count() >= 1


def test_a_missing_index_is_explained_and_the_page_opens_when_it_is_built(browser, tmp_path):
    settings = settings_for(tmp_path)
    def factory(configured):
        return build_runtime(configured, embedder=HashingEmbedder(), llm=scripted())

    service = WebService(settings, runtime_factory=factory, retry_seconds=0)  # nothing was indexed yet
    with running(browser, service) as app:
        page = app.page
        page.wait_for_selector(".setup")
        assert text(page, ".setup h1") == "The library has not been built yet"
        assert "python scripts/build_index.py" in text(page, ".setup code")
        assert page.locator("#question").is_disabled()
        build_the_index(settings)  # the person runs the command in another window
        page.wait_for_selector(".welcome", timeout=12_000)  # the page checks again every few seconds
        assert page.locator("#question").is_enabled()


# ---------------------------------------------------------------------------------------------- settings and export


def test_scope_and_settings_are_sent_with_the_question_and_remembered(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        page.click("#scope-open")
        page.wait_for_selector("#scope[open] .pane")
        page.locator('#scope .group:has(legend:has-text("Where it comes from")) .opt:has-text("US") input').check()
        page.locator('#scope .group:has(legend:has-text("Search method")) .opt:has-text("Keywords only") input').check()
        assert text(page, ".scope-summary") == "Searching US: 2 documents"
        page.keyboard.press("Escape")
        assert page.evaluate("document.activeElement.id") == "scope-open"
        page.reload()
        page.wait_for_selector(".welcome")
        assert "US: 2 documents" in text(page, "#scope-text")
        ask(page, BREACH)
        page.wait_for_selector(".turn .prose, .turn .verdict")
        sent = app.asked[-1]
        assert sent["filters"] == {"jurisdiction": ["US"]} and sent["options"]["search"] == "lexical"
        assert text(page, ".turn .scope") == "Searched US: 2 documents"


def test_the_conversation_can_be_saved_as_markdown_with_its_sources(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        ask(page, BREACH)
        page.wait_for_selector(".turn .prose")
        with page.expect_download() as download:
            page.click("#save-chat")
        assert re.fullmatch(r"reglens-\d{4}-\d{2}-\d{2}\.md", download.value.suggested_filename)
        saved = open(download.value.path(), encoding="utf-8").read()
        assert saved.startswith("# RegLens conversation") and BREACH in saved and "Sources" in saved and "[S1]" in saved


def test_the_theme_choice_is_kept_and_applied_before_the_page_is_drawn(browser, tmp_path):
    with running(browser, library(tmp_path)) as app:
        page = app.page
        page.click("#theme")
        assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")
        chosen = page.evaluate("document.documentElement.dataset.theme")
        page.reload()
        assert page.evaluate("document.documentElement.dataset.theme") == chosen


# ---------------------------------------------------------------------------------------------- small screens


def test_the_page_fits_a_phone_without_sideways_scrolling(browser, tmp_path):
    with running(browser, library(tmp_path), width=390, height=844) as app:
        page = app.page
        page.wait_for_selector(".welcome")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        ask(page, BREACH)
        page.wait_for_selector(".turn .prose")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        box = page.locator("#question").bounding_box()
        assert box is not None and 0 <= box["x"] and box["x"] + box["width"] <= 390 and box["y"] + box["height"] <= 844
        assert page.locator(".turn .margin").count() == 0 or page.locator(".turn .note").first.is_visible()
