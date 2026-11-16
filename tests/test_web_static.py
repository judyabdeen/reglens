"""The files of the web page: what ``index.html`` loads exists and is served, and the scripts obey the server's rules.

The server sends a strict Content-Security-Policy (no inline scripts or styles, nothing from other sites). A page that
broke those rules would work in a quick test and then fail silently in the browser, so these checks keep the files and
the policy in step without needing a browser. (The page's behaviour is tested in ``tests/js`` and, with a real browser,
by the checks described in the README.)
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest
import requests
from webfixtures import LiveServer, make_service, scripted

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from reglens.retrieval import MODES  # noqa: E402
from reglens.web.app import CONTENT_SECURITY_POLICY, SAFE_NAME, SECURITY_HEADERS, STATIC_DIR, STATIC_TYPES, create_app  # noqa: E402
from reglens.web.service import ASK_MODES, FILTER_KEYS  # noqa: E402

SCRIPTS = sorted([*STATIC_DIR.glob("*.mjs"), *STATIC_DIR.glob("*.js")])
# import {a, b} from "./x.mjs";   export {a} from "./x.mjs";   import "./x.mjs";   import("./x.mjs")
IMPORT = re.compile(
    r"""^\s*(?:import|export)\b[^;"']*?\bfrom\s*["']([^"']+)["']|^\s*import\s*["']([^"']+)["']|\bimport\(\s*["']([^"']+)["']\s*\)""",
    re.M,
)


class PageParser(HTMLParser):
    """Collects what matters for the security rules and for people who use a screen reader."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.title = ""
        self.refs: list[tuple[str, str, str]] = []  # tag, attribute, value
        self.inline_scripts = 0
        self.style_tags = 0
        self.style_attributes: list[str] = []
        self.event_attributes: list[str] = []
        self.ids: list[str] = []
        self.label_targets: list[str] = []
        self.dialogs: list[dict] = []
        self.mains = 0
        self.viewport = False
        self.live_regions: list[dict] = []
        self.unnamed_buttons: list[str] = []
        self._in = []
        self._button: dict | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self._in.append(tag)
        if tag == "html":
            self.lang = attributes.get("lang")
        if tag == "main":
            self.mains += 1
        if tag == "meta" and attributes.get("name") == "viewport":
            self.viewport = True
        if tag == "style":
            self.style_tags += 1
        if tag == "script" and "src" not in attributes:
            self.inline_scripts += 1
        for name in ("src", "href", "action", "formaction", "data", "poster", "srcset"):
            if name in attributes and attributes[name] is not None:
                self.refs.append((tag, name, attributes[name]))
        if "style" in attributes:
            self.style_attributes.append(tag)
        self.event_attributes += [f"{tag}[{name}]" for name in attributes if name.startswith("on")]
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "label" and attributes.get("for"):
            self.label_targets.append(attributes["for"])
        if tag == "dialog":
            self.dialogs.append(attributes)
        if "aria-live" in attributes:
            self.live_regions.append(attributes)
        if tag == "button":
            # A button may be named by aria-label, by its text (the script paints the text of the Ask button) or by its title.
            self._button = {"id": attributes.get("id", "(no id)"), "label": attributes.get("aria-label") or attributes.get("title") or "", "text": ""}

    def handle_endtag(self, tag):
        if self._in and self._in[-1] == tag:
            self._in.pop()
        if tag == "button" and self._button is not None:
            if not (self._button["label"].strip() or self._button["text"].strip()):
                self.unnamed_buttons.append(self._button["id"])
            self._button = None

    def handle_data(self, data):
        if self._in and self._in[-1] == "title":
            self.title += data
        if self._in and self._in[-1] == "script" and data.strip():
            self.inline_scripts += 1
        if self._button is not None:
            self._button["text"] += data


def parse_page() -> PageParser:
    parser = PageParser()
    parser.feed((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    parser.close()
    return parser


def own_files(page: PageParser) -> list[str]:
    """The files of the static folder that the page asks for, as the server names them (``app.css``)."""
    return sorted({value.removeprefix("/static/") for tag, name, value in page.refs if value.startswith("/static/")})


def imports_of(path) -> list[str]:
    return [next(group for group in found if group) for found in IMPORT.findall(path.read_text(encoding="utf-8"))]


def needed_files() -> set[str]:
    """Everything the browser will fetch: what the page names, and the modules those modules import, and so on."""
    found: set[str] = set()
    pending = own_files(parse_page())
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        if name.endswith((".mjs", ".js")):
            pending += [ref.removeprefix("./") for ref in imports_of(STATIC_DIR / name) if ref.startswith("./")]
    return found


# ---------------------------------------------------------------------------------------------- what the page loads


def test_the_page_loads_only_files_of_its_own_static_folder():
    page = parse_page()
    for tag, name, value in page.refs:
        parts = urlsplit(value)
        assert not parts.scheme and not parts.netloc, f"<{tag} {name}> points to another site: {value}"
        if tag in ("script", "link", "img"):
            assert value.startswith("/static/"), f"<{tag} {name}={value!r}> is not served by the static folder"
        else:
            assert value == "/" or value.startswith("#"), f"<{tag} {name}={value!r}> leaves the page"
    assert own_files(page), "the page loads nothing from its static folder"


def test_the_page_follows_the_content_security_policy():
    page = parse_page()
    assert page.inline_scripts == 0, "an inline script would be blocked by script-src 'self'"
    assert page.style_tags == 0 and page.style_attributes == [], f"inline styles would be blocked by style-src 'self': {page.style_attributes}"
    assert page.event_attributes == [], f"inline event handlers would be blocked: {page.event_attributes}"
    for directive in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "frame-ancestors 'none'"):
        assert directive in CONTENT_SECURITY_POLICY
    assert "unsafe" not in CONTENT_SECURITY_POLICY


def test_every_file_the_page_needs_exists_and_can_be_named_by_the_server():
    needed = needed_files()
    assert {"app.css", "app.mjs", "theme.js", "favicon.svg"} <= needed
    for name in needed:
        assert (STATIC_DIR / name).is_file(), f"the page needs /static/{name} but the file is missing"
        assert SAFE_NAME.fullmatch(name), f"the server would refuse the name {name}"
        assert "." + name.rsplit(".", 1)[-1] in STATIC_TYPES, f"the server has no content type for {name}"


def test_modules_import_only_each_other_by_relative_name():
    for path in SCRIPTS:
        for reference in imports_of(path):
            assert reference.startswith("./") and not reference.startswith("./../"), f"{path.name} imports {reference!r}"
            assert (STATIC_DIR / reference.removeprefix("./")).is_file(), f"{path.name} imports {reference} which does not exist"


def test_the_stylesheet_loads_nothing_from_outside():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    assert "@import" not in css and "expression(" not in css
    assert not re.search(r"https?:", css), "the stylesheet names another site"
    for target in re.findall(r"url\(\s*([^)]*)\)", css):
        assert target.strip("\"' ").startswith("data:"), f"url({target}) would load a file from outside"


def test_the_scripts_never_turn_text_into_markup():
    """Answers and passages come from a model and from documents: they are always inserted as text, never as markup."""
    forbidden = {
        r"\.innerHTML": "innerHTML",
        r"\.outerHTML": "outerHTML",
        r"insertAdjacentHTML": "insertAdjacentHTML",
        r"document\.write": "document.write",
        r"\beval\s*\(": "eval",
        r"new\s+Function\b": "new Function",
        r"setAttribute\(\s*[\"']style": "a style attribute (the policy blocks it; use the CSSOM)",
        r"setAttribute\(\s*[\"']on": "an event-handler attribute",
        r"srcdoc": "srcdoc",
        r"javascript:": "a javascript: address",
        r"https?://(?!www\.w3\.org/2000/svg)": "an address on another site",
    }
    assert SCRIPTS, "no scripts found"
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        for pattern, meaning in forbidden.items():
            assert not re.search(pattern, text), f"{path.name} uses {meaning}"


def test_the_page_has_the_structure_a_screen_reader_needs():
    page = parse_page()
    assert page.lang == "en" and page.title.strip() and page.viewport and page.mains == 1
    assert len(page.ids) == len(set(page.ids)), "duplicate ids"
    for target in page.label_targets:
        assert target in page.ids, f"a label points to the missing id {target}"
    assert page.dialogs, "no dialogs found"
    for dialog in page.dialogs:
        assert dialog.get("aria-labelledby") in page.ids, f"dialog {dialog.get('id')} has no title to announce"
    assert any(region.get("role") == "status" and region.get("aria-live") == "polite" for region in page.live_regions)
    assert page.unnamed_buttons == [], f"buttons without a name: {page.unnamed_buttons}"
    assert "question" in page.ids, "the skip link needs the question box"


# ---------------------------------------------------------------------------------------------- page and server agree


def test_the_search_methods_of_the_page_are_the_ones_the_server_knows():
    block = re.search(r"export const SEARCH_METHODS = \[(.*?)\n\];", (STATIC_DIR / "lib.mjs").read_text(encoding="utf-8"), re.S)
    assert block, "SEARCH_METHODS not found in lib.mjs"
    assert re.findall(r'\bvalue:\s*"(\w+)"', block.group(1)) == list(MODES)


def test_the_modes_of_the_page_are_the_ones_the_server_knows():
    app = (STATIC_DIR / "app.mjs").read_text(encoding="utf-8")
    assert re.findall(r'\{ value: "(\w+)", label: "[^"]+", needsModel', app) == list(ASK_MODES)


def test_the_filters_of_the_page_are_the_ones_the_server_knows():
    lib = (STATIC_DIR / "lib.mjs").read_text(encoding="utf-8")
    lists = re.findall(r'\[("jurisdiction"[^\]]*)\]', lib)
    assert lists, "the filter names were not found in lib.mjs"
    for found in lists:
        assert re.findall(r'"(\w+)"', found) == list(FILTER_KEYS)


def test_the_real_server_serves_every_file_of_the_page(tmp_path):
    service = make_service(tmp_path, scripted())
    with LiveServer(create_app(service)) as server:  # the real static folder
        page = requests.get(f"{server.base}/", timeout=10)
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert 'id="question"' in page.text
        files = sorted(needed_files() | {path.name for path in STATIC_DIR.iterdir() if path.is_file()})
        for name in files:
            response = requests.get(f"{server.base}/static/{name}", timeout=10)
            assert response.status_code == 200, f"/static/{name} answered {response.status_code}"
            assert response.headers["content-type"] == STATIC_TYPES["." + name.rsplit(".", 1)[-1]], name
            assert response.content == (STATIC_DIR / name).read_bytes()
            for header, value in SECURITY_HEADERS:
                assert response.headers[header] == value, f"{header} is missing on /static/{name}"
            assert response.headers["etag"]
