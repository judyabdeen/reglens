import pytest

from reglens.ingest.models import HEADING, PARAGRAPH
from reglens.ingest.webpage import decode_html, extract_web_page
from fixtures import GUIDANCE_PAGE as PAGE


def extract(tmp_path, html: str | bytes):
    path = tmp_path / "page.html"
    path.write_bytes(html.encode("utf-8") if isinstance(html, str) else html)
    return extract_web_page(path)


def test_title_root_and_main_content(tmp_path):
    page = extract(tmp_path, PAGE)
    assert page.title == "Guidance on AI and data protection | ICO"
    assert page.root == "main"
    text = "\n".join(b.text for b in page.blocks)
    assert "individual rights in our AI systems" in text
    assert "Individuals have the right to access their personal data. This includes training data" in text


def test_menus_banners_footers_scripts_and_hidden_text_are_dropped(tmp_path):
    text = "\n".join(b.text for b in extract(tmp_path, PAGE).blocks)
    for unwanted in ("cookies", "ICO logo", "For organisations", "Guidance", "Share on Twitter", "Copyright ICO", "tracking", "color: red", "Hidden text", "Decorative", "Skip to main content"):
        assert unwanted not in text, unwanted


def test_heading_levels_are_compressed_and_lists_and_tables_become_paragraphs(tmp_path):
    blocks = extract(tmp_path, PAGE).blocks
    headings = [(b.level, b.text) for b in blocks if b.kind == HEADING]
    # h1, h3, h4 on the page become levels 1, 2, 3
    assert headings == [
        (1, "How do we ensure individual rights in our AI systems?"),
        (2, "Right of access"),
        (3, "Practical steps"),
        (2, "Right of access"),
    ]
    paragraphs = [b.text for b in blocks if b.kind == PARAGRAPH]
    assert "• Search for the data" in paragraphs and "• Provide a copy" in paragraphs
    assert "Right | Applies when" in paragraphs and "Access | Always" in paragraphs
    assert "First paragraph inside a div." in paragraphs and "Second paragraph." in paragraphs


def test_the_same_heading_twice_in_a_row_is_kept_once(tmp_path):
    blocks = extract(tmp_path, PAGE).blocks
    texts = [b.text for b in blocks]
    index = texts.index("Text after a repeated heading that is long enough to count as real content for the test page here.")
    assert texts[index - 1] == "Right of access" and texts[index - 2] != "Right of access"


def test_page_without_main_uses_the_body(tmp_path):
    page = extract(tmp_path, "<html><body><h1>Title</h1><p>Only body text here.</p><footer>Foot</footer></body></html>")
    assert page.root == "body"
    assert [b.text for b in page.blocks] == ["Title", "Only body text here."]


def test_navigation_asides_and_footers_are_dropped_when_the_body_is_the_root(tmp_path):
    html = (
        "<html><body><nav><ul><li>Home</li><li>Menu item</li></ul></nav><h1>Title</h1><p>Real text here.</p>"
        "<aside>Related links</aside><footer>Footer text</footer></body></html>"
    )
    page = extract(tmp_path, html)
    assert page.root == "body"
    assert [b.text for b in page.blocks] == ["Title", "Real text here."]


def test_a_short_main_element_is_not_trusted(tmp_path):
    html = "<html><body><main><p>Tiny.</p></main><div><p>The real text is outside the main element.</p></div></body></html>"
    page = extract(tmp_path, html)
    assert page.root == "body"
    assert "The real text is outside the main element." in [b.text for b in page.blocks]


def test_big_wrapper_with_a_boilerplate_looking_class_name_is_kept(tmp_path):
    paragraphs = "".join(f"<p>Paragraph {n} of the real content of the page, written in full sentences.</p>" for n in range(20))
    html = f'<html><body><div class="content-with-sidebar">{paragraphs}</div></body></html>'
    texts = [b.text for b in extract(tmp_path, html).blocks]
    assert len(texts) == 20


def test_boilerplate_lines_and_entities(tmp_path):
    html = "<html><body><p>Print this page</p><p>Fish &amp; chips&nbsp;cost &pound;5.</p><p>BACK TO TOP</p></body></html>"
    assert [b.text for b in extract(tmp_path, html).blocks] == ["Fish & chips cost £5."]


def test_empty_page_gives_no_blocks(tmp_path):
    page = extract(tmp_path, "<html><head><title>Empty</title></head><body></body></html>")
    assert page.blocks == [] and page.title == "Empty"


def test_unreadable_characters_are_counted(tmp_path):
    page = extract(tmp_path, "<html><body><p>one � two (cid:5) three</p></body></html>")
    assert [b.text for b in page.blocks] == ["one two three"]
    assert page.chars["replacement_chars"] == 1 and page.chars["cid_refs"] == 1


# --------------------------------------------------------------------------------------- decoding


def test_decode_utf8_with_and_without_byte_order_mark():
    assert decode_html("café".encode("utf-8")) == "café"
    assert decode_html(b"\xef\xbb\xbf" + "café".encode("utf-8")) == "café"


def test_decode_uses_the_declared_charset_when_the_bytes_are_not_utf8():
    raw = b'<meta charset="iso-8859-1"><p>caf\xe9</p>'
    assert "café" in decode_html(raw)


@pytest.mark.parametrize("raw", [b"<p>caf\xe9 \x93quoted\x94</p>", b'<meta charset="no-such-charset"><p>caf\xe9</p>'])
def test_decode_falls_back_to_cp1252_without_crashing(raw):
    assert "café" in decode_html(raw)
