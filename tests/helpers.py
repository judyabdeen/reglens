"""Test helpers: build tiny but valid PDFs without any PDF library, and describe how fixture lines are typeset."""

from __future__ import annotations

BODY_SIZE = 10.0


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Styled(str):
    """A line of a fixture page that is set in a particular type: ``size`` in points, ``bold`` or not."""

    size: float
    bold: bool

    def __new__(cls, text: str, size: float = BODY_SIZE, bold: bool = False):
        line = super().__new__(cls, text)
        line.size = size
        line.bold = bold
        return line


def small(text: str, size: float = 8.0) -> Styled:
    """A footnote line: smaller than the running text."""
    return Styled(text, size)


def bold(text: str, size: float = BODY_SIZE) -> Styled:
    """A heading line set in bold."""
    return Styled(text, size, True)


def typeset(pages: list[list[str]]) -> tuple[list[str], list[list[tuple[float, bool] | None]]]:
    """What the PDF reader hands over for fixture pages: one string per page and a ``(size, bold)`` per line.

    Plain strings are running text; ``small`` and ``bold`` lines carry their own type. Empty lines have no style.
    """
    texts, styles = [], []
    for lines in pages:
        texts.append("\n".join(lines))
        styles.append([
            (line.size, line.bold) if isinstance(line, Styled) else ((BODY_SIZE, False) if line.strip() else None)
            for line in lines
        ])
    return texts, styles


def make_pdf(lines: list[str] | None, min_size: int = 6000, font_extra: str = "") -> bytes:
    """Return a one-page PDF containing ``lines`` of text (or no text at all if None).

    Padding comments are added so the file passes the downloader's minimum-size check.
    ``font_extra`` is inserted into the font dictionary (for example ``/Encoding /SymbolSetEncoding``).
    """
    if lines:
        commands = ["BT", "/F1 10 Tf"]
        y = 760
        for line in lines:
            commands.append(f"1 0 0 1 40 {y} Tm ({_pdf_escape(line)}) Tj")
            y -= 12
            if y < 40:
                break
        commands.append("ET")
        stream = "\n".join(commands).encode("cp1252")
    else:
        stream = b""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica {font_extra} >>".encode("latin-1"),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    while len(out) < min_size:
        out += b"% padding padding padding padding padding padding padding padding\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


SAMPLE_LINES = [
    "RegLens test document. This sentence exists only so the PDF contains readable text.",
    "Article 1 - Subject matter. The purpose of this regulation is to test a downloader.",
    "Article 2 - Scope. This regulation applies to every unit test that needs a PDF.",
    "Article 3 - Definitions. A downloader is a program that saves files from a server.",
]

GOOD_HTML = (
    "<!doctype html><html><head><title>Guidance on testing downloaders</title></head><body>"
    + "".join(
        f"<p>Paragraph {i}: this page contains enough readable text to look like a real "
        "guidance page rather than an error message or a bot-detection challenge.</p>"
        for i in range(30)
    )
    + "</body></html>"
)

WAF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<script src='https://example.invalid/awswaf/challenge.js'></script>"
    "<noscript>Please enable JavaScript to continue.</noscript></body></html>"
    + " " * 3500  # pad past the minimum-size check so the challenge detection is what fails
)


def make_multipage_pdf(
    pages: list[list[str]],
    outline: list[tuple[str, int, int]] | None = None,
    min_size: int = 6000,
) -> bytes:
    """Return a PDF with one page per list in ``pages``; each string is one line of text, top to bottom.

    A line that is ``""`` leaves a gap without any text. ``outline`` is a list of ``(title, level, page_index)``
    bookmarks (levels start at 1, page_index at 0). Text is written in the WinAnsi encoding, so curly quotes,
    dashes and the section sign work. No PDF library is needed.
    """
    n_pages = len(pages)
    first_page_obj = 3
    content_obj = lambda i: first_page_obj + n_pages + i  # noqa: E731 - tiny local helper
    font_obj = first_page_obj + 2 * n_pages  # the regular face; the bold face follows it
    objects: list[bytes] = [b"", b""]  # catalog and page tree are filled in at the end
    for index in range(n_pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj(index)} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R /F2 {font_obj + 1} 0 R >> >> >>".encode()
        )
    for lines in pages:
        commands = ["BT"]
        y = 770
        for line in lines:
            if line:
                size = getattr(line, "size", BODY_SIZE)
                face = "F2" if getattr(line, "bold", False) else "F1"
                commands.append(f"/{face} {size} Tf 1 0 0 1 40 {y} Tm ({_pdf_escape(line)}) Tj")
            y -= 12
        commands.append("ET")
        stream = "\n".join(commands).encode("cp1252")
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

    catalog_extra = ""
    if outline:
        base = len(objects) + 1  # object number of the outline root
        nodes = []  # (title, level, page_index, children indexes)
        for title, level, page_index in outline:
            nodes.append({"title": title, "level": level, "page": page_index, "children": [], "parent": None})
        stack: list[int] = []
        roots: list[int] = []
        for index, node in enumerate(nodes):
            while stack and nodes[stack[-1]]["level"] >= node["level"]:
                stack.pop()
            if stack:
                nodes[stack[-1]]["children"].append(index)
                node["parent"] = stack[-1]
            else:
                roots.append(index)
            stack.append(index)

        def number(i: int) -> int:
            return base + 1 + i

        def siblings(i: int) -> list[int]:
            parent = nodes[i]["parent"]
            return roots if parent is None else nodes[parent]["children"]

        objects.append(
            f"<< /Type /Outlines /First {number(roots[0])} 0 R /Last {number(roots[-1])} 0 R /Count {len(nodes)} >>".encode()
        )
        for i, node in enumerate(nodes):
            group = siblings(i)
            position = group.index(i)
            parent = base if node["parent"] is None else number(node["parent"])
            body = f"<< /Title ({_pdf_escape(node['title'])}) /Parent {parent} 0 R /Dest [{first_page_obj + node['page']} 0 R /Fit]"
            if position > 0:
                body += f" /Prev {number(group[position - 1])} 0 R"
            if position < len(group) - 1:
                body += f" /Next {number(group[position + 1])} 0 R"
            if node["children"]:
                body += f" /First {number(node['children'][0])} 0 R /Last {number(node['children'][-1])} 0 R /Count {len(node['children'])}"
            objects.append((body + " >>").encode("cp1252"))
        catalog_extra = f" /Outlines {base} 0 R"

    objects[0] = f"<< /Type /Catalog /Pages 2 0 R{catalog_extra} >>".encode()
    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(n_pages))
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number_, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number_} 0 obj\n".encode() + body + b"\nendobj\n"
    while len(out) < min_size:
        out += b"% padding padding padding padding padding padding padding padding\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def make_pdf_from_stream(commands: list[str], min_size: int = 6000) -> bytes:
    """A one-page PDF whose content stream is exactly ``commands`` (for tests of fonts, sizes and positions).

    Two fonts are available: ``/F1`` is Helvetica and ``/F2`` is Helvetica-Bold, both in WinAnsi encoding.
    """
    stream = "\n".join(commands).encode("cp1252")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    while len(out) < min_size:
        out += b"% padding padding padding padding padding padding padding padding\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def make_pdf_with_form(page_commands: list[str], form_commands: list[str], min_size: int = 6000) -> bytes:
    """A one-page PDF whose page content draws a form XObject (``/Fm1``), as headers and footers often are.

    ``/F1`` is Helvetica in both the page and the form. pypdf reports text drawn by a form twice to its visitor.
    """
    page_stream = "\n".join(page_commands).encode("cp1252")
    form_stream = "\n".join(form_commands).encode("cp1252")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Fm1 6 0 R >> >> >>",
        b"<< /Length " + str(len(page_stream)).encode() + b" >>\nstream\n" + page_stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Length "
        + str(len(form_stream)).encode() + b" >>\nstream\n" + form_stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    while len(out) < min_size:
        out += b"% padding padding padding padding padding padding padding padding\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
