"""Synthetic documents that imitate the layout problems of the real corpus (made up, not copied from any law).

``regulation_pages`` imitates an EU Official Journal act: a running header that alternates between odd and even
pages, recitals and list items whose number sits on its own line, "Article N" with the title on the next line,
a list of definitions "(1) ..." that must not be mistaken for footnotes, hyphenated line ends, and footnotes at the
foot of the page - one of which continues on the next page.

``statement_pages`` imitates a Word-exported agency statement: a page number at the top of every page, Roman-numeral
headings, footnote markers glued to the text ("0F" "1"), spaces inside words, and a footnote that continues on the
next page.
"""

from __future__ import annotations

import textwrap

from helpers import bold, small

SENTENCES = [
    "The controller shall implement appropriate technical and organisational measures to ensure and to be able to demonstrate that processing is performed in accordance with this Regulation.",
    "Those measures shall be reviewed and updated where necessary, taking into account the nature, scope, context and purposes of the processing as well as the risks of varying likelihood and severity.",
    "Member States shall ensure that the supervisory authority is provided with the human, technical and financial resources necessary for the effective performance of its tasks.",
    "A data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed.",
    "The provider of a high-risk system shall establish, implement, document and maintain a risk management system throughout the entire lifecycle of that system.",
    "Where the processing is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall carry out an assessment of the impact of the envisaged processing operations.",
]


def _paragraph(seed: int, sentences: int) -> str:
    return " ".join(SENTENCES[(seed + i) % len(SENTENCES)] for i in range(sentences))


def _wrap(text: str, width: int = 92) -> list[str]:
    return textwrap.wrap(text, width=width)


def _numbered(marker: str, text: str) -> list[str]:
    """An item as pypdf returns it from an Official Journal table: the marker alone on a line, then the text."""
    return [marker, *_wrap(text)]


def _article(label: str, title: str, paragraphs: list[str]) -> list[str]:
    lines = [label, title]
    for number, text in enumerate(paragraphs, start=1):
        lines += _wrap(f"{number}.   {text}")
    return lines


# Line-end hyphenation as the publisher's layout engine produces it: the break falls inside a word.
HYPHEN_LINES = [
    "The supervisory authority shall coop-",
    "erate with the other authorities. An organi-",
    "sation shall record every breach. The non-",
    "binding opinion is public, and a non-binding guideline may be added.",
]


def _page(number: int, body: list[str], notes: list[str]) -> list[str]:
    odd = f"L 119/{number} EN Official Journal of the European Union 4.5.2099"
    even = f"4.5.2099 EN Official Journal of the European Union L 119/{number}"
    header = odd if number % 2 else even
    return [header, *body, "", *(small(note) for note in notes)] if notes else [header, *body]


def regulation_pages() -> list[list[str]]:
    """Ten pages of a made-up regulation (one list of lines per page), 12 footnotes numbered continuously."""
    sections: list[tuple[list[str], list[str]]] = []  # (body lines, footnote lines) per page

    sections.append((
        [
            bold("REGULATION (EU) 2099/1 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL", 12),
            "of 27 April 2099",
            "on the protection of natural persons with regard to the processing of test data",
            "THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,",
            "Having regard to the Treaty on the Functioning of the European Union, and in particular Article 16 thereof,",
            "Having regard to the opinion of the European Economic and Social Committee ( 1 ),",
            "Acting in accordance with the ordinary legislative procedure ( 2 ),",
            "Whereas:",
            *_numbered("(1)", "The protection of natural persons in relation to the processing of personal data is a fundamental right."),
            *_numbered("(2)", "The principles of, and rules on the protection of natural persons with regard to the processing of their personal data should respect their fundamental rights and freedoms."),
        ],
        ["(1) OJ C 229, 31.7.2099, p. 90.", "(2) Position of the European Parliament of 12 March 2099."],
    ))
    sections.append((
        [
            *_numbered("(3)", "The economic and social integration resulting from the functioning of the internal market has led to a substantial increase in cross-border flows of personal data."),
            *_numbered("(4)", "Technology has transformed both the economy and social life, and should further facilitate the free flow of personal data within the Union."),
            "HAVE ADOPTED THIS REGULATION:",
            "CHAPTER I",
            "General provisions",
            *_article("Article 1", "Subject-matter and objectives", [
                "This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data.",
                "This Regulation protects fundamental rights and freedoms of natural persons and in particular their right to the protection of personal data.",
            ]),
        ],
        ["(3) Directive 2098/95/EC of the European Parliament and of the Council of 24 October 2098 (OJ L 281, 23.11.2098, p. 31).", "(4) Council Decision 2098/17/JHA of 8 June 2098."],
    ))
    sections.append((
        [
            *_article("Article 2", "Material scope", [_paragraph(2, 2), _paragraph(3, 1)]),
            "Article 3",
            "Definitions",
            "For the purposes of this Regulation:",
            *_wrap("(1) 'personal data' means any information relating to an identified or identifiable natural person ('data subject');"),
            *_wrap("(2) 'processing' means any operation or set of operations which is performed on personal data, whether or not by automated means;"),
        ],
        ["(5) Regulation (EU) 2098/45 of the European Parliament and of the Council of 18 December 2098.", "(6) Commission Recommendation 2098/361/EC of 6 May 2098."],
    ))
    sections.append((
        [
            "Article 4",
            "Principles relating to processing of personal data",
            *_wrap(f"1.   {_paragraph(0, 4)}"),
            "2.   Personal data shall be:",
            *_numbered("(a)", "processed lawfully, fairly and in a transparent manner in relation to the data subject;"),
            *_numbered("(b)", "collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes;"),
            *_wrap(f"3.   {_paragraph(2, 4)}"),
        ],
        ["(7) Article 5 of the Charter is quoted here only to test a note on its own."],
    ))
    sections.append((
        [
            *_wrap(f"4.   {_paragraph(4, 4)}"),
            *HYPHEN_LINES,
            *_wrap(f"5.   {_paragraph(1, 4)}"),
            *_wrap(f"6.   {_paragraph(3, 4)}"),
        ],
        ["(8) See the Opinion of the Board of 2 February 2099.", "(9) Judgment of the Court of 21 December 2098, Test v Test, C-1/99, EU:C:2098:1."],
    ))
    sections.append((
        [
            *_article("Article 5", "Lawfulness of processing", ["Processing shall be lawful only if and to the extent that at least one of the following applies:"]),
            *_numbered("(a)", "the data subject has given consent to the processing of his or her personal data for one or more specific purposes;"),
            *_numbered("(b)", "processing is necessary for the performance of a contract to which the data subject is party;"),
            *_numbered("(c)", "processing is necessary for compliance with a legal obligation to which the controller is subject."),
            *_wrap("2.   10 percent of the cases in the annual report are enough to trigger a review, which the authority shall carry out within three months."),
        ],
        [],
    ))
    sections.append((
        [
            *_article("Article 6", "Conditions for consent", [_paragraph(5, 3), _paragraph(0, 2)]),
        ],
        ["(10) Regulation (EU) 2098/77 of the European Parliament and of the Council of 3 March 2098 on the"],
    ))
    sections.append((
        [
            "CHAPTER II",
            "Rights of the data subject",
            *_article("Article 7", "Transparent information", [_paragraph(3, 3)]),
        ],
        ["protection of natural persons, as amended.", "(11) Test note on the right to information."],
    ))
    sections.append((
        [
            *_article("Article 8", "Entry into force", ["This Regulation shall enter into force on the twentieth day following that of its publication in the Official Journal of the European Union."]),
            "This Regulation shall be binding in its entirety and directly applicable in all Member States.",
            "Done at Brussels, 27 April 2099.",
            "For the European Parliament",
            "The President",
        ],
        ["(12) Test note on the entry into force."],
    ))
    sections.append((
        [
            "ANNEX I",
            "List of examples",
            *_wrap(_paragraph(1, 3)),
        ],
        [],
    ))
    return [_page(number, body, notes) for number, (body, notes) in enumerate(sections, start=1)]


_ALPHABET = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()


def _body_lines(page: int, count: int = 10) -> list[str]:
    """Lines that differ on every page and every row, even after digits are ignored."""
    return [f"{_ALPHABET[(page + row) % 26]} {_ALPHABET[(page * 3 + row * 5) % 26]} {_ALPHABET[(page * 7 + row) % 26]} text" for row in range(count)]


def official_journal_pages() -> list[list[str]]:
    """Pages of a made-up act whose PDF stores the running header in the wrong place, as the real ones do:
    as its own line in the middle of the page, glued to the end of a sentence, and on both sides of a footnote."""
    header = lambda n: f"4.5.2099 L 119/{n} Official Journal of the European Union EN"  # noqa: E731
    pages = []
    for n in range(1, 13):
        body = _body_lines(n, 9)
        notes = [small(f"(1) OJ C {n}, 31.7.2099, p. {n} ({_ALPHABET[(n * 3) % 26]} {_ALPHABET[(n * 5) % 26]}).")]
        if n % 4 == 1:  # as the first line, where margin detection finds it
            page = [header(n), *body, "", *notes]
        elif n % 4 == 2:  # as its own line between the text and the footnotes
            page = [*body, header(n), *notes]
        elif n % 4 == 3:  # glued to the end of the last line of text
            page = [*body[:-1], f"{body[-1]} {header(n)}", *notes]
        else:  # glued in the middle of the text
            page = [*body[:4], f"{body[4]} {header(n)} and more words", *body[5:], *notes]
        pages.append(page)
    return pages


def treaty_pages() -> list[list[str]]:
    """A treaty whose articles are numbered paragraphs "1 ... 2 ..." in the same type as the rest of the text.

    Nothing here is a footnote, although the numbered paragraphs sit at the foot of the pages and count 1, 2 again
    and again - which is exactly what the text of a footnote block looks like.
    """
    pages = []
    for n in range(1, 8):
        pages.append([
            f"Article {2 * n - 1} – Scope {n}",
            *_wrap(f"1 {_paragraph(n, 2)}"),
            *_wrap(f"2 {_paragraph(n + 1, 2)}"),
            f"Article {2 * n} – Duties {n}",
            *_wrap(f"1 {_paragraph(n + 2, 1)}"),
            *_wrap(f"2 {_paragraph(n + 3, 1)}"),
            f"Remark {_ALPHABET[n]} applies in full.",  # a last line that no other page repeats (so it is no footer)
        ])
    return pages


def statement_pages() -> list[list[str]]:
    """Four pages of a made-up Word-exported statement, with footnote artefacts and continued footnotes."""
    paragraph = (
        "Artificial intelligence is reshaping how consumers get information and perform their jobs. 0F",
        "1 As companies have marketed their products to the public, they have represented that their systems aim to give",
        "accurate output that is possible within their technological constraints. The Commission believes that consumers",
        "have a reasonable expectation of truthful output , but a company m ight alter that output. A State law may also",
        "matter. Of course , the Commission will apply the law to every State and to each company in the same way.",
        "This is the last line.",
    )
    pages = [
        [
            "1",
            bold("PROPOSED POLICY STATEMENT CONCERNING THE ACCURACY OF AI SYSTEMS", 14),
            "July 1, 2099",
            " ",
            bold("I. Summary"),
            *paragraph,
            " ",
            small("1 Executive Office of the President, Winning the Race (July 2099), https://example.org/a-long-address-that-continues-"),
            small("on-the-next-line.pdf."),
        ],
        [
            "2",
            bold("II. Background"),
            *paragraph,
            "Another paragraph follows and it is long enough to be a real line of text that is wrapped here to",
            "keep going until the page is full, with the State and the company named again for the vocabulary.",
            " ",
            small("2 See, e.g., Exec. Order No. 1, 90 Fed. Reg. 1 (July 23, 2099)."),
            small("3 Id."),
        ],
        [
            "3",
            bold("III. The Commission's Authority under Section 5"),
            *paragraph,
            " ",
            small("4 Id. at 5."),
            small("5 See the statement of the Commission, which is quoted here at some length so that the"),
            small("footnote does not end on this page and the rest follows below -"),
        ],
        [
            "4",
            "Another paragraph follows and it is long enough to be a real line of text that is wrapped here to",
            "keep going until the page is full, with the State and the company named again for the vocabulary.",
            " ",
            small("continued text of footnote 5 that started on the page before."),
            small("6 Id. at 6."),
        ],
    ]
    return pages


# A saved web page with a cookie banner, menus, a breadcrumb, a share box, hidden text and a footer around the content.
_WEB_TEXT = (
    "This guidance explains how organisations should think about the rights of individuals when they build and "
    "use systems that process personal data, and it is written so that the main element carries enough text to "
    "be recognised as the real content of the page rather than a small wrapper element. "
) * 2

GUIDANCE_PAGE = f"""<!doctype html><html><head><title>Guidance on AI and data protection | ICO</title><meta charset="utf-8"></head>
<body>
<div id="cookie-banner">We use cookies. Accept all cookies.</div>
<header><a href="/">ICO logo</a><nav><ul><li>Home</li><li>For organisations</li></ul></nav></header>
<nav class="breadcrumb"><a href="/">Home</a> &gt; <a href="/x">Guidance</a></nav>
<main>
  <h1>How do we ensure individual rights in our AI systems?</h1>
  <p>Skip to main content</p>
  <p>{_WEB_TEXT}</p>
  <h3>Right of access</h3>
  <p>Individuals have the right to <strong>access</strong> their personal data.<br>This includes training data in some cases.</p>
  <ul><li>Search for the data</li><li>Provide a copy</li></ul>
  <h4>Practical steps</h4>
  <div><p>First paragraph inside a div.</p><p>Second paragraph.</p></div>
  <table><tr><th>Right</th><th>Applies when</th></tr><tr><td>Access</td><td>Always</td></tr></table>
  <div class="share-this-page">Share on Twitter Facebook</div>
  <p hidden>Hidden text must not appear.</p>
  <span aria-hidden="true">Decorative text</span>
  <script>var tracking = 1;</script><style>.x {{ color: red }}</style>
  <h3>Right of access</h3>
  <h3>Right of access</h3>
  <p>Text after a repeated heading that is long enough to count as real content for the test page here.</p>
</main>
<footer>Copyright ICO. Contact us.</footer>
</body></html>"""
