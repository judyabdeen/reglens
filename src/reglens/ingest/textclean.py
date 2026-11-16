"""Character- and word-level repairs for text that was extracted from PDFs.

PDF text extraction is lossy: the file stores glyphs at positions, not words and paragraphs. These
helpers repair the artefacts that are common in the corpus (found by reading real extracted pages):

* private-use bullet glyphs, ligatures, soft hyphens and other invisible characters,
* spaces that appear inside words ("S tate") or around punctuation ("built -in", "AI system -that"),
* words broken over two lines with a hyphen ("organi-" / "zation") - but not real compounds
  ("Retrieval-" / "Augmented"),
* footnote markers that a Word export leaves as "0F" + "1" next to each other,
* raised digits: a footnote mark glued to the word before it ("WP 2186") and an exponent whose raised digits were
  flattened ("10^25" read as "1025").

Every repair is conservative: when the evidence is unclear the text is left alone, because wrong
"fixes" in a legal corpus are worse than ugly spacing. Nothing here deletes sentences.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

# --------------------------------------------------------------------------- characters

# Word and other tools store bullets as private-use glyphs of the Symbol / Wingdings fonts (squares, circles, arrows, ticks).
_PRIVATE_USE_BULLETS = {"", "", "", "", "", "", "", "", "", "", ""}
_OTHER_BULLETS = str.maketrans({c: "•" for c in "●▪◦‣⁃■▫"})
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿‎‏‪‫‬‭‮"))
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNICODE_SPACES = re.compile(r"[   -   　\t]")
_CID = re.compile(r"\(cid:\d+\)")
_PRIVATE_USE = re.compile(r"[-]")


# ------------------------------------------------------------------------ raised digits

_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_EXPONENT_BASE = re.compile(r"(?<![\d.,/])10$")
# "Article 10" followed by a raised "3" is a footnote mark, not 10 to the power 3.
_REFERENCE_WORDS = frozenset(
    "article articles paragraph paragraphs point points section sections chapter chapters annex annexes no number "
    "recital recitals directive regulation decision title part page pages p pp para".split()
)


def superscript_text(before: str, digits: str) -> str:
    """Plain-text form of a run of raised digits that follows ``before`` on the same line.

    After the base 10 it is an exponent ("10" + "25" -> "10^25"). Anywhere else it is a footnote reference mark,
    written " [n]" so that it can never fuse with the number or word before it ("WP 218" + "6" must not read
    "WP 2186"). Reading the raised digits as plain digits would silently change numbers, which is the worst thing
    extraction can do to a legal text.
    """
    base = before.rstrip()
    if _EXPONENT_BASE.search(base):
        word = base[:-2].split()[-1].lower().strip("().,;:") if base[:-2].split() else ""
        if word not in _REFERENCE_WORDS:
            return f"^{digits}"
    return f" [{digits}]"


def _unicode_superscript(match: re.Match) -> str:
    window = match.string[max(0, match.start() - 40) : match.start()]
    return superscript_text(window, match.group(0).translate(_SUPERSCRIPT_DIGITS))


def normalize_chars(text: str, stats: Counter | None = None) -> str:
    """Unicode-normalise ``text`` and map or drop characters that carry no meaning in plain text.

    ``stats`` (optional) is incremented for events worth reporting in the audit:
    ``private_use_dropped`` (and one ``U+XXXX`` entry per glyph), ``replacement_chars`` and ``cid_refs``.
    """
    text = _SUPERSCRIPT_RUN.sub(_unicode_superscript, text)  # before NFKC, which would turn "10²⁵" into "1025"
    text = unicodedata.normalize("NFKC", text)
    for glyph in _PRIVATE_USE_BULLETS:
        text = text.replace(glyph, "•")
    text = text.translate(_OTHER_BULLETS).translate(_INVISIBLE)
    text = _CONTROL.sub("", text)
    text = _UNICODE_SPACES.sub(" ", text)
    if stats is not None:
        stats["cid_refs"] += len(_CID.findall(text))
        dropped = _PRIVATE_USE.findall(text)
        stats["private_use_dropped"] += len(dropped)
        for char in dropped:
            stats[f"U+{ord(char):04X}"] += 1  # which glyph it was, so the audit can say what was lost
        stats["replacement_chars"] += text.count("�")
    text = _CID.sub("", text)
    text = _PRIVATE_USE.sub("", text)
    return text.replace("�", "")


# ----------------------------------------------------------------------------- spacing

_SPACE_BEFORE_PUNCT = re.compile(r"(?<=[\w)\]”’\"'%]) +([,;:!?])")
_SPACE_BEFORE_PERIOD = re.compile(r"(?<=[A-Za-z0-9)\]”’]) +\.(?=\s|$)")
_SPACE_BEFORE_HYPHEN = re.compile(r"(?<=[A-Za-z0-9]) +-(?=[A-Za-z0-9])")
_SPACES_AROUND_EM_DASH = re.compile(r"\s*—\s*")
_APOSTROPHE_GAP = re.compile(r"(?<=[A-Za-z])([’'])\s+(?=s\b)")
_PAREN_GAP = re.compile(r"\(\s+((?:\d{1,3}|[a-z]{1,3}))\s*\)|\(\s*((?:\d{1,3}|[a-z]{1,3}))\s+\)")
_ELLIPSIS = re.compile(r"\.(?: \.){2,}")
_MANY_SPACES = re.compile(r" {2,}")
# A Word export leaves footnote reference n as "(n-1)F" followed by "n" (possibly on the next line, possibly already
# turned into "[n]"): "0F" "1", "17F" "18". Only consecutive pairs count, so ordinary text such as "3F4" is never touched.
_FOOTNOTE_MARKER = re.compile(r"(?<!\d)(\d{1,3})F(\s?)\[?(\d{1,3})\]?(?!\d)")


def _footnote_marker(match: re.Match) -> str:
    if int(match.group(3)) == int(match.group(1)) + 1:
        # A line break inside the pair is kept, so the number of lines of a page never changes.
        return f" [{match.group(3)}] " + ("\n" if "\n" in match.group(2) else "")
    return match.group(0)


def repair_footnote_markers(text: str) -> str:
    """Turn the "0F 1" artefact into a plain "[1]" marker. Run it on a whole page, before splitting into lines."""
    return _FOOTNOTE_MARKER.sub(_footnote_marker, text)


def repair_spacing(text: str) -> str:
    """Fix spaces that the PDF text extraction put in the wrong places. Safe to run twice."""
    text = _ELLIPSIS.sub("...", text)
    text = _SPACES_AROUND_EM_DASH.sub("—", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_BEFORE_PERIOD.sub(".", text)
    text = _SPACE_BEFORE_HYPHEN.sub("-", text)
    text = _APOSTROPHE_GAP.sub(r"\1", text)
    text = _PAREN_GAP.sub(lambda m: f"({m.group(1) or m.group(2)})", text)
    return _MANY_SPACES.sub(" ", text).strip()


# -------------------------------------------------------------------------- vocabulary

_WORD = re.compile(r"[^\W\d_]+(?:[’'\u2010\u2011-][^\W\d_]+)*")  # "re-use" and "re\u2010use" are one word


def build_vocab(texts: Iterable[str]) -> Counter:
    """Lower-cased word counts of a document (inner hyphens and apostrophes stay inside a word)."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(word.lower() for word in _WORD.findall(text))
    return counts


# Prefixes that normally keep their hyphen when a compound is broken over two lines.
_HYPHEN_PREFIXES = frozenset(
    {
        "non", "anti", "self", "cross", "multi", "semi", "quasi", "pre", "post", "co", "re", "sub", "inter",
        "intra", "ex", "vice", "well", "long", "short", "high", "low", "full", "real", "open", "single", "pro",
        "all", "half", "third", "counter", "over", "under", "off", "on", "in", "out", "end", "data", "risk",
    }
)  # fmt: skip
_TAIL_WORD = re.compile(r"([^\W\d_]+(?:[’'-][^\W\d_]+)*)-$")
_HEAD_WORD = re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)*")


def join_wrapped_lines(first: str, second: str, vocab: Counter) -> str:
    """Join two consecutive lines of one paragraph, deciding what to do with a hyphen at the line end.

    * "Retrieval-" + "Augmented"  -> "Retrieval-Augmented" (next word capitalised: a real compound)
    * "organi-" + "zation"        -> "organization"        (the joined word exists elsewhere in the document)
    * "non-" + "parametric"       -> "non-parametric"      (the hyphenated form exists, or a known prefix)
    * anything else               -> a normal space
    """
    first = first.rstrip()
    second = second.lstrip()
    if not first:
        return second
    if not second:
        return first
    if not first.endswith("-") or first.endswith(" -") or len(first) < 2:
        return f"{first} {second}"

    tail = _TAIL_WORD.search(first)
    head = _HEAD_WORD.match(second)
    if tail is None or head is None:
        # "COVID-" + "19", "5-" + "year": keep the hyphen, no space.
        return first + second if second[:1].isalnum() and first[-2:-1].isalnum() else f"{first} {second}"
    left, right = tail.group(1), head.group(0)
    if not second[0].islower():
        return first + second  # "Retrieval-" + "Augmented"
    if "-" in left:
        return first + second  # already a compound: "state-of-the-" + "art"
    joined, hyphenated = (left + right).lower(), f"{left}-{right}".lower()
    if vocab.get(joined, 0) > 0 and vocab.get(hyphenated, 0) == 0:
        return first[:-1] + second
    if vocab.get(hyphenated, 0) > 0 or left.lower() in _HYPHEN_PREFIXES:
        return first + second
    return first[:-1] + second  # line-end hyphenation of a word that is not seen elsewhere


# ----------------------------------------------------------------------- split words

# Every adjacent pair of words, including overlapping ones: the right-hand word is only looked ahead at.
_PAIR = re.compile(r"(?<![\w\u2019'\u2010\u2011-])([^\W\d_]+) (?=([^\W\d_]+)(?:[\u2019']s)?(?![\w\u2019'\u2010\u2011-]))")
_LEGIT_SINGLE = frozenset("aAI")


RARE_FACTOR = 10  # a half is "rare" when the joined word occurs more than this many times as often as the half
FREQUENT_JOINED = 8  # next to a word that is common, a half is only an artefact when the joined word occurs this often


def _rare(part: str, joined: int, vocab: Counter) -> bool:
    """Is ``part`` so much rarer than the word it would form that it is an artefact, not a word of its own?"""
    return vocab.get(part.lower(), 0) * RARE_FACTOR < joined


def _continues_a_word(left: str, right: str) -> bool:
    """Can ``right`` be the rest of a word that began in ``left``? A capitalised right-hand piece starts a word of its
    own ("IEEE Xplore", "Life Cycle"), unless both pieces are in capitals ("HEAL TH")."""
    return not right[:1].isupper() or (right.isupper() and left.isupper())


def _should_join(left: str, right: str, vocab: Counter) -> bool:
    joined = vocab.get((left + right).lower(), 0)
    if joined < 2:
        return False
    one_letter = [part for part in (left, right) if len(part) == 1]
    if one_letter and not any(part in _LEGIT_SINGLE for part in one_letter):
        return True  # "S tate", "harbor s": a lone letter that cannot be a word
    if len(one_letter) == 2:
        return False  # "a n", "A I": a lone letter may belong to the word on its other side, too ambiguous
    if not one_letter and vocab.get(left.lower(), 0) < 2 and vocab.get(right.lower(), 0) < 2:
        return True  # neither half is a word of the document
    rare_left, rare_right = _rare(left, joined, vocab), _rare(right, joined, vocab)
    if rare_left and rare_right and not one_letter:
        return True  # an artefact that repeats ("Mo del" twenty times)
    # One half is a word of its own ("the", "a", "or") and the other much rarer than the word they form: "the ir",
    # "dat a", "A rticle". Only when the joined word is common, so that two real words are not glued together.
    return joined >= FREQUENT_JOINED and rare_left != rare_right and _continues_a_word(left, right)


def repair_split_words(text: str, vocab: Counter) -> str:
    """Re-join a word that the extraction cut in two ("S tate", "harbor s", "stude nts", "the ir", "dat a").

    A pair is joined only if the joined word occurs at least twice elsewhere in the document and the split is not a
    legitimate pair of words. That is so when
      * one half is a single letter that cannot be a word ("S tate"), or
      * neither half is itself a word of the document (seen fewer than twice), or both are very much rarer than the
        word they form (an artefact that repeats is counted too), or
      * the joined word is common (eight times or more) and exactly one half is far rarer than it: the other half
        is then a real word that the artefact was cut from ("the ir", "A rticle", "dat a").
    Two lone letters ("a n") are never joined, and nor is a capitalised right-hand piece ("IEEE Xplore"): both are
    more likely two words than a break. Next to a single "a" or "I" only the rare half can make the case.
    """
    pieces: list[str] = []
    position = 0
    for match in _PAIR.finditer(text):
        if match.start() < position:
            continue  # overlaps a pair that was just joined
        left, right = match.group(1), match.group(2)
        if _should_join(left, right, vocab):
            pieces.append(text[position : match.start()])
            pieces.append(left + right)
            position = match.end() + len(right)
    pieces.append(text[position:])
    return "".join(pieces)


def count_words(text: str) -> int:
    return len(text.split())
