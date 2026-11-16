from collections import Counter

import pytest

from reglens.ingest.textclean import (
    build_vocab,
    count_words,
    join_wrapped_lines,
    normalize_chars,
    repair_footnote_markers,
    repair_spacing,
    repair_split_words,
)

# ----------------------------------------------------------------------------------- normalize_chars


def test_private_use_bullets_become_real_bullets():
    assert normalize_chars(" first\n second") == "• first\n• second"


@pytest.mark.parametrize("glyph", ["\uf0b7", "\uf0a7", "\uf0d8", "\uf076", "\uf0fc", "\uf0a8", "\uf06e", "\uf075", "\uf06f", "\uf06c", "\uf071"])
def test_every_known_wingdings_or_symbol_bullet_is_a_bullet_and_is_not_reported_as_garbled(glyph):
    stats = Counter()
    assert normalize_chars(f"{glyph} a systematic description is provided", stats) == "\u2022 a systematic description is provided"
    assert not [key for key in stats if key.startswith("U+")]


def test_other_bullet_glyphs_become_the_same_bullet():
    assert normalize_chars("● one ▪ two") == "• one • two"


def test_ligatures_soft_hyphens_and_zero_width_characters_are_cleaned():
    assert normalize_chars("oﬃce") == "office"  # NFKC turns the ffi ligature into letters
    assert normalize_chars("organi­sation") == "organisation"
    assert normalize_chars("a​b﻿c") == "abc"


def test_unicode_spaces_and_tabs_become_plain_spaces():
    assert normalize_chars("a b c\td") == "a b c d"


def test_control_characters_are_removed_but_newlines_stay():
    assert normalize_chars("a\x00b\x07c\nd") == "abc\nd"


def test_unreadable_codes_are_removed_and_counted():
    stats: Counter = Counter()
    text = normalize_chars("one (cid:12) two  three � four", stats)
    assert text == "one  two  three  four"
    assert stats["cid_refs"] == 1
    assert stats["private_use_dropped"] == 1
    assert stats["replacement_chars"] == 1


def test_stats_are_optional_and_plain_text_is_untouched():
    plain = "Article 5(1)(a) of Regulation (EU) 2016/679 — “lawfulness”, § 7001."
    assert normalize_chars(plain) == plain


# ------------------------------------------------------------------------ repair_footnote_markers


@pytest.mark.parametrize(
    "raw, marker",
    [
        ("detriment.”17F\n18 This test", "[18]"),
        ("jobs. 0F\n1 As companies", "[1]"),
        ("services.28F29 And we", "[29]"),
    ],
)
def test_word_export_footnote_markers_become_brackets(raw, marker):
    fixed = repair_footnote_markers(raw)
    assert marker in fixed
    assert "F" not in fixed.split(marker)[0][-4:]  # the "17F" part is gone


@pytest.mark.parametrize("text", ["Part 3F7 of the form", "model 1F1", "the 12F 14 pieces", "Form 20F"])
def test_numbers_next_to_an_f_that_are_not_consecutive_are_left_alone(text):
    assert repair_footnote_markers(text) == text


# ----------------------------------------------------------------------------------- repair_spacing


@pytest.mark.parametrize(
    "raw, fixed",
    [
        ("a built -in feature", "a built-in feature"),
        ("the output , but later", "the output, but later"),
        ("It ends here .", "It ends here."),
        ("see Article 5 ( 1 ) and ( a )", "see Article 5 (1) and (a)"),
        ("wait . . . what", "wait ... what"),  # the ellipsis is closed up, the space before it is kept
        ("wait. . . what", "wait... what"),
        ("yes — no", "yes—no"),
        ("America’ s AI plan", "America’s AI plan"),
        ("too    many   spaces", "too many spaces"),
    ],
)
def test_repair_spacing_fixes_extraction_gaps(raw, fixed):
    assert repair_spacing(raw) == fixed


@pytest.mark.parametrize("text", ["pages 3 - 4 of the report", "A - B", "e.g. 10 % of cases", "2016/679 (GDPR)"])
def test_repair_spacing_does_not_touch_legitimate_spacing(text):
    assert repair_spacing(text) == text


def test_repair_spacing_is_idempotent():
    messy = "A built -in rule ( 1 ) , said America’ s lawyer . . . yes — no ."
    once = repair_spacing(messy)
    assert repair_spacing(once) == once


# --------------------------------------------------------------------------------------- vocabulary


def test_vocab_counts_lower_cased_words_and_keeps_compounds_together():
    vocab = build_vocab(["The state-of-the-art State", "state, STATE!"])
    assert vocab["state"] == 3
    assert vocab["state-of-the-art"] == 1
    assert vocab["the"] == 1


# ----------------------------------------------------------------------------- join_wrapped_lines


@pytest.mark.parametrize(
    "first, second, vocab_text, expected",
    [
        ("Retrieval-", "Augmented Generation", "", "Retrieval-Augmented Generation"),  # capital letter: a compound
        ("the organi-", "zation of data", "organization organization", "the organization of data"),
        ("is non-", "parametric memory", "non-parametric non-parametric", "is non-parametric memory"),
        ("a short-", "term plan", "", "a short-term plan"),  # known prefix keeps its hyphen
        ("we coop-", "erate with", "", "we cooperate with"),  # unknown word: line-end hyphenation
        ("COVID-", "19 cases", "", "COVID-19 cases"),
        ("the state-of-the-", "art model", "", "the state-of-the-art model"),
        ("a plain line", "and the next", "", "a plain line and the next"),
        ("a dash -", "then more", "", "a dash - then more"),
    ],
)
def test_join_wrapped_lines(first, second, vocab_text, expected):
    assert join_wrapped_lines(first, second, build_vocab([vocab_text])) == expected


def test_join_wrapped_lines_handles_empty_sides():
    vocab = Counter()
    assert join_wrapped_lines("", " next", vocab) == "next"
    assert join_wrapped_lines("last ", "", vocab) == "last"


def test_a_word_that_exists_only_hyphenated_keeps_its_hyphen():
    vocab = build_vocab(["a well-known fact", "the well-known case"])
    assert join_wrapped_lines("a well-", "known fact", vocab) == "a well-known fact"


# ------------------------------------------------------------------------------ repair_split_words


def _vocab(*texts: str) -> Counter:
    return build_vocab(texts)


def test_split_word_is_rejoined_when_the_whole_word_occurs_elsewhere():
    vocab = _vocab("State State State law", "S tate")
    assert repair_split_words("A S tate law", vocab) == "A State law"


def test_possessive_after_a_split_word_is_handled():
    vocab = _vocab("State State State")
    assert repair_split_words("the S tate’s law", vocab) == "the State’s law"


def test_trailing_letter_split_is_rejoined_and_punctuation_kept():
    vocab = _vocab("harbors harbors harbors")
    assert repair_split_words("safe harbor s—even then", vocab) == "safe harbors—even then"


def test_overlapping_pairs_are_all_examined():
    vocab = _vocab("students students students", "safe safe safe", "harbor harbor harbor", "harbors harbors")
    assert repair_split_words("safe harbor s and stude nts", vocab) == "safe harbors and students"


@pytest.mark.parametrize(
    "text",
    [
        "A part of the plan",  # "a" is a real word even though "apart" exists
        "Plan B is ready",
        "I t is odd",  # "I" is a real word
        "in to the room",  # two common words are not glued together
        "is sue is unclear",
    ],
)
def test_legitimate_word_pairs_are_not_glued_together(text):
    vocab = _vocab("apart apart apart", "plan plan plan", "it it it", "into into into", "issue issue issue", "in in in to to to is is is")
    assert repair_split_words(text, vocab) == text


def test_an_artefact_that_repeats_is_rejoined_when_both_halves_are_very_rare():
    vocab = _vocab(" ".join(["model"] * 50), "Mo del", "Mo del", "Mo del")  # the artefact itself is counted three times
    assert repair_split_words("Mo del extraction", vocab) == "Model extraction"


def test_two_halves_that_are_both_words_of_the_document_are_not_glued_together():
    vocab = _vocab(" ".join(["biometric"] * 100), " ".join(["metric"] * 40), " ".join(["bio"] * 40))
    assert repair_split_words("bio metric data", vocab) == "bio metric data"
    assert repair_split_words("Bio metric data", vocab) == "Bio metric data"  # a capital letter does not make a common word rare


def test_a_fragment_next_to_a_real_word_is_rejoined_when_the_joined_word_is_common():
    vocab = _vocab(" ".join(["their"] * 70), " ".join(["the"] * 1500), "the ir", "the ir")  # "ir" occurs twice, as an artefact
    assert repair_split_words("of the ir AI system and the ir data", vocab) == "of their AI system and their data"
    assert repair_split_words("Th at is why", _vocab(" ".join(["that"] * 45), " ".join(["is"] * 90), "Th at")) == "That is why"
    assert repair_split_words("Backdo or poisoning", _vocab(" ".join(["backdoor"] * 20), " ".join(["or"] * 500), "Backdo or")) == "Backdoor poisoning"


def test_the_fragment_may_come_first_and_the_real_word_may_be_a_single_letter():
    vocab = _vocab(" ".join(["data"] * 300), " ".join(["a"] * 400), "dat a", "dat a")
    assert repair_split_words("of personal dat a and more", vocab) == "of personal data and more"
    assert repair_split_words("Dat a protection", vocab) == "Data protection"


def test_a_lone_a_or_i_followed_by_a_fragment_is_joined_but_not_followed_by_a_real_word():
    vocab = _vocab(" ".join(["article"] * 349), " ".join(["are"] * 411), " ".join(["a"] * 1783), " ".join(["part"] * 60), " ".join(["apart"] * 30), "A rticle", "a re")
    assert repair_split_words("in A rticle 6(3) the rules a re clear", vocab) == "in Article 6(3) the rules are clear"
    assert repair_split_words("A part of the plan", vocab) == "A part of the plan"  # "part" is a word of its own


def test_two_lone_letters_are_never_joined():
    vocab = _vocab(" ".join(["an"] * 155), " ".join(["a"] * 398), "a n", "a n", " ".join(["ai"] * 700), " ".join(["it"] * 90), "I t")
    assert repair_split_words("if a n AI system", vocab) == "if a n AI system"
    assert repair_split_words("A I system and I t does", vocab) == "A I system and I t does"


def test_the_joined_word_must_be_common_to_call_a_half_an_artefact_next_to_a_real_word():
    for count, expected in ((8, "personal data"), (7, "personal dat a")):
        vocab = _vocab(" ".join(["data"] * count), " ".join(["a"] * 400))  # "dat" never occurs elsewhere
        assert repair_split_words("personal dat a", vocab) == expected, count


def test_a_half_is_rare_next_to_a_real_word_only_below_a_tenth_of_the_joined_word():
    for fragment_count, expected in ((9, "their"), (10, "the ir")):
        vocab = _vocab(" ".join(["their"] * 100), " ".join(["the"] * 1000), " ".join(["ir"] * fragment_count))
        assert repair_split_words("the ir", vocab) == expected, fragment_count


def test_a_capitalised_right_hand_piece_starts_a_word_of_its_own():
    vocab = _vocab(" ".join(["ieeexplore"] * 10), " ".join(["ieee"] * 200), " ".join(["lifecycle"] * 40), " ".join(["life"] * 200))
    assert repair_split_words("IEEE Xplore", vocab) == "IEEE Xplore"
    assert repair_split_words("the Life Cycle of", vocab) == "the Life Cycle of"
    vocab = _vocab(" ".join(["health"] * 20), " ".join(["heal"] * 40))
    assert repair_split_words("POLICY Heal TH AND CARE", vocab) == "POLICY Heal TH AND CARE"  # "TH" is capitals, "Heal" is not
    assert repair_split_words("HEAL TH", _vocab(" ".join(["HEALTH"] * 20), " ".join(["HEAL"] * 40))) == "HEALTH"  # all in capitals: one word


def test_a_lone_a_or_i_next_to_a_fragment_that_is_as_rare_as_itself_is_left_alone():
    vocab = _vocab(" ".join(["information"] * 30), "I")  # "I" occurs once: as rare as the fragment "nformation" (never seen)
    assert repair_split_words("I nformation", vocab) == "I nformation"


def test_the_tail_of_a_hyphenated_word_is_not_the_left_half_of_a_pair():
    vocab = _vocab(" ".join(["controller"] * 50))
    for hyphen in ("-", "\u2010", "\u2011"):
        text = f"a pre{hyphen}contr oller here"
        assert repair_split_words(text, vocab) == text, hyphen
    assert repair_split_words("a contr oller here", vocab) == "a controller here"


def test_a_word_with_a_hyphen_of_any_kind_is_not_cut_in_two():
    vocab = _vocab(" ".join(["there"] * 50), " ".join(["the"] * 1000), "re-use re\u2010use re\u2011use")
    for hyphen in ("-", "\u2010", "\u2011"):
        text = f"on the re{hyphen}use of data"
        assert repair_split_words(text, vocab) == text, hyphen
        assert build_vocab([f"re{hyphen}use"])[f"re{hyphen}use"] == 1  # one word, whichever hyphen


def test_a_half_is_rare_when_the_joined_word_is_more_than_ten_times_as_common():
    for joined, expected in ((30, "Mo del"), (31, "Model")):
        vocab = _vocab(" ".join(["model"] * joined), "Mo del", "Mo del", "Mo del")  # each half occurs 3 times
        assert repair_split_words("Mo del", vocab) == expected, joined


def test_a_rare_joined_word_is_not_enough_evidence():
    vocab = _vocab("tomorrow")  # appears once only
    assert repair_split_words("to morrow", vocab) == "to morrow"


def test_count_words():
    assert count_words("one  two\nthree") == 3
    assert count_words("") == 0


# ------------------------------------------------------------------------------------- raised digits


from reglens.ingest.textclean import superscript_text  # noqa: E402


@pytest.mark.parametrize(
    "before, digits, expected",
    [
        ("is greater than 10", "25", "^25"),  # the base 10: an exponent
        ("7.5 ⋅ 10", "22", "^22"),
        ("2.3x10", "25", "^25"),
        ("Article 10", "3", " [3]"),  # a legal reference followed by a footnote mark
        ("paragraph 10", "2", " [2]"),
        ("in 2010", "3", " [3]"),  # not the base 10
        ("version 1.10", "4", " [4]"),
        ("WP 218", "6", " [6]"),  # glued to a number it would change that number
        ("international standards", "9", " [9]"),
        ("", "1", " [1]"),
    ],
)
def test_superscript_text_tells_exponents_from_footnote_marks(before, digits, expected):
    assert superscript_text(before, digits) == expected


def test_unicode_superscripts_are_not_flattened_by_normalisation():
    assert normalize_chars("greater than 10²⁵ operations") == "greater than 10^25 operations"
    assert normalize_chars("see Regulation¹ and Article 10³") == "see Regulation [1] and Article 10 [3]"


def test_footnote_marker_repair_keeps_the_number_of_lines():
    text = "detriment.”17F\n18 This test, which consists\nof three elements."
    repaired = repair_footnote_markers(text)
    assert repaired.count("\n") == text.count("\n")
    assert repaired.splitlines()[0].rstrip().endswith("[18]")


def test_footnote_marker_repair_accepts_a_mark_that_is_already_in_brackets():
    assert repair_footnote_markers("text 3F [4] more").strip() == "text  [4]  more".strip()
    assert "3F" not in repair_footnote_markers("text 3F [4] more")
    assert repair_footnote_markers("version 3F5") == "version 3F5"  # not consecutive: untouched


def test_dropped_private_use_glyphs_are_counted_one_by_one():
    stats = Counter()
    text = normalize_chars("a\ue001b\ue001c\uf123 d\uf0b7 e", stats)  # U+F0B7 is a known bullet and is kept as one
    assert "\ue001" not in text and "\uf123" not in text and "•" in text
    assert stats["private_use_dropped"] == 3 and stats["U+E001"] == 2 and stats["U+F123"] == 1
    assert "U+F0B7" not in stats
