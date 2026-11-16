"""The PDF helpers shared by the downloader and the ingestion code."""

from __future__ import annotations

import logging

import pytest

from reglens.pdfutil import PDF_FIXES, describe_pdf_error, quiet_pypdf


def test_pypdf_messages_are_silenced_only_inside_the_quiet_zone(caplog):
    noisy = logging.getLogger("pypdf._cmap")  # a child logger, like the ones pypdf reports through
    with caplog.at_level(logging.DEBUG):
        with quiet_pypdf():
            noisy.warning("a warning from pypdf")
            noisy.error("an error from pypdf")  # pypdf reports unsupported font encodings at this level
        assert "from pypdf" not in caplog.text
        noisy.error("error after the quiet zone")
        assert "error after the quiet zone" in caplog.text


def test_quiet_zone_restores_the_previous_level_even_when_the_body_raises():
    logger = logging.getLogger("pypdf")
    before = logger.level
    with pytest.raises(RuntimeError):
        with quiet_pypdf():
            assert logger.level == logging.CRITICAL
            raise RuntimeError("boom")
    assert logger.level == before


@pytest.mark.parametrize("name", sorted(PDF_FIXES))
def test_known_failures_get_their_fix(name):
    error = type(name, (Exception,), {})("simulated")  # matched by class name, so a stand-in class is enough
    text = describe_pdf_error(error)
    assert name in text and PDF_FIXES[name] in text


def test_unknown_failures_get_no_invented_fix():
    assert describe_pdf_error(RuntimeError("x")) == "PDF could not be parsed (RuntimeError)"
    assert describe_pdf_error(ValueError("x"), "Cannot read the file") == "Cannot read the file (ValueError)"
