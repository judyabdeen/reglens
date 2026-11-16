"""Small helpers shared by everything that reads PDFs with pypdf (the downloader and the ingestion code)."""

from __future__ import annotations

import contextlib
import logging

# pypdf failures that have a known one-line fix. They are matched by exception name, so no pypdf import is needed.
PDF_FIXES = {
    "DependencyError": 'it is encrypted and needs the "cryptography" package; run: pip install cryptography',
    "DeprecationError": 'pypdf 7.0 has a bug with some PDFs; run: pip install "pypdf<7"',
}


@contextlib.contextmanager
def quiet_pypdf():
    """pypdf prints a very long message for every font encoding it only partly supports; here that is just noise.

    pypdf logs those at ERROR level, so the level has to go up to CRITICAL (ERROR would let them straight through).
    """
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(previous)


def describe_pdf_error(exc: BaseException, what: str = "PDF could not be parsed") -> str:
    """One line naming the failure, with the known fix appended when there is one.

    >>> describe_pdf_error(RuntimeError("boom"))
    'PDF could not be parsed (RuntimeError)'
    """
    name = type(exc).__name__
    fix = PDF_FIXES.get(name)
    return f"{what} ({name}): {fix}" if fix else f"{what} ({name})"
