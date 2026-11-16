"""Small helpers for the command-line tools."""

from __future__ import annotations

import contextlib
import sys


def make_console_safe() -> None:
    """Make printing never crash on a console that cannot show a character (older Windows code pages)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def short_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest:02d} s"
