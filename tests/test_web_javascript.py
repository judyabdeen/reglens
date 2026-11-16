"""The scripts of the web page, checked with Node.js when it is installed.

Node.js is not needed to run RegLens (the browser runs the page); it only lets the logic of the page be tested without a
browser. Without Node these tests are skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "reglens" / "web" / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed (it is only used to test the page's scripts)")


def node(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([NODE, *args], capture_output=True, text=True, timeout=120, cwd=ROOT, check=False)


def test_the_logic_of_the_page_passes_its_own_tests():
    """tests/js/lib.test.mjs: citations, figures, highlighting, filters, saved settings, the exported Markdown."""
    result = node("--test", "tests/js/lib.test.mjs")
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


@pytest.mark.parametrize("path", sorted([*STATIC.glob("*.mjs"), *STATIC.glob("*.js")]), ids=lambda path: path.name)
def test_every_script_of_the_page_is_valid_javascript(path):
    result = node("--check", str(path))
    assert result.returncode == 0, result.stderr
