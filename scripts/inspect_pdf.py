#!/usr/bin/env python3
"""Show how the ingestion sees one PDF (font sizes, bold, footnotes).  Usage:
    python scripts/inspect_pdf.py ID [--pages 5-6] [--find TEXT] [--blocks]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.ingest.inspector import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
