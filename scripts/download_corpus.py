#!/usr/bin/env python3
"""Download the RegLens corpus.  Usage:  python scripts/download_corpus.py [--status] [--only ID ...]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.corpus.download import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
