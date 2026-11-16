#!/usr/bin/env python3
"""Embed the chunks and fill the vector database.  Usage:  python scripts/build_index.py [--backend chroma|numpy|both]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.cli.build_index import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
