#!/usr/bin/env python3
"""Clean and chunk the downloaded corpus.  Usage:  python scripts/ingest.py [--only ID ...] [--export]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.ingest.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
