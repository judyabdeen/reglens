#!/usr/bin/env python3
"""Evaluate RegLens.  Usage:  python scripts/evaluate.py   (see --help for the options)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.cli.evaluate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
