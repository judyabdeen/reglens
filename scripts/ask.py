#!/usr/bin/env python3
"""Ask the indexed documents.  Usage:  python scripts/ask.py "question"   (no question: interactive chat)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.cli.ask import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
