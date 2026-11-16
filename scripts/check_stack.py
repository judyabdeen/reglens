#!/usr/bin/env python3
"""Check ChromaDB, the embedding model, the re-ranker and the language model.  Usage:  python scripts/check_stack.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.stackcheck import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
