#!/usr/bin/env python3
"""Check that this machine is ready to run RegLens.  Usage:  python scripts/check_setup.py [--offline]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.diagnostics import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
