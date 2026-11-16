#!/usr/bin/env python3
"""Start the RegLens web interface.  Usage:  python scripts/serve.py   (then open http://localhost:8000)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.cli.serve import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
