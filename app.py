"""Compatibility wrapper.

The production code lives in the `polarity_iq` package under `src/`.
This file stays to preserve the original `python app.py ...` workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python app.py ...` without requiring `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from polarity_iq.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
