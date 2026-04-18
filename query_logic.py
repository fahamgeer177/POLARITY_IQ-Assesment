"""Compatibility wrapper.

The production code lives in `src/polarity_iq/query_logic.py`.
This module is kept to preserve existing imports like `from query_logic import filter_retrieved`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
	sys.path.insert(0, str(_SRC))

from polarity_iq.query_logic import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
