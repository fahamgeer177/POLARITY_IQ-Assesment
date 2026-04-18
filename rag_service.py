"""Compatibility wrapper.

The production code lives in `src/polarity_iq/rag_service.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
	sys.path.insert(0, str(_SRC))

from polarity_iq.rag_service import *  # noqa: F401,F403


__all__ = [name for name in globals().keys() if not name.startswith("_")]
