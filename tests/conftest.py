from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Allow running tests from a source checkout without `pip install -e .`.
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))