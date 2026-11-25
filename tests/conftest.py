"""Shared pytest fixtures. Module-specific helpers live in tests/helpers.py and tests/session_fixtures.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import LogCapture


@pytest.fixture
def real_tmp(tmp_path: Path) -> Path:
    """A fresh temp directory with symlinks resolved (macOS /var -> /private/var), like `realpath(mkdtemp())`."""
    return tmp_path.resolve()


@pytest.fixture
def log_capture() -> LogCapture:
    """A debug-level logger whose JSON lines are kept in memory (`.lines`, `.records`)."""
    return LogCapture()
