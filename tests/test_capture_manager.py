"""Tests for capture backend selection validation."""

from __future__ import annotations

import pytest

from capture_manager import CaptureManager


def test_capture_manager_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError):
        CaptureManager(backend="unsupported-backend")