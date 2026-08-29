"""Tests for upscaler validation."""

from __future__ import annotations

import pytest

from upscaler import Upscaler


def test_upscaler_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError):
        Upscaler(method="unsupported-method")