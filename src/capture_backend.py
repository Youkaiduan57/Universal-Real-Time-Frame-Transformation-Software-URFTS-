"""Capture backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CaptureBackend(ABC):
    @abstractmethod
    def grab_frame(self) -> Any:
        """Capture and return one frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
        raise NotImplementedError