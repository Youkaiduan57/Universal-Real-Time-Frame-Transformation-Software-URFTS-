"""Runtime paths shared by source and frozen application builds."""

from __future__ import annotations

import os
from pathlib import Path
import sys


APPLICATION_NAME = "UniversalUpscaler"


def is_frozen() -> bool:
    """Return whether the process is running from a PyInstaller bundle."""

    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def resource_root() -> Path:
    """Return the read-only root containing bundled project resources."""

    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Resolve a bundled resource without depending on the working directory."""

    return resource_root().joinpath(*parts)


def user_data_dir(*parts: str, create: bool = True) -> Path:
    """Return a writable per-user application directory."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    target = base.joinpath(APPLICATION_NAME, *parts)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target
