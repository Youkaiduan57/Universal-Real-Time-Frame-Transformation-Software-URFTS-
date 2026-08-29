from __future__ import annotations

from pathlib import Path
import sys

import resource_paths


def test_source_resource_path_is_project_relative() -> None:
    assert resource_paths.resource_path("models", "model.onnx") == (
        Path(__file__).resolve().parents[1] / "models" / "model.onnx"
    )


def test_frozen_resource_path_uses_meipass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert resource_paths.resource_path("configs", "runtime_profile.json") == (
        tmp_path / "configs" / "runtime_profile.json"
    )


def test_user_data_dir_uses_local_app_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    path = resource_paths.user_data_dir("logs")

    assert path == tmp_path / "UniversalUpscaler" / "logs"
    assert path.is_dir()
