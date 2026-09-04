# -*- mode: python ; coding: utf-8 -*-
"""Reproducible PyInstaller one-folder build for the native Windows GUI."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


project_root = Path(SPECPATH).resolve()
src_root = project_root / "src"

datas = [
    (str(project_root / "models" / "SRVGGNetCompact_x2.onnx"), "models"),
    (str(project_root / "models" / "RIFE_v3.6.onnx"), "models"),
    (str(project_root / "models" / "RIFE_v3.6_PROVENANCE.md"), "models"),
    (str(project_root / "models" / "RIFE_v4.25_lite.onnx"), "models"),
    (str(project_root / "models" / "RIFE_v4.25_lite_LICENSE.txt"), "models"),
    (str(project_root / "models" / "RIFE_v4.25_lite_PROVENANCE.md"), "models"),
    (str(project_root / "models" / "IFRNet_S_Vimeo90K.onnx"), "models"),
    (str(project_root / "models" / "IFRNet_LICENSE.txt"), "models"),
    (str(project_root / "models" / "IFRNet_S_Vimeo90K_PROVENANCE.md"), "models"),
    (str(project_root / "configs" / "runtime_profile.json"), "configs"),
]
if (project_root / "assets").is_dir():
    datas.append((str(project_root / "assets"), "assets"))

hiddenimports = []
for package in ("dxcam", "mss"):
    hiddenimports += collect_submodules(package)

binaries = []
for package in ("onnxruntime", "cv2", "dxcam"):
    binaries += collect_dynamic_libs(package)

native_bridge = src_root / "_urfts_directml.pyd"
if native_bridge.is_file():
    binaries.append((str(native_bridge), "."))

for package in ("dxcam", "mss", "comtypes"):
    datas += collect_data_files(package)

for distribution in (
    "onnxruntime-directml",
    "opencv-contrib-python",
    "dxcam",
    "mss",
    "comtypes",
    "pywin32",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

a = Analysis(
    [str(src_root / "ui" / "app.py")],
    pathex=[str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests", "benchmarks", "torch", "onnx"],
    noarchive=False,
    optimize=0,
)
# Some build environments expose Poppler's version-suffixed ICU DLLs on PATH.
# PyInstaller may collect them for Qt6Core even though Qt needs Windows' system
# unversioned ICU shim, causing a procedure-not-found failure at GUI startup.
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.lower() not in {"icuuc.dll", "icudt78.dll"}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UniversalUpscaler",
    icon=str(project_root / "assets" / "UniversalUpscaler.png"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="UniversalUpscaler",
)
