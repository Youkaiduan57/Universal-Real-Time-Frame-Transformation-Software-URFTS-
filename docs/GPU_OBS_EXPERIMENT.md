# GPU and OBS experiment

The working CPU-frame Bicubic baseline is commit `60be1fe`. Its settings remain
unchanged. Native GPU mode is an explicit experiment, not a measured improvement
over that baseline. Do not switch every setting at once.

## GUI trials

Restart the GUI to load the new code. Keep the existing CPU profile for comparison.

1. Select **D3D11 GPU pipeline**, **WGC**, **Bicubic**, frame generation **Off**.
   Check capture, client crop, scaling, resize, and shutdown first.
2. To test GPU frame generation, select **D3D11 + GPU frame gen (experimental)**.
   Select IFRNet-S or RIFE 4.25-lite, Performance, **1 generated per real**,
   temporal and HUD stabilization **On**, presentation buffer **0 ms**.
   Target FPS **60**, fixed pacing. This is not a guarantee of 60 delivered FPS.
3. Test the same pipeline with **OBS Spout (experimental)** only after configuring
   OBS as below. OBS controls capture selection; URFTS's target-window field
   identifies the game for focus/Alt-Tab behavior only.

Native capture, inference, and presentation share **one GPU adapter**. Its name is
logged. The CPU mode's numeric DirectML device selection is not used in this mode.
The GPU path does not currently implement CPU presentation delay, the preview FPS
overlay, or the CPU stationary-inference shortcut. FPS is logged to session output.
The native preview is click-through and does not activate itself. With a selected
game it hides on Alt-Tab and reappears when the game is foreground. Ctrl+Alt+Q or
GUI Stop stops the native loop; do not try to focus the preview and press Q.

## OBS setup

Install the separately distributed OBS plugin from its official releases:
https://github.com/Off-World-Live/obs-spout2-plugin/releases

For OBS 32.1.2, the project now includes `Install OBS Spout.ps1`, pinning Spout
1.12.0 and verifying the official archive checksum. Close OBS before running it:

```powershell
& powershell -ExecutionPolicy Bypass -File '.\Install OBS Spout.ps1'
```

This installs the plugin and its runtime DLLs under
`C:\ProgramData\obs-studio\plugins\win-spout`. It does not create/replace OBS
scenes or begin streaming/recording. Restart OBS after installation.

In OBS:

1. Create a scene containing **Game Capture**, targeting the game.
2. Set the scene/output to 1280×720, SDR. Avoid overlays, webcam, display capture,
   or capturing the URFTS output (which would cause feedback).
3. Open **Tools → Spout Output Settings**, set sender name **URFTS**, and enable it.
4. Use the same GPU for OBS and the receiver. The receiver chooses the sender's
   adapter at initialization and does not migrate an active DirectML session.
5. For the 30 FPS game baseline, set OBS video output to 30 FPS too. OBS emits at
   its own cadence; otherwise duplicate scene frames can distort comparisons.

OBS must remain running. This receives its rendered scene as a shared D3D11
texture; it is not OBS Virtual Camera and does not embed OBS's capture hook into
URFTS. OBS may add scene-render overhead. Missing senders produce an explicit
error, never a silent switch to desktop capture. HDR formats are not supported.

## Build and provenance

Build with Python 3.12, pybind11, Microsoft C++ Build Tools and the Windows SDK:

```powershell
& powershell -ExecutionPolicy Bypass -File native/directml_bridge/build.ps1
& powershell -ExecutionPolicy Bypass -File native/obs_spout/build.ps1 -BuildTestSender
```

DirectML build dependencies are Microsoft.ML.OnnxRuntime.DirectML 1.24.4 and
Microsoft.AI.DirectML 1.15.4 under `native/third_party`.
The OBS receiver uses the upstream BSD-2-Clause SpoutDX SDK:
https://github.com/leadedge/Spout2 at commit
`c2bcc12147711d12ace7d5f08e869d774d840f8a`, checked out at `native/third_party/spout2`.
Its license is retained in `native/obs_spout/Spout_LICENSE.txt`. Downloaded SDKs and
compiled outputs are ignored by Git. The separately installed OBS plugin retains
its own licensing; no OBS/plugin source is copied into URFTS.

```powershell
py -3.12 native/directml_bridge/smoke_test.py
py -3.12 native/directml_bridge/smoke_test.py RIFE_v4.25_lite.onnx
py -3.12 native/obs_spout/smoke_test.py
```

Smoke tests use synthetic textures, not user windows. Pixel readback is used for
test assertions only. Production GPU frame processing does not map pixel buffers
back to CPU; ONNX Runtime may still run shape/control operations on CPU.

## Validation boundaries and comparison

GPU stabilization performs low-resolution motion/residual checks, bounded global
colour correction, and full-resolution endpoint reconstruction in uncertain
regions. HUD detection uses a conservative gradient mask, not exact CPU Canny
parity. Moving-scene flicker and ghosting must be checked manually before treating
this as a replacement. The synchronous native loop and cross-context waits still
need performance work; moving processing to GPU alone does not guarantee speedup.

Compare the same game scene, cap, resolution, GPU, model, and inference preset:
30 seconds moving, then 15 seconds stationary. Record presentation FPS, game FPS,
visible artifacts, and loop/acquisition timings. OBS receipt timestamps are not
game-render timestamps, and sender-side drops are not exposed. `replaced` counts
are meaningful for WGC only; OBS currently reports zero, not verified zero loss.
Native `scale submit` currently includes interpolation and pacing work; it is not
an isolated GPU shader timer. No hardware GPU timestamp measurements are claimed.
