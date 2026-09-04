# Bicubic CPU-frame checkpoint — 2026-09-04

Preserve this configuration for comparison before further native GPU and OBS work.
These are manual gameplay observations, not a controlled benchmark or a guarantee
of display refresh cadence.

## Configuration

- Capture: WGC, game client 1280×720, Subnautica Below Zero.
- Pipeline: CPU-frame; Shader upscaling, Bicubic.
- Frame generation: IFRNet-S Vimeo90K or RIFE **4.25-lite** (the attached log
  identifies `RIFE_v4.25_lite.onnx`, not full RIFE 4.25).
- Provider: DirectML, device 1. Adapter identity has not been verified.
- Performance preset: 160×90 logical inference size, with model-specific padding.
- Generated amount: 1 generated frame per real frame.
- Target FPS / pacing: Auto; queue depth 2; maximum latency 100 ms.
- Presentation buffer: 0 ms. Output refinement: off.
- Temporal stabilization, HUD stabilization, and FPS overlay: enabled.

The saved GUI settings at checkpoint time select IFRNet-S. Test both models
explicitly; settings from one run should not be attributed to every run.

## Observations

User reports roughly 55–60 FPS with the game capped at 30 FPS, and 60–80 FPS
when capped at 45 FPS. The attached sessions contain intervals near 54–59 FPS
and higher intervals around 63–89 FPS. The log does not timestamp game-cap
changes or distinguish camera motion from stationary periods. Presentation
throughput is not proof of evenly spaced, unique model-generated frames:
stationary/fallback shortcuts are enabled. Flicker absence has not been newly
confirmed for every combination in this test.

## Native work status

The checkpoint includes experimental native DirectML bridge source and host
interfaces already present in the workspace. `RUNTIME_VALIDATED` remains false;
the native bridge is not enabled for normal gameplay. Synthetic constant-colour
validation is not motion-quality or end-to-end presentation validation.
OBS capture is not implemented at this checkpoint. Third-party SDK downloads,
compiled binaries, local GUI state, and session logs are not part of this commit.

Future GPU/OBS work must preserve this CPU-frame comparison path and must not
silently remove temporal/HUD stabilization or switch the saved configuration.
