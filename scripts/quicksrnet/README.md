# QuickSRNet Small x2

Official source: https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/quicksrnet/model/models.py

Official weight release: https://github.com/quic/aimet-model-zoo/releases/tag/phase_2_january_artifacts

Download `quicksrnet_small_2x_checkpoint_float32.pth.tar` from that release into
this directory. `export_small.py` checks the recorded SHA256, loads with
`weights_only=True` and a narrow allowlist for training optimizer metadata,
exports the inference graph, and checks CPU ONNX/PyTorch parity at two sizes.
No AIMET package, training, or unrestricted pickle loading is required.

Run from the project root: `.venv\Scripts\python.exe scripts/quicksrnet/export_small.py`.
The BSD-3-Clause license is in LICENSE.upstream.md and distributed beside the ONNX.
The exact input/output contract and checksums are in the model provenance JSON.

## GUI

Restart and select **Open QuickSRNet Test profile** in the Upscaling card.
This creates a separate profile: native input, DirectML, existing device ID,
frame generation off, tiling off, static reuse off, refinement off, FPS Auto.
Existing profiles are preserved. Reopening the profile retains subsequent edits.

## Initial local comparison (2026-09-04)

Same user-provided 1917x1078 BGR screenshot, native input to 3834x2156 output.
DirectML device 1, one warmup, three measured calls, no tiling/reuse.
These are short offline measurements, not live FPS, and the adapter's physical
name was not established. Both sessions reported DML and CPU providers.

| Model | Mean inference | Mean processor total |
| --- | ---: | ---: |
| QuickSRNet Small x2 | 88.05 ms | 471.33 ms |
| SRVGGNetCompact x2 | 539.62 ms | 904.09 ms |

Inference improved about 6.1x but total processing only about 1.9x. CPU image
conversion/output handling remains expensive; there is no stable-30/60 claim.
Visual fidelity and temporal stability still require an identical-scene video
comparison. The QuickSRNet profile deliberately does not disguise runtime using
smaller input or static caching.
