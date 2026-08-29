# RIFE v3.6 model provenance

- Upstream project: `hzwer/ECCV2022-RIFE`
- Repository: https://github.com/hzwer/ECCV2022-RIFE
- Pinned upstream commit: `5d8adbdd40e12c2c8f91930eff838aebe561c086`
- Official v3.6 weights: https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing
- Official archive SHA-256: `4D4970B0953CA679CE5D5C972D47A041BE10D12FA29FE86F1408B4497B2558E8`
- `flownet.pkl` SHA-256: `FE854FC8996547C953F732AAA3B78CAE76CC0A12833AE856EA0749C4C570D7D8`
- Converted ONNX SHA-256: `CD689BACEB0657AB11AC82A2C55587B6146631D00EB78D3EB186455AC8BEF905`

The upstream v3.6 release provides PyTorch weights but no official ONNX model.
`src/rife_v3_6_onnx.py` reproduces the inference-only official v3.6 IFNet,
loads all inference weights strictly, and omits only the training-only teacher
block. The exporter creates a float32 NCHW model with two RGB inputs, one RGB
midpoint output, and dynamic batch, height, and width axes.

Conversion command:

```powershell
.\.venv\Scripts\python.exe src\rife_v3_6_onnx.py --weights models\RIFE_v3.6_flownet.pkl --output models\RIFE_v3.6.onnx --opset 16 --validation-height 32 --validation-width 32 --tolerance 0.0001
```

Deterministic 32x32 validation against PyTorch produced a maximum absolute
error of `0.0000339746` and a mean absolute error of `0.0000049311`.
Additional dynamic-resolution validation at 64x32 produced a maximum absolute
error of `0.0000417233` and a mean absolute error of `0.0000054043`.
