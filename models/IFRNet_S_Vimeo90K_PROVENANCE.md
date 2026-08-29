# IFRNet-S Vimeo90K model provenance

- Architecture and checkpoint: official IFRNet project, https://github.com/ltkong218/IFRNet
- Variant: `IFRNet_S` trained on Vimeo90K
- License: MIT; see `IFRNet_LICENSE.txt`
- Export: `scripts/ifrnet/export_ifrnet_s.py`, midpoint fixed at t=0.5, ONNX opset 17
- URFTS status: experimental; RIFE v3.6 remains the default until live benchmarks show otherwise

The upstream checkpoint is distributed through the download link in the official
IFRNet README. The ONNX graph embeds its family, variant, alignment, license, and
source metadata.
