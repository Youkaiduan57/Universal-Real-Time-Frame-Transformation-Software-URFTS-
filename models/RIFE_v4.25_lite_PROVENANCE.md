# RIFE 4.25 Lite experimental adapter

Source: https://github.com/hzwer/Practical-RIFE
Official package: https://drive.google.com/file/d/1zlKblGuKNatulJNFf5jdB-emp9AqGK05/view
License: MIT, included alongside this file.

The official IFNet_HDv3 architecture and weights were used. Only training-only
teacher/caltime weights were excluded; all inference weights loaded strictly.
The midpoint adapter concatenates two float32 RGB NCHW images and selects the
final merged result using the official [32,16,8,4,1] scale list and timestep 0.5.
Warp uses the existing equivalent align-corners grid-sample implementation.
ONNX opset 17. Dynamic height/width; padding alignment 128 stored in metadata.

Validation JSON records hashes and numerical comparisons at 128x256 and 256x384.
Outputs are not bit-identical to PyTorch: strict initial pointwise tolerances
failed. Final bounded checks required mean absolute error below 0.25/255 and
maximum below 8/255. Observed errors are recorded, not hidden.

Runtime tested on RTX DirectML device 1 with 1080p synthetic output.
Balanced synthetic median: v3.6 22.41 ms; 4.25 Lite 59.32 ms (3 measured calls).
This is not gameplay FPS. New model is for experimental quality testing, not
an established speed upgrade. v3.6 remains the default.

Re-export: run scripts/rife425/export_model.py from this project environment.
Validate: run scripts/rife425/validate_model.py. Requires PyTorch and ONNX.
