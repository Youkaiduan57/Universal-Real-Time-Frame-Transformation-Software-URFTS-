"""Export the official QuickSRNet Small x2 checkpoint without AIMET dependencies.

Architecture adapted from Qualcomm AIMET Model Zoo QuickSRNetBase/Small.
Copyright (c) 2022 Qualcomm Innovation Center, Inc. BSD-3-Clause;
see LICENSE.upstream.md. Only the pretrained inference graph is reproduced.
"""
from pathlib import Path
import collections
import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(__file__).with_name("quicksrnet_small_2x_checkpoint_float32.pth.tar")
SHA256 = "d95d70f1d2366cb9c28d99f8c7aa5bb07e1ffeaf5d7e30d9c66ab0fa28c6d0f8"
SOURCE = "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/quicksrnet_small_2x_checkpoint_float32.pth.tar"


class QuickSRNetSmallX2(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for input_channels in (3, 32, 32):
            layers.extend((nn.Conv2d(input_channels, 32, 3, padding=1), nn.Hardtanh(0, 1)))
        self.cnn = nn.Sequential(*layers)
        self.conv_last = nn.Conv2d(32, 12, 3, padding=1)
        self.clip_output = nn.Hardtanh(0, 1)
        self.depth_to_space = nn.PixelShuffle(2)

    def forward(self, image):
        return self.depth_to_space(self.clip_output(self.conv_last(self.cnn(image))))


def main():
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Checkpoint hash mismatch; refuse to load unexpected weights.")
    # This official training checkpoint includes optimizer metadata. Never use
    # unrestricted pickle loading; allow only the concrete types it requires.
    with torch.serialization.safe_globals([torch.optim.Adam, collections.defaultdict, dict]):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = QuickSRNetSmallX2().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    destination = ROOT / "models" / "QuickSRNetSmall_x2.onnx"
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.onnx.export(model, torch.rand(1, 3, 32, 48), str(destination),
        opset_version=17, input_names=["image"], output_names=["upscaled"],
        dynamic_axes={"image": {2: "height", 3: "width"},
                      "upscaled": {2: "out_height", 3: "out_width"}})
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    for height, width in ((32, 48), (37, 53)):
        sample = torch.rand(1, 3, height, width)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = session.run(None, {"image": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        errors.append(float(np.max(np.abs(actual - expected))))
    metadata = dict(source=SOURCE, checkpoint_sha256=SHA256,
        onnx_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()),
        layout="NCHW", color="RGB", range=[0, 1], scale=2,
        ops=sorted({node.op_type for node in graph.graph.node}),
        cpu_parity_max_errors=errors)
    destination.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
