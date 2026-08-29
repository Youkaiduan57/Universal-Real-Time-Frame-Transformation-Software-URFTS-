"""Export the official Real-ESRGAN SRVGGNetCompact x2 weights to ONNX.

Architecture source:
https://github.com/xinntao/Real-ESRGAN/blob/v0.2.3.0/realesrgan/archs/srvgg_arch.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from torch import nn
from torch.nn import functional as torch_functional


OFFICIAL_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/"
    "RealESRGANv2-animevideo-xsx2.pth"
)
OFFICIAL_WEIGHTS_SHA256 = (
    "27985aa2198711ecd72f9bb274ec7b164e018fc9ce2933daaa7c7ab36a2bd3fe"
)
MODEL_SCALE = 2


class SRVGGNetCompactX2(nn.Module):
    """Official SRVGGNetCompact configuration used by the x2 checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        num_in_ch = 3
        num_out_ch = 3
        num_feat = 64
        num_conv = 16

        self.upscale = MODEL_SCALE
        self.body = nn.ModuleList(
            [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        )
        for _ in range(num_conv):
            self.body.extend(
                [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
            )
        self.body.append(
            nn.Conv2d(num_feat, num_out_ch * self.upscale * self.upscale, 3, 1, 1)
        )
        self.upsampler = nn.PixelShuffle(self.upscale)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = input_tensor
        for layer in self.body:
            output = layer(output)
        output = self.upsampler(output)
        base = torch_functional.interpolate(
            input_tensor,
            scale_factor=self.upscale,
            mode="nearest",
        )
        return output + base


def file_sha256(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_model(weights_path: str | Path) -> SRVGGNetCompactX2:
    """Load the official checkpoint without modifying or remapping weights."""

    resolved_path = Path(weights_path)
    actual_hash = file_sha256(resolved_path)
    if actual_hash.lower() != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError(
            "Official weights checksum mismatch: "
            f"expected {OFFICIAL_WEIGHTS_SHA256}, got {actual_hash}."
        )

    checkpoint = torch.load(
        resolved_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError("Official checkpoint does not contain the expected 'params' state dict.")

    model = SRVGGNetCompactX2()
    model.load_state_dict(checkpoint["params"], strict=True)
    model.eval()
    return model


def export_official_model(
    weights_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export and numerically validate the official x2 model as dynamic ONNX."""

    model = load_official_model(weights_path)
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    sample_input = torch.linspace(
        0.0,
        1.0,
        steps=1 * 3 * 12 * 16,
        dtype=torch.float32,
    ).reshape(1, 3, 12, 16)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample_input,
            resolved_output,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "output_height", 3: "output_width"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(resolved_output)
    helper.set_model_props(
        onnx_model,
        {
            "architecture": "SRVGGNetCompact",
            "scale": "2",
            "source_weights": OFFICIAL_WEIGHTS_URL,
            "source_weights_sha256": OFFICIAL_WEIGHTS_SHA256,
            "conversion": "Local FP32 PyTorch-to-ONNX export; weights unchanged",
        },
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, resolved_output)

    with torch.inference_mode():
        expected_output = model(sample_input).numpy()
    session = ort.InferenceSession(
        str(resolved_output),
        providers=["CPUExecutionProvider"],
    )
    actual_output = session.run(["output"], {"input": sample_input.numpy()})[0]
    maximum_error = float(np.max(np.abs(actual_output - expected_output)))
    if not np.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "Converted ONNX output does not match PyTorch output; "
            f"maximum absolute error is {maximum_error}."
        )

    print(f"Official weights SHA-256: {OFFICIAL_WEIGHTS_SHA256}")
    print(f"ONNX/PyTorch maximum absolute error: {maximum_error:.8f}")
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the official SRVGGNetCompact x2 checkpoint to ONNX.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_official_model(args.weights, args.output)
    print(f"Converted SRVGGNetCompact x2 ONNX model: {output_path}")


if __name__ == "__main__":
    main()
