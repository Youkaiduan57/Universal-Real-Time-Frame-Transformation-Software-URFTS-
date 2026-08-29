"""Export the official IFRNet-S Vimeo90K checkpoint as a midpoint ONNX model.

The official model accepts a third time-step tensor.  URFTS currently generates
evenly spaced frames recursively, so this export deliberately fixes t=0.5 and
keeps the same two-image contract as the existing RIFE interpolator.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
import torch
from torch import nn


def _load_official_model(source_root: Path, weights: Path) -> nn.Module:
    sys.path.insert(0, str(source_root))
    module_path = source_root / "models" / "IFRNet_S.py"
    spec = importlib.util.spec_from_file_location("official_ifrnet_s", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import official IFRNet-S source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    return model.eval()


class MidpointIFRNet(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        timestep = frame_a.new_full((frame_a.shape[0], 1, 1, 1), 0.5)
        return self.model.inference(frame_a, frame_b, timestep)


def export_model(source_root: Path, weights: Path, output: Path, opset: int) -> None:
    model = MidpointIFRNet(_load_official_model(source_root, weights)).eval()
    # Multiples of 16 avoid decoder shape mismatches. Dynamic spatial axes allow
    # URFTS Performance/Balanced/Quality presets to share one model.
    frame_a = torch.rand(1, 3, 192, 320)
    frame_b = torch.rand(1, 3, 192, 320)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = model(frame_a, frame_b).numpy()
        torch.onnx.export(
            model,
            (frame_a, frame_b),
            output,
            input_names=("frame_a", "frame_b"),
            output_names=("frame_mid",),
            dynamic_axes={
                "frame_a": {2: "height", 3: "width"},
                "frame_b": {2: "height", 3: "width"},
                "frame_mid": {2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    graph = onnx.load(str(output))
    helper.set_model_props(graph, {
        "urfts.model_family": "ifrnet",
        "urfts.model_variant": "IFRNet-S Vimeo90K midpoint",
        "urfts.input_alignment": "16",
        "urfts.license": "MIT",
        "urfts.source": "https://github.com/ltkong218/IFRNet",
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"frame_a": frame_a.numpy(), "frame_b": frame_b.numpy()})[0]
    maximum_error = float(np.max(np.abs(reference - actual)))
    if maximum_error > 2e-4:
        raise RuntimeError(f"ONNX validation failed: maximum absolute error {maximum_error}")
    print(f"Exported {output.resolve()}")
    print(f"Maximum PyTorch/ONNX error: {maximum_error:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_model(args.source_root.resolve(), args.weights.resolve(), args.output.resolve(), args.opset)


if __name__ == "__main__":
    main()
