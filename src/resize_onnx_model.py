"""Generate tiny local ONNX resize models for adapter validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ai_processor import SUPPORTED_IMAGE_LAYOUTS


def create_resize_image_model(
    output_path: str | Path,
    scale: int,
    layout: str = "nchw",
) -> Path:
    """Create a nearest-neighbor float32 image model with a fixed integer scale."""

    normalized_layout = layout.strip().lower()
    if normalized_layout not in SUPPORTED_IMAGE_LAYOUTS:
        supported = ", ".join(SUPPORTED_IMAGE_LAYOUTS)
        raise ValueError(f"Unsupported layout: {layout}. Expected one of: {supported}.")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale not in (1, 2, 3, 4):
        raise ValueError("Scale must be one of: 1, 2, 3, 4.")

    input_shape = (
        [1, 3, "height", "width"]
        if normalized_layout == "nchw"
        else [1, "height", "width", 3]
    )
    output_shape = (
        [1, 3, "output_height", "output_width"]
        if normalized_layout == "nchw"
        else [1, "output_height", "output_width", 3]
    )
    scales = (
        np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32)
        if normalized_layout == "nchw"
        else np.array([1.0, float(scale), float(scale), 1.0], dtype=np.float32)
    )
    scales_initializer = numpy_helper.from_array(scales, name="scales")
    resize_node = helper.make_node(
        "Resize",
        ["input", "", "scales"],
        ["output"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [resize_node],
        f"resize_image_{normalized_layout}_{scale}x",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializer=[scales_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="UniversalUpscaler",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, resolved_path)
    return resolved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local ONNX resize validation model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--layout", choices=SUPPORTED_IMAGE_LAYOUTS, default="nchw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = create_resize_image_model(args.output, args.scale, args.layout)
    print(f"Generated {args.layout.upper()} {args.scale}x resize model: {model_path}")


if __name__ == "__main__":
    main()
