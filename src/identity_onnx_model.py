"""Generate a tiny local float32 identity image model for validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from ai_processor import SUPPORTED_IMAGE_LAYOUTS


def create_identity_image_model(
    output_path: str | Path,
    layout: str = "nchw",
) -> Path:
    """Create one dynamic-spatial rank-4 identity ONNX image model."""

    normalized_layout = layout.strip().lower()
    if normalized_layout not in SUPPORTED_IMAGE_LAYOUTS:
        supported = ", ".join(SUPPORTED_IMAGE_LAYOUTS)
        raise ValueError(f"Unsupported layout: {layout}. Expected one of: {supported}.")

    shape = (
        [1, 3, "height", "width"]
        if normalized_layout == "nchw"
        else [1, "height", "width", 3]
    )
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    identity_node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [identity_node],
        f"identity_image_{normalized_layout}",
        [input_info],
        [output_info],
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
        description="Generate a local ONNX identity image model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=SUPPORTED_IMAGE_LAYOUTS,
        default="nchw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = create_identity_image_model(args.output, args.layout)
    print(f"Generated {args.layout.upper()} identity model: {model_path}")


if __name__ == "__main__":
    main()
