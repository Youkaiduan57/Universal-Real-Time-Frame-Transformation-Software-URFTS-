"""Local ONNX export for the official ECCV2022-RIFE v3.6 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F


def _conv(
    input_channels: int,
    output_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.PReLU(output_channels),
    )


def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Match the official v3.6 align-corners backwarp without a shape cache."""

    height = flow.shape[2]
    width = flow.shape[3]
    horizontal = torch.linspace(
        -1.0,
        1.0,
        width,
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 1, width)
    vertical = torch.linspace(
        -1.0,
        1.0,
        height,
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, height, 1)
    base_grid = torch.cat(
        (
            horizontal.expand(image.shape[0], -1, height, -1),
            vertical.expand(image.shape[0], -1, -1, width),
        ),
        dim=1,
    )
    normalized_flow = torch.cat(
        (
            flow[:, 0:1] / ((width - 1.0) / 2.0),
            flow[:, 1:2] / ((height - 1.0) / 2.0),
        ),
        dim=1,
    )
    grid = (base_grid + normalized_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class IFBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int = 90) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(input_channels, channels // 2, 3, 2, 1),
            _conv(channels // 2, channels, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock1 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock2 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock3 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
            nn.PReLU(channels // 2),
            nn.ConvTranspose2d(channels // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
            nn.PReLU(channels // 2),
            nn.ConvTranspose2d(channels // 2, 1, 4, 2, 1),
        )

    def forward(
        self,
        image_pair: torch.Tensor,
        flow: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_pair = F.interpolate(
            image_pair,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        flow = F.interpolate(
            flow,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ) * (1.0 / scale)
        features = self.conv0(torch.cat((image_pair, flow), dim=1))
        features = self.convblock0(features) + features
        features = self.convblock1(features) + features
        features = self.convblock2(features) + features
        features = self.convblock3(features) + features
        flow_delta = self.conv1(features)
        mask_delta = self.conv2(features)
        flow_delta = F.interpolate(
            flow_delta,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ) * scale
        mask_delta = F.interpolate(
            mask_delta,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        return flow_delta, mask_delta


class RIFEv36(nn.Module):
    """Inference-only official v3.6 IFNet producing the midpoint frame."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(11)
        self.block1 = IFBlock(11)
        self.block2 = IFBlock(11)

    def forward(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
    ) -> torch.Tensor:
        image_pair = torch.cat((frame_a, frame_b), dim=1)
        flow = image_pair[:, :4].detach() * 0.0
        mask = image_pair[:, :1].detach() * 0.0
        warped_a = frame_a
        warped_b = frame_b
        merged = frame_a
        for block, scale in zip(
            (self.block0, self.block1, self.block2),
            (4.0, 2.0, 1.0),
        ):
            delta_forward, mask_forward = block(
                torch.cat((warped_a, warped_b, mask), dim=1),
                flow,
                scale,
            )
            delta_backward, mask_backward = block(
                torch.cat((warped_b, warped_a, -mask), dim=1),
                torch.cat((flow[:, 2:4], flow[:, :2]), dim=1),
                scale,
            )
            flow = flow + (
                delta_forward
                + torch.cat(
                    (delta_backward[:, 2:4], delta_backward[:, :2]),
                    dim=1,
                )
            ) / 2.0
            mask = mask + (mask_forward - mask_backward) / 2.0
            warped_a = _warp(frame_a, flow[:, :2])
            warped_b = _warp(frame_b, flow[:, 2:4])
            blend = torch.sigmoid(mask)
            merged = warped_a * blend + warped_b * (1.0 - blend)
        return merged


def load_official_rife_v3_6(weights_path: str | Path) -> RIFEv36:
    state_dict = torch.load(
        Path(weights_path),
        map_location="cpu",
        weights_only=True,
    )
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if not key.removeprefix("module.").startswith("block_tea.")
    }
    model = RIFEv36()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_rife_v3_6(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 16,
) -> Path:
    model = load_official_rife_v3_6(weights_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_a = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    sample_b = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample_a, sample_b),
            str(output_path),
            input_names=("frame_a", "frame_b"),
            output_names=("interpolated",),
            dynamic_axes={
                "frame_a": {0: "batch", 2: "height", 3: "width"},
                "frame_b": {0: "batch", 2: "height", 3: "width"},
                "interpolated": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def validate_export(
    weights_path: str | Path,
    onnx_path: str | Path,
    *,
    height: int = 32,
    width: int = 32,
    tolerance: float = 1e-4,
) -> tuple[float, float]:
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("Validation dimensions must be positive multiples of 32.")
    torch.manual_seed(36)
    frame_a = torch.rand(1, 3, height, width, dtype=torch.float32)
    frame_b = torch.rand(1, 3, height, width, dtype=torch.float32)
    model = load_official_rife_v3_6(weights_path)
    with torch.inference_mode():
        expected = model(frame_a, frame_b).cpu().numpy()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        ["interpolated"],
        {"frame_a": frame_a.numpy(), "frame_b": frame_b.numpy()},
    )[0]
    difference = np.abs(expected - actual)
    maximum_error = float(difference.max())
    mean_error = float(difference.mean())
    if maximum_error > tolerance:
        raise RuntimeError(
            f"ONNX validation failed: maximum error {maximum_error:.8f} "
            f"exceeds tolerance {tolerance:.8f}."
        )
    return maximum_error, mean_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and numerically validate official RIFE v3.6 weights.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--validation-height", type=int, default=32)
    parser.add_argument("--validation-width", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    export_rife_v3_6(args.weights, args.output, opset=args.opset)
    maximum_error, mean_error = validate_export(
        args.weights,
        args.output,
        height=args.validation_height,
        width=args.validation_width,
        tolerance=args.tolerance,
    )
    print(f"Exported ONNX: {args.output.resolve()}")
    print(f"Maximum absolute PyTorch/ONNX error: {maximum_error:.10f}")
    print(f"Mean absolute PyTorch/ONNX error: {mean_error:.10f}")


if __name__ == "__main__":
    main()
