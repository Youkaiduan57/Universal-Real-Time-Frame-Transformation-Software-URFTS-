"""Small D3D11 HLSL compilation and shader-program helpers.

Shader programs own the D3D11 shader interfaces they create.  Compiler blobs
are temporary and are always released before program creation returns.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from wgc_capture import _check_hresult, _release, _vtable_function


FULLSCREEN_SCALE_SHADER_SOURCE = b"""
struct VertexOutput
{
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
};

VertexOutput VSMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float2 texcoord = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    output.texcoord = texcoord;
    return output;
}

Texture2D<float4> sourceTexture : register(t0);
SamplerState sourceSampler : register(s0);

cbuffer ScalingConstants : register(b0)
{
    float2 sourceSize;
    float2 outputSize;
    float edgeStrength;
    float sharpeningStrength;
    float sharpeningEnabled;
    float scalingPadding;
};

float4 PSMain(VertexOutput input) : SV_Target
{
    return sourceTexture.Sample(sourceSampler, input.texcoord);
}

float Sinc(float value)
{
    float distance = abs(value);
    if (distance < 1.0e-5)
    {
        return 1.0;
    }
    float radians = 3.14159265358979323846 * distance;
    return sin(radians) / radians;
}

float Lanczos2Weight(float value)
{
    float distance = abs(value);
    if (distance >= 2.0)
    {
        return 0.0;
    }
    return Sinc(distance) * Sinc(distance * 0.5);
}

float4 PSLanczos(VertexOutput input) : SV_Target
{
    float2 sourcePosition = (input.position.xy / outputSize) * sourceSize - 0.5;
    int2 basePosition = int2(floor(sourcePosition));
    int2 maximumPosition = int2(sourceSize) - 1;
    float4 accumulated = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (int y = -1; y <= 2; ++y)
    {
        float weightY = Lanczos2Weight(float(basePosition.y + y) - sourcePosition.y);
        [unroll]
        for (int x = -1; x <= 2; ++x)
        {
            float weightX = Lanczos2Weight(float(basePosition.x + x) - sourcePosition.x);
            float weight = weightX * weightY;
            int2 samplePosition = clamp(basePosition + int2(x, y), int2(0, 0), maximumPosition);
            accumulated += sourceTexture.Load(int3(samplePosition, 0)) * weight;
            weightSum += weight;
        }
    }

    return accumulated / max(weightSum, 1.0e-6);
}

int2 ClampSourcePosition(int2 position)
{
    return clamp(position, int2(0, 0), int2(sourceSize) - 1);
}

float4 LoadSource(int2 position)
{
    return sourceTexture.Load(int3(ClampSourcePosition(position), 0));
}

float SourceLuma(int2 position)
{
    return dot(LoadSource(position).rgb, float3(0.299, 0.587, 0.114));
}

float4 BilinearAtOutputPixel(float2 outputPixel)
{
    float2 texcoord = clamp(outputPixel / outputSize, 0.0, 1.0);
    return sourceTexture.SampleLevel(sourceSampler, texcoord, 0.0);
}

float SourceEdgeMask(int2 center)
{
    float topLeft = SourceLuma(center + int2(-1, -1));
    float top = SourceLuma(center + int2(0, -1));
    float topRight = SourceLuma(center + int2(1, -1));
    float left = SourceLuma(center + int2(-1, 0));
    float right = SourceLuma(center + int2(1, 0));
    float bottomLeft = SourceLuma(center + int2(-1, 1));
    float bottom = SourceLuma(center + int2(0, 1));
    float bottomRight = SourceLuma(center + int2(1, 1));

    float gradientX =
        -topLeft - 2.0 * left - bottomLeft
        + topRight + 2.0 * right + bottomRight;
    float gradientY =
        -topLeft - 2.0 * top - topRight
        + bottomLeft + 2.0 * bottom + bottomRight;
    float magnitude = abs(gradientX) + abs(gradientY);
    return step(12.0 / 255.0, magnitude);
}

float4 PSFsr1Like(VertexOutput input) : SV_Target
{
    float2 outputPixel = input.position.xy;
    float2 texcoord = clamp(outputPixel / outputSize, 0.0, 1.0);
    float2 sourcePosition = texcoord * sourceSize - 0.5;
    int2 nearestPosition = ClampSourcePosition(int2(floor(sourcePosition + 0.5)));

    float4 bilinear = sourceTexture.SampleLevel(sourceSampler, texcoord, 0.0);
    float4 nearest = LoadSource(nearestPosition);
    float edgeBlend = edgeStrength * SourceEdgeMask(nearestPosition);
    float4 center = lerp(bilinear, nearest, edgeBlend);

    if (sharpeningEnabled < 0.5 || sharpeningStrength <= 0.0)
    {
        return center;
    }

    float3 northWest = BilinearAtOutputPixel(outputPixel + float2(-1.0, -1.0)).rgb;
    float3 north = BilinearAtOutputPixel(outputPixel + float2(0.0, -1.0)).rgb;
    float3 northEast = BilinearAtOutputPixel(outputPixel + float2(1.0, -1.0)).rgb;
    float3 west = BilinearAtOutputPixel(outputPixel + float2(-1.0, 0.0)).rgb;
    float3 east = BilinearAtOutputPixel(outputPixel + float2(1.0, 0.0)).rgb;
    float3 southWest = BilinearAtOutputPixel(outputPixel + float2(-1.0, 1.0)).rgb;
    float3 south = BilinearAtOutputPixel(outputPixel + float2(0.0, 1.0)).rgb;
    float3 southEast = BilinearAtOutputPixel(outputPixel + float2(1.0, 1.0)).rgb;

    float3 blurred = (
        northWest + northEast + southWest + southEast
        + 2.0 * (north + west + east + south)
        + 4.0 * center.rgb
    ) / 16.0;
    float3 sharpened = center.rgb + sharpeningStrength * (center.rgb - blurred);
    float3 localMinimum = min(center.rgb, min(min(north, south), min(west, east)));
    float3 localMaximum = max(center.rgb, max(max(north, south), max(west, east)));

    return float4(clamp(sharpened, localMinimum, localMaximum), center.a);
}
"""


_FULLSCREEN_SCALE_PIXEL_ENTRIES = {
    "nearest": b"PSMain",
    "bilinear": b"PSMain",
    "lanczos": b"PSLanczos",
    "fsr1_like": b"PSFsr1Like",
}


class D3D11ShaderError(RuntimeError):
    """Raised when HLSL compilation cannot produce a usable shader."""


def _blob_bytes(blob: ctypes.c_void_p) -> bytes:
    address = _vtable_function(blob, 3, ctypes.c_void_p)(blob)
    size = _vtable_function(blob, 4, ctypes.c_size_t)(blob)
    return ctypes.string_at(address, size)


def _compile_shader(
    source: bytes,
    *,
    source_name: bytes,
    entry_point: bytes,
    target: bytes,
) -> ctypes.c_void_p:
    try:
        compiler = ctypes.WinDLL("d3dcompiler_47")
    except OSError as error:
        raise D3D11ShaderError(
            "d3dcompiler_47.dll is required to compile the built-in D3D11 shaders."
        ) from error

    compile_shader = compiler.D3DCompile
    compile_shader.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    compile_shader.restype = ctypes.c_long
    code = ctypes.create_string_buffer(source)
    blob = ctypes.c_void_p()
    errors = ctypes.c_void_p()
    result = compile_shader(
        code,
        len(source),
        source_name,
        None,
        None,
        entry_point,
        target,
        0,
        0,
        ctypes.byref(blob),
        ctypes.byref(errors),
    )
    if ctypes.c_uint32(result).value & 0x80000000:
        message = _blob_bytes(errors).decode("utf-8", errors="replace") if errors.value else ""
        _release(blob)
        _release(errors)
        raise D3D11ShaderError(
            f"D3DCompile({entry_point.decode()}/{target.decode()}) failed: {message.strip()}"
        )
    _release(errors)
    return blob


class D3D11ShaderProgram:
    """Owned vertex/pixel shader pair compiled from one HLSL source."""

    def __init__(
        self,
        device: ctypes.c_void_p,
        *,
        source: bytes,
        source_name: bytes,
        vertex_entry: bytes,
        pixel_entry: bytes,
    ) -> None:
        self._vertex_shader = ctypes.c_void_p()
        self._pixel_shader = ctypes.c_void_p()
        self._closed = False
        vertex_blob = ctypes.c_void_p()
        pixel_blob = ctypes.c_void_p()
        try:
            vertex_blob = _compile_shader(
                source,
                source_name=source_name,
                entry_point=vertex_entry,
                target=b"vs_4_0",
            )
            pixel_blob = _compile_shader(
                source,
                source_name=source_name,
                entry_point=pixel_entry,
                target=b"ps_4_0",
            )
            self._vertex_shader = self._create_shader(
                device,
                vertex_blob,
                vtable_slot=12,
                operation="ID3D11Device.CreateVertexShader",
            )
            self._pixel_shader = self._create_shader(
                device,
                pixel_blob,
                vtable_slot=15,
                operation="ID3D11Device.CreatePixelShader",
            )
        except Exception:
            self.close()
            raise
        finally:
            _release(pixel_blob)
            _release(vertex_blob)

    @staticmethod
    def _create_shader(
        device: ctypes.c_void_p,
        blob: ctypes.c_void_p,
        *,
        vtable_slot: int,
        operation: str,
    ) -> ctypes.c_void_p:
        shader_bytes = _blob_bytes(blob)
        shader_buffer = ctypes.create_string_buffer(shader_bytes)
        shader = ctypes.c_void_p()
        result = _vtable_function(
            device,
            vtable_slot,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            device,
            shader_buffer,
            len(shader_bytes),
            None,
            ctypes.byref(shader),
        )
        try:
            _check_hresult(result, operation)
        except Exception:
            _release(shader)
            raise
        return shader

    @classmethod
    def fullscreen_scaler(
        cls,
        device: ctypes.c_void_p,
        *,
        method: str,
    ) -> "D3D11ShaderProgram":
        try:
            pixel_entry = _FULLSCREEN_SCALE_PIXEL_ENTRIES[method]
        except KeyError as error:
            raise D3D11ShaderError(f"Unsupported D3D11 shader scaler method: {method}") from error
        return cls(
            device,
            source=FULLSCREEN_SCALE_SHADER_SOURCE,
            source_name=b"UniversalUpscalerD3D11.hlsl",
            vertex_entry=b"VSMain",
            pixel_entry=pixel_entry,
        )

    @property
    def vertex_shader(self) -> ctypes.c_void_p:
        if self._closed:
            raise D3D11ShaderError("D3D11 shader program is closed.")
        return ctypes.c_void_p(self._vertex_shader.value)

    @property
    def pixel_shader(self) -> ctypes.c_void_p:
        if self._closed:
            raise D3D11ShaderError("D3D11 shader program is closed.")
        return ctypes.c_void_p(self._pixel_shader.value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _release(self._pixel_shader)
        _release(self._vertex_shader)
