#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3d11on12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <DirectML.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#include "stabilize.h"

using Microsoft::WRL::ComPtr;
namespace py = pybind11;

static void check_hr(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        char text[160];
        sprintf_s(text, "%s failed with HRESULT 0x%08X", operation, static_cast<unsigned>(hr));
        throw std::runtime_error(text);
    }
}

static void check_ort(OrtStatus* status) {
    if (!status) return;
    const OrtApi& api = Ort::GetApi();
    std::string message = api.GetErrorMessage(status);
    api.ReleaseStatus(status);
    throw std::runtime_error(message);
}

static const char* kPreprocessShader = R"(
Texture2D<float4> Source : register(t0);
RWStructuredBuffer<float> Output : register(u0);
cbuffer Params : register(b0) { uint SourceWidth; uint SourceHeight; uint Width; uint Height;
    uint LogicalWidth; uint LogicalHeight; uint Reserved0; uint Reserved1; };

float4 sample_bilinear(float2 p) {
    float2 q = clamp(p, 0.0, float2(SourceWidth - 1, SourceHeight - 1));
    int2 a = int2(floor(q));
    int2 b = min(a + 1, int2(SourceWidth - 1, SourceHeight - 1));
    float2 f = frac(q);
    float4 x0 = lerp(Source.Load(int3(a.x, a.y, 0)), Source.Load(int3(b.x, a.y, 0)), f.x);
    float4 x1 = lerp(Source.Load(int3(a.x, b.y, 0)), Source.Load(int3(b.x, b.y, 0)), f.x);
    return lerp(x0, x1, f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    // Resize to the requested aspect ratio, then replicate edges into padding.
    uint2 logical = min(id.xy, uint2(LogicalWidth-1, LogicalHeight-1));
    float2 p = (float2(logical) + 0.5) * float2(SourceWidth, SourceHeight) / float2(LogicalWidth, LogicalHeight) - 0.5;
    float3 rgb = saturate(sample_bilinear(p).rgb);
    uint pixel = id.y * Width + id.x;
    uint plane = Width * Height;
    Output[pixel] = rgb.r;
    Output[plane + pixel] = rgb.g;
    Output[2 * plane + pixel] = rgb.b;
}
)";

static const char* kPostprocessShader = R"(
Texture2D<float4> Source : register(t0);
Texture2D<float4> First : register(t1);
Texture2D<float4> Second : register(t2);
RWTexture2D<float4> Output : register(u0);
cbuffer Params : register(b0) { uint Width; uint Height; uint OutputWidth; uint OutputHeight;
    uint LogicalWidth; uint LogicalHeight; uint Reserved0; uint Reserved1; };

float3 load_rgb(uint2 p) {
    return Source.Load(int3(p,0)).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutputWidth || id.y >= OutputHeight) return;
    float2 p = (float2(id.xy) + 0.5) * float2(LogicalWidth, LogicalHeight) / float2(OutputWidth, OutputHeight) - 0.5;
    p = clamp(p, 0.0, float2(LogicalWidth - 1, LogicalHeight - 1));
    uint2 a = uint2(floor(p));
    uint2 b = min(a + 1, uint2(LogicalWidth - 1, LogicalHeight - 1));
    float2 f = frac(p);
    float3 x0 = lerp(load_rgb(uint2(a.x, a.y)), load_rgb(uint2(b.x, a.y)), f.x);
    float3 x1 = lerp(load_rgb(uint2(a.x, b.y)), load_rgb(uint2(b.x, b.y)), f.x);
    uint2 maskpos = min(id.xy * uint2(LogicalWidth,LogicalHeight) / uint2(OutputWidth,OutputHeight),
                        uint2(LogicalWidth-1,LogicalHeight-1));
    bool fallback = Source.Load(int3(maskpos,0)).a > 0.5;
    float3 fullMidpoint = (First.Load(int3(id.xy,0)).rgb+Second.Load(int3(id.xy,0)).rgb)*0.5;
    Output[id.xy] = float4(fallback ? fullMidpoint : saturate(lerp(x0, x1, f.y)), 1.0);
}
)";

struct DmlAllocationDeleter {
    const OrtDmlApi* api=nullptr;
    void operator()(void* allocation) const { if(allocation && api) api->FreeGPUAllocation(allocation); }
};
struct SharedTensor {
    ComPtr<ID3D12Resource> d3d12;
    ComPtr<ID3D11Buffer> d3d11;
    ComPtr<ID3D11UnorderedAccessView> uav;
    ComPtr<ID3D11ShaderResourceView> srv;
    std::unique_ptr<void,DmlAllocationDeleter> dml_allocation{nullptr,DmlAllocationDeleter{}};
    Ort::Value value{nullptr};
};

struct SharedImage {
    ComPtr<ID3D11Texture2D> texture, compute;
    ComPtr<ID3D11ShaderResourceView> srv;
    uint32_t width=0, height=0;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;
};

class Generator {
public:
    Generator(const std::wstring& model_path, int device_id, uintptr_t device_pointer,
              uint32_t width, uint32_t height)
        : env_(ORT_LOGGING_LEVEL_WARNING, "urfts-directml"), width_(width), height_(height), logical_width_(width), logical_height_(height) {
        if (!device_pointer || !width || !height) throw std::invalid_argument("Invalid bridge configuration");
        if (device_id != -1)
            throw std::invalid_argument("Native shared textures require device -1 (capture adapter); cross-adapter inference is not implemented");
        capture_device_.Attach(reinterpret_cast<ID3D11Device*>(device_pointer));
        capture_device_->AddRef();
        capture_device_->GetImmediateContext(capture_context_.ReleaseAndGetAddressOf());

        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> adapter;
        check_hr(capture_device_.As(&dxgi_device), "IDXGIDevice query");
        check_hr(dxgi_device->GetAdapter(adapter.ReleaseAndGetAddressOf()), "GetAdapter");
        check_hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(d3d12_.ReleaseAndGetAddressOf())), "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check_hr(d3d12_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue_.ReleaseAndGetAddressOf())),
                 "CreateCommandQueue");
        IUnknown* queues[] = {queue_.Get()};
        check_hr(D3D11On12CreateDevice(
                     d3d12_.Get(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                     queues, 1, 0, d3d11_.ReleaseAndGetAddressOf(),
                     context_.ReleaseAndGetAddressOf(), nullptr),
                 "D3D11On12CreateDevice");
        check_hr(d3d11_.As(&d3d11_1_), "ID3D11Device1 query");
        check_hr(d3d11_.As(&on12_), "ID3D11On12Device query");
        check_hr(DMLCreateDevice(d3d12_.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                 IID_PPV_ARGS(dml_.ReleaseAndGetAddressOf())), "DMLCreateDevice");

        compile_shader(kPreprocessShader, preprocess_);
        compile_shader(kPostprocessShader, postprocess_);
        compile_shader(kStabilizeShader, correct_, "Correct");
        compile_shader(kStabilizeShader, composite_, "Composite");
        create_constant_buffer();

        const OrtApi& api = Ort::GetApi();
        check_ort(api.GetExecutionProviderApi("DML", ORT_API_VERSION,
                  reinterpret_cast<const void**>(&dml_api_)));
        options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options_.DisableMemPattern();
        check_ort(dml_api_->SessionOptionsAppendExecutionProvider_DML1(options_, dml_.Get(), queue_.Get()));
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);

        Ort::AllocatorWithDefaultOptions allocator;
        auto metadata = session_->GetModelMetadata();
        auto alignment_text = metadata.LookupCustomMetadataMapAllocated("urfts.input_alignment", allocator);
        const unsigned alignment = alignment_text ? static_cast<unsigned>(std::stoul(alignment_text.get())) : 32;
        if (!alignment || alignment > 1024 || width_ > 8192 || height_ > 8192)
            throw std::invalid_argument("Invalid native inference dimensions or model alignment");
        width_ = ((width_ + alignment - 1) / alignment) * alignment;
        height_ = ((height_ + alignment - 1) / alignment) * alignment;
        if (session_->GetInputCount() != 2 || session_->GetOutputCount() != 1)
            throw std::invalid_argument("Native bridge requires two inputs and one output");
        for (size_t i = 0; i < 2; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_[i] = name.get();
        }
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name.get();

        const size_t elements = static_cast<size_t>(3) * width_ * height_;
        input_a_ = create_tensor(elements);
        input_b_ = create_tensor(elements);
        output_ = create_tensor(elements);
        create_work_image(1, 1, correction_texture_, correction_srv_, correction_uav_);
        create_work_image(logical_width_, logical_height_, stable_texture_, stable_srv_, stable_uav_);
    }

    ~Generator() {
        // Destroy session/provider state before backing buffers and devices.
        session_.reset();
        output_.value = Ort::Value{nullptr};
        input_b_.value = Ort::Value{nullptr};
        input_a_.value = Ort::Value{nullptr};
    }

    uintptr_t interpolate(uintptr_t previous, uintptr_t current, uint32_t source_width, uint32_t source_height) {
        if (!previous || !current) throw std::invalid_argument("Null input texture");
        preprocess(reinterpret_cast<ID3D11Texture2D*>(previous), input_a_, source_width, source_height);
        preprocess(reinterpret_cast<ID3D11Texture2D*>(current), input_b_, source_width, source_height);
        wait_d3d11();

        // Keep tensor ownership intact even if Run throws.
        Ort::IoBinding binding(*session_);
        binding.BindInput(input_names_[0].c_str(), input_a_.value);
        binding.BindInput(input_names_[1].c_str(), input_b_.value);
        binding.BindOutput(output_name_.c_str(), output_.value);
        session_->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();

        stabilize();

        ensure_output(source_width, source_height);
        postprocess(source_width, source_height);
        wait_d3d11();
        // Return an independent snapshot, not a reused texture with another
        // AddRef. A caller may retain this result across future interpolation.
        D3D11_TEXTURE2D_DESC desc{};
        output_texture_->GetDesc(&desc);
        desc.MiscFlags=0;
        ComPtr<ID3D11Texture2D> owned;
        check_hr(capture_device_->CreateTexture2D(&desc, nullptr, &owned), "CreateTexture2D(result snapshot)");
        capture_context_->CopyResource(owned.Get(), output_texture_.Get());
        wait_context(capture_device_.Get(), capture_context_.Get());
        return reinterpret_cast<uintptr_t>(owned.Detach());
    }

private:
    void compile_shader(const char* source, ComPtr<ID3D11ComputeShader>& shader, const char* entry="main") {
        ComPtr<ID3DBlob> bytecode, errors;
        HRESULT hr = D3DCompile(source, strlen(source), nullptr, nullptr, nullptr, entry, "cs_5_0",
                                D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                                bytecode.ReleaseAndGetAddressOf(), errors.ReleaseAndGetAddressOf());
        if (FAILED(hr)) {
            std::string detail = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "shader compile failed";
            throw std::runtime_error(detail);
        }
        check_hr(d3d11_->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr,
                                             shader.ReleaseAndGetAddressOf()), "CreateComputeShader");
    }

    void create_constant_buffer() {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = 32;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        check_hr(d3d11_->CreateBuffer(&desc, nullptr, constants_.ReleaseAndGetAddressOf()), "CreateBuffer(constants)");
    }

    SharedTensor create_tensor(size_t elements) {
        SharedTensor tensor;
        const uint64_t bytes = elements * sizeof(float);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        check_hr(d3d12_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(tensor.d3d12.ReleaseAndGetAddressOf())),
                 "CreateCommittedResource(tensor)");
        D3D11_RESOURCE_FLAGS flags{};
        flags.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        flags.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        flags.StructureByteStride = sizeof(float);
        check_hr(on12_->CreateWrappedResource(
                     tensor.d3d12.Get(), &flags, D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_COMMON,
                     IID_PPV_ARGS(tensor.d3d11.ReleaseAndGetAddressOf())),
                 "CreateWrappedResource(tensor)");

        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = static_cast<UINT>(elements);
        check_hr(d3d11_->CreateUnorderedAccessView(tensor.d3d11.Get(), &uav,
                 tensor.uav.ReleaseAndGetAddressOf()), "CreateUAV(tensor)");
        D3D11_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srv.Buffer.NumElements = static_cast<UINT>(elements);
        check_hr(d3d11_->CreateShaderResourceView(tensor.d3d11.Get(), &srv,
                 tensor.srv.ReleaseAndGetAddressOf()), "CreateSRV(tensor)");

        void* allocation=nullptr;
        check_ort(dml_api_->CreateGPUAllocationFromD3DResource(tensor.d3d12.Get(), &allocation));
        tensor.dml_allocation = std::unique_ptr<void,DmlAllocationDeleter>(allocation,DmlAllocationDeleter{dml_api_});
        std::array<int64_t, 4> shape{1, 3, static_cast<int64_t>(height_), static_cast<int64_t>(width_)};
        Ort::MemoryInfo memory("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        tensor.value = Ort::Value::CreateTensor(memory, tensor.dml_allocation.get(), bytes,
                                                shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        return tensor;
    }

    void update_constants(std::array<uint32_t, 4> values) {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check_hr(context_->Map(constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped), "Map(constants)");
        memcpy(mapped.pData, values.data(), 16);
        const std::array<uint32_t, 4> logical{logical_width_, logical_height_, 0, 0};
        memcpy(static_cast<char*>(mapped.pData) + 16, logical.data(), 16);
        context_->Unmap(constants_.Get(), 0);
    }

    void preprocess(ID3D11Texture2D* input, SharedTensor& tensor, uint32_t source_width, uint32_t source_height) {
        // Separate endpoint textures avoid cross-context overwrite and preserve
        // full-resolution detail for the final confidence composite.
        auto& image = images_[&tensor == &input_a_ ? 0 : 1];
        auto& staging_ = image.texture;
        auto& staging_compute_ = image.compute;
        auto& staging_srv_ = image.srv;
        auto& staging_width_ = image.width;
        auto& staging_height_ = image.height;
        auto& staging_format_ = image.format;
        D3D11_TEXTURE2D_DESC source_desc{};
        input->GetDesc(&source_desc);
        if (source_desc.Width != source_width || source_desc.Height != source_height)
            throw std::runtime_error("Input texture dimensions do not match the frame metadata");
        if (!staging_ || staging_width_ != source_width || staging_height_ != source_height || staging_format_ != source_desc.Format) {
            source_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            source_desc.MiscFlags = 0;
            source_desc.Usage = D3D11_USAGE_DEFAULT;
            source_desc.CPUAccessFlags = 0;
            source_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
            check_hr(capture_device_->CreateTexture2D(&source_desc, nullptr, staging_.ReleaseAndGetAddressOf()),
                     "CreateTexture2D(preprocess)");
            ComPtr<IDXGIResource> shared_resource;
            check_hr(staging_.As(&shared_resource), "IDXGIResource(preprocess)");
            HANDLE shared = nullptr;
            check_hr(shared_resource->GetSharedHandle(&shared), "GetSharedHandle(preprocess)");
            HRESULT opened = d3d11_->OpenSharedResource(
                shared, IID_PPV_ARGS(staging_compute_.ReleaseAndGetAddressOf()));
            check_hr(opened, "OpenSharedResource(preprocess)");
            check_hr(d3d11_->CreateShaderResourceView(staging_compute_.Get(), nullptr, staging_srv_.ReleaseAndGetAddressOf()),
                     "CreateSRV(preprocess)");
            staging_width_ = source_width; staging_height_ = source_height; staging_format_ = source_desc.Format;
        }
        capture_context_->CopyResource(staging_.Get(), input);
        wait_context(capture_device_.Get(), capture_context_.Get());
        update_constants({source_width, source_height, width_, height_});
        ID3D11ShaderResourceView* srvs[] = {staging_srv_.Get()};
        ID3D11UnorderedAccessView* uavs[] = {tensor.uav.Get()};
        ID3D11Buffer* cbs[] = {constants_.Get()};
        context_->CSSetShader(preprocess_.Get(), nullptr, 0);
        ID3D11Resource* wrapped[] = {tensor.d3d11.Get()};
        on12_->AcquireWrappedResources(wrapped, 1);
        context_->CSSetShaderResources(0, 1, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context_->CSSetConstantBuffers(0, 1, cbs);
        context_->Dispatch((width_ + 7) / 8, (height_ + 7) / 8, 1);
        ID3D11ShaderResourceView* null_srv[] = {nullptr};
        ID3D11UnorderedAccessView* null_uav[] = {nullptr};
        context_->CSSetShaderResources(0, 1, null_srv);
        context_->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        on12_->ReleaseWrappedResources(wrapped, 1);
        context_->Flush();
    }

    void ensure_output(uint32_t width, uint32_t height) {
        if (output_texture_ && output_width_ == width && output_height_ == height) return;
        output_srv_.Reset(); output_uav_.Reset(); output_texture_.Reset();
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        check_hr(capture_device_->CreateTexture2D(&desc, nullptr, output_texture_.ReleaseAndGetAddressOf()),
                 "CreateTexture2D(output)");
        ComPtr<IDXGIResource> shared_resource;
        check_hr(output_texture_.As(&shared_resource), "IDXGIResource(output)");
        HANDLE shared = nullptr;
        check_hr(shared_resource->GetSharedHandle(&shared), "GetSharedHandle(output)");
        HRESULT opened = d3d11_->OpenSharedResource(
            shared, IID_PPV_ARGS(output_compute_.ReleaseAndGetAddressOf()));
        check_hr(opened, "OpenSharedResource(output)");
        check_hr(d3d11_->CreateUnorderedAccessView(output_compute_.Get(), nullptr,
                 output_uav_.ReleaseAndGetAddressOf()), "CreateUAV(output)");
        output_width_ = width; output_height_ = height;
    }

    void create_work_image(uint32_t w, uint32_t h, ComPtr<ID3D11Texture2D>& texture,
                          ComPtr<ID3D11ShaderResourceView>& srv, ComPtr<ID3D11UnorderedAccessView>& uav) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=w; desc.Height=h; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count=1;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        check_hr(d3d11_->CreateTexture2D(&desc,nullptr,&texture),"CreateTexture2D(stabilization)");
        check_hr(d3d11_->CreateShaderResourceView(texture.Get(),nullptr,&srv),"CreateSRV(stabilization)");
        check_hr(d3d11_->CreateUnorderedAccessView(texture.Get(),nullptr,&uav),"CreateUAV(stabilization)");
    }

    void stabilize() {
        update_constants({width_,height_,logical_width_,logical_height_});
        ID3D11Resource* wrapped[]={output_.d3d11.Get(),input_a_.d3d11.Get(),input_b_.d3d11.Get()};
        on12_->AcquireWrappedResources(wrapped,3);
        ID3D11ShaderResourceView* views[]={output_.srv.Get(),input_a_.srv.Get(),input_b_.srv.Get(),nullptr};
        ID3D11UnorderedAccessView* correction[]={correction_uav_.Get()};
        ID3D11Buffer* cb[]={constants_.Get()};
        context_->CSSetConstantBuffers(0,1,cb);
        context_->CSSetShaderResources(0,4,views);
        context_->CSSetUnorderedAccessViews(0,1,correction,nullptr);
        context_->CSSetShader(correct_.Get(),nullptr,0);
        context_->Dispatch(1,1,1);
        ID3D11UnorderedAccessView* nullu[]={nullptr};
        context_->CSSetUnorderedAccessViews(0,1,nullu,nullptr);
        views[3]=correction_srv_.Get();
        context_->CSSetShaderResources(0,4,views);
        ID3D11UnorderedAccessView* stable[]={stable_uav_.Get()};
        context_->CSSetUnorderedAccessViews(0,1,stable,nullptr);
        context_->CSSetShader(composite_.Get(),nullptr,0);
        context_->Dispatch((logical_width_+7)/8,(logical_height_+7)/8,1);
        ID3D11ShaderResourceView* nulls[]={nullptr,nullptr,nullptr,nullptr};
        context_->CSSetShaderResources(0,4,nulls);
        context_->CSSetUnorderedAccessViews(0,1,nullu,nullptr);
        on12_->ReleaseWrappedResources(wrapped,3);
        context_->Flush();
    }

    void postprocess(uint32_t width, uint32_t height) {
        update_constants({width_, height_, width, height});
        ID3D11ShaderResourceView* srvs[] = {stable_srv_.Get(), images_[0].srv.Get(), images_[1].srv.Get()};
        ID3D11UnorderedAccessView* uavs[] = {output_uav_.Get()};
        ID3D11Buffer* cbs[] = {constants_.Get()};
        context_->CSSetShader(postprocess_.Get(), nullptr, 0);
        context_->CSSetShaderResources(0, 3, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context_->CSSetConstantBuffers(0, 1, cbs);
        context_->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        ID3D11ShaderResourceView* null_srv[] = {nullptr,nullptr,nullptr};
        ID3D11UnorderedAccessView* null_uav[] = {nullptr};
        context_->CSSetShaderResources(0, 3, null_srv);
        context_->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        context_->Flush();
    }

    static void wait_context(ID3D11Device* device, ID3D11DeviceContext* context) {
        D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT, 0};
        ComPtr<ID3D11Query> query;
        check_hr(device->CreateQuery(&desc, query.ReleaseAndGetAddressOf()), "CreateQuery");
        context->End(query.Get());
        context->Flush();
        const ULONGLONG deadline = GetTickCount64() + 5000;
        HRESULT status;
        while ((status = context->GetData(query.Get(), nullptr, 0, 0)) == S_FALSE) {
            check_hr(device->GetDeviceRemovedReason(), "GPU device removed while waiting");
            if (GetTickCount64() >= deadline)
                throw std::runtime_error("GPU synchronization timed out after 5 seconds");
            Sleep(0);
        }
        check_hr(status, "GPU event query");
    }

    void wait_d3d11() { wait_context(d3d11_.Get(), context_.Get()); }

    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::array<std::string, 2> input_names_;
    std::string output_name_;
    const OrtDmlApi* dml_api_ = nullptr;
    uint32_t width_, height_, logical_width_, logical_height_;
    ComPtr<ID3D11Device> capture_device_, d3d11_;
    ComPtr<ID3D11Device1> d3d11_1_;
    ComPtr<ID3D11DeviceContext> capture_context_, context_;
    ComPtr<ID3D11On12Device> on12_;
    ComPtr<ID3D12Device> d3d12_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<IDMLDevice> dml_;
    ComPtr<ID3D11ComputeShader> preprocess_, postprocess_;
    ComPtr<ID3D11ComputeShader> correct_, composite_;
    ComPtr<ID3D11Texture2D> correction_texture_, stable_texture_;
    ComPtr<ID3D11ShaderResourceView> correction_srv_, stable_srv_;
    ComPtr<ID3D11UnorderedAccessView> correction_uav_, stable_uav_;
    std::array<SharedImage,2> images_;
    ComPtr<ID3D11Buffer> constants_;
    SharedTensor input_a_, input_b_, output_;
    ComPtr<ID3D11Texture2D> output_texture_, output_compute_;
    ComPtr<ID3D11ShaderResourceView> output_srv_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    uint32_t output_width_ = 0, output_height_ = 0;
};

PYBIND11_MODULE(_urfts_directml, module) {
    module.attr("ABI_VERSION") = 1;
    module.attr("GPU_RESIDENT_IO") = true;
    module.attr("RUNTIME_VALIDATED") = false;
    module.attr("STABILIZATION_GPU") = true;
    module.attr("PADDED_INPUT") = true;
    module.attr("OUTPUT_DXGI_FORMAT") = static_cast<int>(DXGI_FORMAT_R8G8B8A8_UNORM);
    py::class_<Generator, std::shared_ptr<Generator>>(module, "FrameGenerator");
    module.def("create_frame_generator", [](const std::string& model_path, int device_id,
                                             uintptr_t device, uint32_t width, uint32_t height) {
        return std::make_shared<Generator>(std::filesystem::path(model_path).wstring(), device_id,
                                           device, width, height);
    });
    module.def("interpolate_d3d11", [](const std::shared_ptr<Generator>& generator,
                                       uintptr_t previous, uintptr_t current,
                                       uint32_t width, uint32_t height) {
        py::gil_scoped_release release;
        return generator->interpolate(previous, current, width, height);
    });
    module.def("close_frame_generator", [](std::shared_ptr<Generator>& generator) { generator.reset(); });
}
