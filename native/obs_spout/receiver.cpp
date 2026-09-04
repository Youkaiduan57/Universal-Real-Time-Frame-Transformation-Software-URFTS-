// Original URFTS wrapper around the BSD-2-Clause SpoutDX SDK.
#include <SpoutDX.h>
#include <wrl/client.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
namespace py = pybind11;
using Microsoft::WRL::ComPtr;

class Receiver {
    spoutDX receiver;
    bool closed = false;
public:
    explicit Receiver(const std::string& name) {
        if (name.empty()) throw std::invalid_argument("An explicit OBS Spout sender name is required");
        receiver.SetReceiverName(name.c_str());
        // Choose the sender's adapter once; never silently migrate a live device
        // already shared with the scaler and DirectML session.
        const int adapter = receiver.GetSenderAdapter(name.c_str());
        if (adapter < 0) throw std::runtime_error("OBS sender not found. Enable Spout output named URFTS in OBS.");
        if (!receiver.SetAdapter(adapter) || !receiver.OpenDirectX11())
            throw std::runtime_error("Could not open the OBS sender's D3D11 adapter");
        receiver.SetAdapterAuto(false);
    }
    ~Receiver() { close(); }
    uintptr_t device() { return reinterpret_cast<uintptr_t>(receiver.GetDX11Device()); }
    uintptr_t context() { return reinterpret_cast<uintptr_t>(receiver.GetDX11Context()); }
    py::object receive() {
        if (closed) throw std::runtime_error("OBS receiver is closed");
        if (!receiver.ReceiveTexture()) return py::none();
        receiver.IsUpdated();
        if (!receiver.IsFrameNew()) return py::none();
        auto source = receiver.GetSenderTexture();
        if (!source) return py::none();
        D3D11_TEXTURE2D_DESC desc{};
        source->GetDesc(&desc);
        if (desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM)
            throw std::runtime_error("OBS Spout requires SDR BGRA8/RGBA8 output; disable HDR for this experiment");
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        ComPtr<ID3D11Texture2D> owned;
        if (FAILED(receiver.GetDX11Device()->CreateTexture2D(&desc, nullptr, &owned)))
            throw std::runtime_error("Could not allocate an owned OBS frame");
        receiver.GetDX11Context()->CopyResource(owned.Get(), source);
        // Copy on the receiver's context before the next receive reuses source.
        auto result = py::make_tuple(reinterpret_cast<uintptr_t>(owned.Get()), desc.Width,
                                     desc.Height, static_cast<unsigned>(desc.Format));
        owned.Detach();
        return result;
    }
    void close() {
        if (closed) return;
        closed = true;
        receiver.ReleaseReceiver();
        receiver.CloseDirectX11();
    }
};
PYBIND11_MODULE(_urfts_obs_spout, m) {
    m.attr("ABI_VERSION") = 1;
    py::class_<Receiver>(m, "Receiver")
        .def(py::init<const std::string&>())
        .def_property_readonly("device", &Receiver::device)
        .def_property_readonly("context", &Receiver::context)
        .def("receive", &Receiver::receive)
        .def("close", &Receiver::close);
}
