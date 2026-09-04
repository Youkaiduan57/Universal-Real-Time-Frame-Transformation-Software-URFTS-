from types import SimpleNamespace
import pytest
import obs_capture
from d3d11_gpu_pipeline import validate_gpu_pipeline_request


def test_obs_requires_native_receiver():
    def missing(_):
        raise ImportError("missing")
    with pytest.raises(RuntimeError, match="Spout receiver"):
        obs_capture.OBSCaptureBackend(loader=missing)


def test_obs_receipt_metadata_and_close(monkeypatch):
    class Receiver:
        device, context = 100, 200
        closed = False
        def __init__(self, name):
            assert name == "URFTS"
        def receive(self):
            return 123, 1280, 720, 87
        def close(self):
            self.closed = True
    monkeypatch.setattr(obs_capture, "D3D11Frame", lambda pointer, **kw: SimpleNamespace(pointer=pointer, **kw))
    capture = obs_capture.OBSCaptureBackend(loader=lambda _: SimpleNamespace(ABI_VERSION=1, Receiver=Receiver))
    frame = capture.grab_gpu_frame()
    assert (frame.width, frame.height, frame.sequence) == (1280, 720, 1)
    assert capture.d3d11_device_pointer.value == 100
    capture.close()
    capture.close()
    assert capture._receiver.closed
    with pytest.raises(RuntimeError, match="owning thread"):
        capture.grab_gpu_frame()


def test_obs_timeout_does_not_fall_back_to_screen_capture():
    receiver = SimpleNamespace(receive=lambda: None, close=lambda: None)
    capture = obs_capture.OBSCaptureBackend(timeout=0, loader=lambda _: SimpleNamespace(ABI_VERSION=1, Receiver=lambda _: receiver))
    with pytest.raises(RuntimeError, match="No new OBS"):
        capture.grab_gpu_frame()
    capture.close()


def test_obs_gpu_scaler_does_not_require_window_selection():
    validate_gpu_pipeline_request(capture_backend="obs", method="bicubic", output_width=1920,
                                  output_height=1080, selected_window=False)


def test_obs_gui_configuration_requires_gpu_pipeline():
    from ui.controller import RuntimeConfiguration
    with pytest.raises(ValueError, match="OBS Spout requires"):
        RuntimeConfiguration(hwnd=0, capture_backend="obs", pipeline="cpu").validate()
    config = RuntimeConfiguration(hwnd=0, capture_backend="obs", pipeline="d3d11",
                                  upscaling_method="bicubic")
    args = config.to_engine_args()
    assert args.capture_backend == "obs" and args.pipeline == "d3d11"
