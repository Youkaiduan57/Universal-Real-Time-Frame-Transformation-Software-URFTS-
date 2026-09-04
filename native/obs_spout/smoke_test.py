"""GPU Spout receiver roundtrip using our synthetic sender, not OBS/game capture."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from obs_capture import OBSCaptureBackend
spec=importlib.util.spec_from_file_location('gpu_test_helpers',ROOT/'native/directml_bridge/smoke_test.py')
helpers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

name='URFTS-test-'+uuid.uuid4().hex[:12]
process=subprocess.Popen([str(ROOT/'native/build/obs_spout/test_sender.exe'),name],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
    creationflags=subprocess.CREATE_NO_WINDOW)
capture=None
first=second=None
try:
    assert process.stdout.readline().strip()=='sent'
    capture=OBSCaptureBackend(name)
    first=capture.grab_gpu_frame()
    assert first.dxgi_format==28
    helpers.verify_pixels(capture.d3d11_device_pointer,capture.d3d11_context_pointer,first.texture_pointer)
    process.stdin.write('green\n'); process.stdin.flush()
    assert process.stdout.readline().strip()=='sent'
    second=capture.grab_gpu_frame()
    helpers.verify_pixels(capture.d3d11_device_pointer,capture.d3d11_context_pointer,
                          second.texture_pointer,[0,255,0])
    helpers.verify_pixels(capture.d3d11_device_pointer,capture.d3d11_context_pointer,first.texture_pointer)
    print('SPOUT GPU ROUNDTRIP + RETAINED FRAME PASS (OBS gameplay not tested)')
finally:
    if first: first.close()
    if second: second.close()
    if capture: capture.close()
    if process.poll() is None:
        process.stdin.write('quit\n'); process.stdin.flush()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
