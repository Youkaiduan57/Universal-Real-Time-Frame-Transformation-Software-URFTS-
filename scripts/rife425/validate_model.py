import sys, pathlib, json, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import torch, onnx, numpy as np, onnxruntime as ort
from ifnet425 import IFNet
torch.set_num_threads(4)
weights=torch.load(str(ROOT/"models"/"RIFE_v4.25_lite_weights.pkl"),map_location="cpu",weights_only=True)
weights={k.removeprefix("module."):v for k,v in weights.items()}
net=IFNet().eval()
weights={k:v for k,v in weights.items() if not k.startswith(("teacher.","caltime."))}
net.load_state_dict(weights,strict=True)
class Adapter(torch.nn.Module):
 def __init__(self):super().__init__();self.net=net
 def forward(self,a,b):
  return self.net(torch.cat((a,b),dim=1),timestep=.5,scale_list=[32,16,8,4,1])[-1][-1]
adapter=Adapter().eval()

ort.set_default_logger_severity(3)
path=ROOT/"models"/"RIFE_v4.25_lite.onnx"
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.enable_mem_pattern=False;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
cpu=ort.InferenceSession(str(path),sess_options=opts,providers=["CPUExecutionProvider"])
dml=ort.InferenceSession(str(path),sess_options=opts,providers=[("DmlExecutionProvider",{"device_id":1})])
assert "DmlExecutionProvider" in dml.get_providers()
torch.manual_seed(17)
checks=[]
for h,w in [(128,256),(256,384)]:
 a=torch.rand(1,3,h,w);b=torch.rand_like(a)
 with torch.no_grad():expected=adapter(a,b).numpy()
 feed={"frame_0":a.numpy(),"frame_1":b.numpy()}
 c=cpu.run(None,feed)[0];d=dml.run(None,feed)[0]
 cpu_err=float(np.max(np.abs(c-expected)));diff=np.abs(d-expected)
 row={"shape":[h,w],"cpu_max_error":cpu_err,"cpu_mean_error":float(np.abs(c-expected).mean()),"dml_max_error":float(diff.max()),"dml_mean_error":float(diff.mean())}
 print(row,flush=True)
 assert np.abs(c-expected).mean()<.25/255 and cpu_err<8/255
 assert np.isfinite(d).all()
 # GPU warp/convolution rounding can magnify on random-noise texture.
 # Require <1/4 code value average error and <8 code values maximum.
 assert diff.mean()<.25/255 and diff.max()<8/255
 checks.append(row)
path.with_suffix(".validation.json").write_text(json.dumps({"source":"https://drive.google.com/file/d/1zlKblGuKNatulJNFf5jdB-emp9AqGK05/view","weights_sha256":hashlib.sha256(pathlib.Path(str(ROOT/"models"/"RIFE_v4.25_lite_weights.pkl")).read_bytes()).hexdigest(),"onnx_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"checks":checks},indent=2))
print("Bounded CPU and DirectML numerical comparison passed (not bit-identical)",flush=True)
