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
a=torch.rand(1,3,128,256);b=torch.rand_like(a)
path=ROOT/"models"/"RIFE_v4.25_lite.onnx"
with torch.no_grad():
 torch.onnx.export(adapter,(a,b),str(path),opset_version=17,
  input_names=["frame_0","frame_1"],output_names=["interpolated"],
  dynamic_axes={name:{2:"height",3:"width"} for name in ("frame_0","frame_1","interpolated")},dynamo=False)
model=onnx.load(path);onnx.checker.check_model(model)
for k,v in {"urfts.input_alignment":"128","urfts.model":"RIFE 4.25 Lite","urfts.timestep":"0.5"}.items():
 item=model.metadata_props.add();item.key=k;item.value=v
onnx.save(model,path)

print("Exported", path)
