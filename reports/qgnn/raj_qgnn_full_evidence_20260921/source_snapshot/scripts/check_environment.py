"""Small runtime checks, not an experiment or training run."""
import json, sys, platform, importlib.metadata as md
from pathlib import Path
import torch
import pennylane as qml
from transformers import GPT2Model
from peft import LoraConfig, get_peft_model
ROOT=Path(__file__).resolve().parents[1]
result={"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"platform":platform.platform(),"packages":{n:md.version(n) for n in ("numpy","scipy","pandas","matplotlib","torch","transformers","peft","accelerate","pennylane","scikit-learn")},"cuda":torch.version.cuda,"gpu_checks":[]}
assert sys.prefix != sys.base_prefix
assert not any("quantum-course-design/lib/python3.11/site-packages" in p for p in sys.path)
assert torch.cuda.device_count()==2
for i in range(torch.cuda.device_count()):
    x=torch.randn(32,32,device=f"cuda:{i}",requires_grad=True)
    loss=(x@x.T).square().mean(); loss.backward()
    assert torch.isfinite(x.grad).all()
    torch.cuda.synchronize(i)
    result["gpu_checks"].append({"index":i,"name":torch.cuda.get_device_name(i),"forward_backward":True})
model=GPT2Model.from_pretrained(ROOT/"models/gpt2",local_files_only=True).to("cuda:0")
model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=["c_attn"],fan_in_fan_out=True))
model.eval()
y=model(input_ids=torch.tensor([[10,20,30,40]],device="cuda:0")).last_hidden_state
assert tuple(y.shape)==(1,4,768) and torch.isfinite(y).all()
y.square().mean().backward()
grads=[p.grad for n,p in model.named_parameters() if "lora_" in n and p.requires_grad]
assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
assert any(g.abs().max()>0 for g in grads)
result["gpt2_lora"]={"local_load":True,"output_shape":list(y.shape),"finite_gradients":True,"nonzero_gradient":True,"scope":"dependency smoke test only; no optimizer step or saved trained weights"}
dev=qml.device("default.qubit",wires=2)
@qml.qnode(dev,interface="torch",diff_method="backprop")
def circuit(a):
    qml.RY(a,wires=0); qml.CNOT(wires=[0,1]); return qml.expval(qml.PauliZ(1))
a=torch.tensor(.31,dtype=torch.float64,requires_grad=True)
z=circuit(a); z.backward()
assert torch.allclose(z,torch.cos(a)) and torch.allclose(a.grad,-torch.sin(a))
result["quantum_autograd"]={"backend":"default.qubit","dtype":"float64","value":z.item(),"gradient":a.grad.item(),"passed":True,"scope":"runtime check only; does not validate planned QGNN"}
result["passed"]=True
(ROOT/"reports/ICCT_environment.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
