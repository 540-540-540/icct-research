"""Random-train controller audit; no validation or test labels."""
import sys,json,argparse,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.quantum import SceneAdaptiveQuantumCore
from prediction.qgnn_adaptive.classical import SceneAdaptiveClassicalCore
from prediction.qgnn_final.classical import AdaptiveClassicalCore
from prediction.qgnn_final.common import physical_graph
from frontend.sind_prediction_dataset import SinDPredictionDataset
p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);args=p.parse_args();torch.set_num_threads(4)
d=ROOT/args.run_dir;cp=torch.load(d/"best.pt",map_location="cpu",weights_only=False);c=cp["config"]
q=SceneAdaptiveQuantumCore(c["depth"],c["channels"],c["seed"]+800003,c["adaptive_mode"]!="history",c["adaptive_mode"]=="phase_feedback",c["adaptive_mode"]=="basis_feedback").eval()
q.load_state_dict({k[6:]:v for k,v in cp["model_state"].items() if k.startswith("graph.")},strict=True)
train=SinDPredictionDataset("train",0.,ROOT,True);idx=torch.randperm(len(train),generator=torch.Generator().manual_seed(4811))[:512]
vals={"lambda2":[],"lambda3":[]};k3=[]
with torch.no_grad():
    for data in DataLoader(Subset(train,idx.tolist()),batch_size=32):
        h=data["history_state"];m=data["vehicle_mask"];own,states,risk,tw,em,trace=q.quantum_states(h,m,True)
        e,w=physical_graph(h,m);desc=q.controller.prepare(e,w,m)
        for key,weight in [("lambda2",desc["pw"]),("lambda3",desc["tw"])]:
            x=torch.stack([t[key] for t in trace],2);weighted=(x*weight[:,None,None]).sum(-1)/weight.sum(-1)[:,None,None].clamp_min(1e-6)
            vals[key].append(weighted)
        k3.append(torch.stack([t["k3"].square().mean((1,2)).sqrt() for t in trace],1))
report={"split":"train","samples":512,"sampling_seed":4811,"test_set_used":False}
for key,chunks in vals.items():
    x=torch.cat(chunks)
    report[key]={"mean_C_D":x.mean(0).tolist(),"scene_std_C_D":x.std(0).tolist(),"mean_scene_std":float(x.std(0).mean())}
report["K3_rms_round_mean"]=torch.cat(k3).mean(0).tolist()
report["status"]="COMPLETED"
(d/"random_train_controller_audit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
