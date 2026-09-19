"""Finite-shot feedback-only sensitivity; final readout stays exact. Not a hardware benchmark."""
import sys,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.model import build_model
import prediction.qgnn_adaptive.quantum as module
from prediction.qgnn_final.relational import cumulants as exact_cumulants
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_adaptive import atomic_json
torch.set_num_threads(4)
directory=ROOT/"reports/qgnn/round3_v2_quantum_small_2026"
cp=torch.load(directory/"best.pt",map_location="cpu",weights_only=False)
model=build_model("quantum",adaptive_mode="phase_feedback").cuda().eval()
missing,extra=model.load_state_dict(cp["model_state"],strict=False)
assert not extra and all(k.startswith("llm.gpt2.") and "lora_" not in k for k in missing)
ds=SinDPredictionDataset("train",0.,ROOT,True);indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(6171))[:16]
batch=next(iter(DataLoader(Subset(ds,indices.tolist()),batch_size=16)))
h=batch["history_state"].cuda();mask=batch["vehicle_mask"].cuda();ts=batch["history_timestamp"].cuda()
with torch.no_grad():reference=model(h,mask,ts)["prediction"]
report={"split":"train","test_set_used":False,"samples":16,"indices":indices.tolist(),"note":"Shot noise only in intermediate K2/K3 feedback; readout exact, no gate/readout hardware noise. Draw prefix Z samples from repeated preparations; not nondestructive measurement.","results":[]}
out=ROOT/"reports/qgnn/round3_finite_shot_feedback.json"
for shots in [256,1024,8192,65536]:
    values=[];started=time.perf_counter()
    def estimator(state,n):
        prob=state.abs().square();sample=torch.multinomial(prob,shots,replacement=True)
        freq=torch.zeros_like(prob).scatter_add(1,sample,torch.ones_like(sample,dtype=prob.dtype))/shots
        return exact_cumulants(freq.sqrt().to(state.dtype),n)
    try:
        module.cumulants=estimator
        with torch.no_grad():
            for repeat in range(3):
                torch.manual_seed(6180+repeat);prediction=model(h,mask,ts)["prediction"]
                diff=(prediction-reference).square().sum(-1)[mask[:,None].expand(-1,20,-1)]
                values.append(float(diff.mean().sqrt()))
    finally:module.cumulants=exact_cumulants
    report["results"].append({"shots_per_intermediate_Z_setting":shots,"feedback_preparations_per_scene":8*shots,"position_vector_RMS_deviation_m":values,"mean_deviation_m":sum(values)/len(values),"simulator_elapsed_s":time.perf_counter()-started})
    atomic_json(out,report);print(json.dumps(report["results"][-1]),flush=True)
report["status"]="COMPLETED";atomic_json(out,report)
