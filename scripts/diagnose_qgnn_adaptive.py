"""Post-training dependency and train-only controller diagnostics, not causal ablations."""
import sys,argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.model import build_model
from prediction.qgnn_final.common import physical_graph
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_adaptive import evaluate,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);a=p.parse_args();torch.set_num_threads(4);directory=ROOT/a.run_dir
    cp=torch.load(directory/"best.pt",map_location="cpu",weights_only=False);c=cp["config"]
    qmodel=build_model("quantum",c["seed"],c["depth"],channels=c["channels"],correction_cap_m=c["correction_cap"],adaptive_mode=c["adaptive_mode"]).cuda().eval()
    missing,extra=qmodel.load_state_dict(cp["model_state"],strict=False)
    assert not extra and all(k.startswith("llm.gpt2.") and "lora_" not in k for k in missing)
    q=qmodel.graph;report={"test_set_used":False,"warning":"Post-hoc dependency checks are out-of-distribution interventions, not retrained causal ablations.","epoch":cp["epoch"],"counterfactuals":{}}
    val=SinDPredictionDataset("val",0.,ROOT,True);loader=DataLoader(val,batch_size=32,shuffle=False)
    for mode in ["normal","no_feedback","no_adaptive_controller","no_ZZZ"]:
        q.controller.enabled=mode!="no_adaptive_controller";q.feedback_enabled=mode!="no_feedback" and c["adaptive_mode"]!="history";q.triple_scale=0. if mode=="no_ZZZ" else 1.
        metric,_=evaluate(qmodel,loader,torch.device("cuda:0"));report["counterfactuals"][mode]=metric;atomic_json(directory/"adaptive_diagnostic.json",report);print(mode,json.dumps(metric),flush=True)
    q.controller.enabled=True;q.feedback_enabled=c["adaptive_mode"]!="history";q.triple_scale=1.
    train=SinDPredictionDataset("train",0.,ROOT,True);batch=next(iter(DataLoader(train,batch_size=64,shuffle=False)))
    h=batch["history_state"].cuda();m=batch["vehicle_mask"].cuda()
    with torch.no_grad():
        own,states,risk,tw,em,trace=q.quantum_states(h,m,True)
        edge,w=physical_graph(h,m);d=q.controller.prepare(edge,w,m)
        report["train_only_controller"]={}
        for key,weight in [("lambda2",d["pw"]),("lambda3",d["tw"])]:
            vals=torch.stack([t[key] for t in trace],2)
            weighted=(vals*weight[:,None,None]).sum(-1)/weight.sum(-1)[:,None,None].clamp_min(1e-6)
            report["train_only_controller"][key]={"mean_C_D":weighted.mean(0).tolist(),"scene_std_C_D":weighted.std(0).tolist(),"min":float(vals.min()),"max":float(vals.max()),"saturation_fraction":float(((vals<.25)|(vals>1.75)).float().mean())}
        report["train_only_controller"]["feedback_local"]=q.controller.feedback_local.detach().cpu().tolist()
        report["train_only_controller"]["feedback_global_weight_norm"]=float(q.controller.feedback_global.weight.norm())
        report["train_only_controller"]["scene_head_weight_norm"]=float(q.controller.scene_head.weight.norm())
        report["train_only_controller"]["K3_rms_by_round"]=torch.stack([x["k3"].square().mean().sqrt() for x in trace]).tolist()
    atomic_json(directory/"adaptive_diagnostic.json",report);print("TRAIN_CONTROLLER",json.dumps(report["train_only_controller"]),flush=True)
if __name__=="__main__":main()
