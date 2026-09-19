"""Release audit adds selected neutral-init replay to independent structural acceptance."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.check_qgnn_round3_final as audit
from prediction.qgnn_adaptive.model import build_model
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_adaptive import evaluate,atomic_json
torch.set_num_threads(4)
audit.OUT=ROOT/"reports/qgnn/round3_release_acceptance.json"
audit.stage("start_release_audit")
audit.audit_rows();audit.containment();audit.provenance()
saved=audit.load("reports/qgnn/round3_neutral_init_summary.json")
values={};audit.report["neutral_calibration"]={}
loader=DataLoader(SinDPredictionDataset("val",0.,ROOT,True),batch_size=32,shuffle=False)
for kind in ("quantum","classical"):
    directory=f"reports/qgnn/round3_neutral_{kind}_small_2026"
    config=audit.load(directory+"/config.json");summary=audit.load(directory+"/summary.json")
    rows=audit.load(directory+"/best_validation_rows.json")["rows"]
    assert config["controller_init"]=="neutral" and config["adaptive_mode"]=="phase_feedback"
    assert summary["status"]=="COMPLETED" and not summary["test_set_used"]
    assert (summary["train_samples"],summary["validation_samples"],summary["global_step"])==(4096,1880,1536)
    for key in ("shared_initialization_sha256","train_indices_sha256"):assert summary[key]==audit.report[key]
    raw=np.array([[r["ADE"],r["FDE"],r["ADE"]+.5*r["FDE"]] for r in rows]);values[kind]=raw
    assert max(abs(raw[:,i].mean()-summary["best_validation"][k]) for i,k in enumerate(("ADE","FDE","J")))<1e-12
    training=audit.load(directory+"/training.json");best=min(training,key=lambda r:r["validation"]["J"])
    assert best["validation"]==summary["best_validation"] and best["epoch"]==summary["best_epoch"]
    cp=torch.load(ROOT/directory/"best.pt",map_location="cpu",weights_only=False)
    model=build_model(kind,controller_init="neutral").cuda().eval()
    missing,extra=model.load_state_dict(cp["model_state"],strict=False)
    assert not extra and all(k.startswith("llm.gpt2.") and "lora_" not in k for k in missing)
    actual,replayed=evaluate(model,loader,torch.device("cuda:0"))
    error=max(abs(actual[k]-summary["best_validation"][k]) for k in ("ADE","FDE","J"))
    rowerror=max(abs(x[k]-y[k]) for x,y in zip(rows,replayed) for k in ("ADE","FDE"));assert error<1e-6 and rowerror<2e-5
    audit.report["neutral_calibration"][kind]={"actual":actual,"mean_error":error,"max_per_window_error":rowerror,"config":config,"best_epoch":best["epoch"],"checkpoint_sha256":audit.sha(directory+"/best.pt")}
    audit.stage("neutral_"+kind+"_raw_and_checkpoint_verified")
    del model;torch.cuda.empty_cache()
assert audit.report["neutral_calibration"]["quantum"]["config"]["source_sha256"]==audit.report["neutral_calibration"]["classical"]["config"]["source_sha256"]
features=audit.load("reports/qgnn/round3_strata_reference_20260919.json")["rows"]
ref=audit.load("reports/qgnn/round3_reference_quantum_small_2026/best_validation_rows.json")["rows"]
assert [(x["scene_id"],x["start_frame"]) for x in ref]==[(x["scene_id"],x["start_frame"]) for x in rows]
final=np.array([r["closing_pairs_30m_gt_0p5"] for r in ref]);recent=np.array([r["closing_pairs_30m_gt_0p5"] for r in features])
k3=np.array([r["q_k3_rms"] for r in features]);bounds=np.quantile(k3,np.linspace(0,1,6))
groups={"all":np.ones(1880,bool),"recent5_wedges_ge35":np.array([r["active_wedges"] for r in features])>=35,"recent5_triangles_ge6":np.array([r["active_triangles"] for r in features])>=6}
for name,x in [("final",final),("recent5",recent)]:groups.update({f"{name}_closing_lt10":x<10,f"{name}_closing_10_14":(x>=10)&(x<15),f"{name}_closing_ge15":x>=15})
for j in range(5):groups[f"frozen_K3_quintile_{j+1}"]=(k3>=bounds[j])&(k3<=bounds[j+1] if j==4 else k3<bounds[j+1])
for name,sel in groups.items():
    assert int(sel.sum())==saved["strata"][name]["windows"]
    for kind in ("quantum","classical"):
        expected=saved["strata"][name]["metrics"][kind];actual=values[kind][sel].mean(0)
        assert max(abs(actual[i]-expected[k]) for i,k in enumerate(("ADE","FDE","J")))<1e-12
assert not saved["full_train_gate"]["PASS"]
combined=audit.load("reports/qgnn/ROUND3_FINAL_RESULTS_20260919.json");config=audit.load("configs/qgnn_round3_adaptive.json")
assert combined["selected_report"]=="neutral_init" and config["controller_init"]=="neutral"
assert not combined["any_full_train_gate_passed"]
assert saved["strata"]["all"]["metrics"]["quantum"]["J"]<audit.load("reports/qgnn/round3_v2_summary.json")["strata"]["all"]["metrics"]["quantum"]["J"]
audit.report.update(status="PASS",performance_status="FAILED_FULL_TRAIN_GATE",selected_initialization="neutral",selected_full_gate=saved["full_train_gate"],neutral_strata_verified=True)
audit.stage("RELEASE_PASS_NO_PERFORMANCE_PROMOTION")
