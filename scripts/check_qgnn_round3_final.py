"""Independent round-3 acceptance: raw metrics, checkpoint replay and baseline containment.
No training and no test dataset are constructed. Results are diagnostic validation evidence.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.qgnn_adaptive.model import build_model
from prediction.qgnn_adaptive.classical import SceneAdaptiveClassicalCore
from prediction.qgnn_final.classical import AdaptiveClassicalCore
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_adaptive import evaluate, atomic_json
OUT = ROOT / "reports/qgnn/round3_final_acceptance.json"
report = {"status": "RUNNING", "test_set_used": False, "stages": [], "experiments": {}, "checkpoint_replay": {}}
def load(path):
    return json.loads((ROOT / path).read_text())
def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def stage(name):
    report["stages"].append({"stage": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    atomic_json(OUT, report)
    print(name, flush=True)

def audit_rows():
    raw_path = "reports/qgnn/history_only_quantum_mode_diagnostic_20260919.json"
    raw = load(raw_path)["per_scene"]
    strata = load("reports/qgnn/round3_strata_reference_20260919.json")
    assert len(raw) == len(strata["rows"]) == 1880
    assert strata["source_sha256"] == sha(raw_path)
    assert all(all(row[k] == source[k] for k in row) for row, source in zip(strata["rows"], raw))
    assert all(not any(k.startswith("delta") or k == "q_win_J" for k in row) for row in strata["rows"])
    metadata = [(x["scene_id"], x["start_frame"]) for x in raw]
    runs = {f"v{v}_{kind}": f"reports/qgnn/round3_v{v}_{kind}_small_2026" for v in (1,2,3) for kind in ("quantum", "classical")}
    runs.update({f"reference_{kind}": f"reports/qgnn/round3_reference_{kind}_small_2026" for kind in ("quantum", "classical")})
    values, summaries, configs = {}, {}, {}
    for label, directory in runs.items():
        summary = load(directory + "/summary.json")
        config = load(directory + "/config.json")
        training = load(directory + "/training.json")
        rr = load(directory + "/best_validation_rows.json")["rows"]
        assert summary["status"] == "COMPLETED" and summary["test_set_used"] is False
        assert (summary["train_samples"], summary["validation_samples"], summary["epochs_completed"], summary["global_step"]) == (4096,1880,12,1536)
        for key, expected in {"seed":2026, "snr":0., "epochs":12, "train_limit":4096, "batch_size":32, "depth":3, "channels":4, "correction_cap":16.}.items():
            assert config[key] == expected, (label,key)
        assert len(rr) == 1880 and [x["index"] for x in rr] == list(range(1880))
        assert [(x["scene_id"],x["start_frame"]) for x in rr] == metadata
        a = np.array([[x["ADE"],x["FDE"],x["ADE"]+.5*x["FDE"]] for x in rr])
        assert np.isfinite(a).all()
        error = max(abs(a[:,i].mean()-summary["best_validation"][k]) for i,k in enumerate(("ADE","FDE","J")))
        assert error < 1e-12
        best = min(training, key=lambda x: x["validation"]["J"])
        assert len(training) == 12 and best["epoch"] == summary["best_epoch"]
        assert best["validation"] == summary["best_validation"]
        values[label], summaries[label], configs[label] = a, summary, config
        report["experiments"][label] = {"directory":directory, "metrics":summary["best_validation"], "best_epoch":summary["best_epoch"], "elapsed_seconds":summary["elapsed_seconds"], "raw_mean_max_error":float(error), "rows_sha256":sha(directory+"/best_validation_rows.json")}
    for key in ("shared_initialization_sha256", "train_indices_sha256"):
        assert len({x[key] for x in summaries.values()}) == 1
        report[key] = next(iter(summaries.values()))[key]
    assert len({x["token_sha256"] for x in configs.values()}) == 1
    assert next(iter(configs.values()))["token_sha256"] == sha("configs/qgnn_final_tokens.json")
    for v in (1,2,3):
        assert configs[f"v{v}_quantum"]["source_sha256"] == configs[f"v{v}_classical"]["source_sha256"]
    k3 = np.array([x["q_k3_rms"] for x in raw]); bounds = np.quantile(k3,np.linspace(0,1,6))
    final_rows = load(runs["reference_quantum"]+"/best_validation_rows.json")["rows"]
    final = np.array([x["closing_pairs_30m_gt_0p5"] for x in final_rows])
    recent = np.array([x["closing_pairs_30m_gt_0p5"] for x in raw])
    groups = {"all":np.ones(1880,bool), "recent5_wedges_ge35":np.array([x["active_wedges"] for x in raw])>=35, "recent5_triangles_ge6":np.array([x["active_triangles"] for x in raw])>=6}
    for name, x in (("final",final),("recent5",recent)):
        groups.update({f"{name}_closing_lt10":x<10, f"{name}_closing_10_14":(x>=10)&(x<15), f"{name}_closing_ge15":x>=15})
    for j in range(5):
        groups[f"frozen_K3_quintile_{j+1}"] = (k3>=bounds[j]) & (k3<=bounds[j+1] if j==4 else k3<bounds[j+1])
    report["strata_counts"] = {k:int(v.sum()) for k,v in groups.items()}
    assert report["strata_counts"]["final_closing_ge15"] == 140
    assert report["strata_counts"]["recent5_closing_ge15"] == 115
    report["full_train_gate"] = {}
    for v in (1,2,3):
        saved = load(f"reports/qgnn/round3_v{v}_summary.json")
        for name, mask in groups.items():
            entry = saved["strata"][name]
            assert int(mask.sum()) == entry["windows"]
            for arm, label in (("quantum",f"v{v}_quantum"),("classical",f"v{v}_classical"),("frozen_quantum","reference_quantum"),("frozen_classical","reference_classical")):
                actual = values[label][mask].mean(0)
                expected = np.array([entry["metrics"][arm][k] for k in ("ADE","FDE","J")])
                assert np.max(np.abs(actual-expected)) < 1e-12
        q,c,fq,fc = (values[k] for k in (f"v{v}_quantum",f"v{v}_classical","reference_quantum","reference_classical"))
        ratio = lambda x,y,m: x[m,2].mean()/y[m,2].mean()
        allmask, low, high = (groups[k] for k in ("all","final_closing_lt10","final_closing_ge15"))
        checks = {"overall_J_improves_at_least1pct":100*(1-ratio(q,fq,allmask))>=1, "relative_gap_shrinks_at_least30pct_or_win":ratio(q,c,allmask)-1<=max(0,.7*(ratio(fq,fc,allmask)-1)), "ordinary_J_improves":ratio(q,fq,low)<1, "high_dynamic_J_not_degraded_over1pct":100*(1-ratio(q,fq,high))>=-1, "high_dynamic_QC_gap_not_worse":ratio(q,c,high)<=ratio(fq,fc,high)}
        checks = {k:bool(x) for k,x in checks.items()}
        assert checks == saved["full_train_gate"]["checks"]
        assert all(checks.values()) == saved["full_train_gate"]["PASS"]
        report["full_train_gate"][f"v{v}"] = {"checks":checks,"PASS":all(checks.values())}
    chosen = min((1,2,3),key=lambda v: values[f"v{v}_quantum"][:,2].mean())
    config = load("configs/qgnn_round3_adaptive.json")
    assert config["main_version"] == f"v{chosen}" and chosen == 2
    assert config["main_mode"] == "phase_feedback" and not config["full_train_executed"]
    report["selected_version"] = "v2"
    report["full_train_authorized"] = any(x["PASS"] for x in report["full_train_gate"].values())
    assert not report["full_train_authorized"]
    stage("raw_metrics_strata_fairness_and_gate_recomputed")

def replay_selected():
    val = SinDPredictionDataset("val",0.,ROOT,True)
    loader = DataLoader(val,batch_size=32,shuffle=False)
    for kind in ("quantum","classical"):
        directory = f"reports/qgnn/round3_v2_{kind}_small_2026"
        cp = torch.load(ROOT/directory/"best.pt",map_location="cpu",weights_only=False)
        c = cp["config"]
        assert c["adaptive_mode"] == "phase_feedback"
        model = build_model(kind,seed=c["seed"],depth=c["depth"],channels=c["channels"],correction_cap_m=c["correction_cap"],adaptive_mode=c["adaptive_mode"]).cuda()
        missing,extra = model.load_state_dict(cp["model_state"],strict=False)
        assert not extra and all(k.startswith("llm.gpt2.") and "lora_" not in k for k in missing)
        start = time.perf_counter()
        actual,rows = evaluate(model,loader,torch.device("cuda:0"))
        cached = load(directory+"/best_validation_rows.json")["rows"]
        difference = {k:abs(actual[k]-cp["validation"][k]) for k in ("ADE","FDE","J")}
        row_error = max(abs(a[k]-b[k]) for a,b in zip(rows,cached) for k in ("ADE","FDE"))
        assert max(difference.values()) < 1e-6 and row_error < 2e-5
        report["checkpoint_replay"][kind] = {"actual":actual,"difference":difference,"per_window_max_error":row_error,"checkpoint_sha256":sha(directory+"/best.pt"),"parameters":model.parameter_summary(),"elapsed_seconds":time.perf_counter()-start}
        stage("selected_"+kind+"_1880_window_checkpoint_replay")
        del model
        torch.cuda.empty_cache()

def containment():
    from prediction.qgnn_adaptive.quantum import SceneAdaptiveQuantumCore
    from prediction.qgnn_final.relational import RelationCarryingQuantumCore
    torch.manual_seed(6917)
    report["neutral_controller_containment"] = {}
    for kind,new_cls,old_cls in (("classical",SceneAdaptiveClassicalCore,AdaptiveClassicalCore),("quantum",SceneAdaptiveQuantumCore,RelationCarryingQuantumCore)):
        old = old_cls().double().eval()
        new = new_cls(**({"phase_mode":True} if kind=="quantum" else {})).double().eval()
        missing,extra = new.load_state_dict(old.state_dict(),strict=False)
        assert not extra and all(k.startswith("controller.") for k in missing)
        # A parameter setting, not a classical/quantum routing flag.
        with torch.no_grad():
            for name in ("scene_head","feedback_global"):
                for p in getattr(new.controller,name).parameters(): p.zero_()
            new.controller.relation_gain.zero_()
            new.controller.feedback_local.zero_()
        errors = {}
        for n in (1,3,8,20):
            x = torch.randn(2,20,n,4,dtype=torch.float64)
            mask = torch.ones(2,n,dtype=torch.bool); mask[0,1::3] = False
            with torch.no_grad(): error = float((old(x,mask)-new(x,mask)).abs().max())
            assert error < 1e-8
            errors[str(n)] = error
        report["neutral_controller_containment"][kind] = {"max_error_by_N":errors,"controller_still_enabled":new.controller.enabled}
    changed = subprocess.check_output(["git","diff","--name-only","591030c","--","prediction/qgnn_final","configs/qgnn_final_architecture.json","configs/qgnn_final_tokens.json","scripts/train_qgnn_final.py"],cwd=ROOT,text=True)
    assert not changed.strip(), changed
    report["round2_frozen_sources_unchanged_from"] = "591030c"
    stage("both_frozen_cores_contained_without_disabling_quantum")

def provenance():
    paths = [str(p.relative_to(ROOT)) for d in ("prediction/qgnn_final","prediction/qgnn_adaptive") for p in (ROOT/d).glob("*.py")]
    paths += ["scripts/check_qgnn_round3_final.py","scripts/train_qgnn_adaptive.py","configs/qgnn_round3_adaptive.json","configs/qgnn_final_tokens.json","models/gpt2/config.json","models/gpt2/model.safetensors"]
    paths += [f"data/sind/{location}/{split}/{name}" for split in ("train","val") for location,name in (("splits","samples.npz"),("isac","sensing_cache.npz"))]
    report["source_and_data_sha256"] = {p:sha(p) for p in paths}
    prior = load("reports/qgnn/additional_checks_and_provenance.json")["sha256"]
    for p in paths:
        if p in prior: assert report["source_and_data_sha256"][p] == prior[p]["sha256"]
    engineering = load("reports/qgnn/round3_engineering_v2.json")
    assert engineering["status"] == "PASS"
    report["selected_engineering"] = {"source":"reports/qgnn/round3_engineering_v2.json","source_sha256":sha("reports/qgnn/round3_engineering_v2.json"),"profiling":engineering["profiling"]}
    resume = load("reports/qgnn/round3_resume_regression.json")
    assert resume["status"] == "PASS" and resume["max_parameter_error"] < 1e-6
    report["existing_resume_regression"] = {"status":resume["status"],"max_parameter_error":resume["max_parameter_error"],"scope":"Existing recorded smoke; not rerun by this acceptance script."}
    report["limitations"] = ["Validation selected, overlapping windows, single pilot seed.","All three variants failed full-train gate; no full-train result is inferred.","Neutral containment and checkpoint replay do not prove quantum advantage.","Finite-shot audit keeps final readout exact; not a hardware experiment."]
    stage("provenance_and_existing_engineering_verified")

def main():
    torch.set_num_threads(4)
    report["source_git_head"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    stage("start_final_acceptance")
    audit_rows()
    replay_selected()
    containment()
    provenance()
    report["status"] = "PASS"
    report["performance_status"] = "FAILED_PREDECLARED_FULL_TRAIN_GATE"
    stage("FINAL_ACCEPTANCE_PASS_PERFORMANCE_NOT_ACCEPTED")
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report.update(status="FAILED",error=repr(exc),traceback=traceback.format_exc())
        atomic_json(OUT,report)
        raise
