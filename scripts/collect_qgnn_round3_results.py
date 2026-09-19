"""Collect every Round3 paired pilot, including one non-structural initialization control."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
reports={name:load(f"reports/qgnn/round3_{name}_summary.json") for name in ("v1","v2","v3","neutral_init")}
selected_structure=min(("v1","v2","v3"),key=lambda n:reports[n]["strata"]["all"]["metrics"]["quantum"]["J"])
assert selected_structure=="v2"
selected=min(("v2","neutral_init"),key=lambda n:reports[n]["strata"]["all"]["metrics"]["quantum"]["J"])
initialization="neutral" if selected=="neutral_init" else "specialized"
res={"revision":"ROUND3_FINAL_20260919","selected_structure":"v2 phase_feedback","selected_initialization":initialization,"selected_report":selected,"selection_metric":"overall validation J=ADE+0.5FDE","structural_revisions_used":2,"extra_initialization_calibrations":1,"test_set_used":False,"full_train_executed":False,"five_snr_multiseed_executed":False,"single_quantum_backup":"v3 basis_feedback; whole-model research comparator only","all_scenes_quantum":True,"protocol":{"train":4096,"validation":1880,"epochs":12,"batch_size":32,"seed":2026,"snr_db":0,"correction_cap_m":16},"pairs":{},"selected_strata":reports[selected]["strata"],"full_train_gates":{k:r["full_train_gate"] for k,r in reports.items()}}
for name,r in reports.items():
    res["pairs"][name]={"quantum":r["runs"]["quantum"],"classical":r["runs"]["classical"],"quantum_vs_matched_gain_pct":r["strata"]["all"]["quantum_vs_matched_gain_pct"],"source":f"reports/qgnn/round3_{name}_summary.json"}
res["frozen_cap16_reference"]={k:reports["v1"]["runs"][k] for k in ("frozen_quantum","frozen_classical")}
res["selected_vs_retained_frozen_classical_gain_pct"]={k:100*(1-res["selected_strata"]["all"]["metrics"]["quantum"][k]/res["selected_strata"]["all"]["metrics"]["frozen_classical"][k]) for k in ("ADE","FDE","J")}
res["engineering"]=load("reports/qgnn/round3_engineering_v2.json")["profiling"]
res["finite_shot_feedback_audit"]=load("reports/qgnn/round3_finite_shot_feedback.json")
res["limitations"]=["Every result is single-seed validation pilot with overlapping windows.","Observed K3 association is not evidence of causal intervention benefit.","Model/initialization chosen on validation; no independent confirmation.","Shot audit keeps final readout exact and is not quantum hardware execution.","Full-run time is extrapolated from steps, not a Round3 full-run measurement."]
res["any_full_train_gate_passed"]=any(x["PASS"] for x in res["full_train_gates"].values())
res["performance_status"]="GATE_PASSED_REQUIRES_FULL_TRAIN" if res["any_full_train_gate_passed"] else "FAILED_FULL_TRAIN_GATE"
res["source_git_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
res["input_report_sha256"]={k:hashlib.sha256((ROOT/v["source"]).read_bytes()).hexdigest() for k,v in res["pairs"].items()}
res["acceptance"]={}
for name in ("round3_release_acceptance","round3_release_resume_regression","round3_neutral_init_preflight"):
    p=ROOT/f"reports/qgnn/{name}.json"
    if p.exists():
        info=json.loads(p.read_text());res["acceptance"][name]={"status":info["status"],"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
        if "max_parameter_error" in info:res["acceptance"][name].update({k:info[k] for k in ("max_parameter_error","validation_ADE_difference","validation_FDE_difference")})
path=ROOT/"reports/qgnn/ROUND3_FINAL_RESULTS_20260919.json";path.write_text(json.dumps(res,indent=2)+"\n")
print(json.dumps({k:res[k] for k in ("selected_structure","selected_initialization","selected_report","performance_status","selected_vs_retained_frozen_classical_gain_pct")},indent=2))
