"""Paired validation analysis with frozen subgroup definitions; never loads test."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(d,name):return json.loads((ROOT/d/name).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument("--quantum-dir",required=True);p.add_argument("--classical-dir",required=True);p.add_argument("--tag",required=True);p.add_argument("--reference-prefix",default="reports/qgnn/round3_reference_");a=p.parse_args()
    dirs={"quantum":a.quantum_dir,"classical":a.classical_dir,"frozen_quantum":a.reference_prefix+"quantum_small_2026","frozen_classical":a.reference_prefix+"classical_small_2026"}
    summaries={k:read(d,"summary.json") for k,d in dirs.items()};rows={k:read(d,"best_validation_rows.json")["rows"] for k,d in dirs.items()}
    assert all(s["status"]=="COMPLETED" for s in summaries.values())
    assert all(s["test_set_used"] is False for s in summaries.values())
    assert len({s["shared_initialization_sha256"] for s in summaries.values()})==1
    assert len({s["train_indices_sha256"] for s in summaries.values()})==1
    features=json.loads((ROOT/"reports/qgnn/round3_strata_reference_20260919.json").read_text())["rows"]
    ref=rows["frozen_quantum"];assert len(ref)==len(features)==1880
    for values in rows.values():
        assert len(values)==1880
        assert all((x["index"],x["scene_id"],x["start_frame"])==(y["index"],y["scene_id"],y["start_frame"]) for x,y in zip(values,ref))
    assert all((x["scene_id"],x["start_frame"])==(y["scene_id"],y["start_frame"]) for x,y in zip(ref,features))
    final=np.array([x["closing_pairs_30m_gt_0p5"] for x in ref]);recent=np.array([x["closing_pairs_30m_gt_0p5"] for x in features])
    k3=np.array([x["q_k3_rms"] for x in features]);bounds=np.quantile(k3,np.linspace(0,1,6))
    groups={"all":np.ones(1880,bool),"final_closing_lt10":final<10,"final_closing_10_14":(final>=10)&(final<15),"final_closing_ge15":final>=15,"recent5_closing_lt10":recent<10,"recent5_closing_10_14":(recent>=10)&(recent<15),"recent5_closing_ge15":recent>=15,"recent5_wedges_ge35":np.array([x["active_wedges"] for x in features])>=35,"recent5_triangles_ge6":np.array([x["active_triangles"] for x in features])>=6}
    for k in range(5):groups[f"frozen_K3_quintile_{k+1}"]=(k3>=bounds[k])&((k3<=bounds[k+1]) if k==4 else (k3<bounds[k+1]))
    values={k:np.array([[x["ADE"],x["FDE"],x["ADE"]+.5*x["FDE"]] for x in rr]) for k,rr in rows.items()}
    report={"tag":a.tag,"test_set_used":False,"notes":["Same cap16 4096/1880/12epoch comparison.","K3 strata use frozen R2 history response, not labels or adaptive outcome selection.","Overlapping windows; all subgroup statistics are descriptive, not independent-sample significance."],"runs":summaries,"directories":dirs,"frozen_K3_bounds":bounds.tolist(),"strata":{}}
    for label,sel in groups.items():
        ms={k:dict(zip(["ADE","FDE","J"],v[sel].mean(0).tolist())) for k,v in values.items()}
        gains={metric:100*(1-ms["quantum"][metric]/ms["classical"][metric]) for metric in ["ADE","FDE","J"]}
        improvements={metric:100*(1-ms["quantum"][metric]/ms["frozen_quantum"][metric]) for metric in ["ADE","FDE","J"]}
        report["strata"][label]={"windows":int(sel.sum()),"metrics":ms,"quantum_vs_matched_gain_pct":gains,"adaptive_vs_frozen_quantum_gain_pct":improvements}
    s=report["strata"];overall=s["all"];m=overall["metrics"]
    oldgap=m["frozen_quantum"]["J"]/m["frozen_classical"]["J"]-1;newgap=m["quantum"]["J"]/m["classical"]["J"]-1
    high=s["final_closing_ge15"]["metrics"];oldhighgap=high["frozen_quantum"]["J"]/high["frozen_classical"]["J"]-1;newhighgap=high["quantum"]["J"]/high["classical"]["J"]-1
    gate={"overall_J_improves_at_least1pct":overall["adaptive_vs_frozen_quantum_gain_pct"]["J"]>=1.,"relative_gap_shrinks_at_least30pct_or_win":newgap<=max(0.,oldgap*.7),"ordinary_J_improves":s["final_closing_lt10"]["adaptive_vs_frozen_quantum_gain_pct"]["J"]>0.,"high_dynamic_J_not_degraded_over1pct":s["final_closing_ge15"]["adaptive_vs_frozen_quantum_gain_pct"]["J"]>=-1.,"high_dynamic_QC_gap_not_worse":newhighgap<=oldhighgap}
    report["full_train_gate"]={"checks":gate,"PASS":all(gate.values()),"old_relative_J_gap":oldgap,"new_relative_J_gap":newgap,"old_high_dynamic_relative_gap":oldhighgap,"new_high_dynamic_relative_gap":newhighgap}
    out=ROOT/f"reports/qgnn/round3_{a.tag}_summary.json";out.write_text(json.dumps(report,indent=2));print(json.dumps({"overall":s["all"],"high_dynamic":s["final_closing_ge15"],"gate":report["full_train_gate"]},indent=2))
if __name__=="__main__":main()
