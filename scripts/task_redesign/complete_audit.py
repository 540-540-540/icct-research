#!/usr/bin/env python3
"""Unit tests and metadata-only audit supplements; no model training/label access."""
import csv,json,hashlib,math,sys,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/task_redesign'))
from audit_history_only import pair_metrics,DT
OUT=ROOT/'reports/task_redesign'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=json.loads((OUT/'history_only_audit_20260920.json').read_text())
    with (OUT/'history_only_membership_20260920.csv').open() as f:
        rr=list(csv.DictReader(f))
    checks={}
    crossing=np.array([[0.,0.,5.,0.],[10.,-10.,0.,5.],[25.,25.,0.,0.]])
    e=pair_metrics(crossing)[0]
    checks['crossing_detected']=bool(e[0,1] and e[1,0])
    following=np.array([[0.,0.,10.,0.],[10.,0.,10.,0.]])
    checks['equal_speed_following_directed']=bool(pair_metrics(following)[0][0,1] and not pair_metrics(following)[0][1,0])
    checks['stationary_density_is_not_interaction']=bool(not pair_metrics(np.array([[0.,0.,0.,0.],[3.,0.,0.,0.]]))[0].any())
    checks['diverging_not_interaction']=bool(not pair_metrics(np.array([[0.,0.,-1.,0.],[10.,0.,1.,0.]]))[0].any())
    z=crossing.copy();z[:,:2]+=[100.,-80.]
    checks['translation_invariant']=bool(np.array_equal(e,pair_metrics(z)[0]))
    angle=.71; R=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    z=np.concatenate((crossing[:,:2]@R,crossing[:,2:]@R),1)
    checks['rotation_invariant']=bool(np.array_equal(e,pair_metrics(z)[0]))
    order=[2,0,1]; eo=pair_metrics(crossing[order])[0]
    checks['permutation_equivariant']=bool(np.array_equal(eo,e[np.ix_(order,order)]))
    checks['tcpa_not_ttc']=bool(abs(pair_metrics(crossing)[6][0,1]-2)<1e-12 and abs(pair_metrics(crossing)[8][0,1]-(2-5/math.sqrt(50)))<1e-12)
    keys=[(r['scene'],r['split'],r['t0'],r['id'],r['horizon']) for r in rr]
    checks['unique_membership_keys']=len(keys)==len(set(keys))
    grouped={}
    for r in rr:
        k=(r['scene'],r['split'],r['t0'],r['id'])
        grouped.setdefault(k,[]).append((int(r['horizon']),int(r['k'])))
    checks['interaction_membership_horizon_independent']=all(len({k for _,k in v})==1 for v in grouped.values())
    checks['future_frame_counts_bounded']=all(0<=int(r['future_points'])<=int(r['horizon']) for r in rr)
    checks['censored_targets_not_removed']=any(r['future_points']=='0' for r in rr)
    checks['test_rows_not_in_manifest']=all(r['split'] in {'train','val'} for r in rr)
    out={'checks':checks,'passed':all(checks.values()),'scope':{'future_labels_opened':False,'model_training':False},'split_summary':{},'train_design_boundaries':{'0':{'fit':[0,9598],'guard':[9599,9658],'design':[9659,11998]},'1':{'fit':[0,5627],'guard':[5628,5687],'design':[5688,7034]}},'design_rebuild_requirement':'Purge identities crossing fit/design using frame metadata, and recompute contexts without those identities. Do not use stale full-train degrees. Within fit carve last15% as stopping-dev with another60frame gap and identity purge. Formal val is not used to tune thresholds or to choose epochs.'}
    for split in ['train','val']:
        rows=[r for r in rr if r['split']==split and r['horizon']=='40']
        crit=[r for r in rows if int(r['k'])>=1]
        deg=np.array([int(r['k']) for r in rows]); nn=np.array([int(r['n50']) for r in rows])
        out['split_summary'][split]={'full_targets':len(rows),'critical_targets':len(crit),'critical_rate':len(crit)/len(rows),'critical_by_scene':{str(s):sum(int(r['scene'])==s for r in crit) for s in [0,1]},'critical_endpoint_coverage':sum(r['endpoint']=='True' for r in crit)/len(crit),'full_endpoint_coverage':sum(r['endpoint']=='True' for r in rows)/len(rows),'target_wedges_sum':int(np.sum(deg*(deg-1)//2)),'target_wedges_mean':float(np.mean(deg*(deg-1)/2)),'target_pair_density_mean':float(np.mean(deg/np.maximum(nn,1))),'target_wedge_density_mean':float(np.mean(np.where(nn>=2,deg*(deg-1)/np.maximum(nn*(nn-1),1),0))),'radius_context_more_than7_fraction':float(np.mean(nn>7)),'risk_neighbors_more_than7_targets':int(np.sum(deg>7)),'critical_zero_future_labels':sum(r['future_points']=='0' for r in crit),'critical_ADE_eligible_origins':len({(r['scene'],r['t0']) for r in crit if int(r['future_points'])>0}),'critical_FDE_eligible_origins':len({(r['scene'],r['t0']) for r in crit if r['endpoint']=='True'}),'common_20_30_40_targets':sum({h for h,_ in v}=={20,30,40} for k,v in grouped.items() if k[1]==split)}
    evidence_paths=['docs/qgnn/QGNN_HISTORICAL_EXPERIMENT_UNIFIED_AUDIT_20260920.md','reports/qgnn/QGNN_HISTORICAL_EXPERIMENT_UNIFIED_AUDIT_20260920.json','docs/task_redesign/GPT6PRO_TASK_REDESIGN_TASK_20260920.md','configs/sind_prediction.json','configs/sind_target_prediction.json','configs/sind_controlled_isac.json','reports/sind/sind_dataset_build.json','reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json','reports/q0/llm_marginal_effect_audit_0db.json','reports/qgnn/raj_pennylane_p1_1/repair_preflight_20260920.json','reports/qgnn/raj_pennylane_p2/preflight_20260920.json','reports/qgnn/sind_target_self_repair/preflight.json','docs/qgnn/RAJ_PENNYLANE_RESIDUAL_MASTER_TASK_20260920.md','docs/qgnn/RAJ_PENNYLANE_P2_RESIDUAL_FREEZE_20260920.md','prediction/qgnn_raj_pennylane/residual.py','prediction/qgnn_raj_pennylane/self_baselines.py','prediction/q0/motion_token_llm.py','prediction/qgnn_final/model.py','tools/data_preprocessing/build_sind_high_interaction.py','frontend/controlled_isac/automatum_frontend.py']
    out['evidence_files']={p:{'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in evidence_paths}
    gz=OUT/'history_only_membership_20260920.csv.gz'
    with gz.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f: f.write((OUT/'history_only_membership_20260920.csv').read_bytes())
    out['membership_gzip']={'sha256':sha(gz),'bytes':gz.stat().st_size,'rows':len(rr),'contains_future_coordinates':False}
    (OUT/'audit_supplement_20260920.json').write_text(json.dumps(out,indent=2)+chr(10))
    print(json.dumps(out,indent=2))
    if not out['passed']: raise SystemExit(1)
if __name__=='__main__':main()
