"""Aggregate completed validation-only pilots without opening any dataset."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'reports/qgnn'
def load(p):return json.loads(p.read_text())
r={'architecture':'RC-HQGNN-v3','test_set_used':False,'status':'ARCHITECTURE_FROZEN_QUANTUM_ADVANTAGE_NOT_ESTABLISHED','pilot_history':{}}
for d in sorted(base.glob('r2_pilot_v*')):
    if (d/'summary.json').is_file():r['pilot_history'][d.name]=load(d/'summary.json')
dirs={k:base/f'r2_final_full_{k}_0db_seed2026' for k in ['quantum','classical']}
summaries={k:load(d/'summary.json') for k,d in dirs.items()}
configs={k:load(d/'config.json') for k,d in dirs.items()}
assert all(s['status']=='COMPLETED' and s['epochs_completed']==20 and not s['test_set_used'] for s in summaries.values())
keys=['seed','snr','epochs','batch_size','train_limit','depth','channels','quantum_version','correction_cap','lr','source_sha256','token_sha256','git_head']
assert all(configs['quantum'][k]==configs['classical'][k] for k in keys)
for k in ['shared_initialization_sha256','train_indices_sha256','train_samples','validation_samples','global_step']:
    assert summaries['quantum'][k]==summaries['classical'][k]
r['final_protocol']={k:configs['quantum'][k] for k in keys};r['final_runs']=summaries
r['fairness_checks']={'matching_configs':keys,'identical_data_order_and_shared_initialization':True,'equal_steps':9880}
rows={k:load(d/'best_validation_rows.json')['rows'] for k,d in dirs.items()}
assert len(rows['quantum'])==len(rows['classical'])==1880
assert all((a['scene_id'],a['start_frame'])==(b['scene_id'],b['start_frame']) for a,b in zip(rows['quantum'],rows['classical']))
closing=np.array([a['closing_pairs_30m_gt_0p5'] for a in rows['quantum']]);r['strata']={}
for label,mask in [('all',np.ones(1880,dtype=bool)),('closing_ge10',closing>=10),('closing_ge15',closing>=15),('closing_lt10',closing<10)]:
    group={'windows':int(mask.sum())}
    for metric in ['ADE','FDE']:
        q=np.array([x[metric] for x in rows['quantum']])[mask];c=np.array([x[metric] for x in rows['classical']])[mask]
        group[metric]={'quantum':float(q.mean()),'classical':float(c.mean()),'quantum_gain_percent':float(100*(1-q.mean()/c.mean())),'quantum_better_window_percent':float(100*(q<c).mean())}
    r['strata'][label]=group
r['statistical_scope']='Single-seed validation, overlapping windows; descriptive paired comparison only, no IID-window confidence claims.'
r['measurement_scope']='Exact quantum-circuit simulation on RTX4090, not hardware quantum speedup.'
r['engineering']=load(base/'engineering_checks_final.json')['profiling']
r['postfix_checks']=load(base/'additional_checks_and_provenance.json')
for name,path in [('resume_regression',base/'resume_regression.json'),('final_mechanism_diagnostic',dirs['quantum']/'mechanism_diagnostic.json')]:
    if path.exists():r[name]=load(path)
r['artifact_sha256']={}
for d in dirs.values():
    for filename in ['config.json','summary.json','training.json','best_validation_rows.json']:
        p=d/filename;r['artifact_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
out=base/'final_round2_results_20260919.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out)
print(json.dumps({'final_runs':summaries,'strata':r['strata']},indent=2),flush=True)
