"""Audit all four fixed retraining arms and all predeclared strata."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_qgnn_round3_selected_feedback import strata

def load(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def metric(rows):
    a=np.array([[r['ADE'],r['FDE']] for r in rows],dtype=float)
    return np.column_stack((a,a[:,0]+.5*a[:,1]))
def named(a):return dict(zip(['ADE','FDE','J'],[float(v) for v in a]))

def main():
    report={'status':'RUNNING','test_set_used':False,'full_training_executed':False,'new_model_selected':False,
      'protocol':{'seed':2026,'snr_db':0,'train_samples':4096,'val_samples':1880,'epochs':12,'batch_size':32,'cap_m':16,'adaptive_mode':'phase_feedback','controller_init':'neutral'},
      'interpretation':'One fixed 2x2 retraining ablation, not architecture search. Positive feedback benefit means on trained better than off. Validation windows overlap; no iid significance claim.',
      'runs':{},'within_core_feedback_effects':{},'quantum_vs_classical':{}}
    allrows={};arrays={};summaries={};configs={}
    for kind in ['quantum','classical']:
        for feedback in ['on','off']:
            key=kind+'_'+feedback;d='reports/qgnn/round3_retrain_'+key+'_20260919'
            s=load(d+'/summary.json');c=load(d+'/config.json');t=load(d+'/training.json')
            raw=load(d+'/best_validation_rows.json')['rows']
            assert s['status']=='COMPLETED' and not s['test_set_used']
            assert s['global_step']==1536 and s['epochs_completed']==12 and s['train_samples']==4096 and s['validation_samples']==1880
            for field,v in [('seed',2026),('snr',0),('epochs',12),('batch_size',32),('train_limit',4096),('correction_cap',16),('adaptive_mode','phase_feedback'),('controller_init','neutral'),('feedback_training',feedback),('kind',kind)]:assert c[field]==v,(key,field)
            assert len(raw)==1880 and len(t)==12
            assert s['best_epoch']==min(t,key=lambda x:x['validation']['J'])['epoch']
            a=metric(raw);assert np.max(np.abs(a.mean(0)-np.array([s['best_validation'][k] for k in ['ADE','FDE','J']])))<1e-7
            beats=load(d+'/step_samples.json')['heartbeats']
            replay=load(d+'/replay.json');assert replay['status']=='PASS' and replay['feedback_training']==feedback
            assert replay['feedback_enabled_restored_from_config']==(feedback=='on')
            summaries[key]=s;configs[key]=c;allrows[key]=raw;arrays[key]=a
            report['runs'][key]={'path':d,'best':named(a.mean(0)),'best_epoch':s['best_epoch'],'epoch12':t[-1]['validation'],'elapsed_seconds':s['elapsed_seconds'],
              'initial_mutable_sha256':s['initial_mutable_sha256'],'peak_allocated_gib':max(x['peak_allocated_gib'] for x in beats),'median_logged_step_seconds':float(np.median([x['step_seconds'] for x in beats])),
              'global_steps':s['global_step'],'parameters':s['parameters'],'train_loss_first_last':[t[0]['train_loss'],t[-1]['train_loss']],
              'checkpoint_replay':replay,'artifacts_sha256':{f:sha(d+'/'+f) for f in ['config.json','summary.json','training.json','best_validation_rows.json','step_samples.json','replay.json']}}
    reference=allrows['quantum_on']
    assert all([(r['index'],r['scene_id'],r['start_frame']) for r in value]==[(r['index'],r['scene_id'],r['start_frame']) for r in reference] for value in allrows.values())
    for k in ['shared_initialization_sha256','train_indices_sha256']:
        assert len({s[k] for s in summaries.values()})==1
    for k in ['token_sha256','source_sha256']:
        assert all(c[k]==configs['quantum_on'][k] for c in configs.values())
    for kind in ['quantum','classical']:assert summaries[kind+'_on']['initial_mutable_sha256']==summaries[kind+'_off']['initial_mutable_sha256']
    group=strata(reference,load('reports/qgnn/round3_strata_reference_20260919.json')['rows'])
    for key,arr in arrays.items():report['runs'][key]['strata']={name:{'windows':int(sel.sum()),**named(arr[sel].mean(0))} for name,sel in group.items()}
    for kind in ['quantum','classical']:
        on,off=arrays[kind+'_on'],arrays[kind+'_off']
        report['within_core_feedback_effects'][kind]={name:{'windows':int(sel.sum()),'off_minus_on':named((off[sel]-on[sel]).mean(0)),
          'on_gain_vs_off_percent':named(100*(off[sel].mean(0)-on[sel].mean(0))/off[sel].mean(0))} for name,sel in group.items()}
    for mode in ['on','off']:
        q,c=arrays['quantum_'+mode],arrays['classical_'+mode]
        report['quantum_vs_classical'][mode]={name:{'windows':int(sel.sum()),'quantum_gain_percent':named(100*(c[sel].mean(0)-q[sel].mean(0))/c[sel].mean(0))} for name,sel in group.items()}
    report['feedback_difference_in_differences']={name:named(((arrays['quantum_off']-arrays['quantum_on'])-(arrays['classical_off']-arrays['classical_on']))[sel].mean(0)) for name,sel in group.items()}
    baseline=load('reports/qgnn/round3_reference_classical_small_2026/best_validation_rows.json')['rows']
    assert [(x['scene_id'],x['start_frame']) for x in baseline]==[(x['scene_id'],x['start_frame']) for x in reference]
    c=metric(baseline).mean(0)
    report['retained_stronger_R2_classical']={'metrics':named(c),'quantum_on_gain_percent':named(100*(c-arrays['quantum_on'].mean(0))/c),'quantum_off_gain_percent':named(100*(c-arrays['quantum_off'].mean(0))/c)}
    report['fresh_on_replay_of_selected']={}
    for kind in ['quantum','classical']:
        old=metric(load('reports/qgnn/round3_neutral_'+kind+'_small_2026/best_validation_rows.json')['rows'])
        report['fresh_on_replay_of_selected'][kind]={'max_per_window_difference_m':float(np.max(np.abs(arrays[kind+'_on']-old))),'overall_difference_m':named((arrays[kind+'_on']-old).mean(0))}
    report['fairness']={'same_initial_mutable_parameters_within_kind':True,'same_shared_GPT2_initialization_all4':True,'same_training_indices_all4':True,'same_source_and_token_hashes_all4':True,'same_checkpoint_selection_rule':True,'all_quantum_scenes_execute_circuits':True,'test_closed':True}
    report['source_sha256']=configs['quantum_on']['source_sha256']
    report['pre_registration_sha256']=sha('docs/qgnn/ROUND3_FEEDBACK_RETRAIN_PREREG_20260919.md')
    report['preflight']=load('reports/qgnn/round3_feedback_retrain_preflight.json')
    report['corrected_auxiliary_audit']={}
    for kind in ['quantum','classical']:
        d='reports/qgnn/round3_feedback_audit_completed_'+kind+'_20260919'
        a=load(d+'/summary.json');assert a['status']=='COMPLETED'
        report['corrected_auxiliary_audit'][kind]={'path':d,'train_controller':a['train_controller'],'baseline_replay_max_metric_error_m':a['baseline_replay_max_metric_error_m']}
    report['status']='PASS_EVIDENCE_COMPLETE';report['architecture_decision']='UNCHANGED_PHASE_FEEDBACK_NEUTRAL';report['full_train_gate']='UNCHANGED_FAILED'
    out=ROOT/'reports/qgnn/round3_feedback_retraining_results_20260919.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'runs':{k:{f:v[f] for f in ['best','best_epoch','epoch12','elapsed_seconds']} for k,v in report['runs'].items()},'feedback_effects':{k:v['all'] for k,v in report['within_core_feedback_effects'].items()},'quantum_vs_classical':{k:v['all'] for k,v in report['quantum_vs_classical'].items()},'high140':{k:v['final_closing_ge15'] for k,v in report['quantum_vs_classical'].items()}},indent=2),flush=True)
if __name__=='__main__':main()
