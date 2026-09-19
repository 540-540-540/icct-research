"""Independent acceptance of the single neutral-initialization calibration.
No new architecture, training, parameter tuning, or test access.
"""
from __future__ import annotations
import hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_adaptive.model import build_model
from scripts.train_qgnn_adaptive import evaluate, atomic_json
OUT = ROOT/'reports/qgnn/round3_neutral_independent_acceptance.json'
report = {'status':'RUNNING','test_set_used':False,'runs':{},'checkpoint_replay':{},'scope':'Single paired neutral initialization of unchanged v2 phase_feedback; no training by this script.'}
def read(path):return json.loads((ROOT/path).read_text())
def save(stage):
    report['stage']=stage;atomic_json(OUT,report);print(stage,flush=True)
def main():
    torch.set_num_threads(4)
    configs={};summaries={};arrays={}
    aggregate=read('reports/qgnn/round3_neutral_init_summary.json')
    save('checking_complete_paired_protocol')
    for kind in ('quantum','classical'):
        directory='reports/qgnn/round3_neutral_'+kind+'_small_2026'
        c=read(directory+'/config.json');s=read(directory+'/summary.json')
        rows=read(directory+'/best_validation_rows.json')['rows']
        assert s['status']=='COMPLETED' and s['test_set_used'] is False
        assert (s['train_samples'],s['validation_samples'],s['epochs_completed'],s['global_step'])==(4096,1880,12,1536)
        for k,v in {'controller_init':'neutral','adaptive_mode':'phase_feedback','seed':2026,'snr':0.,'correction_cap':16.,'channels':4,'depth':3,'batch_size':32}.items():assert c[k]==v
        a=np.array([[r['ADE'],r['FDE'],r['ADE']+.5*r['FDE']] for r in rows])
        assert np.isfinite(a).all() and a.shape==(1880,3)
        assert max(abs(a[:,i].mean()-s['best_validation'][k]) for i,k in enumerate(('ADE','FDE','J')))<1e-12
        configs[kind],summaries[kind],arrays[kind]=c,s,a
        report['runs'][kind]={'directory':directory,'summary':s}
    assert configs['quantum']['source_sha256']==configs['classical']['source_sha256']
    for k in ('shared_initialization_sha256','train_indices_sha256'):assert summaries['quantum'][k]==summaries['classical'][k]
    assert configs['quantum']['token_sha256']==configs['classical']['token_sha256']
    assert not aggregate['full_train_gate']['PASS']
    report['full_train_gate']=aggregate['full_train_gate']
    report['strata']=aggregate['strata']
    ref=aggregate['strata']['all']['metrics']
    report['against_unmodified_classical_gain_pct']={k:100*(1-ref['quantum'][k]/ref['frozen_classical'][k]) for k in ('ADE','FDE','J')}
    report['classical_adaptation_gain_pct']={k:100*(1-ref['classical'][k]/ref['frozen_classical'][k]) for k in ('ADE','FDE','J')}
    save('raw_pair_verified_and_original_classical_retained')
    val=SinDPredictionDataset('val',0.,ROOT,True);loader=DataLoader(val,batch_size=32,shuffle=False)
    for kind in ('quantum','classical'):
        directory=report['runs'][kind]['directory'];c=configs[kind]
        cp=torch.load(ROOT/directory/'best.pt',map_location='cpu',weights_only=False)
        model=build_model(kind,seed=c['seed'],depth=c['depth'],channels=c['channels'],correction_cap_m=c['correction_cap'],adaptive_mode=c['adaptive_mode'],controller_init=c['controller_init']).cuda()
        missing,extra=model.load_state_dict(cp['model_state'],strict=False)
        assert not extra and all(k.startswith('llm.gpt2.') and 'lora_' not in k for k in missing)
        assert model.graph.controller.enabled and model.graph.feedback_enabled
        start=time.perf_counter();actual,rows=evaluate(model,loader,torch.device('cuda:0'))
        expected=read(directory+'/best_validation_rows.json')['rows']
        diff={k:abs(actual[k]-cp['validation'][k]) for k in ('ADE','FDE','J')}
        row_error=max(abs(a[k]-b[k]) for a,b in zip(rows,expected) for k in ('ADE','FDE'))
        assert max(diff.values())<1e-6 and row_error<2e-5
        report['checkpoint_replay'][kind]={'metrics':actual,'difference':diff,'per_window_max_error':row_error,'seconds':time.perf_counter()-start,'controller_enabled':True,'feedback_enabled':True}
        save(kind+'_1880_window_replay_pass')
        del model;torch.cuda.empty_cache()
    report['status']='PASS'
    report['performance_status']='FAILED_PREDECLARED_FULL_TRAIN_GATE'
    report['limitations']=['Validation-only single pilot seed; dependent overlapping windows.','Gap reduction partly reflects worse new-classical results; compare against original classical too.','Negative gate does not prove all adaptive quantum architectures ineffective.','No full training or five-SNR/multi-seed sweep authorized.']
    save('ACCEPTANCE_PASS_PERFORMANCE_GATE_FAILED')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());atomic_json(OUT,report);raise
