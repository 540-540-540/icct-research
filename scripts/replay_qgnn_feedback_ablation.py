"""Replay a trained feedback arm with its stored non-tensor feedback flag."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.model import build_model
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_feedback_ablation import evaluate, atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);directory=ROOT/a.run_dir;out=ROOT/a.output
    assert not out.exists(),out
    started=time.perf_counter();cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);c=cp['config']
    assert c['adaptive_mode']=='phase_feedback' and c['controller_init']=='neutral' and c['feedback_training'] in ['on','off']
    for path,digest in cp['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert hashlib.sha256((ROOT/'configs/qgnn_final_tokens.json').read_bytes()).hexdigest()==cp['token_sha256']
    model=build_model(c['kind'],c['seed'],c['depth'],channels=c['channels'],snr_db=c['snr'],correction_cap_m=c['correction_cap'],adaptive_mode=c['adaptive_mode'],controller_init=c['controller_init']).cuda().eval()
    model.graph.feedback_enabled=c['feedback_training']=='on'
    missing,extra=model.load_state_dict(cp['model_state'],strict=False)
    assert not extra and all(k.startswith('llm.gpt2.') and 'lora_' not in k for k in missing)
    feedback_norms={k:float(v.norm()) for k,v in model.graph.controller.state_dict().items() if k in ['feedback_local','feedback_global.weight']}
    if c['feedback_training']=='off':assert all(v==0 for v in feedback_norms.values())
    else:assert all(v>0 for v in feedback_norms.values())
    report={'status':'RUNNING','kind':c['kind'],'feedback_training':c['feedback_training'],'test_set_used':False,'new_training':False,'feedback_enabled_restored_from_config':model.graph.feedback_enabled,'feedback_parameter_norms':feedback_norms}
    atomic_json(out,report)
    ds=SinDPredictionDataset('val',0.,ROOT,True)
    metric,rows=evaluate(model,DataLoader(ds,batch_size=32,shuffle=False),torch.device('cuda:0'))
    old=json.loads((directory/'best_validation_rows.json').read_text())['rows']
    assert len(rows)==len(old)==1880
    assert all((r['index'],r['scene_id'],r['start_frame'])==(s['index'],s['scene_id'],s['start_frame']) for r,s in zip(rows,old))
    error=max(abs(r[k]-s[k]) for r,s in zip(rows,old) for k in ['ADE','FDE'])
    assert error<2e-5,error
    report.update(status='PASS',validation=metric,max_per_window_metric_error_m=error,elapsed_seconds=time.perf_counter()-started,checkpoint_epoch=cp['epoch'])
    atomic_json(out,report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
