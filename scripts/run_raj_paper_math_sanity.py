from __future__ import annotations
import hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_paper_native.model import build_paper_model
from prediction.q0.metrics import trajectory_metrics
from scripts.train_q0_motion_llm import token_loss

OUT=ROOT/'reports/qgnn/round4_raj_paper_math_sanity_20260919'
OUT.mkdir(parents=True,exist_ok=True)

def atomic(name,payload):
    p=OUT/name; t=p.with_suffix('.tmp'); t.write_text(json.dumps(payload,indent=2)+'\n'); t.replace(p)

def main():
    seed=2026
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds=SinDPredictionDataset('train',0.,ROOT,True)
    selected=np.random.default_rng(seed).permutation(len(ds))[:32]
    model=build_paper_model('raj_paper_math_quantum',seed=seed,rounds=3,j=3,correction_cap_m=16.).to(device)
    groups=[
      {'params':[v for n,v in model.named_parameters() if v.requires_grad and 'lora_' not in n],'lr':3e-4},
      {'params':[v for n,v in model.named_parameters() if v.requires_grad and 'lora_' in n],'lr':7.5e-5},
    ]
    opt=torch.optim.AdamW(groups,weight_decay=2e-4)
    config={'seed':seed,'snr_db':0,'selected_indices':selected.tolist(),'selected_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
            'batch_size':4,'steps':32,'kind':'raj_paper_math_quantum','j':3,'D':6,'k':3,'rounds':3,'lr':3e-4,'lora_lr':7.5e-5,
            'weight_decay':2e-4,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'test_set_used':False}
    atomic('config.json',config)
    records=[]; started=time.perf_counter(); global_step=0
    torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
    for epoch in range(1,5):
        order=torch.randperm(32,generator=torch.Generator().manual_seed(seed+1009*epoch)).tolist()
        loader=DataLoader(Subset(ds,selected[order].tolist()),batch_size=4,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
        for batch in loader:
            global_step+=1; tick=time.perf_counter()
            h=batch['history_state'].to(device);m=batch['vehicle_mask'].to(device);y=batch['future_state'].to(device);ts=batch['history_timestamp'].to(device)
            opt.zero_grad(set_to_none=True); out=model(h,m,ts)
            metric=trajectory_metrics(out['prediction'],y,m)
            tok=token_loss(out['token_logits'],model.llm.future_token_ids(h,y),m)
            loss=metric['loss']+.035*tok
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            graph_sq=0.
            for p in model.graph.parameters():
                if p.grad is not None: graph_sq += float(p.grad.detach().square().sum())
            graph_grad=graph_sq**0.5
            total_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),3.))
            if not np.isfinite(graph_grad) or graph_grad<=0 or not np.isfinite(total_grad): raise FloatingPointError('invalid gradient')
            opt.step()
            if device.type=='cuda': torch.cuda.synchronize()
            rec={'step':global_step,'epoch':epoch,'loss':float(loss.detach()),'trajectory_loss':float(metric['loss'].detach()),
                 'token_loss':float(tok.detach()),'ADE':float(metric['ADE'].detach()),'FDE':float(metric['FDE'].detach()),
                 'graph_grad_norm':graph_grad,'total_grad_norm_preclip':total_grad,'step_seconds':time.perf_counter()-tick}
            records.append(rec)
            atomic('heartbeat.json',rec|{'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started})
            print('STEP',json.dumps(rec),flush=True)
    first=float(np.mean([r['loss'] for r in records[:8]])); last=float(np.mean([r['loss'] for r in records[-8:]]))
    improvement=(first-last)/first*100.
    passed=all(np.isfinite(r['loss']) and np.isfinite(r['graph_grad_norm']) and r['graph_grad_norm']>0 for r in records) and improvement>=10.
    summary={'status':'PASS' if passed else 'FAIL','first8_mean_loss':first,'last8_mean_loss':last,'loss_improvement_pct':improvement,
             'min_graph_grad_norm':min(r['graph_grad_norm'] for r in records),'max_graph_grad_norm':max(r['graph_grad_norm'] for r in records),
             'elapsed_seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
             'steps':len(records),'test_set_used':False}
    atomic('training.json',records); atomic('summary.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)
    if not passed: raise SystemExit(2)

if __name__=='__main__':
    main()
