"""Reproducible synthetic correctness preflight for the two QGNN finalists and matched controls."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_finalists.trc import TRCQuantumCore,RTCNCore
from prediction.qgnn_finalists.toj import TOJQGNNCore,TRTGNCore

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',default='reports/qgnn/finalists_preflight/core_check_latest.json');args=p.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() else 'cpu');torch.manual_seed(97)
    h=torch.randn(1,20,8,4,device=dev);h[...,2:]*=.4;m=torch.ones(1,8,dtype=torch.bool,device=dev);ts=torch.arange(20,device=dev,dtype=torch.float64)[None]*.1001001
    perm=torch.tensor([3,0,7,2,6,1,5,4],device=dev);rows={}
    for name,ctor in [('trc',TRCQuantumCore),('rtcn',RTCNCore),('toj',TOJQGNNCore),('trtgn',TRTGNCore)]:
        model=ctor().to(dev).train();
        if dev.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        tick=time.perf_counter();out=model(h,m,ts);loss=out['readout'].square().mean();loss.backward()
        if dev.type=='cuda':torch.cuda.synchronize()
        grads=[x.grad for x in model.parameters() if x.requires_grad];nz=sum(int(g is not None and torch.isfinite(g).all() and float(g.abs().sum())>0) for g in grads)
        model.eval()
        with torch.no_grad():
            a=model(h,m,ts)['readout'];b=model(h[:,:,perm],m[:,perm],ts)['readout'];equiv=float((b-a[:,perm]).abs().max())
            mp=m.clone();mp[:,7]=False;hp=h.clone();hp[:,:,7]=0;c=model(hp,mp,ts)['readout'];hp[:,:,7]=9999*torch.randn_like(hp[:,:,7]);d=model(hp,mp,ts)['readout'];pad=float((c[:,:7]-d[:,:7]).abs().max())
        rows[name]={'shape':list(out['readout'].shape),'finite':bool(torch.isfinite(out['readout']).all()),'trainable_grad_tensors':len(grads),'nonzero_finite_grad_tensors':nz,'permutation_max_abs':equiv,'padding_active_max_abs':pad,'seconds_including_backward_and_checks':time.perf_counter()-tick,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if dev.type=='cuda' else 0.0}
        del model,out,loss
    payload={'status':'PASS' if all(r['finite'] and r['nonzero_finite_grad_tensors']==r['trainable_grad_tensors'] and r['permutation_max_abs']<1e-5 and r['padding_active_max_abs']<1e-7 for r in rows.values()) else 'FAIL','test_set_used':False,'rows':rows}
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
    if payload['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
