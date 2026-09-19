from __future__ import annotations
import sys,time,json,torch
from pathlib import Path
ROOT=Path('/home/dell/YrM/ICCT');sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_paper_native import RajSubsetQGNNCore,RajJohnsonGINCore

@torch.no_grad()
def perm_error(model,h,m):
    n=h.shape[2];p=torch.randperm(n,device=h.device);inv=torch.argsort(p)
    a=model(h,m);b=model(h[:,:,p],m[:,p])[:,inv]
    return float((a-b).abs().max())

def nparam(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds=SinDPredictionDataset('val',0.,ROOT,True)
    batch=[ds[i] for i in range(32)]
    h=torch.stack([x['history_state'] for x in batch]).to(dev)
    m=torch.stack([x['vehicle_mask'] for x in batch]).to(dev)
    out={}
    for j in (2,3):
      for kind,cls in [('quantum',RajSubsetQGNNCore),('johnson_gin',RajJohnsonGINCore)]:
        model=cls(j=j,rounds=3).to(dev)
        model.train()
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats()
        y=model(h,m);loss=y.square().mean();loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite=all(torch.isfinite(g).all() for g in grads)
        model.eval();pe=perm_error(model,h[:4],m[:4])
        model.train();times=[]
        for _ in range(6):
          model.zero_grad(set_to_none=True);t=time.perf_counter();z=model(h,m);z.square().mean().backward()
          if dev.type=='cuda':torch.cuda.synchronize()
          times.append(time.perf_counter()-t)
        out[f'{kind}_j{j}']={
          'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'grad_finite':bool(finite),
          'params':nparam(model),'perm_max_abs':pe,'step_median_s':float(torch.tensor(times).median()),
          'peak_gib':torch.cuda.max_memory_allocated()/2**30 if dev.type=='cuda' else 0}
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
