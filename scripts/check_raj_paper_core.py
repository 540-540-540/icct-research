from __future__ import annotations
import sys,time,json,torch
from pathlib import Path
ROOT=Path('/home/dell/YrM/ICCT');sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_paper_native.raj_paper import RajPaperQGNNCore,RajPaperJohnsonGINCore,conditional_rdm

def params(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
@torch.no_grad()
def perm(model,h,m):
    n=h.shape[2];p=torch.randperm(n,device=h.device);inv=torch.argsort(p)
    a=model(h,m);b=model(h[:,:,p],m[:,p])[:,inv]
    return float((a-b).abs().max())

def run(cls,j,h,m,device):
    model=cls(j=j,depth=3).to(device)
    y=model(h,m);loss=y.square().mean();loss.backward()
    grad=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    pe=perm(model.eval(),h[:4],m[:4])
    model.train();times=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        model.zero_grad(set_to_none=True);t=time.perf_counter();z=model(h,m);z.square().mean().backward()
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
    return {'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'grad_finite':bool(grad),'perm_max_abs':pe,'params':params(model),'step_median_s':float(torch.tensor(times).median()),'peak_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0}

def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 1-RDM sanity: trace must equal k for normalized states.
    x=torch.randn(7,5,20,dtype=torch.complex64,device=device);x=x/(x.norm(dim=-1,keepdim=True)+1e-8)
    g=conditional_rdm(x)
    tr=g.diagonal(dim1=-2,dim2=-1).real.sum(-1)
    print('RDM_TRACE_MAX_ERR',float((tr-3).abs().max()))
    ds=SinDPredictionDataset('val',0.,ROOT,True);xs=[ds[i] for i in range(32)]
    h=torch.stack([x['history_state'] for x in xs]).to(device);m=torch.stack([x['vehicle_mask'] for x in xs]).to(device)
    out={}
    for j in (2,3):
        out[f'quantum_j{j}']=run(RajPaperQGNNCore,j,h,m,device)
        out[f'johnson_j{j}']=run(RajPaperJohnsonGINCore,j,h,m,device)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
