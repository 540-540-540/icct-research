"""History-only representation bottleneck audit for frozen Round2 RC-HQGNN.
Teacher target is the frozen strong classical core's interaction residual, never future GT.
Test split is never constructed.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_final.model import build_model
from prediction.qgnn_final.common import physical_graph
from prediction.qgnn_final.quantum import evolve,moments,tables
from prediction.qgnn_final.relational import cumulants,rooted_to_global

QCK=ROOT/'reports/qgnn/r2_final_full_quantum_0db_seed2026/best.pt'
CCK=ROOT/'reports/qgnn/r2_final_full_classical_0db_seed2026/best.pt'
OUT=ROOT/'reports/qgnn/round4_representation_bottleneck_audit_20260919.json'

def load_model(kind,ck,device):
    m=build_model(kind,2026,3,channels=4,quantum_version=3,correction_cap_m=16.).to(device)
    p=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(p['model_state'],strict=False);m.eval();return m

def gather_raw(own,edge,risk,mask,tw_global,tf_global,tri):
    b,n,_=own.shape;rows=[]
    # History-only risk-sorted pair and rooted-triplet descriptors.
    pair_all=[];tri_all=[]
    rootmap=rooted_to_global(n,own.device)
    for i in range(n):
        valid=mask.clone();valid[:,i]=False
        score=risk[:,i].masked_fill(~valid,-1e9)
        idx=torch.argsort(score,dim=1,descending=True,stable=True)[:,:n-1]
        ei=edge[:,i].gather(1,idx[...,None].expand(-1,-1,edge.shape[-1]))
        ri=risk[:,i].gather(1,idx)[...,None]
        pair_all.append(torch.cat((ei,ri),-1).flatten(1))
        gi=rootmap[i][None].expand(b,-1)
        tf=tf_global.gather(1,gi[...,None].expand(-1,-1,tf_global.shape[-1]))
        tw=tw_global.gather(1,gi)
        order=torch.argsort(tw,dim=1,descending=True,stable=True)
        tf=tf.gather(1,order[...,None].expand(-1,-1,tf.shape[-1]));tw=tw.gather(1,order)[...,None]
        tri_all.append(torch.cat((tf,tw),-1).flatten(1))
    return torch.cat((own,torch.stack(pair_all,1),torch.stack(tri_all,1)),-1)*mask[...,None]

def representations(qcore,h,m):
    own,erisk,etw,ry,rz,pa,ta,rx,em=qcore.circuit_inputs(h,m)
    b,n,_=own.shape;ch=qcore.channels;d=qcore.depth
    edge,risk=physical_graph(h,m);_,_,pairs,tri,_,_=tables(n,h.device,h.dtype)
    # Rebuild symmetric triple descriptor exactly as circuit compiler uses it.
    a,c,k=tri.unbind(-1)
    es=torch.stack((edge[:,a,c],edge[:,a,k],edge[:,c,k]),-2)
    tf=torch.cat((es.mean(-2),es.var(-2,unbiased=False).add(1e-6).sqrt()),-1)
    raw=gather_raw(own,edge,risk,m,etw.reshape(b,ch,-1)[:,0],tf,tri)
    # Angle representation: own + local node angles + all incident pair/triple phase controls.
    ry0=ry.reshape(b,ch,d,n).permute(0,3,1,2);rz0=rz.reshape(b,ch,d,n).permute(0,3,1,2)
    pmap=torch.full((n,n),-1,device=h.device,dtype=torch.long)
    pmap[pairs[:,0],pairs[:,1]]=torch.arange(len(pairs),device=h.device);pmap[pairs[:,1],pairs[:,0]]=torch.arange(len(pairs),device=h.device)
    pidx=torch.stack([pmap[i,torch.arange(n,device=h.device)!=i] for i in range(n)])
    pa0=pa.reshape(b,ch,d,-1)
    pg=torch.stack([pa0[...,pidx[i]] for i in range(n)],1).flatten(2)
    tmap=rooted_to_global(n,h.device);ta0=ta.reshape(b,ch,d,-1)
    tg=torch.stack([ta0[...,tmap[i]] for i in range(n)],1).flatten(2)
    angles=torch.cat((own,ry0.flatten(2),rz0.flatten(2),pg,tg),-1)*m[...,None]
    states=evolve(ry,rz,pa,ta,rx,em)
    base=qcore.readout_features(states,em,erisk,etw,b)
    relation=qcore.relational_features(states,own,h,m)
    current=torch.cat((own,base,relation),-1)*m[...,None]
    gate=torch.sigmoid(qcore.gate(current))
    compressed=(gate*qcore.readout(torch.cat((base,relation),-1)))*m[...,None]
    local64=qcore.local(own)*m[...,None]
    final64=(local64+compressed)*m[...,None]
    # Rich identity-preserving quantum observables before learned readout compression.
    rich_layers=[]
    rmap=rooted_to_global(n,h.device)
    for st in states:
        mom=moments(st,em,erisk,etw)[...,:3].reshape(b,ch,n,3).permute(0,2,1,3)
        c2,c3=cumulants(st,n);c2=c2.reshape(b,ch,n,n).permute(0,2,1,3)
        c3=c3.reshape(b,ch,-1)
        c3r=torch.stack([c3[...,rmap[i]] for i in range(n)],1)
        rich_layers.append(torch.cat((mom.flatten(2),c2.flatten(2),c3r.flatten(2)),-1))
    rich=torch.cat((own,*rich_layers),-1)*m[...,None]
    return {'raw':raw,'angles':angles,'rich_quantum':rich,'current_readout_input':current,'compressed_interaction64':compressed,'local_only64':local64,'final_graph64':final64}

@torch.no_grad()
def extract(indices,split,qmodel,cmodel,device,batch=32):
    ds=SinDPredictionDataset(split,0.,ROOT,True);sub=Subset(ds,indices) if indices is not None else ds
    dl=DataLoader(sub,batch_size=batch,shuffle=False,num_workers=0)
    buckets={};targets=[]
    for bb in dl:
        h=bb['history_state'].to(device);m=bb['vehicle_mask'].to(device).bool()
        rep=representations(qmodel.graph,h,m)
        ownc=cmodel.graph.encoder(h,m);full=cmodel.graph(h,m);local=cmodel.graph.local(ownc);target=(full-local)*m[...,None]
        valid=m.reshape(-1);targets.append(target.reshape(-1,64)[valid].cpu())
        for k,v in rep.items():buckets.setdefault(k,[]).append(v.reshape(-1,v.shape[-1])[valid].cpu())
    return {k:torch.cat(v).numpy().astype(np.float32) for k,v in buckets.items()},torch.cat(targets).numpy().astype(np.float32)

class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,64))
    def forward(self,x):return self.net(x)

def fit_probe(xtr,ytr,xv,yv,device):
    xt=torch.from_numpy(xtr).to(device);yt=torch.from_numpy(ytr).to(device);xval=torch.from_numpy(xv).to(device);yval=torch.from_numpy(yv).to(device)
    model=Probe().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    g=torch.Generator(device='cpu').manual_seed(2026)
    best=1e9
    for ep in range(35):
        order=torch.randperm(len(xt),generator=g)
        model.train()
        for st in range(0,len(xt),512):
            ids=order[st:st+512].to(device);pred=model(xt[ids]);loss=(pred-yt[ids]).square().mean();opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): mse=float((model(xval)-yval).square().mean())
        best=min(best,mse)
    var=float(np.mean((yv-ytr.mean(0,keepdims=True))**2));return best,1-best/max(var,1e-12)

def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');q=load_model('quantum',QCK,device);c=load_model('classical',CCK,device)
    sel=np.random.default_rng(2026).permutation(15802)[:4096].tolist()
    tr,yt=extract(sel,'train',q,c,device);va,yv=extract(None,'val',q,c,device)
    ys=StandardScaler().fit(yt);ytr=ys.transform(yt).astype(np.float32);yv0=ys.transform(yv).astype(np.float32)
    report={'scope':{'train_scenes':4096,'val_scenes':1880,'snr_db':0,'test_set_used':False,'target':'frozen strong-classical graph interaction residual','common_probe_dim':128},'representations':{}}
    for name in tr:
        sc=StandardScaler().fit(tr[name]);a=sc.transform(tr[name]);b=sc.transform(va[name])
        pca=PCA(n_components=min(128,a.shape[1]),svd_solver='randomized',random_state=2026);a=pca.fit_transform(a);b=pca.transform(b)
        if a.shape[1]<128:
            a=np.pad(a,((0,0),(0,128-a.shape[1])));b=np.pad(b,((0,0),(0,128-b.shape[1])))
        mse,r2=fit_probe(a.astype(np.float32),ytr,b.astype(np.float32),yv0,device)
        report['representations'][name]={'raw_dim':int(tr[name].shape[1]),'pca_explained':float(pca.explained_variance_ratio_.sum()),'standardized_val_mse':mse,'val_r2':r2}
        print(name,report['representations'][name],flush=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
