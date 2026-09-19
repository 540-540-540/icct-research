from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from prediction.qgnn_paper_native.raj_joint import fixed_weight_subsets
from prediction.qgnn_paper_native.raj_paper_math import ControlledFeatureLoader, EquivariantSubsetAdjacency, ConditionalRDMReadout, RajPaperMathQGNNCore


def permute_subset(x, pi, n, j):
    subs=fixed_weight_subsets(n,j); lookup={s:i for i,s in enumerate(subs)}
    out=torch.zeros_like(x)
    for old,s in enumerate(subs):
        new=tuple(sorted(pi[q] for q in s)); out[:,lookup[new]]=x[:,old]
    return out


def permute_graph(edge,risk,mask,pi):
    B,n,_,d=edge.shape
    ep=torch.zeros_like(edge); rp=torch.zeros_like(risk); mp=torch.zeros_like(mask)
    for oi,ni in enumerate(pi):
        mp[:,ni]=mask[:,oi]
        for oj,nj in enumerate(pi):
            ep[:,ni,nj]=edge[:,oi,oj]; rp[:,ni,nj]=risk[:,oi,oj]
    return ep,rp,mp


def main():
    torch.manual_seed(123)
    report={'status':'RUNNING','test_set_used':False}

    # Loader gates.
    B,S,F,E=2,6,11,20
    feat=torch.randn(B,S,F,dtype=torch.float64)
    valid=torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0]],dtype=torch.bool)
    loader=ControlledFeatureLoader(F,E).double()
    state,cond=loader(feat,valid)
    loader_global_norm=float((state.flatten(1).norm(dim=1)-1).abs().max())
    cond_norm=cond.norm(dim=-1)
    loader_cond_norm=float((cond_norm[valid]-1).abs().max())
    ref=loader.reference(cond,valid).to(torch.complex128)
    prepared=loader.reupload(ref,cond,valid)
    loader_prepare_err=float((prepared[valid]-cond.to(torch.complex128)[valid]).abs().max())
    rand=torch.complex(torch.randn(B,S,E,dtype=torch.float64),torch.randn(B,S,E,dtype=torch.float64))*valid[...,None]
    before=rand.norm(dim=-1); after=loader.reupload(rand,cond,valid).norm(dim=-1)
    loader_reupload_norm=float((before[valid]-after[valid]).abs().max())
    assert loader_global_norm<1e-10 and loader_cond_norm<1e-10 and loader_prepare_err<1e-10 and loader_reupload_norm<1e-10
    report['loader']={'global_norm_error':loader_global_norm,'conditional_norm_error':loader_cond_norm,'prepare_error':loader_prepare_err,'reupload_norm_error':loader_reupload_norm}

    # Adjacency norm + permutation equivariance.
    n,j=4,2; S2=len(fixed_weight_subsets(n,j)); Dm=5
    adj=EquivariantSubsetAdjacency(n,j).double()
    x=torch.complex(torch.randn(1,S2,Dm,dtype=torch.float64),torch.randn(1,S2,Dm,dtype=torch.float64)); x=x/x.norm()
    risk=torch.tensor([[[0,.91,.72,.53],[.91,0,.84,.31],[.72,.84,0,.65],[.53,.31,.65,0]]],dtype=torch.float64)
    edge=torch.zeros(1,n,n,16,dtype=torch.float64)
    for a in range(n):
        for b in range(n):
            if a!=b:
                edge[0,a,b,0]=risk[0,a,b]; edge[0,a,b,1]=(a+b+1)*.07; edge[0,a,b,2]=risk[0,a,b]**2
    # Make feature tensor permutation-equivariant: symmetrise non-risk channels by node-invariant risk functions.
    edge[...,1]=risk*0.37; edge[...,2]=risk.square()
    mask=torch.ones(1,n,dtype=torch.bool)
    y=adj(x,edge,risk,mask)
    adj_norm_err=float((y.norm()-x.norm()).abs())
    pi=[2,0,3,1]
    xp=permute_subset(x,pi,n,j); ep,rp,mp=permute_graph(edge,risk,mask,pi)
    yp=adj(xp,ep,rp,mp); expected=permute_subset(y,pi,n,j)
    adj_equiv_err=float((yp-expected).abs().max())
    assert adj_norm_err<1e-10 and adj_equiv_err<1e-10
    report['adjacency']={'norm_error':adj_norm_err,'permutation_equivariance_error':adj_equiv_err}

    # Conditional 1-RDM trace = k.
    n3,j3,D,k=8,3,6,3; S3=len(fixed_weight_subsets(n3,j3)); E3=20
    readout=ConditionalRDMReadout(n3,j3,D,k).double()
    c=torch.complex(torch.randn(1,S3,E3,dtype=torch.float64),torch.randn(1,S3,E3,dtype=torch.float64)); c=c/c.norm(dim=-1,keepdim=True)
    gamma=readout.subset_rdm(c)
    trace=gamma.diagonal(dim1=-2,dim2=-1).real.sum(-1)
    rdm_trace_err=float((trace-k).abs().max())
    assert rdm_trace_err<1e-10
    report['rdm']={'trace_error':rdm_trace_err}

    # Full production-size forward/backward and masking.
    core=RajPaperMathQGNNCore().float()
    hist=torch.randn(1,20,8,4); hist[...,0:2]*=12.; hist[...,2:4]*=4.
    m=torch.tensor([[1,1,1,1,1,0,0,0]],dtype=torch.bool)
    hist=torch.where(m[:,None,:,None],hist,0.)
    z=core(hist,m)
    assert z.shape==(1,8,64) and torch.isfinite(z).all() and float(z[:,5:].abs().max())==0.0
    loss=z[:,:5].square().mean(); loss.backward()
    groups={
      'loader':sum(float(p.grad.norm()) for mod in core.loaders for p in mod.parameters() if p.grad is not None),
      'adjacency':sum(float(p.grad.norm()) for mod in core.adj for p in mod.parameters() if p.grad is not None),
      'W':sum(float(p.grad.norm()) for mod in core.evolution for p in mod.parameters() if p.grad is not None),
      'M':sum(float(p.grad.norm()) for p in core.joint.parameters() if p.grad is not None),
      'readout':sum(float(p.grad.norm()) for p in core.readout.parameters() if p.grad is not None),
    }
    assert all(torch.isfinite(torch.tensor(v)) and v>1e-12 for v in groups.values())
    report['full_core']={'shape':list(z.shape),'max_abs':float(z.abs().max()),'masked_max_abs':float(z[:,5:].abs().max()),'gradient_norm_sums':groups}
    report['status']='PASS'
    out=Path('reports/qgnn/round4_raj_paper_math_unit_20260919.json'); out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
