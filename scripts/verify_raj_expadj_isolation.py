"""Unit/smoke checks for the preregistered Raj A(G) operator isolation."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from prediction.qgnn_paper_native.raj_mechanism import HamiltonianExpSubsetAdjacency
from prediction.qgnn_paper_native.raj_paper import subset_tensors
from prediction.qgnn_paper_native.model import build_paper_model


def permute_subset(x, pi, n, j):
    subs,_,_=subset_tensors(n,j)
    lookup={tuple(s.tolist()):i for i,s in enumerate(subs)}
    out=torch.zeros_like(x)
    for old,s in enumerate(subs):
        ns=tuple(sorted(pi[q] for q in s.tolist()))
        out[:,lookup[ns]]=x[:,old]
    return out


def permute_nodes(t, pi):
    out=torch.zeros_like(t)
    for old_i,new_i in enumerate(pi):
        for old_j,new_j in enumerate(pi):
            out[:,new_i,new_j]=t[:,old_i,old_j]
    return out


def main():
    torch.manual_seed(123)
    n,j,E=5,2,4
    S=len(subset_tensors(n,j)[0])
    layer=HamiltonianExpSubsetAdjacency(n,j).double()
    xr=torch.randn(2,S,E,dtype=torch.float64)
    xi=torch.randn(2,S,E,dtype=torch.float64)
    x=torch.complex(xr,xi)
    x=x/x.flatten(1).norm(dim=1)[:,None,None]
    base=torch.randn(2,n,n,16,dtype=torch.float64)
    edge=(base+base.transpose(1,2))/2
    mask=torch.ones(2,n,dtype=torch.bool)
    risk=torch.ones(2,n,n,dtype=torch.float64)

    with torch.no_grad():
        for p in layer.angle.parameters(): p.zero_()
    y0=layer(x,edge,risk,mask)
    identity_err=float((y0-x).abs().max())
    assert identity_err < 1e-10

    torch.manual_seed(124)
    layer=HamiltonianExpSubsetAdjacency(n,j).double()
    y=layer(x,edge,risk,mask)
    norm_err=float((y.flatten(1).norm(dim=1)-x.flatten(1).norm(dim=1)).abs().max())
    assert norm_err < 1e-10

    pi=[2,4,0,3,1]
    xp=permute_subset(x,pi,n,j)
    ep=permute_nodes(edge,pi)
    rp=permute_nodes(risk,pi)
    mp=torch.zeros_like(mask)
    for old,new in enumerate(pi): mp[:,new]=mask[:,old]
    yp=layer(xp,ep,rp,mp)
    expected=permute_subset(y,pi,n,j)
    equiv_err=float((yp-expected).abs().max())
    assert equiv_err < 1e-10

    layer.zero_grad(set_to_none=True)
    probe=torch.randn_like(y.real)
    loss=(layer(x,edge,risk,mask).real*probe).sum()
    loss.backward()
    grad=sum(float(p.grad.norm()) for p in layer.angle.parameters() if p.grad is not None)
    assert grad > 1e-10

    torch.manual_seed(2026)
    model=build_paper_model('raj_paper_math_expadj_quantum',seed=2026,rounds=3,j=3,correction_cap_m=16.)
    graph_params=sum(p.numel() for p in model.graph.parameters() if p.requires_grad)
    assert graph_params == 56763, graph_params

    B,T,N=2,20,8
    history=torch.randn(B,T,N,4)
    history[...,0:2]*=8.
    history[...,2:4]*=2.
    mask2=torch.ones(B,N,dtype=torch.bool)
    mask2[1,-2:]=False
    history=torch.where(mask2[:,None,:,None],history,0.)
    z=model.graph(history,mask2)
    assert z.shape==(B,N,64)
    assert torch.isfinite(z).all()
    assert float(z[1,-2:].abs().max())==0.0
    model.graph.zero_grad(set_to_none=True)
    probe2=torch.randn_like(z)
    (z*probe2).sum().backward()
    gnorm=sum(float(p.grad.norm()) for p in model.graph.parameters() if p.grad is not None)
    assert gnorm > 1e-10

    payload={
      'status':'PASS',
      'identity_max_abs_error':identity_err,
      'norm_max_abs_error':norm_err,
      'permutation_equivariance_max_abs_error':equiv_err,
      'adjacency_gradient_norm_sum':grad,
      'full_graph_gradient_norm_sum':gnorm,
      'graph_params':graph_params,
      'graph_shape':list(z.shape),
      'masked_zero_max':float(z[1,-2:].abs().max()),
      'test_used':False,
    }
    out=ROOT/'reports/qgnn/round4_raj_expadj_unit_smoke_20260919.json'
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':
    main()

