"""Experimental Raj-style higher-order subset QGNN for SinD.
Paper-native inspiration: Raj et al. 2026 mp-qgnns.
This is a project adaptation, not a verbatim reproduction.
"""
from __future__ import annotations
import itertools
from functools import lru_cache
import torch
from torch import nn
from .common import BoundedCore, OwnHistoryEncoder, physical_graph

def _subsets(n: int, j: int, device):
    vals=list(itertools.combinations(range(n),j)) if n>=j else []
    return torch.tensor(vals,device=device,dtype=torch.long).reshape(-1,j)

@lru_cache(maxsize=32)
def _johnson_cpu(n: int, j: int):
    vals=list(itertools.combinations(range(n),j)) if n>=j else []
    s=len(vals)
    a=torch.zeros(s,s,dtype=torch.float64)
    for x in range(s):
        sx=set(vals[x])
        for y in range(x+1,s):
            if len(sx.intersection(vals[y]))==j-1:
                a[x,y]=a[y,x]=1.
    if s:
        w,v=torch.linalg.eigh(a)
    else:
        w=torch.empty(0,dtype=torch.float64);v=torch.empty(0,0,dtype=torch.float64)
    inc=torch.zeros(s,n,dtype=torch.float64)
    for si,t in enumerate(vals):
        for i in t: inc[si,i]=1.
    return w,v,inc

class SubsetFeatureBuilder(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.own=OwnHistoryEncoder(width)
        self.out_dim=2*width+2*16+2
    def forward(self,history,mask,j:int):
        own=self.own(history,mask);edge,risk=physical_graph(history,mask)
        b,n,w=own.shape;subs=_subsets(n,j,history.device)
        if subs.numel()==0:return own,own.new_zeros(b,0,self.out_dim),mask.new_zeros(b,0),subs
        nodes=own[:,subs]
        nmean=nodes.mean(-2);nstd=nodes.var(-2,unbiased=False).add(1e-6).sqrt()
        ep=[];rp=[]
        for a,c in itertools.combinations(range(j),2):
            ia,ic=subs[:,a],subs[:,c];ep.append(edge[:,ia,ic]);rp.append(risk[:,ia,ic])
        e=torch.stack(ep,-2);r=torch.stack(rp,-1)
        feat=torch.cat((nmean,nstd,e.mean(-2),e.var(-2,unbiased=False).add(1e-6).sqrt(),
                        r.mean(-1,keepdim=True),r.var(-1,unbiased=False,keepdim=True).add(1e-6).sqrt()),-1)
        valid=mask[:,subs].all(-1)
        return own,feat*valid[...,None],valid,subs

class _SubsetReadout(nn.Module):
    def __init__(self,own_width,embed_dim,rounds):
        super().__init__();dim=(rounds+1)*2*embed_dim
        self.local=nn.Sequential(nn.Linear(own_width,64),nn.SiLU(),nn.LayerNorm(64))
        self.inter=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,64))
        self.norm=nn.LayerNorm(64)
    def pool_nodes(self,zs,valid,subs,n,mask):
        _,_,inc0=_johnson_cpu(n,subs.shape[-1]);inc=inc0.to(zs[0].device,zs[0].real.dtype)
        den=torch.einsum('bs,sn->bn',valid.to(inc.dtype),inc).clamp_min(1.)
        pools=[]
        for z in zs:
            rr=torch.cat((z.real,z.imag),-1)*valid[...,None]
            pools.append(torch.einsum('bse,sn->bne',rr,inc)/den[...,None])
        return torch.cat(pools,-1)*mask[...,None]
    def forward_out(self,own,zs,valid,subs,mask):
        p=self.pool_nodes(zs,valid,subs,own.shape[1],mask)
        return self.norm(self.local(own)+self.inter(p))*mask[...,None]

class RajJohnsonGINCore(BoundedCore):
    def __init__(self,j=3,D=6,k=3,rounds=3,hidden=None,encoder_hidden=64):
        super().__init__();self.j=j;self.rounds=rounds
        import math
        self.hidden=hidden or math.comb(D,k);self.builder=SubsetFeatureBuilder(32);f=self.builder.out_dim
        self.input=nn.Sequential(nn.Linear(f,encoder_hidden),nn.SiLU(),nn.Linear(encoder_hidden,self.hidden))
        self.layers=nn.ModuleList(nn.Linear(2*self.hidden,self.hidden) for _ in range(rounds))
        self.readout=_SubsetReadout(32,self.hidden,rounds)
    @staticmethod
    def _unit(x):return x/(torch.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-8)
    def forward_patch(self,history,mask):
        own,feat,valid,subs=self.builder(history,mask,self.j)
        if subs.shape[0]==0:return self.readout.local(own)*mask[...,None]
        s=subs.shape[0];a=feat.new_zeros(s,s)
        for u in range(s):
            su=set(subs[u].tolist())
            for v in range(u+1,s):
                if len(su.intersection(subs[v].tolist()))==self.j-1:a[u,v]=a[v,u]=1
        h=self._unit(torch.tanh(self.input(feat)))*valid[...,None];hs=[h]
        for layer in self.layers:
            agg=torch.einsum('uv,bvh->buh',a,h)
            h=self._unit(torch.tanh(layer(torch.cat((h,agg),-1))))*valid[...,None];hs.append(h)
        zs=[torch.complex(hh,torch.zeros_like(hh)) for hh in hs]
        return self.readout.forward_out(own,zs,valid,subs,mask)

class MultiJFusion(nn.Module):
    """Per-agent fusion only; no new cross-agent classical message passing."""
    def __init__(self):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1))
        self.delta=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,64))
        self.norm=nn.LayerNorm(64)
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,z2,z3,mask):
        both=torch.cat((z2,z3),-1)
        g=torch.sigmoid(self.gate(both))
        base=(1-g)*z2+g*z3
        return self.norm(base+self.delta(both))*mask[...,None]

class RajMultiJJohnsonCore(BoundedCore):
    """Matched classical: same j=2+j=3 subset lifts and same per-agent fusion."""
    def __init__(self,rounds=3,hidden=64):
        super().__init__();self.j2=RajJohnsonGINCore(j=2,rounds=rounds,hidden=hidden);self.j3=RajJohnsonGINCore(j=3,rounds=rounds,hidden=hidden);self.fuse=MultiJFusion()
    def forward_patch(self,history,mask):
        return self.fuse(self.j2.forward_patch(history,mask),self.j3.forward_patch(history,mask),mask)
