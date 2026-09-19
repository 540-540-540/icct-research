"""Raj-style Fidelity-A SinD adaptation.

Faithful public-paper blocks:
- j-subset node register
- graph-conditioned number-preserving node mixing
- fixed-particle-number embedding register D=6,k=3
- trainable compound/RBS evolution
- layerwise data re-upload
- 1-RDM-style embedding readout

Not yet included:
- final cross-register joint mixer M(theta_M)

This is therefore a faithful intermediate adaptation, not a verbatim full-paper reproduction.
"""
from __future__ import annotations
import itertools, math
from functools import lru_cache
import torch
from torch import nn
from prediction.qgnn_final.common import BoundedCore, OwnHistoryEncoder, physical_graph

def subsets(n,j,device):
    return torch.tensor(list(itertools.combinations(range(n),j)),device=device,dtype=torch.long).reshape(-1,j)

@lru_cache(maxsize=32)
def subset_inc_cpu(n,j):
    ss=list(itertools.combinations(range(n),j)); inc=torch.zeros(len(ss),n,dtype=torch.float64)
    for sidx,s in enumerate(ss):
        for i in s: inc[sidx,i]=1
    return inc

def compound_batched(U: torch.Tensor, k: int) -> torch.Tensor:
    B,D,_=U.shape
    ss=torch.tensor(list(itertools.combinations(range(D),k)),device=U.device,dtype=torch.long)
    rows=U[:,ss]
    blocks=rows[:,:,:,ss].permute(0,1,3,2,4)
    return torch.linalg.det(blocks)

class CompoundLayer(nn.Module):
    def __init__(self,D=6,k=3,init_std=.1):
        super().__init__();self.D=D;self.k=k
        self.theta=nn.Parameter(torch.randn(D*(D-1)//2)*init_std)
        ij=torch.triu_indices(D,D,1);self.register_buffer('ri',ij[0]);self.register_buffer('ci',ij[1]);self.register_buffer('eye',torch.eye(D))
    def orthogonal(self):
        A=torch.zeros(self.D,self.D,device=self.theta.device,dtype=self.theta.dtype)
        A[self.ri,self.ci]=self.theta;A=A-A.T;I=self.eye.to(A)
        return torch.linalg.solve(I+A,I-A)
    def forward(self,x):
        C=compound_batched(self.orthogonal()[None],self.k)[0].to(x.dtype)
        return torch.einsum('...e,fe->...f',x,C)

class SubsetFeatures(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.own=OwnHistoryEncoder(width);self.out_dim=2*width+2*16+2
    def forward(self,hist,mask,j):
        own=self.own(hist,mask);edge,risk=physical_graph(hist,mask);b,n,_=own.shape;ss=subsets(n,j,hist.device)
        nv=own[:,ss];nmean=nv.mean(-2);nstd=nv.var(-2,unbiased=False).add(1e-6).sqrt()
        ep=[];rp=[]
        for a,bj in itertools.combinations(range(j),2):
            ia,ib=ss[:,a],ss[:,bj];ep.append(edge[:,ia,ib]);rp.append(risk[:,ia,ib])
        e=torch.stack(ep,-2);r=torch.stack(rp,-1)
        feat=torch.cat((nmean,nstd,e.mean(-2),e.var(-2,unbiased=False).add(1e-6).sqrt(),
                        r.mean(-1,keepdim=True),r.var(-1,unbiased=False,keepdim=True).add(1e-6).sqrt()),-1)
        valid=mask[:,ss].all(-1)
        return own,feat*valid[...,None],valid,ss,edge,risk

def lifted_weighted_hamiltonian(risk,ss):
    b,n,_=risk.shape;s=ss.shape[0];H=risk.new_zeros(b,s,s)
    sets=[set(x.tolist()) for x in ss]
    for u in range(s):
        Su=sets[u]
        for v in range(u+1,s):
            Sv=sets[v]
            rem=list(Su-Sv);add=list(Sv-Su)
            if len(rem)==len(add)==1:
                w=risk[:,rem[0],add[0]]
                H[:,u,v]=w;H[:,v,u]=w
    deg=H.abs().sum(-1).clamp_min(1e-6)
    return H/(deg[:,:,None]*deg[:,None,:]).sqrt()

@lru_cache(maxsize=16)
def fermion_1rdm_ops(D,k):
    ss=list(itertools.combinations(range(D),k));lookup={s:i for i,s in enumerate(ss)};E=len(ss)
    ops=torch.zeros(D,D,E,E,dtype=torch.float64)
    def parity(bits,pos): return (-1.)**sum(1 for x in bits if x<pos)
    for col,s in enumerate(ss):
        st=set(s)
        for q in range(D):
            if q not in st: continue
            sign1=parity(st,q);t=set(st);t.remove(q)
            for p in range(D):
                if p in t: continue
                sign2=parity(t,p);u=tuple(sorted(t|{p}));row=lookup[u]
                ops[p,q,row,col]=sign1*sign2
    return ops

class FidelityAReadout(nn.Module):
    def __init__(self,own_width=32,D=6,k=3):
        super().__init__();self.D=D;self.k=k;dim=2*D*D
        self.local=nn.Sequential(nn.Linear(own_width,64),nn.SiLU(),nn.LayerNorm(64))
        self.inter=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,64))
        self.gate=nn.Sequential(nn.Linear(own_width+dim,32),nn.SiLU(),nn.Linear(32,1))
        self.norm=nn.LayerNorm(64);nn.init.constant_(self.gate[-1].bias,-1.)
    def node_features(self,state,valid,ss,n):
        ops=fermion_1rdm_ops(self.D,self.k).to(state.device,state.real.dtype).to(state.dtype)
        rho=torch.einsum('bse,pqef,bsf->bspq',state.conj(),ops,state)
        rr=torch.cat((rho.real,rho.imag),-1).flatten(-2)*valid[...,None]
        inc=subset_inc_cpu(n,ss.shape[-1]).to(rr.device,rr.dtype)
        den=torch.einsum('bs,sn->bn',valid.to(rr.dtype),inc).clamp_min(1.)
        return torch.einsum('bsd,sn->bnd',rr,inc)/den[...,None]
    def forward(self,own,state,valid,ss,mask):
        q=self.node_features(state,valid,ss,own.shape[1])
        g=torch.sigmoid(self.gate(torch.cat((own,q),-1)))
        return self.norm(self.local(own)+g*self.inter(q))*mask[...,None]

class RajFidelityAQGNNCore(BoundedCore):
    def __init__(self,j=3,D=6,k=3,rounds=3,enc_hidden=64):
        super().__init__();self.j=j;self.D=D;self.k=k;self.rounds=rounds;self.E=math.comb(D,k)
        self.builder=SubsetFeatures(32);f=self.builder.out_dim
        self.loaders=nn.ModuleList(nn.Sequential(nn.Linear(f,enc_hidden),nn.SiLU(),nn.Linear(enc_hidden,self.E)) for _ in range(rounds+1))
        self.alphas=nn.Parameter(torch.randn(rounds)*.25)
        self.embed_layers=nn.ModuleList(CompoundLayer(D,k) for _ in range(rounds))
        self.readout=FidelityAReadout(32,D,k)
    @staticmethod
    def unit(x):return x/(torch.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-8)
    def forward_patch(self,hist,mask):
        own,feat,valid,ss,edge,risk=self.builder(hist,mask,self.j)
        ctype=torch.complex64 if feat.dtype==torch.float32 else torch.complex128
        x=self.unit(self.loaders[0](feat)).to(ctype)*valid[...,None]
        H=lifted_weighted_hamiltonian(risk,ss)
        for l in range(self.rounds):
            U=torch.matrix_exp((1j*self.alphas[l]).to(ctype)*H.to(ctype))
            x=torch.einsum('bst,bte->bse',U,x)
            x=self.embed_layers[l](x)
            x=x+self.unit(self.loaders[l+1](feat)).to(ctype)
            x=self.unit(x)*valid[...,None]
        return self.readout(own,x,valid,ss,mask)

class RajWeightedJohnsonGINCore(BoundedCore):
    def __init__(self,j=3,rounds=3,hidden=128):
        super().__init__();self.j=j;self.rounds=rounds;self.builder=SubsetFeatures(32);f=self.builder.out_dim
        self.input=nn.Sequential(nn.Linear(f,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.layers=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(rounds))
        self.local=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.LayerNorm(64))
        self.readout=nn.Sequential(nn.Linear(hidden,128),nn.SiLU(),nn.Linear(128,64))
        self.gate=nn.Sequential(nn.Linear(32+hidden,32),nn.SiLU(),nn.Linear(32,1));self.norm=nn.LayerNorm(64)
        nn.init.constant_(self.gate[-1].bias,-1.)
    @staticmethod
    def unit(x):return x/(torch.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-8)
    def forward_patch(self,hist,mask):
        own,feat,valid,ss,edge,risk=self.builder(hist,mask,self.j);H=lifted_weighted_hamiltonian(risk,ss)
        h=self.unit(torch.tanh(self.input(feat)))*valid[...,None]
        for layer in self.layers:
            agg=torch.einsum('bst,bth->bsh',H,h)
            h=self.unit(layer(torch.cat((h,agg),-1)))*valid[...,None]
        inc=subset_inc_cpu(own.shape[1],self.j).to(h.device,h.dtype)
        den=torch.einsum('bs,sn->bn',valid.to(h.dtype),inc).clamp_min(1.)
        p=torch.einsum('bsh,sn->bnh',h,inc)/den[...,None]
        g=torch.sigmoid(self.gate(torch.cat((own,p),-1)))
        return self.norm(self.local(own)+g*self.readout(p))*mask[...,None]
