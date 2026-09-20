"""Paper-faithful Raj-style two-register reduced-basis QGNN adaptation for SinD.

Inspired by Raj et al. (2026), arXiv:2606.26873:
- j-subset node register / Johnson lift
- graph-conditioned particle-number-preserving node mixing
- D=6,k=3 embedding register with trainable compound/RBS evolution
- data re-uploading
- full complex embedding-register 1-RDM readout

The public paper repository does not expose the complete joint-register mixer/readout
as a reusable module. This file is therefore an ICCT adaptation of the published
mathematical construction, not a verbatim copy of official code.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import comb
import torch
from torch import nn

from .common import BoundedCore, OwnHistoryEncoder, physical_graph


@lru_cache(maxsize=32)
def subset_tensors(n: int, j: int):
    subs=list(itertools.combinations(range(n),j))
    links=[]
    for u in range(len(subs)):
        su=set(subs[u])
        for v in range(u+1,len(subs)):
            sv=set(subs[v])
            rem=sorted(su-sv); add=sorted(sv-su)
            if len(rem)==len(add)==1:
                links.append((u,v,rem[0],add[0]))
    inc=torch.zeros(len(subs),n,dtype=torch.float32)
    for s,t in enumerate(subs):
        inc[s,list(t)]=1.
    return torch.tensor(subs,dtype=torch.long),torch.tensor(links,dtype=torch.long).reshape(-1,4),inc


@lru_cache(maxsize=16)
def embed_subsets(D:int,k:int):
    return list(itertools.combinations(range(D),k))


@lru_cache(maxsize=16)
def rdm_operators_cpu(D:int,k:int):
    """Matrices of a_p^dagger a_q restricted to the weight-k basis."""
    subs=embed_subsets(D,k);index={s:i for i,s in enumerate(subs)}
    ops=torch.zeros(D,D,len(subs),len(subs),dtype=torch.complex128)
    for col,S in enumerate(subs):
        occ=set(S)
        for q in range(D):
            if q not in occ: continue
            sign1=(-1)**sum(x<q for x in occ)
            after=occ-{q}
            for p in range(D):
                if p in after: continue
                sign2=(-1)**sum(x<p for x in after)
                T=tuple(sorted(after|{p}))
                row=index[T]
                ops[p,q,row,col]=sign1*sign2
    return ops


class CompoundEvolution(nn.Module):
    """Official-paper style SO(D) Cayley layer lifted by the k-th compound."""

    def __init__(self,D=6,k=3,init_std=.1):
        super().__init__();self.D=D;self.k=k;self.dim=comb(D,k)
        self.theta=nn.Parameter(torch.randn(D*(D-1)//2)*init_std)
        idx=torch.triu_indices(D,D,offset=1)
        self.register_buffer('_row',idx[0]);self.register_buffer('_col',idx[1]);self.register_buffer('_eye',torch.eye(D))

    def orthogonal(self):
        D=self.D;skew=torch.zeros(D,D,device=self.theta.device,dtype=self.theta.dtype)
        skew[self._row,self._col]=self.theta;skew=skew-skew.T
        eye=self._eye.to(self.theta.dtype)
        return torch.linalg.solve(eye+skew,eye-skew)

    def compound(self):
        O=self.orthogonal()
        if self.k==1:return O
        subs=torch.tensor(embed_subsets(self.D,self.k),device=O.device,dtype=torch.long)
        blocks=O[subs][:,:,subs].permute(0,2,1,3)
        return torch.linalg.det(blocks)

    def forward(self,x):
        return torch.matmul(x,self.compound().T.to(x.dtype))


class ContinuousSubsetLoader(nn.Module):
    """Permutation-symmetric continuous physical features for each j-subset."""

    def __init__(self,own_width=32,D=6,k=3,hidden=64):
        super().__init__();self.own=OwnHistoryEncoder(own_width);self.emb=comb(D,k)
        # own mean/std + physical-edge mean/std + risk mean/std + dynamic extrema
        self.feat_dim=2*own_width+2*16+6
        self.net=nn.Sequential(nn.Linear(self.feat_dim,hidden),nn.SiLU(),nn.Linear(hidden,self.emb))

    def forward(self,history,mask,j):
        own=self.own(history,mask);edge,risk=physical_graph(history,mask)
        b,n,w=own.shape;subs,_,_=subset_tensors(n,j);subs=subs.to(history.device)
        nodes=own[:,subs];nmean=nodes.mean(-2);nstd=nodes.var(-2,unbiased=False).add(1e-6).sqrt()
        ep=[];rp=[]
        for a,c in itertools.combinations(range(j),2):
            ia,ic=subs[:,a],subs[:,c];ep.append(edge[:,ia,ic]);rp.append(risk[:,ia,ic])
        e=torch.stack(ep,-2);r=torch.stack(rp,-1)
        # Retain richer interaction shape than a single gamma scalar.
        rmean=r.mean(-1,keepdim=True);rstd=r.var(-1,unbiased=False,keepdim=True).add(1e-6).sqrt()
        rmax=r.max(-1,keepdim=True).values;rmin=r.min(-1,keepdim=True).values
        # closing feature is index 5 in current physical edge definition; distance index 4.
        dist=e[...,0];closing=e[...,1]
        extra=torch.cat((rmean,rstd,rmax,rmin,dist.mean(-1,keepdim=True),closing.mean(-1,keepdim=True)),-1)
        feat=torch.cat((nmean,nstd,e.mean(-2),e.var(-2,unbiased=False).add(1e-6).sqrt(),extra),-1)
        valid=mask[:,subs].all(-1)
        amp=self.net(feat)
        amp=amp/(torch.linalg.vector_norm(amp,dim=-1,keepdim=True)+1e-8)
        return own,edge,risk,feat,amp*valid[...,None],valid,subs


class WeightedJohnsonMix(nn.Module):
    """Graph-conditioned hopping Hamiltonian on the j-subset node register."""

    def __init__(self,j:int,depth:int,edge_dim=16,hidden=16):
        super().__init__();self.j=j;self.depth=depth
        self.angle=nn.ModuleList(nn.Sequential(nn.Linear(edge_dim,hidden),nn.SiLU(),nn.Linear(hidden,1)) for _ in range(depth))
        self.scale=nn.Parameter(torch.full((depth,),0.35))

    def hamiltonian(self,edge,mask,layer):
        b,n,_,_=edge.shape;subs,links,_=subset_tensors(n,self.j)
        links=links.to(edge.device);S=len(subs)
        H=edge.new_zeros(b,S,S)
        if len(links)==0:return H
        a,c=links[:,2],links[:,3]
        ef=edge[:,a,c]
        theta=torch.tanh(self.angle[layer](ef).squeeze(-1))*self.scale[layer]
        valid=(mask[:,a]&mask[:,c]).to(theta.dtype);theta=theta*valid
        u,v=links[:,0],links[:,1]
        H[:,u,v]=theta;H[:,v,u]=theta
        return H

    def forward(self,x,edge,mask,layer):
        H=self.hamiltonian(edge,mask,layer)
        U=torch.matrix_exp(1j*H.to(x.dtype))
        return torch.bmm(U,x)


def conditional_rdm(x,D=6,k=3):
    """Full complex 1-RDM for each conditional subset embedding state."""
    x=x/(torch.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-8)
    op=rdm_operators_cpu(D,k).to(x.device,x.dtype)
    # gamma[p,q] = <x| a_p^dagger a_q |x>
    return torch.einsum('bsf,pqfg,bsg->bspq',x.conj(),op,x)


class IncidenceRDMReadout(nn.Module):
    def __init__(self,D=6,out=64):
        super().__init__();self.D=D
        self.net=nn.Sequential(nn.Linear(2*D*D+2,out),nn.SiLU(),nn.Linear(out,out),nn.LayerNorm(out))

    def forward(self,x,valid,subs,n,mask):
        gamma=conditional_rdm(x,self.D,3)
        _,_,inc0=subset_tensors(n,subs.shape[-1]);inc=inc0.to(x.device,x.real.dtype)
        w=valid.to(x.real.dtype)
        den=torch.einsum('bs,sn->bn',w,inc).clamp_min(1.)
        # Per-agent incidence average of conditional RDMs.
        gr=torch.einsum('bspq,bs,sn->bnpq',gamma.real,w,inc)/den[...,None,None]
        gi=torch.einsum('bspq,bs,sn->bnpq',gamma.imag,w,inc)/den[...,None,None]
        # Projection/incidence statistics retained explicitly.
        count=den/valid.sum(1,keepdim=True).clamp_min(1.)
        purity=(gr.square()+gi.square()).sum((-1,-2))
        raw=torch.cat((gr.flatten(-2),gi.flatten(-2),count[...,None],purity[...,None]),-1)
        return self.net(raw)*mask[...,None]


class RajPaperGINCore(BoundedCore):
    """Original-graph 1-WL GIN-style paper-native baseline adapted to weighted SinD graph."""
    def __init__(self,depth=3,hidden=128):
        super().__init__();self.depth=depth;self.hidden=hidden
        self.encoder=OwnHistoryEncoder(32)
        self.input=nn.Linear(32,hidden)
        self.layers=nn.ModuleList(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(depth))
        self.readout=nn.Sequential(nn.Linear((depth+1)*hidden,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64))
        self.eps=nn.Parameter(torch.zeros(depth))
    def forward_patch(self,history,mask):
        own=self.encoder(history,mask);_,risk=physical_graph(history,mask)
        h=torch.relu(self.input(own))*mask[...,None];hs=[h]
        for l,layer in enumerate(self.layers):
            agg=torch.bmm(risk,h)
            h=torch.relu(layer((1+self.eps[l])*h+agg))*mask[...,None];hs.append(h)
        return self.readout(torch.cat(hs,-1))*mask[...,None]

class PPGNBlock(nn.Module):
    """Compact PPGN-style 2-tensor block with channelwise matrix multiplication."""
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.a=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.b=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.update=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
    def forward(self,x,pairmask):
        a=self.a(x)*pairmask[...,None];b=self.b(x)*pairmask[...,None]
        prod=torch.einsum('bikc,bkjc->bijc',a,b)
        den=pairmask.sum(-1,keepdim=True).clamp_min(1).to(x.dtype)
        prod=prod/den[...,None]
        y=self.update(torch.cat((x,prod),-1))
        return self.norm(x+y)*pairmask[...,None]

class RajPaperPPGNCore(BoundedCore):
    """PPGN-style higher-order classical reference on node/edge 2-tensors."""
    def __init__(self,depth=3,width=64):
        super().__init__();self.depth=depth;self.width=width
        self.encoder=OwnHistoryEncoder(32)
        self.node=nn.Linear(32,width);self.edge=nn.Linear(16,width)
        self.layers=nn.ModuleList(PPGNBlock(width) for _ in range(depth))
        self.readout=nn.Sequential(nn.Linear((depth+1)*3*width,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64))
    def forward_patch(self,history,mask):
        own=self.encoder(history,mask);edge,_=physical_graph(history,mask);b,n,_=own.shape
        pairmask=mask[:,:,None]&mask[:,None,:]
        x=self.edge(edge)*pairmask[...,None]
        idx=torch.arange(n,device=x.device)
        x[:,idx,idx]=self.node(own)
        layers=[x]
        for layer in self.layers:
            x=layer(x,pairmask);layers.append(x)
        feats=[]
        off=pairmask&~torch.eye(n,device=x.device,dtype=torch.bool)[None]
        den=off.sum(-1,keepdim=True).clamp_min(1).to(x.dtype)
        for z in layers:
            d=z[:,idx,idx]
            row=(z*off[...,None]).sum(2)/den
            col=(z*off[...,None]).sum(1)/den
            feats.extend((d,row,col))
        return self.readout(torch.cat(feats,-1))*mask[...,None]

class RajWeightedSubsetQGNNCore(BoundedCore):
    """Successful ICCT subset interface + Raj paper graph-conditioned node mixing and embedding evolution."""
    def __init__(self,j=3,rounds=3,D=6,k=3,encoder_hidden=64):
        super().__init__();self.j=j;self.rounds=rounds;self.D=D;self.k=k
        from .classical import SubsetFeatureBuilder, _SubsetReadout
        self.builder=SubsetFeatureBuilder(32);f=self.builder.out_dim;self.embed_dim=comb(D,k)
        self.enc0=nn.Sequential(nn.Linear(f,encoder_hidden),nn.SiLU(),nn.Linear(encoder_hidden,self.embed_dim))
        self.encs=nn.ModuleList(nn.Sequential(nn.Linear(f,encoder_hidden),nn.SiLU(),nn.Linear(encoder_hidden,self.embed_dim)) for _ in range(rounds))
        self.mix=WeightedJohnsonMix(j,rounds)
        self.embed_layers=nn.ModuleList(CompoundEvolution(D,k) for _ in range(rounds))
        self.readout=_SubsetReadout(32,self.embed_dim,rounds)
    @staticmethod
    def _unit(x):return x/(torch.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-8)
    def forward_patch(self,history,mask):
        own,feat,valid,subs=self.builder(history,mask,self.j)
        if subs.shape[0]==0:return self.readout.local(own)*mask[...,None]
        edge,_=physical_graph(history,mask)
        ctype=torch.complex64 if feat.dtype==torch.float32 else torch.complex128
        x=self._unit(self.enc0(feat)).to(ctype)*valid[...,None];zs=[x]
        for l in range(self.rounds):
            x=self.mix(x,edge,mask,l)*valid[...,None]
            x=self.embed_layers[l](x)
            x=x+self._unit(self.encs[l](feat)).to(x.dtype)*valid[...,None]
            x=self._unit(x)*valid[...,None];zs.append(x)
        return self.readout.forward_out(own,zs,valid,subs,mask)

class RajWeightedMultiJQGNNCore(BoundedCore):
    """All scenes run both weighted quantum j=2 and j=3 branches."""
    def __init__(self,rounds=3):
        super().__init__()
        from .classical import MultiJFusion
        self.j2=RajWeightedSubsetQGNNCore(j=2,rounds=rounds)
        self.j3=RajWeightedSubsetQGNNCore(j=3,rounds=rounds)
        self.fuse=MultiJFusion()
    def forward_patch(self,history,mask):
        return self.fuse(self.j2.forward_patch(history,mask),self.j3.forward_patch(history,mask),mask)
