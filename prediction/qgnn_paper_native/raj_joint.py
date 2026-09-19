"""Batched joint-register primitives for Raj-style QGNN fidelity experiments."""
from __future__ import annotations
import itertools
from functools import lru_cache
from math import comb
import torch
from torch import nn

@lru_cache(maxsize=64)
def fixed_weight_subsets(m:int, weight:int):
    return tuple(itertools.combinations(range(m), weight))

@lru_cache(maxsize=64)
def fixed_weight_index(m:int, weight:int):
    return {s:i for i,s in enumerate(fixed_weight_subsets(m,weight))}

@lru_cache(maxsize=64)
def factored_to_joint_indices(n:int,D:int,j:int,k:int):
    ns=fixed_weight_subsets(n,j); es=fixed_weight_subsets(D,k)
    lookup=fixed_weight_index(n+D,j+k)
    out=torch.empty(len(ns),len(es),dtype=torch.long)
    for si,s in enumerate(ns):
        for ei,e in enumerate(es):
            out[si,ei]=lookup[tuple(sorted((*s,*(n+q for q in e))))]
    return out

@lru_cache(maxsize=512)
def givens_shell_pairs(m:int,weight:int,a:int,b:int):
    if not (0<=a<m and 0<=b<m and a!=b): raise ValueError((m,weight,a,b))
    basis=fixed_weight_subsets(m,weight); lookup=fixed_weight_index(m,weight)
    left=[];right=[]
    for idx,s in enumerate(basis):
        st=set(s)
        if a in st and b not in st:
            partner=tuple(sorted((st-{a})|{b}))
            left.append(idx);right.append(lookup[partner])
    return torch.tensor(left,dtype=torch.long),torch.tensor(right,dtype=torch.long)

@lru_cache(maxsize=64)
def oriented_pair_tables(m:int,weight:int):
    p=comb(m-2,weight-1)
    left=torch.zeros(m,m,p,dtype=torch.long);right=torch.zeros(m,m,p,dtype=torch.long)
    for a in range(m):
        for b in range(m):
            if a==b: continue
            ia,ib=givens_shell_pairs(m,weight,a,b)
            if len(ia)!=p: raise RuntimeError((m,weight,a,b,len(ia),p))
            left[a,b]=ia;right[a,b]=ib
    return left,right

class JointRegisterMixer(nn.Module):
    """Final total-weight mixer. forward is batched; _forward_reference is the frozen slow reference."""
    def __init__(self,n=8,D=6,j=3,k=3,init_std=.03):
        super().__init__();self.n=n;self.D=D;self.j=j;self.k=k
        self.m=n+D;self.total_weight=j+k;self.joint_dim=comb(self.m,self.total_weight)
        self.node_pairs=tuple(itertools.combinations(range(n),2))
        self.emb_pairs=tuple(itertools.combinations(range(D),2))
        self.theta_node=nn.Parameter(torch.randn(len(self.node_pairs))*init_std)
        self.theta_emb=nn.Parameter(torch.randn(len(self.emb_pairs))*init_std)
        self.theta_cross=nn.Parameter(torch.randn(n,D)*init_std)
        self.register_buffer("_factor_joint",factored_to_joint_indices(n,D,j,k))
        li,ri=oriented_pair_tables(self.m,self.total_weight)
        self.register_buffer("_pair_left",li);self.register_buffer("_pair_right",ri)

    def factor_to_joint(self,factor_state):
        expected=(len(fixed_weight_subsets(self.n,self.j)),len(fixed_weight_subsets(self.D,self.k)))
        if factor_state.shape[-2:]!=expected: raise ValueError((factor_state.shape,expected))
        flat=factor_state.reshape(-1,*expected)
        out=flat.new_zeros(flat.shape[0],self.joint_dim)
        out[:,self._factor_joint.reshape(-1)]=flat.reshape(flat.shape[0],-1)
        return out.reshape(*factor_state.shape[:-2],self.joint_dim)

    def project_factor_sector(self,joint_state):
        return joint_state[...,self._factor_joint]

    def _apply_pair(self,x,a,b,theta):
        ia,ib=givens_shell_pairs(self.m,self.total_weight,a,b);ia=ia.to(x.device);ib=ib.to(x.device)
        va,vb=x[...,ia],x[...,ib];c=torch.cos(theta/2).to(x.real.dtype);s=torch.sin(theta/2).to(x.real.dtype)
        y=x.clone();y[...,ia]=c*va-s*vb;y[...,ib]=s*va+c*vb
        return y

    def _apply_pair_batch(self,x,a,b,theta,enabled):
        safe_a=torch.where(enabled,a,torch.zeros_like(a));safe_b=torch.where(enabled,b,torch.ones_like(b))
        ia=self._pair_left[safe_a,safe_b];ib=self._pair_right[safe_a,safe_b]
        va=torch.gather(x,1,ia);vb=torch.gather(x,1,ib)
        th=torch.where(enabled,theta,torch.zeros_like(theta));c=torch.cos(th/2).to(x.real.dtype)[:,None];s=torch.sin(th/2).to(x.real.dtype)[:,None]
        y=x.scatter(1,ia,c*va-s*vb);y=y.scatter(1,ib,s*va+c*vb)
        return y

    def _canonical_node_order(self,weights,active):
        degree=weights.abs().sum(-1);nodes=[i for i in range(self.n) if bool(active[i])]
        return sorted(nodes,key=lambda i:(-float(degree[i]),i))

    def _canonical_pair_order(self,weights,active):
        order=self._canonical_node_order(weights,active);rank={node:r for r,node in enumerate(order)}
        rows=[]
        for a,b in self.node_pairs:
            if bool(active[a] and active[b]):
                u,v=(a,b) if rank[a]<rank[b] else (b,a)
                rows.append((float(weights[a,b]),rank[u],rank[v],u,v))
        rows.sort(key=lambda z:(-z[0],z[1],z[2]))
        return [(u,v) for _,_,_,u,v in rows]

    def _orders_cpu(self,weights,mask):
        wc=weights.detach().cpu();mc=mask.detach().cpu()
        return ([self._canonical_pair_order(wc[b],mc[b]) for b in range(len(wc))],
                [self._canonical_node_order(wc[b],mc[b]) for b in range(len(wc))])

    def _forward_reference(self,factor_state,weights,mask):
        joint=self.factor_to_joint(factor_state);wc=weights.detach().cpu();mc=mask.detach().cpu();rows=[]
        for bi in range(joint.shape[0]):
            x=joint[bi]
            pairs=self._canonical_pair_order(wc[bi],mc[bi])
            for rank,(a,b) in enumerate(pairs): x=self._apply_pair(x,a,b,self.theta_node[rank])
            for rank,(ea,eb) in enumerate(self.emb_pairs): x=self._apply_pair(x,self.n+ea,self.n+eb,self.theta_emb[rank])
            nodes=self._canonical_node_order(wc[bi],mc[bi])
            for rank,node in enumerate(nodes):
                for eb in range(self.D): x=self._apply_pair(x,node,self.n+eb,self.theta_cross[rank,eb])
            rows.append(x)
        return torch.stack(rows,0)

    def forward(self,factor_state,weights,mask):
        if factor_state.ndim!=3: raise ValueError("factor_state must be [B,S,E]")
        x=self.factor_to_joint(factor_state);B=x.shape[0];device=x.device
        pair_orders,node_orders=self._orders_cpu(weights,mask)
        for rank in range(max((len(o) for o in pair_orders),default=0)):
            en=torch.tensor([rank<len(o) for o in pair_orders],device=device,dtype=torch.bool)
            a=torch.tensor([o[rank][0] if rank<len(o) else 0 for o in pair_orders],device=device)
            b=torch.tensor([o[rank][1] if rank<len(o) else 1 for o in pair_orders],device=device)
            x=self._apply_pair_batch(x,a,b,self.theta_node[rank].expand(B),en)
        en=torch.ones(B,device=device,dtype=torch.bool)
        for rank,(ea,eb) in enumerate(self.emb_pairs):
            a=torch.full((B,),self.n+ea,device=device,dtype=torch.long);b=torch.full((B,),self.n+eb,device=device,dtype=torch.long)
            x=self._apply_pair_batch(x,a,b,self.theta_emb[rank].expand(B),en)
        for rank in range(max((len(o) for o in node_orders),default=0)):
            en=torch.tensor([rank<len(o) for o in node_orders],device=device,dtype=torch.bool)
            node=torch.tensor([o[rank] if rank<len(o) else 0 for o in node_orders],device=device)
            for eb in range(self.D):
                emb=torch.full((B,),self.n+eb,device=device,dtype=torch.long)
                x=self._apply_pair_batch(x,node,emb,self.theta_cross[rank,eb].expand(B),en)
        return x

def conditional_embedding_states(mixer,joint_state,eps=1e-12):
    factor=mixer.project_factor_sector(joint_state)
    prob=factor.abs().square().sum(-1)
    cond=factor/prob.clamp_min(eps).sqrt()[...,None]
    cond=torch.where((prob>eps)[...,None],cond,torch.zeros_like(cond))
    return cond,prob

