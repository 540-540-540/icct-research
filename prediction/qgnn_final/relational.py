"""Relation-carrying quantum readout. No classical message passing precedes U(G)."""
from __future__ import annotations
import itertools
from functools import lru_cache
import torch
from torch import nn
from .quantum import HypergraphQuantumCore, evolve, tables
from .common import physical_graph
from .classical import triplet_indices

@lru_cache(maxsize=32)
def rooted_to_global(n, device):
    triples=list(itertools.combinations(range(n),3))
    lookup={t:i for i,t in enumerate(triples)}
    rooted=triplet_indices(n,device).cpu().tolist()
    return torch.tensor([[lookup[tuple(sorted((i,j,k)))] for j,k in pairs] for i,pairs in enumerate(rooted)],device=device,dtype=torch.long)

def cumulants(state, n):
    _,z,pairs,tri,zz,zzz=tables(n,state.device,state.real.dtype)
    prob=state.abs().square(); mz=prob@z; raw2=prob@zz
    c2=raw2-mz[:,pairs[:,0]]*mz[:,pairs[:,1]]
    full=state.real.new_zeros(state.shape[0],n,n)
    full[:,pairs[:,0],pairs[:,1]]=c2;full[:,pairs[:,1],pairs[:,0]]=c2
    if n<3: return full, state.real.new_zeros(state.shape[0],0)
    lookup=torch.zeros(n,n,device=state.device,dtype=torch.long)
    lookup[pairs[:,0],pairs[:,1]]=torch.arange(len(pairs),device=state.device)
    i,j,k=tri.unbind(-1)
    c3=prob@zzz-mz[:,i]*raw2[:,lookup[j,k]]-mz[:,j]*raw2[:,lookup[i,k]]-mz[:,k]*raw2[:,lookup[i,j]]+2*mz[:,i]*mz[:,j]*mz[:,k]
    return full,c3

class RelationCarryingQuantumCore(HypergraphQuantumCore):
    """Joint quantum cumulants route relation-valued messages; product states carry none."""
    def __init__(self,depth=3,channels=4):
        super().__init__(depth,channels,True)
        self.message_width=16
        self.pair_value=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,16))
        self.triple_value=nn.Sequential(nn.Linear(112,32),nn.SiLU(),nn.Linear(32,16))
        features=depth*channels*(7+2*self.message_width)
        self.readout=nn.Sequential(nn.Linear(features,64),nn.SiLU(),nn.Linear(64,64))
        self.gate=nn.Sequential(nn.Linear(32+features,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.gate[-1].bias,-1.)

    def relation_values(self, own, history, mask):
        edge,risk=physical_graph(history,mask);b,n,w=own.shape
        neighbors=own[:,None].expand(-1,n,-1,-1)
        pair=self.pair_value(torch.cat((neighbors,edge),-1))
        if n<3: return pair,own.new_zeros(b,n,0,16),risk,own.new_zeros(b,n,0)
        indices=triplet_indices(n,own.device);j,k=indices.unbind(-1)
        i=torch.arange(n,device=own.device)[:,None]
        hj,hk=own[:,j],own[:,k]
        eij,eik,ejk=edge[:,i,j],edge[:,i,k],edge[:,j,k]
        inp=torch.cat((hj+hk,(hj-hk).abs(),eij+eik,(eij-eik).abs(),ejk),-1)
        tri=self.triple_value(inp)
        r1,r2,r3=risk[:,i,j],risk[:,i,k],risk[:,j,k]
        tw=(r1*r2+r1*r3+r2*r3)/3*(mask[:,:,None]&mask[:,j]&mask[:,k])
        return pair,tri,risk,tw

    def relational_features(self,states,own,history,mask):
        b,n,_=own.shape;ch=self.channels
        pair,tri,risk,tw=self.relation_values(own,history,mask)
        message=[]
        for state in states:
            c2,c3=cumulants(state,n)
            c2=c2.reshape(b,ch,n,n)
            a2=torch.asinh(c2/.02)*risk[:,None]
            m2=torch.einsum("bcij,bijf->bcif",a2,pair)/risk.sum(-1)[:,None,:,None].clamp_min(1e-6)
            if n>=3:
                c3=c3[:,rooted_to_global(n,own.device)].reshape(b,ch,n,-1)
                a3=torch.asinh(c3/.005)*tw[:,None]
                m3=torch.einsum("bcik,bikf->bcif",a3,tri)/tw.sum(-1)[:,None,:,None].clamp_min(1e-6)
            else: m3=torch.zeros_like(m2)
            message.append(torch.cat((m2,m3),-1))
        return torch.cat(message,-1).permute(0,2,1,3).flatten(-2)*mask[...,None]

    def features_and_own(self,history,mask):
        own,risk,tw,ry,rz,pa,ta,rx,expanded_mask=self.circuit_inputs(history,mask)
        states=evolve(ry,rz,pa,ta,rx,expanded_mask)
        base=self.readout_features(states,expanded_mask,risk,tw,own.shape[0])
        relation=self.relational_features(states,own,history,mask)
        return own,torch.cat((base,relation),-1)

    def forward_patch(self,history,mask):
        own,features=self.features_and_own(history,mask)
        gate=torch.sigmoid(self.gate(torch.cat((own,features),-1)))
        return (self.local(own)+gate*self.readout(features))*mask[...,None]
