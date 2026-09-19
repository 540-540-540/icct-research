"""Adaptive pair and rooted-triplet classical countermodel."""
from __future__ import annotations
import itertools
from functools import lru_cache
import torch
from torch import nn
from .common import BoundedCore, OwnHistoryEncoder, physical_graph

@lru_cache(maxsize=32)
def triplet_indices(n,device):
    entries = [[(j,k) for j,k in itertools.combinations(range(n),2) if i not in (j,k)] for i in range(n)]
    return torch.tensor(entries,device=device,dtype=torch.long).reshape(n,-1,2)

def weighted_attention(value,score,weight,valid):
    heads=score.shape[-1];width=value.shape[-1]
    if width%heads: raise ValueError('width must be divisible by heads')
    score=score+torch.log(weight.clamp_min(1e-4))[...,None]
    score=score.masked_fill(~valid[...,None],-1e9)
    alpha=torch.softmax(score,dim=-2)*valid[...,None]
    alpha=alpha/alpha.sum(-2,keepdim=True).clamp_min(1e-8)
    values=value.reshape(*value.shape[:-1],heads,width//heads)
    return (alpha[...,None]*values).sum(-3).flatten(-2)

class AdaptiveHypergraphLayer(nn.Module):
    def __init__(self,width=64,channels=4):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(2*width+16,128),nn.SiLU(),nn.Linear(128,width+channels))
        self.triplet = nn.Sequential(nn.Linear(3*width+48,128),nn.SiLU(),nn.Linear(128,width+channels))
        self.mix = nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,3))
        self.update = nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,width))
        self.norm = nn.LayerNorm(width)
    def forward(self,h,edge,risk,mask):
        b,n,w = h.shape
        hi = h[:,:,None].expand(-1,-1,n,-1)
        hj = h[:,None].expand(-1,n,-1,-1)
        p = self.pair(torch.cat((hi,hj,edge),-1))
        valid = mask[:,:,None]&mask[:,None,:]&~torch.eye(n,device=h.device,dtype=torch.bool)[None]
        pair = weighted_attention(p[...,:w],p[...,w:],risk,valid)
        trip = torch.zeros_like(pair)
        if n>=3:
            indices = triplet_indices(n,h.device)
            j,k = indices[...,0],indices[...,1]
            i = torch.arange(n,device=h.device)[:,None]
            hi = h[:,:,None].expand(-1,-1,j.shape[1],-1)
            hj,hk = h[:,j],h[:,k]
            eij,eik,ejk = edge[:,i,j],edge[:,i,k],edge[:,j,k]
            inp = torch.cat((hi,hj+hk,(hj-hk).abs(),eij+eik,(eij-eik).abs(),ejk),-1)
            tr = self.triplet(inp)
            r1,r2,r3 = risk[:,i,j],risk[:,i,k],risk[:,j,k]
            weights = (r1*r2+r1*r3+r2*r3)/3
            validtr = mask[:,:,None]&mask[:,j]&mask[:,k]
            trip = weighted_attention(tr[...,:w],tr[...,w:],weights,validtr)
        joined = torch.cat((h,pair,trip),-1)
        gates = torch.softmax(self.mix(joined),-1)
        delta = self.update(torch.cat((gates[...,0:1]*h,gates[...,1:2]*pair,gates[...,2:3]*trip),-1))
        return self.norm(h+delta)*mask[...,None]

class AdaptiveClassicalCore(BoundedCore):
    def __init__(self,depth=3,channels=4):
        super().__init__()
        self.encoder = OwnHistoryEncoder(32)
        self.local = nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.LayerNorm(64))
        self.input = nn.Linear(32,64)
        self.layers = nn.ModuleList([AdaptiveHypergraphLayer(channels=channels) for _ in range(depth)])
        self.readout = nn.Sequential(nn.Linear(depth*64,64),nn.SiLU(),nn.Linear(64,64))
        self.gate = nn.Sequential(nn.Linear(32+depth*64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.gate[-1].bias,-1.)
    def forward_patch(self,history,mask):
        own = self.encoder(history,mask)
        edge,risk = physical_graph(history,mask)
        h = self.input(own)*mask[...,None]
        layers=[]
        for layer in self.layers:
            h=layer(h,edge,risk,mask); layers.append(h)
        joint = torch.cat(layers,-1)
        gate = torch.sigmoid(self.gate(torch.cat((own,joint),-1)))
        return (self.local(own)+gate*self.readout(joint))*mask[...,None]
