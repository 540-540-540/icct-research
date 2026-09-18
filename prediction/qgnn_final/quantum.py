"""Exact batched graph-conditioned ZZ/ZZZ circuit, not a classical GNN postprocessor."""
from __future__ import annotations
import itertools, math
from functools import lru_cache
import torch
from torch import nn
from .common import BoundedCore, OwnHistoryEncoder, physical_graph

@lru_cache(maxsize=32)
def tables(n, device, dtype):
    index = torch.arange(2**n,device=device)
    z = torch.stack([1-2*((index >> (n-1-i)) & 1) for i in range(n)],1).to(dtype)
    pairs = torch.tensor(list(itertools.combinations(range(n),2)),device=device,dtype=torch.long).reshape(-1,2)
    triples = torch.tensor(list(itertools.combinations(range(n),3)),device=device,dtype=torch.long).reshape(-1,3)
    zz = z[:,pairs].prod(-1)
    zzz = z[:,triples].prod(-1)
    return index,z,pairs,triples,zz,zzz

def rotate(state, angle, qubit, axis):
    n = state.shape[-1].bit_length()-1
    x = state.reshape(state.shape[0],2**qubit,2,2**(n-qubit-1))
    a,b = x[:,:,0],x[:,:,1]
    c,s = torch.cos(angle/2)[:,None,None],torch.sin(angle/2)[:,None,None]
    if axis=='Y':
        out = torch.stack((c*a-s*b,s*a+c*b),2)
    elif axis=='X':
        out = torch.stack((c*a-1j*s*b,c*b-1j*s*a),2)
    else:
        raise ValueError(axis)
    return out.reshape_as(state)

def evolve(ry, rz, pair_angles, triple_angles, rx, mask):
    """Shapes: [B,L,N], [B,L,P], [B,L,H], rx [L]; chronological snapshots."""
    b,depth,n = ry.shape
    _,z,_,_,zz,zzz = tables(n,ry.device,ry.dtype)
    state = torch.zeros(b,2**n,device=ry.device,dtype=torch.complex128 if ry.dtype==torch.float64 else torch.complex64)
    state[:,0] = 1
    snapshots=[]
    for layer in range(depth):
        for i in range(n):
            state = rotate(state,ry[:,layer,i],i,'Y')
        phase = rz[:,layer]@z.T + pair_angles[:,layer]@zz.T + triple_angles[:,layer]@zzz.T
        state = state*torch.exp(-0.5j*phase)
        for i in range(n):
            state = rotate(state,rx[layer].expand(b)*mask[:,i],i,'X')
        snapshots.append(state)
    return snapshots

def moments(state, mask, risk, triple_weight):
    b,n = mask.shape
    idx,z,pairs,tri,zz,zzz = tables(n,state.device,state.real.dtype)
    prob = state.abs().square()
    mz = prob@z
    flips = idx[None,:] ^ (1 << torch.arange(n-1,-1,-1,device=state.device))[:,None]
    coherence = state.conj()[:,None,:]*state[:,flips]
    mx = coherence.sum(-1).real
    my = (-1j*z.T[None]*coherence).sum(-1).real
    raw2 = prob@zz
    conn2 = raw2-mz[:,pairs[:,0]]*mz[:,pairs[:,1]]
    pair_weight = risk[:,pairs[:,0],pairs[:,1]]
    inc2 = torch.nn.functional.one_hot(pairs,num_classes=n).sum(1).to(mz.dtype)
    inc3 = torch.nn.functional.one_hot(tri,num_classes=n).sum(1).to(mz.dtype)
    if tri.shape[0]:
        pair_lookup = torch.zeros(n,n,device=state.device,dtype=torch.long)
        pair_lookup[pairs[:,0],pairs[:,1]] = torch.arange(len(pairs),device=state.device)
        a,c,d = tri.unbind(-1)
        conn3 = prob@zzz-mz[:,a]*raw2[:,pair_lookup[c,d]]-mz[:,c]*raw2[:,pair_lookup[a,d]]-mz[:,d]*raw2[:,pair_lookup[a,c]]+2*mz[:,a]*mz[:,c]*mz[:,d]
    else:
        conn3 = mz.new_zeros(b,0)
    def pool(value,weight,inc):
        den = weight@inc
        mean = (value*weight)@inc/den.clamp_min(1e-6)
        rms = (((value.square()*weight)@inc/den.clamp_min(1e-6)).clamp_min(0)+1e-12).sqrt()
        return mean*(den>0),rms*(den>0)
    mean2,rms2 = pool(conn2,pair_weight,inc2)
    mean3,rms3 = pool(conn3,triple_weight,inc3)
    return torch.stack((mx,my,mz,mean2,rms2,mean3,rms3),-1)*mask[...,None]

class HypergraphQuantumCore(BoundedCore):
    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
        self.encoder = OwnHistoryEncoder(32)
        self.local = nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.LayerNorm(64))
        self.node_angle = nn.Linear(32,2*depth)
        self.pair_angle = nn.Sequential(nn.Linear(16,16),nn.SiLU(),nn.Linear(16,depth))
        self.triple_angle = nn.Sequential(nn.Linear(32,16),nn.SiLU(),nn.Linear(16,depth))
        self.rx = nn.Parameter(torch.full((depth,),0.4))
        self.readout = nn.Sequential(nn.Linear(7*depth,64),nn.SiLU(),nn.Linear(64,64))
        self.gate = nn.Sequential(nn.Linear(32+7*depth,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.gate[-1].bias,-1.)
        self.interaction_scale = 1.0
        self.triple_scale = 1.0

    def circuit_inputs(self, history, mask):
        h = self.encoder(history,mask)
        edge,risk = physical_graph(history,mask)
        b,n,_ = h.shape
        _,_,pairs,tri,_,_ = tables(n,h.device,h.dtype)
        angle = 1.2*torch.tanh(self.node_angle(h)).reshape(b,n,self.depth,2).permute(0,2,1,3)
        ry,rz = angle[...,0],angle[...,1]
        start = torch.zeros_like(ry); start[:,0] = math.pi/2
        ry = (ry+start)*mask[:,None]; rz=rz*mask[:,None]
        deg = (mask.sum(1)-1).clamp_min(1).to(h.dtype)
        pe = edge[:,pairs[:,0],pairs[:,1]]
        pw = risk[:,pairs[:,0],pairs[:,1]]
        pa = 1.5*torch.tanh(self.pair_angle(pe)+0.5)*pw[...,None]/deg[:,None,None].sqrt()
        a,c,d = tri.unbind(-1)
        es = torch.stack((edge[:,a,c],edge[:,a,d],edge[:,c,d]),-2)
        tf = torch.cat((es.mean(-2),es.var(-2,unbiased=False).add(1e-6).sqrt()),-1)
        r1,r2,r3 = risk[:,a,c],risk[:,a,d],risk[:,c,d]
        tw = (r1*r2+r1*r3+r2*r3)/3
        tw = tw*(mask[:,a]&mask[:,c]&mask[:,d])
        denom = (deg*(deg-1)/2).clamp_min(1).sqrt()
        ta = 1.5*torch.tanh(self.triple_angle(tf)+0.5)*tw[...,None]/denom[:,None,None]
        return h,risk,tw,ry,rz,pa.permute(0,2,1)*self.interaction_scale,ta.permute(0,2,1)*self.interaction_scale*self.triple_scale

    def forward_patch(self, history, mask):
        h,risk,tw,ry,rz,pa,ta = self.circuit_inputs(history,mask)
        states = evolve(ry,rz,pa,ta,self.rx,mask)
        q = torch.cat([moments(state,mask,risk,tw) for state in states],-1)
        gate = torch.sigmoid(self.gate(torch.cat((h,q),-1)))
        return (self.local(h)+gate*self.readout(q))*mask[...,None]
