from __future__ import annotations
import math
from functools import lru_cache
import torch
from torch import nn

HISTORY=20
BLOCKS=4
BLOCK=5

class PrefixHistoryEncoder(nn.Module):
    """Shared single-agent temporal encoder. No cross-agent operation."""
    def __init__(self, anchor: str = "first", width: int = 32):
        super().__init__()
        if anchor not in {"first","last"}: raise ValueError(anchor)
        self.anchor=anchor
        self.input=nn.Linear(6,width)
        self.gru=nn.GRU(width,width,batch_first=True)
        self.norm=nn.LayerNorm(width)
    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b,t,n,_=history.shape
        if t!=HISTORY: raise ValueError("Expected 20 history frames")
        a=history[:,0:1] if self.anchor=="first" else history[:,-1:]
        feat=torch.cat(((history[...,:2]-a[...,:2])/10., history[...,2:]/10., a[...,:2].expand(-1,t,-1,-1)/50.),-1)
        x=feat.permute(0,2,1,3).reshape(b*n,t,6)
        seq,_=self.gru(torch.nn.functional.silu(self.input(x)))
        idx=torch.tensor([4,9,14,19],device=history.device)
        h=self.norm(seq.index_select(1,idx)).reshape(b,n,BLOCKS,-1).permute(0,2,1,3)
        return h*mask[:,None,:,None]

def block_statistics(
    history: torch.Tensor,
    timestamps: torch.Tensor | None,
    mask: torch.Tensor,
    acceleration: str = "delta",
):
    b,t,n,_=history.shape
    p=history[...,:2].reshape(b,BLOCKS,BLOCK,n,2).mean(2)
    v=history[...,2:].reshape(b,BLOCKS,BLOCK,n,2).mean(2)
    if timestamps is None:
        ts=torch.arange(t,device=history.device,dtype=history.dtype)[None].expand(b,-1)*0.1
    else:
        ts=timestamps.to(history.dtype)
    tb=ts.reshape(b,BLOCKS,BLOCK).mean(-1)
    if acceleration == "delta":
        acc=torch.zeros_like(v)
        tt=ts[:,:BLOCK]-ts[:,:BLOCK].mean(-1,keepdim=True)
        vv=history[:,:BLOCK,:,2:]-history[:,:BLOCK,:,2:].mean(1,keepdim=True)
        den=tt.square().sum(-1).clamp_min(1e-8)
        acc[:,0]=(tt[:,:,None,None]*vv).sum(1)/den[:,None,None]
        dt=(tb[:,1:]-tb[:,:-1]).clamp_min(1e-6)
        acc[:,1:]=(v[:,1:]-v[:,:-1])/dt[:,:,None,None]
    elif acceleration == "window_ls":
        tt=ts.reshape(b,BLOCKS,BLOCK)
        tc=tt-tt.mean(-1,keepdim=True)
        vv=history[...,2:].reshape(b,BLOCKS,BLOCK,n,2)
        vc=vv-vv.mean(2,keepdim=True)
        den=tc.square().sum(-1).clamp_min(1e-8)
        acc=(tc[...,None,None]*vc).sum(2)/den[...,None,None]
    else:
        raise ValueError(f"Unknown acceleration mode: {acceleration}")
    valid=mask[:,None,:,None]
    return p*valid,v*valid,acc*valid,tb

def pair_physics(p: torch.Tensor, v: torch.Tensor, acc: torch.Tensor, mask: torch.Tensor):
    """Directed r(j->k)=p_k-p_j and derived quantities for all 4 blocks."""
    r=p[:,:,:,None,:]-p[:,:,None,:,:]  # p_j-p_k first
    r=-r                         # p_k-p_j
    u=v[:,:,:,None,:]-v[:,:,None,:,:]
    u=-u
    da=acc[:,:,:,None,:]-acc[:,:,None,:,:]
    da=-da
    d=r.square().sum(-1).add(1e-6).sqrt()
    dot=(r*u).sum(-1)
    c=-dot/d
    tau=(-dot/u.square().sum(-1).clamp_min(1e-4)).clamp(0,4)
    dcpa=(r+tau[...,None]*u).square().sum(-1).add(1e-6).sqrt()
    align=(v[:,:,:,None,:]*v[:,:,None,:,:]).sum(-1)/(v.square().sum(-1).sqrt()[:,:,:,None]*v.square().sum(-1).sqrt()[:,:,None,:]+.04)
    pair=mask[:,None,:,None]&mask[:,None,None,:]
    eye=torch.eye(mask.shape[1],device=mask.device,dtype=torch.bool)[None,None]
    pair=pair&~eye
    return dict(r=r,u=u,da=da,d=d,c=c,tau=tau,dcpa=dcpa,align=align,pair=pair)

def trc_edge_features(phys):
    c=phys['c'];dcp=phys['dcpa']
    dc=torch.zeros_like(c);dd=torch.zeros_like(dcp)
    dc[:,1:]=c[:,1:]-c[:,:-1];dd[:,1:]=dcp[:,1:]-dcp[:,:-1]
    e=torch.cat((phys['r']/30.,phys['u']/10.,phys['d'][...,None]/30.,c[...,None]/10.,phys['tau'][...,None]/4.,dcp[...,None]/10.,phys['da']/10.,dc[...,None]/10.,dd[...,None]/10.),-1)
    risk=torch.exp(-phys['d']/30.)*(.25+torch.sigmoid(c/2.))*torch.exp(-dcp/20.)/1.25
    return e*phys['pair'][...,None],risk*phys['pair']

def to_edge_features(phys):
    even=torch.stack((phys['d']/30.,phys['c']/10.,phys['tau']/4.,phys['dcpa']/10.,phys['u'].square().sum(-1).sqrt()/10.,phys['align']),-1)
    odd=torch.cat((phys['r']/30.,phys['u']/10.,phys['da']/5.),-1)
    risk=torch.exp(-phys['d']/30.)*(.25+.75*torch.sigmoid(phys['c']/2.))*torch.exp(-phys['dcpa']/10.)
    return even*phys['pair'][...,None],odd*phys['pair'][...,None],risk*phys['pair']

def to_patch_indices(risk: torch.Tensor, phys, mask: torch.Tensor, cap: int = 6, history: torch.Tensor | None = None):
    """Per-target root+top-risk sensed-history context. Fixed tensor width cap."""
    b,s,n,_=risk.shape;k=min(cap,n)
    score=.5*risk.max(1).values+.5*risk[:,-1]
    distance_sum=phys['d'].sum(1)
    canonical=(history.permute(0,2,1,3).reshape(b,n,-1) if history is not None else None)
    slots=[];valid=[]
    allidx=torch.arange(n,device=mask.device)
    def stable_sort(order,key,descending=False):
        return order.gather(1,torch.argsort(key.gather(1,order),dim=-1,descending=descending,stable=True))
    for root in range(n):
        ok=mask.clone();ok[:,root]=False
        order=allidx.expand(b,-1)
        # Full sensed history is the canonical final tie-break; exact ties are physically indistinguishable.
        if canonical is not None:
            for column in range(canonical.shape[-1]-1,-1,-1):
                order=stable_sort(order,canonical[...,column])
        order=stable_sort(order,distance_sum[:,root].masked_fill(~ok,float('inf')))
        order=stable_sort(order,score[:,root].masked_fill(~ok,-float('inf')),descending=True)
        neigh=order[:,:max(k-1,0)]
        nv=ok.gather(1,neigh)
        rootcol=torch.full((b,1),root,device=mask.device,dtype=torch.long)
        rvalid=mask[:,root:root+1]
        sl=torch.cat((rootcol,neigh),1);va=torch.cat((rvalid,nv),1)
        if k<cap:
            sl=torch.cat((sl,rootcol.expand(-1,cap-k)),1);va=torch.cat((va,torch.zeros(b,cap-k,device=mask.device,dtype=torch.bool)),1)
        slots.append(sl);valid.append(va)
    return torch.stack(slots,1),torch.stack(valid,1)

@lru_cache(maxsize=512)
def _pauli_map(q: int, spec: tuple[tuple[int,str], ...], device_str: str):
    device=torch.device(device_str)
    idx=torch.arange(2**q,device=device,dtype=torch.long)
    source=idx.clone();phase=torch.ones(2**q,device=device,dtype=torch.complex64)
    flip=0
    # output index i pulls from source i xor flips; phase must be evaluated on that source.
    for qubit,axis in spec:
        if axis in ('X','Y'): flip ^= 1 << (q-1-qubit)
    source=idx ^ flip
    for qubit,axis in spec:
        bit=(source >> (q-1-qubit)) & 1
        if axis=='Z': phase=phase*(1-2*bit).to(torch.complex64)
        elif axis=='Y': phase=phase*(1j*(1-2*bit).to(torch.float32)).to(torch.complex64)
        elif axis=='X': pass
        else: raise ValueError(axis)
    return source,phase

def apply_pauli_rotation(state: torch.Tensor, angle: torch.Tensor, spec):
    q=state.shape[-1].bit_length()-1
    tup=tuple((int(a),str(b)) for a,b in spec)
    source,phase=_pauli_map(q,tup,str(state.device))
    phase=phase.to(state.dtype)
    ps=state.index_select(-1,source)*phase
    c=torch.cos(angle/2).to(state.real.dtype)[...,None]
    s=torch.sin(angle/2).to(state.real.dtype)[...,None]
    return c*state-1j*s*ps

def expect_pauli(state: torch.Tensor, spec):
    q=state.shape[-1].bit_length()-1
    tup=tuple((int(a),str(b)) for a,b in spec)
    source,phase=_pauli_map(q,tup,str(state.device))
    ps=state.index_select(-1,source)*phase.to(state.dtype)
    return (state.conj()*ps).sum(-1).real

def apply_hadamard(state: torch.Tensor, qubit: int):
    q=state.shape[-1].bit_length()-1
    x=state.reshape(*state.shape[:-1],2**qubit,2,2**(q-qubit-1))
    a,b=x[...,0,:],x[...,1,:]
    y=torch.stack(((a+b)/math.sqrt(2),(a-b)/math.sqrt(2)),-2)
    return y.reshape_as(state)

def apply_hadamards(state: torch.Tensor, qubits):
    for q in qubits: state=apply_hadamard(state,int(q))
    return state

_PAULI2=None
def pauli2(device,dtype):
    global _PAULI2
    ctype=torch.complex128 if dtype==torch.float64 else torch.complex64
    I=torch.eye(2,device=device,dtype=ctype);X=torch.tensor([[0,1],[1,0]],device=device,dtype=ctype)
    Y=torch.tensor([[0,-1j],[1j,0]],device=device,dtype=ctype);Z=torch.tensor([[1,0],[0,-1]],device=device,dtype=ctype)
    kron=torch.kron
    return [kron(Z,I),kron(Y,I),kron(X,I),kron(I,Z),kron(I,Y),kron(I,X),kron(X,X),kron(Y,Y),kron(Z,Z),kron(X,Z)],kron(Y,X)

def local_twoqubit_unitary(angles: torch.Tensor, eta: torch.Tensor | None = None):
    mats,wmat=pauli2(angles.device,angles.dtype);ctype=mats[0].dtype
    shape=angles.shape[:-1];U=torch.eye(4,device=angles.device,dtype=ctype).expand(*shape,4,4).clone()
    I=torch.eye(4,device=angles.device,dtype=ctype)
    for d,P in enumerate(mats):
        a=angles[...,d];G=torch.cos(a/2)[...,None,None]*I-1j*torch.sin(a/2)[...,None,None]*P
        U=G@U
    if eta is not None:
        G=torch.cos(eta/2)[...,None,None]*I-1j*torch.sin(eta/2)[...,None,None]*wmat
        U=G@U
    return U

def apply_adjacent_twoqubit_unitary(state: torch.Tensor, U: torch.Tensor, node: int):
    q=state.shape[-1].bit_length()-1;pre=2**(2*node);post=2**(q-2*node-2)
    x=state.reshape(*state.shape[:-1],pre,4,post)
    y=torch.einsum('...uv,...pvq->...puq',U,x)
    return y.reshape_as(state)

def init_product_plus(valid: torch.Tensor, dtype: torch.dtype = torch.complex64):
    """valid [...,M], returns [...,4**M] with |++> for valid nodes, |00> for padding."""
    *lead,m=valid.shape
    state=torch.ones(*lead,1,device=valid.device,dtype=dtype)
    plus=torch.full((*lead,4),.5,device=valid.device,dtype=dtype)
    zero=torch.zeros(*lead,4,device=valid.device,dtype=dtype);zero[...,0]=1
    for j in range(m):
        loc=torch.where(valid[...,j,None],plus,zero)
        state=(state[..., :,None]*loc[...,None,:]).reshape(*lead,-1)
    return state

def parameter_summary(module: nn.Module):
    train=sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen=sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {'trainable':int(train),'frozen':int(frozen),'total':int(train+frozen)}
