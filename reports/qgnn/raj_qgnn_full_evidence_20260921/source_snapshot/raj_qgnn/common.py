"""History-only features and deterministic bounded ego covers."""
from __future__ import annotations
import torch
from torch import nn
CAP = 8

def physical_graph(history, mask):
    def at(state):
        p, v = state[..., :2], state[..., 2:]
        r = p[:, None] - p[:, :, None]
        u = v[:, None] - v[:, :, None]
        d = r.square().sum(-1).clamp_min(1e-6).sqrt()
        dot = (r*u).sum(-1)
        c = -dot/d
        tau = (-dot/u.square().sum(-1).clamp_min(1e-4)).clamp(0, 4)
        dcpa = (r+tau[...,None]*u).square().sum(-1).clamp_min(1e-6).sqrt()
        speed = u.square().sum(-1).clamp_min(1e-6).sqrt()
        alignment = (v[:,:,None]*v[:,None]).sum(-1)
        e = torch.stack((d/30,c/10,tau/4,dcpa/10,speed/10,alignment/100,r[...,0].abs()/30,r[...,1].abs()/30),-1)
        risk = torch.exp(-d/30)*(0.25+torch.sigmoid(c/2))*torch.exp(-dcpa/20)
        return e, risk
    early, _ = at(history[:,:5].mean(1))
    recent, risk = at(history[:,-5:].mean(1))
    n = mask.shape[1]
    pair = mask[:,:,None] & mask[:,None,:] & ~torch.eye(n,device=mask.device,dtype=torch.bool)[None]
    return torch.cat((recent,recent-early),-1)*pair[...,None], risk*pair

class OwnHistoryEncoder(nn.Module):
    """Shared per-agent encoder; no cross-agent operation."""
    def __init__(self, width=32):
        super().__init__()
        self.input = nn.Linear(6,width)
        self.gru = nn.GRU(width,width,batch_first=True)
        self.norm = nn.LayerNorm(width)
    def forward(self, history, mask):
        b,t,n,_ = history.shape
        own = torch.cat(((history[...,:2]-history[:,-1:,:,:2])/10, history[...,2:]/10, history[:,-1:,:,:2].expand(-1,t,-1,-1)/50),-1)
        x = own.permute(0,2,1,3).reshape(b*n,t,6)
        x = torch.nn.functional.silu(self.input(x))
        _, h = self.gru(x)
        return self.norm(h[0]).reshape(b,n,-1)*mask[...,None]

def patch_inputs(history, mask):
    """Full N<=8 graph, otherwise one history-ranked 8-node ego patch per target."""
    b,t,n,c = history.shape
    if n <= CAP:
        return history, mask, None
    with torch.no_grad():
        _, risk = physical_graph(history, mask)
        canonical = torch.arange(n,device=history.device)[None].expand(b,-1)
        key = history.permute(0,2,1,3).reshape(b,n,-1)
        for k in range(key.shape[-1]-1,-1,-1):
            order = torch.argsort(key[...,k].gather(1,canonical),dim=1,stable=True)
            canonical = canonical.gather(1,order)
        canonical = canonical[:,None].expand(-1,n,-1)
        valid = mask[:,None].expand(-1,n,-1) & ~torch.eye(n,device=mask.device,dtype=torch.bool)[None]
        score = risk.masked_fill(~valid,-float('inf'))
        ranked = torch.argsort(score.gather(2,canonical),dim=2,descending=True,stable=True)
        neighbors = canonical.gather(2,ranked)[...,:CAP-1]
        root = torch.arange(n,device=mask.device)[None,:,None].expand(b,-1,-1)
        slots = torch.cat((root,neighbors),-1)
    agents = history.permute(0,2,1,3)
    batch = torch.arange(b,device=mask.device)[:,None,None]
    hp = agents[batch,slots].reshape(b*n,CAP,t,c).permute(0,2,1,3)
    # A filler can have the root index; it must not become another active ego.
    neighbor_valid = valid.gather(2,neighbors)
    mp = torch.cat((mask[:,:,None],neighbor_valid),-1).reshape(b*n,CAP)
    hp = torch.where(mp[:,None,:,None],hp,0.)
    return hp,mp,(b,n)

class BoundedCore(nn.Module):
    def forward(self, history, mask):
        if history.ndim!=4 or history.shape[1]!=20 or history.shape[-1]!=4:
            raise ValueError('Expected [B,20,N,4]')
        if mask.shape!=(history.shape[0],history.shape[2]) or mask.dtype!=torch.bool:
            raise ValueError('Expected boolean mask [B,N]')
        history = torch.where(mask[:,None,:,None],history,0.)
        hp,mp,restore = patch_inputs(history,mask)
        z = self.forward_patch(hp,mp)
        if restore is not None:
            z = z[:,0].reshape(*restore,-1)
        return z*mask[...,None]
