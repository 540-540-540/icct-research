"""Light history-only coupling controller; no learned cross-node message passing."""
from __future__ import annotations
import math
import torch
from torch import nn
from prediction.qgnn_final.quantum import tables

def masked_stats(x, valid):
    den=valid.sum(1,keepdim=True).clamp_min(1).to(x.dtype)
    mean=(x*valid[...,None]).sum(1)/den
    var=((x-mean[:,None]).square()*valid[...,None]).sum(1)/den
    std=(var+1e-6).sqrt()*(valid.any(1)[:,None])
    return torch.cat((mean,std),-1)

def descriptors(edge, risk, mask):
    b,n=mask.shape
    _,_,pairs,tri,_,_=tables(n,edge.device,edge.dtype)
    pe=edge[:,pairs[:,0],pairs[:,1]]
    pv=mask[:,pairs[:,0]]&mask[:,pairs[:,1]]
    pw=risk[:,pairs[:,0],pairs[:,1]]
    a,j,k=tri.unbind(-1)
    es=torch.stack((edge[:,a,j],edge[:,a,k],edge[:,j,k]),-2)
    tf=torch.cat((es.mean(-2),es.var(-2,unbiased=False).add(1e-6).sqrt()),-1) if len(tri) else edge.new_zeros(b,0,32)
    tv=mask[:,a]&mask[:,j]&mask[:,k]
    x,y,z=risk[:,a,j],risk[:,a,k],risk[:,j,k]
    tw=(x*y+x*z+y*z)/3*tv
    scene=torch.cat((masked_stats(pe,pv),masked_stats(pw[...,None],pv),masked_stats(tw[...,None],tv)),-1)
    return dict(pe=pe,tf=tf,pw=pw,tw=tw,pv=pv,tv=tv,scene=scene,pairs=pairs,tri=tri)

class CouplingController(nn.Module):
    """Only generates bounded interaction multipliers; no path to prediction head."""
    def __init__(self,depth=3,channels=4):
        super().__init__();self.depth=depth;self.channels=channels
        self.scene_encoder=nn.Linear(36,16)
        self.scene_head=nn.Linear(16,2*channels*depth)
        self.query=nn.Linear(16,16)
        self.pair_key=nn.Linear(16,8,bias=False)
        self.triple_key=nn.Linear(32,8,bias=False)
        self.relation_gain=nn.Parameter(torch.full((2,channels,depth),.10))
        self.feedback_global=nn.Linear(4*channels,2*channels*depth,bias=False)
        self.feedback_local=nn.Parameter(torch.full((2,channels,depth,2),.05))
        nn.init.zeros_(self.scene_head.weight);nn.init.zeros_(self.feedback_global.weight)
        # Biases seed mild order/channel specialization, but are freely trainable.
        prior2=torch.linspace(1.10,.80,channels)[:,None].expand(-1,depth)
        prior3=torch.linspace(.75,1.20,channels)[:,None]*torch.linspace(.9,1.1,depth)[None]
        prior=torch.stack((prior2,prior3)).clamp(.3,1.7)
        with torch.no_grad():self.scene_head.bias.copy_(torch.atanh((prior-1)/.8).flatten())
        self.enabled=True

    def prepare(self,edge,risk,mask):
        d=descriptors(edge,risk,mask)
        s=torch.nn.functional.silu(self.scene_encoder(torch.asinh(d["scene"])))
        query=self.query(s).reshape(s.shape[0],2,8)
        d["base"]=self.scene_head(s).reshape(-1,2,self.channels,self.depth)
        d["rel2"]=(torch.tanh(self.pair_key(d["pe"]))*query[:,0,None]).sum(-1)/math.sqrt(8)
        d["rel3"]=(torch.tanh(self.triple_key(d["tf"]))*query[:,1,None]).sum(-1)/math.sqrt(8)
        return d

    def forward(self,layer,d,feedback=None):
        b=d["base"].shape[0];ch=self.channels
        if not self.enabled:
            return d["pe"].new_ones(b,ch,d["pe"].shape[1]),d["tf"].new_ones(b,ch,d["tf"].shape[1])
        fglobal=d["base"].new_zeros(b,2,ch)
        if feedback is not None:
            def avg(v,w):return (v*w[:,None]).sum(-1)/w.sum(-1)[:,None].clamp_min(1e-6)
            f2,f3=feedback
            summary=torch.cat((avg(f2,d["pw"]),avg(f2.square(),d["pw"]),avg(f3,d["tw"]),avg(f3.square(),d["tw"])),-1)
            fglobal=self.feedback_global(summary).reshape(b,2,ch,self.depth)[...,layer]
        out=[]
        for order,key in enumerate(["rel2","rel3"]):
            logit=d["base"][:,order,:,layer,None]+fglobal[:,order,:,None]+self.relation_gain[order,:,layer][None,:,None]*d[key][:,None]
            if feedback is not None:
                v=feedback[order];gain=self.feedback_local[order,:,layer]
                logit=logit+gain[None,:,0,None]*v+gain[None,:,1,None]*v.square()
            out.append(1+.8*torch.tanh(logit))
        return tuple(out)


def node_feedback(feedback,d,n):
    """Four incident-relation response moments per node and channel."""
    pooled=[]
    for val,weight,indices in zip(feedback,[d["pw"],d["tw"]],[d["pairs"],d["tri"]]):
        inc=torch.nn.functional.one_hot(indices,num_classes=n).sum(1).to(val.dtype)
        den=(weight@inc).clamp_min(1e-6)[:,None]
        pooled.extend(((val*weight[:,None])@inc/den,(val.square()*weight[:,None])@inc/den))
    return torch.stack(pooled,-1)
