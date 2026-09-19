"""Unweakened R2 higher-order attention plus matched scene/relation feedback."""
from __future__ import annotations
import torch
from prediction.qgnn_final.classical import AdaptiveClassicalCore,triplet_indices
from prediction.qgnn_final.relational import rooted_to_global
from prediction.qgnn_final.common import physical_graph
from .controller import CouplingController

def modulated_attention(value,score,weight,valid,multiplier):
    heads=score.shape[-1];width=value.shape[-1]
    score=score+torch.log(weight.clamp_min(1e-4))[...,None]+torch.log(multiplier)
    score=score.masked_fill(~valid[...,None],-1e9)
    alpha=torch.softmax(score,-2)*valid[...,None]
    alpha=alpha/alpha.sum(-2,keepdim=True).clamp_min(1e-8)
    values=value.reshape(*value.shape[:-1],heads,width//heads)
    pooled=(alpha[...,None]*values).sum(-3)
    # Global scene gains must not cancel inside normalized attention.
    w=weight*valid;gain=(multiplier*w[...,None]).sum(-2)/w.sum(-1,keepdim=True).clamp_min(1e-6)
    return (pooled*gain[...,None]).flatten(-2)

def adaptive_layer(layer,h,edge,risk,mask,d,l2,l3):
    b,n,w=h.shape;ch=l2.shape[1];pairs=d["pairs"]
    hi=h[:,:,None].expand(-1,-1,n,-1);hj=h[:,None].expand(-1,n,-1,-1)
    p=layer.pair(torch.cat((hi,hj,edge),-1))
    valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,device=h.device,dtype=torch.bool)[None]
    lm=h.new_ones(b,n,n,ch)
    lm[:,pairs[:,0],pairs[:,1]]=l2.transpose(1,2);lm[:,pairs[:,1],pairs[:,0]]=l2.transpose(1,2)
    pair=modulated_attention(p[...,:w],p[...,w:],risk,valid,lm)
    f2=torch.tanh((p[:,pairs[:,0],pairs[:,1],w:]+p[:,pairs[:,1],pairs[:,0],w:])/2).transpose(1,2)
    trip=torch.zeros_like(pair);f3=h.new_zeros(b,ch,d["tri"].shape[0])
    if n>=3:
        indices=triplet_indices(n,h.device);j,k=indices.unbind(-1);i=torch.arange(n,device=h.device)[:,None]
        hi=h[:,:,None].expand(-1,-1,j.shape[1],-1);hj,hk=h[:,j],h[:,k]
        eij,eik,ejk=edge[:,i,j],edge[:,i,k],edge[:,j,k]
        tr=layer.triplet(torch.cat((hi,hj+hk,(hj-hk).abs(),eij+eik,(eij-eik).abs(),ejk),-1))
        r1,r2,r3=risk[:,i,j],risk[:,i,k],risk[:,j,k];weights=(r1*r2+r1*r3+r2*r3)/3
        vt=mask[:,:,None]&mask[:,j]&mask[:,k]
        mapping=rooted_to_global(n,h.device)
        mods=l3[:,:,mapping].permute(0,2,3,1)
        trip=modulated_attention(tr[...,:w],tr[...,w:],weights,vt,mods)
        sums=h.new_zeros(b,d["tri"].shape[0],ch).index_add(1,mapping.flatten(),(tr[...,w:]*vt[...,None]).reshape(b,-1,ch))
        counts=h.new_zeros(b,d["tri"].shape[0]).index_add(1,mapping.flatten(),vt.to(h.dtype).reshape(b,-1))
        f3=torch.tanh(sums/counts.clamp_min(1)[...,None]).transpose(1,2)
    joined=torch.cat((h,pair,trip),-1);gates=torch.softmax(layer.mix(joined),-1)
    delta=layer.update(torch.cat((gates[...,0:1]*h,gates[...,1:2]*pair,gates[...,2:3]*trip),-1))
    return layer.norm(h+delta)*mask[...,None],(f2,f3)

class SceneAdaptiveClassicalCore(AdaptiveClassicalCore):
    def __init__(self,depth=3,channels=4,controller_seed=802029,feedback=True):
        super().__init__(depth,channels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(controller_seed);self.controller=CouplingController(depth,channels)
        self.feedback_enabled=feedback;self.depth=depth;self.channels=channels
    def forward_patch(self,history,mask):
        own=self.encoder(history,mask);edge,risk=physical_graph(history,mask)
        d=self.controller.prepare(edge,risk,mask);h=self.input(own)*mask[...,None]
        feedback=None;layers=[]
        for l,layer in enumerate(self.layers):
            l2,l3=self.controller(l,d,feedback if self.feedback_enabled else None)
            h,feedback=adaptive_layer(layer,h,edge,risk,mask,d,l2,l3);layers.append(h)
        joint=torch.cat(layers,-1);gate=torch.sigmoid(self.gate(torch.cat((own,joint),-1)))
        return (self.local(own)+gate*self.readout(joint))*mask[...,None]
