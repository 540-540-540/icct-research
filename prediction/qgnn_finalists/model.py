from __future__ import annotations
import torch
from torch import nn
from prediction.q0.contracts import CANONICAL_DT
from .common import parameter_summary
from .llm import FinalistLLM
from .trc import TRCQuantumCore,RTCNCore
from .toj import TOJQGNNCore,TRTGNCore

class FinalistModel(nn.Module):
    def __init__(self,core:nn.Module,llm:FinalistLLM,name:str):
        super().__init__();self.core=core;self.llm=llm;self.name=name
    def forward(self,history,mask,timestamps=None):
        history=torch.where(mask[:,None,:,None],history,0.)
        core=self.core(history,mask,timestamps)
        out=self.llm(history,mask,core['readout'],core['own'],core.get('interaction_mask'))
        if timestamps is not None:
            dt=((timestamps[:,-1]-timestamps[:,0])/19).to(history.dtype)
            step=torch.arange(1,21,device=history.device,dtype=history.dtype)[None,:,None,None]
            correction=step*(dt-CANONICAL_DT)[:,None,None,None]*history[:,-1:,:,2:]
            out['prediction']=out['prediction']+correction*mask[:,None,:,None]
        out['interaction_readout']=core['readout'];return out
    def parameter_summary(self):
        out=parameter_summary(self);out['core_trainable']=sum(p.numel() for p in self.core.parameters() if p.requires_grad)
        out['llm_trainable']=sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        return out

def build_model(kind:str,seed:int=2026,correction_cap_m:float=16.,classical_width:int=64,use_checkpoint:bool=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        if kind=='trc': core=TRCQuantumCore(use_checkpoint=use_checkpoint);mode='trc';dim=35;tokens=8
        elif kind=='rtcn': core=RTCNCore(width=classical_width);mode='trc';dim=35;tokens=8
        elif kind=='toj': core=TOJQGNNCore(cap=6,use_checkpoint=use_checkpoint);mode='toj';dim=31;tokens=4
        elif kind=='trtgn': core=TRTGNCore(cap=6,width=32,rank=16);mode='toj';dim=31;tokens=4
        else: raise ValueError(kind)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+300003)
        llm=FinalistLLM(mode,dim,tokens,correction_cap_m)
    return FinalistModel(core,llm,kind)
