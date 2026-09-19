"""Shared motion/interaction-token GPT-2 interface for the final core comparison."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from prediction.q0.motion_token_llm import AutomatumMotionTokenizer,AutomatumMotionTokenizerConfig,MotionTokenGPT2Core
from prediction.q0.graph_motion_llm import GraphMotionLLM
from prediction.q0.contracts import CANONICAL_DT
from .quantum import HypergraphQuantumCore
from .classical import AdaptiveClassicalCore

class NonuniformTokenizer(AutomatumMotionTokenizer):
    def __init__(self, payload):
        self.forward_centers=payload['forward_centers'];self.lateral_centers=payload['lateral_centers']
        self.config=AutomatumMotionTokenizerConfig(forward_bins=len(self.forward_centers),lateral_bins=len(self.lateral_centers),forward_min=self.forward_centers[0],forward_max=self.forward_centers[-1],lateral_min=self.lateral_centers[0],lateral_max=self.lateral_centers[-1],temperature_m=.12)
    def axis_tables(self,device,dtype):
        return (torch.tensor(self.forward_centers,device=device,dtype=dtype),torch.tensor(self.lateral_centers,device=device,dtype=dtype))
    def _tables(self,device,dtype):
        f,l=self.axis_tables(device,dtype)
        return f[:,None].expand(-1,len(l)).reshape(-1),l[None].expand(len(f),-1).reshape(-1)
    def corners(self,f,l):
        ft,lt=self.axis_tables(f.device,f.dtype)
        def indices(x,table):
            right=torch.searchsorted(table,x.contiguous()).clamp(1,len(table)-1)
            left=right-1
            frac=((x-table[left])/(table[right]-table[left])).clamp(0,1)
            return left,right,frac,x-x.clamp(table[0],table[-1])
        fl,fr,fw,fo=indices(f,ft);ll,lr,lw,lo=indices(l,lt)
        ids=torch.stack((fl*len(lt)+ll,fl*len(lt)+lr,fr*len(lt)+ll,fr*len(lt)+lr),-1)
        weight=torch.stack(((1-fw)*(1-lw),(1-fw)*lw,fw*(1-lw),fw*lw),-1)
        return ids,weight,torch.stack((fo,lo),-1)
    def soft_probabilities(self,positions,velocities):
        f,l=self.local_deltas(positions,velocities)
        ids,weight,_=self.corners(f,l)
        return f.new_zeros(*f.shape,self.config.vocab_size).scatter_add(-1,ids,weight)
    def token_ids(self,positions,velocities):
        f,l=self.local_deltas(positions,velocities)
        ft,lt=self.axis_tables(f.device,f.dtype)
        return (f[...,None]-ft).abs().argmin(-1)*len(lt)+(l[...,None]-lt).abs().argmin(-1)

class ScaledCoordinateHead(nn.Sequential):
    """Preserve the local gradient while extending the shared residual range."""
    def __init__(self,*modules,scale=1.):
        super().__init__(*modules);self.output_scale=float(scale)
    def forward(self,x):
        return super().forward(x)*self.output_scale

class FinalMotionGPT2(MotionTokenGPT2Core):
    def __init__(self,payload,correction_cap_m=16.):
        tokenizer=NonuniformTokenizer(payload)
        if correction_cap_m<4.: raise ValueError("correction_cap_m must be at least4m")
        super().__init__(graph_dim=64,llm_layers=4,lora_rank=8,tokenizer_config=tokenizer.config,correction_scale_m=correction_cap_m)
        self.coordinate_head=ScaledCoordinateHead(*list(self.coordinate_head.children()),scale=4./correction_cap_m)
        self.tokenizer=tokenizer
        self.continuous_motion=nn.Linear(4,self.d_llm,bias=False)
        nn.init.normal_(self.continuous_motion.weight,std=.01)
    def _history_motion_embeddings(self,history):
        f,l=self.tokenizer.local_deltas(history[...,:2],history[...,2:])
        ids,weight,overflow=self.tokenizer.corners(f,l)
        table=torch.nn.functional.layer_norm(self.motion_embedding.weight,(self.d_llm,))
        embedded=(table[ids]*weight[...,None]).sum(-2)
        raw=torch.cat((torch.stack((f,l),-1)/4,overflow/4),-1)
        return embedded+self.continuous_motion(raw),None

class LegacyHigherOrder(nn.Module):
    """Unweakened previous 64-wide, per-frame pair+triplet core as extra control."""
    def __init__(self,snr_db=0.):
        super().__init__()
        from prediction.q0.graph import GatedPairTriplet
        from prediction.q0.normalization import load_normalization,normalization_path
        self.inner=GatedPairTriplet()
        self.stats=load_normalization(normalization_path(Path(__file__).resolve().parents[2],snr_db,'sind'))
    def forward(self,history,mask):
        from prediction.q0.features import normalize_state_and_edges
        state,edge,pair=normalize_state_and_edges(history,mask,self.stats)
        return self.inner(state,edge,pair,mask)[:,-1]

class FinalModel(nn.Module):
    def __init__(self,core,llm):
        super().__init__();self.graph=core;self.llm=llm
    def forward(self,history,mask,timestamps=None):
        graph=self.graph(history,mask)
        out=self.llm(history,mask,graph[:,None].expand(-1,20,-1,-1))
        if timestamps is not None:
            dt=((timestamps[:,-1]-timestamps[:,0])/19).to(history.dtype)
            step=torch.arange(1,21,device=history.device,dtype=history.dtype)[None,:,None,None]
            correction=step*(dt-CANONICAL_DT)[:,None,None,None]*history[:,-1:, :,2:]
            out['prediction']=out['prediction']+correction*mask[:,None,:,None]
        out['graph_features']=graph
        return out
    def parameter_summary(self):
        return GraphMotionLLM.parameter_summary(self)

def build_model(kind,seed=2026,depth=3,token_path=None,channels=4,enhanced=True,snr_db=0.,quantum_version=3,correction_cap_m=16.):
    root=Path(__file__).resolve().parents[2]
    payload=json.loads(Path(token_path or root/'configs/qgnn_final_tokens.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        if kind=='quantum':
            if quantum_version==3:
                from .relational import RelationCarryingQuantumCore
                core=RelationCarryingQuantumCore(depth,channels)
            else: core=HypergraphQuantumCore(depth,channels,enhanced)
        elif kind=='classical': core=AdaptiveClassicalCore(depth,channels)
        elif kind=='legacy_classical': core=LegacyHigherOrder(snr_db)
        else: raise ValueError(kind)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+300003)
        llm=FinalMotionGPT2(payload,correction_cap_m)
    return FinalModel(core,llm)
