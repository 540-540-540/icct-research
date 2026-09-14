"""QGNN-conditioned trust calibration for an LLM trajectory correction."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
from future_token_model import FutureTokenCoreGraphLLM

class InteractionTrustFutureLLM(FutureTokenCoreGraphLLM):
    """QGNN says who matters; the joint state decides how much to trust the LLM."""
    def __init__(self,graph):
        super().__init__(graph);self._trust_attention=[]
        self._trust_hooks=[x.register_forward_hook(self._capture) for x in self.graph_backbone.graph_layers]
        w=128
        self.trust_projection=nn.Sequential(nn.LayerNorm(self.d_llm),nn.Linear(self.d_llm,w),nn.GELU())
        # 512 routed interaction features + 4 QGNN confidence statistics +
        # 3 graph/LLM correction statistics + token entropy + horizon.
        self.trust_head=nn.Sequential(nn.LayerNorm(4*w+9),nn.Linear(4*w+9,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.trust_head[-1].weight);nn.init.zeros_(self.trust_head[-1].bias)
        self.trust_attention_mode='learned';self.last_trust_mean=None
    def _capture(self,module,inputs,output):
        relation=module.last_effective_relation if getattr(self,'use_effective_relation',False) else output[1]
        self._trust_attention.append(relation)
    def _weights(self,a,adj):
        n=a.shape[1];mask=adj&~torch.eye(n,dtype=torch.bool,device=a.device)[None]
        if self.trust_attention_mode=='off':z=torch.zeros_like(a)
        elif self.trust_attention_mode=='uniform':z=mask.to(a.dtype)
        else:z=a*mask
        return z/z.sum(-1,keepdim=True).clamp_min(1e-6)
    def forward(self,history,target_mask):
        self._trust_attention=[];o=super().forward(history,target_mask);assert len(self._trust_attention)==2
        b,_,n,_=history.shape;t=self.config.prediction_length
        own=self.trust_projection(o['future_hidden_features'].view(b,n,t,self.d_llm));parts=[];confidence=[]
        for a in self._trust_attention:
            w=self._weights(a,o['adjacency']);neighbor=torch.einsum('bij,bjtd->bitd',w,own);active=(w.sum(-1)>0)[:,:,None,None];d=(neighbor-own)*active;parts.extend([d,own*d])
            entropy=-(w*w.clamp_min(1e-8).log()).sum(-1)/torch.log(torch.tensor(float(max(n-1,2)),device=w.device));peak=w.max(-1).values
            confidence.extend([entropy[:,:,None,None].expand(-1,-1,t,-1),peak[:,:,None,None].expand(-1,-1,t,-1)])
        base=o['future_position'];graph=o['graph_future_position'];correction=base-graph
        correction_nt=correction.permute(0,2,1,3)/.75;correction_features=torch.cat([correction_nt,correction_nt.norm(dim=-1,keepdim=True)],-1)
        probs=torch.softmax(o['token_logits'],dim=-1);token_entropy=-(probs*probs.clamp_min(1e-8).log()).sum(-1)/torch.log(torch.tensor(float(probs.shape[-1]),device=probs.device));token_entropy=token_entropy.permute(0,2,1)[:,:,:,None]
        horizon=torch.linspace(0.,1.,t,device=history.device)[None,None,:,None].expand(b,n,-1,-1)
        joint=torch.cat(parts+confidence+[correction_features,token_entropy,horizon],-1)
        if getattr(self,'segment_level',False):
            pooled=torch.cat([x.mean(dim=2) for x in torch.tensor_split(joint,self.segment_count,dim=2)],dim=-1)
            segment_logits=self.trust_segment_head(pooled);segment_gain=1.+torch.tanh(segment_logits)
            lengths=[x.shape[2] for x in torch.tensor_split(joint,self.segment_count,dim=2)]
            logits=torch.cat([segment_logits[:,:,k,None].expand(-1,-1,lengths[k]) for k in range(self.segment_count)],dim=2)
            gain=torch.cat([segment_gain[:,:,k,None].expand(-1,-1,lengths[k]) for k in range(self.segment_count)],dim=2)
        elif getattr(self,'target_level',False):
            pooled=.5*joint.mean(dim=2)+.5*joint[:,:,-1]
            target_logits=self.trust_head(pooled).squeeze(-1);target_gain=1.+torch.tanh(target_logits)
            logits=target_logits[:,:,None].expand(-1,-1,t);gain=target_gain[:,:,None].expand(-1,-1,t)
        else:logits=self.trust_head(joint).squeeze(-1);gain=1.+torch.tanh(logits)
        fused=graph+gain.permute(0,2,1)[...,None]*correction;fused=fused*target_mask[:,None,:,None]
        o['ungated_future_position']=base;o['future_position']=fused;o['displacement']=fused-history[:,-1,None,:,:2]
        o['trust_logits']=logits.permute(0,2,1);o['trust_gain']=gain.permute(0,2,1)
        if getattr(self,'segment_level',False):o['segment_gain']=segment_gain
        self.last_trust_mean=gain.detach().mean();return o

class TargetInteractionTrustFutureLLM(InteractionTrustFutureLLM):
    """Use one stable QGNN-conditioned trust value for each target trajectory."""
    def __init__(self,graph):super().__init__(graph);self.target_level=True

class SegmentInteractionTrustFutureLLM(InteractionTrustFutureLLM):
    """Three stable temporal trust values per target: early, middle, and late."""
    def __init__(self,graph,segments=3):
        super().__init__(graph);self.segment_level=True;self.segment_count=int(segments);d=4*128+9
        self.trust_segment_head=nn.Sequential(nn.LayerNorm(d*self.segment_count),nn.Linear(d*self.segment_count,256),nn.GELU(),nn.Linear(256,self.segment_count))
        nn.init.zeros_(self.trust_segment_head[-1].weight);nn.init.zeros_(self.trust_segment_head[-1].bias)

class EffectiveSegmentInteractionTrustFutureLLM(SegmentInteractionTrustFutureLLM):
    """Route LLM states by the QGNN's effective alpha-times-gate influence."""
    def __init__(self,graph,segments=3):super().__init__(graph,segments);self.use_effective_relation=True
