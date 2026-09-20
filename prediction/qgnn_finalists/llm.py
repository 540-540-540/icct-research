from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from prediction.q0.temporal import FusedQKVLoRA,constant_velocity_baseline
from prediction.q0.motion_token_llm import MotionTokenGPT2Core
from prediction.qgnn_final.model import FinalMotionGPT2

class FinalistLLM(nn.Module):
    """Shared frozen GPT-2+LoRA motion-token reader with finalist-specific interaction tokens."""
    def __init__(self,mode:str,readout_dim:int,interaction_tokens:int,correction_cap_m:float=16.):
        super().__init__()
        if mode not in {'trc','toj'}: raise ValueError(mode)
        root=Path(__file__).resolve().parents[2]
        payload=json.loads((root/'configs/qgnn_final_tokens.json').read_text())
        self.mode=mode;self.readout_dim=int(readout_dim);self.interaction_tokens=int(interaction_tokens)
        self.base=FinalMotionGPT2(payload,correction_cap_m)
        self.base.graph_projection.requires_grad_(False)
        if mode=='toj': self.base.coordinate_head.requires_grad_(False)
        d=self.base.d_llm
        self.own_projection=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,d),nn.GELU())
        if mode=='trc':
            self.interaction_pre=nn.Sequential(nn.LayerNorm(readout_dim),nn.Linear(readout_dim,128),nn.GELU())
            self.interaction_out=nn.Sequential(nn.Linear(128,d),nn.GELU())
            self.type_embedding=nn.Parameter(torch.randn(2,d)*.02)
            self.time_embedding=nn.Parameter(torch.randn(4,d)*.02)
            self.head_context=nn.Sequential(nn.LayerNorm(160),nn.Linear(160,64),nn.GELU())
        else:
            self.interaction_pre=nn.Sequential(nn.LayerNorm(readout_dim),nn.Linear(readout_dim,d),nn.GELU())
            self.interaction_out=nn.Identity()
            self.time_embedding=nn.Parameter(torch.randn(4,d)*.02)
            self.head_context=None
            self.to_coordinate_head=nn.Sequential(nn.LayerNorm(d+2),nn.Linear(d+2,256),nn.GELU(),nn.Linear(256,2))
            nn.init.zeros_(self.to_coordinate_head[-1].weight);nn.init.zeros_(self.to_coordinate_head[-1].bias)
        self.correction_cap_m=float(correction_cap_m)
    @property
    def gpt2(self): return self.base.gpt2
    @property
    def tokenizer(self): return self.base.tokenizer
    def future_token_ids(self,history,future): return self.base.future_token_ids(history,future)
    def _motion(self,h):
        emb,_=self.base._history_motion_embeddings(h)
        return emb
    def forward(self,history,mask,readout,own,interaction_mask=None):
        b,t,n,_=history.shape;d=self.base.d_llm
        flat_hist=history.permute(0,2,1,3).reshape(b*n,t,4)
        flat_own=own.reshape(b*n,32)
        active=mask.reshape(-1).nonzero(as_tuple=True)[0]
        prediction=history.new_zeros((b*n,20,2));logits_full=history.new_zeros((b*n,20,self.base.vocab_size))
        if active.numel()==0:
            return {'prediction':prediction.reshape(b,n,20,2).permute(0,2,1,3),'token_logits':logits_full.reshape(b,n,20,-1).permute(0,2,1,3),'origin_eligible':mask}
        h=flat_hist[active];o=flat_own[active]
        own_tok=self.own_projection(o).unsqueeze(1)
        motion=self._motion(h)
        if self.mode=='trc':
            z=readout.reshape(b*n,8,self.readout_dim)[active]
            vm=(interaction_mask.reshape(b*n,8)[active] if interaction_mask is not None else torch.ones(z.shape[:2],device=z.device,dtype=torch.bool))
            a=self.interaction_pre(z)
            q=self.interaction_out(a)
            typ=torch.arange(2,device=z.device).repeat(4)
            tim=torch.arange(4,device=z.device).repeat_interleave(2)
            q=(q+self.type_embedding[typ][None]+self.time_embedding[tim][None])*vm[...,None]
            den=vm.sum(1,keepdim=True).clamp_min(1).to(q.dtype)
            ctx=(q.sum(1)/den);ap=(a*vm[...,None]).sum(1)/den
            hist_in=motion+ctx[:,None]
            hist_tok=hist_in+self.base.history_adapter(hist_in)
            inter_mask=vm
            seq=torch.cat((own_tok,q,hist_tok,self.base.future_queries[None].expand(len(active),-1,-1)),1)
            attn=torch.cat((torch.ones(len(active),1,device=z.device,dtype=torch.bool),inter_mask,torch.ones(len(active),39,device=z.device,dtype=torch.bool)),1)
            hidden=self.base.gpt2(inputs_embeds=seq,attention_mask=attn,use_cache=False).last_hidden_state[:,-20:]
            token_logits=self.base.token_head(hidden);prob=torch.softmax(token_logits,-1)
            ef,el=self.base.tokenizer.expected_motion(prob);em=torch.stack((ef,el),-1)
            hc=self.head_context(torch.cat((ap,o),-1))[:,None].expand(-1,20,-1)
            raw=self.base.coordinate_head(torch.cat((hidden,hc,em),-1))
            corr=torch.tanh(raw)*self.base.correction_scale_m
        else:
            z=readout.reshape(b*n,4,self.readout_dim)[active]
            q=self.interaction_out(self.interaction_pre(z))+self.time_embedding[None]
            hist_tok=motion+self.base.history_adapter(motion)
            seq=torch.cat((own_tok,q,hist_tok,self.base.future_queries[None].expand(len(active),-1,-1)),1)
            hidden=self.base.gpt2(inputs_embeds=seq,use_cache=False).last_hidden_state[:,-20:]
            token_logits=self.base.token_head(hidden);prob=torch.softmax(token_logits,-1)
            ef,el=self.base.tokenizer.expected_motion(prob);em=torch.stack((ef,el),-1)
            raw=self.to_coordinate_head(torch.cat((hidden,em),-1))*(4./self.correction_cap_m)
            corr=torch.tanh(raw)*self.correction_cap_m
        cv=constant_velocity_baseline(h.reshape(len(active),20,1,4))[:,:,0]
        prediction=prediction.index_copy(0,active,cv+corr);logits_full=logits_full.index_copy(0,active,token_logits)
        return {'prediction':prediction.reshape(b,n,20,2).permute(0,2,1,3),'token_logits':logits_full.reshape(b,n,20,-1).permute(0,2,1,3),'origin_eligible':mask}
