"""Core message replacement, not a parallel residual adapter."""
import sys, math
from pathlib import Path
import torch
from torch import nn
ROOT=Path('/home/js_cn/sensing')
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from target_interaction_graph import TargetInteractionGNN, ForecasterConfig
from quantum_core import BatchedStatevectorPQC, QuantumCircuitConfig
from MultiTargetTimeLLM import MultiTargetGraphLLM

class ClassicalCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(6,6),nn.SiLU(),nn.Linear(6,6))
    def forward(self,x):
        y=torch.tanh(self.layers(x))
        return torch.cat([y,torch.sin(math.pi*y)],-1)

class CoreMessageLayer(nn.Module):
    def __init__(self,config,kind,quantum_backend='torch'):
        super().__init__()
        h=config.hidden_dim
        self.heads=config.graph_heads; self.head_dim=h//self.heads
        self.input_norm=nn.LayerNorm(2*h+7)
        self.encoder=nn.Linear(2*h+7,6)
        # Core initialization must not advance the RNG used by shared readouts.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2028)
            if kind=='quantum' and quantum_backend=='torch':
                self.core=BatchedStatevectorPQC(QuantumCircuitConfig())
            elif kind=='quantum' and quantum_backend=='pennylane':
                # Lazy import keeps the retained default model usable without PennyLane.
                from pennylane_core import PennyLaneStatevectorPQC
                self.core=PennyLaneStatevectorPQC(QuantumCircuitConfig())
            elif kind=='quantum':
                raise ValueError(f'Unknown quantum backend: {quantum_backend}')
            else:
                self.core=ClassicalCore()
        self.latent_norm=nn.LayerNorm(12)
        self.score=nn.Linear(12,self.heads,bias=False)
        self.gate=nn.Linear(12,self.heads,bias=False)
        self.value_norm=nn.LayerNorm(h)
        self.value=nn.Linear(h,h,bias=False)
        self.edge_value=nn.Sequential(nn.Linear(7,h),nn.SiLU(),nn.Linear(h,h,bias=False))
        self.output_norm=nn.LayerNorm(h)
        self.ffn=nn.Sequential(nn.Linear(h,2*h),nn.SiLU(),nn.Dropout(config.dropout),nn.Linear(2*h,h))
        self.final_norm=nn.LayerNorm(h)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,nodes,edge_features,adjacency):
        b,n,h=nodes.shape
        pairs=torch.cat([nodes[:,:,None,:].expand(-1,-1,n,-1),nodes[:,None,:,:].expand(-1,n,-1,-1),edge_features],-1)
        selected=adjacency.reshape(-1).nonzero().squeeze(-1)
        x=pairs.reshape(-1,2*h+7).index_select(0,selected)
        angles=math.pi*torch.tanh(self.encoder(self.input_norm(x)))
        z=self.latent_norm(self.core(angles))
        value=self.value(self.value_norm(nodes)).view(b,n,self.heads,self.head_dim)
        edge_value=self.edge_value(edge_features).view(b,n,n,self.heads,self.head_dim)
        base_msg=value[:,None,:,:,:]+edge_value
        score=nodes.new_zeros(b*n*n,self.heads).index_copy(0,selected,self.score(z)).view(b,n,n,self.heads)
        # The core is the sole source of attention logits and pairwise message gates.
        gate=nodes.new_zeros(b*n*n,self.heads).index_copy(0,selected,self.gate(z)).view(b,n,n,self.heads)
        gate=2.0*torch.sigmoid(gate)
        attn=torch.softmax(score.masked_fill(~adjacency[...,None],-1e4),dim=2)
        aggregate=(self.dropout(attn)[...,None]*gate[...,None]*base_msg).sum(2).reshape(b,n,h)
        # No classical value/edge-value bypass and no zero scalar message gate.
        nodes=self.output_norm(nodes+self.dropout(aggregate))
        nodes=self.final_norm(nodes+self.dropout(self.ffn(nodes)))
        return nodes,attn.mean(-1)

def build_graph(arm,seed=2026,quantum_backend='torch'):
    # Reset before shared and replacement initialization; match all common tensors.
    torch.manual_seed(seed)
    config=ForecasterConfig()
    model=TargetInteractionGNN(config)
    if arm!='plain':
        torch.manual_seed(seed+1)
        model.graph_layers[1]=CoreMessageLayer(config,arm,quantum_backend=quantum_backend)
    return model

class CoreGraphLLM(MultiTargetGraphLLM):
    def __init__(self,graph):
        super().__init__(graph,graph.config,llm_layers=4,lora_rank=8)
        for p in self.graph_backbone.parameters(): p.requires_grad_(True)
    def forward(self,history,target_mask):
        # The graph participates in final trajectory/token losses; no no_grad.
        b,t,n,_=history.shape
        g=self.graph_backbone(history,target_mask)
        nodes=g['node_features'].reshape(b*n,-1)
        target_history=history.permute(0,2,1,3).reshape(b*n,t,4)
        soft=self._soft_history_embeddings(target_history,nodes)
        graph_token=self.graph_projection(nodes).unsqueeze(1)
        motion=soft['embeddings']+graph_token
        seq=torch.cat([graph_token,motion+self.token_adapter(motion)],1)
        hidden=self.gpt2(inputs_embeds=seq).last_hidden_state[:,-1]
        future_hidden=hidden[:,None,:]+self.future_queries[None,:,:]
        logits=self.token_head(future_hidden).reshape(b,n,self.config.prediction_length,self.vocab_size).permute(0,2,1,3)
        correction=self.coordinate_head(torch.cat([hidden,nodes],-1))
        correction=torch.tanh(correction.view(b,n,self.config.prediction_length,2))*self.correction_scale_m
        correction=correction.permute(0,2,1,3)*target_mask[:,None,:,None]
        future=g['future_position']+correction
        return dict(future_position=future,graph_future_position=g['future_position'],token_logits=logits,
                    token_temperature=soft['temperature'],node_features=g['node_features'],attention=g['attention'],adjacency=g['adjacency'])

def compact_state(model):
    return {n:t.detach().cpu().clone() for n,t in model.state_dict().items() if not n.startswith('gpt2.') or 'lora_' in n}

def restore_compact(model,state):
    missing,extra=model.load_state_dict(state,strict=False)
    assert not extra and all(n.startswith('gpt2.') and 'lora_' not in n for n in missing)
