"""Function-preserving message-basis extension of the strongest dual QGNN."""
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
from target_interaction_graph import ForecasterConfig,TargetInteractionGNN
from physics_aligned_dual_model import PhysicsAlignedMessageLayer

class WarmBasisLayer(PhysicsAlignedMessageLayer):
    def __init__(self,config,kind,role,core_seed):
        super().__init__(config,kind,role,core_seed)
        h=config.hidden_dim;self.bases=4
        # Rename the original value transform so old weights can be mapped explicitly.
        self.sender_value=nn.Linear(h,h,bias=False);del self.value
        self.difference_value=nn.Linear(h,h,bias=False);nn.init.zeros_(self.difference_value.weight)
        self.joint_value=nn.Linear(h,h,bias=False);nn.init.zeros_(self.joint_value.weight)
        self.basis_readout=nn.Linear(12,self.heads*self.bases,bias=False);nn.init.zeros_(self.basis_readout.weight)
        bias=torch.full((self.heads,self.bases),-3.0);bias[:,0]=3.0
        self.basis_bias=nn.Parameter(bias)
        self.risk_head=nn.Linear(12,1)
        self.last_risk_logits=None

    def forward(self,nodes,edge_features,adjacency):
        b,n,h=nodes.shape
        relations=self.relation_features(nodes,edge_features)
        selected=adjacency.reshape(-1).nonzero().squeeze(-1)
        angles=self.encode_angles(relations.reshape(-1,12).index_select(0,selected))
        latent=self.latent_norm(self.core(angles));total=b*n*n
        score=nodes.new_zeros(total,self.heads).index_copy(0,selected,self.score(latent)).view(b,n,n,self.heads)
        gate=nodes.new_zeros(total,self.heads).index_copy(0,selected,self.gate(latent)).view(b,n,n,self.heads)
        mix=nodes.new_zeros(total,self.heads,self.bases).index_copy(0,selected,self.basis_readout(latent).view(-1,self.heads,self.bases)).view(b,n,n,self.heads,self.bases)
        mix=mix+self.basis_bias[None,None,None]
        risk=nodes.new_zeros(total).index_copy(0,selected,self.risk_head(latent).squeeze(-1)).view(b,n,n);self.last_risk_logits=risk
        norm=self.value_norm(nodes);receiver=norm[:,:,None,:].expand(-1,-1,n,-1);sender=norm[:,None,:,:].expand(-1,n,-1,-1)
        edge=self.edge_value(edge_features)
        original=self.sender_value(norm)[:,None,:,:].expand(-1,n,-1,-1)+edge
        bases=[original,self.difference_value(sender-receiver),self.joint_value(sender*receiver),edge]
        stack=torch.stack([x.view(b,n,n,self.heads,self.head_dim) for x in bases],-2)
        message=(torch.softmax(mix,-1)[...,None]*stack).sum(-2)
        attention=torch.softmax(score.masked_fill(~adjacency[...,None],-1e4),2)
        aggregate=(self.dropout(attention)[...,None]*(2*torch.sigmoid(gate))[...,None]*message).sum(2).reshape(b,n,h)
        nodes=self.output_norm(nodes+torch.sigmoid(self.residual_logit)*self.dropout(aggregate))
        nodes=self.final_norm(nodes+self.dropout(self.ffn(nodes)))
        return nodes,attention.mean(-1)

def build_warm_basis_graph(arm,seed=2026):
    if arm not in ('warm_classical_dual','warm_quantum_dual'):raise ValueError(arm)
    kind='quantum' if arm=='warm_quantum_dual' else 'classical';torch.manual_seed(seed);cfg=ForecasterConfig();model=TargetInteractionGNN(cfg)
    torch.manual_seed(seed+3);model.graph_layers[0]=WarmBasisLayer(cfg,kind,'physical',seed+103)
    torch.manual_seed(seed+1);model.graph_layers[1]=WarmBasisLayer(cfg,kind,'contextual',seed+101)
    return model

def restore_strong(model,arm):
    source_arm='physics_quantum_dual' if arm=='warm_quantum_dual' else 'physics_classical_dual'
    path=BASE/'physics_aligned_dual_seed2026_v1'/(source_arm+'_graph_selected.pt')
    source=torch.load(path,map_location='cpu',weights_only=True)['state'];own=model.state_dict();loaded=[]
    for name,value in source.items():
        target=name.replace('.value.weight','.sender_value.weight') if '.value.weight' in name else name
        if target in own and own[target].shape==value.shape:own[target]=value;loaded.append(target)
    model.load_state_dict(own,strict=True);return path,len(loaded)
