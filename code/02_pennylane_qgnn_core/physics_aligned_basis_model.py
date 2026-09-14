"""Hierarchical dual QGNN whose quantum outputs also mix message content bases."""
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))

from target_interaction_graph import ForecasterConfig, TargetInteractionGNN
from physics_aligned_dual_model import DualAxisPennyLanePQC, LowRankClassicalCore


class BasisMixMessageLayer(nn.Module):
    """Use a 12-value core latent for attention, gating, basis mixing and risk."""

    def __init__(self, config, kind, role, core_seed):
        super().__init__()
        if kind not in ('quantum', 'classical'): raise ValueError(kind)
        if role not in ('physical', 'contextual'): raise ValueError(role)
        h = config.hidden_dim
        self.kind, self.role = kind, role
        self.heads, self.head_dim, self.bases = config.graph_heads, h // config.graph_heads, 4
        self.node_norm = nn.LayerNorm(h)
        init_scale = (1.0 - 0.5) / 3.5
        self.angle_scale_raw = nn.Parameter(torch.full((12,), math.log(init_scale/(1-init_scale))))
        self.angle_bias = nn.Parameter(torch.zeros(12))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(core_seed)
            self.core = DualAxisPennyLanePQC(3) if kind == 'quantum' else LowRankClassicalCore()
        self.latent_norm = nn.LayerNorm(12)
        self.score = nn.Linear(12, self.heads, bias=False)
        self.gate = nn.Linear(12, self.heads, bias=False)
        self.basis_readout = nn.Linear(12, self.heads*self.bases, bias=False)
        self.risk_head = nn.Linear(12, 1)
        nn.init.zeros_(self.gate.weight)
        self.sender_value = nn.Linear(h, h, bias=False)
        self.edge_value = nn.Sequential(nn.Linear(7, h), nn.SiLU(), nn.Linear(h, h, bias=False))
        self.difference_value = nn.Linear(h, h, bias=False)
        self.joint_value = nn.Linear(h, h, bias=False)
        self.residual_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.output_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 2*h), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(2*h, h))
        self.final_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(config.dropout)
        self.last_risk_logits = None

    def relation_features(self, nodes, edge_features):
        normalized = self.node_norm(nodes)
        b,n,h = normalized.shape
        receiver = normalized[:, :, None, :].expand(-1,-1,n,-1)
        sender = normalized[:, None, :, :].expand(-1,n,-1,-1)
        products = (receiver*sender).reshape(b,n,n,4,h//4).mean(-1)
        rms = (receiver-sender).square().mean(-1,keepdim=True).add(1e-6).sqrt()
        physical, context = torch.tanh(edge_features), torch.tanh(torch.cat([products,rms],-1))
        # Layer one privileges direct kinematics; layer two privileges propagated context.
        return torch.cat([physical, .35*context],-1) if self.role == 'physical' else torch.cat([context, .35*physical],-1)

    def encode_angles(self, features):
        scale = .5 + 3.5*torch.sigmoid(self.angle_scale_raw)
        return 2*torch.atan(scale*features + .25*torch.tanh(self.angle_bias))

    def forward(self, nodes, edge_features, adjacency):
        b,n,h = nodes.shape
        norm = self.node_norm(nodes)
        relations = self.relation_features(nodes, edge_features)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        angles = self.encode_angles(relations.reshape(-1,12).index_select(0,selected))
        latent = self.latent_norm(self.core(angles))
        total = b*n*n
        score = nodes.new_zeros(total,self.heads).index_copy(0,selected,self.score(latent)).view(b,n,n,self.heads)
        gate = nodes.new_zeros(total,self.heads).index_copy(0,selected,self.gate(latent)).view(b,n,n,self.heads)
        mix = nodes.new_zeros(total,self.heads,self.bases).index_copy(
            0, selected, self.basis_readout(latent).view(-1,self.heads,self.bases)).view(b,n,n,self.heads,self.bases)
        risk = nodes.new_zeros(total).index_copy(0,selected,self.risk_head(latent).squeeze(-1)).view(b,n,n)
        self.last_risk_logits = risk

        receiver = norm[:, :, None, :].expand(-1,-1,n,-1)
        sender = norm[:, None, :, :].expand(-1,n,-1,-1)
        base = [self.sender_value(sender), self.edge_value(edge_features),
                self.difference_value(sender-receiver), self.joint_value(sender*receiver)]
        bases = torch.stack([x.view(b,n,n,self.heads,self.head_dim) for x in base], dim=-2)
        message = (torch.softmax(mix,dim=-1)[...,None]*bases).sum(-2)
        attention = torch.softmax(score.masked_fill(~adjacency[...,None],-1e4),dim=2)
        aggregate = (self.dropout(attention)[...,None]*(2*torch.sigmoid(gate))[...,None]*message).sum(2).reshape(b,n,h)
        nodes = self.output_norm(nodes + torch.sigmoid(self.residual_logit)*self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


def build_basis_dual_graph(arm, seed=2026):
    if arm not in ('basis_classical_dual','basis_quantum_dual'): raise ValueError(arm)
    kind = 'quantum' if arm == 'basis_quantum_dual' else 'classical'
    torch.manual_seed(seed); config=ForecasterConfig(); model=TargetInteractionGNN(config)
    torch.manual_seed(seed+3); model.graph_layers[0]=BasisMixMessageLayer(config,kind,'physical',seed+103)
    torch.manual_seed(seed+1); model.graph_layers[1]=BasisMixMessageLayer(config,kind,'contextual',seed+101)
    return model
