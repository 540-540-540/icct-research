"""Inherited QGNN with zero-initialized directed context residuals."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _archive


class _DirectedContextMessageLayer(_archive.PhysicsAlignedMessageLayer):
    """Add receiver/sender roles without changing the inherited mapping at zero."""

    def __init__(self, original):
        nn.Module.__init__(self)
        for name in ('kind', 'role', 'heads', 'head_dim', 'context_gain'):
            setattr(self, name, getattr(original, name))
        for name, module in original.named_children():
            self.add_module(name, module)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        width = self.heads * self.head_dim
        self.receiver_context = nn.Parameter(original.node_norm.weight.new_zeros(5, width))
        self.sender_context = nn.Parameter(original.node_norm.weight.new_zeros(5, width))

    def relation_features(self, nodes, edge_features):
        normalized = self.node_norm(nodes)
        b, n, h = normalized.shape
        receiver = normalized[:, :, None, :].expand(-1, -1, n, -1)
        sender = normalized[:, None, :, :].expand(-1, n, -1, -1)
        products = (receiver * sender).reshape(b, n, n, 4, h // 4).mean(-1)
        rms_difference = (receiver - sender).square().mean(-1, keepdim=True).add(1e-6).sqrt()
        context = torch.cat((products, rms_difference), -1)
        directed = F.linear(receiver, self.receiver_context) + F.linear(sender, self.sender_context)
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=nodes.device)[None, :, :, None]
        context = context + directed * off_diagonal
        return torch.cat((torch.tanh(edge_features), self.context_gain * torch.tanh(context)), -1)


class DirectedContextQGNNGraph(InheritedQGNNGraph):
    """Two inherited quantum layers plus 2,560 classical interface parameters."""

    def __init__(self, seed=2026):
        super().__init__(seed=seed)
        self.graph_layers = nn.ModuleList([_DirectedContextMessageLayer(layer) for layer in self.graph_layers])
