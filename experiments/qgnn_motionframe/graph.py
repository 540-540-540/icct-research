"""Inherited QGNN with receiver-motion coordinates only in quantum relations."""
import torch
from torch import nn

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _archive, _base


def receiver_motion_edges(edge_features, velocity):
    """For i<-j, project the four relative-motion channels onto receiver i.

    Input edges retain their existing /30 and /15 scaling. Speed below the
    preset 1 m/s threshold uses global right=(1,0), forward=(0,1).
    """
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    fallback = velocity.new_tensor([0., 1.])
    forward = torch.where(speed >= 1., velocity / speed.clamp_min(1.), fallback)
    right = torch.stack((forward[..., 1], -forward[..., 0]), dim=-1)
    axes = (right[:, :, None, :], forward[:, :, None, :])
    rotated = [(edge_features[..., start:start+2] * axis).sum(-1, keepdim=True)
               for start in (0, 2) for axis in axes]
    return torch.cat([*rotated, edge_features[..., 4:]], dim=-1)


class _MotionFrameMessageLayer(_archive.PhysicsAlignedMessageLayer):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name in ('kind', 'role', 'heads', 'head_dim', 'context_gain'):
            setattr(self, name, getattr(original, name))
        # Adopt the original modules/parameters without new random draws.
        for name, module in original.named_children():
            self.add_module(name, module)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)

    def forward(self, nodes, edge_features, adjacency, quantum_edges):
        b, n, h = nodes.shape
        relations = self.relation_features(nodes, quantum_edges)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        angles = self.encode_angles(relations.reshape(-1, 12).index_select(0, selected))
        latent = self.latent_norm(self.core(angles))

        values = self.value(self.value_norm(nodes)).view(b, n, self.heads, self.head_dim)
        # Classical message content continues to use the unrotated SI edges.
        edge_values = self.edge_value(edge_features).view(b, n, n, self.heads, self.head_dim)
        base_message = values[:, None, :, :, :] + edge_values
        score = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, self.score(latent)).view(b, n, n, self.heads)
        gate = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, self.gate(latent)).view(b, n, n, self.heads)
        gate = 2.0 * torch.sigmoid(gate)
        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        aggregate = (self.dropout(attention)[..., None] * gate[..., None] * base_message).sum(2).reshape(b, n, h)
        residual_strength = torch.sigmoid(self.residual_logit)
        nodes = self.output_norm(nodes + residual_strength * self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


class MotionFrameQGNNGraph(InheritedQGNNGraph):
    """Same state_dict and graph contract; no new parameters or readouts."""

    def __init__(self, seed=2026):
        super().__init__(seed=seed)
        self.graph_layers = nn.ModuleList([_MotionFrameMessageLayer(layer) for layer in self.graph_layers])

    def forward(self, state_hat, standardized_state, track_exists, detected):
        if state_hat.shape != standardized_state.shape or state_hat.ndim < 3 or state_hat.shape[-1] != 4:
            raise ValueError('Expected matching [...,N,4] state tensors')
        if track_exists.shape != state_hat.shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        n = state_hat.shape[-2]
        if not 1 <= n <= 8:
            raise ValueError('One to eight slots are required')
        mask = track_exists.bool()
        physical = torch.where(mask[..., None], state_hat, 0)
        z = torch.where(mask[..., None], standardized_state, 0)
        detection = torch.where(mask, detected, 0).to(z)
        if not torch.isfinite(physical).all() or not torch.isfinite(z).all() or not torch.isfinite(detection).all():
            raise ValueError('Existing states and detection flags must be finite')
        nodes = self.input(torch.cat((z, detection[..., None], mask[..., None].to(z)), -1))
        current = mask.reshape(-1, n)
        nodes = (nodes * mask[..., None]).reshape(-1, n, 128)
        active = current.any(-1).nonzero().squeeze(-1)
        if active.numel() == 0:
            return nodes.reshape(*state_hat.shape[:-1], 128)
        live = current.index_select(0, active)
        h = nodes.index_select(0, active)
        physical = physical.reshape(-1, n, 4).index_select(0, active)
        edges, distances = _base.build_edge_features(physical[..., :2], physical[..., 2:])
        quantum_edges = receiver_motion_edges(edges, physical[..., 2:])
        pairs = live[:, :, None] & live[:, None, :]
        adjacency = pairs & (distances <= 45.)
        adjacency |= torch.eye(n, device=state_hat.device, dtype=torch.bool)[None] & pairs
        for layer in self.graph_layers:
            h, _ = layer(h, edges, adjacency, quantum_edges)
            h = h * live[..., None]
        nodes = nodes.index_copy(0, active, h)
        return nodes.reshape(*state_hat.shape[:-1], 128)
