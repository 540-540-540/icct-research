"""Mechanism-only Raj diagnostics kept separate from accepted paper-native models."""
from __future__ import annotations

import torch
from torch import nn

from prediction.qgnn_final.common import BoundedCore, physical_graph
from .raj_subset import SubsetFeatureBuilder
from .raj_paper import WeightedJohnsonMix, subset_tensors


class RajSubsetBuilderWeightedJohnsonCore(BoundedCore):
    """Weighted/no-local Johnson control using the full-paper prototype feature builder.

    This mirrors RajPaperJohnsonGINCore except that ContinuousSubsetLoader is
    replaced by SubsetFeatureBuilder. It is a diagnostic class, not a new primary
    baseline.
    """

    def __init__(self, j: int = 3, depth: int = 3, hidden: int = 128):
        super().__init__()
        self.j, self.depth, self.hidden = j, depth, hidden
        self.builder = SubsetFeatureBuilder(32)
        self.edge_weight = WeightedJohnsonMix(j, depth)
        self.input = nn.Sequential(
            nn.Linear(self.builder.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(depth)
        )
        self.readout = nn.Sequential(
            nn.Linear((depth + 1) * hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

    @staticmethod
    def _unit(x):
        return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-8)

    def forward_patch(self, history, mask):
        own, feat, valid, subs = self.builder(history, mask, self.j)
        if subs.shape[0] == 0:
            return history.new_zeros(history.shape[0], history.shape[2], 64)

        edge, _ = physical_graph(history, mask)
        h = self._unit(self.input(feat)) * valid[..., None]
        hs = [h]
        for l, layer in enumerate(self.layers):
            H = self.edge_weight.hamiltonian(edge, mask, l)
            agg = torch.bmm(H, h)
            h = self._unit(torch.tanh(layer(torch.cat((h, agg), -1)))) * valid[..., None]
            hs.append(h)

        joint = torch.cat(hs, -1)
        _, _, inc0 = subset_tensors(history.shape[2], self.j)
        inc = inc0.to(h.device, h.dtype)
        den = torch.einsum("bs,sn->bn", valid.to(h.dtype), inc).clamp_min(1.0)
        pooled = torch.einsum("bsh,sn->bnh", joint * valid[..., None], inc) / den[..., None]
        return self.readout(pooled) * mask[..., None]

