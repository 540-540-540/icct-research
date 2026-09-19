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

class RowLocalFeatureLoader(nn.Module):
    """Same feature MLP as the paper-math loader, but row-local/additive semantics."""

    def __init__(self, feat_dim: int, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, embed_dim))

    def _conditional(self, feat, valid):
        raw = torch.nn.functional.softplus(self.net(feat)) * valid[..., None]
        cond = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.where(valid[..., None], cond, torch.zeros_like(cond))

    def _scale(self, valid, dtype):
        count = valid.sum(-1, keepdim=True).clamp_min(1).to(dtype)
        return count.rsqrt()[..., None]

    def forward(self, feat, valid):
        cond = self._conditional(feat, valid)
        return cond * self._scale(valid, cond.dtype), cond

    def reupload(self, x, cond, valid):
        scale = self._scale(valid, x.real.dtype).to(x.dtype)
        target = cond.to(x.dtype) * scale
        y = (x + target) * valid[..., None]
        row = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
        return row * scale * valid[..., None]


class RajPaperMathRowLocalQGNNCore(BoundedCore):
    """Full paper-math pipeline with only V/re-upload semantics replaced."""

    def __init__(self, j: int = 3, rounds: int = 3, D: int = 6, k: int = 3, n: int = 8):
        super().__init__()
        import math
        from .raj_paper_math import EquivariantSubsetAdjacency, ConditionalRDMReadout
        from .raj_paper import CompoundEvolution
        from .raj_joint import JointRegisterMixer

        if n != 8:
            raise ValueError("BoundedCore patch size is fixed at N=8 for this diagnostic")
        self.n, self.j, self.rounds, self.D, self.k = n, j, rounds, D, k
        self.builder = SubsetFeatureBuilder(32)
        self.embed_dim = math.comb(D, k)
        self.loaders = nn.ModuleList(
            RowLocalFeatureLoader(self.builder.out_dim, self.embed_dim) for _ in range(rounds + 1)
        )
        self.adj = nn.ModuleList(EquivariantSubsetAdjacency(n, j) for _ in range(rounds))
        self.evolution = nn.ModuleList(CompoundEvolution(D, k) for _ in range(rounds))
        self.joint = JointRegisterMixer(n=n, D=D, j=j, k=k)
        self.readout = ConditionalRDMReadout(n, j, D, k, 64)

    def forward_patch(self, history, mask):
        own, feat, valid, subs = self.builder(history, mask, self.j)
        if not bool(valid.any()):
            return history.new_zeros(history.shape[0], history.shape[2], 64)
        edge, risk = physical_graph(history, mask)
        x0, _ = self.loaders[0](feat, valid)
        ctype = torch.complex64 if x0.dtype == torch.float32 else torch.complex128
        x = x0.to(ctype)
        for l in range(self.rounds):
            x = self.adj[l](x, edge, risk, mask)
            x = self.evolution[l](x)
            _, cond = self.loaders[l + 1](feat, valid)
            x = self.loaders[l + 1].reupload(x, cond, valid)
        joint = self.joint(x, risk, mask)
        from .raj_joint import conditional_embedding_states
        conditional, prob = conditional_embedding_states(self.joint, joint)
        z, _ = self.readout(conditional, prob, valid, mask)
        return z

