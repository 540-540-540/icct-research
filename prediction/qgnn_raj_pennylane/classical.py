from __future__ import annotations

import torch
from torch import nn

from prediction.qgnn_final.common import patch_inputs, physical_graph
from prediction.qgnn_paper_native.raj_paper import subset_tensors
from prediction.qgnn_paper_native.raj_subset import RajJohnsonGINCore, SubsetFeatureBuilder

from .common import (
    NODE_PAIRS,
    branch_agent_mask,
    directed_pair_features,
    pair_lookup,
    pair_valid_mask,
    subset_incidence,
)


def _mlp(in_dim: int, hidden: int, out_dim: int):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class WeightedJohnsonClassicalBranch(nn.Module):
    """Capacity-controlled classical counterpart to one Raj j-subset quantum branch."""

    def __init__(self, j: int, rounds: int = 3, hidden: int = 32, encoder_hidden: int = 64):
        super().__init__()
        if j not in (2, 3):
            raise ValueError(j)
        self.j = int(j)
        self.rounds = int(rounds)
        self.hidden = int(hidden)
        self.builder = SubsetFeatureBuilder(32)
        feat_dim = self.builder.out_dim
        self.input = _mlp(feat_dim, encoder_hidden, hidden)
        self.edge_weight = nn.ModuleList(_mlp(32, 32, 1) for _ in range(rounds))
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden, 64),
                nn.SiLU(),
                nn.Linear(64, hidden),
                nn.LayerNorm(hidden),
            )
            for _ in range(rounds)
        )
        self.readout = nn.Sequential(
            nn.LayerNorm((rounds + 1) * hidden),
            nn.Linear((rounds + 1) * hidden, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
        )

    def _johnson_matrix(self, history, mask, layer):
        edge, _ = physical_graph(history, mask)
        pair_feat = directed_pair_features(edge)
        theta = torch.tanh(self.edge_weight[layer](pair_feat).squeeze(-1))
        theta = theta * pair_valid_mask(mask).to(theta.dtype)

        _, links, _ = subset_tensors(mask.shape[1], self.j)
        links = links.to(mask.device)
        s = len(subset_tensors(mask.shape[1], self.j)[0])
        H = history.new_zeros(history.shape[0], s, s)
        if links.numel() == 0:
            return H
        lookup = pair_lookup().to(mask.device)
        pair_index = lookup[links[:, 2], links[:, 3]]
        weight = theta[:, pair_index]
        H[:, links[:, 0], links[:, 1]] = weight
        H[:, links[:, 1], links[:, 0]] = weight
        return H

    def _pool(self, states, valid, mask):
        inc = subset_incidence(self.j).to(states[0].device, states[0].dtype)
        weight = valid.to(states[0].dtype)
        den = torch.einsum("bs,sn->bn", weight, inc).clamp_min(1.0)
        pooled = [
            torch.einsum("bsh,bs,sn->bnh", h, weight, inc) / den[..., None]
            for h in states
        ]
        return torch.cat(pooled, -1) * mask[..., None]

    def forward_patch(self, history, mask):
        _, feat, valid, _ = self.builder(history, mask, self.j)
        active = branch_agent_mask(mask, self.j)
        h = torch.tanh(self.input(feat)) * valid[..., None]
        states = [h]
        for layer, update in enumerate(self.layers):
            H = self._johnson_matrix(history, mask, layer)
            agg = torch.bmm(H, h)
            h = update(torch.cat((h, agg), -1)) * valid[..., None]
            states.append(h)
        pooled = self._pool(states, valid, active)
        return self.readout(pooled) * active[..., None]


class MatchedRajJohnsonTokenCore(nn.Module):
    """Near-capacity classical j=2+j=3 token core for Raj-PennyLane comparisons."""

    readout_dim = 64
    interaction_tokens = 2

    def __init__(self, rounds: int = 3, hidden: int = 32):
        super().__init__()
        self.j2 = WeightedJohnsonClassicalBranch(2, rounds, hidden)
        self.j3 = WeightedJohnsonClassicalBranch(3, rounds, hidden)

    def forward(self, history, mask, timestamps=None):
        del timestamps
        history = torch.where(mask[:, None, :, None], history, 0.0)
        hp, mp, restore = patch_inputs(history, mask)
        z2 = self.j2.forward_patch(hp, mp)
        z3 = self.j3.forward_patch(hp, mp)
        tokens = torch.stack((z2, z3), 2)
        token_mask = torch.stack(
            (branch_agent_mask(mp, 2), branch_agent_mask(mp, 3)), 2
        )
        if restore is not None:
            b, n = restore
            tokens = tokens[:, 0].reshape(b, n, 2, 64)
            token_mask = token_mask[:, 0].reshape(b, n, 2)
        return {"readout": tokens, "interaction_mask": token_mask}


class HistoricalRajJohnsonTokenCore(nn.Module):
    """Strong historical JohnsonGIN branches exposed as separate j=2/j=3 tokens."""

    readout_dim = 64
    interaction_tokens = 2

    def __init__(self, rounds: int = 3, hidden: int = 64):
        super().__init__()
        self.j2 = RajJohnsonGINCore(j=2, rounds=rounds, hidden=hidden)
        self.j3 = RajJohnsonGINCore(j=3, rounds=rounds, hidden=hidden)

    def forward(self, history, mask, timestamps=None):
        del timestamps
        history = torch.where(mask[:, None, :, None], history, 0.0)
        hp, mp, restore = patch_inputs(history, mask)
        z2 = self.j2.forward_patch(hp, mp)
        z3 = self.j3.forward_patch(hp, mp)
        tokens = torch.stack((z2, z3), 2)
        token_mask = torch.stack(
            (branch_agent_mask(mp, 2), branch_agent_mask(mp, 3)), 2
        )
        if restore is not None:
            b, n = restore
            tokens = tokens[:, 0].reshape(b, n, 2, 64)
            token_mask = token_mask[:, 0].reshape(b, n, 2)
        return {"readout": tokens, "interaction_mask": token_mask}
