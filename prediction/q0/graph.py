"""Classical graph cores used to establish the Q0 diagnostic baseline ladder."""
from __future__ import annotations

import torch
from torch import nn


class IndependentTemporalEncoder(nn.Module):
    """Shared causal encoder applied independently to each vehicle."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Linear(4, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, standardized_state: torch.Tensor, vehicle_mask: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = standardized_state.shape
        x = standardized_state.permute(0, 2, 1, 3).reshape(b * n, t, 4)
        x = torch.nn.functional.silu(self.input(x))
        h, _ = self.gru(x)
        h = self.norm(h).reshape(b, n, t, self.hidden_dim).permute(0, 2, 1, 3)
        return torch.where(vehicle_mask[:, None, :, None], h, torch.zeros_like(h))


class NoGraphCore(nn.Module):
    """No cross-vehicle reasoning: only the shared independent temporal encoder."""

    display_name = "NoGraph"

    def __init__(self, hidden_dim: int = 64, graph_dim: int = 64) -> None:
        super().__init__()
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        self.graph_dim = int(graph_dim)

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        del standardized_edges, pair_mask
        h = self.encoder(standardized_state, vehicle_mask)
        z = self.output(h)
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))


class ResidualEdgeMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 8) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h: [B,T,N,H], edge: [B,T,N,N,E], pair_mask: [B,N,N].
        n = h.shape[2]
        receiver = h.unsqueeze(3).expand(-1, -1, -1, n, -1)
        sender = h.unsqueeze(2).expand(-1, -1, n, -1, -1)
        message_input = torch.cat([receiver, sender, edge], dim=-1)
        message = self.message(message_input)
        valid = pair_mask[:, None, :, :, None]
        message = torch.where(valid, message, torch.zeros_like(message))
        degree = pair_mask.sum(dim=-1).clamp_min(1).to(h.dtype)[:, None, :, None]
        aggregate = message.sum(dim=3) / degree
        delta = self.update(torch.cat([h, aggregate], dim=-1))
        out = self.norm(h + delta)
        return torch.where(vehicle_mask[:, None, :, None], out, torch.zeros_like(out))


class EdgeResidualMPNN(nn.Module):
    """Strong pairwise baseline with explicit physical edge features."""

    display_name = "EdgeResidualMPNN"

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_dim: int = 64,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            [ResidualEdgeMPNNLayer(hidden_dim) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        self.graph_dim = int(graph_dim)

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(standardized_state, vehicle_mask)
        for layer in self.layers:
            h = layer(h, standardized_edges, pair_mask, vehicle_mask)
        z = self.output(h)
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))



class EdgeGATv2Layer(nn.Module):
    """Edge-aware dynamic attention layer following the GATv2 scoring principle."""

    def __init__(self, hidden_dim: int, edge_dim: int = 8, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.attn_features = nn.Linear(2 * hidden_dim + edge_dim, hidden_dim)
        self.attn_vector = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vector)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = h.shape[2]
        receiver = h.unsqueeze(3).expand(-1, -1, -1, n, -1)
        sender = h.unsqueeze(2).expand(-1, -1, n, -1, -1)
        joined = torch.cat([receiver, sender, edge], dim=-1)
        a = torch.nn.functional.leaky_relu(self.attn_features(joined), negative_slope=0.2)
        a = a.reshape(*a.shape[:-1], self.heads, self.head_dim)
        score = (a * self.attn_vector).sum(dim=-1)

        valid = pair_mask[:, None, :, :, None]
        # Stable masked softmax; padded receiver rows stay exactly zero.
        masked_score = torch.where(valid, score, torch.full_like(score, -1e9))
        max_score = masked_score.max(dim=3, keepdim=True).values
        weight = torch.where(valid, torch.exp(masked_score - max_score), torch.zeros_like(score))
        weight = weight / weight.sum(dim=3, keepdim=True).clamp_min(1e-12)

        value = self.value(sender).reshape(*sender.shape[:-1], self.heads, self.head_dim)
        aggregate = (weight[..., None] * value).sum(dim=3).reshape_as(h)
        out = self.norm(h + self.output(aggregate))
        return torch.where(vehicle_mask[:, None, :, None], out, torch.zeros_like(out))


class EdgeGATv2(nn.Module):
    """Strong edge-aware attention baseline for G1 interaction diagnostics."""

    display_name = "EdgeGATv2"

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_dim: int = 64,
        layers: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeGATv2Layer(hidden_dim, heads=heads) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        self.graph_dim = int(graph_dim)

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(standardized_state, vehicle_mask)
        for layer in self.layers:
            h = layer(h, standardized_edges, pair_mask, vehicle_mask)
        z = self.output(h)
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))



class GatedEdgeResidualMPNN(nn.Module):
    """Pairwise MPNN with an explicit NoGraph bypass and learned interaction gate.

    The local encoder/output path is construction-compatible with NoGraphCore.
    The interaction correction is initialized to exactly zero, so this model starts
    from the same local representation and can always revert to it.
    """

    display_name = "GatedEdgeResidualMPNN"

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_dim: int = 64,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        # Construct before the interaction branch so, under the paired graph RNG,
        # this path matches NoGraphCore exactly.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        self.layers = nn.ModuleList(
            [ResidualEdgeMPNNLayer(hidden_dim) for _ in range(layers)]
        )
        self.delta = nn.Linear(hidden_dim, graph_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Exact NoGraph starting point without blocking learning of the residual head.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.graph_dim = int(graph_dim)

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        local_h = self.encoder(standardized_state, vehicle_mask)
        local_z = self.output(local_h)
        h = local_h
        for layer in self.layers:
            h = layer(h, standardized_edges, pair_mask, vehicle_mask)
        interaction = h - local_h
        gate = torch.sigmoid(self.gate(torch.cat([local_h, interaction], dim=-1)))
        z = local_z + gate * self.delta(interaction)
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))



class RootedTripletLayer(nn.Module):
    """Permutation-equivariant explicit i<-{j,k} interaction over unordered neighbor pairs."""

    def __init__(self, hidden_dim: int, edge_dim: int = 8) -> None:
        super().__init__()
        # [h_i, h_j+h_k, |h_j-h_k|] +
        # [e_ij+e_ik, |e_ij-e_ik|, e_jk+e_kj, |e_jk-e_kj|]
        input_dim = 3 * hidden_dim + 4 * edge_dim
        self.message = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        roots, first, second = [], [], []
        for i in range(8):
            neighbors = [j for j in range(8) if j != i]
            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    roots.append(i)
                    first.append(neighbors[a])
                    second.append(neighbors[b])
        self.register_buffer("root_index", torch.tensor(roots, dtype=torch.long), persistent=False)
        self.register_buffer("first_index", torch.tensor(first, dtype=torch.long), persistent=False)
        self.register_buffer("second_index", torch.tensor(second, dtype=torch.long), persistent=False)
        root_onehot = torch.nn.functional.one_hot(self.root_index, num_classes=8).float()
        self.register_buffer("root_onehot", root_onehot, persistent=False)

    def forward(
        self,
        h: torch.Tensor,
        edge: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h [B,T,N,H], edge [B,T,N,N,E]; P=168 rooted unordered triplets.
        ri, ji, ki = self.root_index, self.first_index, self.second_index
        hi, hj, hk = h[:, :, ri], h[:, :, ji], h[:, :, ki]
        eij = edge[:, :, ri, ji]
        eik = edge[:, :, ri, ki]
        ejk = edge[:, :, ji, ki]
        ekj = edge[:, :, ki, ji]
        features = torch.cat(
            [
                hi,
                hj + hk,
                (hj - hk).abs(),
                eij + eik,
                (eij - eik).abs(),
                ejk + ekj,
                (ejk - ekj).abs(),
            ],
            dim=-1,
        )
        message = self.message(features)
        valid = (
            vehicle_mask[:, ri]
            & vehicle_mask[:, ji]
            & vehicle_mask[:, ki]
        )
        message = torch.where(valid[:, None, :, None], message, torch.zeros_like(message))
        weights = valid.to(message.dtype)[:, :, None] * self.root_onehot[None, :, :]
        # Sum P rooted-triplet messages into receiver nodes, independently for B,T.
        aggregate = torch.einsum("btph,bpn->btnh", message, weights)
        count = weights.sum(dim=1).clamp_min(1.0)[:, None, :, None]
        aggregate = aggregate / count
        has_triplet = (weights.sum(dim=1) > 0)[:, None, :, None]
        delta = self.update(torch.cat([h, aggregate], dim=-1))
        out = self.norm(h + delta)
        return torch.where(has_triplet, out, h)


class GatedPairTriplet(nn.Module):
    """Strong classical higher-order diagnostic: gated pairwise + explicit rooted triplets."""

    display_name = "GatedPairTriplet"

    def __init__(self, hidden_dim: int = 64, graph_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        # Same construction order as GatedEdgeResidualMPNN through this branch.
        self.pair_layers = nn.ModuleList(
            [ResidualEdgeMPNNLayer(hidden_dim) for _ in range(layers)]
        )
        self.pair_delta = nn.Linear(hidden_dim, graph_dim)
        self.pair_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.pair_delta.weight)
        nn.init.zeros_(self.pair_delta.bias)
        nn.init.zeros_(self.pair_gate[-1].weight)
        nn.init.zeros_(self.pair_gate[-1].bias)

        self.triplet_layer = RootedTripletLayer(hidden_dim)
        self.triplet_delta = nn.Linear(hidden_dim, graph_dim)
        self.triplet_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.triplet_delta.weight)
        nn.init.zeros_(self.triplet_delta.bias)
        nn.init.zeros_(self.triplet_gate[-1].weight)
        nn.init.zeros_(self.triplet_gate[-1].bias)
        self.graph_dim = int(graph_dim)

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        local_h = self.encoder(standardized_state, vehicle_mask)
        local_z = self.output(local_h)

        pair_h = local_h
        for layer in self.pair_layers:
            pair_h = layer(pair_h, standardized_edges, pair_mask, vehicle_mask)
        pair_interaction = pair_h - local_h
        pair_gate = torch.sigmoid(
            self.pair_gate(torch.cat([local_h, pair_interaction], dim=-1))
        )
        pair_correction = pair_gate * self.pair_delta(pair_interaction)

        trip_h = self.triplet_layer(local_h, standardized_edges, vehicle_mask)
        trip_interaction = trip_h - local_h
        trip_gate = torch.sigmoid(
            self.triplet_gate(torch.cat([local_h, trip_interaction], dim=-1))
        )
        trip_correction = trip_gate * self.triplet_delta(trip_interaction)
        z = local_z + pair_correction + trip_correction
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))



class PhysicsRoutedEdgeMPNN(nn.Module):
    """Local bypass plus a scene-level gate learned from physical interaction summaries.

    The graph branch is a normal edge-aware MPNN. The routing gate only decides
    how much of its interaction residual should enter the node representation.
    This is a strong classical control for heterogeneous interaction relevance.
    """

    display_name = "PhysicsRoutedEdgeMPNN"

    def __init__(self, hidden_dim: int = 64, graph_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.encoder = IndependentTemporalEncoder(hidden_dim)
        # Construction-compatible local path.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim),
            nn.SiLU(),
            nn.LayerNorm(graph_dim),
        )
        self.layers = nn.ModuleList(
            [ResidualEdgeMPNNLayer(hidden_dim) for _ in range(layers)]
        )
        self.delta = nn.Linear(hidden_dim, graph_dim)
        # N plus six physical edge summaries:
        # mean/min distance, mean/max closing, min tCPA, min dCPA.
        self.router = nn.Sequential(
            nn.Linear(7, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        # Conservative start: mostly local, but not saturated.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, -1.5)
        self.graph_dim = int(graph_dim)

    @staticmethod
    def _scene_features(edge: torch.Tensor, pair_mask: torch.Tensor, vehicle_mask: torch.Tensor) -> torch.Tensor:
        # Use last-frame standardized physical edges. Statistics stay permutation invariant.
        e = edge[:, -1]  # [B,N,N,E]
        valid = pair_mask
        vf = valid.to(e.dtype)
        count = vf.sum(dim=(1, 2)).clamp_min(1.0)

        def masked_mean(x):
            return (x * vf).sum(dim=(1, 2)) / count

        inf = torch.full_like(e[..., 0], float("inf"))
        neg_inf = torch.full_like(e[..., 0], -float("inf"))
        dist = e[..., 4]
        closing = e[..., 5]
        tcpa = e[..., 6]
        dcpa = e[..., 7]
        n = vehicle_mask.sum(dim=1).to(e.dtype) / vehicle_mask.shape[1]
        feats = torch.stack(
            [
                n,
                masked_mean(dist),
                torch.where(valid, dist, inf).amin(dim=(1, 2)),
                masked_mean(closing),
                torch.where(valid, closing, neg_inf).amax(dim=(1, 2)),
                torch.where(valid, tcpa, inf).amin(dim=(1, 2)),
                torch.where(valid, dcpa, inf).amin(dim=(1, 2)),
            ],
            dim=-1,
        )
        return feats

    def forward(
        self,
        standardized_state: torch.Tensor,
        standardized_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        local_h = self.encoder(standardized_state, vehicle_mask)
        local_z = self.output(local_h)

        h = local_h
        for layer in self.layers:
            h = layer(h, standardized_edges, pair_mask, vehicle_mask)
        interaction = h - local_h

        scene_features = self._scene_features(standardized_edges, pair_mask, vehicle_mask)
        gate = torch.sigmoid(self.router(scene_features))[:, None, None, :]
        z = local_z + gate * self.delta(interaction)
        return torch.where(vehicle_mask[:, None, :, None], z, torch.zeros_like(z))
