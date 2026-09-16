"""Edge-level QGNN relations condition LLM future states to refine trajectories."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from future_token_model import FutureTokenCoreGraphLLM


class TemporalPairRefinementFutureLLM(FutureTokenCoreGraphLLM):
    """Decode future neighbour effects from QGNN edges and LLM time states.

    The QGNN contributes an edge-specific latent and effective message strength.
    GPT-2 contributes a different hidden state at every future step.  Their joint
    decoder predicts the displacement induced by sender j on receiver i.
    """

    def __init__(self, graph, relation_dim=32, temporal_dim=64):
        super().__init__(graph)
        self._pair_relations = []
        self._pair_hooks = [
            layer.register_forward_hook(self._capture)
            for layer in self.graph_backbone.graph_layers
        ]
        self.pair_hidden_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, temporal_dim), nn.GELU()
        )
        self.pair_relation_projection = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(12), nn.Linear(12, relation_dim), nn.GELU())
            for _ in self.graph_backbone.graph_layers
        ])
        decoder_input = 4 * temporal_dim + relation_dim + 4
        self.pair_decoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(decoder_input),
                nn.Linear(decoder_input, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )
            for _ in self.graph_backbone.graph_layers
        ])
        for decoder in self.pair_decoders:
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)
        self.pair_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.pair_relation_mode = 'learned'

    def _capture(self, module, inputs, output):
        self._pair_relations.append((module.last_relation_latent, module.last_effective_relation))

    def _edge_weights(self, strength, adjacency):
        n = strength.shape[1]
        mask = adjacency & ~torch.eye(n, dtype=torch.bool, device=strength.device)[None]
        if self.pair_relation_mode == 'off':
            weight = torch.zeros_like(strength)
        elif self.pair_relation_mode == 'uniform':
            weight = mask.to(strength.dtype)
        else:
            weight = strength * mask
        return weight / weight.sum(-1, keepdim=True).clamp_min(1e-6), mask

    def _history_interaction_risk(self, history, adjacency):
        """Outcome-blind gate: close, converging neighbours need interaction reasoning."""
        position = history[:, -1, :, :2]
        velocity = history[:, -1, :, 2:4]
        relative_position = position[:, None, :, :] - position[:, :, None, :]
        relative_velocity = velocity[:, None, :, :] - velocity[:, :, None, :]
        distance = relative_position.norm(dim=-1).clamp_min(1e-3)
        closing = -(relative_position * relative_velocity).sum(dim=-1) / distance
        n = position.shape[1]
        mask = adjacency & ~torch.eye(n, dtype=torch.bool, device=history.device)[None]
        proximity = torch.sigmoid((30.0 - distance) / 5.0)
        convergence = torch.sigmoid((closing - 0.20) / 0.20)
        edge_risk = proximity * convergence * mask
        target_risk = edge_risk.max(dim=-1).values
        return target_risk[:, :, None, None]

    def forward(self, history, target_mask):
        self._pair_relations = []
        output = super().forward(history, target_mask)
        assert len(self._pair_relations) == len(self.pair_decoders)

        batch, _, targets, _ = history.shape
        horizon = self.config.prediction_length
        own = self.pair_hidden_projection(
            output['future_hidden_features'].view(batch, targets, horizon, self.d_llm)
        )
        receiver = own[:, :, None, :, :].expand(-1, -1, targets, -1, -1)
        sender = own[:, None, :, :, :].expand(-1, targets, -1, -1, -1)
        temporal_pair = torch.cat(
            [receiver, sender, receiver - sender, receiver * sender], dim=-1
        )

        base_nt = output['future_position'].permute(0, 2, 1, 3)
        base_relative = base_nt[:, None, :, :, :] - base_nt[:, :, None, :, :]
        base_relative = base_relative / 20.0
        time = torch.linspace(0.0, 1.0, horizon, device=history.device)
        time = time[None, None, None, :, None].expand(batch, targets, targets, -1, -1)

        layer_corrections = []
        normalized_weights = []
        for index, ((latent, strength), relation_projection, decoder) in enumerate(
            zip(self._pair_relations, self.pair_relation_projection, self.pair_decoders)
        ):
            weights, mask = self._edge_weights(strength, output['adjacency'])
            relation = relation_projection(latent)[:, :, :, None, :].expand(
                -1, -1, -1, horizon, -1
            )
            strength_feature = strength[:, :, :, None, None].expand(
                -1, -1, -1, horizon, -1
            )
            features = torch.cat(
                [temporal_pair, relation, base_relative, strength_feature, time], dim=-1
            )
            # Predict the future sender-minus-receiver relative-position error.
            # Moving the receiver by -delta/2 (and symmetrically the sender by
            # +delta/2 through its reverse edge) corrects that pair geometry.
            relative_delta = torch.tanh(decoder(features)) * 1.5
            relative_delta = relative_delta * mask[:, :, :, None, None]
            aggregate = -0.5 * (
                weights[:, :, :, None, None] * relative_delta
            ).sum(dim=2)
            layer_corrections.append(aggregate)
            normalized_weights.append(weights)
            if index == 0:
                pair_relative_deltas = [relative_delta]
            else:
                pair_relative_deltas.append(relative_delta)

        correction = torch.stack(layer_corrections, dim=0).mean(dim=0)
        interaction_risk = self._history_interaction_risk(history, output['adjacency'])
        correction = correction * torch.sigmoid(self.pair_scale_raw) * interaction_risk
        correction = correction * target_mask[:, :, None, None]
        refined_nt = base_nt + correction
        refined = refined_nt.permute(0, 2, 1, 3)

        output['source_future_position'] = output['future_position']
        output['future_position'] = refined
        output['displacement'] = refined - history[:, -1, None, :, :2]
        output['pair_refinement'] = correction.permute(0, 2, 1, 3)
        output['pair_weights'] = normalized_weights
        output['pair_relative_deltas'] = pair_relative_deltas
        output['pair_interaction_risk'] = interaction_risk
        return output
