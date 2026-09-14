"""QGNN-conditioned, three-stage semantic intent planner for future trajectories."""
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


class IntentTokenPlannerFutureLLM(FutureTokenCoreGraphLLM):
    """Let QGNN relations route LLM states, then plan three semantic stages."""

    def __init__(self, graph, segments=((0, 7), (7, 14), (14, 20)), width=64):
        super().__init__(graph)
        self.intent_segments = tuple(tuple(x) for x in segments)
        self._intent_relations = []
        self._intent_hooks = [
            layer.register_forward_hook(self._capture)
            for layer in self.graph_backbone.graph_layers
        ]
        self.intent_hidden_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, width), nn.GELU()
        )
        # own state + two QGNN-routed neighbour differences + entropy/peak per layer
        context_dim = 3 * width + 4
        self.intent_context_norm = nn.LayerNorm(context_dim)
        self.intent_longitudinal_head = nn.Linear(context_dim, 3)
        self.intent_lateral_head = nn.Linear(context_dim, 3)
        self.intent_interaction_head = nn.Linear(context_dim, 3)
        step_dim = context_dim + 9
        self.intent_residual_head = nn.Sequential(
            nn.LayerNorm(step_dim), nn.Linear(step_dim, 128), nn.GELU(), nn.Linear(128, 2)
        )
        self.intent_local_gate = nn.Sequential(
            nn.LayerNorm(step_dim), nn.Linear(step_dim, 1)
        )
        nn.init.zeros_(self.intent_local_gate[-1].weight)
        nn.init.zeros_(self.intent_local_gate[-1].bias)
        # Exact epoch-0 fallback.  Separate strengths let the late stage carry
        # more responsibility when validation supports it.
        self.intent_stage_strength = nn.Parameter(torch.zeros(len(self.intent_segments)))
        self.intent_residual_scale_m = 1.5
        self.intent_relation_mode = 'learned'
        # Diagnostic-only batch override.  Normal training/inference keeps None.
        self.intent_probability_override = None

    def _capture(self, module, inputs, output):
        self._intent_relations.append(module.last_effective_relation)

    def _weights(self, relation, adjacency):
        n = relation.shape[1]
        mask = adjacency & ~torch.eye(n, dtype=torch.bool, device=relation.device)[None]
        if self.intent_relation_mode == 'off':
            value = torch.zeros_like(relation)
        elif self.intent_relation_mode == 'uniform':
            value = mask.to(relation.dtype)
        else:
            value = relation * mask
        return value / value.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, history, target_mask):
        self._intent_relations = []
        output = super().forward(history, target_mask)
        assert len(self._intent_relations) == 2
        batch, _, targets, _ = history.shape
        horizon = self.config.prediction_length
        own = self.intent_hidden_projection(
            output['future_hidden_features'].view(batch, targets, horizon, self.d_llm)
        )
        routed = []
        statistics = []
        for relation in self._intent_relations:
            weights = self._weights(relation, output['adjacency'])
            neighbour = torch.einsum('bij,bjtd->bitd', weights, own)
            routed.append(neighbour - own)
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
            entropy = entropy / torch.log(torch.tensor(float(max(targets - 1, 2)), device=history.device))
            peak = weights.max(-1).values
            statistics.extend([entropy, peak])
        stats = torch.stack(statistics, dim=-1)[:, :, None, :].expand(-1, -1, horizon, -1)
        step_context = self.intent_context_norm(torch.cat([own] + routed + [stats], dim=-1))

        logits = {'longitudinal': [], 'lateral': [], 'interaction': []}
        probabilities = []
        corrections = []
        gains = []
        for segment_index, (start, end) in enumerate(self.intent_segments):
            pooled = 0.5 * step_context[:, :, start:end].mean(2) + 0.5 * step_context[:, :, end - 1]
            long_logits = self.intent_longitudinal_head(pooled)
            lateral_logits = self.intent_lateral_head(pooled)
            interaction_logits = self.intent_interaction_head(pooled)
            logits['longitudinal'].append(long_logits)
            logits['lateral'].append(lateral_logits)
            logits['interaction'].append(interaction_logits)
            if self.intent_probability_override is None:
                probs = torch.cat([
                    torch.softmax(long_logits, -1),
                    torch.softmax(lateral_logits, -1),
                    torch.softmax(interaction_logits, -1),
                ], dim=-1)
            else:
                probs = self.intent_probability_override[:, :, segment_index]
            probabilities.append(probs)
            segment_probs = probs[:, :, None, :].expand(-1, -1, end - start, -1)
            decoder_input = torch.cat([step_context[:, :, start:end], segment_probs], dim=-1)
            candidate = torch.tanh(self.intent_residual_head(decoder_input)) * self.intent_residual_scale_m
            local = torch.sigmoid(self.intent_local_gate(decoder_input))
            strength = torch.tanh(self.intent_stage_strength[segment_index])
            corrections.append(candidate)
            gains.append(strength * local)

        candidate_nt = torch.cat(corrections, dim=2)
        gain_nt = torch.cat(gains, dim=2)
        source_nt = output['future_position'].permute(0, 2, 1, 3)
        fused_nt = source_nt + gain_nt * candidate_nt
        fused_nt = fused_nt * target_mask[:, :, None, None]
        fused = fused_nt.permute(0, 2, 1, 3)
        output['source_future_position'] = output['future_position']
        output['future_position'] = fused
        output['displacement'] = fused - history[:, -1, None, :, :2]
        output['intent_logits'] = {
            name: torch.stack(values, dim=2) for name, values in logits.items()
        }
        output['intent_probabilities'] = torch.stack(probabilities, dim=2)
        output['intent_candidate_correction'] = candidate_nt.permute(0, 2, 1, 3)
        output['intent_gain'] = gain_nt.permute(0, 2, 1, 3)
        return output
