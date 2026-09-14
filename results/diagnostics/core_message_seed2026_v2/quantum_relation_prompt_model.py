"""Direct QGNN-to-LLM interface using quantum relation observables as prompts."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from future_token_model import FutureTokenCoreGraphLLM
from target_interaction_graph import build_edge_features


class RelationPromptCoreGraphLLM(FutureTokenCoreGraphLLM):
    """Cross-attend future queries to edge-level relation observables.

    Each of the two graph layers contributes one 12-D circuit readout per
    directed edge.  The readout, its Pauli-Z uncertainty proxy (1-z^2), and
    seven physical edge features form a relation token.  Future queries attend
    to these tokens before entering GPT-2.  The prompt strength starts at zero,
    so epoch 0 exactly reproduces the selected future-token checkpoint.
    """

    def __init__(self, graph):
        super().__init__(graph)
        self._captured_relation_latents = []
        self._relation_hooks = []
        for layer in self.graph_backbone.graph_layers:
            self._relation_hooks.append(
                layer.latent_norm.register_forward_hook(self._capture_relation_latent)
            )
        relation_dim = 12 + 12 + 7
        attention_dim = 96
        self.relation_projection = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(relation_dim),
                nn.Linear(relation_dim, 192),
                nn.GELU(),
                nn.Linear(192, self.d_llm),
            )
            for _ in self.graph_backbone.graph_layers
        ])
        self.relation_layer_type = nn.Parameter(
            torch.randn(len(self.graph_backbone.graph_layers), self.d_llm) * 0.01
        )
        self.relation_query = nn.Linear(self.d_llm, attention_dim, bias=False)
        self.relation_key = nn.Linear(self.d_llm, attention_dim, bias=False)
        self.relation_value = nn.Linear(self.d_llm, self.d_llm, bias=False)
        self.relation_output_norm = nn.LayerNorm(self.d_llm)
        self.relation_prompt_strength = nn.Parameter(torch.zeros(()))
        # Positive scale for a future-time-aligned physical risk prior.
        self.relation_risk_bias_raw = nn.Parameter(torch.tensor(1.8545866))  # softplus ~= 2
        self.relation_llm_fusion_head = nn.Sequential(
            nn.LayerNorm(2 * self.d_llm),
            nn.Linear(2 * self.d_llm, 192),
            nn.GELU(),
            nn.Linear(192, 2),
        )
        self.relation_coordinate_strength = nn.Parameter(torch.zeros(()))
        self.last_relation_attention_entropy = None
        self.last_relation_risk_activation = None
        self._last_relation_context = None

    def _capture_relation_latent(self, module, inputs, output):
        self._captured_relation_latents.append(output)

    def forward(self, history, target_mask):
        self._captured_relation_latents = []
        output = super().forward(history, target_mask)
        batch, _, targets, _ = history.shape
        fusion_features = torch.cat(
            [output["future_hidden_features"], self._last_relation_context], dim=-1
        )
        relation_delta = torch.tanh(self.relation_llm_fusion_head(fusion_features)) * 0.75
        relation_delta = relation_delta * torch.tanh(self.relation_coordinate_strength)
        relation_delta = relation_delta.view(
            batch, targets, self.config.prediction_length, 2
        ).permute(0, 2, 1, 3)
        relation_delta = relation_delta * target_mask[:, None, :, None]
        output["future_position"] = output["future_position"] + relation_delta
        output["displacement"] = output["future_position"] - history[:, -1, None, :, :2]
        output["relation_prompt_strength"] = torch.tanh(self.relation_prompt_strength)
        output["relation_coordinate_strength"] = torch.tanh(self.relation_coordinate_strength)
        output["relation_attention_entropy"] = self.last_relation_attention_entropy
        output["relation_risk_activation"] = self.last_relation_risk_activation
        return output

    def _future_query_addition(self, history, graph_output, flat_nodes):
        batch, _, targets, _ = history.shape
        adjacency = graph_output["adjacency"]
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        assert len(self._captured_relation_latents) == len(self.relation_projection)
        edge_features, _ = build_edge_features(history[:, -1, :, :2], history[:, -1, :, 2:4])

        tokens = []
        masks = []
        risks = []
        eye = torch.eye(targets, dtype=torch.bool, device=history.device)[None]
        relation_mask = adjacency & ~eye
        for layer_index, latent in enumerate(self._captured_relation_latents):
            dense = flat_nodes.new_zeros(batch * targets * targets, 12)
            dense = dense.index_copy(0, selected, latent)
            dense = dense.view(batch, targets, targets, 12)
            uncertainty = (1.0 - dense.square()).clamp_min(0.0)
            relation_input = torch.cat([dense, uncertainty, edge_features], dim=-1)
            embedded = self.relation_projection[layer_index](relation_input)
            embedded = embedded + self.relation_layer_type[layer_index]
            tokens.append(embedded.reshape(batch * targets, targets, self.d_llm))
            masks.append(relation_mask.reshape(batch * targets, targets))
            # Physical event locator: a future query should focus on an edge when
            # its time-to-collision is near that query's horizon, with additional
            # preference for closing and nearby pairs. Quantum observables then
            # describe the selected relation; the prior does not replace them.
            distance = edge_features[..., 4].reshape(batch * targets, targets)
            closing = edge_features[..., 5].reshape(batch * targets, targets)
            ttc_seconds = 20.0 * edge_features[..., 6].reshape(batch * targets, targets)
            future_time = (
                torch.arange(1, self.config.prediction_length + 1, device=history.device, dtype=history.dtype)
                * self.config.dt
            )
            time_match = -torch.abs(ttc_seconds[:, None, :] - future_time[None, :, None]) / 2.0
            physical_risk = time_match + 0.75 * torch.relu(closing)[:, None, :] - 0.35 * distance[:, None, :]
            risks.append(physical_risk)

        relation_tokens = torch.cat(tokens, dim=1)
        relation_masks = torch.cat(masks, dim=1)
        relation_risk = torch.cat(risks, dim=2)
        query = self.relation_query(self.future_queries)[None, :, :].expand(batch * targets, -1, -1)
        key = self.relation_key(relation_tokens)
        scores = torch.einsum("bhd,bkd->bhk", query, key) / math.sqrt(key.shape[-1])
        scores = scores + torch.nn.functional.softplus(self.relation_risk_bias_raw) * relation_risk
        scores = scores.masked_fill(~relation_masks[:, None, :], -1.0e4)
        weights = torch.softmax(scores, dim=-1) * relation_masks[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        context = torch.einsum("bhk,bkd->bhd", weights, self.relation_value(relation_tokens))
        context = self.relation_output_norm(context)
        valid = relation_masks.any(dim=-1)[:, None, None]
        masked_risk = relation_risk.masked_fill(~relation_masks[:, None, :], -1.0e4)
        maximum_risk = masked_risk.max(dim=-1).values
        # Activate relation prompts only when an interaction event lies near the
        # prediction window; far-away neighbors must not be amplified by LayerNorm.
        risk_activation = torch.sigmoid(3.0 * maximum_risk)
        risk_activation = risk_activation * relation_masks.any(dim=-1)[:, None]
        context = context * valid * risk_activation[..., None]
        self._last_relation_context = context
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=-1)
        self.last_relation_attention_entropy = entropy.detach().mean()
        self.last_relation_risk_activation = risk_activation.detach().mean()
        return torch.tanh(self.relation_prompt_strength) * context
