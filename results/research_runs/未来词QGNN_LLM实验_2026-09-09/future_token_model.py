"""Future-token QGNN+LLM model with an exact retained-model fallback."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from model import CoreGraphLLM


class FutureTokenCoreGraphLLM(CoreGraphLLM):
    """Use GPT-2 future states and motion-token rollout without losing the baseline.

    The retained QGNN+LLM path is kept unchanged.  A separate LoRA GPT-2 branch
    receives both history tokens and causal future-query tokens.  Its token
    distribution is differentiably decoded into coordinates.  A zero-initialized
    scalar gate makes the initial prediction exactly equal to the retained model.
    """

    def __init__(self, graph):
        super().__init__(graph)
        # The original GPT path stays fixed and reproduces the retained checkpoint.
        for parameter in self.gpt2.parameters():
            parameter.requires_grad_(False)

        # A distinct future branch lets LoRA learn future-token reasoning without
        # changing the retained coordinate path used by the exact fallback.
        self.future_gpt2 = copy.deepcopy(self.gpt2)
        for name, parameter in self.future_gpt2.named_parameters():
            parameter.requires_grad_("lora_" in name)

        h = self.config.hidden_dim
        self.future_type_embedding = nn.Parameter(torch.zeros(self.d_llm))
        self.token_residual_head = nn.Sequential(
            nn.LayerNorm(self.d_llm + h),
            nn.Linear(self.d_llm + h, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.token_residual_head[-1].weight)
        nn.init.zeros_(self.token_residual_head[-1].bias)
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(self.d_llm + h),
            nn.Linear(self.d_llm + h, 1),
        )
        nn.init.zeros_(self.adaptive_gate[-1].weight)
        nn.init.zeros_(self.adaptive_gate[-1].bias)
        self.fusion_strength = nn.Parameter(torch.zeros(()))
        self.token_residual_scale_m = 0.75
        # Future-branch-only switches used by controlled ablations.  The
        # retained coordinate path always stays intact for an exact epoch-0
        # fallback to the same accepted QGNN+LLM checkpoint.
        self.future_use_motion_tokens = True
        self.future_use_graph_token = True
        self.future_use_queries = True

    def initialize_future_branch_from_baseline(self):
        self.future_gpt2.load_state_dict(self.gpt2.state_dict(), strict=True)
        for name, parameter in self.future_gpt2.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def _future_query_addition(self, history, graph_output, flat_nodes):
        """Extension point for graph-relation prompts; base route adds nothing."""
        return flat_nodes.new_zeros(
            flat_nodes.shape[0], self.config.prediction_length, self.d_llm
        )

    def forward(self, history, target_mask):
        batch, history_length, targets, _ = history.shape
        graph_output = self.graph_backbone(history, target_mask)
        flat_nodes = graph_output["node_features"].reshape(batch * targets, -1)
        target_history = history.permute(0, 2, 1, 3).reshape(batch * targets, history_length, 4)
        soft = self._soft_history_embeddings(target_history, flat_nodes)
        graph_token = self.graph_projection(flat_nodes).unsqueeze(1)
        motion = soft["embeddings"] + graph_token
        history_sequence = torch.cat(
            [graph_token, motion + self.token_adapter(motion)], dim=1
        )

        # Retained coordinate path: identical operations to CoreGraphLLM.
        retained_hidden = self.gpt2(inputs_embeds=history_sequence).last_hidden_state[:, -1]
        retained_correction = self.coordinate_head(torch.cat([retained_hidden, flat_nodes], dim=-1))
        retained_correction = torch.tanh(
            retained_correction.view(batch, targets, self.config.prediction_length, 2)
        ) * self.correction_scale_m
        retained_correction = retained_correction.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        retained_future = graph_output["future_position"] + retained_correction

        # New LLM path: future queries are inside GPT-2, so each future state can
        # attend to the graph/history and all earlier future-query states.
        future_graph_token = graph_token if self.future_use_graph_token else torch.zeros_like(graph_token)
        if self.future_use_motion_tokens:
            future_motion = soft["embeddings"] + future_graph_token
            future_history_sequence = torch.cat(
                [future_graph_token, future_motion + self.token_adapter(future_motion)], dim=1
            )
        else:
            future_history_sequence = future_graph_token
        query_content = self.future_queries if self.future_use_queries else torch.zeros_like(self.future_queries)
        future_inputs = (
            query_content[None, :, :]
            + self.future_type_embedding[None, None, :]
            + future_graph_token
            + self._future_query_addition(history, graph_output, flat_nodes)
        )
        full_sequence = torch.cat([future_history_sequence, future_inputs], dim=1)
        future_hidden = self.future_gpt2(inputs_embeds=full_sequence).last_hidden_state[
            :, -self.config.prediction_length :, :
        ]
        token_logits_flat = self.token_head(future_hidden)
        token_probs_flat = torch.softmax(token_logits_flat, dim=-1)

        start_pos = history[:, -1, :, :2].reshape(batch * targets, 2)
        start_velocity = history[:, -1, :, 2:4].reshape(batch * targets, 2)
        start_heading = torch.atan2(start_velocity[:, 1], start_velocity[:, 0])
        token_trajectory = self.motion_tokenizer.soft_decode(
            start_pos, start_heading, token_probs_flat
        )
        future_nodes = flat_nodes if self.future_use_graph_token else torch.zeros_like(flat_nodes)
        step_nodes = future_nodes[:, None, :].expand(-1, self.config.prediction_length, -1)
        step_context = torch.cat([future_hidden, step_nodes], dim=-1)
        token_residual = torch.tanh(self.token_residual_head(step_context)) * self.token_residual_scale_m
        token_candidate = token_trajectory + token_residual

        retained_flat = retained_future.permute(0, 2, 1, 3).reshape(
            batch * targets, self.config.prediction_length, 2
        )
        local_gate = torch.sigmoid(self.adaptive_gate(step_context))
        effective_gate = torch.tanh(self.fusion_strength) * local_gate
        fused_flat = retained_flat + effective_gate * (token_candidate - retained_flat)
        fused = fused_flat.view(batch, targets, self.config.prediction_length, 2).permute(0, 2, 1, 3)
        fused = fused * target_mask[:, None, :, None]

        token_logits = token_logits_flat.view(
            batch, targets, self.config.prediction_length, self.vocab_size
        ).permute(0, 2, 1, 3)
        token_position = token_candidate.view(
            batch, targets, self.config.prediction_length, 2
        ).permute(0, 2, 1, 3)
        token_position = token_position * target_mask[:, None, :, None]
        return {
            "future_position": fused,
            "displacement": fused - history[:, -1, None, :, :2],
            "retained_future_position": retained_future,
            "token_future_position": token_position,
            "graph_future_position": graph_output["future_position"],
            "token_logits": token_logits,
            "token_temperature": soft["temperature"],
            "fusion_strength": torch.tanh(self.fusion_strength),
            "fusion_gate_mean": effective_gate.mean(),
            "future_hidden_features": future_hidden,
            "node_features": graph_output["node_features"],
            "attention": graph_output["attention"],
            "adjacency": graph_output["adjacency"],
        }


def future_compact_state(model):
    state = {}
    for name, tensor in model.state_dict().items():
        frozen_original = name.startswith("gpt2.") and "lora_" not in name
        frozen_future = name.startswith("future_gpt2.") and "lora_" not in name
        if not frozen_original and not frozen_future:
            state[name] = tensor.detach().cpu().clone()
    return state


def restore_future_compact(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    allowed = [
        name for name in missing
        if (name.startswith("gpt2.") or name.startswith("future_gpt2.")) and "lora_" not in name
    ]
    assert len(allowed) == len(missing), missing
