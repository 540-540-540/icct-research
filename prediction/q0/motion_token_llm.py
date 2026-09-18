"""Automatum motion-token GPT-2 core.

The LLM is a primary forecaster: future query states produce motion-token
probabilities and coordinate refinements. The graph core supplies indispensable
cross-vehicle context but does not directly emit the final trajectory here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from .contracts import CANONICAL_DT, HISTORY_LENGTH, PREDICTION_LENGTH
from .temporal import FusedQKVLoRA, constant_velocity_baseline


@dataclass(frozen=True)
class AutomatumMotionTokenizerConfig:
    forward_bins: int = 31
    lateral_bins: int = 31
    forward_min: float = -1.0
    forward_max: float = 3.5
    lateral_min: float = -1.8
    lateral_max: float = 1.8
    temperature_m: float = 0.18

    @property
    def vocab_size(self) -> int:
        return self.forward_bins * self.lateral_bins


class AutomatumMotionTokenizer:
    def __init__(self, config: AutomatumMotionTokenizerConfig | None = None):
        self.config = config or AutomatumMotionTokenizerConfig()

    def _tables(self, device, dtype):
        c = self.config
        f = torch.linspace(c.forward_min, c.forward_max, c.forward_bins, device=device, dtype=dtype)
        l = torch.linspace(c.lateral_min, c.lateral_max, c.lateral_bins, device=device, dtype=dtype)
        fg = f[:, None].expand(-1, c.lateral_bins)
        lg = l[None, :].expand(c.forward_bins, -1)
        return fg.reshape(-1), lg.reshape(-1)

    @staticmethod
    def _heading(velocity: torch.Tensor, displacement: torch.Tensor | None = None) -> torch.Tensor:
        heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        if displacement is not None:
            fallback = torch.atan2(displacement[..., 1], displacement[..., 0])
            slow = torch.linalg.vector_norm(velocity, dim=-1) < 0.2
            heading = torch.where(slow, fallback, heading)
        return heading

    def local_deltas(self, positions: torch.Tensor, velocities: torch.Tensor):
        displacement = positions[:, 1:] - positions[:, :-1]
        heading = self._heading(velocities[:, :-1], displacement)
        c, s = torch.cos(heading), torch.sin(heading)
        forward = displacement[..., 0] * c + displacement[..., 1] * s
        lateral = -displacement[..., 0] * s + displacement[..., 1] * c
        return forward, lateral

    def logits(self, positions: torch.Tensor, velocities: torch.Tensor, temperature: float | None = None):
        forward, lateral = self.local_deltas(positions, velocities)
        token_f, token_l = self._tables(forward.device, forward.dtype)
        tau = float(self.config.temperature_m if temperature is None else temperature)
        return -(
            ((forward[..., None] - token_f) / tau).square()
            + ((lateral[..., None] - token_l) / tau).square()
        )

    def soft_probabilities(self, positions: torch.Tensor, velocities: torch.Tensor):
        return torch.softmax(self.logits(positions, velocities), dim=-1)

    def token_ids(self, positions: torch.Tensor, velocities: torch.Tensor):
        return self.logits(positions, velocities).argmax(dim=-1)

    def expected_motion(self, probabilities: torch.Tensor):
        token_f, token_l = self._tables(probabilities.device, probabilities.dtype)
        return (probabilities * token_f).sum(dim=-1), (probabilities * token_l).sum(dim=-1)

    def soft_decode(
        self,
        start_position: torch.Tensor,
        start_velocity: torch.Tensor,
        probabilities: torch.Tensor,
    ):
        token_f, token_l = self._tables(probabilities.device, probabilities.dtype)
        forward = (probabilities * token_f).sum(dim=-1)
        lateral = (probabilities * token_l).sum(dim=-1)
        heading = self._heading(start_velocity)
        position = start_position
        trajectory = []
        for t in range(probabilities.shape[1]):
            f, l = forward[:, t], lateral[:, t]
            dx = torch.cos(heading) * f - torch.sin(heading) * l
            dy = torch.sin(heading) * f + torch.cos(heading) * l
            position = position + torch.stack([dx, dy], dim=-1)
            trajectory.append(position)
            heading = heading + torch.atan2(l, f.abs().clamp_min(1e-3))
        return torch.stack(trajectory, dim=1)


class MotionTokenGPT2Core(nn.Module):
    """Graph-conditioned motion-token GPT-2 used as the primary temporal core."""

    def __init__(
        self,
        graph_dim: int = 64,
        *,
        checkpoint: str | Path | None = None,
        llm_layers: int = 4,
        lora_rank: int = 8,
        correction_scale_m: float = 4.0,
        tokenizer_config: AutomatumMotionTokenizerConfig | None = None,
    ):
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.tokenizer = AutomatumMotionTokenizer(tokenizer_config)
        self.vocab_size = self.tokenizer.config.vocab_size
        self.correction_scale_m = float(correction_scale_m)

        checkpoint = checkpoint or Path(__file__).resolve().parents[2] / "models/gpt2"
        config = GPT2Config.from_pretrained(str(checkpoint), local_files_only=True)
        config.num_hidden_layers = int(llm_layers)
        config.use_cache = False
        self.d_llm = int(config.n_embd)
        self.gpt2 = GPT2Model.from_pretrained(
            str(checkpoint), config=config, local_files_only=True, attn_implementation="eager"
        )
        self.gpt2.requires_grad_(False)
        for block in self.gpt2.h:
            block.attn.c_attn = FusedQKVLoRA(block.attn.c_attn, rank=lora_rank, alpha=2 * lora_rank)

        self.motion_embedding = nn.Embedding(self.vocab_size, self.d_llm)
        with torch.no_grad():
            source = self.gpt2.wte.weight.detach()
            ids = torch.linspace(0, source.shape[0] - 1, self.vocab_size).round().long()
            self.motion_embedding.weight.copy_(0.20 * source.index_select(0, ids))

        self.graph_projection = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, self.d_llm),
            nn.GELU(),
        )
        self.history_adapter = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.future_queries = nn.Parameter(torch.randn(PREDICTION_LENGTH, self.d_llm) * 0.02)
        self.token_head = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.vocab_size),
        )
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(self.d_llm + graph_dim + 2),
            nn.Linear(self.d_llm + graph_dim + 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)

    def _history_motion_embeddings(self, history: torch.Tensor):
        # history [BN,T,4], transitions = T-1.
        probs = self.tokenizer.soft_probabilities(history[..., :2], history[..., 2:4])
        table = F.layer_norm(self.motion_embedding.weight, (self.d_llm,))
        embeddings = probs @ table
        return embeddings, probs

    def forward(
        self,
        state_hat: torch.Tensor,
        vehicle_mask: torch.Tensor,
        graph_features: torch.Tensor,
    ):
        b, t, n, c = state_hat.shape
        if (t, c) != (HISTORY_LENGTH, 4):
            raise ValueError("Expected [B,20,N,4] state history")
        if graph_features.shape != (b, t, n, self.graph_dim):
            raise ValueError("graph_features must be [B,20,N,graph_dim]")

        history = state_hat.permute(0, 2, 1, 3).reshape(b * n, t, 4)
        graph_node = graph_features[:, -1].reshape(b * n, self.graph_dim)
        active = vehicle_mask.reshape(-1).nonzero(as_tuple=True)[0]

        prediction = state_hat.new_zeros((b * n, PREDICTION_LENGTH, 2))
        logits_full = state_hat.new_zeros((b * n, PREDICTION_LENGTH, self.vocab_size))
        if active.numel():
            h = history[active]
            g = graph_node[active]
            graph_token = self.graph_projection(g).unsqueeze(1)
            motion_emb, _ = self._history_motion_embeddings(h)
            history_tokens = motion_emb + graph_token + self.history_adapter(motion_emb + graph_token)
            future_tokens = self.future_queries[None].expand(len(active), -1, -1)
            sequence = torch.cat([graph_token, history_tokens, future_tokens], dim=1)
            hidden = self.gpt2(inputs_embeds=sequence, use_cache=False).last_hidden_state
            future_hidden = hidden[:, -PREDICTION_LENGTH:]
            token_logits = self.token_head(future_hidden)
            token_probs = torch.softmax(token_logits, dim=-1)
            expected_f, expected_l = self.tokenizer.expected_motion(token_probs)
            expected_motion = torch.stack([expected_f, expected_l], dim=-1)
            g_expanded = g[:, None, :].expand(-1, PREDICTION_LENGTH, -1)
            correction = torch.tanh(
                self.coordinate_head(torch.cat([future_hidden, g_expanded, expected_motion], dim=-1))
            ) * self.correction_scale_m
            cv = constant_velocity_baseline(
                h.reshape(len(active), HISTORY_LENGTH, 1, 4)
            )[:, :, 0]
            trajectory = cv + correction
            prediction = prediction.index_copy(0, active, trajectory)
            logits_full = logits_full.index_copy(0, active, token_logits)

        prediction = prediction.reshape(b, n, PREDICTION_LENGTH, 2).permute(0, 2, 1, 3)
        logits_full = logits_full.reshape(b, n, PREDICTION_LENGTH, self.vocab_size).permute(0, 2, 1, 3)
        return {
            "prediction": prediction,
            "token_logits": logits_full,
            "origin_eligible": vehicle_mask,
        }

    def future_token_ids(self, history: torch.Tensor, future: torch.Tensor):
        # Build transitions [last observed -> future1 -> ... -> future20].
        b, _, n, _ = history.shape
        start = history[:, -1:].permute(0, 2, 1, 3).reshape(b * n, 1, 4)
        fut = future.permute(0, 2, 1, 3).reshape(b * n, PREDICTION_LENGTH, 4)
        states = torch.cat([start, fut], dim=1)
        ids = self.tokenizer.token_ids(states[..., :2], states[..., 2:4])
        return ids.reshape(b, n, PREDICTION_LENGTH).permute(0, 2, 1)

