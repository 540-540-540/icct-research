"""Shared GPT-2 + LoRA downstream for every Q0 graph core."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Model

from .contracts import CANONICAL_DT, HISTORY_LENGTH, PREDICTION_LENGTH


class FusedQKVLoRA(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.lora_A = nn.Parameter(base.weight.new_empty(base.weight.shape[0], rank))
        self.lora_B = nn.Parameter(base.weight.new_zeros(rank, base.weight.shape[1]))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)


class UnifiedTokenAdapter(nn.Module):
    def __init__(self, graph_dim: int, llm_dim: int = 768) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.state_projection = nn.Linear(4, llm_dim)
        self.graph_projection = nn.Linear(graph_dim, llm_dim, bias=False)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, standardized_state: torch.Tensor, graph_features: torch.Tensor) -> torch.Tensor:
        if graph_features.shape[:-1] != standardized_state.shape[:-1]:
            raise ValueError("State and graph token grids must match")
        token = self.state_projection(standardized_state) + self.graph_projection(graph_features)
        return self.norm(token)


def constant_velocity_baseline(state_hat: torch.Tensor) -> torch.Tensor:
    origin = state_hat[:, -1]
    horizon = (
        torch.arange(
            1,
            PREDICTION_LENGTH + 1,
            device=state_hat.device,
            dtype=state_hat.dtype,
        )[None, :, None, None]
        * CANONICAL_DT
    )
    return origin[:, None, :, :2] + horizon * origin[:, None, :, 2:4]


class SharedTrajectoryPredictor(nn.Module):
    """Vehicle-wise temporal predictor; cross-vehicle information enters only through graph_features."""

    def __init__(
        self,
        graph_dim: int,
        *,
        checkpoint: str | Path | None = None,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        checkpoint = checkpoint or Path(__file__).resolve().parents[2] / "models/gpt2"
        self.adapter = UnifiedTokenAdapter(graph_dim)
        self.llm = GPT2Model.from_pretrained(
            str(checkpoint), local_files_only=True, attn_implementation="eager"
        )
        if (self.llm.config.n_layer, self.llm.config.n_head, self.llm.config.n_embd) != (12, 12, 768):
            raise ValueError("Q0 expects GPT-2 base 12/12/768")
        self.llm.requires_grad_(False)
        self.llm.config.use_cache = False
        for block in self.llm.h:
            block.attn.c_attn = FusedQKVLoRA(block.attn.c_attn, rank=lora_rank)

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, PREDICTION_LENGTH * 2),
        )
        # Every graph method starts from exactly the same CV predictor.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        state_hat: torch.Tensor,
        standardized_state: torch.Tensor,
        vehicle_mask: torch.Tensor,
        graph_features: torch.Tensor,
        token_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, n, _ = state_hat.shape
        if t != HISTORY_LENGTH:
            raise ValueError("Unexpected history length")

        tokens = self.adapter(standardized_state, graph_features)
        if token_delta is not None:
            if token_delta.shape != tokens.shape:
                raise ValueError("token_delta must be [B,T,N,768]")
            tokens = tokens + token_delta
        flat_tokens = tokens.permute(0, 2, 1, 3).reshape(b * n, t, 768)
        active = vehicle_mask.reshape(-1).nonzero(as_tuple=True)[0]

        correction = flat_tokens.new_zeros((b * n, PREDICTION_LENGTH, 2))
        if active.numel():
            hidden = self.llm(inputs_embeds=flat_tokens[active], use_cache=False).last_hidden_state[:, -1]
            delta = self.head(hidden).reshape(-1, PREDICTION_LENGTH, 2)
            correction = correction.index_copy(0, active, delta)

        correction = correction.reshape(b, n, PREDICTION_LENGTH, 2).permute(0, 2, 1, 3)
        prediction = constant_velocity_baseline(state_hat) + correction
        prediction = torch.where(
            vehicle_mask[:, None, :, None], prediction, torch.zeros_like(prediction)
        )
        return {
            "prediction": prediction,
            "origin_eligible": vehicle_mask,
            "correction": correction,
        }

