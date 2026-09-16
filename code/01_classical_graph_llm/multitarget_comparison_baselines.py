"""Conventional LSTM, TCN and Transformer baselines for multi-target scenes.

Each model predicts every target independently and uses the same normalized
state features, constant-velocity anchor and multi-step decoder.  This keeps the
comparison focused on the temporal encoder and matches the baseline families in
the first paper while using the second-point data interface.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from target_interaction_graph import ForecasterConfig


def target_history_features(history: torch.Tensor, config: ForecasterConfig) -> torch.Tensor:
    last_position = history[:, -1, :, :2]
    relative_position = (history[..., :2] - last_position[:, None, :, :]) / config.position_scale
    velocity = history[..., 2:4] / config.velocity_scale
    acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1]) / config.dt
    return torch.cat([relative_position, velocity, acceleration], dim=-1)


class IndependentTemporalForecaster(nn.Module):
    def __init__(self, config: ForecasterConfig, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.decoder_time_embedding = nn.Embedding(config.prediction_length, 24)
        decoder_hidden = min(max(latent_dim, 128), 384)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 24 + 2, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.GELU(),
            nn.Linear(decoder_hidden // 2, 2),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, target_history: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, history: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, history_length, targets, _ = history.shape
        features = target_history_features(history, self.config)
        target_major = features.permute(0, 2, 1, 3).reshape(batch * targets, history_length, 6)
        latent = self.encode(target_major).view(batch, targets, self.latent_dim)
        latent = latent * target_mask[..., None]

        last_position = history[:, -1, :, :2]
        last_velocity = history[:, -1, :, 2:4]
        steps = torch.arange(self.config.prediction_length, device=history.device)
        times = (steps.to(history.dtype) + 1.0) * self.config.dt
        constant_velocity = last_velocity[:, :, None, :] * times[None, None, :, None]
        time_features = self.decoder_time_embedding(steps)[None, None].expand(batch, targets, -1, -1)
        decoder_input = torch.cat(
            [
                latent[:, :, None].expand(-1, -1, self.config.prediction_length, -1),
                time_features,
                constant_velocity / self.config.position_scale,
            ],
            dim=-1,
        )
        residual = self.decoder(decoder_input) * self.config.residual_scale
        displacement = (constant_velocity + residual).permute(0, 2, 1, 3)
        displacement = displacement * target_mask[:, None, :, None]
        future_position = last_position[:, None] + displacement
        return {"future_position": future_position, "displacement": displacement}


class MultiTargetLSTMBaseline(IndependentTemporalForecaster):
    def __init__(self, config: ForecasterConfig, hidden_dim: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__(config, latent_dim=hidden_dim, dropout=dropout)
        self.encoder = nn.LSTM(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def encode(self, target_history: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(target_history)
        return self.norm(hidden[-1])


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def _causal(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., :-self.padding] if self.padding > 0 else tensor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.activation(self._causal(self.conv1(inputs))))
        output = self.dropout(self.activation(self._causal(self.conv2(output))))
        return self.activation(output + self.residual(inputs))


class MultiTargetTCNBaseline(IndependentTemporalForecaster):
    def __init__(
        self,
        config: ForecasterConfig,
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(config, latent_dim=int(channels[-1]), dropout=dropout)
        blocks = []
        for level, out_channels in enumerate(channels):
            in_channels = 6 if level == 0 else int(channels[level - 1])
            blocks.append(CausalTemporalBlock(in_channels, int(out_channels), kernel_size, 2 ** level, dropout))
        self.network = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(int(channels[-1]))

    def encode(self, target_history: torch.Tensor) -> torch.Tensor:
        encoded = self.network(target_history.transpose(1, 2)).transpose(1, 2)
        return self.norm(encoded[:, -1])


class MultiTargetTransformerBaseline(IndependentTemporalForecaster):
    def __init__(
        self,
        config: ForecasterConfig,
        d_model: int = 768,
        heads: int = 12,
        layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__(config, latent_dim=d_model, dropout=dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(6, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.history_time_embedding = nn.Embedding(config.history_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
    def encode(self, target_history: torch.Tensor) -> torch.Tensor:
        length = target_history.shape[1]
        steps = torch.arange(length, device=target_history.device)
        embedded = self.input_projection(target_history) + self.history_time_embedding(steps)[None]
        causal_mask = torch.triu(
            torch.full((length, length), float("-inf"), device=target_history.device), diagonal=1
        )
        return self.encoder(embedded, mask=causal_mask)[:, -1]
