"""Own-history temporal baselines for the Raj residual Self stage."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from prediction.q0.contracts import CANONICAL_DT
from prediction.q0.temporal import constant_velocity_baseline


class CausalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.padding = 2 * int(dilation)
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def _conv(self, conv, x):
        return conv(F.pad(x, (self.padding, 0)))

    def forward(self, x):
        residual = x
        x = self._conv(self.conv1, x).transpose(1, 2)
        x = F.gelu(self.norm1(x)).transpose(1, 2)
        x = self._conv(self.conv2, x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        return F.gelu(x + residual)


class SelfTemporalBaseline(nn.Module):
    """LSTM, Transformer, or TCN with the same own-history CV-residual contract."""

    def __init__(self, kind: str, width: int = 192, correction_cap_m: float = 16.0):
        super().__init__()
        if kind not in {"lstm", "transformer", "tcn"}:
            raise ValueError(kind)
        self.kind = kind
        self.width = int(width)
        self.correction_cap_m = float(correction_cap_m)
        self.input_projection = nn.Sequential(
            nn.Linear(4, width), nn.LayerNorm(width), nn.GELU()
        )
        self.future_queries = nn.Parameter(torch.randn(20, width) * 0.02)

        if kind == "lstm":
            self.encoder = nn.LSTM(width, width, 2, batch_first=True)
            self.decoder = nn.LSTM(width, width, 2, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                width,
                nhead=6,
                dim_feedforward=4 * width,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.position = nn.Parameter(torch.randn(40, width) * 0.02)
            self.encoder = nn.TransformerEncoder(
                layer, 4, enable_nested_tensor=False
            )
        else:
            self.tcn = nn.ModuleList(
                CausalResidualBlock(width, dilation) for dilation in (1, 2, 4, 8)
            )
            self.future_adapter = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
            )

        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _motion_features(history):
        displacement = torch.zeros_like(history[..., :2])
        displacement[:, 1:] = history[:, 1:, :2] - history[:, :-1, :2]
        return torch.cat((displacement, history[..., 2:4]), -1)

    def _future_states(self, history):
        encoded = self.input_projection(self._motion_features(history))
        future_queries = self.future_queries[None].expand(len(history), -1, -1)
        if self.kind == "lstm":
            _, state = self.encoder(encoded)
            future, _ = self.decoder(future_queries, state)
            return future
        if self.kind == "transformer":
            sequence = torch.cat((encoded, future_queries), 1) + self.position[None]
            return self.encoder(sequence)[:, -20:]
        x = encoded.transpose(1, 2)
        for block in self.tcn:
            x = block(x)
        return self.future_adapter(x[:, :, -1][:, None] + future_queries)

    def forward(self, history, mask, timestamps=None):
        if history.ndim != 4 or history.shape[1] != 20 or history.shape[-1] != 4:
            raise ValueError("Expected history [B,20,N,4]")
        if mask.shape != (history.shape[0], history.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean mask [B,N]")
        history = torch.where(mask[:, None, :, None], history, 0.0)
        b, t, n, _ = history.shape
        flat = history.permute(0, 2, 1, 3).reshape(b * n, t, 4)
        active = mask.reshape(-1).nonzero(as_tuple=True)[0]
        prediction = history.new_zeros((b * n, 20, 2))
        if active.numel():
            own_history = flat[active]
            correction = torch.tanh(
                self.head(self._future_states(own_history))
            ) * self.correction_cap_m
            cv = constant_velocity_baseline(
                own_history.reshape(len(active), 20, 1, 4)
            )[:, :, 0]
            prediction = prediction.index_copy(0, active, cv + correction)
        prediction = prediction.reshape(b, n, 20, 2).permute(0, 2, 1, 3)
        if timestamps is not None:
            dt = ((timestamps[:, -1] - timestamps[:, 0]) / 19).to(history.dtype)
            step = torch.arange(1, 21, device=history.device, dtype=history.dtype)
            timing = (
                step[None, :, None, None]
                * (dt - CANONICAL_DT)[:, None, None, None]
                * history[:, -1:, :, 2:]
            )
            prediction = prediction + timing * mask[:, None, :, None]
        return {"prediction": prediction, "origin_eligible": mask}

    def parameter_summary(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": total, "total": total}


def build_self_baseline(kind: str, seed: int = 2026):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 300003)
        return SelfTemporalBaseline(kind)
