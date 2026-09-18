"""
Shared front-end for geometry-aware multi-BS feature fusion.
"""

import math

import torch
import torch.nn as nn

try:
    from dataset_feature_config import LEGACY_FEATURE_CONFIG
except ImportError:
    from .dataset_feature_config import LEGACY_FEATURE_CONFIG

BS_POSITIONS = torch.tensor([[0.0, 0.0], [150.0, 0.0], [75.0, 150.0]], dtype=torch.float32)


def build_motion_anchor_from_state(state_norm, pred_len, dt=0.05):
    """
    state_norm: [B, 4] = [x_norm, y_norm, vx_norm, vy_norm]
    return: [B, pred_len, 2] normalized anchor trajectory
    """
    pos = state_norm[:, :2]
    vel = state_norm[:, 2:4]
    steps = torch.arange(1, pred_len + 1, dtype=state_norm.dtype, device=state_norm.device).view(1, -1, 1)
    return pos.unsqueeze(1) + steps * dt * vel.unsqueeze(1)


class VectorTokenEncoder(nn.Module):
    def __init__(self, in_dim, token_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GeometryEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        coords = []
        for x, y in BS_POSITIONS:
            x_norm = float(x) / 150.0
            y_norm = float(y) / 150.0
            coords.append(
                [
                    x_norm,
                    y_norm,
                    math.sin(math.pi * x_norm),
                    math.cos(math.pi * x_norm),
                    math.sin(math.pi * y_norm),
                    math.cos(math.pi * y_norm),
                ]
            )
        self.register_buffer("geo_inputs", torch.tensor(coords, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(6, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self):
        return self.net(self.geo_inputs)


class GeometryConditionedFusion(nn.Module):
    """
    Weighted fusion + attention refinement for one modality across BSs.
    """

    def __init__(self, dim, bs_count=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.bs_count = bs_count
        self.score_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.query_proj = nn.Linear(dim, dim)
        self.refine_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bs_tokens):
        logits = self.score_net(bs_tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=2)
        fused = torch.sum(weights.unsqueeze(-1) * bs_tokens, dim=2)

        batch_size, seq_len, _, dim = bs_tokens.shape
        query = self.query_proj(fused).reshape(batch_size * seq_len, 1, dim)
        key_value = bs_tokens.reshape(batch_size * seq_len, self.bs_count, dim)
        refined, _ = self.refine_attn(query, key_value, key_value)
        refined = self.norm(query + refined).reshape(batch_size, seq_len, dim)

        return self.out_proj(fused + refined), weights


class SequenceContextAggregator(nn.Module):
    """
    Aggregate a temporal token sequence with global and recent attention pooling.
    """

    def __init__(self, dim, recent_steps=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.recent_steps = recent_steps
        self.global_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.recent_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.recent_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, seq_tokens):
        batch_size = seq_tokens.shape[0]
        global_query = self.global_query.expand(batch_size, -1, -1)
        global_ctx, _ = self.global_attn(global_query, seq_tokens, seq_tokens)

        recent_steps = min(self.recent_steps, seq_tokens.shape[1])
        recent_tokens = seq_tokens[:, -recent_steps:, :]
        recent_query = self.recent_query.expand(batch_size, -1, -1)
        recent_ctx, _ = self.recent_attn(recent_query, recent_tokens, recent_tokens)
        merged = torch.cat([global_ctx.squeeze(1), recent_ctx.squeeze(1)], dim=-1)
        return self.out_proj(merged)


class StateSummaryEstimator(nn.Module):
    """
    Estimate current state and short-term motion summary from fused sensing history.
    """

    def __init__(self, token_dim, hidden_dim=256, recent_steps=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.aggregator = SequenceContextAggregator(
            token_dim,
            recent_steps=recent_steps,
            num_heads=num_heads,
            dropout=dropout,
        )
        core_dim = token_dim * 2
        self.core_proj = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim + 8, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, sensor_steps):
        last_step = sensor_steps[:, -1, :]
        context = self.aggregator(sensor_steps)
        core = self.core_proj(torch.cat([last_step, context], dim=-1))
        state_pred = self.state_head(core)
        summary_pred = self.summary_head(core)
        state_token = self.token_head(torch.cat([core, state_pred, summary_pred], dim=-1))
        return {
            "state_pred": state_pred,
            "summary_pred": summary_pred,
            "state_token": state_token,
            "state_context": core,
        }


class StructuredISACFrontEnd(nn.Module):
    BS_COUNT = 3
    DIST_DIM = LEGACY_FEATURE_CONFIG.dist_dim
    VEL_DIM = LEGACY_FEATURE_CONFIG.vel_dim
    ANG_DIM = LEGACY_FEATURE_CONFIG.ang_dim
    BS_FEATURE_DIM = DIST_DIM + VEL_DIM + ANG_DIM
    RF_TOTAL = BS_COUNT * BS_FEATURE_DIM
    N_STAT = 12

    def __init__(self, configs):
        super().__init__()
        dropout = configs.dropout
        self.bs_hidden = getattr(configs, "bs_hidden", 64)
        self.token_dim = getattr(configs, "rf_token_dim", 256)
        self.bs_dropout_prob = getattr(configs, "bs_dropout", 0.0)
        self.rf_noise_std = getattr(configs, "rf_noise_std", 0.0)
        self.out_dim = self.token_dim
        self.cross_attn_heads = getattr(configs, "cross_attn_heads", 4)
        self.dist_dim = int(getattr(configs, "dist_dim", self.DIST_DIM))
        self.vel_dim = int(getattr(configs, "vel_dim", self.VEL_DIM))
        self.ang_dim = int(getattr(configs, "ang_dim", self.ANG_DIM))
        self.bs_feature_dim = int(getattr(configs, "bs_feature_dim", self.dist_dim + self.vel_dim + self.ang_dim))
        self.rf_total = int(getattr(configs, "rf_total_dim", self.BS_COUNT * self.bs_feature_dim))
        self.DIST_DIM = self.dist_dim
        self.VEL_DIM = self.vel_dim
        self.ANG_DIM = self.ang_dim
        self.BS_FEATURE_DIM = self.bs_feature_dim
        self.RF_TOTAL = self.rf_total

        self.geo_encoder = GeometryEncoder(self.bs_hidden)
        self.dist_encoder = VectorTokenEncoder(self.dist_dim, self.bs_hidden, dropout=dropout)
        self.vel_encoder = VectorTokenEncoder(self.vel_dim, self.bs_hidden, dropout=dropout)
        self.ang_encoder = VectorTokenEncoder(self.ang_dim, self.bs_hidden, dropout=dropout)

        self.dist_fusion = GeometryConditionedFusion(self.bs_hidden, bs_count=self.BS_COUNT, num_heads=self.cross_attn_heads, dropout=dropout)
        self.vel_fusion = GeometryConditionedFusion(self.bs_hidden, bs_count=self.BS_COUNT, num_heads=self.cross_attn_heads, dropout=dropout)
        self.ang_fusion = GeometryConditionedFusion(self.bs_hidden, bs_count=self.BS_COUNT, num_heads=self.cross_attn_heads, dropout=dropout)

        self.dist_out = self._make_token_head(dropout)
        self.vel_out = self._make_token_head(dropout)
        self.ang_out = self._make_token_head(dropout)
        self.sensor_fuse = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("bs_geo_features", self._compute_bs_geometry())

    def _make_token_head(self, dropout):
        return nn.Sequential(
            nn.Linear(self.bs_hidden, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
        )

    def _compute_bs_geometry(self):
        center = torch.tensor([75.0, 75.0], dtype=torch.float32)
        d01 = torch.norm(BS_POSITIONS[0] - BS_POSITIONS[1]) / 150.0
        d02 = torch.norm(BS_POSITIONS[0] - BS_POSITIONS[2]) / 150.0
        d12 = torch.norm(BS_POSITIONS[1] - BS_POSITIONS[2]) / 150.0
        c0 = torch.norm(BS_POSITIONS[0] - center) / 150.0
        c1 = torch.norm(BS_POSITIONS[1] - center) / 150.0
        c2 = torch.norm(BS_POSITIONS[2] - center) / 150.0
        return torch.tensor([d01, d02, d12, c0, c1, c2], dtype=torch.float32)

    def _reshape_rf(self, x_enc):
        batch_size, seq_len, _ = x_enc.shape
        rf = x_enc[:, :, : self.rf_total].contiguous()
        return rf.view(batch_size, seq_len, self.BS_COUNT, self.bs_feature_dim)

    def _condition_bs_tokens(self, tokens):
        geo = self.geo_encoder().to(tokens.device).view(1, 1, self.BS_COUNT, self.bs_hidden)
        return tokens + geo

    def compute_signal_stats(self, rf):
        batch_size = rf.shape[0]
        dist = rf[:, :, :, : self.dist_dim]
        vel = rf[:, :, :, self.dist_dim : self.dist_dim + self.vel_dim]
        ang = rf[:, :, :, self.dist_dim + self.vel_dim :]
        stats = torch.stack(
            [
                dist.abs().mean(dim=(1, 2, 3)),
                dist.std(dim=(1, 2, 3)),
                vel.abs().mean(dim=(1, 2, 3)),
                vel.std(dim=(1, 2, 3)),
                ang.abs().mean(dim=(1, 2, 3)),
                ang.std(dim=(1, 2, 3)),
            ],
            dim=-1,
        )
        geo = self.bs_geo_features.unsqueeze(0).expand(batch_size, -1).to(rf.device)
        return torch.cat([stats, geo], dim=-1)

    def forward(self, x_enc):
        batch_size = x_enc.shape[0]
        device = x_enc.device
        rf = self._reshape_rf(x_enc)

        if self.training and self.rf_noise_std > 0:
            rf = rf + torch.randn_like(rf) * self.rf_noise_std
        if self.training and self.bs_dropout_prob > 0:
            keep = (torch.rand(batch_size, 1, self.BS_COUNT, 1, device=device) > self.bs_dropout_prob).float()
            rf = rf * keep

        dist_tokens = self._condition_bs_tokens(self.dist_encoder(rf[:, :, :, : self.dist_dim]))
        vel_tokens = self._condition_bs_tokens(self.vel_encoder(rf[:, :, :, self.dist_dim : self.dist_dim + self.vel_dim]))
        ang_tokens = self._condition_bs_tokens(self.ang_encoder(rf[:, :, :, self.dist_dim + self.vel_dim :]))

        dist_fused, dist_weights = self.dist_fusion(dist_tokens)
        vel_fused, vel_weights = self.vel_fusion(vel_tokens)
        ang_fused, ang_weights = self.ang_fusion(ang_tokens)

        dist_steps = self.dist_out(dist_fused)
        vel_steps = self.vel_out(vel_fused)
        ang_steps = self.ang_out(ang_fused)
        modality_tokens = torch.stack([dist_steps, vel_steps, ang_steps], dim=2)
        sensor_steps = self.sensor_fuse(torch.cat([dist_steps, vel_steps, ang_steps], dim=-1))

        return {
            "rf_hist": rf,
            "dist_bs_tokens": dist_tokens,
            "vel_bs_tokens": vel_tokens,
            "ang_bs_tokens": ang_tokens,
            "dist_steps": dist_steps,
            "vel_steps": vel_steps,
            "ang_steps": ang_steps,
            "modality_tokens": modality_tokens,
            "sensor_steps": sensor_steps,
            "step_context": sensor_steps,
            "fusion_weights": {
                "dist": dist_weights,
                "vel": vel_weights,
                "ang": ang_weights,
            },
        }
