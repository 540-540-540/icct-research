"""
TimeLLM V7: Oracle-aligned RF-to-state bridge.

Main stages:
1. Oracle motion-token GPT
2. Bridge-only imitation
3. Bridge + GPT future token forecasting
4. Decoded trajectory evaluation
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2Config, GPT2Model

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

try:
    from .structured_frontend import StructuredISACFrontEnd
except ImportError:
    from structured_frontend import StructuredISACFrontEnd

try:
    from .motion_tokenizer_v7 import MotionTokenizerConfig, MotionTokenizerV7
except ImportError:
    from motion_tokenizer_v7 import MotionTokenizerConfig, MotionTokenizerV7

transformers.logging.set_verbosity_error()

BS_POSITIONS = torch.tensor([[0.0, 0.0], [150.0, 0.0], [75.0, 150.0]], dtype=torch.float32)
RANGE_BINS = 32
VR_BINS = 33
PHI_BINS = 36
RANGE_MAX = 220.0
VR_MIN, VR_MAX = -20.0, 20.0


def resolve_gpt2_source(configs):
    model_name = getattr(configs, "llm_model", "gpt2")
    here = Path(__file__).resolve().parent
    candidates = [
        (Path("./gpt2"), True),
        (here / "gpt2", True),
        (Path(model_name), True),
    ]
    for path, local_only in candidates:
        if path.exists():
            return str(path), local_only
    return model_name.lower() if model_name.upper() == "GPT2" else model_name, False


def _make_centers(min_val, max_val, num_bins, dtype=torch.float32):
    return torch.linspace(min_val, max_val, steps=num_bins, dtype=dtype)


class RFGeometryBridge(nn.Module):
    def __init__(self, configs, tokenizer: MotionTokenizerV7):
        super().__init__()
        self.seq_len = configs.seq_len
        self.dt = getattr(configs, "dt", 0.05)
        self.frontend = StructuredISACFrontEnd(configs)
        self.bs_hidden = self.frontend.bs_hidden
        self.rf_token_dim = getattr(configs, "rf_token_dim", 256)
        self.d_llm = getattr(configs, "llm_dim", 768)
        self.tokenizer = tokenizer

        self.range_head = nn.Linear(self.bs_hidden, RANGE_BINS)
        self.vr_head = nn.Linear(self.bs_hidden, VR_BINS)
        self.phi_head = nn.Linear(self.bs_hidden, PHI_BINS)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.map_dist_proj = nn.Linear(64, self.bs_hidden)
        self.map_vel_proj = nn.Linear(64, self.bs_hidden)
        self.map_ang_proj = nn.Linear(64, self.bs_hidden)

        self.rf_latent_encoder = nn.Sequential(
            nn.Linear(self.bs_hidden * 3, self.rf_token_dim),
            nn.GELU(),
            nn.LayerNorm(self.rf_token_dim),
        )
        self.bridge_adapter = nn.Sequential(
            nn.Linear(self.rf_token_dim + 5, self.rf_token_dim),
            nn.GELU(),
            nn.Dropout(getattr(configs, "dropout", 0.1)),
            nn.Linear(self.rf_token_dim, self.rf_token_dim),
            nn.LayerNorm(self.rf_token_dim),
        )

        self.register_buffer("bs_positions", BS_POSITIONS.clone(), persistent=False)
        self.register_buffer("range_centers", _make_centers(0.0, RANGE_MAX, RANGE_BINS), persistent=False)
        self.register_buffer("vr_centers", _make_centers(VR_MIN, VR_MAX, VR_BINS), persistent=False)
        self.register_buffer("phi_centers", _make_centers(-math.pi, math.pi, PHI_BINS), persistent=False)

        self.temporal_encoder = nn.GRU(
            input_size=self.rf_token_dim + 5,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.state_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )
        self.motion_factor_backbone = nn.Sequential(
            nn.Linear(256 * 2, 192),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.forward_head = nn.Linear(192, tokenizer.config.forward_bins)
        self.lateral_head = nn.Linear(192, tokenizer.config.lateral_bins)
        self.forward_reg_head = nn.Linear(192, 1)
        self.motion_embed_head = nn.Sequential(
            nn.Linear(192, self.d_llm),
            nn.LayerNorm(self.d_llm),
        )

    @staticmethod
    def _continuous_to_index(values, centers):
        diff = torch.abs(values.unsqueeze(-1) - centers.view(*([1] * values.ndim), -1))
        return torch.argmin(diff, dim=-1)

    @staticmethod
    def _soft_expected(probs, centers):
        shape = [1] * (probs.ndim - 1) + [-1]
        return torch.sum(probs * centers.view(*shape), dim=-1)

    @staticmethod
    def _entropy(probs):
        return -torch.sum(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8)), dim=-1)

    def _measurement_posteriors(self, front):
        r_logits = self.range_head(front["dist_bs_tokens"])
        vr_logits = self.vr_head(front["vel_bs_tokens"])
        phi_logits = self.phi_head(front["ang_bs_tokens"])

        r_probs = F.softmax(r_logits, dim=-1)
        vr_probs = F.softmax(vr_logits, dim=-1)
        phi_probs = F.softmax(phi_logits, dim=-1)

        r_exp = self._soft_expected(r_probs, self.range_centers.to(r_probs.device))
        vr_exp = self._soft_expected(vr_probs, self.vr_centers.to(vr_probs.device))
        phi_exp = self._soft_expected(phi_probs, self.phi_centers.to(phi_probs.device))

        peak = (r_probs.max(dim=-1).values + vr_probs.max(dim=-1).values + phi_probs.max(dim=-1).values) / 3.0
        entropy = (self._entropy(r_probs) + self._entropy(vr_probs) + self._entropy(phi_probs)) / 3.0
        conf = peak / (1.0 + 0.3 * entropy)

        return {
            "r_logits": r_logits,
            "vr_logits": vr_logits,
            "phi_logits": phi_logits,
            "r_probs": r_probs,
            "vr_probs": vr_probs,
            "phi_probs": phi_probs,
            "r_exp": r_exp,
            "vr_exp": vr_exp,
            "phi_exp": phi_exp,
            "conf": conf.clamp_min(1e-4),
        }

    def _front_tokens_from_maps(self, rf_maps):
        rd = rf_maps["rd"]
        ra = rf_maps["ra"]
        if rd.ndim != 6 or ra.ndim != 6:
            raise ValueError(f"RF map shapes must be [B,T,BS,C,H,W], got rd={tuple(rd.shape)}, ra={tuple(ra.shape)}")
        if ra.shape[-1] < rd.shape[-1]:
            pad_total = rd.shape[-1] - ra.shape[-1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            ra = F.pad(ra, (pad_left, pad_right, 0, 0))
        elif ra.shape[-1] > rd.shape[-1]:
            ra = ra[..., : rd.shape[-1]]

        rf_map = torch.cat([rd, ra], dim=3)  # [B,T,BS,4,H,W]
        B, T, W, C, H, WW = rf_map.shape
        feat = self.map_encoder(rf_map.reshape(B * T * W, C, H, WW)).flatten(1)
        dist = self.map_dist_proj(feat).view(B, T, W, self.bs_hidden)
        vel = self.map_vel_proj(feat).view(B, T, W, self.bs_hidden)
        ang = self.map_ang_proj(feat).view(B, T, W, self.bs_hidden)
        return {
            "dist_bs_tokens": dist,
            "vel_bs_tokens": vel,
            "ang_bs_tokens": ang,
        }

    def _extract_front_tokens(self, x_enc, rf_maps=None):
        if rf_maps is not None:
            return self._front_tokens_from_maps(rf_maps)
        return self.frontend(x_enc)

    def _rf_latent_tokens(self, front, meas):
        rf_feat = torch.cat(
            [
                front["dist_bs_tokens"],
                front["vel_bs_tokens"],
                front["ang_bs_tokens"],
            ],
            dim=-1,
        )
        rf_latent = self.rf_latent_encoder(rf_feat)
        meas_stats = torch.stack(
            [
                meas["r_exp"] / RANGE_MAX,
                meas["vr_exp"] / max(abs(VR_MIN), abs(VR_MAX)),
                torch.sin(meas["phi_exp"]),
                torch.cos(meas["phi_exp"]),
                meas["conf"],
            ],
            dim=-1,
        )
        adapted_bs = self.bridge_adapter(torch.cat([rf_latent, meas_stats], dim=-1))
        weights = meas["conf"] / meas["conf"].sum(dim=2, keepdim=True).clamp_min(1e-4)
        fused_rf = torch.sum(adapted_bs * weights.unsqueeze(-1), dim=2)
        return rf_latent, adapted_bs, fused_rf

    def _temporal_features(self, meas, fused_rf):
        weights = meas["conf"] / meas["conf"].sum(dim=2, keepdim=True).clamp_min(1e-4)
        r_mean = torch.sum(weights * (meas["r_exp"] / RANGE_MAX), dim=2)
        vr_mean = torch.sum(weights * (meas["vr_exp"] / max(abs(VR_MIN), abs(VR_MAX))), dim=2)
        sin_phi = torch.sum(weights * torch.sin(meas["phi_exp"]), dim=2)
        cos_phi = torch.sum(weights * torch.cos(meas["phi_exp"]), dim=2)
        conf_mean = meas["conf"].mean(dim=2)
        temporal_in = torch.cat(
            [
                fused_rf,
                r_mean.unsqueeze(-1),
                vr_mean.unsqueeze(-1),
                sin_phi.unsqueeze(-1),
                cos_phi.unsqueeze(-1),
                conf_mean.unsqueeze(-1),
            ],
            dim=-1,
        )
        h, _ = self.temporal_encoder(temporal_in)
        return h

    def _state_from_temporal(self, temporal_context):
        raw = self.state_head(temporal_context)
        pos_ref = 150.0 * torch.sigmoid(raw[..., :2])
        vel_ref = 20.0 * torch.tanh(raw[..., 2:4])
        heading_ref = torch.atan2(raw[..., 4], raw[..., 5])
        return torch.cat([pos_ref, vel_ref, heading_ref.unsqueeze(-1)], dim=-1)

    def _bridge_motion_outputs(self, temporal_context):
        pair_feat = torch.cat(
            [
                temporal_context[:, :-1, :],
                temporal_context[:, 1:, :],
            ],
            dim=-1,
        )
        pair_feat = F.dropout(pair_feat, p=0.15, training=self.training)
        hidden = self.motion_factor_backbone(pair_feat)
        forward_logits = self.forward_head(hidden)
        lateral_logits = self.lateral_head(hidden)
        joint_logits = (
            forward_logits.unsqueeze(-1) + lateral_logits.unsqueeze(-2)
        ).reshape(hidden.shape[0], hidden.shape[1], -1)
        forward_reg = self.forward_reg_head(hidden).squeeze(-1)
        motion_embed = self.motion_embed_head(hidden)
        return joint_logits, motion_embed, forward_reg, forward_logits, lateral_logits

    def forward(self, x_enc, rf_maps=None):
        front = self._extract_front_tokens(x_enc, rf_maps=rf_maps)
        meas = self._measurement_posteriors(front)
        rf_latent, adapted_bs, fused_rf = self._rf_latent_tokens(front, meas)
        temporal_context = self._temporal_features(meas, fused_rf)
        state_seq = self._state_from_temporal(temporal_context)
        bridge_token_logits, bridge_motion_embed, bridge_forward_reg, forward_logits, lateral_logits = self._bridge_motion_outputs(temporal_context)
        bridge_token_probs = F.softmax(bridge_token_logits, dim=-1)
        return {
            "front": front,
            "meas": meas,
            "rf_latent": rf_latent,
            "adapted_bs": adapted_bs,
            "fused_rf": fused_rf,
            "state_seq": state_seq,
            "smooth_context": temporal_context,
            "bridge_forward_reg": bridge_forward_reg,
            "forward_logits": forward_logits,
            "lateral_logits": lateral_logits,
            "bridge_motion_embed": bridge_motion_embed,
            "bridge_token_logits": bridge_token_logits,
            "bridge_token_probs": bridge_token_probs,
        }


class MotionGPTForecaster(nn.Module):
    def __init__(self, configs, vocab_size: int, start_token_id: int):
        super().__init__()
        self.pred_len = configs.pred_len
        self.d_llm = configs.llm_dim
        self.vocab_size = vocab_size
        self.start_token_id = start_token_id

        self.token_embed = nn.Embedding(vocab_size + 1, self.d_llm)
        self.time_embed = nn.Embedding(256, self.d_llm)
        self.type_embed = nn.Embedding(2, self.d_llm)  # history / future

        model_source, local_only = resolve_gpt2_source(configs)
        config = GPT2Config.from_pretrained(model_source, local_files_only=local_only)
        config.num_hidden_layers = getattr(configs, "llm_layers", 6)
        self.gpt2 = GPT2Model.from_pretrained(model_source, config=config, local_files_only=local_only)
        for p in self.gpt2.parameters():
            p.requires_grad = False

        if getattr(configs, "islora", 1) == 1 and get_peft_model is not None and LoraConfig is not None:
            r = getattr(configs, "lora_r", 16)
            self.gpt2 = get_peft_model(
                self.gpt2,
                LoraConfig(
                    r=r,
                    lora_alpha=r * 2,
                    target_modules=["c_attn", "c_proj"],
                    lora_dropout=getattr(configs, "dropout", 0.1),
                ),
            )

        self.token_head = nn.Linear(self.d_llm, vocab_size)

    def history_embed_from_ids(self, token_ids):
        return self.token_embed(token_ids)

    def history_embed_from_probs(self, token_probs):
        return token_probs @ self.token_embed.weight[: self.vocab_size]

    def history_embed_from_logits(self, token_logits):
        return self.history_embed_from_probs(F.softmax(token_logits, dim=-1))

    def _add_pos_type(self, embeds, token_type: int, start_index: int = 0):
        B, T, _ = embeds.shape
        t_ids = torch.arange(start_index, start_index + T, device=embeds.device).view(1, T)
        return embeds + self.time_embed(t_ids) + self.type_embed.weight[token_type].view(1, 1, -1)

    def forward_teacher(self, history_embeds, future_token_ids):
        B, Th, _ = history_embeds.shape
        start_ids = torch.full((B, 1), self.start_token_id, device=history_embeds.device, dtype=torch.long)
        future_in = torch.cat([start_ids, future_token_ids[:, :-1]], dim=1)
        future_embeds = self.token_embed(future_in)

        hist = self._add_pos_type(history_embeds, token_type=0, start_index=0)
        fut = self._add_pos_type(future_embeds, token_type=1, start_index=Th)
        seq = torch.cat([hist, fut], dim=1)
        out = self.gpt2(inputs_embeds=seq).last_hidden_state[:, -future_token_ids.shape[1] :, :]
        logits = self.token_head(out)
        return logits

    @torch.no_grad()
    def generate(self, history_embeds, future_len):
        B, Th, _ = history_embeds.shape
        hist = self._add_pos_type(history_embeds, token_type=0, start_index=0)

        generated = []
        prev_ids = torch.full((B, 1), self.start_token_id, device=history_embeds.device, dtype=torch.long)
        future_embeds = []
        for step in range(future_len):
            step_embed = self.token_embed(prev_ids)
            step_embed = self._add_pos_type(step_embed, token_type=1, start_index=Th + step)
            future_embeds.append(step_embed)
            seq = torch.cat([hist] + future_embeds, dim=1)
            out = self.gpt2(inputs_embeds=seq).last_hidden_state[:, -1:, :]
            logits = self.token_head(out[:, 0, :])
            pred = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(pred)
            prev_ids = pred
        return torch.cat(generated, dim=1)


class AnchorEstimator(nn.Module):
    """
    Estimate the current anchor from frozen front-end sensor features and measurements.
    """

    def __init__(
        self,
        num_heads=4,
        hidden_dim=192,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pos_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.15,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 2),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, front, meas, temporal_context):
        sensor_steps = front.get("sensor_steps")
        if sensor_steps is None:
            sensor_steps = torch.cat(
                [
                    front["dist_bs_tokens"].mean(dim=2),
                    front["vel_bs_tokens"].mean(dim=2),
                    front["ang_bs_tokens"].mean(dim=2),
                ],
                dim=-1,
            )
        meas_stats = torch.cat(
            [
                meas["r_exp"] / RANGE_MAX,
                torch.sin(meas["phi_exp"]),
                torch.cos(meas["phi_exp"]),
                meas["vr_exp"] / max(abs(VR_MIN), abs(VR_MAX)),
                meas["conf"],
            ],
            dim=-1,
        )
        frozen_feat = torch.cat([sensor_steps, meas_stats], dim=-1)
        temporal_feat = temporal_context.detach()
        seq_feat = self.input_proj(torch.cat([frozen_feat, temporal_feat], dim=-1))
        batch_size = seq_feat.shape[0]
        query = self.pos_query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query, seq_feat, seq_feat)
        pooled = self.norm(pooled.squeeze(1))
        pos = 150.0 * torch.sigmoid(self.pos_head(pooled))
        hd = self.heading_head(pooled)
        heading = torch.atan2(hd[:, 0], hd[:, 1])
        return pos, heading


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dt = getattr(configs, "dt", 0.05)
        self.tokenizer = MotionTokenizerV7(
            MotionTokenizerConfig(
                forward_bins=getattr(configs, "motion_forward_bins", 21),
                lateral_bins=getattr(configs, "motion_lateral_bins", 21),
                forward_min=getattr(configs, "motion_forward_min", -0.5),
                forward_max=getattr(configs, "motion_forward_max", 1.5),
                lateral_min=getattr(configs, "motion_lateral_min", -1.0),
                lateral_max=getattr(configs, "motion_lateral_max", 1.0),
                dt=self.dt,
            )
        )
        self.bridge = RFGeometryBridge(configs, tokenizer=self.tokenizer)
        self.forecaster = MotionGPTForecaster(
            configs,
            vocab_size=self.tokenizer.vocab_size,
            start_token_id=self.tokenizer.start_token_id,
        )
        self.history_bridge_norm = nn.LayerNorm(getattr(configs, "llm_dim", 768))
        self.anchor_estimator = AnchorEstimator(
            num_heads=4,
            hidden_dim=128,
        )

    def encode_oracle_history(self, history_real):
        state_seq = self.tokenizer.build_state_sequence(history_real, dt=self.dt)
        token_ids = self.tokenizer.motion_token_ids_from_state_sequence(state_seq)
        hist_ids = token_ids[:, : self.seq_len - 1]
        return hist_ids, state_seq

    def encode_oracle_future(self, full_real):
        state_seq = self.tokenizer.build_state_sequence(full_real, dt=self.dt)
        token_ids = self.tokenizer.motion_token_ids_from_state_sequence(state_seq)
        hist_ids = token_ids[:, : self.seq_len - 1]
        future_ids = token_ids[:, self.seq_len - 1 : self.seq_len - 1 + self.pred_len]
        return hist_ids, future_ids, state_seq

    def forward_oracle(self, history_real, future_real):
        full_real = torch.cat([history_real, future_real], dim=1)
        hist_ids, future_ids, hist_state = self.encode_oracle_future(full_real)
        hist_embeds = self.forecaster.history_embed_from_ids(hist_ids)
        future_logits = self.forecaster.forward_teacher(hist_embeds, future_ids)
        future_probs = F.softmax(future_logits, dim=-1)
        start_pos = history_real[:, -1, :]
        start_heading = hist_state[:, self.seq_len - 1, 4]
        pred_real = self.tokenizer.soft_decode(start_pos, start_heading, future_probs)
        return {
            "history_token_ids": hist_ids,
            "future_token_ids": future_ids,
            "future_token_logits": future_logits,
            "pred_real": pred_real,
            "start_pos": start_pos,
            "start_heading": start_heading,
        }

    def forward_rf_pretrain(self, x_enc, rf_maps=None):
        front = self.bridge._extract_front_tokens(x_enc, rf_maps=rf_maps)
        meas = self.bridge._measurement_posteriors(front)
        return {
            "front": front,
            "meas": meas,
        }

    def forward_bridge(self, x_enc, rf_maps=None):
        bridge_out = self.bridge(x_enc, rf_maps=rf_maps)
        token_embed = self.forecaster.history_embed_from_probs(bridge_out["bridge_token_probs"])
        token_embed = self.history_bridge_norm(token_embed + bridge_out["bridge_motion_embed"])
        bridge_out["bridge_token_embed"] = token_embed
        anchor_pos, anchor_heading = self.anchor_estimator(
            bridge_out["front"],
            bridge_out["meas"],
            bridge_out["smooth_context"],
        )
        bridge_out["anchor_pos"] = anchor_pos
        bridge_out["anchor_heading"] = anchor_heading
        return bridge_out

    def forward_anchor_calib(self, x_enc, future_token_ids, rf_maps=None, history_real=None, history_state=None):
        with torch.no_grad():
            bridge_out = self.bridge(x_enc, rf_maps=rf_maps)
            history_embeds = self.forecaster.history_embed_from_probs(bridge_out["bridge_token_probs"])
            history_embeds = self.history_bridge_norm(history_embeds + bridge_out["bridge_motion_embed"])
            future_logits = self.forecaster.forward_teacher(history_embeds, future_token_ids)
            future_probs = F.softmax(future_logits, dim=-1)
            state_last = bridge_out["state_seq"][:, -1, :]
            pred_real_raw_anchor = self.tokenizer.soft_decode(
                state_last[:, :2],
                state_last[:, 4],
                future_probs,
            )
            pred_real_gt_anchor = None
            if history_real is not None:
                if history_state is None:
                    history_state = self.tokenizer.build_state_sequence(history_real, dt=self.dt)
                gt_state_last = history_state[:, -1, :]
                pred_real_gt_anchor = self.tokenizer.soft_decode(
                    history_real[:, -1, :],
                    gt_state_last[:, 4],
                    future_probs,
                )

        anchor_pos, anchor_heading = self.anchor_estimator(
            bridge_out["front"],
            bridge_out["meas"],
            bridge_out["smooth_context"],
        )
        pred_real = self.tokenizer.soft_decode(
            anchor_pos,
            anchor_heading,
            future_probs.detach(),
        )
        bridge_out.update(
            {
                "history_token_embed": history_embeds,
                "future_token_logits": future_logits,
                "future_token_probs": future_probs,
                "pred_real": pred_real,
                "pred_real_raw_anchor": pred_real_raw_anchor,
                "pred_real_gt_anchor": pred_real_gt_anchor,
                "state_last": state_last,
                "anchor_pos": anchor_pos,
                "anchor_heading": anchor_heading,
            }
        )
        return bridge_out

    def forward_forecast(self, x_enc, future_token_ids=None, rf_maps=None, history_real=None, history_state=None):
        if any(p.requires_grad for p in self.bridge.parameters()):
            bridge_out = self.bridge(x_enc, rf_maps=rf_maps)
        else:
            with torch.no_grad():
                bridge_out = self.bridge(x_enc, rf_maps=rf_maps)
        history_embeds = self.forecaster.history_embed_from_probs(bridge_out["bridge_token_probs"])
        history_embeds = self.history_bridge_norm(history_embeds + bridge_out["bridge_motion_embed"])
        if future_token_ids is not None:
            future_logits = self.forecaster.forward_teacher(history_embeds, future_token_ids)
            future_probs = F.softmax(future_logits, dim=-1)
        else:
            future_ids = self.forecaster.generate(history_embeds, self.pred_len)
            future_probs = F.one_hot(future_ids, num_classes=self.tokenizer.vocab_size).float()
            future_logits = None

        state_last = bridge_out["state_seq"][:, -1, :]
        decode_probs = future_probs.detach()
        pred_real_raw_anchor = self.tokenizer.soft_decode(
            state_last[:, :2],
            state_last[:, 4],
            decode_probs,
        )
        anchor_pos, anchor_heading = self.anchor_estimator(
            bridge_out["front"],
            bridge_out["meas"],
            bridge_out["smooth_context"],
        )
        pred_real = self.tokenizer.soft_decode(
            anchor_pos,
            anchor_heading,
            decode_probs,
        )
        pred_real_gt_anchor = None
        if history_real is not None:
            if history_state is None:
                history_state = self.tokenizer.build_state_sequence(history_real, dt=self.dt)
            gt_state_last = history_state[:, -1, :]
            pred_real_gt_anchor = self.tokenizer.soft_decode(
                history_real[:, -1, :],
                gt_state_last[:, 4],
                decode_probs,
            )
        bridge_out.update(
            {
                "history_token_embed": history_embeds,
                "future_token_logits": future_logits,
                "future_token_probs": future_probs,
                "pred_real": pred_real,
                "pred_real_raw_anchor": pred_real_raw_anchor,
                "pred_real_gt_anchor": pred_real_gt_anchor,
                "state_last": state_last,
                "anchor_pos": anchor_pos,
                "anchor_heading": anchor_heading,
            }
        )
        return bridge_out
