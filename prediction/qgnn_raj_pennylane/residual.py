from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from prediction.q0.temporal import FusedQKVLoRA, constant_velocity_baseline
from prediction.qgnn_final.model import FinalMotionGPT2


class RajResidualLLM(nn.Module):
    """Self-motion GPT-2 plus a graph-exclusive interaction-residual interface."""

    def __init__(
        self,
        readout_dim: int = 64,
        correction_cap_m: float = 16.0,
        query_dim: int = 32,
    ):
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        payload = json.loads((root / "configs/qgnn_final_tokens.json").read_text())
        self.base = FinalMotionGPT2(payload, correction_cap_m)
        self.base.graph_projection.requires_grad_(False)
        self.base.coordinate_head.requires_grad_(False)
        self.correction_cap_m = float(correction_cap_m)
        self.readout_dim = int(readout_dim)
        self.query_dim = int(query_dim)
        self.training_phase = "joint"

        d = self.base.d_llm
        self.self_head = nn.Sequential(
            nn.LayerNorm(d + 2),
            nn.Linear(d + 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        self.interaction_projection = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, query_dim),
            nn.GELU(),
        )
        self.future_projection = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, query_dim, bias=False),
        )
        self.future_step_embedding = nn.Parameter(torch.randn(20, query_dim) * 0.02)
        self.interaction_residual_head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 2),
        )
        self.interaction_gate = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, 1),
        )

        nn.init.zeros_(self.self_head[-1].weight)
        nn.init.zeros_(self.self_head[-1].bias)
        nn.init.normal_(self.interaction_residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.interaction_residual_head[-1].bias)
        nn.init.zeros_(self.interaction_gate[-1].weight)
        nn.init.constant_(self.interaction_gate[-1].bias, -2.0)

        self._base_default_trainable = {
            name: p.requires_grad for name, p in self.base.named_parameters()
        }

    @property
    def gpt2(self):
        return self.base.gpt2

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def future_token_ids(self, history, future):
        return self.base.future_token_ids(history, future)

    def _interaction_modules(self):
        return (
            self.interaction_projection,
            self.future_projection,
            self.interaction_residual_head,
            self.interaction_gate,
        )

    def _set_interaction_trainable(self, enabled: bool):
        for module in self._interaction_modules():
            module.requires_grad_(enabled)
        self.future_step_embedding.requires_grad_(enabled)

    def _set_base_default(self):
        for name, p in self.base.named_parameters():
            p.requires_grad_(self._base_default_trainable[name])

    def _set_lora_only(self):
        self.base.requires_grad_(False)
        for module in self.base.modules():
            if isinstance(module, FusedQKVLoRA):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)

    def set_training_phase(self, phase: str):
        if phase not in {"self", "interaction", "joint"}:
            raise ValueError(phase)
        self.training_phase = phase
        if phase == "self":
            self._set_base_default()
            self.self_head.requires_grad_(True)
            self._set_interaction_trainable(False)
        elif phase == "interaction":
            self.base.requires_grad_(False)
            self.self_head.requires_grad_(False)
            self._set_interaction_trainable(True)
        else:
            self._set_lora_only()
            self.self_head.requires_grad_(False)
            self._set_interaction_trainable(True)

    def _self_motion(self, history):
        motion, _ = self.base._history_motion_embeddings(history)
        history_tokens = motion + self.base.history_adapter(motion)
        sequence = torch.cat(
            (
                history_tokens,
                self.base.future_queries[None].expand(len(history), -1, -1),
            ),
            1,
        )
        hidden = self.base.gpt2(
            inputs_embeds=sequence,
            use_cache=False,
        ).last_hidden_state[:, -20:]
        token_logits = self.base.token_head(hidden)
        probability = torch.softmax(token_logits, -1)
        ef, el = self.base.tokenizer.expected_motion(probability)
        expected_motion = torch.stack((ef, el), -1)
        raw = self.self_head(torch.cat((hidden, expected_motion), -1))
        correction = torch.tanh(raw) * self.correction_cap_m
        return hidden, token_logits, expected_motion, correction

    def _interaction(self, hidden, readout, interaction_mask):
        graph = self.interaction_projection(readout)[:, None]
        future = self.future_projection(hidden)
        fused = torch.nn.functional.gelu(
            future + graph + self.future_step_embedding[None]
        )
        raw = self.interaction_residual_head(fused)
        residual = torch.tanh(raw) * self.correction_cap_m
        gate = torch.sigmoid(self.interaction_gate(fused))
        gate = gate * interaction_mask[:, None, None].to(gate.dtype)
        if self.training_phase == "self":
            gate = gate * 0.0
        return gate, residual

    def forward(self, history, mask, readout, interaction_mask):
        b, t, n, _ = history.shape
        flat_history = history.permute(0, 2, 1, 3).reshape(b * n, t, 4)
        if readout.shape != (b, n, self.readout_dim):
            raise ValueError("readout must be [B,N,readout_dim]")
        if interaction_mask.shape != (b, n) or interaction_mask.dtype != torch.bool:
            raise ValueError("interaction_mask must be boolean [B,N]")
        flat_readout = readout.reshape(b * n, self.readout_dim)
        flat_imask = interaction_mask.reshape(b * n)
        active = mask.reshape(-1).nonzero(as_tuple=True)[0]

        prediction = history.new_zeros((b * n, 20, 2))
        logits_full = history.new_zeros((b * n, 20, self.base.vocab_size))
        gate_full = history.new_zeros((b * n, 20, 1))
        residual_full = history.new_zeros((b * n, 20, 2))
        self_full = history.new_zeros((b * n, 20, 2))

        if active.numel() == 0:
            return self._reshape_outputs(
                prediction, logits_full, gate_full, residual_full, self_full, b, n, mask
            )

        h = flat_history[active]
        z = flat_readout[active]
        vm = flat_imask[active]
        hidden, token_logits, _, self_corr = self._self_motion(h)
        gate, interaction_corr = self._interaction(hidden, z, vm)
        cv = constant_velocity_baseline(h.reshape(len(active), 20, 1, 4))[:, :, 0]
        pred = cv + self_corr + gate * interaction_corr

        prediction = prediction.index_copy(0, active, pred)
        logits_full = logits_full.index_copy(0, active, token_logits)
        gate_full = gate_full.index_copy(0, active, gate)
        residual_full = residual_full.index_copy(0, active, interaction_corr)
        self_full = self_full.index_copy(0, active, self_corr)
        return self._reshape_outputs(
            prediction, logits_full, gate_full, residual_full, self_full, b, n, mask
        )

    def interface_parameter_count(self):
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith(("base.", "self_head."))
        )

    @staticmethod
    def _reshape_outputs(
        prediction, logits, gate, residual, self_corr, b, n, mask
    ):
        return {
            "prediction": prediction.reshape(b, n, 20, 2).permute(0, 2, 1, 3),
            "token_logits": logits.reshape(b, n, 20, -1).permute(0, 2, 1, 3),
            "interaction_gate": gate.reshape(b, n, 20, 1).permute(0, 2, 1, 3),
            "interaction_residual": residual.reshape(b, n, 20, 2).permute(0, 2, 1, 3),
            "self_residual": self_corr.reshape(b, n, 20, 2).permute(0, 2, 1, 3),
            "origin_eligible": mask,
        }
