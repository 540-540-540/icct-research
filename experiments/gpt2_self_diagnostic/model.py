"""Isolated Self diagnostics; production predictors and checkpoints stay intact.

The random arm changes only GPT blocks, position embeddings and final norm.
Motion embeddings still originate from the same pretrained vocabulary table;
this is a backbone-pretraining test, not removal of every language-derived weight.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import GPT2Model

from prediction.q0.contracts import CANONICAL_DT
from prediction.q0.temporal import FusedQKVLoRA, constant_velocity_baseline
from prediction.qgnn_raj_pennylane.model import RajResidualModel
from prediction.qgnn_raj_pennylane.residual import RajResidualLLM
from prediction.qgnn_raj_pennylane.self_baselines import SelfTemporalBaseline


class DiagnosticTCN(SelfTemporalBaseline):
    """Identical TCN weights; optionally apply the shared ego interface bundle."""

    def __init__(self, frame):
        super().__init__("tcn")
        self.frame = frame
        self.diagnostic_kind = "tcn"

    def forward(self, history, mask, timestamps=None):
        if self.frame == "global":
            return super().forward(history, mask, timestamps)
        if history.ndim != 4 or history.shape[1] != 20 or history.shape[-1] != 4:
            raise ValueError("Expected history [B,20,N,4]")
        if mask.shape != (history.shape[0], history.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean mask [B,N]")
        history = torch.where(mask[:, None, :, None], history, 0.0)
        b, t, n, _ = history.shape
        flat = history.permute(0, 2, 1, 3).reshape(b * n, t, 4)
        frame, has_direction = RajResidualLLM._ego_frame(flat)
        position = (flat[..., :2] - flat[:, -1:, :2]) @ frame
        velocity = flat[..., 2:] @ frame
        local = torch.cat((position, velocity), -1)
        local = local.reshape(b, n, t, 4).permute(0, 2, 1, 3)
        local_prediction = super().forward(local, mask)["prediction"]
        local_residual = local_prediction - constant_velocity_baseline(local)
        local_residual = local_residual.permute(0, 2, 1, 3).reshape(b * n, 20, 2)
        residual = (local_residual @ frame.transpose(-1, -2)) * has_direction[:, None, None]
        residual = residual.reshape(b, n, 20, 2).permute(0, 2, 1, 3)
        prediction = constant_velocity_baseline(history) + residual
        if timestamps is not None:
            dt = ((timestamps[:, -1] - timestamps[:, 0]) / 19).to(history.dtype)
            step = torch.arange(1, 21, device=history.device, dtype=history.dtype)
            prediction = prediction + (
                step[None, :, None, None]
                * (dt - CANONICAL_DT)[:, None, None, None]
                * history[:, -1:, :, 2:]
            )
        prediction = torch.where(mask[:, None, :, None], prediction, 0.0)
        return {"prediction": prediction, "origin_eligible": mask}

    def parameter_summary(self):
        return parameter_summary(self)


class DiagnosticResidualLLM(RajResidualLLM):
    def __init__(self, frame, seed):
        super().__init__(self_frame="ego_v1")
        self.frame = frame
        # All arms carry the same extra state key, without perturbing any common
        # initialization or the caller's CPU/CUDA dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 500003)
            self.heading_adapter = nn.Linear(2, self.base.d_llm, bias=False)
            nn.init.zeros_(self.heading_adapter.weight)
        self.heading_adapter.requires_grad_(frame == "ego_heading")

    def _self_motion(self, history):
        if self.frame != "ego_heading":
            return super()._self_motion(history)
        # Same ego_v1 token/continuous-state/head path, with one explicit heading
        # injection. No hook, neighbor data, or change to future-token targets.
        motion, _ = self.base._history_motion_embeddings(history)
        history_tokens = motion + self.base.history_adapter(motion)
        frame, has_direction = self._ego_frame(history)
        position = (history[..., :2] - history[:, -1:, :2]) @ frame
        velocity = history[..., 2:4] @ frame
        own_state = torch.cat((position / 10.0, velocity / 10.0), -1)
        history_tokens = history_tokens + self.own_history_adapter(own_state[:, 1:])
        heading = frame[:, :, 0] * has_direction[:, None]
        history_tokens = history_tokens + self.heading_adapter(heading)[:, None]
        sequence = torch.cat((history_tokens, self.base.future_queries[None].expand(len(history), -1, -1)), 1)
        hidden = self.base.gpt2(inputs_embeds=sequence, use_cache=False).last_hidden_state[:, -20:]
        token_logits = self.base.token_head(hidden)
        ef, el = self.base.tokenizer.expected_motion(torch.softmax(token_logits, -1))
        expected_motion = torch.stack((ef, el), -1)
        raw = self.self_head(torch.cat((hidden, expected_motion), -1))
        correction = torch.tanh(raw) * self.correction_cap_m
        correction = (correction @ frame.transpose(-1, -2)) * has_direction[:, None, None]
        return hidden, token_logits, expected_motion, correction


class _SelfOnlyCore(nn.Module):
    readout_dim = 64

    def forward(self, *args, **kwargs):
        raise RuntimeError("The Self diagnostic must never execute an interaction core")


class DiagnosticLLM(RajResidualModel):
    def __init__(self, llm, frame, initialization, adaptation):
        super().__init__(_SelfOnlyCore(), llm, "gpt2_self_diagnostic")
        self.diagnostic_kind = "llm"
        self.frame = frame
        self.initialization = initialization
        self.adaptation = adaptation
        self.set_training_phase("self")

    def set_training_phase(self, phase):
        if phase != "self":
            raise ValueError("Only the Self phase is supported by this diagnostic")
        super().set_training_phase(phase)
        gpt = self.llm.gpt2
        gpt.requires_grad_(False)
        if self.adaptation == "full":
            gpt.h.requires_grad_(True)
            gpt.wpe.requires_grad_(True)
            gpt.ln_f.requires_grad_(True)
        for module in gpt.modules():
            if isinstance(module, FusedQKVLoRA):
                module.lora_A.requires_grad_(self.adaptation == "lora")
                module.lora_B.requires_grad_(self.adaptation == "lora")
                if self.adaptation == "full":
                    with torch.no_grad():
                        module.lora_B.zero_()
        self.llm.heading_adapter.requires_grad_(self.frame == "ego_heading")

    def parameter_summary(self):
        return parameter_summary(self)


def _randomize_backbone(llm, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 400003)
        random_gpt = GPT2Model(copy.deepcopy(llm.gpt2.config))
    random_state = random_gpt.state_dict()
    with torch.no_grad():
        for name, parameter in llm.gpt2.named_parameters():
            if name.startswith("wte.") or "lora_" in name:
                continue
            source_name = name.replace(".c_attn.base.", ".c_attn.")
            parameter.copy_(random_state[source_name])


def build_diagnostic_model(
    kind="llm", seed=2026, frame="ego", initialization="pretrained", adaptation="lora"
):
    if kind not in {"tcn", "llm"}:
        raise ValueError(f"Unknown diagnostic model: {kind}")
    if frame not in {"global", "ego", "ego_heading"}:
        raise ValueError(f"Unknown frame: {frame}")
    if kind == "llm" and frame == "global":
        raise ValueError("LLM supports ego/ego_heading only; legacy is not a valid global-state control")
    if initialization not in {"pretrained", "random"} or adaptation not in {"lora", "full"}:
        raise ValueError("Expected pretrained/random initialization and lora/full adaptation")
    if kind == "tcn" and (frame == "ego_heading" or initialization != "pretrained" or adaptation != "lora"):
        raise ValueError("TCN supports global/ego only; initialization/adaptation are GPT-only factors")
    with torch.random.fork_rng(devices=[]):
        # CPU-only seed preserves CUDA dropout RNG; same weight seed as the
        # production builders. Extra/random initializations have separate forks.
        torch.random.default_generator.manual_seed(seed + 300003)
        if kind == "tcn":
            return DiagnosticTCN(frame)
        llm = DiagnosticResidualLLM(frame, seed)
    if initialization == "random":
        _randomize_backbone(llm, seed)
    return DiagnosticLLM(llm, frame, initialization, adaptation)


def parameter_groups(model, backbone_lr, peripheral_lr):
    groups = {"backbone": [], "peripheral": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "backbone" if name.startswith("llm.base.gpt2.") else "peripheral"
            groups[group].append(parameter)
    return [{"name": name, "params": parameters,
             "lr": backbone_lr if name == "backbone" else peripheral_lr}
            for name, parameters in groups.items() if parameters]


def parameter_summary(model):
    named = list(model.named_parameters())
    backbone = [(name, p) for name, p in named if name.startswith("llm.base.gpt2.")]
    backbone_used = [(name, p) for name, p in backbone
                     if not name.startswith("llm.base.gpt2.wte.")
                     and not (getattr(model, "adaptation", None) == "full" and "lora_" in name)]
    if model.diagnostic_kind == "tcn":
        used = named
    else:
        used = backbone_used + [(name, p) for name, p in named
            if ((name.startswith("llm.base.") and not name.startswith((
                    "llm.base.gpt2.", "llm.base.graph_projection.", "llm.base.coordinate_head.")))
                or name.startswith(("llm.self_head.", "llm.own_history_adapter."))
                or (model.frame == "ego_heading" and name.startswith("llm.heading_adapter.")))]
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    backbone_trainable = sum(p.numel() for _, p in backbone if p.requires_grad)
    return {
        "total": sum(p.numel() for _, p in named),
        "trainable": trainable,
        "effective_total": sum(p.numel() for _, p in used),
        "backbone_used": sum(p.numel() for _, p in backbone_used),
        "backbone_trainable": backbone_trainable,
        "peripheral_trainable": trainable - backbone_trainable,
        "heading_total": sum(p.numel() for name, p in named if name.startswith("llm.heading_adapter.")),
        "heading_trainable": sum(p.numel() for name, p in named
                                 if name.startswith("llm.heading_adapter.") and p.requires_grad),
    }
