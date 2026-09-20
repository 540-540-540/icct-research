from __future__ import annotations

import torch
from torch import nn

from prediction.q0.contracts import CANONICAL_DT

from .classical import HistoricalRajJohnsonTokenCore, MatchedRajJohnsonTokenCore
from .common import parameter_summary
from .quantum import PennyLaneRajMultiJCore
from .residual import RajResidualLLM


class RajResidualModel(nn.Module):
    """CV + self-motion GPT-2 residual + graph-exclusive interaction residual."""

    def __init__(self, core: nn.Module, llm: RajResidualLLM, name: str):
        super().__init__()
        self.core = core
        self.llm = llm
        self.name = str(name)
        self.training_phase = "joint"

    def set_training_phase(self, phase: str):
        if phase not in {"self", "interaction", "joint"}:
            raise ValueError(phase)
        self.training_phase = phase
        self.core.requires_grad_(phase != "self")
        self.llm.set_training_phase(phase)

    def forward(self, history, mask, timestamps=None):
        history = torch.where(mask[:, None, :, None], history, 0.0)
        core = self.core(history, mask, timestamps)
        out = self.llm(
            history,
            mask,
            core["readout"],
            core["interaction_mask"],
        )
        if timestamps is not None:
            dt = ((timestamps[:, -1] - timestamps[:, 0]) / 19).to(history.dtype)
            step = torch.arange(
                1, 21, device=history.device, dtype=history.dtype
            )[None, :, None, None]
            correction = (
                step
                * (dt - CANONICAL_DT)[:, None, None, None]
                * history[:, -1:, :, 2:]
            )
            out["prediction"] = (
                out["prediction"] + correction * mask[:, None, :, None]
            )
        out["interaction_readout"] = core["readout"]
        out["interaction_mask"] = core["interaction_mask"]
        if "fused_readout" in core:
            out["fused_readout"] = core["fused_readout"]
        return out

    def parameter_summary(self):
        total = parameter_summary(self)
        total["core_trainable"] = sum(
            p.numel() for p in self.core.parameters() if p.requires_grad
        )
        total["llm_trainable"] = sum(
            p.numel() for p in self.llm.parameters() if p.requires_grad
        )
        return total


def build_model(
    kind: str = "raj_pennylane",
    seed: int = 2026,
    rounds: int = 3,
    correction_cap_m: float = 16.0,
    matched_hidden: int = 32,
    device_name: str = "lightning.gpu",
    diff_method: str = "adjoint",
    phase: str = "joint",
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100003)
        if kind == "raj_pennylane":
            core = PennyLaneRajMultiJCore(
                rounds=rounds,
                device_name=device_name,
                diff_method=diff_method,
            )
        elif kind == "raj_matched":
            core = MatchedRajJohnsonTokenCore(
                rounds=rounds,
                hidden=matched_hidden,
            )
        elif kind == "raj_strong":
            core = HistoricalRajJohnsonTokenCore(
                rounds=rounds,
                hidden=64,
            )
        else:
            raise ValueError(kind)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 300003)
        llm = RajResidualLLM(
            readout_dim=core.readout_dim,
            interaction_tokens=core.interaction_tokens,
            correction_cap_m=correction_cap_m,
        )

    model = RajResidualModel(core, llm, kind)
    model.set_training_phase(phase)
    return model
