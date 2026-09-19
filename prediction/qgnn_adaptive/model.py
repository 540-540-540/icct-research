"""Unmodified common tokenizer/GPT-2; only the interaction core changes."""
from pathlib import Path
import json
import torch
from prediction.qgnn_final.model import FinalModel,FinalMotionGPT2
from .quantum import SceneAdaptiveQuantumCore
from .classical import SceneAdaptiveClassicalCore

def build_model(kind,seed=2026,depth=3,token_path=None,channels=4,enhanced=True,snr_db=0.,quantum_version=3,correction_cap_m=16.,adaptive_mode="phase_feedback",controller_init="specialized"):
    if adaptive_mode not in ("feedback","history","phase_feedback","basis_feedback"):raise ValueError(adaptive_mode)
    root=Path(__file__).resolve().parents[2]
    payload=json.loads(Path(token_path or root/"configs/qgnn_final_tokens.json").read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        cls={"quantum":SceneAdaptiveQuantumCore,"classical":SceneAdaptiveClassicalCore}[kind]
        core=cls(depth,channels,seed+800003,adaptive_mode!="history",basis_feedback=adaptive_mode=="basis_feedback",**({"phase_mode":adaptive_mode=="phase_feedback"} if kind=="quantum" else {}))
    if controller_init not in ("specialized","neutral"):raise ValueError(controller_init)
    if controller_init=="neutral":
        with torch.no_grad():
            for layer in (core.controller.scene_head,core.controller.feedback_global):
                for parameter in layer.parameters():parameter.zero_()
            core.controller.relation_gain.zero_();core.controller.feedback_local.zero_()
            if hasattr(core,"basis_weights"):core.basis_weights.zero_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+300003);llm=FinalMotionGPT2(payload,correction_cap_m)
    return FinalModel(core,llm)
