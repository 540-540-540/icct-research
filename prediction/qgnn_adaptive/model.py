"""Unmodified common tokenizer/GPT-2; only the interaction core changes."""
from pathlib import Path
import json
import torch
from prediction.qgnn_final.model import FinalModel,FinalMotionGPT2
from .quantum import SceneAdaptiveQuantumCore
from .classical import SceneAdaptiveClassicalCore

def build_model(kind,seed=2026,depth=3,token_path=None,channels=4,enhanced=True,snr_db=0.,quantum_version=3,correction_cap_m=16.,adaptive_mode="feedback"):
    if adaptive_mode not in ("feedback","history"):raise ValueError(adaptive_mode)
    root=Path(__file__).resolve().parents[2]
    payload=json.loads(Path(token_path or root/"configs/qgnn_final_tokens.json").read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        cls={"quantum":SceneAdaptiveQuantumCore,"classical":SceneAdaptiveClassicalCore}[kind]
        core=cls(depth,channels,seed+800003,adaptive_mode=="feedback")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+300003);llm=FinalMotionGPT2(payload,correction_cap_m)
    return FinalModel(core,llm)
