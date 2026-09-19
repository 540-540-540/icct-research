from __future__ import annotations
import json
from pathlib import Path
import torch
from prediction.qgnn_final.model import FinalMotionGPT2,FinalModel
from .raj_subset import RajSubsetQGNNCore,RajJohnsonGINCore,RajMultiJQGNNCore,RajMultiJJohnsonCore
from .raj_paper import RajPaperQGNNCore,RajPaperJohnsonGINCore,RajPaperGINCore,RajPaperPPGNCore,RajWeightedMultiJQGNNCore

def build_paper_model(kind,seed=2026,rounds=3,j=3,correction_cap_m=16.):
    root=Path(__file__).resolve().parents[2]
    payload=json.loads((root/'configs/qgnn_final_tokens.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        if kind=='raj_quantum':
            core=RajSubsetQGNNCore(j=j,rounds=rounds)
        elif kind=='raj_johnson':
            core=RajJohnsonGINCore(j=j,rounds=rounds,hidden=64)
        elif kind=='raj_multij_quantum':
            core=RajMultiJQGNNCore(rounds=rounds)
        elif kind=='raj_multij_johnson':
            core=RajMultiJJohnsonCore(rounds=rounds,hidden=64)
        elif kind=='raj_paper_quantum':
            core=RajPaperQGNNCore(j=j,depth=rounds)
        elif kind=='raj_paper_johnson':
            core=RajPaperJohnsonGINCore(j=j,depth=rounds,hidden=128)
        elif kind=='raj_paper_gin':
            core=RajPaperGINCore(depth=rounds,hidden=128)
        elif kind=='raj_paper_ppgn':
            core=RajPaperPPGNCore(depth=rounds,width=64)
        elif kind=='raj_weighted_multij_quantum':
            core=RajWeightedMultiJQGNNCore(rounds=rounds)
        else: raise ValueError(kind)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+300003)
        llm=FinalMotionGPT2(payload,correction_cap_m)
    return FinalModel(core,llm)
