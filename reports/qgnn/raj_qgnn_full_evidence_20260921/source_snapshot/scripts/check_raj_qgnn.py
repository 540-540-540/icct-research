"""Fast import, shape, gradient, and parameter checks for the retained RAJ models."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.raj_qgnn.model import build_model


def synthetic_batch(batch=2):
    torch.manual_seed(7)
    history = torch.randn(batch, 20, 8, 4)
    history[..., :2] = history[..., :2].cumsum(1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    history = torch.where(mask[:, None, :, None], history, 0.0)
    timestamps = torch.arange(20, dtype=torch.float32)[None].expand(batch, -1) * 0.1
    return history, mask, timestamps


def main():
    history, mask, timestamps = synthetic_batch()
    summaries = {}
    shared = None
    for kind in ("quantum", "matched_classical", "gin", "ppgn"):
        model = build_model(kind, seed=2026, rounds=3)
        output = model(history, mask, timestamps)
        assert output["prediction"].shape == (2, 20, 8, 2)
        assert output["graph_features"].shape == (2, 8, 64)
        assert torch.isfinite(output["prediction"]).all()
        (output["prediction"].square().mean() + output["token_logits"].square().mean()).backward()
        graph_grad = sum(float(p.grad.abs().sum()) for p in model.graph.parameters() if p.grad is not None)
        assert graph_grad > 0.0
        current = {n: p.detach().clone() for n, p in model.llm.named_parameters()}
        if shared is None:
            shared = current
        else:
            assert shared.keys() == current.keys()
            assert all(torch.equal(shared[n], current[n]) for n in shared)
        summaries[kind] = model.parameter_summary()
    print({"status": "PASS", "models": summaries})


if __name__ == "__main__":
    main()
