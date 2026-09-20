#!/usr/bin/env python3
"""Short Self coordinate-interface checks; synthetic inputs, no training/test.

Optional legacy-reference is a pre-change output snapshot on train histories.
Both legacy checkpoints are also checked for allowed missing keys explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.q0.temporal import constant_velocity_baseline
from prediction.qgnn_raj_pennylane import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--legacy-reference", type=Path)
    parser.add_argument("--legacy-run-dir", type=Path,
                        default=ROOT / "results/qgnn/raj_pennylane_p2/self_llm_0db_seed2026")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    device = torch.device(args.device)
    checks = {}

    def record(name, passed, **evidence):
        checks[name] = {"passed": bool(passed), **evidence}
        print(json.dumps({"check": name, **checks[name]}), flush=True)
        assert passed, name

    def close(name, actual, expected, tolerance=2e-4):
        error = float((actual - expected).abs().max())
        record(name, torch.isfinite(actual).all() and error <= tolerance, max_abs_error=error)

    history = torch.zeros(2, 20, 3, 4, device=device)
    time = torch.arange(20, device=device) * 0.1001001001
    history[:, :, 0, 0] = 3 * time + 0.4 * time.square()
    history[:, :, 0, 1] = 0.25 * time.square()
    history[:, :, 0, 2] = 3 + 0.8 * time
    history[:, :, 0, 3] = 0.5 * time
    history[0, :, 1, 0] = 0.005 * time
    history[0, :, 1, 2] = 0.005
    history[1, :, 1, :2] = 5.0  # Truly stationary: no direction in input.
    mask = torch.tensor([[True, True, False], [True, True, False]], device=device)
    model = build_model(phase="self", self_frame="ego_v1").to(device).eval()

    def forbid_core(*unused):
        raise AssertionError("Raj core executed during Self phase")

    hook = model.core.register_forward_pre_hook(forbid_core)
    with torch.no_grad():
        initial = model(history, mask)
        cv = constant_velocity_baseline(history) * mask[:, None, :, None]
        close("zero_head_is_exact_cv", initial["prediction"], cv, 0.0)
        model.llm.self_head[-1].weight.normal_(std=0.003)
        model.llm.self_head[-1].bias.copy_(torch.tensor([0.03, -0.02], device=device))
        original = model(history, mask)
        record("nonzero_head_test", original["self_residual"][:, :, 0].abs().max() > 0.01)
        for angle in (math.pi, 0.73):
            c, s = math.cos(angle), math.sin(angle)
            rotation = history.new_tensor([[c, -s], [s, c]])
            rotated = history.clone()
            rotated[..., :2] = history[..., :2] @ rotation.T
            rotated[..., 2:] = history[..., 2:] @ rotation.T
            transformed = model(rotated, mask)
            close(f"rotation_{angle:.3f}", transformed["prediction"],
                  original["prediction"] @ rotation.T)
        translated = history.clone()
        offset = history.new_tensor([11.0, -7.0])
        translated[..., :2] += offset
        close("translation", model(translated, mask)["prediction"],
              original["prediction"] + offset * mask[:, None, :, None])
        changed = history.clone()
        changed[:, :, 1] = torch.randn_like(changed[:, :, 1]) * 100
        close("neighbor_isolation", model(changed, mask)["prediction"][:, :, 0],
              original["prediction"][:, :, 0], 0.0)
        padded = torch.cat((history, torch.full_like(history[:, :, :1], float("nan"))), 2)
        padded_mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
        padded_out = model(padded, padded_mask)
        close("padding_invariance", padded_out["prediction"][:, :, :3],
              original["prediction"], 0.0)
        close("padding_zero", padded_out["prediction"][:, :, 3],
              torch.zeros_like(padded_out["prediction"][:, :, 3]), 0.0)
        close("fully_static_no_direction", original["self_residual"][1, :, 1],
              torch.zeros_like(original["self_residual"][1, :, 1]), 0.0)
        record("low_speed_not_suppressed", original["self_residual"][0, :, 1].abs().max() > 0.01)
        stopped = history[0, :, :1].transpose(0, 1).clone()
        stopped[:, -4:, 2:] = 0
        _, usable = model.llm._ego_frame(stopped)
        record("stopped_uses_history_direction", usable.all())
        for fallback_name, positions, speed in (("displacement_fallback", True, False),
                                                ("tiny_velocity_fallback", False, True)):
            fallback = stopped.clone()
            fallback[..., 2:] = 0.0005 if speed else 0
            if not positions:
                fallback[..., :2] = 0
            _, usable = model.llm._ego_frame(fallback)
            record(fallback_name, usable.all())
        empty = model(history, torch.zeros_like(mask))
        close("all_padding_zero", empty["prediction"], torch.zeros_like(cv), 0.0)
        close("self_has_no_interaction", original["interaction_gate"],
              torch.zeros_like(original["interaction_gate"]), 0.0)
        # CE target semantics must remain the original per-transition local motion.
        future = history.clone()
        close("future_token_targets_unchanged", model.llm.future_token_ids(history, future),
              model.llm.base.future_token_ids(history, future), 0.0)

    model.zero_grad(set_to_none=True)
    loss = model(history, mask)["prediction"].square().mean()
    loss.backward()
    own_grad = model.llm.own_history_adapter.weight.grad
    lora_grads = [p.grad for name, p in model.named_parameters()
                  if "lora_" in name and p.grad is not None]
    record("self_adapter_gradient", own_grad is not None and torch.isfinite(own_grad).all()
           and own_grad.abs().sum() > 0, sum_abs=float(own_grad.abs().sum()))
    record("self_lora_gradient", bool(lora_grads) and all(torch.isfinite(g).all() for g in lora_grads)
           and sum(float(g.abs().sum()) for g in lora_grads) > 0)
    hook.remove()
    record("self_core_not_executed", True)
    for phase in ("interaction", "joint", "self"):
        model.set_training_phase(phase)
        record(f"adapter_phase_{phase}",
               model.llm.own_history_adapter.weight.requires_grad == (phase == "self"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    legacy = build_model(phase="self", self_frame="legacy").to(device).eval()
    record("legacy_has_no_new_keys", not any("own_history_adapter" in k for k in legacy.state_dict()))
    reference = (torch.load(args.legacy_reference, map_location="cpu", weights_only=False)
                 if args.legacy_reference else None)
    for filename in ("best.pt", "last.pt"):
        saved = torch.load(args.legacy_run_dir / filename, map_location="cpu", weights_only=False)
        missing, unexpected = legacy.load_state_dict(saved["model_state"], strict=False)
        disallowed = [name for name in missing if not name.startswith("core.")
                      and not (name.startswith("llm.base.gpt2.") and "lora_" not in name)]
        record(f"legacy_load_{filename}", not disallowed and not unexpected,
               missing_disallowed=disallowed, unexpected=unexpected)
        if reference:
            with torch.no_grad():
                replay = legacy(reference["history"].to(device), reference["mask"].to(device))
            for key, expected in reference[filename].items():
                close(f"legacy_replay_{filename}_{key}", replay[key].cpu(), expected, 0.0)

    report = {"schema": "raj_self_frame_ego_v1_checks", "checks": checks,
              "passed": all(row["passed"] for row in checks.values()),
              "optimizer_steps": 0, "test_opened": False,
              "legacy_reference_compared": reference is not None}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
