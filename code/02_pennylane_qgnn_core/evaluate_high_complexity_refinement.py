"""Locked test evaluation of the matched high-complexity refinement checkpoints."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
SOURCE = BASE / "high_complexity_llm_refine_seed2026_v1"
OUT = BASE / "high_complexity_llm_test_seed2026_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

import experiment_utils as r
import run as retained
from analyze_interaction_strata import scene_features
from model import CoreGraphLLM, restore_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph

SEED = 2026
SNRS = ("5", "10", "15", "20")
REPS = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(arm):
    model = CoreGraphLLM(build_physics_aligned_dual_graph(arm))
    checkpoint = torch.load(SOURCE / f"{arm}_high_llm_selected.pt", map_location="cpu", weights_only=True)
    restore_compact(model, checkpoint["state"])
    return model, checkpoint


def metric(arrays, indices):
    result = []
    for snr in SNRS:
        x = arrays[snr][indices]
        denominator = x[:, 2].sum()
        result.append((x[:, 0].sum() / (denominator * 20), x[:, 1].sum() / denominator))
    return np.mean(result, axis=0)


def paired_bootstrap(reference, candidate, indices, rng):
    reference_point = metric(reference, indices)
    candidate_point = metric(candidate, indices)
    draws = np.empty((REPS, 2), dtype=np.float64)
    for repetition in range(REPS):
        sample = indices[rng.integers(0, len(indices), len(indices))]
        ref = metric(reference, sample)
        cand = metric(candidate, sample)
        draws[repetition] = 100 * (ref - cand) / ref
    point = 100 * (reference_point - candidate_point) / reference_point
    return {
        key: {
            "point_improvement_percent": float(point[column]),
            "bootstrap_median_percent": float(np.median(draws[:, column])),
            "ci95_percent": [float(value) for value in np.quantile(draws[:, column], [0.025, 0.975])],
            "bootstrap_probability_improvement": float((draws[:, column] > 0).mean()),
        }
        for column, key in enumerate(("ade", "fde"))
    }, draws


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn" and torch.cuda.is_available()
    assert (SOURCE / "completed.json").exists()
    OUT.mkdir(exist_ok=False)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        states = data["test_states"].copy()
        masks = data["test_mask"].copy()
    bank = {
        "history": torch.from_numpy(states[:, :20]).float().cuda(),
        "future": torch.from_numpy(states[:, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks).bool().cuda(),
    }
    arrays = {}
    summaries = {}
    for arm in ("physics_classical_dual", "physics_quantum_dual"):
        r.set_seed(SEED)
        model, checkpoint = load_model(arm)
        model = model.cuda()
        started = time.time()
        metrics, collected = retained.evaluate(model, bank, collect=True)
        elapsed = time.time() - started
        arrays[arm] = collected
        summaries[arm] = {
            "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "evaluation_seconds": elapsed,
        }
        np.savez_compressed(OUT / f"{arm}.npz", **collected, indices=np.arange(len(states)))
        r.atomic_json(OUT / "progress.json", {"status": "evaluating", "last": arm})
        print(json.dumps({arm: summaries[arm]}), flush=True)
        del model
        torch.cuda.empty_cache()

    score = scene_features(states[:, :20], masks)["composite"]
    cuts = np.quantile(score, [1 / 3, 2 / 3])
    groups = {
        "all": np.arange(len(score)),
        "low": np.flatnonzero(score <= cuts[0]),
        "medium": np.flatnonzero((score > cuts[0]) & (score <= cuts[1])),
        "high": np.flatnonzero(score > cuts[1]),
    }
    rng = np.random.default_rng(SEED)
    bootstrap = {}
    draws = {}
    for name, indices in groups.items():
        bootstrap[name], draws[name] = paired_bootstrap(arrays["physics_classical_dual"], arrays["physics_quantum_dual"], indices, rng)
    interaction = draws["high"] - draws["low"]
    bootstrap["high_minus_low_interaction"] = {
        key: {
            "median_percentage_points": float(np.median(interaction[:, column])),
            "ci95_percentage_points": [float(value) for value in np.quantile(interaction[:, column], [0.025, 0.975])],
            "bootstrap_probability_positive": float((interaction[:, column] > 0).mean()),
        }
        for column, key in enumerate(("ade", "fde"))
    }
    result = {
        "status": "completed",
        "seed": SEED,
        "split": "official test; historically used by prior project phases and not described as blind",
        "summaries": summaries,
        "paired_scene_bootstrap_repetitions": REPS,
        "complexity_bootstrap": bootstrap,
        "no_multiple_seeds": True,
    }
    r.atomic_json(OUT / "results.json", result)
    r.atomic_json(OUT / "completed.json", {"status": "completed", "results_sha256": sha(OUT / "results.json")})
    r.atomic_json(OUT / "progress.json", {"status": "completed"})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
