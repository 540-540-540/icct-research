"""Post-hoc control: select classical/plain fusion on discovery blocks, then confirm once."""
import json
import platform
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
import run_two_hop_rich_quantum as rich_run
import run_two_hop_score2_quantum as score2_run
import run_two_hop_sparse_experiment as sparse


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "two_hop_sparse_radius20_seed2026_v1"
OUTPUT = BASE / "two_hop_plain_fusion_control_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SEED = 2026


def load_model(name):
    if name in ("plain", "classical", "quantum"):
        model = sparse.build_model(f"{name}_radius20")
        checkpoint = REFERENCE / f"{name}_radius20_graph_selected.pt"
    elif name == "quantum_score2":
        model = score2_run.build_model(True)
        checkpoint = score2_run.OUTPUT / "quantum_score2_radius20_graph_selected.pt"
    elif name == "quantum_rich":
        model = rich_run.build_model(True)
        checkpoint = rich_run.OUTPUT / "quantum_rich_radius20_graph_selected.pt"
    else:
        raise ValueError(name)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state"], strict=True)
    return model.cuda().eval(), saved


@torch.no_grad()
def collect(model, bank):
    output = {}
    for snr in (5, 10, 15, 20):
        generator = torch.Generator(device="cuda").manual_seed(SEED + 100000)
        predictions, targets, masks = [], [], []
        for start in range(0, len(bank["history"]), 24):
            idx = torch.arange(start, min(start + 24, len(bank["history"])), device="cuda")
            history, future, mask = retained.make_batch(bank, idx, snr, generator)
            prediction = model(history, mask)["future_position"]
            predictions.append(prediction.cpu())
            targets.append(future[..., :2].cpu())
            masks.append(mask.cpu())
        output[str(snr)] = {
            "prediction": torch.cat(predictions),
            "target": torch.cat(targets),
            "mask": torch.cat(masks),
        }
    return output


def metrics(predictions):
    by_snr = {}
    scene_arrays = {}
    for snr, payload in predictions.items():
        distance = (payload["prediction"] - payload["target"]).norm(dim=-1).double()
        mask = payload["mask"]
        sum_t = (distance * mask[:, None, :]).sum((1, 2))
        sum_f = (distance[:, -1] * mask).sum(1)
        count = mask.sum(1).double()
        array = torch.stack([sum_t, sum_f, count], dim=1).numpy()
        denominator = array[:, 2].sum()
        by_snr[snr] = {
            "ade_m": float(array[:, 0].sum() / (denominator * 20)),
            "fde_m": float(array[:, 1].sum() / denominator),
        }
        scene_arrays[snr] = array
    aggregate = {
        key: float(np.mean([value[key] for value in by_snr.values()]))
        for key in ("ade_m", "fde_m")
    }
    return {"aggregate": aggregate, "by_snr": by_snr}, scene_arrays


def fuse(classical, quantum, alpha):
    result = {}
    for snr in classical:
        assert torch.equal(classical[snr]["target"], quantum[snr]["target"])
        assert torch.equal(classical[snr]["mask"], quantum[snr]["mask"])
        result[snr] = {
            "prediction": (1.0 - alpha) * classical[snr]["prediction"] + alpha * quantum[snr]["prediction"],
            "target": classical[snr]["target"],
            "mask": classical[snr]["mask"],
        }
    return result


def score(metric):
    return metric["aggregate"]["ade_m"] + 0.35 * metric["aggregate"]["fde_m"]


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    OUTPUT.mkdir(exist_ok=False)
    protocol = json.loads((REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    with np.load(DATA, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
        starts = data["train_start_index"].copy()
    selection_indices = np.asarray(protocol["selection_indices"], dtype=np.int64)
    confirmation_indices = np.asarray(protocol["confirmation_indices"], dtype=np.int64)
    selection = sparse.bank(states, masks, selection_indices)
    confirmation = sparse.bank(states, masks, confirmation_indices)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")

    names = ("classical", "plain")
    selection_predictions = {}
    selection_checkpoints = {}
    for name in names:
        model, saved = load_model(name)
        selection_predictions[name] = collect(model, selection)
        selection_checkpoints[name] = {
            "epoch": int(saved["epoch"]), "metrics": saved["metrics"]
        }
        del model
        torch.cuda.empty_cache()

    candidates = []
    for quantum_name in names[1:]:
        for alpha in np.linspace(0.0, 1.0, 21):
            metric, _ = metrics(fuse(selection_predictions["classical"], selection_predictions[quantum_name], float(alpha)))
            candidates.append({
                "quantum_branch": quantum_name,
                "alpha": float(alpha),
                "metrics": metric,
                "score": score(metric),
            })
    selected = min(candidates, key=lambda item: item["score"])

    # Confirmation is opened only after branch and alpha have been frozen above.
    confirmation_predictions = {}
    for name in ("classical", selected["quantum_branch"]):
        model, _ = load_model(name)
        confirmation_predictions[name] = collect(model, confirmation)
        del model
        torch.cuda.empty_cache()
    confirmation_metric, confirmation_arrays = metrics(fuse(
        confirmation_predictions["classical"],
        confirmation_predictions[selected["quantum_branch"]],
        selected["alpha"],
    ))
    classical_metric, classical_arrays = metrics(confirmation_predictions["classical"])
    np.savez_compressed(
        OUTPUT / "fusion_confirmation.npz",
        **{f"fusion_{key}": value for key, value in confirmation_arrays.items()},
        **{f"classical_{key}": value for key, value in classical_arrays.items()},
        indices=confirmation_indices,
        starts=starts[confirmation_indices],
    )
    result = {
        "status": "completed",
        "seed": SEED,
        "selection_rule": "minimum ADE + 0.35 FDE on discovery time blocks",
        "alpha_grid": np.linspace(0.0, 1.0, 21).tolist(),
        "selected": selected,
        "confirmation": confirmation_metric,
        "classical_confirmation": classical_metric,
        "confirmation_used_after_selection_only": True,
        "no_llm": True,
        "no_test": True,
        "checkpoint_selection_rechecks": selection_checkpoints,
    }
    r.atomic_json(OUTPUT / "completed.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
