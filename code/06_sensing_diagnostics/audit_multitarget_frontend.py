"""SENS-AUDIT-01 cache-level audit: read-only inspection of the F01-D sensing cache.

Reads data/f01d for train/V_select only; never writes inside data/f01d.
Outputs land in reports/sensing_audit/sens_audit_01/.

Usage: python -u code/06_sensing_diagnostics/audit_multitarget_frontend.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/sensing_audit/sens_audit_01"
DATA = ROOT / "data/f01d"
SNRS = [5, 10, 15, 20]
ALLOWED_SPLITS = ("train", "V_select")

AUDITED_FILES = {
    "configs/symbol_frontend.json": "789168e8a9e323960ab5ade567af561b9424b5a119de4ddfc6d85764429eb7d8",
    "scripts/run_f01d_gpu.py": "f4d3b4bae329802ee5bc3af5271775ec9b10480ba26f1f1e02dd48a91aaedf9b",
    "scripts/rebuild_f01d_gpu.py": "74320c6a453c36f54ab99623b19ce85a7c197f851608d69d2965bc38a805e085",
    "frontend/generate_symbol_gpu.py": "234a9663cc6891fb057c0efadfd237373bfda37a2b86270dcaeadc0d0473c9ec",
    "frontend/symbol_level_gpu.py": "193849acb2e7b1a87de887c7bc6846634510fa35423ec2ac4ba0ef72e13269c7",
    "frontend/symbol_level.py": "4a0f00058be221cd0caa2994841bb923dcb64e3b21001fb9f8dc11e25f6f41b0",
    "frontend/echo_source.py": "5c59a41a04ae588765c8df4ffd7e16e2ed612e31ddb1352538323f7d652ccde4",
    "frontend/ofdm_echo.py": "43265204a3edd81aeadc1cb66dda10ac3200c8637c981a3f94161ba6f8e47ef0",
    "frontend/pack_symbol_dataset.py": "f964fef067a008e4474fd03b315c4ed033a7463e616d45707c978e6c438d192d",
    "frontend/symbol_dataset.py": "5eae742edaa773716d88544d366fb72fbaf1af1d99617484303b47317d078452",
    "frontend/run_symbol_frontend.py": "0ae6f6652f3455b49326fb4a3b5fa5a55667214672e795e857bed42bd6f8d802",
    "frontend/scene_manifest.py": "4482c0c61dd24cea6e869c35a927c013c74c9f8d82ef3755ea74b96c3b4e15a4",
    "frontend/check_symbol_dataset.py": "ddc6dea478b7f42b1da31f4dd957ca55ea3c14eaaa11bec258b926c1b2c9ac72",
    "frontend/detector.py": "db3bf5ce2aa521fcccad38fd8092f7e10b0d3a824a42b95f6f5103078ee48d2f",
    "frontend/jpda.py": "bde4408daf09fbc20b2deb8455de7b46bf6d07a494fc098fcd024d7019ca48ed",
    "frontend/run_frontend.py": "edb8ec5f296cad983efc22f6312e0b297d76bd5128488e54ee74405d27887f89",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(f, "")) for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inputs(split: str, snr: int) -> dict:
    with np.load(DATA / "inputs" / f"{split}_snr_{snr}.npz", allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def build_truth(source) -> dict:
    truth = {}
    for index, episode in enumerate(source.episodes):
        if episode["split"] not in ALLOWED_SPLITS:
            continue
        start = int(episode["start_ms"])
        arr = np.full((199, 8, 4), np.nan)
        for f in range(199):
            states, slots, _ = source.at_time(episode, (start + (f + 1) * 100) * 1_000_000)
            for state, slot in zip(states, slots):
                arr[f, int(slot)] = state
        truth[index] = arr
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from frontend.echo_source import SourceEpisodes

    OUT.mkdir(parents=True, exist_ok=True)
    checks = {"audit": "SENS-AUDIT-01",
              "generated_by": "code/06_sensing_diagnostics/audit_multitarget_frontend.py",
              "read_only_data_root": "data/f01d", "allowed_splits": list(ALLOWED_SPLITS),
              "source_hashes": {}, "checks": []}
    for name, expected in AUDITED_FILES.items():
        path = ROOT / name
        actual = sha256(path) if path.exists() else "missing"
        checks["source_hashes"][name] = {"sha256": actual, "expected": expected, "match": bool(actual == expected)}
    config = json.loads((ROOT / "configs/symbol_frontend.json").read_text())
    for key in ("revision", "scope", "target_access", "power_resource_model", "snr_definition", "seed_recipe"):
        checks[f"config_{key}"] = config[key]
    checks["checks"].append({"name": "audited_source_hashes_match_static_audit",
                             "passed": all(v["match"] for v in checks["source_hashes"].values())})

    generation = json.loads((ROOT / "reports/f01d/generation.json").read_text())
    run_status = json.loads((ROOT / "reports/f01d/run_status.json").read_text())
    packing = json.loads((ROOT / "reports/f01d/packing.json").read_text())
    checks["cache_provenance"] = {
        "generation_config_sha256": generation.get("frontend_hashes", {}).get("configs/symbol_frontend.json"),
        "run_status": {k: run_status.get(k) for k in ("status", "stage", "structural_validation_passed",
                                                      "search_failures", "estimated_target_states",
                                                      "prediction_training_started")},
        "packing_inputs": packing.get("input_keys"),
        "packing_rule": packing.get("rule"),
    }
    checks["checks"].append({"name": "cache_generated_by_audited_config_hash",
                             "passed": generation.get("frontend_hashes", {}).get("configs/symbol_frontend.json")
                             == AUDITED_FILES["configs/symbol_frontend.json"]})

    source = SourceEpisodes()
    expected_outputs = {}
    for row in generation["episodes"]:
        for cond in row["conditions"]:
            expected_outputs[(int(row["episode_index"]), int(cond["snr_db"]))] = int(cond.get("outputs", -1))

    sequence_rows = []
    sequence_checks = []
    for split in ALLOWED_SPLITS:
        episodes = [(i, ep) for i, ep in enumerate(source.episodes) if ep["split"] == split]
        for index, episode in episodes:
            for snr in SNRS:
                path = DATA / "sequences" / f"snr_{snr}" / f"episode_{index:03d}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    exists = archive["track_exists"]
                    detected = archive["detected"]
                    state = archive["state_hat"]
                outputs = int(exists.sum())
                sequence_rows.append({
                    "split": split, "episode_index": index, "snr_db": snr, "outputs": outputs,
                    "expected_outputs": expected_outputs.get((index, snr), -1),
                    "detected_true": int(detected.sum()), "exists_true": outputs,
                    "detected_equals_exists": bool(np.array_equal(detected, exists)),
                    "detected_without_exists": int((detected & ~exists).sum()),
                    "exists_without_detected": int((exists & ~detected).sum()),
                    "zero_state_on_absent": bool(np.all(state[~exists] == 0)),
                })
        rows = [r for r in sequence_rows if r["split"] == split]
        sequence_checks.append({
            "split": split, "episodes": len(episodes),
            "outputs_match_generation_json": all(r["outputs"] == r["expected_outputs"] for r in rows),
            "detected_equals_exists_all": all(r["detected_equals_exists"] for r in rows),
            "zero_state_on_absent_all": all(r["zero_state_on_absent"] for r in rows),
        })
    write_csv(OUT / "tables/sequence_outputs.csv", sequence_rows,
              ["split", "episode_index", "snr_db", "outputs", "expected_outputs", "detected_true",
               "exists_true", "detected_equals_exists", "detected_without_exists", "exists_without_detected",
               "zero_state_on_absent"])
    checks["checks"].append({"name": "sequence_outputs_match_generation_and_detected_is_presence",
                             "passed": all(c["outputs_match_generation_json"] and c["detected_equals_exists_all"]
                                           and c["zero_state_on_absent_all"] for c in sequence_checks),
                             "detail": sequence_checks})

    truth = build_truth(source)
    starts = {i: int(ep["start_ms"]) for i, ep in enumerate(source.episodes)}
    diff_rows, error_rows, detected_rows, figures = [], [], [], []
    for split in ALLOWED_SPLITS:
        arrays = {snr: load_inputs(split, snr) for snr in SNRS}
        base = arrays[20]
        for snr in SNRS:
            cur = arrays[snr]
            valid = base["track_exists"] | cur["track_exists"]
            delta = np.zeros(base["state_hat"].shape, dtype=np.float64)
            delta[valid] = np.abs(base["state_hat"][valid].astype(np.float64) - cur["state_hat"][valid].astype(np.float64))
            per_entry = delta.max(axis=-1)
            identical = np.all(base["state_hat"] == cur["state_hat"], axis=-1) & (base["track_exists"] == cur["track_exists"])
            n_valid = int(valid.sum())
            n_ident = int((identical & valid).sum())
            diff_rows.append({
                "split": split, "snr_db": snr, "valid_entries": n_valid, "identical_entries": n_ident,
                "identical_fraction": n_ident / n_valid if n_valid else 1.0,
                "nonzero_delta_entries": int((per_entry > 0).sum()),
                "max_abs_component_delta": float(per_entry.max()) if n_valid else 0.0,
                "p95_max_abs_component_delta": float(np.quantile(per_entry[valid], 0.95)) if n_valid else 0.0,
                "p99_max_abs_component_delta": float(np.quantile(per_entry[valid], 0.99)) if n_valid else 0.0,
                "mask_mismatch_entries": int((base["track_exists"] != cur["track_exists"]).sum()),
            })
            detected_rows.append({
                "split": split, "snr_db": snr,
                "detected_without_exists": int((cur["detected"] & ~cur["track_exists"]).sum()),
                "exists_without_detected": int((cur["track_exists"] & ~cur["detected"]).sum()),
                "detected_true": int(cur["detected"].sum()), "track_exists_true": int(cur["track_exists"].sum()),
                "detected_equals_exists": bool(np.array_equal(cur["detected"], cur["track_exists"])),
                "target_count_per_origin_min": int(cur["track_exists"].sum(axis=(1, 2)).min()),
                "target_count_per_origin_max": int(cur["track_exists"].sum(axis=(1, 2)).max()),
            })

        metadata = json.loads((DATA / "metadata" / f"{split}.json").read_text())["samples"]
        for snr in SNRS:
            cur = arrays[snr]
            pos_errs, vel_errs = [], []
            truth_missing = 0
            for j, record in enumerate(metadata):
                index = int(record["episode_index"])
                origin = int(record["origin_ms"])
                f_origin = (origin - starts[index]) // 100
                window = truth[index][f_origin - 20:f_origin]
                mask = cur["track_exists"][j]
                if not mask.any():
                    continue
                est = cur["state_hat"][j][mask].astype(np.float64)
                gt = window[mask]
                bad = np.isnan(gt).any(axis=1)
                truth_missing += int(bad.sum())
                est, gt = est[~bad], gt[~bad]
                if len(gt):
                    pos_errs.append(np.linalg.norm(est[:, :2] - gt[:, :2], axis=1))
                    vel_errs.append(np.linalg.norm(est[:, 2:] - gt[:, 2:], axis=1))
            pos = np.concatenate(pos_errs) if pos_errs else np.zeros(1)
            vel = np.concatenate(vel_errs) if vel_errs else np.zeros(1)
            entries = int(len(pos))
            error_rows.append({
                "split": split, "snr_db": snr, "entries": entries, "truth_missing": truth_missing,
                "position_rmse_m": float(np.sqrt(np.mean(pos ** 2))),
                "position_median_m": float(np.median(pos)), "position_p95_m": float(np.quantile(pos, 0.95)),
                "position_max_m": float(pos.max()),
                "velocity_rmse_mps": float(np.sqrt(np.mean(vel ** 2))),
                "velocity_median_mps": float(np.median(vel)), "velocity_p95_mps": float(np.quantile(vel, 0.95)),
                "velocity_max_mps": float(vel.max()),
            })

    write_csv(OUT / "tables/cache_snr_diff.csv", diff_rows,
              ["split", "snr_db", "valid_entries", "identical_entries", "identical_fraction",
               "nonzero_delta_entries", "max_abs_component_delta", "p95_max_abs_component_delta",
               "p99_max_abs_component_delta", "mask_mismatch_entries"])
    write_csv(OUT / "tables/cache_error_vs_snr.csv", error_rows,
              ["split", "snr_db", "entries", "truth_missing", "position_rmse_m", "position_median_m",
               "position_p95_m", "position_max_m", "velocity_rmse_mps", "velocity_median_mps",
               "velocity_p95_mps", "velocity_max_mps"])
    write_csv(OUT / "tables/detected_mask_audit.csv", detected_rows,
              ["split", "snr_db", "detected_true", "track_exists_true", "detected_equals_exists",
               "detected_without_exists", "exists_without_detected",
               "target_count_per_origin_min", "target_count_per_origin_max"])

    if not args.skip_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
            for split, color in (("train", "#1f77b4"), ("V_select", "#d62728")):
                rows = [r for r in error_rows if r["split"] == split]
                axes[0].plot([r["snr_db"] for r in rows], [r["position_rmse_m"] for r in rows], "o-", color=color, label=split)
                axes[1].plot([r["snr_db"] for r in rows], [r["velocity_rmse_mps"] for r in rows], "o-", color=color, label=split)
            axes[0].set(xlabel="SNR (dB)", ylabel="position RMSE (m)", title="Estimate vs source truth")
            axes[1].set(xlabel="SNR (dB)", ylabel="velocity RMSE (m/s)", title="Estimate vs source truth")
            for ax in axes:
                ax.axvspan(5, 20, color="#cccccc", alpha=0.35, zorder=0)
                ax.grid(alpha=0.3)
            axes[0].legend()
            fig.tight_layout()
            path = OUT / "figures/cache_error_vs_snr.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures.append(str(path.relative_to(ROOT)))

            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for split, color in (("train", "#1f77b4"), ("V_select", "#d62728")):
                rows = [r for r in diff_rows if r["split"] == split]
                ax.plot([r["snr_db"] for r in rows], [r["identical_fraction"] for r in rows], "o-", color=color, label=split)
            ax.set(xlabel="SNR (dB)", ylabel="fraction of identical target states vs 20 dB",
                   ylim=(0, 1.01), title="Cache state equality across SNR")
            ax.axvspan(5, 20, color="#cccccc", alpha=0.35, zorder=0)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = OUT / "figures/cache_identical_fraction.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures.append(str(path.relative_to(ROOT)))
        except Exception as error:
            checks["figure_error"] = repr(error)

    summary = {
        "audit": "SENS-AUDIT-01",
        "config_revision": config["revision"],
        "snr_db": SNRS,
        "splits": list(ALLOWED_SPLITS),
        "cache_snr_diff": diff_rows,
        "cache_error_vs_snr": error_rows,
        "detected_mask_audit": detected_rows,
        "sequence_checks": sequence_checks,
        "figures": figures,
    }
    write_json(OUT / "summary.json", summary)
    checks["checks"].append({"name": "detected_is_presence_mask_not_detector_output",
                             "passed": all(r["detected_equals_exists"] for r in detected_rows),
                             "detail": "detected==track_exists in every audited cache file"})
    write_json(OUT / "checks.json", checks)
    print(json.dumps({"config_revision": config["revision"], "cache_snr_diff": diff_rows,
                      "cache_error_vs_snr": error_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()