from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate_a.data import ego_transform, pair_metrics
from frontend.controlled_isac.automatum_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import calibration_from_config, setup_from_config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def intervals(states: np.ndarray, cfg: dict) -> dict[str, tuple[int, int]]:
    start, end = int(states["time_ms"].min()), int(states["time_ms"].max()) + 100
    fractions = list(cfg["split_fractions"].values())
    cuts = np.rint((start + (end - start) * np.cumsum([0.0] + fractions)) / 100).astype(np.int64) * 100
    cuts[0], cuts[-1] = start, end
    guard = int(round(cfg["guard_s"] * 1000))
    names = list(cfg["split_fractions"])
    return {name: (int(cuts[i] + (guard if i else 0)),
                   int(cuts[i + 1] - (guard if i < len(names) - 1 else 0)))
            for i, name in enumerate(names)}


def sensing_setup(root: Path) -> tuple[dict, dict, dict]:
    cfg = json.loads((root / "configs/automatum_controlled_isac.json").read_text())
    geometry = json.loads((root / "reports/f01a/geometry.json").read_text())
    cfg["scenes"] = [{"scene_id": 0, "stations_xy_m": geometry["stations_xy_m"],
                      "boresights_deg": geometry["boresight_deg"]}]
    cfg["visibility"] = {"range_min_m": geometry["range_m"][0],
                         "range_max_m": geometry["range_m"][1],
                         "fov_half_angle_deg": geometry["fov_half_angle_deg"],
                         "height_difference_m": geometry["station_height_m"] - geometry["vehicle_height_m"]}
    return setup_from_config(cfg), calibration_from_config(cfg), geometry


def history_ok(states: np.ndarray) -> np.ndarray:
    ok = np.zeros(len(states), dtype=bool)
    for _, indices in group_indices(states["source_key"]).items():
        if len(indices) >= 20:
            t = states["time_ms"][indices]
            good = (t[19:] - t[:-19]) == 1900
            ok[indices[19:][good]] = True
    return ok


def group_indices(keys: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    values, starts = np.unique(keys[order], return_index=True)
    return {int(k): x for k, x in zip(values, np.split(order, starts[1:]))}


def assign_splits(states: np.ndarray, bounds: dict) -> tuple[np.ndarray, dict]:
    row_split = np.full(len(states), -1, np.int8)
    names = ["train", "val", "test"]
    for code, name in enumerate(names):
        lo, hi = bounds[name]
        row_split[(states["time_ms"] >= lo) & (states["time_ms"] < hi) & states["owned"]] = code
    owners = {}
    crossing = set()
    by_group = defaultdict(set)
    for group, code in zip(states["source_group"], row_split):
        if code >= 0:
            by_group[int(group)].add(int(code))
    for group, codes in by_group.items():
        if len(codes) == 1:
            owners[group] = next(iter(codes))
        else:
            crossing.add(group)
    for group in crossing:
        row_split[states["source_group"] == group] = -1
    return row_split, {"source_groups_owned": len(owners), "source_groups_purged_cross_split": len(crossing)}


def sense_rows(states: np.ndarray, row_split: np.ndarray, setup: dict, calibration: dict,
               snr: float) -> tuple[np.ndarray, Counter]:
    result = np.zeros((len(states), 4), np.float32)
    counts = Counter()
    origin = int(states["time_ms"].min())
    for number, i in enumerate(np.flatnonzero(row_split >= 0), 1):
        row = states[i]
        estimate = sense_vehicle(0, (int(row["time_ms"]) - origin) // 100,
                                 int(row["source_key"]), [row["x"], row["y"]],
                                 [row["vx"], row["vy"]], snr, calibration, setup)
        result[i] = [estimate[k] for k in ("x_hat", "y_hat", "vx_hat", "vy_hat")]
        counts[int(estimate["n_bs"])] += 1
        if number % 200000 == 0:
            print(f"sensed {number} legal rows", flush=True)
    return result, counts


def neighbor_order(target: int, metrics: dict, keys: np.ndarray) -> list[int]:
    choices = np.flatnonzero(metrics["near"][target]).tolist()
    def key(j):
        tcpa = metrics["tcpa"][target, j]
        valid_tcpa = 0 < tcpa <= 4.0
        return (-int(metrics["edge"][target, j]), metrics["ttc"][target, j],
                metrics["dcpa"][target, j] if valid_tcpa else np.inf,
                -metrics["closing"][target, j], metrics["distance"][target, j], int(keys[j]))
    return sorted(choices, key=key)


def build_records(states: np.ndarray, sensed: np.ndarray, row_split: np.ndarray, hist_ok: np.ndarray,
                  cfg: dict, split_code: int) -> list[dict]:
    by_key = group_indices(states["source_key"])
    position_in_track = np.empty(len(states), np.int32)
    for indices in by_key.values():
        position_in_track[indices] = np.arange(len(indices), dtype=np.int32)
    current = defaultdict(list)
    for i in np.flatnonzero((row_split == split_code) & hist_ok):
        track = by_key[int(states["source_key"][i])]
        p = int(position_in_track[i])
        selected = track[p - 19:p + 1]
        if len(selected) == 20 and np.all(row_split[selected] == split_code):
            current[int(states["time_ms"][i])].append(int(i))
    stride_ms = int(round(cfg["origin_stride_steps"] * cfg["dt_s"] * 1000))
    records = []
    for origin_no, time_ms in enumerate(sorted(current)):
        if time_ms % stride_ms != int(states["time_ms"].min()) % stride_ms:
            continue
        rows = np.asarray(current[time_ms], np.int64)
        if len(rows) < 2:
            continue
        estimates = sensed[rows]
        metrics = pair_metrics(estimates, cfg)
        keys = states["source_key"][rows]
        for target in range(len(rows)):
            ordered = neighbor_order(target, metrics, keys)
            nodes = [target] + ordered
            node_rows = rows[nodes]
            histories = []
            for row_index in node_rows:
                track = by_key[int(states["source_key"][row_index])]
                p = int(position_in_track[row_index])
                selected = track[p - 19:p + 1]
                if len(selected) != 20 or not np.array_equal(states["time_ms"][selected], time_ms + np.arange(-19, 1) * 100):
                    raise RuntimeError("history contract violation")
                histories.append(sensed[selected])
            history = np.stack(histories, axis=1)
            future = np.zeros((40, 2), np.float32)
            future_mask = np.zeros(40, bool)
            target_row = int(rows[target])
            track = by_key[int(states["source_key"][target_row])]
            p = int(position_in_track[target_row])
            for step in range(1, 41):
                q = p + step
                if (q < len(track) and row_split[track[q]] == split_code
                        and states["time_ms"][track[q]] == time_ms + step * 100):
                    future[step - 1] = [states["x"][track[q]], states["y"][track[q]]]
                    future_mask[step - 1] = True
            history, future, _, _ = ego_transform(history, future, future_mask)
            edge_row = metrics["edge"][target]
            near = metrics["near"][target]
            valid_tcpa = near & (metrics["tcpa"][target] > 0) & (metrics["tcpa"][target] <= cfg["tcpa_s"])
            records.append({"history": history.astype(np.float32), "future": future,
                            "future_mask": future_mask, "source_key": int(keys[target]),
                            "source_group": int(states["source_group"][target_row]),
                            "time_ms": time_ms, "n_context": int(near.sum()),
                            "k": int(edge_row.sum()), "has_cpa": bool(metrics["cpa"][target].any()),
                            "has_following": bool(metrics["following"][target].any()),
                            "max_closing": float(metrics["closing"][target, near].max()) if near.any() else 0.0,
                            "min_dcpa": float(metrics["dcpa"][target, valid_tcpa].min()) if valid_tcpa.any() else 999.0})
        if origin_no % 100 == 0:
            print(f"split {split_code}: origin {origin_no}/{len(current)}, samples {len(records)}", flush=True)
    return records


def pack(path: Path, records: list[dict], max_nodes: int):
    n = len(records)
    history = np.zeros((n, 20, max_nodes, 4), np.float32)
    node_mask = np.zeros((n, max_nodes), bool)
    future = np.zeros((n, 40, 2), np.float32)
    future_mask = np.zeros((n, 40), bool)
    scalars = {k: [] for k in ("source_key", "source_group", "time_ms", "n_context", "k",
                                "has_cpa", "has_following", "max_closing", "min_dcpa")}
    for i, record in enumerate(records):
        nodes = record["history"].shape[1]
        history[i, :, :nodes] = record["history"]
        node_mask[i, :nodes] = True
        future[i], future_mask[i] = record["future"], record["future_mask"]
        for key in scalars:
            scalars[key].append(record[key])
    arrays = dict(history_state=history, node_mask=node_mask, future_xy=future,
                  future_mask=future_mask, **{k: np.asarray(v) for k, v in scalars.items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/task_redesign/lankershim_gate_a_v1")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/gate_a_lankershim_ic4.json").read_text())
    raw = ROOT / "data/Lankershim_Vehicle_Trajectories.csv"
    if sha256(raw) != cfg["raw_sha256"]:
        raise RuntimeError("raw source hash mismatch")
    states = np.load(ROOT / cfg["source_states"], allow_pickle=False)
    bounds = intervals(states, cfg)
    row_split, split_audit = assign_splits(states, bounds)
    if args.repair_existing:
        for name in ("train", "val"):
            path = out / f"{name}.npz"
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            lo, hi = bounds[name]
            keep = arrays["time_ms"] >= lo + 1900
            arrays = {k: v[keep] for k, v in arrays.items()}
            legal_future = arrays["time_ms"][:, None] + np.arange(1, 41)[None] * 100 < hi
            arrays["future_mask"] &= legal_future
            arrays["future_xy"][~arrays["future_mask"]] = 0.0
            tmp = path.with_suffix(".repair.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
        args.finalize_existing = True
    if args.finalize_existing:
        summaries, groups, max_nodes = {}, {}, 0
        for name in ("train", "val"):
            with np.load(out / f"{name}.npz", allow_pickle=False) as z:
                summaries[name] = {"samples": int(len(z["history_state"])),
                                   "origins": int(len(np.unique(z["time_ms"]))),
                                   "ic": int(np.sum(z["k"] >= 1)),
                                   "future4_any": int(np.sum(z["future_mask"].any(1))),
                                   "future4_endpoint": int(np.sum(z["future_mask"][:, -1]))}
                groups[name] = set(map(int, z["source_group"]))
                max_nodes = max(max_nodes, int(z["history_state"].shape[2]))
        if groups["train"] & groups["val"]:
            raise RuntimeError("cross-split source group leakage")
        manifest = {"status": "COMPLETE", "revision": cfg["revision"], "strict_causal": True,
                    "history_dependency": "raw Local_X/Y at t<=t0; OLS velocity uses current plus <=4 past frames",
                    "future_dependency": "future raw Local_X/Y only in future_xy/future_mask supervision",
                    "sensing": "frozen controlled 3-BS polar measurement and same-frame fusion at 0 dB",
                    "raw_sha256": cfg["raw_sha256"], "source_states_sha256": sha256(ROOT / cfg["source_states"]),
                    "intervals_ms": bounds, "test_materialized": False, "test_access": "time bounds only; no test cache",
                    "split_audit": split_audit, "visible_bs_row_counts": "validated during build; not replayed in finalize-existing",
                    "max_nodes": max_nodes, "splits": summaries,
                    "thresholds": {k: cfg[k] for k in ("radius_m", "closing_mps", "tcpa_s", "dcpa_m",
                                                                   "following_heading_deg", "following_headway_s",
                                                                   "following_lateral_m")},
                    "files": {f"{name}.npz": {"bytes": (out / f"{name}.npz").stat().st_size,
                                                "sha256": sha256(out / f"{name}.npz")}
                              for name in ("train", "val")}}
        write_json(out / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2)); return
    ok = history_ok(states)
    setup, calibration, geometry = sensing_setup(ROOT)
    sensed, bs_counts = sense_rows(states, row_split, setup, calibration, cfg["snr_db"])
    all_records = {}
    for code, name in enumerate(("train", "val")):
        all_records[name] = build_records(states, sensed, row_split, ok, cfg, code)
    max_nodes = max(r["history"].shape[1] for records in all_records.values() for r in records)
    for name, records in all_records.items():
        pack(out / f"{name}.npz", records, max_nodes)
    groups = {name: set(r["source_group"] for r in records) for name, records in all_records.items()}
    if groups["train"] & groups["val"]:
        raise RuntimeError("cross-split source group leakage")
    manifest = {"status": "COMPLETE", "revision": cfg["revision"], "strict_causal": True,
                "history_dependency": "raw Local_X/Y at t<=t0; OLS velocity uses current plus <=4 past frames",
                "future_dependency": "future raw Local_X/Y only in future_xy/future_mask supervision",
                "sensing": "frozen controlled 3-BS polar measurement and same-frame fusion at 0 dB",
                "raw_sha256": cfg["raw_sha256"], "source_states_sha256": sha256(ROOT / cfg["source_states"]),
                "intervals_ms": bounds, "test_materialized": False, "test_access": "time bounds only; no test cache",
                "split_audit": split_audit, "visible_bs_row_counts": dict(bs_counts), "max_nodes": max_nodes,
                "splits": {name: {"samples": len(records), "origins": len({r["time_ms"] for r in records}),
                                  "ic": int(sum(r["k"] >= 1 for r in records)),
                                  "future4_any": int(sum(r["future_mask"].any() for r in records)),
                                  "future4_endpoint": int(sum(r["future_mask"][-1] for r in records))}
                           for name, records in all_records.items()},
                "thresholds": {k: cfg[k] for k in ("radius_m", "closing_mps", "tcpa_s", "dcpa_m",
                                                               "following_heading_deg", "following_headway_s",
                                                               "following_lateral_m")},
                "files": {f"{name}.npz": {"bytes": (out / f"{name}.npz").stat().st_size,
                                            "sha256": sha256(out / f"{name}.npz")}
                          for name in all_records}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
