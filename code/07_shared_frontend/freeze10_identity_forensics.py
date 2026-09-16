"""F01E-IDENTITY-AUDIT-10: forensic attribution of same-segment tracker identity switches (CPU-only).

Train split only. ``data/f01e`` is strictly read-only: per-subtree SHA256 manifests are taken
before and after the analysis and must be identical. No sensing, tracking, matcher, label or
loader code is modified; the frozen official matcher is re-instantiated here
(``freeze07_generate_f01e.origin_eligibility`` / ``gt_track_xy`` plus the identical cost, gate and
Hungarian rules) to rebuild slot x GT cost matrices, and every reconstructed window assignment and
cost is compared against the frozen metadata before any classification is reported.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/identity_audit_10"
SNR_LIST = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
EXPECTED_TRANSITIONS = 139987
EXPECTED_SWITCHES = 4676
EXPECTED_SWITCH_RATE = 0.03340310171658797
EXPECTED_LABELLED_SLOTS = 161703
EXPECTED_VALID_SLOT_LABELS = 3168589
EXPECTED_ELIGIBLE_SLOTS = 173862
GATE_M = 5.0
MARGIN_M = 0.5
CROWD_GT_M = 1.0
WINDOW_RADIUS = 3
INTEGRITY_SUBTREES = ("inputs", "sequences", "labels", "metadata")

CAUSES = ("crowded_ambiguous", "global_assignment_competition", "matcher_jitter_clear",
          "true_tracker_handoff_clear", "unresolved")

CSV_FIELDS = (
    "switch_id", "episode", "episode_id", "density", "snr", "slot", "window_offset", "origin_ms",
    "state_x", "state_y", "assigned_gt", "nearest_gt", "second_nearest_gt", "nearest_margin_m",
    "d_state_A", "d_state_B", "gt_AB_distance", "detected", "coast",
    "gt_before", "gt_after", "detected_before", "detected_after", "detection_transition",
    "assigned_cost_m", "row_best_gt", "row_best_cost_m", "row_second_best_cost_m",
    "row_cost_margin_m", "assigned_is_row_best", "row_best_taken_by_other_slot",
    "crowded_criterion", "competition_criterion", "jitter_criterion", "handoff_criterion", "cause",
    "persistence_b_origins", "persistent_ge_3", "persistent_to_segment_end",
    "bounce_within_1", "bounce_within_2", "bounce_within_3",
    "nearest_p", "nearest_c", "nearest_r", "jitter_bounce_strict")


def snr_key(snr: float) -> str:
    return str(int(snr))


def density_bucket(episode: dict) -> str:
    count = len(episode["source_keys"])
    return "low" if count <= 3 else ("mid" if count <= 6 else "high")


# ---------------------------------------------------------------------------
# integrity manifests (read-only proof for data/f01e)
# ---------------------------------------------------------------------------

def subtree_manifest(base: Path) -> dict:
    entries = []
    total_bytes = 0
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        total_bytes += path.stat().st_size
        entries.append(f"{path.relative_to(base).as_posix()}:{digest}")
    return {"files": len(entries), "bytes": total_bytes,
            "combined_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}


def manifest_set() -> dict:
    return {name: subtree_manifest(DATA / name) for name in INTEGRITY_SUBTREES}


# ---------------------------------------------------------------------------
# frozen-matcher re-instantiation
# ---------------------------------------------------------------------------

def reconstruct_window(sequence: dict, episode: dict, source, origin_ms: int) -> dict | None:
    """Rebuild the official slot x vehicle cost matrix for one prediction origin.

    Identical rules to ``freeze07_generate_f01e.align_window``: origin-eligible slots only, per-slot
    cost over its own contiguous alive segment, mean Euclidean xy distance over >=3 common frames,
    5 m gate, Hungarian. Returns None when no slot is origin-eligible.
    """
    indices, exists, segments = generator.origin_eligibility(sequence, origin_ms)
    if not segments:
        return None
    times = (int(origin_ms) + np.arange(-19, 1, dtype=np.int64) * 100).astype(np.int64)
    states = sequence["state_hat"][indices]
    vehicles = [int(key) for key in episode["source_keys"]]
    gt_xy = np.full((len(vehicles), 20, 2), np.nan)
    for vehicle_index, key in enumerate(vehicles):
        gt_xy[vehicle_index], _ = generator.gt_track_xy(source.tracks[key], times)
    slots = sorted(segments)
    cost = np.full((len(slots), len(vehicles)), 1e9)
    common = np.zeros((len(slots), len(vehicles)), np.int64)
    for row, slot in enumerate(slots):
        start = segments[slot]["segment_start_history_index"]
        for vehicle_index in range(len(vehicles)):
            mask = exists[start:, slot] & np.isfinite(gt_xy[vehicle_index, start:, 0])
            common[row, vehicle_index] = int(mask.sum())
            if common[row, vehicle_index] >= 3:
                distance = np.linalg.norm(states[start:, slot, :2][mask]
                                          - gt_xy[vehicle_index, start:][mask], axis=1)
                cost[row, vehicle_index] = float(distance.mean())
    small = np.where(cost <= GATE_M, cost, 1e9)
    rows, columns = linear_sum_assignment(small) if small.size else ([], [])
    assignment = {}
    for row, vehicle_index in zip(np.asarray(rows).tolist(), np.asarray(columns).tolist()):
        if small[row, vehicle_index] < 1e9:
            assignment[int(slots[row])] = int(vehicles[vehicle_index])
    return {"slots": slots, "vehicles": vehicles, "cost": cost, "small": small, "common": common,
            "assignment": assignment, "indices": indices, "segments": segments, "states": states}


def gt_positions_at(source, episode: dict, origin_ms: int) -> dict:
    positions = {}
    for key in episode["source_keys"]:
        xy, exact = generator.gt_track_xy(source.tracks[int(key)],
                                          np.asarray([origin_ms], np.int64))
        if bool(exact[0]) and np.isfinite(xy[0, 0]):
            positions[int(key)] = (float(xy[0, 0]), float(xy[0, 1]))
    return positions


def nearest_evidence(point, gt_positions: dict) -> tuple:
    if point is None:
        return None, None, None
    distances = {key: float(np.linalg.norm(point - np.asarray(xy)))
                 for key, xy in gt_positions.items()}
    ordered = sorted(distances.items(), key=lambda item: item[1])
    if not ordered:
        return None, None, None
    second = ordered[1][1] if len(ordered) > 1 else None
    margin = (ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else None
    return (int(ordered[0][0]),
            int(ordered[1][0]) if len(ordered) > 1 else None,
            margin)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    before_manifest = manifest_set()

    from frontend.echo_source import SourceEpisodes
    source = SourceEpisodes()
    metadata = json.loads((DATA / "metadata" / "train.json").read_text())
    samples = metadata["samples"]
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)

    label_valid = {}
    baseline = {"valid_slot_labels": 0, "labelled_slots": 0, "eligible_slots": 0}
    for snr in SNR_LIST:
        with np.load(DATA / "labels" / f"train_{generator.snr_name(snr)}.npz",
                     allow_pickle=False) as payload:
            valid = payload["label_valid"]
        label_valid[snr] = valid
        labelled = int(valid.any(axis=1).sum())
        baseline["valid_slot_labels"] += int(valid.sum())
        baseline["labelled_slots"] += labelled

    # ------------------------------------------------------------------
    # phase A: reproduction of the frozen switch statistics
    # ------------------------------------------------------------------
    transitions = Counter()
    switches = []
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        density = density_bucket(episode)
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        sample_indices = [int(sample["sample_index"]) for sample in episode_samples]
        for snr in SNR_LIST:
            sequence = sequences[snr]
            segments_by_window = []
            detected_by_window = []
            maps = []
            for sample in episode_samples:
                indices, _, segments = generator.origin_eligibility(sequence,
                                                                    int(sample["origin_ms"]))
                segments_by_window.append(segments)
                detected_by_window.append(sequence["detected"][int(indices[19])].copy())
                maps.append({int(slot): int(key) for slot, key in
                             sample["slot_alignment_by_snr"][snr_key(snr)].items()})
                baseline["eligible_slots"] += len(segments)
            for window in range(1, len(episode_samples)):
                gap_frames = (int(episode_samples[window]["origin_ms"])
                              - int(episode_samples[window - 1]["origin_ms"])) // 100
                for slot, entry in segments_by_window[window].items():
                    if entry["contiguous_alive_length"] < gap_frames + 1:
                        continue
                    if slot not in maps[window - 1] or slot not in maps[window]:
                        continue
                    detected_before = bool(detected_by_window[window - 1][slot])
                    detected_after = bool(detected_by_window[window][slot])
                    detection_transition = ("detected" if detected_before else "coast") + "_to_" + \
                                           ("detected" if detected_after else "coast")
                    transitions["all"] += 1
                    transitions[f"snr_{snr_key(snr)}"] += 1
                    transitions[f"density_{density}"] += 1
                    transitions[f"dt_{detection_transition}"] += 1
                    if maps[window - 1][slot] != maps[window][slot]:
                        switches.append({
                            "episode": episode_index, "episode_id": episode["episode_id"],
                            "density": density, "snr": snr, "slot": slot, "window": window,
                            "window_before": window - 1,
                            "origin_ms": int(episode_samples[window]["origin_ms"]),
                            "origin_before_ms": int(episode_samples[window - 1]["origin_ms"]),
                            "gt_before": int(maps[window - 1][slot]),
                            "gt_after": int(maps[window][slot]),
                            "detected_before": detected_before, "detected_after": detected_after,
                            "detection_transition": detection_transition})
        print(json.dumps({"phase": "scan", "episode": episode_index, "switches": len(switches),
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

    switch_count = len(switches)
    switch_rate = switch_count / max(transitions["all"], 1)
    baseline["coverage_ratio"] = baseline["labelled_slots"] / max(baseline["eligible_slots"], 1)
    reproduction_pass = (transitions["all"] == EXPECTED_TRANSITIONS
                         and switch_count == EXPECTED_SWITCHES
                         and abs(switch_rate - EXPECTED_SWITCH_RATE) < 1e-9
                         and baseline["labelled_slots"] == EXPECTED_LABELLED_SLOTS
                         and baseline["valid_slot_labels"] == EXPECTED_VALID_SLOT_LABELS
                         and baseline["eligible_slots"] == EXPECTED_ELIGIBLE_SLOTS)
    if not reproduction_pass:
        after_manifest = manifest_set()
        summary = {
            "stage": "F01E-IDENTITY-AUDIT-10", "dataset": "F01E", "split": "train",
            "f01d_used": False, "expected_switches": EXPECTED_SWITCHES,
            "reproduced_switches": switch_count,
            "reproduced_transitions": transitions["all"], "switch_rate": switch_rate,
            "reproduction_pass": False, "next_action": "STOP_REPRODUCTION_FAILED",
            "baseline_labels": baseline}
        integrity = {"stage": "F01E-IDENTITY-AUDIT-10", "before": before_manifest,
                     "after": after_manifest,
                     "unchanged": {name: before_manifest[name] == after_manifest[name]
                                   for name in INTEGRITY_SUBTREES},
                     "reproduction_pass": False, "aborted_before_classification": True}
        for name, payload in (("summary.json", summary), ("integrity.json", integrity)):
            with (OUT / name).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
        print(json.dumps({"overall": "REPRODUCTION_FAILED", "transitions": transitions["all"],
                          "switches": switch_count, "baseline": baseline}, ensure_ascii=False))
        return

    # ------------------------------------------------------------------
    # phase B: per-switch forensics
    # ------------------------------------------------------------------
    switches_by_key = defaultdict(list)
    for entry in switches:
        switches_by_key[(entry["episode"], entry["snr"])].append(entry)

    csv_rows = []
    cause_counts = Counter()
    criteria_overlap = Counter()
    cause_by_density = defaultdict(Counter)
    cause_by_snr = defaultdict(Counter)
    cause_by_dt = defaultdict(Counter)
    handoff_by_density = Counter()
    handoff_by_snr = Counter()
    handoff_by_dt = Counter()
    bounce_exact = Counter()
    persistence_runs = Counter()
    matcher_mismatch = Counter()
    matcher_cost_mismatch = 0
    chain_dominance = Counter()
    dominant_fractions = []
    handoff_segments = set()
    handoff_labelled_cells = set()
    handoff_pm2_cells = set()
    policy_sets = {"P1_exact": set(), "P2_pm1": set(), "P3_pm2": set()}
    switch_counter = 0

    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        density = density_bucket(episode)
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        n_windows = len(episode_samples)
        origins = [int(sample["origin_ms"]) for sample in episode_samples]
        sample_indices = [int(sample["sample_index"]) for sample in episode_samples]
        gt_position_cache = {}

        def gt_at(origin_ms, _cache=gt_position_cache):
            if origin_ms not in _cache:
                _cache[origin_ms] = gt_positions_at(source, episode, origin_ms)
            return _cache[origin_ms]

        for snr in SNR_LIST:
            sequence = sequences[snr]
            recon_cache = {}
            maps = []
            detected = []
            segments_by_window = []
            for sample in episode_samples:
                indices, _, segments = generator.origin_eligibility(sequence,
                                                                    int(sample["origin_ms"]))
                segments_by_window.append(segments)
                detected.append(sequence["detected"][int(indices[19])].copy())
                maps.append({int(slot): int(key) for slot, key in
                             sample["slot_alignment_by_snr"][snr_key(snr)].items()})
            edge = np.zeros((max(n_windows - 1, 0), 8), bool)
            for window in range(1, n_windows):
                gap_frames = (origins[window] - origins[window - 1]) // 100
                for slot, info in segments_by_window[window].items():
                    if info["contiguous_alive_length"] >= gap_frames + 1:
                        edge[window - 1, slot] = True
            chains = {slot: [] for slot in range(8)}
            chain_of = {slot: {} for slot in range(8)}
            for slot in range(8):
                current = None
                for window in range(n_windows):
                    if slot not in segments_by_window[window]:
                        current = None
                        continue
                    if current is None or not edge[window - 1, slot]:
                        current = len(chains[slot])
                        chains[slot].append([])
                    chain_of[slot][window] = current
                    chains[slot][current].append(window)

            # section 22: per-chain GT identity dominance (labels from the frozen metadata)
            for slot in range(8):
                for windows in chains[slot]:
                    labels = [maps[window][slot] for window in windows if slot in maps[window]]
                    chain_dominance["segments_total"] += 1
                    if len(windows) == 1:
                        chain_dominance["single_window_segments"] += 1
                    else:
                        chain_dominance["segments_len_ge2"] += 1
                    if not labels:
                        chain_dominance["segments_with_no_assigned_label"] += 1
                        continue
                    chain_dominance["labelled_segments"] += 1
                    if len(windows) >= 2:
                        fraction = max(Counter(labels).values()) / len(labels)
                        dominant_fractions.append(fraction)
                        if fraction >= 0.90:
                            chain_dominance["dominant_ge_0p90"] += 1
                        if fraction >= 0.95:
                            chain_dominance["dominant_ge_0p95"] += 1
                        if fraction < 0.5:
                            chain_dominance["no_clear_dominant"] += 1

            def recon_at(origin_ms, _sequence=sequence, _cache=recon_cache):
                if origin_ms not in _cache:
                    _cache[origin_ms] = reconstruct_window(_sequence, episode, source, origin_ms)
                return _cache[origin_ms]

            for entry in switches_by_key.get((episode_index, snr), []):
                slot = entry["slot"]
                window = entry["window"]
                key_a = entry["gt_before"]
                key_b = entry["gt_after"]
                switch_id = switch_counter
                switch_counter += 1

                # verify the reconstructed matcher against the frozen metadata for this switch
                for probe in (window, entry["window_before"]):
                    recon_probe = recon_at(origins[probe])
                    recorded_probe = {int(k): int(v) for k, v in
                                      episode_samples[probe]["slot_alignment_by_snr"][
                                          snr_key(snr)].items()}
                    if recon_probe is None or recon_probe["assignment"] != recorded_probe:
                        matcher_mismatch["assignment"] += 1
                recon_switch = recon_at(origins[window])
                for detail in episode_samples[window]["slot_alignment_detail_by_snr"][snr_key(snr)]:
                    row = recon_switch["slots"].index(int(detail["slot"]))
                    column = recon_switch["vehicles"].index(int(detail["source_key"]))
                    if abs(float(detail["cost_m"]) - float(recon_switch["cost"][row, column])) > 1e-6 \
                            or int(detail["common_frames"]) != int(recon_switch["common"][row, column]):
                        matcher_cost_mismatch += 1

                window_rows = {}
                for offset in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
                    position = window + offset
                    if not 0 <= position < n_windows:
                        continue
                    origin_ms = origins[position]
                    recon = recon_at(origin_ms)
                    assigned = maps[position].get(slot)
                    point = None
                    if recon is not None and slot in recon["segments"]:
                        point = recon["states"][19, slot, :2].astype(np.float64)
                    gt_positions = gt_at(origin_ms)
                    distances = ({key: float(np.linalg.norm(point - np.asarray(xy)))
                                  for key, xy in gt_positions.items()} if point is not None else {})
                    nearest, second_nearest, nearest_margin = nearest_evidence(point, gt_positions)
                    d_state_a = distances.get(key_a)
                    d_state_b = distances.get(key_b)
                    gt_ab = (float(np.linalg.norm(np.asarray(gt_positions[key_a])
                                                  - np.asarray(gt_positions[key_b])))
                             if key_a in gt_positions and key_b in gt_positions else None)
                    detected_flag = bool(detected[position][slot])
                    row_info = {"assigned_cost_m": None, "row_best_gt": None,
                                "row_best_cost_m": None, "row_second_best_cost_m": None,
                                "row_cost_margin_m": None, "assigned_is_row_best": None,
                                "row_best_taken_by_other_slot": None}
                    if offset == 0 and recon is not None and slot in recon["segments"]:
                        row = recon["slots"].index(slot)
                        row_costs = recon["small"][row]
                        finite = np.flatnonzero(row_costs < 1e9)
                        order = finite[np.argsort(row_costs[finite], kind="stable")]
                        if len(order):
                            best_column = int(order[0])
                            best_gt = int(recon["vehicles"][best_column])
                            assigned_cost = (float(row_costs[recon["vehicles"].index(assigned)])
                                             if assigned is not None and assigned in recon["vehicles"]
                                             else None)
                            second_cost = float(row_costs[int(order[1])]) if len(order) > 1 else None
                            taken_by = [other for other, key in recon["assignment"].items()
                                        if key == best_gt and other != slot]
                            row_info = {
                                "assigned_cost_m": assigned_cost, "row_best_gt": best_gt,
                                "row_best_cost_m": float(row_costs[best_column]),
                                "row_second_best_cost_m": second_cost,
                                "row_cost_margin_m": ((second_cost - float(row_costs[best_column]))
                                                      if second_cost is not None else None),
                                "assigned_is_row_best": bool(assigned == best_gt),
                                "row_best_taken_by_other_slot": bool(taken_by)}
                    window_rows[offset] = {
                        "switch_id": switch_id, "episode": episode_index,
                        "episode_id": entry["episode_id"], "density": density, "snr": snr,
                        "slot": slot, "window_offset": offset, "origin_ms": origin_ms,
                        "state_x": float(point[0]) if point is not None else None,
                        "state_y": float(point[1]) if point is not None else None,
                        "assigned_gt": assigned, "nearest_gt": nearest,
                        "second_nearest_gt": second_nearest, "nearest_margin_m": nearest_margin,
                        "d_state_A": d_state_a, "d_state_B": d_state_b, "gt_AB_distance": gt_ab,
                        "detected": detected_flag, "coast": not detected_flag,
                        **row_info}

                # persistence / bounce inside the same no-gap chain
                chain = chains[slot][chain_of[slot][window]]
                chain_position = chain.index(window)
                run = 1
                stop_reason = "segment_end"
                for later in chain[chain_position + 1:]:
                    later_key = maps[later].get(slot)
                    if later_key is None:
                        stop_reason = "unassigned"
                        break
                    if later_key != key_b:
                        stop_reason = "identity_changed"
                        break
                    run += 1
                exact_return = None
                for step in (1, 2, 3):
                    later = window + step
                    if later >= n_windows or chain_of[slot].get(later) != chain_of[slot][window]:
                        break
                    if maps[later].get(slot) == key_a:
                        exact_return = step
                        break
                nearest_r = None
                jitter_bounce_strict = False
                if exact_return is not None:
                    recon_r = recon_at(origins[window + exact_return])
                    point_r = (recon_r["states"][19, slot, :2].astype(np.float64)
                               if recon_r is not None and slot in recon_r["segments"] else None)
                    nearest_r, _, margin_r = nearest_evidence(point_r, gt_at(origins[window + exact_return]))
                    margins = (window_rows[-1]["nearest_margin_m"], window_rows[0]["nearest_margin_m"],
                               margin_r)
                    jitter_bounce_strict = bool(
                        nearest_r is not None and window_rows[-1]["nearest_gt"] == key_a
                        and window_rows[0]["nearest_gt"] == key_a and nearest_r == key_a
                        and all(value is not None and value >= MARGIN_M for value in margins))

                row_switch = window_rows[0]
                row_before = window_rows[-1]
                crowded = bool(
                    (row_switch["gt_AB_distance"] is not None
                     and row_switch["gt_AB_distance"] <= CROWD_GT_M)
                    or (row_switch["d_state_A"] is not None and row_switch["d_state_B"] is not None
                        and abs(row_switch["d_state_A"] - row_switch["d_state_B"]) < MARGIN_M))
                competition = bool(
                    row_switch["assigned_is_row_best"] is False
                    and row_switch["assigned_cost_m"] is not None
                    and row_switch["row_best_cost_m"] is not None
                    and row_switch["row_best_taken_by_other_slot"]
                    and (row_switch["assigned_cost_m"] - row_switch["row_best_cost_m"]) >= MARGIN_M)
                jitter = bool(
                    (not crowded)
                    and ((row_before["nearest_gt"] == key_a == row_switch["nearest_gt"]
                          and row_before["nearest_margin_m"] is not None
                          and row_before["nearest_margin_m"] >= MARGIN_M
                          and row_switch["nearest_margin_m"] is not None
                          and row_switch["nearest_margin_m"] >= MARGIN_M)
                         or jitter_bounce_strict))
                handoff = bool(
                    (not crowded) and (not competition)
                    and row_before["nearest_gt"] == key_a and row_switch["nearest_gt"] == key_b
                    and row_before["d_state_A"] is not None and row_before["d_state_B"] is not None
                    and row_before["d_state_A"] + MARGIN_M <= row_before["d_state_B"]
                    and row_switch["d_state_A"] is not None and row_switch["d_state_B"] is not None
                    and row_switch["d_state_B"] + MARGIN_M <= row_switch["d_state_A"]
                    and run >= 3)
                if crowded:
                    cause = "crowded_ambiguous"
                elif competition:
                    cause = "global_assignment_competition"
                elif jitter:
                    cause = "matcher_jitter_clear"
                elif handoff:
                    cause = "true_tracker_handoff_clear"
                else:
                    cause = "unresolved"

                flags = (("crowded" if crowded else ""), ("competition" if competition else ""),
                         ("jitter" if jitter else ""), ("handoff" if handoff else ""))
                criteria_overlap["+".join(part for part in flags if part) or "none"] += 1
                cause_counts[cause] += 1
                cause_by_density[density][cause] += 1
                cause_by_snr[snr_key(snr)][cause] += 1
                cause_by_dt[entry["detection_transition"]][cause] += 1
                persistence_runs[run] += 1
                if exact_return is not None:
                    bounce_exact[exact_return] += 1

                switch_fields = {
                    "gt_before": key_a, "gt_after": key_b,
                    "detected_before": entry["detected_before"],
                    "detected_after": entry["detected_after"],
                    "detection_transition": entry["detection_transition"],
                    "crowded_criterion": crowded, "competition_criterion": competition,
                    "jitter_criterion": jitter, "handoff_criterion": handoff, "cause": cause,
                    "persistence_b_origins": run, "persistent_ge_3": run >= 3,
                    "persistent_to_segment_end": stop_reason == "segment_end",
                    "bounce_within_1": exact_return == 1,
                    "bounce_within_2": exact_return is not None and exact_return <= 2,
                    "bounce_within_3": exact_return is not None and exact_return <= 3,
                    "nearest_p": window_rows[-1]["nearest_gt"],
                    "nearest_c": row_switch["nearest_gt"], "nearest_r": nearest_r,
                    "jitter_bounce_strict": jitter_bounce_strict}
                # cost/row columns stay window-level (populated at window_offset == 0 only)
                for offset in sorted(window_rows):
                    merged = dict(switch_fields)
                    merged.update(window_rows[offset])
                    csv_rows.append(merged)

                # policy simulation sets (deduplicated) and handoff impact accounting
                for policy, radius in (("P1_exact", 0), ("P2_pm1", 1), ("P3_pm2", 2)):
                    for offset in range(-radius, radius + 1):
                        position = window + offset
                        if 0 <= position < n_windows:
                            policy_sets[policy].add((episode_index, sample_indices[position],
                                                     slot, snr))
                if cause == "true_tracker_handoff_clear":
                    handoff_segments.add((episode_index, snr, slot, chain[0]))
                    handoff_by_density[density] += 1
                    handoff_by_snr[snr_key(snr)] += 1
                    handoff_by_dt[entry["detection_transition"]] += 1
                    for later in chain:
                        cell = (episode_index, sample_indices[later], slot, snr)
                        if label_valid[snr][sample_indices[later], :, slot].any():
                            handoff_labelled_cells.add(cell)
                    for offset in range(-2, 3):
                        position = window + offset
                        if 0 <= position < n_windows:
                            cell = (episode_index, sample_indices[position], slot, snr)
                            if label_valid[snr][sample_indices[position], :, slot].any():
                                handoff_pm2_cells.add(cell)

            print(json.dumps({"phase": "forensics", "episode": episode_index, "snr": snr,
                              "switches_so_far": sum(cause_counts.values()),
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)

    # ------------------------------------------------------------------
    # aggregation and policy impact
    # ------------------------------------------------------------------
    def labelled(cell):
        episode_index, sample_index, slot, snr = cell
        return bool(label_valid[snr][sample_index, :, slot].any())

    policy_impact = {
        "stage": "F01E-IDENTITY-AUDIT-10", "note": "simulation only; labels were not modified",
        "baseline": baseline,
        "policies": {}}
    for policy, description in (("P1_exact", "mask the switch-origin slot label"),
                                ("P2_pm1", "mask switch origin +/- 1 origin for that slot"),
                                ("P3_pm2", "mask switch origin +/- 2 origins for that slot")):
        masked = {cell for cell in policy_sets[policy] if labelled(cell)}
        masked_by_snr = Counter(snr_key(cell[3]) for cell in masked)
        masked_by_density = Counter(density_bucket(source.episodes[cell[0]]) for cell in masked)
        remaining = baseline["labelled_slots"] - len(masked)
        policy_impact["policies"][policy] = {
            "description": description,
            "masked_labelled_slots": len(masked),
            "remaining_labelled_slots": remaining,
            "coverage_ratio": remaining / max(baseline["eligible_slots"], 1),
            "retention_ratio": remaining / max(baseline["labelled_slots"], 1),
            "masked_by_snr": {key: masked_by_snr.get(key, 0) for key in map(snr_key, SNR_LIST)},
            "masked_by_density": {key: masked_by_density.get(key, 0)
                                  for key in ("low", "mid", "high")}}

    transitions_by_snr = {tag: transitions[f"snr_{tag}"] for tag in map(snr_key, SNR_LIST)}
    switches_by_snr = Counter(snr_key(entry["snr"]) for entry in switches)
    transitions_by_density = {name: transitions[f"density_{name}"]
                              for name in ("low", "mid", "high")}
    switches_by_density = Counter(entry["density"] for entry in switches)
    by_density = {}
    for name in ("low", "mid", "high"):
        bucket_causes = {cause: cause_by_density[name].get(cause, 0) for cause in CAUSES}
        bucket_switches = switches_by_density[name]
        by_density[name] = {
            "transitions": transitions_by_density[name],
            "switches": bucket_switches,
            "switch_rate": bucket_switches / max(transitions_by_density[name], 1),
            "cause": bucket_causes,
            "cause_fractions": {cause: count / max(bucket_switches, 1)
                                for cause, count in bucket_causes.items()},
            "true_tracker_handoff_clear": handoff_by_density.get(name, 0)}
    by_snr = {}
    for tag in map(snr_key, SNR_LIST):
        bucket_causes = {cause: cause_by_snr[tag].get(cause, 0) for cause in CAUSES}
        bucket_switches = switches_by_snr.get(tag, 0)
        by_snr[tag] = {
            "transitions": transitions_by_snr[tag],
            "switches": bucket_switches,
            "switch_rate": bucket_switches / max(transitions_by_snr[tag], 1),
            "cause": bucket_causes,
            "cause_fractions": {cause: count / max(bucket_switches, 1)
                                for cause, count in bucket_causes.items()},
            "true_tracker_handoff_clear": handoff_by_snr.get(tag, 0)}
    by_dt = {}
    for name in ("detected_to_detected", "detected_to_coast", "coast_to_detected", "coast_to_coast"):
        total = transitions[f"dt_{name}"]
        bucket_causes = {cause: cause_by_dt[name].get(cause, 0) for cause in CAUSES}
        bucket_switches = sum(bucket_causes.values())
        by_dt[name] = {
            "transitions": total,
            "switches": bucket_switches,
            "switch_rate": bucket_switches / max(total, 1),
            "cause": bucket_causes,
            "cause_fractions": {cause: count / max(bucket_switches, 1)
                                for cause, count in bucket_causes.items()},
            "true_tracker_handoff_clear": handoff_by_dt.get(name, 0)}

    dominance = {
        "definition": ("chains of consecutive origin-eligible windows of one (episode, SNR, slot) "
                       "joined by no-gap same-segment edges"),
        "segments_total": chain_dominance["segments_total"],
        "segments_len_ge2": chain_dominance["segments_len_ge2"],
        "single_window_segments": chain_dominance["single_window_segments"],
        "labelled_segments": chain_dominance["labelled_segments"],
        "segments_with_no_assigned_label": chain_dominance["segments_with_no_assigned_label"],
        "dominant_ge_0p90": chain_dominance["dominant_ge_0p90"],
        "dominant_ge_0p95": chain_dominance["dominant_ge_0p95"],
        "no_clear_dominant": chain_dominance["no_clear_dominant"],
        "no_clear_dominant_threshold": 0.5,
        "dominant_fraction_median_len_ge2": (float(np.median(dominant_fractions))
                                             if dominant_fractions else None)}

    handoff_events = cause_counts["true_tracker_handoff_clear"]
    handoff_fraction = len(handoff_labelled_cells) / max(baseline["labelled_slots"], 1)

    cause_fractions = {cause: cause_counts.get(cause, 0) / max(switch_count, 1) for cause in CAUSES}
    summary = {
        "stage": "F01E-IDENTITY-AUDIT-10", "dataset": "F01E", "split": "train",
        "f01d_used": False, "script": "code/07_shared_frontend/freeze10_identity_forensics.py",
        "expected_switches": EXPECTED_SWITCHES, "reproduced_switches": switch_count,
        "expected_transitions": EXPECTED_TRANSITIONS,
        "reproduced_transitions": transitions["all"], "switch_rate": switch_rate,
        "reproduction_pass": True,
        "classification_priority": list(CAUSES),
        "definitions": {
            "same_segment_transition": ("consecutive prediction origins of one (episode, SNR, slot) "
                                        "with the slot origin-eligible at the later origin and a "
                                        "contiguous alive run covering the whole gap "
                                        "(no track_exists=0 gap)"),
            "crowded_ambiguous": ("GT A-B distance <= 1.0 m at the switch origin or "
                                  "abs(d_state_A - d_state_B) < 0.5 m at the switch origin"),
            "global_assignment_competition": ("assigned GT != row-best GT, assigned cost - row-best "
                                              "cost >= 0.5 m and the row-best GT is assigned to "
                                              "another slot by the same Hungarian solution"),
            "matcher_jitter_clear": ("clear nearest-GT identity stays A (margins >= 0.5 m at "
                                     "switch-before and switch origins) while the assignment moves "
                                     "away from A, or an A->B->A return within <=2 origins with the "
                                     "same clear nearest-A evidence at all three origins"),
            "true_tracker_handoff_clear": ("d_A + 0.5 <= d_B before, d_B + 0.5 <= d_A at the switch "
                                           "origin, nearest identity A->B, new identity persists "
                                           ">= 3 origins, and neither crowded nor competition"),
            "persistence_b_origins": ("consecutive same-chain origins assigned to B starting at the "
                                      "switch origin"),
            "bounce_within_k": ("the first return of the assignment to A happens at the k-th "
                                "following same-chain origin"),
            "density": "low: <=3 source vehicles, mid: 4-6, high: >=7",
            "cause_priority": list(CAUSES)},
        "cause": {cause: cause_counts.get(cause, 0) for cause in CAUSES},
        "cause_fractions": cause_fractions,
        "criteria_overlap": dict(criteria_overlap),
        "bounce": {
            "within_1": sum(count for step, count in bounce_exact.items() if step <= 1),
            "within_2": sum(count for step, count in bounce_exact.items() if step <= 2),
            "within_3": sum(count for step, count in bounce_exact.items() if step <= 3),
            "exact": {str(step): bounce_exact.get(step, 0) for step in (1, 2, 3)}},
        "persistence": {
            "persistent_new_identity_ge_3_origins": sum(count for run, count in persistence_runs.items()
                                                        if run >= 3),
            "persistent_to_segment_end": sum(1 for row in csv_rows
                                             if row["window_offset"] == 0
                                             and row["persistent_to_segment_end"]),
            "run_length_histogram": {str(run): persistence_runs[run]
                                     for run in sorted(persistence_runs)},
            "run_length_median": (float(np.median([run for run, count in persistence_runs.items()
                                                   for _ in range(count)]))
                                  if persistence_runs else None)},
        "by_density": by_density,
        "by_snr": by_snr,
        "by_detection_transition": by_dt,
        "segment_dominance": dominance,
        "true_tracker_handoff": {
            "events": handoff_events,
            "unique_segments": len(handoff_segments),
            "affected_labelled_slots_in_segments": len(handoff_labelled_cells),
            "affected_labelled_slots_in_pm2_windows": len(handoff_pm2_cells),
            "fraction_of_all_labelled_slots": handoff_fraction,
            "by_density": {name: handoff_by_density.get(name, 0)
                           for name in ("low", "mid", "high")},
            "by_snr": {tag: handoff_by_snr.get(tag, 0) for tag in map(snr_key, SNR_LIST)},
            "by_detection_transition": {name: handoff_by_dt.get(name, 0)
                                        for name in by_dt}},
        "true_tracker_handoff_fraction_all_labelled_slots": handoff_fraction,
        "baseline_labels": baseline,
        "matcher_reproduction": {
            "assignment_mismatch_windows": matcher_mismatch["assignment"],
            "cost_or_common_mismatch": matcher_cost_mismatch},
        "hard_checks": {
            "reproduced_transitions": transitions["all"] == EXPECTED_TRANSITIONS,
            "reproduced_switches": switch_count == EXPECTED_SWITCHES,
            "switch_rate": abs(switch_rate - EXPECTED_SWITCH_RATE) < 1e-9,
            "baseline_labelled_slots": baseline["labelled_slots"] == EXPECTED_LABELLED_SLOTS,
            "baseline_valid_slot_labels": (baseline["valid_slot_labels"]
                                           == EXPECTED_VALID_SLOT_LABELS),
            "baseline_eligible_slots": baseline["eligible_slots"] == EXPECTED_ELIGIBLE_SLOTS,
            "cause_sum": sum(cause_counts.values()) == switch_count,
            "switch_ids_consistent": switch_counter == switch_count,
            "matcher_assignment_reproduced": matcher_mismatch["assignment"] == 0,
            "matcher_cost_reproduced": matcher_cost_mismatch == 0,
            "f01d_free": not False},
        "next_action": "REVIEW_REQUIRED"}
    summary["overall"] = "PASS" if all(summary["hard_checks"].values()) else "FAIL"

    after_manifest = manifest_set()
    integrity = {
        "stage": "F01E-IDENTITY-AUDIT-10",
        "data_f01e_read_only": True, "production_sensing_modified": False, "f01d_used": False,
        "before": before_manifest, "after": after_manifest,
        "unchanged": {name: before_manifest[name] == after_manifest[name]
                      for name in INTEGRITY_SUBTREES},
        "baseline_labels": baseline,
        "matcher_reproduction": summary["matcher_reproduction"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__},
        "runtime_seconds": round(time.time() - started, 1)}

    with (OUT / "switch_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    with (OUT / "policy_impact.json").open("w", encoding="utf-8") as handle:
        json.dump(policy_impact, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (OUT / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    print(json.dumps({"overall": summary["overall"], "switches": switch_count,
                      "switch_rate": switch_rate,
                      "cause": summary["cause"], "bounce": summary["bounce"],
                      "policy": {name: entry["remaining_labelled_slots"]
                                 for name, entry in policy_impact["policies"].items()},
                      "declared_next_action": summary["next_action"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()