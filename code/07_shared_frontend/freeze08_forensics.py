"""F01E-AUDIT-08 forensics (CPU-only): contract introspection, tail attribution, identity audits.

Train only for performance analysis. F01-D is never read. The current alignment is reproduced
from the F01-E metadata (`slot_alignment_by_snr`) written by the generator; no matcher change.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/audit_08"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
VEHICLE_VISIBLE = (10.0, 300.0, 70.0)
TAIL_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
JUMP_LIMIT_M = 10.0
STICKY_JUMP_M = 20.0
FRAGMENTATION_GAP = 10
NFRAMES = 199
CANDIDATE_CAP = 80


def percentile(values, fraction):
    return float(np.percentile(values, fraction * 100)) if len(values) else None


def contract_introspection() -> dict:
    source = (HERE / "freeze07_generate_f01e.py").read_text()
    lines = source.splitlines()

    def find(pattern: str) -> list:
        return [index + 1 for index, line in enumerate(lines) if pattern in line]

    facts = {
        "generator_path": "code/07_shared_frontend/freeze07_generate_f01e.py",
        "generator_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_path_pattern": "inputs/{split}_snr_{s}.npz",
        "label_path_pattern": "labels/{split}_snr_{s}.npz",
        "reference_label_write_lines": find("labels\" / f\"{split}.npz\""),
        "alignment_function_lines": find("def align_window"),
        "hungarian_lines": find("linear_sum_assignment"),
        "gate_lines": find("gate_m"),
        "common_frames_rule_lines": find("common[slot, vehicle_index] >= 3"),
        "history_window_lines": find("np.arange(-19, 1"),
        "sequence_frames": generator.SEQUENCE_FRAMES,
        "sequence_grid": "episode start + (1..199)*100 ms",
        "snr_name_convention": generator.snr_name(-5),
        "slot_track_key_written": False,
        "assigned_gt_saved": True,
        "assigned_gt_storage": "metadata/{split}.json samples[].slot_alignment_by_snr (per window and SNR)",
    }
    answers = {
        "Q1_input_paths_per_snr": {str(snr): f"data/f01e/inputs/{{split}}_{generator.snr_name(snr)}.npz"
                                   for snr in SNR_LIST},
        "Q2_label_paths_per_snr": {str(snr): f"data/f01e/labels/{{split}}_{generator.snr_name(snr)}.npz"
                                   for snr in SNR_LIST},
        "Q3_generic_labels": "labels/{split}.npz is written by freeze07_generate_f01e.pack from the "
                              "max-SNR reference alignment",
        "Q4_reference_snr": max(SNR_LIST),
        "Q5_mapping_granularity": "per window (20 history frames) and per SNR; Hungarian over slots x "
                                  "source vehicles",
        "Q6_cost": "mean Euclidean distance over frames where the slot is alive and the GT sample exists "
                   "(>=3 common frames)",
        "Q7_hungarian": True,
        "Q8_gate_m": 5.0,
        "Q9_identity_changes_within_track": "assignment is recomputed independently per window and per SNR; "
                                            "nothing enforces temporal identity across windows, and true "
                                            "track_key is not persisted",
        "Q10_future_labels": "aligned_labels() copies the aligned source vehicle's raw future xy "
                             "(origin+1..20 frames, exact samples) into the slot row",
        "Q11_slot_track_key_in_cache": False,
        "Q12_assigned_gt_saved": True,
        "Q13_assignment_reproducible": True,
        "Q13_reproduction_route": "read metadata/{split}.json samples[].slot_alignment_by_snr",
    }
    required = ("alignment_function_lines", "hungarian_lines", "gate_lines", "common_frames_rule_lines")
    facts["introspection_complete"] = all(facts[key] for key in required)
    return {"facts": facts, "answers": answers, "formal_dataset": "F01E", "f01d_used_for_audit": False}


class EpisodeGT:
    """Raw A01 source tracks indexed on the frozen 199-frame grid with per-frame visibility."""

    def __init__(self, source, episode, stations, boresights, height):
        self.keys = [int(key) for key in episode["source_keys"]]
        self.times = int(episode["start_ms"]) + np.arange(1, NFRAMES + 1, dtype=np.int64) * 100
        self.xy, self.vel, self.exact = {}, {}, {}
        for key in self.keys:
            track = source.tracks[key]
            positions = np.searchsorted(track["time_ms"], self.times)
            in_range = positions < len(track)
            safe = np.minimum(positions, max(0, len(track) - 1))
            exact = in_range & (track["time_ms"][safe] == self.times)
            xy = np.stack([track["x"][safe], track["y"][safe]], axis=-1).astype(np.float64)
            vel = np.stack([track["vx"][safe], track["vy"][safe]], axis=-1).astype(np.float64)
            xy[~exact] = np.nan
            self.xy[key], self.vel[key], self.exact[key] = xy, vel, exact
        self.samples = []
        self.in_fov = []
        for frame in range(NFRAMES):
            samples, fov = {}, {}
            for key in self.keys:
                if not self.exact[key][frame]:
                    continue
                point = self.xy[key][frame]
                samples[key] = point
                for station, boresight in zip(stations, boresights):
                    delta = np.asarray(station, dtype=float) - point
                    slant = float(np.hypot(np.linalg.norm(delta), height))
                    bearing = float(np.arctan2(-delta[1], -delta[0]) - boresight + np.pi) \
                        % (2 * np.pi) - np.pi
                    if VEHICLE_VISIBLE[0] <= slant <= VEHICLE_VISIBLE[1] \
                            and abs(bearing) <= np.radians(VEHICLE_VISIBLE[2]):
                        fov[key] = True
                        break
            self.samples.append(samples)
            self.in_fov.append(fov)


def classify(assigned_error, nearest_key, assigned_key, nearest_error, second_error, margin, count):
    if nearest_error > 1.0:
        return "true_tracking_tail"
    if nearest_key != assigned_key and nearest_error < 0.5 and margin >= 0.5:
        return "alignment_suspect"
    if nearest_error <= 1.0 and (margin < 0.5 or (count > 1 and second_error <= 1.0)):
        return "ambiguous_close_vehicle"
    return "unresolved"


def main() -> None:
    from frontend.echo_source import SourceEpisodes

    import _common

    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    stations = geometry["stations"]
    boresights = geometry["boresights"]
    height = float(geometry["height_difference_m"])
    source = SourceEpisodes()
    metadata = json.loads((DATA / "metadata" / "train.json").read_text())
    samples = metadata["samples"]
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    labels = {}
    for snr in SNR_LIST:
        with np.load(DATA / "labels" / f"train_{generator.snr_name(snr)}.npz", allow_pickle=False) as payload:
            labels[snr] = (payload["future_position"], payload["label_valid"])

    contract = contract_introspection()
    contract["train_windows"] = len(samples)
    contract["train_episodes"] = len(by_episode)

    tail = Counter()
    cause = defaultdict(Counter)
    tail_candidates = defaultdict(list)
    state_counts = defaultdict(Counter)
    window_class = defaultdict(Counter)
    window_p90 = {"assigned": defaultdict(list), "nearest": defaultdict(list)}
    label_stats = defaultdict(Counter)
    origin_identity = defaultdict(Counter)
    label_mismatch_examples = []
    switch_cases = []
    track_stats = Counter()
    fragmentation = defaultdict(Counter)
    transitions = defaultdict(int)
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        gt = EpisodeGT(source, episode, stations, boresights, height)
        density = "low" if len(episode["source_keys"]) <= 3 else ("mid" if len(episode["source_keys"]) <= 6
                                                                  else "high")
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        origins = sorted(int(sample["origin_ms"]) for sample in episode_samples)
        frame_origin = np.empty(NFRAMES, dtype=np.int64)
        for frame in range(NFRAMES):
            time = gt.times[frame]
            frame_origin[frame] = min(origins, key=lambda origin: (abs(origin - time), origin))
        canonical = {snr: {} for snr in SNR_LIST}
        for sample in episode_samples:
            for snr in SNR_LIST:
                canonical[snr][int(sample["origin_ms"])] = {
                    int(slot): int(key) for slot, key in sample["slot_alignment_by_snr"][str(snr)].items()}
        segments = {snr: defaultdict(list) for snr in SNR_LIST}
        for snr in SNR_LIST:
            exists_all = sequences[snr]["track_exists"]
            for slot in range(8):
                start = None
                for frame in range(NFRAMES + 1):
                    alive = frame < NFRAMES and bool(exists_all[frame, slot])
                    if alive and start is None:
                        start = frame
                    if not alive and start is not None:
                        segments[snr][slot].append((start, frame - 1))
                        start = None
        for sample in episode_samples:
            origin = int(sample["origin_ms"])
            window_index = sample["sample_index"]
            indices = generator.history_indices(sequences[SNR_LIST[0]], origin)
            for snr in SNR_LIST:
                sequence = sequences[snr]
                exists = sequence["track_exists"][indices]
                detected = sequence["detected"][indices]
                states = sequence["state_hat"][indices]
                mapping = canonical[snr][origin]
                window_errors, window_nearest = [], []
                violations = 0
                for slot in range(8):
                    for frame in range(1, 20):
                        if exists[frame - 1, slot] and exists[frame, slot]:
                            if float(np.linalg.norm(states[frame, slot, :2]
                                                    - states[frame - 1, slot, :2])) > STICKY_JUMP_M:
                                violations += 1
                for slot in range(8):
                    for frame in range(20):
                        if not exists[frame, slot]:
                            continue
                        state_counts["total"]["valid"] += 1
                        state_counts[f"snr_{snr}"]["valid"] += 1
                        state_counts[f"density_{density}"]["valid"] += 1
                        point = states[frame, slot, :2].astype(np.float64)
                        frame_index = int(indices[frame])
                        samples = gt.samples[frame_index]
                        assigned = mapping.get(slot)
                        bucket = "coast" if not detected[frame, slot] else "detected"
                        state_counts[bucket]["valid"] += 1
                        if assigned is None:
                            state_counts["missing"]["assigned"] += 1
                            state_counts["unassigned"]["valid"] += 1
                            if samples:
                                distances_unassigned = [float(np.linalg.norm(point - xy))
                                                        for xy in samples.values()]
                                nearest_unassigned = min(distances_unassigned)
                                for threshold in TAIL_THRESHOLDS:
                                    if nearest_unassigned > threshold:
                                        state_counts["unassigned"][f"gt_{threshold}"] += 1
                            continue
                        if assigned not in samples:
                            state_counts["missing"]["gt_sample"] += 1
                            continue
                        state_counts["total"]["comparable"] += 1
                        state_counts[f"snr_{snr}"]["comparable"] += 1
                        state_counts[f"density_{density}"]["comparable"] += 1
                        state_counts[bucket]["comparable"] += 1
                        if assigned not in gt.in_fov[frame_index]:
                            state_counts["fov"]["assigned_out_of_fov"] += 1
                        distances = {key: float(np.linalg.norm(point - xy))
                                     for key, xy in samples.items()}
                        assigned_error = distances[assigned]
                        ordered = sorted(distances.items(), key=lambda item: item[1])
                        nearest_key, nearest_error = ordered[0]
                        second_error = ordered[1][1] if len(ordered) > 1 else float("inf")
                        margin = second_error - nearest_error
                        window_errors.append(assigned_error)
                        window_nearest.append(nearest_error)
                        for threshold in TAIL_THRESHOLDS:
                            if assigned_error > threshold:
                                tail[f"gt_{threshold}"] += 1
                                state_counts[f"snr_{snr}"][f"gt_{threshold}"] += 1
                                state_counts[f"density_{density}"][f"gt_{threshold}"] += 1
                                state_counts[bucket][f"gt_{threshold}"] += 1
                        if assigned_error > 1.0:
                            category = classify(assigned_error, nearest_key, assigned, nearest_error,
                                                second_error, margin, len(distances))
                            slot_exists = exists[:, slot]
                            rebirth = bool(np.any((~slot_exists[:-1]) & slot_exists[1:]))
                            cause["all"][category] += 1
                            cause[bucket][category] += 1
                            cause[f"snr_{snr}"][category] += 1
                            cause[f"density_{density}"][category] += 1
                            if category == "alignment_suspect":
                                cause["subtype"]["alignment_suspect_slot_rebirth_in_window"] += int(rebirth)
                                cause["subtype"]["alignment_suspect_total"] += 1
                            candidates = tail_candidates[(str(snr), density, bucket, category)]
                            if len(candidates) < CANDIDATE_CAP:
                                candidates.append({
                                    "snr_db": snr, "episode": episode_index, "frame": int(indices[frame]),
                                    "origin_ms": origin, "slot": slot, "detected": bucket == "detected",
                                    "density": density, "assigned_gt_id": int(assigned),
                                    "nearest_gt_id": int(nearest_key),
                                    "assigned_error_m": round(assigned_error, 4),
                                    "nearest_error_m": round(nearest_error, 4),
                                    "second_nearest_error_m": round(second_error, 4) if np.isfinite(second_error) else None,
                                    "ambiguity_margin_m": round(margin, 4) if np.isfinite(margin) else None,
                                    "slot_rebirth_in_window": rebirth,
                                    "tail_cause": category, "visible_gt_count": len(distances)})
                assigned_p90 = percentile(window_errors, 0.90)
                nearest_p90 = percentile(window_nearest, 0.90)
                window_p90["assigned"][str(snr)].append(assigned_p90 or 0.0)
                window_p90["nearest"][str(snr)].append(nearest_p90 or 0.0)
                coast_ratio = float((exists & ~detected).sum() / max(exists.sum(), 1))
                if violations:
                    window_class[str(snr)]["sticky_slot_violation"] += 1
                elif (assigned_p90 or 0.0) > 1.0 and coast_ratio > 0.30:
                    window_class[str(snr)]["both"] += 1
                elif (assigned_p90 or 0.0) > 1.0:
                    window_class[str(snr)]["position_only"] += 1
                elif coast_ratio > 0.30:
                    window_class[str(snr)]["coast_only"] += 1
                else:
                    window_class[str(snr)]["neither"] += 1
                future_all, valid_all = labels[snr]
                future, valid_row = future_all[window_index], valid_all[window_index]
                for slot, key in mapping.items():
                    slot_valid = valid_row[:, slot]
                    if not slot_valid.any():
                        continue
                    label_stats[str(snr)]["valid_slot_labels"] += 1
                    future_times = origin + np.arange(1, 21, dtype=np.int64) * 100
                    positions = np.searchsorted(gt.times, future_times)
                    in_range = positions < NFRAMES
                    safe = np.minimum(positions, NFRAMES - 1)
                    expected = gt.xy[key][safe]
                    exact = in_range & (gt.times[safe] == future_times) & gt.exact[key][safe]
                    slot_future = future[:, slot]
                    if not (np.array_equal(exact, slot_valid)
                            and np.all(np.abs(slot_future[exact] - expected[exact]) < 5e-3)):
                        label_stats[str(snr)]["identity_mismatch"] += 1
                        if len(label_mismatch_examples) < 10:
                            label_mismatch_examples.append({"snr": snr, "episode": episode_index,
                                                            "window": window_index, "slot": slot,
                                                            "assigned_key": key})
                    label_stats[str(snr)]["label_windows"] += 1
                    alive_at_origin = bool(exists[19, slot])
                    origin_identity[str(snr)]["valid_slot_labels"] += 1
                    origin_identity[f"density_{density}"]["valid_slot_labels"] += 1
                    if not alive_at_origin:
                        origin_identity[str(snr)]["labels_on_dead_slot"] += 1
                    else:
                        origin_identity[str(snr)]["alive_at_origin"] += 1
                        point = states[19, slot, :2].astype(np.float64)
                        if gt.exact[key][indices[19]]:
                            error = float(np.linalg.norm(point - gt.xy[key][indices[19]]))
                            if error > 1.0:
                                origin_identity[str(snr)]["inconsistent_gt_1m"] += 1
                                origin_identity[f"density_{density}"]["inconsistent_gt_1m"] += 1
                            if error > 0.5:
                                origin_identity[str(snr)]["inconsistent_gt_0p5m"] += 1
                        else:
                            origin_identity[str(snr)]["gt_missing_at_origin"] += 1
                    origin_position = gt.xy[key][indices[19]]
                    if np.isfinite(origin_position).all() and slot_valid[0]:
                        if float(np.linalg.norm(slot_future[0] - origin_position)) > JUMP_LIMIT_M:
                            label_stats[str(snr)]["label_jump_suspect"] += 1
                    for frame in range(1, 20):
                        if slot_valid[frame] and slot_valid[frame - 1]:
                            if float(np.linalg.norm(slot_future[frame] - slot_future[frame - 1])) \
                                    > JUMP_LIMIT_M:
                                label_stats[str(snr)]["label_jump_suspect"] += 1
        for snr in SNR_LIST:
            assignment = np.full((NFRAMES, 8), -1, dtype=np.int64)
            for frame in range(NFRAMES):
                mapping = canonical[snr].get(int(frame_origin[frame]))
                if mapping:
                    for slot, key in mapping.items():
                        assignment[frame, slot] = key
            vehicle_segments = defaultdict(list)
            for slot, runs in segments[snr].items():
                for start, end in runs:
                    track_id = f"{episode_index}:{snr}:{slot}:{start}"
                    assigned_ids = [int(assignment[frame, slot]) for frame in range(start, end + 1)
                                    if assignment[frame, slot] >= 0]
                    if not assigned_ids:
                        continue
                    dominant = Counter(assigned_ids).most_common(1)[0][0]
                    vehicle_segments[dominant].append((start, end, slot, track_id))
                    switches = 0
                    previous = None
                    for frame in range(start, end + 1):
                        current = int(assignment[frame, slot])
                        if current < 0:
                            continue
                        if previous is not None and current != previous:
                            switches += 1
                            if len(switch_cases) < 40:
                                switch_cases.append({
                                    "episode": episode_index, "snr": snr, "slot": slot, "track": track_id,
                                    "frame_before": frame - 1, "frame_after": frame, "gt_before": previous,
                                    "gt_after": current, "density": density,
                                    "detected_after": bool(sequence["detected"][frame, slot]),
                                    "track_age": frame - start})
                        previous = current
                    track_stats["tracks_total"] += 1
                    track_stats["tracks_with_multiple_gt_ids"] += int(len(set(assigned_ids)) > 1)
                    track_stats["identity_switches"] += switches
                    transitions[str(snr)] += max(len(assigned_ids) - 1, 0)
            for key, runs in vehicle_segments.items():
                runs.sort()
                for (start_a, end_a, _, _), (start_b, end_b, _, _) in zip(runs, runs[1:]):
                    if 0 < start_b - end_a <= FRAGMENTATION_GAP:
                        fragmentation[str(snr)]["count"] += 1
                fragmentation[str(snr)]["instances"] += 1

    total_valid = state_counts["total"]["valid"]
    total_comparable = state_counts["total"]["comparable"]
    tail_summary = {f"gt_{threshold}m": {"count": tail[f"gt_{threshold}"],
                                         "fraction": tail[f"gt_{threshold}"] / max(total_comparable, 1)}
                    for threshold in TAIL_THRESHOLDS}
    tail_gt1 = tail["gt_1.0"]
    attribution = {name: {"count_of_tail": cause["all"][name],
                          "fraction_of_tail": cause["all"][name] / max(tail_gt1, 1),
                          "fraction_of_all_valid": cause["all"][name] / max(total_comparable, 1)}
                   for name in ("alignment_suspect", "true_tracking_tail", "ambiguous_close_vehicle",
                                "unresolved")}
    by_detection = {}
    for bucket in ("detected", "coast"):
        counts = cause[bucket]
        by_detection[bucket] = {
            "alive_states": state_counts[bucket]["valid"],
            "valid_states": state_counts[bucket]["comparable"],
            "tail_gt_1": state_counts[bucket]["gt_1.0"],
            "tail_gt_1_fraction": state_counts[bucket]["gt_1.0"] / max(state_counts[bucket]["comparable"], 1),
            "cause_count": {name: counts[name] for name in ("alignment_suspect", "true_tracking_tail",
                                                            "ambiguous_close_vehicle", "unresolved")},
            "cause_fraction_of_tail": {name: counts[name] / max(sum(counts.values()), 1)
                                       for name in ("alignment_suspect", "true_tracking_tail",
                                                    "ambiguous_close_vehicle", "unresolved")}}
    by_snr = {}
    for snr in SNR_LIST:
        valid = state_counts[f"snr_{snr}"]["comparable"]
        counts = cause[f"snr_{snr}"]
        by_snr[str(snr)] = {
            "alive_states": state_counts[f"snr_{snr}"]["valid"],
            "valid_states": valid, "tail_gt_1": state_counts[f"snr_{snr}"]["gt_1.0"],
            "tail_gt_1_fraction": state_counts[f"snr_{snr}"]["gt_1.0"] / max(valid, 1),
            "cause_count": {name: counts[name] for name in ("alignment_suspect", "true_tracking_tail",
                                                            "ambiguous_close_vehicle", "unresolved")}}
    by_density = {}
    for density in ("low", "mid", "high"):
        valid = state_counts[f"density_{density}"]["comparable"]
        counts = cause[f"density_{density}"]
        by_density[density] = {
            "alive_states": state_counts[f"density_{density}"]["valid"],
            "valid_states": valid, "tail_gt_1": state_counts[f"density_{density}"]["gt_1.0"],
            "tail_gt_1_fraction": state_counts[f"density_{density}"]["gt_1.0"] / max(valid, 1),
            "cause_fraction_of_tail": {name: counts[name] / max(sum(counts.values()), 1)
                                       for name in ("alignment_suspect", "true_tracking_tail",
                                                    "ambiguous_close_vehicle", "unresolved")}}
    decomposition = {}
    for snr in SNR_LIST:
        counts = window_class[str(snr)]
        total = sum(counts.values()) or 1
        decomposition[str(snr)] = {key: counts[key] for key in ("position_only", "coast_only", "both",
                                                                "sticky_slot_violation", "neither")}
        decomposition[str(snr)].update({
            "windows": total, "problematic_fraction": (total - counts["neither"]) / total,
            "position_p90_assigned_median": percentile(window_p90["assigned"][str(snr)], 0.50),
            "position_p90_nearest_median": percentile(window_p90["nearest"][str(snr)], 0.50)})
    tail_attribution = {
        "dataset": "F01E", "split": "train", "snr_levels": SNR_LIST, "f01d_used": False,
        "alive_states_total": total_valid, "valid_states": total_comparable,
        "missing_alive_states": {"assigned_gt": state_counts["missing"]["assigned"],
                                 "gt_sample_unavailable": state_counts["missing"]["gt_sample"]},
        "unassigned_alive_states": {
            "alive_states": state_counts["unassigned"]["valid"],
            "fraction_of_alive": state_counts["unassigned"]["valid"] / max(total_valid, 1),
            "nearest_gt_error_fractions": {f"gt_{threshold}m": state_counts["unassigned"][f"gt_{threshold}"]
                                           / max(state_counts["unassigned"]["valid"], 1)
                                           for threshold in TAIL_THRESHOLDS},
            "note": "alive states whose slot has no current alignment entry; they carry no assigned identity "
                    "and are reported separately, not forced into the four tail causes"},
        "assigned_gt_out_of_simulator_fov_states": state_counts["fov"]["assigned_out_of_fov"],
        "tail_thresholds": tail_summary,
        "tail_gt_1m": {"count": tail_gt1,
                       "fraction_all_valid_states": tail_gt1 / max(total_comparable, 1),
                       "cause_fraction_of_tail": {name: value["fraction_of_tail"]
                                                  for name, value in attribution.items()}},
        "cause_detail": attribution, "by_detection_state": by_detection, "by_snr": by_snr,
        "by_density": by_density, "problematic_window_decomposition": decomposition,
        "alignment_suspect_subtypes": {
            "slot_rebirth_inside_window": cause["subtype"]["alignment_suspect_slot_rebirth_in_window"],
            "alignment_suspect_total": cause["subtype"]["alignment_suspect_total"],
            "rule": "a slot whose track_exists run starts inside the 20-frame window was reused by a new "
                    "track; the window matcher then cannot represent both tracks"},
        "notes": {"cause_rules": "true_tracking_tail: nearest>1; alignment_suspect: nearest!=assigned, "
                                 "nearest<0.5, margin>=0.5; ambiguous_close_vehicle: nearest<=1 and "
                                 "(margin<0.5 or another visible GT<=1); else unresolved",
                  "coast_is_a_state_dimension_not_a_cause": True}}
    switch_rate = track_stats["identity_switches"] / max(track_stats["tracks_total"], 1)
    transition_total = sum(transitions.values())
    label_identity = {
        "tracks_total": track_stats["tracks_total"],
        "tracks_with_multiple_gt_ids": track_stats["tracks_with_multiple_gt_ids"],
        "identity_switches": track_stats["identity_switches"],
        "identity_switch_rate": switch_rate,
        "identity_switch_rate_per_track": switch_rate,
        "identity_switch_rate_per_transition": track_stats["identity_switches"]
        / max(transition_total, 1),
        "track_key_available_in_cache": False,
        "track_identity_proxy": "maximal runs of consecutive track_exists=1 frames per (episode, snr, slot); "
                                "true track_key is not persisted in F01-E",
        "fragmentation": {str(snr): {"count": fragmentation[str(snr)]["count"],
                                     "instances": fragmentation[str(snr)]["instances"],
                                     "rate": fragmentation[str(snr)]["count"]
                                     / max(fragmentation[str(snr)]["instances"], 1),
                                     "gap_frames": FRAGMENTATION_GAP} for snr in SNR_LIST},
        "window_labels": {
            "label_windows": sum(label_stats[str(snr)]["label_windows"] for snr in SNR_LIST),
            "valid_slot_labels": sum(label_stats[str(snr)]["valid_slot_labels"] for snr in SNR_LIST),
            "history_future_identity_mismatch_count": sum(label_stats[str(snr)]["identity_mismatch"]
                                                          for snr in SNR_LIST),
            "history_future_identity_mismatch_fraction": sum(label_stats[str(snr)]["identity_mismatch"]
                                                             for snr in SNR_LIST)
            / max(sum(label_stats[str(snr)]["valid_slot_labels"] for snr in SNR_LIST), 1),
            "identity_unverifiable_count": 0,
            "label_jump_suspect_count": sum(label_stats[str(snr)]["label_jump_suspect"] for snr in SNR_LIST),
            "mismatch_examples": label_mismatch_examples},
        "training_pair_identity_check": {
            "rule": "for each valid slot label, the origin-frame state must be within 1 m of the label "
                    "vehicle's position at the origin",
            "valid_slot_labels_total": sum(origin_identity[str(snr)]["valid_slot_labels"]
                                           for snr in SNR_LIST),
            "alive_at_origin_total": sum(origin_identity[str(snr)]["alive_at_origin"] for snr in SNR_LIST),
            "inconsistent_gt_1m_total": sum(origin_identity[str(snr)]["inconsistent_gt_1m"]
                                            for snr in SNR_LIST),
            "inconsistent_gt_1m_fraction": sum(origin_identity[str(snr)]["inconsistent_gt_1m"]
                                               for snr in SNR_LIST)
            / max(sum(origin_identity[str(snr)]["valid_slot_labels"] for snr in SNR_LIST), 1),
            "labels_on_dead_slot_total": sum(origin_identity[str(snr)]["labels_on_dead_slot"]
                                             for snr in SNR_LIST),
            "gt_missing_at_origin_total": sum(origin_identity[str(snr)]["gt_missing_at_origin"]
                                              for snr in SNR_LIST),
            "by_snr": {str(snr): dict(origin_identity[str(snr)]) for snr in SNR_LIST},
            "by_density": {density: dict(origin_identity[f"density_{density}"])
                           for density in ("low", "mid", "high")}},
        "by_snr": {str(snr): dict(label_stats[str(snr)]) for snr in SNR_LIST},
        "switch_event_examples": switch_cases[:20],
        "notes": {"label_identity_check": "future label rows verified geometrically against the assigned "
                                          "vehicle's raw future samples; independent consistency check",
                  "legacy_generic_labels": "labels/{split}.npz is the 20 dB reference file, not the formal "
                                           "per-SNR label source"}}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (("data_contract.json", contract),
                          ("tail_attribution_summary.json", tail_attribution),
                          ("label_identity_audit.json", label_identity)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
    rows = []
    for key, candidates in sorted(tail_candidates.items()):
        candidates.sort(key=lambda item: -item["assigned_error_m"])
        rows.extend(candidates[:3])
    for case in switch_cases[:10]:
        rows.append({"snr_db": case["snr"], "episode": case["episode"], "frame": case["frame_after"],
                     "slot": case["slot"], "detected": case["detected_after"], "density": case["density"],
                     "assigned_gt_id": case["gt_after"], "nearest_gt_id": case["gt_before"],
                     "tail_cause": "identity_switch_event"})
    with (OUT / "tail_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"valid_states": total_valid, "tail": tail_summary,
                      "attribution": {name: round(value["fraction_of_tail"], 4)
                                      for name, value in attribution.items()},
                      "detected_only": by_detection["detected"]["cause_fraction_of_tail"],
                      "tracks": dict(track_stats),
                      "labels": label_identity["window_labels"], "cases": len(rows)}))


if __name__ == "__main__":
    main()
