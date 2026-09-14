"""Source alignment, fixed input strata, and paired descriptive analysis for BDX-01."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from experiments.qgnn_bottleneck.metrics import correlation, percentile_summary, worst_overlap


def source_components(samples):
    episodes = sorted({row["episode_id"] for row in samples})
    groups = {episode: set() for episode in episodes}
    for row in samples:
        groups[row["episode_id"]].update(row["source_groups"])
    remaining, components = set(episodes), []
    while remaining:
        component = {min(remaining)}
        while True:
            identities = set().union(*(groups[e] for e in component))
            joined = {e for e in remaining if groups[e] & identities}
            if joined <= component:
                break
            component |= joined
        remaining -= component
        components.append(sorted(component))
    lookup = {episode: i for i, component in enumerate(components) for episode in component}
    return lookup, components


def _history_proxy(state, exists, slot):
    indices = np.flatnonzero(exists[:, slot])
    count = len(indices)
    first, last = indices[indices < 5], indices[indices >= 15]
    speed_change = heading = None
    if len(first) >= 3 and len(last) >= 3:
        va, vb = state[first, slot, 2:].astype(float), state[last, slot, 2:].astype(float)
        speed_change = abs(np.linalg.norm(vb, axis=1).mean() - np.linalg.norm(va, axis=1).mean())
        a, b = va.mean(0), vb.mean(0)
        if min(np.linalg.norm(a), np.linalg.norm(b)) >= 1.0:
            heading = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    return count, speed_change, heading


def input_feature_rows(split, state, exists, samples):
    rows = []
    component_lookup, _ = source_components(samples)
    for origin, meta in enumerate(samples):
        live = np.flatnonzero(exists[origin, -1])
        positions = state[origin, -1, :, :2].astype(float)
        for slot in live:
            history_count, speed_change, heading = _history_proxy(state[origin], exists[origin], slot)
            others = live[live != slot]
            neighbors = int((np.linalg.norm(positions[others] - positions[slot], axis=1) <= 45).sum())
            rows.append(dict(split=split, origin=origin, slot=int(slot), episode_id=meta["episode_id"],
                source_block=int(meta["source_block"]), source_component=component_lookup[meta["episode_id"]],
                source_key=int(meta["source_keys"][slot]) if slot < len(meta["source_keys"]) else None,
                active_tracks_last=int(len(live)), neighbor_count=neighbors,
                history_count=int(history_count), speed_change_proxy=speed_change,
                heading_change_proxy_deg=heading,
                forward_speed_mps=float(np.linalg.norm(state[origin, -1, slot, 2:])),
                forward_vx_mps=float(state[origin, -1, slot, 2]),
                forward_vy_mps=float(state[origin, -1, slot, 3])))
    return rows


def define_speed_tertiles(train_rows):
    values = np.asarray([r["speed_change_proxy"] for r in train_rows if r["speed_change_proxy"] is not None], float)
    return np.quantile(values, [1/3, 2/3]).tolist() if len(values) else []


def assign_strata(rows, speed_edges):
    for row in rows:
        tracks = row["active_tracks_last"]
        row["stratum_active_tracks"] = "1-2" if tracks <= 2 else "3-5" if tracks <= 5 else "6-8"
        degree = row["neighbor_count"]
        row["stratum_neighbors"] = "0" if degree == 0 else "1-2" if degree <= 2 else ">=3"
        history = row["history_count"]
        row["stratum_history"] = "3-9" if history <= 9 else "10-19" if history <= 19 else "20"
        speed = row["speed_change_proxy"]
        row["stratum_speed_change"] = "unavailable" if speed is None or len(speed_edges) != 2 else f"T{np.searchsorted(speed_edges, speed, side='right') + 1}"
        heading = row["heading_change_proxy_deg"]
        row["stratum_heading_change"] = "unavailable" if heading is None else "<5deg" if heading < 5 else "5-15deg" if heading < 15 else ">=15deg"
    return rows


class SourceLookup:
    """Binary-search only requested keys/times in the sorted mixed source memmap."""
    def __init__(self, rows):
        self.rows = rows
        self.keys = rows["source_key"]

    def at(self, key, time_ms):
        left = int(np.searchsorted(self.keys, key, side="left"))
        right = int(np.searchsorted(self.keys, key, side="right"))
        if left == right:
            return None
        times = self.rows["time_ms"][left:right]
        pos = int(np.searchsorted(times, time_ms))
        if pos >= len(times) or int(times[pos]) != int(time_ms):
            return None
        return self.rows[left + pos]


def aligned_reference(samples, eligible, source_rows):
    lookup = SourceLookup(source_rows)
    position = np.full((*eligible.shape, 2), np.nan, np.float64)
    velocity = np.full((*eligible.shape, 2), np.nan, np.float64)
    available = np.zeros_like(eligible, bool)
    reasons = defaultdict(int)
    for origin, meta in enumerate(samples):
        for slot in range(eligible.shape[1]):
            if not eligible[origin, slot]:
                reasons["not_model_eligible"] += 1
                continue
            if slot >= len(meta["source_keys"]):
                reasons["missing_source_slot"] += 1
                continue
            row = lookup.at(int(meta["source_keys"][slot]), int(meta["origin_ms"]))
            if row is None:
                reasons["missing_exact_origin_row"] += 1
                continue
            if int(row["past_frames"]) < 3:
                reasons["source_history_below_three"] += 1
                continue
            position[origin, slot] = [row["x"], row["y"]]
            velocity[origin, slot] = [row["vx"], row["vy"]]
            available[origin, slot] = True
    return dict(position=position, velocity=velocity, available=available,
                reasons=dict(reasons), requested_eligible=int(eligible.sum()), aligned=int(available.sum()))


def paired_error_summary(rows, left="Q_best", right="G_best"):
    keys = [(int(r["origin"]), int(r["slot"])) for r in rows]
    result = {}
    for metric in ("ADE20", "FDE20"):
        a = np.asarray([r.get(f"{left}_{metric}", np.nan) for r in rows], float)
        b = np.asarray([r.get(f"{right}_{metric}", np.nan) for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(b)
        k, av, bv = [x for x, keep in zip(keys, ok) if keep], a[ok], b[ok]
        delta = av - bv
        result[metric] = dict(
            pearson=correlation(av, bv, "pearson"), spearman=correlation(av, bv, "spearman"),
            worst10=worst_overlap(k, av, bv), delta=percentile_summary(delta),
            left_better=int((delta < 0).sum()), right_better=int((delta > 0).sum()), equal=int((delta == 0).sum()))
    return result


def aggregate_groups(rows, value_fields, group_fields):
    output = []
    for field in group_fields:
        groups = sorted({str(r[field]) for r in rows})
        for group in groups:
            part = [r for r in rows if str(r[field]) == group]
            row = dict(group_type=field, group_value=group, targets=len(part),
                       origins=len({r["origin"] for r in part}), episodes=len({r["episode_id"] for r in part}))
            for value in value_fields:
                vals = [r[value] for r in part if r.get(value) is not None and np.isfinite(r[value])]
                row[value] = float(np.mean(vals)) if vals else None
            output.append(row)
    return output


def leave_one_group_out(rows, value, group):
    groups = sorted({r[group] for r in rows})
    all_values = np.asarray([r[value] for r in rows if r.get(value) is not None], float)
    estimates = []
    for excluded in groups:
        values = np.asarray([r[value] for r in rows if r[group] != excluded and r.get(value) is not None], float)
        if len(values):
            estimates.append(float(values.mean()))
    return dict(overall=float(all_values.mean()) if len(all_values) else None,
                groups=len(groups), min=min(estimates) if estimates else None,
                max=max(estimates) if estimates else None,
                sign_flip=bool(estimates and np.any(np.asarray(estimates) * all_values.mean() < 0)))
