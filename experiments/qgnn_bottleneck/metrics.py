"""Metric primitives for BDX-01.  All functions are inference-only NumPy code."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


HORIZONS = (5, 10, 20)


def eligible_mask(track_exists: np.ndarray) -> np.ndarray:
    track_exists = np.asarray(track_exists, dtype=bool)
    if track_exists.ndim != 3:
        raise ValueError("track_exists must be [scene,history,slot]")
    return track_exists[:, -1] & (track_exists.sum(axis=1) >= 3)


def masked_distances(prediction, future, label_valid, eligible):
    prediction, future = np.asarray(prediction), np.asarray(future)
    label_valid, eligible = np.asarray(label_valid, bool), np.asarray(eligible, bool)
    if prediction.shape != future.shape or prediction.ndim != 4 or prediction.shape[1:] != (20, 8, 2):
        raise ValueError("prediction/future must be matching [scene,20,8,2]")
    if label_valid.shape != prediction.shape[:-1] or eligible.shape != (len(prediction), 8):
        raise ValueError("mask shape mismatch")
    valid = label_valid & eligible[:, None, :]
    distance = np.linalg.norm(prediction.astype(np.float64) - future.astype(np.float64), axis=-1)
    distance[~valid] = 0.0
    return distance, valid


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else None


def evaluate_metrics(prediction, future, label_valid, eligible, horizons: Iterable[int] = HORIZONS):
    """Return protocol macro, training-rule, and fixed-complete-target metrics."""
    distance, valid = masked_distances(prediction, future, label_valid, eligible)
    rows = []
    full_targets = eligible & label_valid.all(axis=1)
    for horizon in horizons:
        if horizon < 1 or horizon > 20:
            raise ValueError("horizon must be within 1..20")
        prefix = valid[:, :horizon]
        counts = prefix.sum(axis=1)
        target_ok = counts > 0
        target_ade = np.divide(distance[:, :horizon].sum(axis=1), counts,
                               out=np.zeros_like(counts, dtype=np.float64), where=target_ok)
        end_ok = valid[:, horizon - 1]
        target_fde = distance[:, horizon - 1]
        scene_ade_n = target_ok.sum(axis=1)
        scene_fde_n = end_ok.sum(axis=1)
        scene_ade = np.divide((target_ade * target_ok).sum(axis=1), scene_ade_n,
                              out=np.zeros(len(prediction)), where=scene_ade_n > 0)
        scene_fde = np.divide((target_fde * end_ok).sum(axis=1), scene_fde_n,
                              out=np.zeros(len(prediction)), where=scene_fde_n > 0)
        ade_scenes = scene_ade_n > 0
        fde_scenes = scene_fde_n > 0
        point_n = prefix.sum(axis=(1, 2))
        point_ade = np.divide(distance[:, :horizon].sum(axis=(1, 2)), point_n,
                              out=np.zeros(len(prediction)), where=point_n > 0)
        fixed_n = full_targets.sum(axis=1)
        fixed_ade = np.divide((distance[:, :horizon].mean(axis=1) * full_targets).sum(axis=1), fixed_n,
                              out=np.zeros(len(prediction)), where=fixed_n > 0)
        fixed_fde = np.divide((distance[:, horizon - 1] * full_targets).sum(axis=1), fixed_n,
                              out=np.zeros(len(prediction)), where=fixed_n > 0)
        ade = _mean_or_none(scene_ade[ade_scenes])
        fde = _mean_or_none(scene_fde[fde_scenes])
        rows.append(dict(
            horizon_steps=horizon, horizon_seconds=horizon * 0.1,
            ADE=ade, FDE=fde, J=None if ade is None or fde is None else ade + 0.5 * fde,
            ade_scenes=int(ade_scenes.sum()), fde_scenes=int(fde_scenes.sum()),
            ade_targets=int(target_ok.sum()), fde_targets=int(end_ok.sum()),
            valid_points=int(prefix.sum()), total_scenes=len(prediction),
            scenes_without_supervision=int((point_n == 0).sum()),
            ADE_point_trainrule_eval=float(point_ade.mean()),
            FDE_trainrule_eval=float(scene_fde.mean()),
            J_trainrule_eval=float((point_ade + 0.5 * scene_fde).mean()),
            ADE_point_scoreable_scenes=_mean_or_none(point_ade[point_n > 0]),
            FDE_scoreable_scenes=_mean_or_none(scene_fde[fde_scenes]),
            fixed_complete_targets=int(full_targets.sum()),
            fixed_complete_scenes=int((fixed_n > 0).sum()),
            fixed_ADE=_mean_or_none(fixed_ade[fixed_n > 0]),
            fixed_FDE=_mean_or_none(fixed_fde[fixed_n > 0]),
        ))
    return rows


def scene_rows(prediction, future, label_valid, eligible, horizon=20):
    distance, valid = masked_distances(prediction, future, label_valid, eligible)
    prefix = valid[:, :horizon]
    counts = prefix.sum(axis=1)
    target_ok = counts > 0
    target_ade = np.divide(distance[:, :horizon].sum(axis=1), counts,
                           out=np.zeros_like(counts, dtype=np.float64), where=target_ok)
    end_ok = valid[:, horizon - 1]
    rows = []
    for origin in range(len(prediction)):
        an, fn = int(target_ok[origin].sum()), int(end_ok[origin].sum())
        rows.append(dict(origin=origin,
            ADE=float(target_ade[origin, target_ok[origin]].mean()) if an else None,
            FDE=float(distance[origin, horizon - 1, end_ok[origin]].mean()) if fn else None,
            ade_targets=an, fde_targets=fn,
            valid_points=int(prefix[origin].sum())))
    return rows


def target_rows(prediction, future, label_valid, eligible):
    distance, valid = masked_distances(prediction, future, label_valid, eligible)
    rows = []
    for origin in range(len(prediction)):
        for slot in range(8):
            if not eligible[origin, slot]:
                continue
            row = dict(origin=origin, slot=slot, eligible=True,
                       valid_steps=int(valid[origin, :, slot].sum()))
            for horizon in HORIZONS:
                mask = valid[origin, :horizon, slot]
                row[f"ADE{horizon}"] = float(distance[origin, :horizon, slot][mask].mean()) if mask.any() else None
                row[f"FDE{horizon}"] = float(distance[origin, horizon - 1, slot]) if valid[origin, horizon - 1, slot] else None
            for step in range(20):
                row[f"step_{step+1:02d}"] = float(distance[origin, step, slot]) if valid[origin, step, slot] else None
            rows.append(row)
    return rows


def percentile_summary(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(x):
        return dict(count=0, mean=None, median=None, p90=None, p99=None)
    return dict(count=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), p99=float(np.quantile(x, .99)))


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def correlation(left, right, method="pearson"):
    a, b = np.asarray(left, float), np.asarray(right, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(value=None, count=int(len(a)), reason="fewer_than_two_or_zero_variance")
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    elif method != "pearson":
        raise ValueError("method must be pearson or spearman")
    return dict(value=float(np.corrcoef(a, b)[0, 1]), count=int(len(a)), reason=None)


def worst_overlap(keys, left, right, fraction=.1):
    if len(keys) != len(left) or len(keys) != len(right):
        raise ValueError("paired arrays differ")
    n, k = len(keys), int(math.ceil(fraction * len(keys)))
    ordered_left = sorted(zip(left, keys), key=lambda x: (-x[0], x[1]))[:k]
    ordered_right = sorted(zip(right, keys), key=lambda x: (-x[0], x[1]))[:k]
    a, b = {key for _, key in ordered_left}, {key for _, key in ordered_right}
    intersection = a & b
    return dict(n=n, k=k, intersection=len(intersection), union=len(a | b),
                jaccard=len(intersection) / len(a | b) if a | b else None)
