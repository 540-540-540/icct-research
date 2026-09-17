#!/usr/bin/env python3
"""Validate the canonical ICCT trajectory dataset (read-only).

Checks (frozen rules from reports/data_cleaning/source_session_audit.md and the
canonical build script):
 1. vehicle_id is 1..N contiguous and unique;
 2. no duplicate (vehicle_id, timestamp);
 3. per vehicle timestamp strictly increasing;
 4. consecutive time gap is 0.1 s;
 5. no NaN;
 6. no Inf;
 7. no implausible position jumps (hard fail > 10 m per 0.1 s step);
 8. vx/vy finite;
 9. speed sqrt(vx^2+vy^2) physical (hard fail > 150 m/s);
10. raw v_Vel consistency (scalar speed sanity check only);
11. the 4 duplicated vehicles exist as exactly 4 vehicle_ids with 2 sources each;
12. complementary session vehicles all kept (per-source row accounting);
13. totals (vehicles, rows);
14. every deleted row is explained (merge drops only; first frame kept);
15. first-frame velocity uses forward difference and equals the second row's
    backward difference; every later row uses backward difference.

Usage:
    python tools/data_cleaning/validate_canonical_trajectories.py \
        --clean /home/dell/YrM/ICCT/data/processed/trajectories_clean.csv \
        --mapping /home/dell/YrM/ICCT/data/processed/vehicle_id_mapping.csv \
        --csv /home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv \
        --output reports/data_cleaning/canonical_validation_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEET_TO_M = 0.3048
DT_S = 0.1
EXPECTED_OFFSETS_MS = [1118935679900, 1118936699900]
DUPLICATE_PAIRS = [(1418, 2), (1421, 5), (1420, 6), (1422, 7)]
STEP_FAIL_M = 10.0
SPEED_FAIL_MPS = 150.0


def log(*args) -> None:
    print('[%7.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate canonical trajectories (read-only)')
    parser.add_argument('--clean', default='/home/dell/YrM/ICCT/data/processed/trajectories_clean.csv')
    parser.add_argument('--mapping', default='/home/dell/YrM/ICCT/data/processed/vehicle_id_mapping.csv')
    parser.add_argument('--csv', default='/home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv')
    parser.add_argument('--output', default='reports/data_cleaning/canonical_validation_stats.json')
    args = parser.parse_args()

    clean = pd.read_csv(args.clean)
    mapping = pd.read_csv(args.mapping)
    raw = pd.read_csv(args.csv, usecols=['Vehicle_ID', 'Frame_ID', 'Global_Time', 'v_Vel'])
    log('loaded clean', len(clean), 'mapping', len(mapping), 'raw', len(raw))

    global_time = raw.Global_Time.to_numpy()
    clock_offset = global_time - raw.Frame_ID.to_numpy() * 100
    offsets = np.sort(np.unique(clock_offset))
    if offsets.tolist() != EXPECTED_OFFSETS_MS:
        raise SystemExit(f'unexpected raw clock offsets: {offsets.tolist()}')
    raw['session'] = np.searchsorted(offsets, clock_offset)
    t_origin = int(global_time.min())
    boundary_ms = int(global_time[raw.session.to_numpy() == 1].min())
    raw_key = raw.set_index(['session', 'Vehicle_ID', 'Global_Time']).v_Vel * FEET_TO_M

    checks = {}

    vehicles = np.sort(clean.vehicle_id.unique())
    checks['vehicle_id_contiguous_1_to_N'] = {
        'count': int(len(vehicles)), 'min': int(vehicles.min()), 'max': int(vehicles.max()),
        'pass': bool(np.array_equal(vehicles, np.arange(1, len(vehicles) + 1))),
    }

    duplicate_rows = int(clean.duplicated(['vehicle_id', 'timestamp']).sum())
    checks['no_duplicate_vehicle_timestamp'] = {'duplicates': duplicate_rows, 'pass': duplicate_rows == 0}

    ordered = clean.sort_values(['vehicle_id', 'timestamp'], kind='mergesort')
    diffs = ordered.groupby('vehicle_id', sort=False).timestamp.diff().to_numpy()
    within = diffs[~np.isnan(diffs)]
    checks['timestamps_strictly_increasing'] = {
        'non_positive_steps': int((within <= 0).sum()), 'pass': bool(np.all(within > 0)),
    }
    dt_ok = np.allclose(within, DT_S, atol=1e-6)
    checks['time_gap_0p1s'] = {
        'steps': int(len(within)), 'max_abs_deviation': float(np.max(np.abs(within - DT_S))) if len(within) else None,
        'pass': bool(dt_ok),
    }

    nan_counts = {column: int(clean[column].isna().sum()) for column in clean.columns}
    checks['no_nan'] = {'counts': nan_counts, 'pass': bool(sum(nan_counts.values()) == 0)}
    inf_counts = {column: int(np.isinf(clean[column].to_numpy(dtype=float)).sum()) for column in clean.columns}
    checks['no_inf'] = {'counts': inf_counts, 'pass': bool(sum(inf_counts.values()) == 0)}

    dx_step = ordered.groupby('vehicle_id', sort=False).x.diff().to_numpy()
    dy_step = ordered.groupby('vehicle_id', sort=False).y.diff().to_numpy()
    step = np.hypot(dx_step, dy_step)
    step = step[~np.isnan(step)]
    big_steps = int((step > STEP_FAIL_M).sum())
    checks['position_steps'] = {
        'max_step_m': float(step.max()) if len(step) else None,
        'steps_gt_3m': int((step > 3).sum()), 'steps_gt_5m': int((step > 5).sum()),
        'steps_gt_10m': big_steps, 'pass': big_steps == 0,
    }

    finite_velocity = bool(np.all(np.isfinite(clean.vx)) and np.all(np.isfinite(clean.vy)))
    checks['velocity_finite'] = {'pass': finite_velocity}

    speed = np.hypot(clean.vx.to_numpy(), clean.vy.to_numpy())
    checks['speed_physical'] = {
        'max_mps': float(speed.max()), 'p50_mps': float(np.median(speed)),
        'p99_mps': float(np.percentile(speed, 99)), 'count_gt_30_mps': int((speed > 30).sum()),
        'pass': bool(speed.max() < SPEED_FAIL_MPS),
    }

    check_rows = clean.copy()
    check_rows['Global_Time'] = np.rint(check_rows.timestamp * 1000).astype(np.int64) + t_origin
    collision = {}
    for row in mapping.itertuples(index=False):
        collision.setdefault(int(row.vehicle_id), []).append((int(row.source_session_id),
                                                             int(row.original_vehicle_id)))
    groups = {int(v): g for v, g in check_rows.groupby('vehicle_id', sort=False)}
    matched_raw, matched_computed = [], []
    for vehicle_id, sources in collision.items():
        rows = groups[vehicle_id]
        for s, vid in sources:
            if len(sources) == 1:
                subset = rows
            elif s == 0:
                subset = rows[rows.Global_Time < boundary_ms]
            else:
                subset = rows[rows.Global_Time >= boundary_ms]
            for row in subset.itertuples(index=False):
                key = (s, vid, row.Global_Time)
                if key in raw_key.index:
                    matched_raw.append(float(raw_key.loc[key]))
                    matched_computed.append(float(np.hypot(row.vx, row.vy)))
    matched_raw = np.asarray(matched_raw)
    matched_computed = np.asarray(matched_computed)
    if len(matched_raw) > 2:
        pearson = float(np.corrcoef(matched_raw, matched_computed)[0, 1])
    else:
        pearson = None
    matched_fraction = len(matched_raw) / max(len(clean), 1)
    checks['v_vel_consistency'] = {
        'matched_rows': int(len(matched_raw)), 'matched_fraction': float(matched_fraction),
        'median_raw_mps': float(np.median(matched_raw)) if len(matched_raw) else None,
        'median_computed_mps': float(np.median(matched_computed)) if len(matched_computed) else None,
        'median_abs_diff_mps': float(np.median(np.abs(matched_computed - matched_raw)))
        if len(matched_raw) else None,
        'pearson': pearson,
        'pass': bool(matched_fraction > 0.99 and pearson is not None and pearson > 0.8),
    }

    merged_ids = sorted(vehicle_id for vehicle_id, sources in collision.items() if len(sources) > 1)
    merged_sources = {vehicle_id: sorted(collision[vehicle_id]) for vehicle_id in merged_ids}
    expected_sources = sorted([(0, a) for a, _ in DUPLICATE_PAIRS] + [(1, b) for _, b in DUPLICATE_PAIRS])
    actual_merged_sources = sorted(sum((list(collision[v]) for v in merged_ids), []))
    checks['duplicates_merged'] = {
        'merged_vehicle_ids': merged_ids, 'merged_vehicle_count': len(merged_ids),
        'sources': {str(k): v for k, v in merged_sources.items()},
        'expected_pairs': DUPLICATE_PAIRS,
        'pass': len(merged_ids) == len(DUPLICATE_PAIRS) and actual_merged_sources == expected_sources,
    }

    raw_sizes = raw.groupby(['session', 'Vehicle_ID']).size()
    merged_lookup = {}
    for index, (vid0, vid1) in enumerate(DUPLICATE_PAIRS):
        merged_lookup[(0, vid0)] = index
        merged_lookup[(1, vid1)] = index
    expected_rows_by_vehicle = {}
    for (s, vid), size in raw_sizes.items():
        if (s, vid) in merged_lookup:
            key_index = merged_lookup[(s, vid)]
            track = raw[(raw.session == s) & (raw.Vehicle_ID == vid)]
            if s == 0:
                keep = int((track.Global_Time < boundary_ms).sum())
            else:
                keep = int((track.Global_Time >= boundary_ms).sum())
            expected_rows_by_vehicle[-(key_index + 1)] = expected_rows_by_vehicle.get(-(key_index + 1), 0) + keep
        else:
            expected_rows_by_vehicle[s * 1_000_000 + vid] = int(size)
    actual_sizes = clean.groupby('vehicle_id').size()
    source_counts = mapping.groupby('source_session_id').size()
    expected_total = int(sum(expected_rows_by_vehicle.values()))
    checks['row_accounting'] = {
        'raw_rows': int(len(raw)), 'merge_dropped_rows': int(len(raw) - expected_total),
        'expected_final_rows': expected_total, 'actual_final_rows': int(len(clean)),
        'source_tracks': int(len(raw_sizes)),
        'source_tracks_session0': int(source_counts.get(0, 0)),
        'source_tracks_session1': int(source_counts.get(1, 0)),
        'complementary_tracks_kept_expected': int(len(raw_sizes) - len(DUPLICATE_PAIRS)),
        'canonical_vehicles': int(len(vehicles)),
        'pass': bool(expected_total == len(clean)
                     and len(vehicles) == len(raw_sizes) - len(DUPLICATE_PAIRS)),
    }
    # per-vehicle row counts: canonical vehicle_id -> sum of its source rows after merge filter
    expected_by_canonical = {}
    for vehicle_id, sources in collision.items():
        total = 0
        counted_merged = set()
        for s, vid in sources:
            if (s, vid) in merged_lookup:
                index = merged_lookup[(s, vid)]
                if index not in counted_merged:
                    total += expected_rows_by_vehicle[-(index + 1)]
                    counted_merged.add(index)
            else:
                total += expected_rows_by_vehicle[s * 1_000_000 + vid]
        expected_by_canonical[vehicle_id] = total
    per_vehicle_ok = True
    details = []
    for vehicle_id in vehicles:
        expected = expected_by_canonical.get(int(vehicle_id))
        got = int(actual_sizes.get(vehicle_id, 0))
        if expected != got:
            per_vehicle_ok = False
            details.append({'vehicle_id': int(vehicle_id), 'expected': expected, 'actual': got})
    checks['per_vehicle_row_counts'] = {'mismatches': details[:20], 'pass': bool(per_vehicle_ok),
                                        'vehicles_checked': int(len(vehicles))}

    first_frame_ok = True
    first_frame_mismatch = []
    for vehicle_id, group in ordered.groupby('vehicle_id', sort=False):
        xs = group.x.to_numpy()
        ys = group.y.to_numpy()
        vx = group.vx.to_numpy()
        vy = group.vy.to_numpy()
        if len(group) < 2:
            first_frame_ok = False
            first_frame_mismatch.append({'vehicle_id': int(vehicle_id), 'points': len(group)})
            continue
        expected_vx = (xs[1:] - xs[:-1]) / DT_S
        expected_vy = (ys[1:] - ys[:-1]) / DT_S
        if not np.allclose(vx[1:], expected_vx, rtol=1e-5, atol=1e-4) or \
                not np.allclose(vy[1:], expected_vy, rtol=1e-5, atol=1e-4):
            first_frame_ok = False
            first_frame_mismatch.append({'vehicle_id': int(vehicle_id), 'rule': 'backward difference mismatch'})
            continue
        if abs(vx[0] - expected_vx[0]) > 1e-4 or abs(vy[0] - expected_vy[0]) > 1e-4:
            first_frame_ok = False
            first_frame_mismatch.append({'vehicle_id': int(vehicle_id), 'rule': 'first-frame forward difference mismatch'})
    checks['first_frame_velocity_rule'] = {
        'checked_vehicles': int(len(vehicles)), 'mismatches': first_frame_mismatch[:20],
        'pass': bool(first_frame_ok),
    }

    expected_order = np.lexsort((clean.vehicle_id.to_numpy(), clean.timestamp.to_numpy()))
    sorted_ok = bool(np.array_equal(expected_order, np.arange(len(clean))))
    checks['file_sorted_by_timestamp_vehicle'] = {'pass': sorted_ok}

    totals = {'vehicles': int(len(vehicles)), 'rows': int(len(clean)),
              'columns': list(clean.columns)}
    overall = all(check.get('pass', False) for check in checks.values())
    report = {'kind': 'canonical_validation',
              'clean_csv': args.clean, 'mapping_csv': args.mapping, 'raw_csv': args.csv,
              'totals': totals, 'checks': checks, 'overall_pass': bool(overall)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf-8')
    log('validation written', output)
    print(json.dumps({'overall_pass': bool(overall),
                      'totals': totals,
                      'failed_checks': [k for k, v in checks.items() if not v.get('pass', False)]},
                     default=float))


if __name__ == '__main__':
    START = time.time()
    main()