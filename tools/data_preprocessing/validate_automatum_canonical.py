#!/usr/bin/env python3
"""Independent validation of the canonical 10 Hz Automatum T-Crossing dataset.

Reads the raw recordings and the canonical outputs directly and re-derives every
critical quantity with its own implementation (no shared code with the builder):

- schema, scene separation, vehicle_id continuity
- 0.1 s grid, strict per-track ordering, duplicates, NaN/Inf
- nearest-source-frame half-frame rule and no source frame reuse
- x/y copied from the real source frame, world velocity rotation recomputed
- speed norm preservation, position jumps, physical speed ranges
- mapping/metadata/SHA256 consistency, "raw untouched" evidence
- 2 s + 2 s window supply statistics (40 discrete 10 Hz states per window)

Exits non-zero if any check fails.

Usage:
    python tools/data_preprocessing/validate_automatum_canonical.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW_FPS = 29.97
SRC_DT = 1.0 / RAW_FPS
HALF_FRAME_S = 0.5 * SRC_DT
CANON_DT = 0.1
EPS_S = 1e-6
POS_TOL_M = 2e-6
VEL_TOL_MPS = 2e-6
JUMP_LIMIT_MPS = 45.0
SPEED_LIMIT_MPS = 35.0
FOUR_S = 4.0
SCENE_NAMES = {
    0: 'T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2',
    1: 'T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_recording_dir(raw_root: Path, name: str) -> Path:
    for candidate in (raw_root / name, raw_root / 'extracted' / name):
        if (candidate / 'dynamicWorld.json').is_file():
            return candidate
    for world in raw_root.rglob('dynamicWorld.json'):
        if world.parent.name == name:
            return world.parent
    raise FileNotFoundError(name)


def read_canonical(path: Path) -> tuple:
    header = None
    rows = []
    with path.open(encoding='utf-8', newline='') as fh:
        reader = csv.reader(fh)
        for i, fields in enumerate(reader):
            if i == 0:
                header = fields
                continue
            rows.append((int(fields[0]), int(fields[1]), float(fields[2]),
                         float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6])))
    return header, rows


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, targets)
    idx = np.clip(idx, 1, len(times) - 1)
    left = times[idx - 1]
    right = times[idx]
    choose_left = (targets - left) <= (right - targets)
    return np.where(choose_left, idx - 1, idx)


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate the canonical Automatum dataset')
    parser.add_argument('--raw-dir', default='data/automatum_t_crossing/raw')
    parser.add_argument('--processed-dir', default='data/automatum_t_crossing/processed')
    parser.add_argument('--stats', default='reports/data_preprocessing/automatum_canonical_stats.json')
    parser.add_argument('--out', default='reports/data_preprocessing/automatum_canonical_validation.json')
    args = parser.parse_args()

    raw_root = Path(args.raw_dir).resolve()
    processed = Path(args.processed_dir).resolve()
    stats_path = Path(args.stats).resolve()
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    checks = {}

    def record(name: str, passed: bool, detail) -> None:
        checks[name] = {'pass': bool(passed), 'detail': detail}

    header, rows = read_canonical(processed / 'trajectories_10hz.csv')
    record('schema_columns',
           header == ['scene_id', 'vehicle_id', 'timestamp', 'x', 'y', 'vx', 'vy'],
           {'header': header})
    scene_ids = np.array([r[0] for r in rows])
    vehicle_ids = np.array([r[1] for r in rows])
    timestamps = np.array([r[2] for r in rows])
    x = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows])
    vx = np.array([r[5] for r in rows])
    vy = np.array([r[6] for r in rows])
    record('rows_nonzero', len(rows) > 0, {'rows': len(rows)})

    record('scene_ids_only_0_1', set(np.unique(scene_ids).tolist()) == {0, 1},
           {'scene_ids': sorted(set(np.unique(scene_ids).tolist()))})
    id_set = set(np.unique(vehicle_ids).tolist())
    record('vehicle_id_continuous_1_683', id_set == set(range(1, 684)),
           {'count': len(id_set), 'min': min(id_set), 'max': max(id_set)})
    scene0_ids = set(vehicle_ids[scene_ids == 0].tolist())
    scene1_ids = set(vehicle_ids[scene_ids == 1].tolist())
    record('scenes_do_not_share_vehicles', not (scene0_ids & scene1_ids),
           {'scene_0_vehicles': len(scene0_ids), 'scene_1_vehicles': len(scene1_ids),
            'overlap': len(scene0_ids & scene1_ids)})

    keys = list(zip(scene_ids.tolist(), vehicle_ids.tolist(), timestamps.tolist()))
    record('no_duplicate_state_rows', len(keys) == len(set(keys)),
           {'rows': len(keys), 'unique': len(set(keys))})

    grid_error = np.abs(timestamps / CANON_DT - np.round(timestamps / CANON_DT))
    record('timestamp_on_0.1s_grid', bool(grid_error.max() <= EPS_S / CANON_DT),
           {'max_grid_error_s': float(grid_error.max() * CANON_DT), 'min_timestamp': float(timestamps.min())})
    record('finite_values', bool(np.isfinite(np.column_stack([timestamps, x, y, vx, vy])).all()),
           {'nan': int(np.isnan(np.column_stack([timestamps, x, y, vx, vy])).sum()),
            'inf': int(np.isinf(np.column_stack([timestamps, x, y, vx, vy])).sum())})

    tracks = defaultdict(list)
    for row in rows:
        tracks[(row[0], row[1])].append(row)
    strict = gaps = 0
    gap_examples = []
    for key, entries in tracks.items():
        entries.sort(key=lambda e: e[2])
        t = np.array([e[2] for e in entries])
        d = np.diff(t)
        if d.size and not np.all(d > 0):
            strict += 1
        off = np.abs(d - CANON_DT) > EPS_S
        if off.any():
            gaps += 1
            if len(gap_examples) < 5:
                gap_examples.append({'scene_id': key[0], 'vehicle_id': key[1],
                                     'max_dt_s': float(d.max()), 'min_dt_s': float(d.min())})
    record('per_track_strictly_increasing', strict == 0, {'violating_tracks': strict})
    record('per_track_dt_is_0.1s', gaps == 0, {'tracks_with_other_dt': gaps, 'examples': gap_examples})

    speeds = np.hypot(vx, vy)
    fd_vx = np.array([])
    fd_vy = np.array([])
    speed_violations = 0
    jump_violations = 0
    max_step_speed = 0.0
    for key, entries in tracks.items():
        entries.sort(key=lambda e: e[2])
        ex = np.array([e[3] for e in entries])
        ey = np.array([e[4] for e in entries])
        if len(entries) < 2:
            continue
        step_speed = np.hypot(np.diff(ex), np.diff(ey)) / CANON_DT
        max_step_speed = max(max_step_speed, float(step_speed.max()))
        jump_violations += int((step_speed > JUMP_LIMIT_MPS).sum())
    speed_violations = int((speeds > SPEED_LIMIT_MPS).sum())
    record('no_position_jumps', jump_violations == 0,
           {'max_step_speed_mps': max_step_speed, 'violations': jump_violations})
    record('speed_range_physical', speed_violations == 0,
           {'max_speed_mps': float(speeds.max()), 'violations_over_35mps': speed_violations})

    # independent raw re-derivation: nearest source frame, copy checks, rotation
    raw_by_scene = {}
    for scene_id, name in SCENE_NAMES.items():
        raw_dir = find_recording_dir(raw_root, name)
        world = json.loads((raw_dir / 'dynamicWorld.json').read_text(encoding='utf-8'))
        raw_by_scene[scene_id] = {'dir': raw_dir, 'world': world,
                                  'by_uuid': {o['UUID']: o for o in world['objects']}}
        recorded = stats['scenes'][str(scene_id)]['raw_file_sha256']
        record('raw_file_sha256_scene_%d' % scene_id,
               sha256_file(raw_dir / 'dynamicWorld.json') == recorded['dynamicWorld.json']
               and sha256_file(raw_dir / 'staticWorld.xodr') == recorded['staticWorld.xodr'],
               {'dynamicWorld.json': recorded['dynamicWorld.json'], 'staticWorld.xodr': recorded['staticWorld.xodr']})

    mapping_path = processed / 'vehicle_id_mapping.csv'
    mapping = list(csv.DictReader(mapping_path.open(encoding='utf-8', newline='')))
    record('mapping_rows_683', len(mapping) == 683, {'rows': len(mapping)})
    map_by_id = {int(m['vehicle_id']): m for m in mapping}
    record('mapping_ids_match_canonical', set(map_by_id) == id_set, {'mapping_ids': len(map_by_id)})
    record('mapping_uuids_unique', len({m['original_uuid'] for m in mapping}) == len(mapping),
           {'unique_uuids': len({m['original_uuid'] for m in mapping})})

    time_errors = []
    reuse_violations = 0
    copy_violations = 0
    rotation_violations = 0
    norm_violations = 0
    mapping_mismatches = 0
    fd_err_vx, fd_err_vy, dir_err = [], [], []
    for (scene_id, vehicle_id), entries in tracks.items():
        entry = map_by_id.get(vehicle_id)
        if entry is None or int(entry['scene_id']) != scene_id:
            mapping_mismatches += 1
            continue
        raw = raw_by_scene[scene_id]['by_uuid'].get(entry['original_uuid'])
        if raw is None:
            mapping_mismatches += 1
            continue
        if (abs(float(entry['original_first_time']) - float(raw['time'][0])) > 1e-6
                or abs(float(entry['original_last_time']) - float(raw['time'][-1])) > 1e-6
                or entry['obj_type'] != raw['objType']
                or abs(float(entry['length']) - float(raw['length'])) > 1e-6
                or abs(float(entry['width']) - float(raw['width'])) > 1e-6):
            mapping_mismatches += 1
        t = np.asarray(raw['time'], dtype=np.float64)
        rx = np.asarray(raw['x_vec'], dtype=np.float64)
        ry = np.asarray(raw['y_vec'], dtype=np.float64)
        rvx = np.asarray(raw['vx_vec'], dtype=np.float64)
        rvy = np.asarray(raw['vy_vec'], dtype=np.float64)
        rpsi = np.asarray(raw['psi_vec'], dtype=np.float64)
        entries.sort(key=lambda e: e[2])
        times = np.array([e[2] for e in entries])
        idx = nearest_indices(t, times)
        time_errors.append(np.abs(t[idx] - times))
        if np.any(np.diff(idx) <= 0):
            reuse_violations += 1
        if (np.abs(rx[idx] - np.array([e[3] for e in entries])) > POS_TOL_M).any() \
                or (np.abs(ry[idx] - np.array([e[4] for e in entries])) > POS_TOL_M).any():
            copy_violations += 1
        wx = np.cos(rpsi[idx]) * rvx[idx] - np.sin(rpsi[idx]) * rvy[idx]
        wy = np.sin(rpsi[idx]) * rvx[idx] + np.cos(rpsi[idx]) * rvy[idx]
        if (np.abs(wx - np.array([e[5] for e in entries])) > VEL_TOL_MPS).any() \
                or (np.abs(wy - np.array([e[6] for e in entries])) > VEL_TOL_MPS).any():
            rotation_violations += 1
        if (np.abs(np.hypot(rvx[idx], rvy[idx]) - np.hypot(wx, wy)) > VEL_TOL_MPS).any():
            norm_violations += 1
        if len(entries) >= 2:
            mid_vx = 0.5 * (np.array([e[5] for e in entries])[:-1] + np.array([e[5] for e in entries])[1:])
            mid_vy = 0.5 * (np.array([e[6] for e in entries])[:-1] + np.array([e[6] for e in entries])[1:])
            fd_true_vx = (np.array([e[3] for e in entries])[1:] - np.array([e[3] for e in entries])[:-1]) / np.diff(t[idx])
            fd_true_vy = (np.array([e[4] for e in entries])[1:] - np.array([e[4] for e in entries])[:-1]) / np.diff(t[idx])
            fd_err_vx.append(fd_true_vx - mid_vx)
            fd_err_vy.append(fd_true_vy - mid_vy)
            moving = np.hypot(mid_vx, mid_vy) > 1.0
            if moving.any():
                cross = fd_true_vx[moving] * mid_vy[moving] - fd_true_vy[moving] * mid_vx[moving]
                dot = fd_true_vx[moving] * mid_vx[moving] + fd_true_vy[moving] * mid_vy[moving]
                dir_err.append(np.degrees(np.abs(np.arctan2(cross, dot))))
    all_time_errors = np.concatenate(time_errors)
    max_error_ms = float(all_time_errors.max() * 1000.0)
    record('half_frame_rule', max_error_ms <= HALF_FRAME_S * 1000.0 + 1e-6,
           {'max_error_ms': max_error_ms, 'half_frame_ms': HALF_FRAME_S * 1000.0,
            'median_ms': float(np.median(all_time_errors) * 1000.0),
            'p95_ms': float(np.percentile(all_time_errors, 95) * 1000.0),
            'p99_ms': float(np.percentile(all_time_errors, 99) * 1000.0)})
    record('no_source_frame_reuse', reuse_violations == 0, {'violating_tracks': reuse_violations})
    record('positions_copied_from_source', copy_violations == 0, {'violating_tracks': copy_violations})
    record('rotation_reproduces_world_velocity', rotation_violations == 0,
           {'violating_tracks': rotation_violations})
    record('speed_norm_preserved', norm_violations == 0, {'violating_tracks': norm_violations,
                                                          'tolerance_mps': VEL_TOL_MPS})
    record('mapping_traceable_to_raw', mapping_mismatches == 0, {'mismatches': mapping_mismatches})

    fd_err_vx = np.concatenate(fd_err_vx)
    fd_err_vy = np.concatenate(fd_err_vy)
    dir_err = np.concatenate(dir_err)
    dir_q = np.percentile(dir_err, [50, 95, 99, 100])
    record('velocity_fd_consistency', bool(np.abs(fd_err_vx).mean() < 0.5 and np.abs(fd_err_vy).mean() < 0.5
                                           and dir_q[1] < 10.0),
           {'vx_mae_mps': float(np.abs(fd_err_vx).mean()), 'vy_mae_mps': float(np.abs(fd_err_vy).mean()),
            'vx_p95_mps': float(np.percentile(np.abs(fd_err_vx), 95)),
            'vy_p95_mps': float(np.percentile(np.abs(fd_err_vy), 95)),
            'direction_median_deg': float(dir_q[0]), 'direction_p95_deg': float(dir_q[1]),
            'direction_max_deg': float(dir_q[3]), 'pairs': int(len(fd_err_vx))})

    # ordering / scene block check
    order_ok = True
    for a, b in zip(rows, rows[1:]):
        if (b[0], b[2], b[1]) < (a[0], a[2], a[1]):
            order_ok = False
            break
    record('file_sorted_scene_time_vehicle', order_ok, {'sorted': order_ok})

    # metadata + stats consistency
    metadata = json.loads((processed / 'scene_metadata.json').read_text(encoding='utf-8'))
    meta_ok = True
    meta_detail = {}
    for scene_id in (0, 1):
        meta = metadata['scenes'][str(scene_id)]
        count = int((scene_ids == scene_id).sum())
        ts_min = float(timestamps[scene_ids == scene_id].min())
        ts_max = float(timestamps[scene_ids == scene_id].max())
        vehicles = len(set(vehicle_ids[scene_ids == scene_id].tolist()))
        scene_ok = (meta['canonical_rows'] == count and meta['num_vehicles'] == vehicles
                    and abs(meta['canonical_timestamp_range_s'][0] - ts_min) < EPS_S
                    and abs(meta['canonical_timestamp_range_s'][1] - ts_max) < EPS_S)
        meta_ok = meta_ok and scene_ok
        meta_detail[str(scene_id)] = {'rows': count, 'vehicles': vehicles,
                                      'timestamp_range': [ts_min, ts_max], 'stats_ok': scene_ok}
    record('scene_metadata_consistent', meta_ok, meta_detail)

    durations = {}
    for scene_id in (0, 1):
        counts = defaultdict(int)
        for row in rows:
            if row[0] == scene_id:
                counts[row[1]] = max(counts[row[1]], row[2])
        firsts = {}
        for row in rows:
            if row[0] == scene_id:
                firsts.setdefault(row[1], row[2])
        durations[scene_id] = sum(1 for v in counts if counts[v] - firsts[v] >= FOUR_S - EPS_S)
    record('four_second_counts_match_stats',
           durations[0] == stats['four_second_trajectories']['scene_0']
           and durations[1] == stats['four_second_trajectories']['scene_1'],
           {'scene_0': durations[0], 'scene_1': durations[1],
            'stats': [stats['four_second_trajectories']['scene_0'],
                      stats['four_second_trajectories']['scene_1']]})

    sha_ok = {}
    for name in ('trajectories_10hz.csv', 'vehicle_id_mapping.csv'):
        actual = sha256_file(processed / name)
        sha_ok[name] = actual == stats['outputs'][name]['sha256']
        record('sha256_%s' % name.replace('.', '_'), sha_ok[name],
               {'actual': actual, 'stats': stats['outputs'][name]['sha256']})
    actual_meta = sha256_file(processed / 'scene_metadata.json')
    sha_ok['scene_metadata.json'] = actual_meta == stats['outputs']['scene_metadata.json']['sha256']
    record('sha256_scene_metadata_json', sha_ok['scene_metadata.json'],
           {'actual': actual_meta, 'stats': stats['outputs']['scene_metadata.json']['sha256']})

    failed = [name for name, entry in checks.items() if not entry['pass']]
    payload = {
        'dataset': 'automatum_t_crossing_canonical',
        'validator': 'tools/data_preprocessing/validate_automatum_canonical.py',
        'inputs': {'trajectories_10hz.csv': str(processed / 'trajectories_10hz.csv'),
                   'stats': str(stats_path)},
        'checks': checks,
        'summary': {'checks_total': len(checks), 'checks_passed': len(checks) - len(failed),
                    'checks_failed': len(failed), 'failed_checks': failed,
                    'verdict': 'PASS' if not failed else 'FAIL'},
        'velocity_validation_true_source_interval': {
            'vx_mae_mps': float(np.abs(fd_err_vx).mean()), 'vy_mae_mps': float(np.abs(fd_err_vy).mean()),
            'vx_rmse_mps': float(np.sqrt(np.mean(fd_err_vx ** 2))),
            'vy_rmse_mps': float(np.sqrt(np.mean(fd_err_vy ** 2))),
            'vx_p95_abs_mps': float(np.percentile(np.abs(fd_err_vx), 95)),
            'vy_p95_abs_mps': float(np.percentile(np.abs(fd_err_vy), 95)),
            'direction_median_deg': float(dir_q[0]), 'direction_p95_deg': float(dir_q[1]),
            'direction_p99_deg': float(dir_q[2]), 'direction_max_deg': float(dir_q[3]),
            'pairs': int(len(fd_err_vx))},
        'time_mapping_error_ms': {'median': float(np.median(all_time_errors) * 1000.0),
                                  'p95': float(np.percentile(all_time_errors, 95) * 1000.0),
                                  'p99': float(np.percentile(all_time_errors, 99) * 1000.0),
                                  'max': max_error_ms, 'half_frame': HALF_FRAME_S * 1000.0},
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + '\n')

    print(json.dumps({'rows': len(rows), 'checks': payload['summary'],
                      'main_checks': {k: v['pass'] for k, v in checks.items()},
                      'output': str(out_path)}, indent=1, ensure_ascii=False))
    if failed:
        print('FAILED CHECKS:', failed, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()