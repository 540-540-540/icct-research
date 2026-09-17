#!/usr/bin/env python3
"""Independent validation of the canonical Automatum T-Crossing dataset (stride 3).

Reads the raw recordings and the canonical outputs directly and re-derives every
critical quantity with its own implementation (no shared code with the builder):

- schema, scene separation, vehicle_id continuity and stable-id rule
- source-frame recovery, frame %% 3 == 0, stride exactly 3, no stride 2/4
- true source timestamps (never a 0.1 s relabel), constant physical dt
- scene-global phase: one source frame per (scene_id, timestamp)
- x/y copied from the real source frame, world velocity rotation recomputed
- speed norm preservation, position jumps, physical speed ranges
- velocity FD consistency with no grid-label artifact
- mapping/metadata/SHA256 consistency, "raw untouched" evidence
- 20 + 20 sample window supply statistics

Exits non-zero if any check fails.

Usage:
    python tools/data_preprocessing/validate_automatum_canonical.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW_FPS = 29.97
SRC_DT = 1.0 / RAW_FPS
STRIDE = 3
CANON_DT = STRIDE * SRC_DT
EPS_S = 1e-9
EXACT_TOL_S = 1e-12
POS_TOL_M = 2e-6
VEL_TOL_MPS = 2e-6
JUMP_LIMIT_MPS = 45.0
SPEED_LIMIT_MPS = 35.0
WINDOW_SAMPLES = 40
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
            rows.append((int(fields[0]), int(fields[1]), fields[2],
                         float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6])))
    return header, rows


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
    record('rows_nonzero', len(rows) > 0, {'rows': len(rows)})
    scene_ids = np.array([r[0] for r in rows])
    vehicle_ids = np.array([r[1] for r in rows])
    timestamps = np.array([float(r[2]) for r in rows])
    x = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows])
    vx = np.array([r[5] for r in rows])
    vy = np.array([r[6] for r in rows])

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
    record('finite_values', bool(np.isfinite(np.column_stack([timestamps, x, y, vx, vy])).all()),
           {'nan': int(np.isnan(np.column_stack([timestamps, x, y, vx, vy])).sum()),
            'inf': int(np.isinf(np.column_stack([timestamps, x, y, vx, vy])).sum())})

    tracks = defaultdict(list)
    for row in rows:
        tracks[(row[0], row[1])].append(row)

    strict = 0
    dt_bad = 0
    dt_values = []
    for key, entries in tracks.items():
        entries.sort(key=lambda e: float(e[2]))
        t = np.array([float(e[2]) for e in entries])
        d = np.diff(t)
        dt_values.append(d)
        if d.size and not np.all(d > 0):
            strict += 1
        if d.size and np.any(np.abs(d - CANON_DT) > EPS_S):
            dt_bad += 1
    dt_all = np.concatenate([d for d in dt_values if d.size])
    record('per_track_strictly_increasing', strict == 0, {'violating_tracks': strict})
    record('physical_dt_constant', dt_bad == 0,
           {'violating_tracks': dt_bad, 'median_s': float(np.median(dt_all)),
            'p95_s': float(np.percentile(dt_all, 95)), 'p99_s': float(np.percentile(dt_all, 99)),
            'min_s': float(dt_all.min()), 'max_s': float(dt_all.max()), 'nominal_s': CANON_DT})

    speeds = np.hypot(vx, vy)
    step_violations = 0
    max_step_speed = 0.0
    for key, entries in tracks.items():
        entries.sort(key=lambda e: float(e[2]))
        ex = np.array([e[3] for e in entries])
        ey = np.array([e[4] for e in entries])
        if len(entries) < 2:
            continue
        step_speed = np.hypot(np.diff(ex), np.diff(ey)) / CANON_DT
        max_step_speed = max(max_step_speed, float(step_speed.max()))
        step_violations += int((step_speed > JUMP_LIMIT_MPS).sum())
    record('no_position_jumps', step_violations == 0,
           {'max_step_speed_mps': max_step_speed, 'violations': step_violations})
    record('speed_range_physical', bool((speeds <= SPEED_LIMIT_MPS).all()),
           {'max_speed_mps': float(speeds.max()),
            'violations_over_35mps': int((speeds > SPEED_LIMIT_MPS).sum())})

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

    expected_ids = {}
    next_id = 1
    for scene_id in (0, 1):
        entries = sorted(((float(m['original_first_time']), m['original_uuid'])
                          for m in mapping if int(m['scene_id']) == scene_id))
        for _t0, uuid in entries:
            expected_ids[uuid] = next_id
            next_id += 1
    actual_ids = {m['original_uuid']: int(m['vehicle_id']) for m in mapping}
    id_rule_ok = expected_ids == actual_ids
    record('vehicle_id_rule_reproduced', id_rule_ok,
           {'rule': '(scene_id, raw first timestamp, original UUID)',
            'mismatches': int(sum(1 for u in expected_ids if expected_ids[u] != actual_ids.get(u)))})

    stride_bad = 0
    stride_counts = defaultdict(int)
    no_reuse = 0
    copy_bad = 0
    rotation_bad = 0
    norm_bad = 0
    mapping_bad = 0
    frame_phase_bad = 0
    timestamp_bad = 0
    fd_err_vx, fd_err_vy, dir_err = [], [], []
    phase_groups = defaultdict(set)
    for (scene_id, vehicle_id), entries in tracks.items():
        entry = map_by_id.get(vehicle_id)
        if entry is None or int(entry['scene_id']) != scene_id:
            mapping_bad += 1
            continue
        raw = raw_by_scene[scene_id]['by_uuid'].get(entry['original_uuid'])
        if raw is None:
            mapping_bad += 1
            continue
        if (abs(float(entry['original_first_time']) - float(raw['time'][0])) > 1e-6
                or abs(float(entry['original_last_time']) - float(raw['time'][-1])) > 1e-6
                or entry['obj_type'] != raw['objType']
                or abs(float(entry['length']) - float(raw['length'])) > 1e-6
                or abs(float(entry['width']) - float(raw['width'])) > 1e-6):
            mapping_bad += 1
        t = np.asarray(raw['time'], dtype=np.float64)
        rx = np.asarray(raw['x_vec'], dtype=np.float64)
        ry = np.asarray(raw['y_vec'], dtype=np.float64)
        rvx = np.asarray(raw['vx_vec'], dtype=np.float64)
        rvy = np.asarray(raw['vy_vec'], dtype=np.float64)
        rpsi = np.asarray(raw['psi_vec'], dtype=np.float64)
        frame_of_raw = np.round(t / SRC_DT).astype(np.int64)
        k0 = int(frame_of_raw[0])
        entries.sort(key=lambda e: float(e[2]))
        ts = np.array([float(e[2]) for e in entries])
        frames = np.round(ts / SRC_DT).astype(np.int64)
        local = frames - k0
        if np.any(local < 0) or np.any(local >= len(t)):
            stride_bad += len(frames)
            continue
        if np.any(np.abs(t[local] - ts) > EXACT_TOL_S):
            timestamp_bad += 1
        if np.any(frames % STRIDE != 0):
            frame_phase_bad += 1
        step = np.diff(frames)
        for value in step.tolist():
            stride_counts[int(value)] += 1
        if np.any(step != STRIDE):
            stride_bad += 1
        if np.any(step <= 0):
            no_reuse += 1
        for frame, timestamp in zip(frames.tolist(), ts.tolist()):
            phase_groups[(scene_id, timestamp)].add(int(frame))
        if (np.abs(rx[local] - np.array([e[3] for e in entries])) > POS_TOL_M).any() \
                or (np.abs(ry[local] - np.array([e[4] for e in entries])) > POS_TOL_M).any():
            copy_bad += 1
        wx = np.cos(rpsi[local]) * rvx[local] - np.sin(rpsi[local]) * rvy[local]
        wy = np.sin(rpsi[local]) * rvx[local] + np.cos(rpsi[local]) * rvy[local]
        if (np.abs(wx - np.array([e[5] for e in entries])) > VEL_TOL_MPS).any() \
                or (np.abs(wy - np.array([e[6] for e in entries])) > VEL_TOL_MPS).any():
            rotation_bad += 1
        if (np.abs(np.hypot(rvx[local], rvy[local]) - np.hypot(wx, wy)) > VEL_TOL_MPS).any():
            norm_bad += 1
        if len(entries) >= 2:
            cx = np.array([e[3] for e in entries])
            cy = np.array([e[4] for e in entries])
            cvx = np.array([e[5] for e in entries])
            cvy = np.array([e[6] for e in entries])
            dt = np.diff(ts)
            fd_vx, fd_vy = np.diff(cx) / dt, np.diff(cy) / dt
            mid_vx = 0.5 * (cvx[:-1] + cvx[1:])
            mid_vy = 0.5 * (cvy[:-1] + cvy[1:])
            fd_err_vx.append(fd_vx - mid_vx)
            fd_err_vy.append(fd_vy - mid_vy)
            moving = np.hypot(mid_vx, mid_vy) > 1.0
            if moving.any():
                cross = fd_vx[moving] * mid_vy[moving] - fd_vy[moving] * mid_vx[moving]
                dot = fd_vx[moving] * mid_vx[moving] + fd_vy[moving] * mid_vy[moving]
                dir_err.append(np.degrees(np.abs(np.arctan2(cross, dot))))

    record('fixed_stride_3', stride_bad == 0,
           {'pairs_by_stride': {str(k): int(v) for k, v in sorted(stride_counts.items())},
            'violating_tracks': stride_bad})
    record('no_stride_2', stride_counts.get(2, 0) == 0, {'stride_2_pairs': int(stride_counts.get(2, 0))})
    record('no_stride_4', stride_counts.get(4, 0) == 0, {'stride_4_pairs': int(stride_counts.get(4, 0))})
    record('true_timestamp_preserved', timestamp_bad == 0,
           {'tracks_with_non_source_timestamp': timestamp_bad, 'tolerance_s': EXACT_TOL_S})
    record('scene_global_phase_consistent',
           all(len(frames) == 1 for frames in phase_groups.values()),
           {'checked_timestamps': len(phase_groups),
            'violations': int(sum(1 for frames in phase_groups.values() if len(frames) != 1))})
    record('all_frames_on_global_phase', frame_phase_bad == 0,
           {'tracks_with_frame_not_multiple_of_3': frame_phase_bad})
    record('no_source_frame_reuse', no_reuse == 0, {'violating_tracks': no_reuse})
    record('positions_copied_from_source', copy_bad == 0, {'violating_tracks': copy_bad})
    record('rotation_reproduces_world_velocity', rotation_bad == 0, {'violating_tracks': rotation_bad})
    record('speed_norm_preserved', norm_bad == 0, {'violating_tracks': norm_bad,
                                                  'tolerance_mps': VEL_TOL_MPS})
    record('mapping_traceable_to_raw', mapping_bad == 0, {'mismatches': mapping_bad})

    fd_err_vx = np.concatenate(fd_err_vx)
    fd_err_vy = np.concatenate(fd_err_vy)
    dir_err = np.concatenate(dir_err)
    dir_q = np.percentile(dir_err, [50, 95, 99, 100])
    max_abs = max(float(np.abs(fd_err_vx).max()), float(np.abs(fd_err_vy).max()))
    record('velocity_fd_no_grid_label_artifact', bool(max_abs < 1.0),
           {'max_abs_error_mps': max_abs, 'threshold_mps': 1.0,
            'vx_mae_mps': float(np.abs(fd_err_vx).mean()),
            'vy_mae_mps': float(np.abs(fd_err_vy).mean())})
    record('velocity_fd_consistency',
           bool(np.abs(fd_err_vx).mean() < 0.5 and np.abs(fd_err_vy).mean() < 0.5 and dir_q[1] < 10.0),
           {'vx_mae_mps': float(np.abs(fd_err_vx).mean()), 'vy_mae_mps': float(np.abs(fd_err_vy).mean()),
            'vx_rmse_mps': float(np.sqrt(np.mean(fd_err_vx ** 2))),
            'vy_rmse_mps': float(np.sqrt(np.mean(fd_err_vy ** 2))),
            'vx_p95_abs_mps': float(np.percentile(np.abs(fd_err_vx), 95)),
            'vy_p95_abs_mps': float(np.percentile(np.abs(fd_err_vy), 95)),
            'vx_p99_abs_mps': float(np.percentile(np.abs(fd_err_vx), 99)),
            'vy_p99_abs_mps': float(np.percentile(np.abs(fd_err_vy), 99)),
            'direction_median_deg': float(dir_q[0]), 'direction_p95_deg': float(dir_q[1]),
            'direction_p99_deg': float(dir_q[2]), 'direction_max_deg': float(dir_q[3]),
            'pairs': int(len(fd_err_vx))})

    order_ok = True
    for a, b in zip(rows, rows[1:]):
        if (b[0], float(b[2]), b[1]) < (a[0], float(a[2]), a[1]):
            order_ok = False
            break
    record('file_sorted_scene_time_vehicle', order_ok, {'sorted': order_ok})

    metadata = json.loads((processed / 'scene_metadata.json').read_text(encoding='utf-8'))
    meta_ok = True
    meta_detail = {}
    for scene_id in (0, 1):
        meta = metadata['scenes'][str(scene_id)]
        mask = scene_ids == scene_id
        count = int(mask.sum())
        ts_min = float(timestamps[mask].min())
        ts_max = float(timestamps[mask].max())
        vehicles = len(set(vehicle_ids[mask].tolist()))
        scene_ok = (meta['canonical_rows'] == count and meta['num_vehicles'] == vehicles
                    and abs(meta['canonical_timestamp_range_s'][0] - ts_min) < EPS_S
                    and abs(meta['canonical_timestamp_range_s'][1] - ts_max) < EPS_S
                    and abs(meta['canonical_fps'] - RAW_FPS / STRIDE) < 1e-9
                    and int(meta['stride']) == STRIDE)
        meta_ok = meta_ok and scene_ok
        meta_detail[str(scene_id)] = {'rows': count, 'vehicles': vehicles,
                                      'timestamp_range': [ts_min, ts_max], 'ok': scene_ok}
    record('scene_metadata_consistent', meta_ok, meta_detail)

    window_counts = {}
    for scene_id in (0, 1):
        counts = defaultdict(int)
        for row in rows:
            if row[0] == scene_id:
                counts[row[1]] += 1
        window_counts[scene_id] = sum(1 for v in counts.values() if v >= WINDOW_SAMPLES)
    record('prediction_window_supply_match',
           window_counts[0] == stats['prediction_window_supply']['scene_0']
           and window_counts[1] == stats['prediction_window_supply']['scene_1'],
           {'scene_0': window_counts[0], 'scene_1': window_counts[1],
            'stats': [stats['prediction_window_supply']['scene_0'],
                      stats['prediction_window_supply']['scene_1']]})

    for name in ('trajectories_10hz.csv', 'vehicle_id_mapping.csv', 'scene_metadata.json'):
        actual = sha256_file(processed / name)
        record('sha256_%s' % name.replace('.', '_'), actual == stats['outputs'][name]['sha256'],
               {'actual': actual, 'stats': stats['outputs'][name]['sha256']})

    failed = [name for name, entry in checks.items() if not entry['pass']]
    payload = {
        'dataset': 'automatum_t_crossing_canonical',
        'validator': 'tools/data_preprocessing/validate_automatum_canonical.py',
        'sampling': {'source_fps': RAW_FPS, 'stride': STRIDE, 'canonical_fps': RAW_FPS / STRIDE,
                     'canonical_dt_s': CANON_DT,
                     'timestamp_rule': 'true source time frame / 29.97'},
        'inputs': {'trajectories_10hz.csv': (processed / 'trajectories_10hz.csv').name,
                   'stats': stats_path.name},
        'checks': checks,
        'summary': {'checks_total': len(checks), 'checks_passed': len(checks) - len(failed),
                    'checks_failed': len(failed), 'failed_checks': failed,
                    'verdict': 'PASS' if not failed else 'FAIL'},
        'velocity_fd_validation': {
            'vx_mae_mps': float(np.abs(fd_err_vx).mean()), 'vy_mae_mps': float(np.abs(fd_err_vy).mean()),
            'vx_rmse_mps': float(np.sqrt(np.mean(fd_err_vx ** 2))),
            'vy_rmse_mps': float(np.sqrt(np.mean(fd_err_vy ** 2))),
            'vx_p95_abs_mps': float(np.percentile(np.abs(fd_err_vx), 95)),
            'vy_p95_abs_mps': float(np.percentile(np.abs(fd_err_vy), 95)),
            'vx_p99_abs_mps': float(np.percentile(np.abs(fd_err_vx), 99)),
            'vy_p99_abs_mps': float(np.percentile(np.abs(fd_err_vy), 99)),
            'vx_max_abs_mps': float(np.abs(fd_err_vx).max()),
            'vy_max_abs_mps': float(np.abs(fd_err_vy).max()),
            'direction_median_deg': float(dir_q[0]), 'direction_p95_deg': float(dir_q[1]),
            'direction_p99_deg': float(dir_q[2]), 'direction_max_deg': float(dir_q[3]),
            'pairs': int(len(fd_err_vx))},
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