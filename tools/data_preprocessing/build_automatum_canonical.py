#!/usr/bin/env python3
"""Build the canonical 10 Hz Automatum T-Crossing trajectory dataset.

Deterministic and read-only on the raw recordings. One CSV per requirement:

    data/automatum_t_crossing/processed/trajectories_10hz.csv
    data/automatum_t_crossing/processed/vehicle_id_mapping.csv
    data/automatum_t_crossing/processed/scene_metadata.json

Rules (frozen by the preprocessing order, see reports/data_preprocessing/):
- two scenes stay separate: scene_id 0 = Gaimersheim Stadtweg, 1 = St2214 Duenzlau
- 29.97 Hz -> 10 Hz by nearest 0.1 s grid sample, no interpolation, no smoothing
- only source samples within half a source frame (0.5/29.97 s) are eligible
- stored timestamp is the target grid time, not the source time
- vx/vy are rotated from body frame to world frame with the raw continuous psi
- x/y stay in the official local metric frame, no shifting/scaling/normalisation
- all 683 vehicles kept (car/van/truck), no ROI, no N<=8 filtering, no split

Usage:
    python tools/data_preprocessing/build_automatum_canonical.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RAW_FPS = 29.97
SRC_DT = 1.0 / RAW_FPS
HALF_FRAME_S = 0.5 * SRC_DT
CANON_DT = 0.1
TIME_EPS_S = 1e-9
POS_TOL_M = 2e-6
VEL_TOL_MPS = 2e-6
JUMP_LIMIT_MPS = 45.0
FOUR_S = 4.0
AUDIT_COMMIT = 'f03e230d094abb0a3f268c2b7df9db7c6c675552'
AUDIT_REPORT = 'reports/data_audit/automatum_t_crossing_audit.md'
SCENES = [
    {'scene_id': 0, 'name': 'T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2',
     'label': 'Gaimersheim Stadtweg'},
    {'scene_id': 1, 'name': 'T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f',
     'label': 'St2214 Duenzlau Umgehung'},
]
START = time.time()


def log(*args) -> None:
    print('[%6.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_recording_dir(raw_root: Path, name: str) -> Path:
    candidates = [raw_root / name, raw_root / 'extracted' / name]
    for candidate in candidates:
        if (candidate / 'dynamicWorld.json').is_file():
            return candidate
    for world in raw_root.rglob('dynamicWorld.json'):
        if world.parent.name == name:
            return world.parent
    raise FileNotFoundError('recording dir not found for %s under %s' % (name, raw_root))


def qstats_ms(values_s: np.ndarray) -> dict:
    values_ms = np.asarray(values_s, dtype=np.float64) * 1000.0
    if values_ms.size == 0:
        return {'count': 0}
    q = np.percentile(values_ms, [50, 95, 99, 100])
    return {'count': int(values_ms.size), 'median_ms': float(q[0]), 'p95_ms': float(q[1]),
            'p99_ms': float(q[2]), 'max_ms': float(q[3]), 'mean_ms': float(values_ms.mean())}


def load_scene(raw_dir: Path, spec: dict) -> dict:
    world = json.loads((raw_dir / 'dynamicWorld.json').read_text(encoding='utf-8'))
    objects = []
    for index, raw in enumerate(world['objects']):
        objects.append({
            'index': index, 'uuid': raw['UUID'], 'obj_type': raw['objType'],
            'length': float(raw['length']), 'width': float(raw['width']),
            't': np.asarray(raw['time'], dtype=np.float64),
            'x': np.asarray(raw['x_vec'], dtype=np.float64),
            'y': np.asarray(raw['y_vec'], dtype=np.float64),
            'psi': np.asarray(raw['psi_vec'], dtype=np.float64),
            'vx': np.asarray(raw['vx_vec'], dtype=np.float64),
            'vy': np.asarray(raw['vy_vec'], dtype=np.float64),
        })
    return {'spec': spec, 'dir': raw_dir, 'world': world, 'objects': objects}


def resample_scene(scene: dict) -> dict:
    """Nearest 0.1 s grid selection over every track. Returns per-track selection."""
    objects = scene['objects']
    t_last = max(float(o['t'][-1]) for o in objects)
    n_targets = int(math.floor(t_last / CANON_DT + TIME_EPS_S)) + 1
    targets = np.arange(n_targets, dtype=np.float64) * CANON_DT
    global_index = np.round(targets / SRC_DT).astype(np.int64)
    grid_error = np.abs(global_index * SRC_DT - targets)
    assert float(grid_error.max()) <= HALF_FRAME_S + TIME_EPS_S, 'grid error bound violated'
    tracks = []
    for obj in objects:
        n = len(obj['t'])
        k0 = int(round(float(obj['t'][0]) / SRC_DT))
        k_last = int(round(float(obj['t'][-1]) / SRC_DT))
        assert abs(obj['t'][0] - k0 * SRC_DT) < TIME_EPS_S, 'track start not on source grid'
        assert abs(obj['t'][-1] - k_last * SRC_DT) < TIME_EPS_S, 'track end not on source grid'
        assert k_last == k0 + n - 1, 'track is not contiguous on the source grid'
        local = global_index - k0
        valid = (local >= 0) & (local < n)
        local = local[valid]
        err = np.abs(obj['t'][local] - targets[valid])
        assert float(err.max()) <= HALF_FRAME_S + TIME_EPS_S, 'half-frame rule violated'
        psi = obj['psi'][local]
        body_vx = obj['vx'][local]
        body_vy = obj['vy'][local]
        world_vx = np.cos(psi) * body_vx - np.sin(psi) * body_vy
        world_vy = np.sin(psi) * body_vx + np.cos(psi) * body_vy
        norm_error = np.abs(np.hypot(body_vx, body_vy) - np.hypot(world_vx, world_vy))
        tracks.append({
            'uuid': obj['uuid'], 'obj_type': obj['obj_type'], 'length': obj['length'],
            'width': obj['width'], 'n_raw': n,
            'raw_t0': float(obj['t'][0]), 'raw_t1': float(obj['t'][-1]),
            'target_k': np.nonzero(valid)[0].astype(np.int64),
            'target_t': targets[valid], 'source_index': local.astype(np.int64),
            'source_t': obj['t'][local],
            'time_error_s': err, 'x': obj['x'][local], 'y': obj['y'][local],
            'vx_world': world_vx, 'vy_world': world_vy,
            'body_speed': np.hypot(body_vx, body_vy),
            'world_speed': np.hypot(world_vx, world_vy),
            'norm_error': norm_error,
        })
    return {'scene': scene, 'tracks': tracks, 'targets': targets,
            'global_index': global_index, 'grid_error': grid_error}


def assign_vehicle_ids(resampled: list) -> list:
    keys = []
    for item in resampled:
        scene_id = item['scene']['spec']['scene_id']
        for track in item['tracks']:
            keys.append((scene_id, float(track['target_t'][0]), track['uuid'], track))
    keys.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    mapping = []
    for vehicle_id, (scene_id, _first_t, uuid, track) in enumerate(keys, start=1):
        track['vehicle_id'] = vehicle_id
        mapping.append({'scene_id': scene_id, 'vehicle_id': vehicle_id, 'original_uuid': uuid,
                        'obj_type': track['obj_type'], 'length': track['length'],
                        'width': track['width'], 'original_first_time': track['raw_t0'],
                        'original_last_time': track['raw_t1']})
    return mapping


def rows_of(resampled: list) -> list:
    rows = []
    for item in resampled:
        scene_id = item['scene']['spec']['scene_id']
        for track in item['tracks']:
            vehicle_id = track['vehicle_id']
            for i in range(len(track['target_k'])):
                rows.append((scene_id, int(track['target_k'][i]), vehicle_id,
                             float(track['x'][i]), float(track['y'][i]),
                             float(track['vx_world'][i]), float(track['vy_world'][i])))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return rows


def _error_block(err_vx, err_vy, direction) -> dict:
    def block(values):
        abs_values = np.abs(values)
        q = np.percentile(abs_values, [50, 95, 99])
        return {'mae_mps': float(abs_values.mean()), 'rmse_mps': float(np.sqrt(np.mean(values ** 2))),
                'bias_mps': float(values.mean()), 'median_abs_mps': float(q[0]),
                'p95_abs_mps': float(q[1]), 'p99_abs_mps': float(q[2]),
                'max_abs_mps': float(abs_values.max())}
    dir_q = np.percentile(direction, [50, 95, 99, 100]) if direction.size else [float('nan')] * 4
    return {'vx': block(err_vx), 'vy': block(err_vy),
            'direction_error_deg': {'median': float(dir_q[0]), 'p95': float(dir_q[1]),
                                    'p99': float(dir_q[2]), 'max': float(dir_q[3]),
                                    'moving_points': int(direction.size)},
            'pairs_total': int(len(err_vx))}


def velocity_validation(tracks: list) -> dict:
    grid_vx, grid_vy, grid_dir = [], [], []
    true_vx, true_vy, true_dir = [], [], []
    strides = []
    for track in tracks:
        x, y = track['x'], track['y']
        if len(x) < 2:
            continue
        step = np.diff(track['source_index'])
        strides.append(step)
        delta_x, delta_y = np.diff(x), np.diff(y)
        mid_vx = 0.5 * (track['vx_world'][:-1] + track['vx_world'][1:])
        mid_vy = 0.5 * (track['vy_world'][:-1] + track['vy_world'][1:])
        fd_grid_vx, fd_grid_vy = delta_x / CANON_DT, delta_y / CANON_DT
        dt_source = np.diff(track['source_t'])
        fd_true_vx, fd_true_vy = delta_x / dt_source, delta_y / dt_source
        grid_vx.append(fd_grid_vx - mid_vx)
        grid_vy.append(fd_grid_vy - mid_vy)
        true_vx.append(fd_true_vx - mid_vx)
        true_vy.append(fd_true_vy - mid_vy)
        moving = np.hypot(mid_vx, mid_vy) > 1.0
        if moving.any():
            for fd_vx, fd_vy, bucket in ((fd_grid_vx, fd_grid_vy, grid_dir),
                                         (fd_true_vx, fd_true_vy, true_dir)):
                cross = fd_vx[moving] * mid_vy[moving] - fd_vy[moving] * mid_vx[moving]
                dot = fd_vx[moving] * mid_vx[moving] + fd_vy[moving] * mid_vy[moving]
                bucket.append(np.degrees(np.abs(np.arctan2(cross, dot))))
    strides = np.concatenate(strides)
    values, counts = np.unique(strides, return_counts=True)
    return {
        'grid_interval_chord': _error_block(np.concatenate(grid_vx), np.concatenate(grid_vy),
                                            np.concatenate(grid_dir) if grid_dir else np.array([])),
        'true_source_interval': _error_block(np.concatenate(true_vx), np.concatenate(true_vy),
                                             np.concatenate(true_dir) if true_dir else np.array([])),
        'source_stride_counts': {str(int(v)): int(c) for v, c in zip(values, counts)},
        'note': 'grid_interval_chord divides by the 0.1 s label (as stored); true_source_interval '
                'divides by the real source time delta and is the rotation-correctness check',
    }


def scene_stats(item: dict) -> dict:
    spec = item['scene']['spec']
    world = item['scene']['world']
    tracks = item['tracks']
    all_err = np.concatenate([t['time_error_s'] for t in tracks])
    all_x = np.concatenate([t['x'] for t in tracks])
    all_y = np.concatenate([t['y'] for t in tracks])
    all_norm = np.concatenate([t['norm_error'] for t in tracks])
    durations = np.array([float(t['target_t'][-1] - t['target_t'][0]) for t in tracks])
    dist_bins = {'<2s': int((durations < 2).sum()), '2-4s': int(((durations >= 2) & (durations < 4)).sum()),
                 '4-6s': int(((durations >= 4) & (durations < 6)).sum()),
                 '6-10s': int(((durations >= 6) & (durations < 10)).sum()),
                 '10-20s': int(((durations >= 10) & (durations < 20)).sum()),
                 '>20s': int((durations >= 20).sum())}
    raw_points = int(sum(t['n_raw'] for t in tracks))
    canonical_points = int(sum(len(t['target_k']) for t in tracks))
    used_first = float(min(t['target_t'][0] for t in tracks))
    used_last = float(max(t['target_t'][-1] for t in tracks))
    strides = np.concatenate([np.diff(t['source_index']) for t in tracks if len(t['source_index']) > 1])
    stride_values, stride_counts = np.unique(strides, return_counts=True)
    raw_last = max(float(o['t'][-1]) for o in item['scene']['objects'])
    raw_first = min(float(o['t'][0]) for o in item['scene']['objects'])
    return {
        'scene_id': spec['scene_id'], 'scene_name': spec['name'], 'scene_label': spec['label'],
        'recording_name': world['RecordingName'], 'recording_uuid': world['UUID'],
        'source_dir': item['scene']['dir'].name,
        'raw_fps': RAW_FPS, 'canonical_fps': 1.0 / CANON_DT,
        'raw_points': raw_points, 'canonical_points': canonical_points,
        'raw_vehicles': len(tracks), 'canonical_vehicles': len(tracks),
        'raw_duration_s': raw_last - raw_first,
        'raw_time_range_s': [raw_first, raw_last],
        'canonical_timestamp_range_s': [used_first, used_last],
        'canonical_grid_range_s': [0.0, float(item['targets'][-1])],
        'canonical_x_range_m': [float(all_x.min()), float(all_x.max())],
        'canonical_y_range_m': [float(all_y.min()), float(all_y.max())],
        'time_mapping_error_ms': qstats_ms(all_err),
        'source_stride_counts': {str(int(v)): int(c) for v, c in zip(stride_values, stride_counts)},
        'track_duration_bins': dist_bins,
        'tracks_ge_4s': int((durations >= FOUR_S).sum()),
        'tracks_with_ge_40_canonical_points': int(sum(1 for t in tracks if len(t['target_k']) >= 40)),
        'norm_error_max_mps': float(all_norm.max()),
        'utm_reference': world['UTM-ReferencePoint'], 'wgs84_reference': world['WGS84-ReferencePoint'],
        'static_world_xodr': str((item['scene']['dir'] / 'staticWorld.xodr')),
        'raw_file_sha256': {
            'dynamicWorld.json': sha256_file(item['scene']['dir'] / 'dynamicWorld.json'),
            'staticWorld.xodr': sha256_file(item['scene']['dir'] / 'staticWorld.xodr'),
        },
    }


def write_trajectories(rows: list, path: Path) -> dict:
    with path.open('w', encoding='utf-8', newline='') as fh:
        fh.write('scene_id,vehicle_id,timestamp,x,y,vx,vy\n')
        writer = csv.writer(fh, lineterminator='\n')
        for scene_id, k, vehicle_id, x, y, vx, vy in rows:
            writer.writerow([scene_id, vehicle_id, '%.1f' % (k * CANON_DT),
                             '%.6f' % x, '%.6f' % y, '%.6f' % vx, '%.6f' % vy])
    return {'rows': len(rows), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def write_mapping(mapping: list, path: Path) -> dict:
    fields = ['scene_id', 'vehicle_id', 'original_uuid', 'obj_type', 'length', 'width',
              'original_first_time', 'original_last_time']
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(fields)
        for entry in sorted(mapping, key=lambda e: e['vehicle_id']):
            writer.writerow([entry['scene_id'], entry['vehicle_id'], entry['original_uuid'],
                             entry['obj_type'], '%.6f' % entry['length'], '%.6f' % entry['width'],
                             '%.6f' % entry['original_first_time'], '%.6f' % entry['original_last_time']])
    return {'rows': len(mapping), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def write_json(payload: dict, path: Path) -> dict:
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    with path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(text + '\n')
    return {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the canonical 10 Hz Automatum T-Crossing dataset')
    parser.add_argument('--raw-dir', default='data/automatum_t_crossing/raw')
    parser.add_argument('--out-dir', default='data/automatum_t_crossing/processed')
    parser.add_argument('--stats-out', default='reports/data_preprocessing/automatum_canonical_stats.json')
    args = parser.parse_args()

    raw_root = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    stats_path = Path(args.stats_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    resampled = []
    for spec in SCENES:
        raw_dir = find_recording_dir(raw_root, spec['name'])
        log('loading scene %d from %s' % (spec['scene_id'], raw_dir))
        scene = load_scene(raw_dir, spec)
        log('resampling scene %d: %d tracks' % (spec['scene_id'], len(scene['objects'])))
        resampled.append(resample_scene(scene))

    mapping = assign_vehicle_ids(resampled)
    assert len(mapping) == 683, 'expected 683 vehicles, got %d' % len(mapping)
    rows = rows_of(resampled)
    log('canonical rows:', len(rows))

    scene_meta = {}
    scene_stats_all = {}
    for item, spec in zip(resampled, SCENES):
        stats = scene_stats(item)
        scene_stats_all[str(spec['scene_id'])] = stats
        scene_meta[str(spec['scene_id'])] = {
            'scene_id': stats['scene_id'], 'scene_name': stats['scene_name'],
            'scene_label': stats['scene_label'], 'recording_name': stats['recording_name'],
            'recording_uuid': stats['recording_uuid'], 'source_dir': stats['source_dir'],
            'recording_duration_raw_s': stats['raw_duration_s'],
            'raw_fps': stats['raw_fps'], 'canonical_fps': stats['canonical_fps'],
            'num_vehicles': stats['canonical_vehicles'],
            'x_range_m': stats['canonical_x_range_m'], 'y_range_m': stats['canonical_y_range_m'],
            'UTM_reference': stats['utm_reference'], 'WGS84_reference': stats['wgs84_reference'],
            'static_world_xodr': stats['static_world_xodr'],
            'canonical_rows': stats['canonical_points'],
            'canonical_timestamp_range_s': stats['canonical_timestamp_range_s'],
            'time_mapping_error_ms': stats['time_mapping_error_ms'],
            'track_duration_bins': stats['track_duration_bins'],
            'tracks_ge_4s': stats['tracks_ge_4s'],
            'raw_file_sha256': stats['raw_file_sha256'],
        }

    metrics = {}
    metrics['trajectories_10hz.csv'] = write_trajectories(rows, out_dir / 'trajectories_10hz.csv')
    metrics['vehicle_id_mapping.csv'] = write_mapping(mapping, out_dir / 'vehicle_id_mapping.csv')
    metadata_payload = {
        'dataset': 'automatum_t_crossing_canonical',
        'canonical_fps': 1.0 / CANON_DT,
        'columns': ['scene_id', 'vehicle_id', 'timestamp', 'x', 'y', 'vx', 'vy'],
        'units': {'timestamp': 's', 'x': 'm', 'y': 'm', 'vx': 'm/s', 'vy': 'm/s'},
        'velocity_frame': 'world (local metric scene frame, same as x/y and staticWorld.xodr)',
        'scenes': scene_meta,
    }
    meta_metric = write_json(metadata_payload, out_dir / 'scene_metadata.json')
    metrics['scene_metadata.json'] = dict(meta_metric, rows=len(scene_meta))

    per_scene_tracks = {str(item['scene']['spec']['scene_id']): item['tracks'] for item in resampled}
    vv = {}
    for sid, tracks in per_scene_tracks.items():
        vv[sid] = velocity_validation(tracks)
    vv['all_scenes'] = velocity_validation([t for tracks in per_scene_tracks.values() for t in tracks])

    stats = {
        'dataset': 'automatum_t_crossing_canonical',
        'mode': 'deterministic build; raw recordings read-only',
        'provenance': {'audit_commit': AUDIT_COMMIT, 'audit_report': AUDIT_REPORT,
                       'builder': 'tools/data_preprocessing/build_automatum_canonical.py'},
        'rules': {
            'source_fps': RAW_FPS, 'canonical_fps': 1.0 / CANON_DT,
            'downsampling': 'nearest 0.1 s grid sample, no interpolation, no smoothing',
            'max_time_error_s': HALF_FRAME_S,
            'timestamp_written': 'target 0.1 s grid time',
            'velocity': 'world frame: vx_w = cos(psi)*vx_body - sin(psi)*vy_body, '
                        'vy_w = sin(psi)*vx_body + cos(psi)*vy_body',
            'coordinates': 'official local metric frame, unchanged',
            'vehicle_id': 'sorted by (scene_id, first canonical timestamp, original UUID)',
        },
        'outputs': metrics,
        'scenes': scene_stats_all,
        'totals': {
            'scenes': len(SCENES),
            'vehicles': len(mapping),
            'rows': len(rows),
            'raw_points': int(sum(s['raw_points'] for s in scene_stats_all.values())),
            'canonical_bytes': int(sum(m['bytes'] for m in metrics.values())),
        },
        'time_mapping_error_ms': {
            'all_scenes': qstats_ms(np.concatenate([t['time_error_s']
                                                    for item in resampled for t in item['tracks']])),
            'half_source_frame_ms': HALF_FRAME_S * 1000.0,
        },
        'velocity_rotation_validation': vv,
        'four_second_trajectories': {
            'definition': 'canonical tracks with last - first timestamp >= 4.0 s',
            'scene_0': scene_stats_all['0']['tracks_ge_4s'],
            'scene_1': scene_stats_all['1']['tracks_ge_4s'],
            'total': scene_stats_all['0']['tracks_ge_4s'] + scene_stats_all['1']['tracks_ge_4s'],
        },
    }
    stats_metric = write_json(stats, stats_path)
    log('wrote', stats_path)
    print(json.dumps({'trajectories': metrics['trajectories_10hz.csv'],
                      'mapping': metrics['vehicle_id_mapping.csv'],
                      'metadata': metrics['scene_metadata.json'],
                      'stats': stats_metric,
                      'scenes': {k: {'rows': v['canonical_points'], 'vehicles': v['canonical_vehicles'],
                                     'tracks_ge_4s': v['tracks_ge_4s']}
                                 for k, v in scene_stats_all.items()}},
                     indent=1, default=float))


if __name__ == '__main__':
    main()