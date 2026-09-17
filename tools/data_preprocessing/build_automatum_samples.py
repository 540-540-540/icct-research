#!/usr/bin/env python3
"""Build deterministic 20+20 multi-vehicle prediction samples from the splits.

Frozen rules:
- window = 40 consecutive canonical frames (history 20 + future 20), stride 1
- eligible vehicle = present in all 40 frames of the window
- keep window iff 2 <= N <= 8, reject whole window when N > 8 (no vehicle picking)
- vehicles ordered by vehicle_id ascending, padded to 8 nodes:
  history/future 0, vehicle_ids -1, mask 0
- state = [x, y, vx, vy] straight from trajectories.csv (float32 storage)

Outputs per split (data/ is git-ignored):
    data/automatum_t_crossing/splits/<split>/trajectories.csv   (byte copy, migrated)
    data/automatum_t_crossing/splits/<split>/samples.npz
    data/automatum_t_crossing/splits/<split>/sample_manifest.json
plus git mirrors under reports/data_preprocessing/sample_manifests/ and a stats
file. The npz writer pins ZipInfo timestamps and uses ZIP_STORED so the file is
byte-reproducible across machines.

Usage:
    python tools/data_preprocessing/build_automatum_samples.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

FPS = 29.97 / 3
SOURCE_FPS = 29.97
STRIDE = 3
HISTORY_FRAMES = 20
FUTURE_FRAMES = 20
TOTAL_FRAMES = HISTORY_FRAMES + FUTURE_FRAMES
WINDOW_STRIDE = 1
MIN_VEHICLES = 2
MAX_VEHICLES = 8
MAX_NODES = 8
SPLIT_SHA256 = {
    'train': 'd75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11',
    'val': 'c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d',
    'test': '6fb3117e95368e2c5b66cca5a28cb45a1ff91691d4afea4fb9b8c1a17eb75371',
}
NPZ_KEYS = ['history', 'future', 'vehicle_ids', 'vehicle_mask', 'num_vehicles',
            'scene_id', 'start_frame', 'start_timestamp']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene_tables(path: Path) -> dict:
    """Parse trajectories.csv into per-scene dense grids indexed [vehicle, frame]."""
    scenes = {}
    with path.open(encoding='utf-8', newline='') as fh:
        header = fh.readline().rstrip('\n')
        assert header == 'scene_id,vehicle_id,timestamp,x,y,vx,vy', 'unexpected header'
        rows = {0: [], 1: []}
        for line in fh:
            parts = line.rstrip('\n').split(',')
            scene = int(parts[0])
            vehicle = int(parts[1])
            timestamp = float(parts[2])
            source_frame = int(round(timestamp * SOURCE_FPS))
            assert source_frame % STRIDE == 0, 'timestamp not on the stride-3 phase'
            assert abs(source_frame / SOURCE_FPS - timestamp) < 1e-9, 'frame roundtrip error'
            rows[scene].append((vehicle, source_frame // STRIDE, timestamp,
                                float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])))
    for scene_id in (0, 1):
        entries = rows[scene_id]
        assert entries, 'scene %d has no rows' % scene_id
        vehicles = np.array(sorted({e[0] for e in entries}), dtype=np.int64)
        frames = np.array([e[1] for e in entries], dtype=np.int64)
        f_first, f_last = int(frames.min()), int(frames.max())
        grid = np.full((len(vehicles), f_last - f_first + 1, 4), np.nan)
        timestamps = np.full(f_last - f_first + 1, np.nan)
        vidx = np.searchsorted(vehicles, np.array([e[0] for e in entries]))
        for i, entry in enumerate(entries):
            j = int(entry[1] - f_first)
            assert np.isnan(grid[vidx[i], j, 0]), 'duplicate vehicle/frame state'
            grid[vidx[i], j] = entry[3:7]
            if np.isnan(timestamps[j]):
                timestamps[j] = entry[2]
            else:
                assert timestamps[j] == entry[2], 'timestamp differs within a frame'
        v_first = np.empty(len(vehicles), dtype=np.int64)
        v_last = np.empty(len(vehicles), dtype=np.int64)
        for k in range(len(vehicles)):
            present = np.nonzero(~np.isnan(grid[k, :, 0]))[0]
            assert present[-1] - present[0] + 1 == len(present), 'vehicle frames are not contiguous'
            v_first[k] = present[0] + f_first
            v_last[k] = present[-1] + f_first
        scenes[scene_id] = {'vehicles': vehicles, 'grid': grid, 'timestamps': timestamps,
                            'f_first': f_first, 'f_last': f_last,
                            'v_first': v_first, 'v_last': v_last}
    return scenes


def build_split(split: str, path: Path) -> dict:
    scenes = load_scene_tables(path)
    hist, fut, ids, mask, nums, scene_ids, starts, start_ts = [], [], [], [], [], [], [], []
    per_scene = {0: 0, 1: 0}
    rejected = 0
    n_dist = {str(n): 0 for n in range(MIN_VEHICLES, MAX_VEHICLES + 1)}
    for scene_id in (0, 1):
        scene = scenes[scene_id]
        grid = scene['grid']
        v_first, v_last = scene['v_first'], scene['v_last']
        f_first, f_last = scene['f_first'], scene['f_last']
        for f in range(f_first, f_last - TOTAL_FRAMES + 2):
            eligible = np.nonzero((v_first <= f) & (v_last >= f + TOTAL_FRAMES - 1))[0]
            n = int(eligible.size)
            if n < MIN_VEHICLES:
                continue
            if n > MAX_VEHICLES:
                rejected += 1
                continue
            offset = f - f_first
            states = grid[eligible, offset:offset + TOTAL_FRAMES, :]  # (N, 40, 4)
            h = np.zeros((HISTORY_FRAMES, MAX_NODES, 4), np.float32)
            u = np.zeros((FUTURE_FRAMES, MAX_NODES, 4), np.float32)
            h[:, :n, :] = np.transpose(states[:, :HISTORY_FRAMES, :], (1, 0, 2)).astype(np.float32)
            u[:, :n, :] = np.transpose(states[:, HISTORY_FRAMES:, :], (1, 0, 2)).astype(np.float32)
            vids = np.full(MAX_NODES, -1, np.int32)
            vids[:n] = scene['vehicles'][eligible]
            m = np.zeros(MAX_NODES, np.bool_)
            m[:n] = True
            hist.append(h)
            fut.append(u)
            ids.append(vids)
            mask.append(m)
            nums.append(n)
            scene_ids.append(scene_id)
            starts.append(f)
            start_ts.append(scene['timestamps'][offset])
            per_scene[scene_id] += 1
            n_dist[str(n)] += 1
    arrays = {
        'history': np.stack(hist).astype(np.float32) if hist else np.zeros((0, HISTORY_FRAMES, MAX_NODES, 4), np.float32),
        'future': np.stack(fut).astype(np.float32) if fut else np.zeros((0, FUTURE_FRAMES, MAX_NODES, 4), np.float32),
        'vehicle_ids': np.stack(ids).astype(np.int32) if ids else np.zeros((0, MAX_NODES), np.int32),
        'vehicle_mask': np.stack(mask).astype(np.bool_) if mask else np.zeros((0, MAX_NODES), np.bool_),
        'num_vehicles': np.array(nums, np.uint8),
        'scene_id': np.array(scene_ids, np.uint8),
        'start_frame': np.array(starts, np.int32),
        'start_timestamp': np.array(start_ts, np.float64),
    }
    return {'arrays': arrays, 'total': len(hist), 'per_scene': per_scene, 'rejected_gt_8': rejected,
            'n_distribution': n_dist,
            'start_frame_range': [int(min(starts)), int(max(starts))] if starts else None,
            'start_timestamp_range': [float(min(start_ts)), float(max(start_ts))] if start_ts else None}


def write_npz_deterministic(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for key in NPZ_KEYS:
            info = zipfile.ZipInfo('%s.npy' % key, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0
            info.internal_attr = 0
            info.flag_bits = 0
            with zf.open(info, 'w') as fid:
                npy_format.write_array(fid, np.ascontiguousarray(arrays[key]), allow_pickle=False)


def write_json(payload: dict, path: Path) -> str:
    text = json.dumps(payload, indent=1, ensure_ascii=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description='Build 20+20 Automatum samples')
    parser.add_argument('--splits-dir', default='data/automatum_t_crossing/splits')
    parser.add_argument('--stats-out', default='reports/data_preprocessing/automatum_sample_stats.json')
    parser.add_argument('--git-manifest-dir',
                        default='reports/data_preprocessing/sample_manifests')
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir).resolve()
    stats_path = Path(args.stats_out).resolve()
    git_manifest_dir = Path(args.git_manifest_dir).resolve()

    stats = {
        'dataset': 'automatum_t_crossing_samples',
        'task': {'history_frames': HISTORY_FRAMES, 'future_frames': FUTURE_FRAMES,
                 'total_frames': TOTAL_FRAMES, 'window_stride': WINDOW_STRIDE,
                 'canonical_fps': FPS, 'state': ['x', 'y', 'vx', 'vy'],
                 'vehicle_rule': '2 <= N <= 8, eligible = present in all 40 frames, '
                                 'N > 8 rejects the whole window, order by vehicle_id ascending',
                 'padding': {'max_nodes': MAX_NODES, 'state': 0.0, 'vehicle_id': -1}},
        'splits': {},
    }
    for split in ('train', 'val', 'test'):
        trajectory_path = splits_dir / split / 'trajectories.csv'
        actual_sha = sha256_file(trajectory_path)
        if actual_sha != SPLIT_SHA256[split]:
            print('FATAL: %s SHA256 mismatch: %s' % (trajectory_path, actual_sha), file=sys.stderr)
            sys.exit(2)
        print('[ok] %s trajectories sha256=%s' % (split, actual_sha))
        result = build_split(split, trajectory_path)
        arrays = result['arrays']
        npz_path = splits_dir / split / 'samples.npz'
        write_npz_deterministic(npz_path, arrays)
        npz_size = npz_path.stat().st_size
        npz_sha = sha256_file(npz_path)
        manifest = {
            'split_name': split,
            'parent_trajectory_path': 'data/automatum_t_crossing/splits/%s/trajectories.csv' % split,
            'parent_trajectory_sha256': actual_sha,
            'sampling': {'fps': FPS, 'history_frames': HISTORY_FRAMES,
                         'future_frames': FUTURE_FRAMES, 'total_frames': TOTAL_FRAMES,
                         'window_stride': WINDOW_STRIDE},
            'vehicle_rule': {'min_vehicles': MIN_VEHICLES, 'max_vehicles': MAX_VEHICLES,
                             'eligible_rule': 'present_in_all_40_frames',
                             'over_max_policy': 'reject_window',
                             'ordering': 'vehicle_id ascending'},
            'padding': {'max_nodes': MAX_NODES, 'state_padding': 0.0, 'vehicle_id_padding': -1},
            'samples': {'total': result['total'], 'per_scene': {str(k): v for k, v in result['per_scene'].items()},
                        'n_distribution': result['n_distribution'],
                        'rejected_n_gt_8': result['rejected_gt_8'],
                        'start_frame_range': result['start_frame_range'],
                        'start_timestamp_range': result['start_timestamp_range']},
            'output': {'samples_npz_path': 'data/automatum_t_crossing/splits/%s/samples.npz' % split,
                       'size_bytes': npz_size, 'sha256': npz_sha,
                       'arrays': {k: {'shape': list(arrays[k].shape), 'dtype': str(arrays[k].dtype)}
                                  for k in NPZ_KEYS}},
        }
        runtime_manifest = splits_dir / split / 'sample_manifest.json'
        text = write_json(manifest, runtime_manifest)
        mirror_path = git_manifest_dir / ('%s_sample_manifest.json' % split)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with mirror_path.open('w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
        assert sha256_file(runtime_manifest) == sha256_file(mirror_path)
        print('[ok] %s samples=%d rejected_gt_8=%d npz_bytes=%d npz_sha256=%s'
              % (split, result['total'], result['rejected_gt_8'], npz_size, npz_sha))
        stats['splits'][split] = {
            'trajectory_sha256': actual_sha, 'samples': result['total'],
            'per_scene': {str(k): v for k, v in result['per_scene'].items()},
            'n_distribution': result['n_distribution'],
            'rejected_n_gt_8': result['rejected_gt_8'],
            'npz': {'path': 'data/automatum_t_crossing/splits/%s/samples.npz' % split,
                    'size_bytes': npz_size, 'sha256': npz_sha},
            'sample_manifest': {'runtime': 'data/automatum_t_crossing/splits/%s/sample_manifest.json' % split,
                                'git_mirror': 'reports/data_preprocessing/sample_manifests/%s_sample_manifest.json' % split,
                                'sha256': sha256_file(runtime_manifest)},
        }
    stats['totals'] = {
        'samples': sum(stats['splits'][s]['samples'] for s in ('train', 'val', 'test')),
        'rejected_n_gt_8': sum(stats['splits'][s]['rejected_n_gt_8'] for s in ('train', 'val', 'test')),
        'n_distribution': {str(n): sum(stats['splits'][s]['n_distribution'][str(n)]
                                       for s in ('train', 'val', 'test'))
                           for n in range(MIN_VEHICLES, MAX_VEHICLES + 1)},
    }
    write_json(stats, stats_path)
    print('[ok] stats', stats_path)
    print(json.dumps({'samples': {s: stats['splits'][s]['samples'] for s in ('train', 'val', 'test')},
                      'rejected': {s: stats['splits'][s]['rejected_n_gt_8'] for s in ('train', 'val', 'test')},
                      'n_distribution': {s: stats['splits'][s]['n_distribution'] for s in ('train', 'val', 'test')}},
                     indent=1))


if __name__ == '__main__':
    main()