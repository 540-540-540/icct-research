#!/usr/bin/env python3
"""Independent validation of the Automatum 20+20 multi-vehicle samples.

Re-reads trajectories.csv, samples.npz and the manifests and re-derives every
property with its own implementation (no shared code with the builder):

- anchor reconciliation: every 2<=N<=8 window appears exactly once, every N>8
  window is absent, N distribution and per-scene counts match
- per-sample QA: shapes, dtypes, mask/ids/padding, 40-frame continuity, history
  and future slice placement, exact traceability back to float64 CSV states,
  scene and split containment, start timestamp
- float32 casting error only (no other transformation)
- frozen trajectory SHA256, npz SHA256, manifest mirrors byte-identical

Exits non-zero if any check fails.

Usage:
    python tools/data_preprocessing/validate_automatum_samples.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS = 29.97 / 3
SOURCE_FPS = 29.97
STRIDE = 3
HISTORY_FRAMES = 20
FUTURE_FRAMES = 20
TOTAL_FRAMES = 40
MIN_VEHICLES = 2
MAX_VEHICLES = 8
MAX_NODES = 8
SPLIT_SHA256 = {
    'train': 'd75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11',
    'val': 'c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d',
    'test': '6fb3117e95368e2c5b66cca5a28cb45a1ff91691d4afea4fb9b8c1a17eb75371',
}
EXPECTED_DTYPES = {'history': 'float32', 'future': 'float32', 'vehicle_ids': 'int32',
                   'vehicle_mask': 'bool', 'num_vehicles': 'uint8', 'scene_id': 'uint8',
                   'start_frame': 'int32', 'start_timestamp': 'float64'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_trajectories(path: Path) -> dict:
    """Independent parse into per-scene dense float32 grids plus float64 metadata."""
    raw = np.loadtxt(path, delimiter=',', skiprows=1)
    scene = raw[:, 0].astype(np.int64)
    vehicle = raw[:, 1].astype(np.int64)
    timestamp = raw[:, 2]
    states = raw[:, 3:7]
    source_frame = np.round(timestamp * SOURCE_FPS).astype(np.int64)
    assert np.all(source_frame % STRIDE == 0), 'timestamp not on the stride-3 phase'
    assert np.max(np.abs(source_frame / SOURCE_FPS - timestamp)) < 1e-9
    frame = source_frame // STRIDE
    cast_error_max = float(np.max(np.abs(states - states.astype(np.float32).astype(np.float64))))
    out = {'cast_error_max': cast_error_max, 'states_checked': int(states.size)}
    for scene_id in (0, 1):
        mask = scene == scene_id
        vehicles = np.unique(vehicle[mask])
        vehicles.sort()
        frames = frame[mask]
        f_first, f_last = int(frames.min()), int(frames.max())
        grid = np.full((len(vehicles), f_last - f_first + 1, 4), np.nan, np.float32)
        vidx = np.searchsorted(vehicles, vehicle[mask])
        offsets = frames - f_first
        seen = np.zeros(grid.shape[:2], dtype=bool)
        assert not np.any(seen[vidx, offsets]), 'duplicate vehicle/frame state'
        seen[vidx, offsets] = True
        grid[vidx, offsets] = states[mask].astype(np.float32)
        ts = np.full(f_last - f_first + 1, np.nan)
        ts[offsets] = timestamp[mask]
        present = ~np.isnan(grid[:, :, 0])
        v_first = np.argmax(present, axis=1) + f_first
        v_last = present.shape[1] - 1 - np.argmax(present[:, ::-1], axis=1) + f_first
        contiguous = np.array([np.count_nonzero(present[k]) == v_last[k] - v_first[k] + 1
                               for k in range(len(vehicles))])
        assert contiguous.all(), 'vehicle frames are not contiguous'
        out[scene_id] = {'vehicles': vehicles, 'grid': grid, 'timestamps': ts,
                         'f_first': f_first, 'f_last': f_last,
                         'v_first': v_first, 'v_last': v_last}
    return out


def anchors_for_scene(scene: dict, f_from: int, f_to: int) -> tuple:
    """Return (anchors dict frame -> eligible vehicle list, rejected count) for a frame range."""
    v_first, v_last = scene['v_first'], scene['v_last']
    vehicles = scene['vehicles']
    n_frames = scene['f_last'] - scene['f_first'] + 1
    delta = np.zeros(n_frames + 1, dtype=np.int64)
    for k in range(len(vehicles)):
        lo = int(v_first[k])
        hi = int(v_last[k]) - TOTAL_FRAMES + 1
        if hi >= lo:
            delta[lo - scene['f_first']] += 1
            delta[hi + 1 - scene['f_first']] -= 1
    counts = np.cumsum(delta[:-1])
    anchors = {}
    rejected = 0
    for f in range(f_from, f_to + 1):
        n = int(counts[f - scene['f_first']])
        if n < MIN_VEHICLES:
            continue
        if n > MAX_VEHICLES:
            rejected += 1
            continue
        eligible = (v_first <= f) & (v_last >= f + TOTAL_FRAMES - 1)
        anchors[f] = [int(v) for v in vehicles[eligible]]
    return anchors, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate the Automatum 20+20 samples')
    parser.add_argument('--splits-dir', default='data/automatum_t_crossing/splits')
    parser.add_argument('--split-manifest', default='data/automatum_t_crossing/splits/split_manifest.json')
    parser.add_argument('--git-manifest-dir', default='reports/data_preprocessing/sample_manifests')
    parser.add_argument('--out', default='reports/data_preprocessing/automatum_sample_validation.json')
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir).resolve()
    split_manifest = json.loads(Path(args.split_manifest).resolve().read_text(encoding='utf-8'))
    checks = {}
    per_sample_issues = defaultdict(int)
    cast_error_max = 0.0
    checked_states = 0

    def record(name, passed, detail):
        checks[name] = {'pass': bool(passed), 'detail': detail}

    summary = {}
    for split in ('train', 'val', 'test'):
        print('[validate] %s ...' % split, file=sys.stderr, flush=True)
        split_dir = splits_dir / split
        trajectory_path = split_dir / 'trajectories.csv'
        npz_path = split_dir / 'samples.npz'
        manifest = json.loads((split_dir / 'sample_manifest.json').read_text(encoding='utf-8'))
        mirror = Path(args.git_manifest_dir).resolve() / ('%s_sample_manifest.json' % split)

        record('%s_trajectory_sha256' % split, sha256_file(trajectory_path) == SPLIT_SHA256[split],
               {'actual': sha256_file(trajectory_path), 'expected': SPLIT_SHA256[split]})
        record('%s_npz_sha256_matches_manifest' % split,
               sha256_file(npz_path) == manifest['output']['sha256'],
               {'actual': sha256_file(npz_path), 'manifest': manifest['output']['sha256']})
        record('%s_manifest_mirror_byte_identical' % split,
               sha256_file(mirror) == sha256_file(split_dir / 'sample_manifest.json'),
               {'runtime': sha256_file(split_dir / 'sample_manifest.json'), 'mirror': sha256_file(mirror)})

        trajectory = read_trajectories(trajectory_path)
        cast_error_max = max(cast_error_max, trajectory['cast_error_max'])
        checked_states += trajectory['states_checked']

        expected = {}
        rejected_total = 0
        for scene_id in (0, 1):
            scene = trajectory[scene_id]
            ranges = split_manifest['scenes']['scene_%d' % scene_id]['%s_frame_range' % split]
            f_from = max(scene['f_first'], ranges[0])
            f_to = min(scene['f_last'] - TOTAL_FRAMES + 1, ranges[1] - TOTAL_FRAMES + 1)
            anchors, rejected = anchors_for_scene(scene, f_from, f_to)
            expected.update({(scene_id, f): v for f, v in anchors.items()})
            rejected_total += rejected

        data = np.load(npz_path)
        arrays_ok = all(str(data[key].dtype) == EXPECTED_DTYPES[key] for key in EXPECTED_DTYPES)
        record('%s_array_dtypes' % split, arrays_ok,
               {key: str(data[key].dtype) for key in EXPECTED_DTYPES})
        shape_ok = (data['history'].shape[1:] == (HISTORY_FRAMES, MAX_NODES, 4)
                    and data['future'].shape[1:] == (FUTURE_FRAMES, MAX_NODES, 4)
                    and data['vehicle_ids'].shape[1:] == (MAX_NODES,)
                    and data['vehicle_mask'].shape[1:] == (MAX_NODES,))
        record('%s_array_shapes' % split, shape_ok,
               {key: list(data[key].shape) for key in
                ('history', 'future', 'vehicle_ids', 'vehicle_mask')})

        n_samples = int(data['history'].shape[0])
        record('%s_count_matches_recomputed_anchors' % split, n_samples == len(expected),
               {'samples': n_samples, 'recomputed_2_to_8_anchors': len(expected),
                'manifest_total': manifest['samples']['total']})
        record('%s_rejected_gt_8_match' % split,
               rejected_total == manifest['samples']['rejected_n_gt_8'],
               {'recomputed': rejected_total, 'manifest': manifest['samples']['rejected_n_gt_8']})

        sample_keys = []
        n_dist = defaultdict(int)
        per_scene = defaultdict(int)
        history = data['history']
        future = data['future']
        for i in range(n_samples):
            scene_id = int(data['scene_id'][i])
            start_frame = int(data['start_frame'][i])
            n = int(data['num_vehicles'][i])
            ids = data['vehicle_ids'][i]
            mask = data['vehicle_mask'][i]
            if not (MIN_VEHICLES <= n <= MAX_VEHICLES):
                per_sample_issues['num_vehicles_out_of_range'] += 1
                continue
            if int(mask.sum()) != n:
                per_sample_issues['mask_sum_mismatch'] += 1
                continue
            head = [int(x) for x in ids[:n]]
            if any(x <= 0 for x in head) or any(int(ids[j]) != -1 for j in range(n, MAX_NODES)):
                per_sample_issues['vehicle_id_padding_or_value'] += 1
                continue
            if head != sorted(head) or len(set(head)) != n:
                per_sample_issues['vehicle_id_order_or_duplicate'] += 1
                continue
            if np.any(history[i, :, n:, :] != 0) or np.any(future[i, :, n:, :] != 0):
                per_sample_issues['state_padding_not_zero'] += 1
                continue
            anchor = expected.get((scene_id, start_frame))
            if anchor is None:
                per_sample_issues['sample_not_a_recomputed_anchor'] += 1
                continue
            if head != anchor:
                per_sample_issues['eligible_vehicle_set_mismatch'] += 1
                continue
            scene = trajectory[scene_id]
            ranges = split_manifest['scenes']['scene_%d' % scene_id]['%s_frame_range' % split]
            if start_frame < ranges[0] or start_frame + TOTAL_FRAMES - 1 > ranges[1]:
                per_sample_issues['window_crosses_split_boundary'] += 1
                continue
            expected_ts = float(scene['timestamps'][start_frame - scene['f_first']])
            if float(data['start_timestamp'][i]) != expected_ts:
                per_sample_issues['start_timestamp_mismatch'] += 1
                continue
            rows = np.searchsorted(scene['vehicles'], np.array(head))
            block = scene['grid'][rows, start_frame - scene['f_first']:
                                  start_frame - scene['f_first'] + TOTAL_FRAMES, :]
            stored = np.concatenate([history[i, :, :n, :], future[i, :, :n, :]], axis=0).transpose(1, 0, 2)
            if not np.array_equal(stored, block) or np.any(np.isnan(block)):
                per_sample_issues['state_not_exact_float32_of_csv'] += 1
                continue
            sample_keys.append((scene_id, start_frame))
            n_dist[str(n)] += 1
            per_scene[str(scene_id)] += 1

        record('%s_per_sample_qa' % split, not per_sample_issues,
               {'samples_checked': n_samples, 'issues': dict(per_sample_issues)})
        record('%s_samples_sorted_scene_frame' % split,
               all(sample_keys[i] < sample_keys[i + 1] for i in range(len(sample_keys) - 1)),
               {'first': sample_keys[0] if sample_keys else None,
                'last': sample_keys[-1] if sample_keys else None})
        record('%s_all_anchors_covered_exactly_once' % split,
               set(sample_keys) == set(expected) and len(sample_keys) == len(expected),
               {'recomputed': len(expected), 'validated_samples': len(sample_keys)})
        n_dist_full = {k: n_dist.get(k, 0) for k in map(str, range(MIN_VEHICLES, MAX_VEHICLES + 1))}
        record('%s_n_distribution_matches_manifest' % split,
               n_dist_full == manifest['samples']['n_distribution'],
               {'recomputed': n_dist_full, 'manifest': manifest['samples']['n_distribution']})
        per_scene_full = {k: per_scene.get(k, 0) for k in ('0', '1')}
        record('%s_per_scene_matches_manifest' % split,
               per_scene_full == manifest['samples']['per_scene'],
               {'recomputed': per_scene_full, 'manifest': manifest['samples']['per_scene']})
        summary[split] = {'samples': n_samples, 'rejected_gt_8': rejected_total,
                          'n_distribution': n_dist_full}
        print('[validate] %s done: %d samples, %d rejected, issues=%s'
              % (split, n_samples, rejected_total, dict(per_sample_issues)), file=sys.stderr, flush=True)

    record('float32_cast_error_only', cast_error_max < 1e-3,
           {'max_abs_cast_error': cast_error_max, 'states_checked': checked_states,
            'note': 'stored values equal np.float32(csv float64) exactly'})

    failed = [name for name, entry in checks.items() if not entry['pass']]
    payload = {
        'dataset': 'automatum_t_crossing_samples',
        'validator': 'tools/data_preprocessing/validate_automatum_samples.py',
        'checks': checks,
        'summary': {'checks_total': len(checks), 'checks_passed': len(checks) - len(failed),
                    'checks_failed': len(failed), 'failed_checks': failed,
                    'verdict': 'PASS' if not failed else 'FAIL'},
        'splits': summary,
        'float32_cast_max_abs_error': cast_error_max,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'checks': payload['summary'], 'splits': summary,
                      'float32_cast_max_abs_error': cast_error_max,
                      'output': str(out_path)}, indent=1, ensure_ascii=False))
    if failed:
        print('FAILED CHECKS:', failed, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()