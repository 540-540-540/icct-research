#!/usr/bin/env python3
"""Build deterministic temporal Train/Val/Test = 8:1:1 splits for Automatum.

Reads only the frozen canonical CSV (SHA256 checked before anything else) and
writes a pure row subset per split. Splitting rules (frozen by the work order):

- each scene is split independently along its own continuous canonical time axis
- ratio is defined on canonical frame slots, never on CSV row counts
- one shared 40-frame gap between Train/Val and one between Val/Test
- gap start candidates searched within +/- 30 s (+/- 300 frames) of the nominal
  ratio cut; candidate pairs must keep every scene ratio within +/- 2 percentage
  points of 0.80 / 0.10 / 0.10
- deterministic lexicographic ranking of feasible pairs:
    1. total cross-gap vehicles
    2. total gap rows
    3. total mean active vehicles in gaps
    4. total max active vehicles in gaps
    5. total offset from the nominal boundaries
    6. earliest (gap1, gap2)
- canonical frame recovery: source_frame = round(timestamp * 29.97), must be a
  multiple of 3; canonical_frame = source_frame // 3

Outputs (row text copied verbatim from canonical, so every split file is a pure
subset, byte for byte):
    data/automatum_t_crossing/splits/{train,val,test}.csv
    data/automatum_t_crossing/splits/split_manifest.json
    reports/data_preprocessing/automatum_split_manifest.json   (byte-identical mirror)
    reports/data_preprocessing/automatum_split_stats.json

Usage:
    python tools/data_preprocessing/build_automatum_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CANONICAL_SHA256 = 'a41ac4e24a8f21039dde60064868871b21093289a661ca9bef28ee7d1aab3442'
PARENT_COMMIT = 'c9721eaf1c36509744cbf4d34b2145d0eeaf1b7a'
SOURCE_FPS = 29.97
STRIDE = 3
CANONICAL_FPS = SOURCE_FPS / STRIDE
CANONICAL_DT = STRIDE / SOURCE_FPS
GAP_FRAMES = 40
TARGET_RATIO = (0.8, 0.1, 0.1)
RATIO_TOL = 0.02
SEARCH_SECONDS = 30
SEARCH_FRAMES = int(round(SEARCH_SECONDS * CANONICAL_FPS))
EXPECTED_CANONICAL_ROWS = 81450


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_timestamp(frame: int) -> float:
    return (frame * STRIDE) / SOURCE_FPS


def window_sum(prefix: np.ndarray, start: int, length: int) -> int:
    return int(prefix[start + length] - prefix[start])


def window_max(values: np.ndarray, start: int, length: int) -> int:
    return int(values[start:start + length].max()) if length else 0


def scene_candidates(scene_id: int, rows_per_frame: np.ndarray,
                     active_per_frame: np.ndarray, vehicle_first: np.ndarray,
                     vehicle_last: np.ndarray, f_first: int, f_last: int) -> dict:
    total_slots = f_last - f_first + 1
    usable = total_slots - 2 * GAP_FRAMES
    train_slots = int(round(TARGET_RATIO[0] * usable))
    val_slots = int(round(TARGET_RATIO[1] * usable))
    test_slots = usable - train_slots - val_slots
    g1_nominal = f_first + train_slots
    g2_nominal = g1_nominal + GAP_FRAMES + val_slots

    rows_prefix = np.concatenate([[0], np.cumsum(rows_per_frame)])
    active_prefix = np.concatenate([[0], np.cumsum(active_per_frame)])
    offset = f_first

    def metrics(candidates: np.ndarray, nominal: int) -> dict:
        cross = np.array([int(np.count_nonzero((vehicle_first < g) & (vehicle_last > g + GAP_FRAMES - 1)))
                          for g in candidates])
        rows = np.array([window_sum(rows_prefix, int(g) - offset, GAP_FRAMES) for g in candidates])
        mean_active = np.array([float(active_prefix[int(g) - offset + GAP_FRAMES]
                                      - active_prefix[int(g) - offset]) / GAP_FRAMES
                                for g in candidates])
        max_active = np.array([window_max(active_per_frame, int(g) - offset, GAP_FRAMES)
                               for g in candidates])
        return {'frames': candidates, 'cross': cross, 'rows': rows,
                'mean_active': mean_active, 'max_active': max_active,
                'offset': np.abs(candidates - nominal)}

    r = SEARCH_FRAMES
    g1_low = max(f_first + 1, g1_nominal - r)
    g1_high = min(f_last - 2 * GAP_FRAMES - 2, g1_nominal + r)
    g2_low = max(g1_low + GAP_FRAMES + 2, g2_nominal - r)
    g2_high = min(f_last - GAP_FRAMES - 1, g2_nominal + r)
    g1_cands = np.arange(g1_low, g1_high + 1, dtype=np.int64)
    g2_cands = np.arange(g2_low, g2_high + 1, dtype=np.int64)
    m1 = metrics(g1_cands, int(g1_nominal))
    m2 = metrics(g2_cands, int(g2_nominal))

    g1v = g1_cands[:, None]
    g2v = g2_cands[None, :]
    train_ratio = (g1v - f_first) / usable
    val_ratio = (g2v - g1v - GAP_FRAMES) / usable
    test_ratio = (f_last - g2v - GAP_FRAMES + 1) / usable
    feasible = ((np.abs(train_ratio - TARGET_RATIO[0]) <= RATIO_TOL + 1e-12)
                & (np.abs(val_ratio - TARGET_RATIO[1]) <= RATIO_TOL + 1e-12)
                & (np.abs(test_ratio - TARGET_RATIO[2]) <= RATIO_TOL + 1e-12))
    candidate_count = int(feasible.sum())
    if candidate_count == 0:
        return {'scene_id': scene_id, 'feasible_pairs': 0, 'status': 'NO_VALID_NATURAL_PAIR',
                'f_first': f_first, 'f_last': f_last, 'usable_slots': usable,
                'nominal_gap1_start_frame': int(g1_nominal),
                'nominal_gap2_start_frame': int(g2_nominal)}

    cross_sum = m1['cross'][:, None] + m2['cross'][None, :]
    rows_sum = m1['rows'][:, None] + m2['rows'][None, :]
    mean_sum = m1['mean_active'][:, None] + m2['mean_active'][None, :]
    max_sum = m1['max_active'][:, None] + m2['max_active'][None, :]
    offset_sum = m1['offset'][:, None] + m2['offset'][None, :]
    g1_grid = np.broadcast_to(g1v, feasible.shape)
    g2_grid = np.broadcast_to(g2v, feasible.shape)
    mask = feasible.ravel()
    order = np.lexsort((g2_grid.ravel()[mask], g1_grid.ravel()[mask], offset_sum.ravel()[mask],
                        max_sum.ravel()[mask], mean_sum.ravel()[mask], rows_sum.ravel()[mask],
                        cross_sum.ravel()[mask]))
    flat_cross = cross_sum.ravel()[mask][order]
    flat_rows = rows_sum.ravel()[mask][order]
    flat_mean = mean_sum.ravel()[mask][order]
    flat_max = max_sum.ravel()[mask][order]
    flat_offset = offset_sum.ravel()[mask][order]
    flat_g1 = g1_grid.ravel()[mask][order]
    flat_g2 = g2_grid.ravel()[mask][order]

    def per_candidate(metric: dict, g: int) -> dict:
        i = int(np.nonzero(metric['frames'] == g)[0][0])
        return {'cross_gap_vehicles': int(metric['cross'][i]), 'gap_rows': int(metric['rows'][i]),
                'mean_active_vehicles_in_gap': float(metric['mean_active'][i]),
                'max_active_vehicles_in_gap': int(metric['max_active'][i]),
                'distance_from_nominal_frames': int(metric['offset'][i])}

    selected_g1 = int(flat_g1[0])
    selected_g2 = int(flat_g2[0])
    return {
        'scene_id': scene_id, 'status': 'OK',
        'f_first': f_first, 'f_last': f_last, 'total_slots': total_slots,
        'usable_slots': usable, 'nominal_train_slots': train_slots,
        'nominal_val_slots': val_slots, 'nominal_test_slots': test_slots,
        'nominal_gap1_start_frame': int(g1_nominal),
        'nominal_gap2_start_frame': int(g2_nominal),
        'selected_gap1_start_frame': selected_g1,
        'selected_gap2_start_frame': selected_g2,
        'gap1_shift_frames': selected_g1 - int(g1_nominal),
        'gap2_shift_frames': selected_g2 - int(g2_nominal),
        'gap1': per_candidate(m1, selected_g1),
        'gap2': per_candidate(m2, selected_g2),
        'candidate_gap1_starts': int(len(g1_cands)),
        'candidate_gap2_starts': int(len(g2_cands)),
        'feasible_pairs': candidate_count,
        'runner_up': ({'gap1': int(flat_g1[1]), 'gap2': int(flat_g2[1]),
                       'cross_total': int(flat_cross[1]), 'rows_total': int(flat_rows[1])}
                      if len(flat_g1) > 1 else None),
        'selected_pair_rank_key': {
            'cross_gap_vehicles_total': int(flat_cross[0]),
            'gap_rows_total': int(flat_rows[0]),
            'mean_active_total': float(flat_mean[0]),
            'max_active_total': int(flat_max[0]),
            'nominal_offset_total_frames': int(flat_offset[0])},
        'train_slots': selected_g1 - f_first,
        'val_slots': selected_g2 - selected_g1 - GAP_FRAMES,
        'test_slots': f_last - selected_g2 - GAP_FRAMES + 1,
        'train_ratio': (selected_g1 - f_first) / usable,
        'val_ratio': (selected_g2 - selected_g1 - GAP_FRAMES) / usable,
        'test_ratio': (f_last - selected_g2 - GAP_FRAMES + 1) / usable,
        'rows_per_frame': rows_per_frame, 'active_per_frame': active_per_frame,
    }


def window_supply(scene_id: int, split_frame_ranges: dict, vehicle_slices: dict) -> dict:
    out = {}
    for split, (lo, hi) in split_frame_ranges.items():
        anchors = np.zeros(hi - lo + 1, dtype=np.int64)
        windows = 0
        vehicles_with = 0
        for vehicle, (v_first, v_last) in vehicle_slices.items():
            first = max(v_first, lo)
            last = min(v_last, hi)
            count = last - first + 1
            if count >= 40:
                windows += count - 39
                vehicles_with += 1
                start_lo = first
                start_hi = last - 39
                anchors[start_lo - lo] += 1
                if start_hi - lo + 1 < len(anchors):
                    anchors[start_hi - lo + 1] -= 1
        eligible = np.cumsum(anchors)
        active_anchors = eligible[eligible > 0]
        out[split] = {
            'potential_single_vehicle_40pt_windows': int(windows),
            'unique_vehicles_with_40pt_window': int(vehicles_with),
            'anchor_count': int(active_anchors.size),
            'anchors_ge_2': int((active_anchors >= 2).sum()),
            'anchors_2_to_8': int(((active_anchors >= 2) & (active_anchors <= 8)).sum()),
            'anchors_gt_8': int((active_anchors > 8).sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Build deterministic temporal 8:1:1 splits')
    parser.add_argument('--canonical', default='data/automatum_t_crossing/processed/trajectories_10hz.csv')
    parser.add_argument('--out-dir', default='data/automatum_t_crossing/splits')
    parser.add_argument('--stats-out', default='reports/data_preprocessing/automatum_split_stats.json')
    parser.add_argument('--manifest-git-mirror',
                        default='reports/data_preprocessing/automatum_split_manifest.json')
    args = parser.parse_args()

    canonical = Path(args.canonical).resolve()
    out_dir = Path(args.out_dir).resolve()
    stats_path = Path(args.stats_out).resolve()
    git_manifest = Path(args.manifest_git_mirror).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    sha = sha256_file(canonical)
    if sha != CANONICAL_SHA256:
        print('FATAL: canonical SHA256 mismatch: %s != %s' % (sha, CANONICAL_SHA256), file=sys.stderr)
        sys.exit(2)
    print('[ok] canonical SHA256', sha)

    lines = canonical.read_text(encoding='utf-8').splitlines()
    header = lines[0]
    data_lines = lines[1:]
    assert len(data_lines) == EXPECTED_CANONICAL_ROWS, 'unexpected canonical row count'
    scene = np.empty(len(data_lines), dtype=np.int64)
    frame = np.empty(len(data_lines), dtype=np.int64)
    vehicle = np.empty(len(data_lines), dtype=np.int64)
    for i, line in enumerate(data_lines):
        parts = line.split(',')
        scene[i] = int(parts[0])
        vehicle[i] = int(parts[1])
        source_frame = int(round(float(parts[2]) * SOURCE_FPS))
        assert source_frame % STRIDE == 0, 'canonical timestamp is not on the stride-3 phase'
        assert abs(source_frame / SOURCE_FPS - float(parts[2])) < 1e-9, 'frame roundtrip error'
        frame[i] = source_frame // STRIDE

    scene_results = {}
    split_label = np.empty(len(data_lines), dtype='U5')
    for scene_id in (0, 1):
        selection = scene == scene_id
        scene_frames = frame[selection]
        f_first, f_last = int(scene_frames.min()), int(scene_frames.max())
        rows_per_frame = np.bincount(scene_frames - f_first, minlength=f_last - f_first + 1)
        active_per_frame = np.zeros_like(rows_per_frame)
        vehicle_first = []
        vehicle_last = []
        vehicle_uuids = np.unique(vehicle[selection])
        for vid in vehicle_uuids:
            rows = frame[selection & (vehicle == vid)]
            assert rows.max() - rows.min() + 1 == len(rows), 'vehicle frames are not contiguous'
            vehicle_first.append(int(rows.min()))
            vehicle_last.append(int(rows.max()))
        vehicle_first = np.array(vehicle_first)
        vehicle_last = np.array(vehicle_last)
        for f in range(f_first, f_last + 1):
            active_per_frame[f - f_first] = int(np.count_nonzero(
                (vehicle_first <= f) & (vehicle_last >= f)))
        assert np.array_equal(active_per_frame, rows_per_frame), \
            'active vehicle count does not match canonical rows per frame'
        result = scene_candidates(scene_id, rows_per_frame, active_per_frame,
                                  vehicle_first, vehicle_last, f_first, f_last)
        if result.get('status') != 'OK':
            print('FATAL: NO_VALID_NATURAL_PAIR for scene %d' % scene_id, file=sys.stderr)
            sys.exit(3)
        g1, g2 = result['selected_gap1_start_frame'], result['selected_gap2_start_frame']
        labels = np.full(len(data_lines), '', dtype='U5')
        scene_rows = np.nonzero(selection)[0]
        f = frame[scene_rows]
        lab = np.where(f < g1, 'train', np.where(f < g1 + GAP_FRAMES, 'gap',
                       np.where(f < g2, 'val', np.where(f < g2 + GAP_FRAMES, 'gap', 'test'))))
        labels[scene_rows] = lab
        split_label[scene_rows] = lab
        ranges = {
            'train': (f_first, g1 - 1), 'val': (g1 + GAP_FRAMES, g2 - 1),
            'test': (g2 + GAP_FRAMES, f_last)}
        result['split_frame_ranges'] = {k: [int(a), int(b)] for k, (a, b) in ranges.items()}
        result['gap1_frame_range'] = [g1, g1 + GAP_FRAMES - 1]
        result['gap2_frame_range'] = [g2, g2 + GAP_FRAMES - 1]
        result['gap1_timestamp_range_s'] = [frame_timestamp(g1), frame_timestamp(g1 + GAP_FRAMES - 1)]
        result['gap2_timestamp_range_s'] = [frame_timestamp(g2), frame_timestamp(g2 + GAP_FRAMES - 1)]
        result['nominal_gap1_start_timestamp_s'] = frame_timestamp(result['nominal_gap1_start_frame'])
        result['nominal_gap2_start_timestamp_s'] = frame_timestamp(result['nominal_gap2_start_frame'])
        result['selected_gap1_start_timestamp_s'] = frame_timestamp(g1)
        result['selected_gap2_start_timestamp_s'] = frame_timestamp(g2)
        result['first_active_timestamp_s'] = frame_timestamp(f_first)
        result['last_active_timestamp_s'] = frame_timestamp(f_last)
        result['scene_rows'] = int(selection.sum())
        result['gap_rows'] = {
            'gap1': window_sum(np.concatenate([[0], np.cumsum(rows_per_frame)]), g1 - f_first, GAP_FRAMES),
            'gap2': window_sum(np.concatenate([[0], np.cumsum(rows_per_frame)]), g2 - f_first, GAP_FRAMES)}
        vehicle_slices = {int(v): (int(a), int(b)) for v, a, b in
                          zip(vehicle_uuids, vehicle_first, vehicle_last)}
        result['window_supply'] = window_supply(scene_id, ranges, vehicle_slices)
        scene_results[scene_id] = result

    split_lines = {'train': [], 'val': [], 'test': []}
    for i, line in enumerate(data_lines):
        label = split_label[i]
        if label in split_lines:
            split_lines[label].append(line)

    outputs = {}
    for split in ('train', 'val', 'test'):
        path = out_dir / ('%s.csv' % split)
        with path.open('w', encoding='utf-8', newline='') as fh:
            fh.write(header + '\n')
            fh.write('\n'.join(split_lines[split]) + '\n')
        rows = len(split_lines[split])
        outputs[split + '.csv'] = {'path': 'data/automatum_t_crossing/splits/%s.csv' % split,
                                   'rows': rows, 'size_bytes': path.stat().st_size,
                                   'sha256': sha256_file(path)}
        print('[ok] %s.csv rows=%d sha256=%s' % (split, rows, outputs[split + '.csv']['sha256']))

    manifest = {
        'dataset_name': 'automatum_t_crossing_splits',
        'parent_canonical_path': 'data/automatum_t_crossing/processed/trajectories_10hz.csv',
        'parent_canonical_sha256': CANONICAL_SHA256,
        'parent_commit': PARENT_COMMIT,
        'sampling': {'source_fps': SOURCE_FPS, 'stride': STRIDE, 'canonical_fps': CANONICAL_FPS,
                     'canonical_dt_s': CANONICAL_DT,
                     'canonical_frame_rule': 'source_frame = round(timestamp*29.97); '
                                             'source_frame % 3 == 0; canonical_frame = source_frame // 3'},
        'split': {
            'target_ratio': list(TARGET_RATIO), 'gap_frames': GAP_FRAMES,
            'search_radius_seconds': SEARCH_SECONDS, 'search_radius_frames': SEARCH_FRAMES,
            'ratio_tolerance_percentage_points': RATIO_TOL * 100,
            'ratio_definition': 'per scene, slots/usable_slots with usable_slots = F - 2*gap_frames',
            'boundary_selection_algorithm':
                'per scene: joint search of (gap1_start, gap2_start) over +/-300 canonical frames '
                'around the nominal 8:1:1 cuts; feasible pairs keep train/val/test slot ratios '
                'within +/-2 percentage points of 0.80/0.10/0.10; feasible pairs are ranked '
                'lexicographically ascending by (total cross-gap vehicles, total gap rows, total '
                'mean active vehicles in gaps, total max active vehicles in gaps, total offset from '
                'nominal boundaries, earliest gap1, earliest gap2); the first pair wins, no manual pick',
        },
        'scenes': {},
        'outputs': outputs,
        'gap_ranges': {},
    }
    for scene_id, result in scene_results.items():
        key = 'scene_%d' % scene_id
        f_first = result['f_first']
        f_last = result['f_last']
        ranges = result['split_frame_ranges']
        manifest['scenes'][key] = {
            'scene_id': scene_id,
            'first_active_frame': f_first, 'last_active_frame': f_last,
            'first_active_timestamp_s': result['first_active_timestamp_s'],
            'last_active_timestamp_s': result['last_active_timestamp_s'],
            'active_slots': result['total_slots'], 'usable_slots': result['usable_slots'],
            'nominal_gap1_start_frame': result['nominal_gap1_start_frame'],
            'nominal_gap2_start_frame': result['nominal_gap2_start_frame'],
            'nominal_gap1_start_timestamp_s': result['nominal_gap1_start_timestamp_s'],
            'nominal_gap2_start_timestamp_s': result['nominal_gap2_start_timestamp_s'],
            'selected_gap1_start_frame': result['selected_gap1_start_frame'],
            'selected_gap2_start_frame': result['selected_gap2_start_frame'],
            'selected_gap1_start_timestamp_s': result['selected_gap1_start_timestamp_s'],
            'selected_gap2_start_timestamp_s': result['selected_gap2_start_timestamp_s'],
            'gap1_shift_frames': result['gap1_shift_frames'],
            'gap2_shift_frames': result['gap2_shift_frames'],
            'gap1': result['gap1'], 'gap2': result['gap2'],
            'candidate_gap1_starts': result['candidate_gap1_starts'],
            'candidate_gap2_starts': result['candidate_gap2_starts'],
            'feasible_pairs': result['feasible_pairs'],
            'selected_pair_rank_key': result['selected_pair_rank_key'],
            'train_frame_range': ranges['train'], 'val_frame_range': ranges['val'],
            'test_frame_range': ranges['test'],
            'gap1_frame_range': result['gap1_frame_range'], 'gap2_frame_range': result['gap2_frame_range'],
            'train_timestamp_range_s': [frame_timestamp(ranges['train'][0]),
                                        frame_timestamp(ranges['train'][1])],
            'val_timestamp_range_s': [frame_timestamp(ranges['val'][0]),
                                      frame_timestamp(ranges['val'][1])],
            'test_timestamp_range_s': [frame_timestamp(ranges['test'][0]),
                                       frame_timestamp(ranges['test'][1])],
            'gap1_timestamp_range_s': result['gap1_timestamp_range_s'],
            'gap2_timestamp_range_s': result['gap2_timestamp_range_s'],
            'train_slots': result['train_slots'], 'val_slots': result['val_slots'],
            'test_slots': result['test_slots'],
            'train_duration_s': result['train_slots'] * CANONICAL_DT,
            'val_duration_s': result['val_slots'] * CANONICAL_DT,
            'test_duration_s': result['test_slots'] * CANONICAL_DT,
            'train_ratio': result['train_ratio'], 'val_ratio': result['val_ratio'],
            'test_ratio': result['test_ratio'],
        }
        manifest['gap_ranges'][key] = {
            'gap1_gap_rows': result['gap_rows']['gap1'], 'gap2_gap_rows': result['gap_rows']['gap2'],
            'total_gap_rows': result['gap_rows']['gap1'] + result['gap_rows']['gap2']}
    manifest_text = json.dumps(manifest, indent=1, ensure_ascii=False) + '\n'
    runtime_manifest = out_dir / 'split_manifest.json'
    for target in (runtime_manifest, git_manifest):
        with target.open('w', encoding='utf-8', newline='\n') as fh:
            fh.write(manifest_text)
    manifest_sha = sha256_file(runtime_manifest)
    assert manifest_sha == sha256_file(git_manifest), 'manifest mirror is not byte-identical'
    print('[ok] manifest sha256', manifest_sha)

    per_scene_supply = {str(k): v['window_supply'] for k, v in scene_results.items()}
    combined_supply = {}
    for split in ('train', 'val', 'test'):
        combined_supply[split] = {field: sum(per_scene_supply[str(s)][split][field] for s in (0, 1))
                                  for field in per_scene_supply['0']['train']}
    split_rows = {split: len(split_lines[split]) for split in ('train', 'val', 'test')}
    gap_rows = {str(s): scene_results[s]['gap_rows']['gap1'] + scene_results[s]['gap_rows']['gap2']
                for s in (0, 1)}
    vehicle_sets = {}
    for split in ('train', 'val', 'test'):
        vehicle_sets[split] = {line.split(',')[0] + ':' + line.split(',')[1]
                               for line in split_lines[split]}
    overlaps = {
        'train_val': len(vehicle_sets['train'] & vehicle_sets['val']),
        'val_test': len(vehicle_sets['val'] & vehicle_sets['test']),
        'train_test': len(vehicle_sets['train'] & vehicle_sets['test']),
    }
    stats = {
        'dataset': 'automatum_t_crossing_splits',
        'canonical_sha256_verified': True,
        'parent_canonical_path': 'data/automatum_t_crossing/processed/trajectories_10hz.csv',
        'sampling': manifest['sampling'],
        'split': manifest['split'],
        'scenes': {str(k): {field: v[field] for field in (
            'scene_id', 'f_first', 'f_last', 'total_slots', 'usable_slots',
            'nominal_gap1_start_frame', 'nominal_gap2_start_frame',
            'selected_gap1_start_frame', 'selected_gap2_start_frame',
            'gap1_shift_frames', 'gap2_shift_frames', 'gap1', 'gap2',
            'candidate_gap1_starts', 'candidate_gap2_starts', 'feasible_pairs',
            'selected_pair_rank_key', 'split_frame_ranges',
            'gap1_frame_range', 'gap2_frame_range', 'gap1_timestamp_range_s', 'gap2_timestamp_range_s',
            'train_slots', 'val_slots', 'test_slots', 'train_ratio', 'val_ratio', 'test_ratio',
            'gap_rows', 'window_supply')} for k, v in scene_results.items()},
        'totals': {
            'canonical_rows': len(data_lines),
            'train_rows': split_rows['train'], 'val_rows': split_rows['val'],
            'test_rows': split_rows['test'],
            'gap_rows_by_scene': gap_rows,
            'gap_rows_total': gap_rows['0'] + gap_rows['1'],
            'split_rows_total': split_rows['train'] + split_rows['val'] + split_rows['test'],
        },
        'vehicle_ids': {
            'train_unique': len(vehicle_sets['train']), 'val_unique': len(vehicle_sets['val']),
            'test_unique': len(vehicle_sets['test']), 'overlaps': overlaps},
        'window_supply': {'per_scene': per_scene_supply, 'combined': combined_supply},
        'outputs': outputs,
        'manifest': {'runtime_path': 'data/automatum_t_crossing/splits/split_manifest.json',
                     'git_mirror_path': 'reports/data_preprocessing/automatum_split_manifest.json',
                     'sha256': manifest_sha},
    }
    with stats_path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(stats, indent=1, ensure_ascii=False) + '\n')
    print('[ok] stats', stats_path)
    print(json.dumps({'rows': {k: split_rows[k] for k in ('train', 'val', 'test')},
                      'gap_rows': gap_rows, 'manifest_sha': manifest_sha,
                      'scene_0_ratio': [round(scene_results[0]['train_ratio'], 5),
                                        round(scene_results[0]['val_ratio'], 5),
                                        round(scene_results[0]['test_ratio'], 5)],
                      'scene_1_ratio': [round(scene_results[1]['train_ratio'], 5),
                                        round(scene_results[1]['val_ratio'], 5),
                                        round(scene_results[1]['test_ratio'], 5)]}, indent=1))


if __name__ == '__main__':
    main()