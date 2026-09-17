#!/usr/bin/env python3
"""Independent validation of the Automatum temporal 8:1:1 splits.

Reads canonical, train/val/test, both manifests and the stats file directly and
re-derives every property with its own implementation (no shared code with the
builder). The boundary selection rule is re-executed from the canonical data and
compared against the manifest, which is the determinism evidence available to a
single run (byte-level reproducibility is proven by running the builder twice).

Exits non-zero if any check fails.

Usage:
    python tools/data_preprocessing/validate_automatum_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

CANONICAL_SHA256 = 'a41ac4e24a8f21039dde60064868871b21093289a661ca9bef28ee7d1aab3442'
SOURCE_FPS = 29.97
STRIDE = 3
CANONICAL_FPS = SOURCE_FPS / STRIDE
CANONICAL_DT = STRIDE / SOURCE_FPS
GAP_FRAMES = 40
TARGET_RATIO = (0.8, 0.1, 0.1)
RATIO_TOL = 0.02
SEARCH_FRAMES = 300
EXPECTED_ROWS = 81450
HEADER = 'scene_id,vehicle_id,timestamp,x,y,vx,vy'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple:
    lines = path.read_text(encoding='utf-8').splitlines()
    return lines[0], lines[1:]


def frame_of(timestamp_text: str) -> int:
    source_frame = int(round(float(timestamp_text) * SOURCE_FPS))
    assert source_frame % STRIDE == 0
    assert abs(source_frame / SOURCE_FPS - float(timestamp_text)) < 1e-9
    return source_frame // STRIDE


def reselect_boundaries(canonical_rows: list, manifest: dict) -> dict:
    """Re-execute the documented selection rule with an independent implementation."""
    per_scene = {}
    for scene_id in (0, 1):
        frames_of_rows = []
        vehicle_frames = defaultdict(list)
        for scene_text, vehicle_text, timestamp_text in canonical_rows:
            if int(scene_text) != scene_id:
                continue
            f = frame_of(timestamp_text)
            frames_of_rows.append(f)
            vehicle_frames[int(vehicle_text)].append(f)
        f_first = min(frames_of_rows)
        f_last = max(frames_of_rows)
        total_slots = f_last - f_first + 1
        usable = total_slots - 2 * GAP_FRAMES
        n_train = int(round(TARGET_RATIO[0] * usable))
        n_val = int(round(TARGET_RATIO[1] * usable))
        g1_nominal = f_first + n_train
        g2_nominal = g1_nominal + GAP_FRAMES + n_val
        spans = [(min(v), max(v)) for v in vehicle_frames.values()]
        rows_per_frame = defaultdict(int)
        for f in frames_of_rows:
            rows_per_frame[f] += 1
        vehicles_per_frame = defaultdict(int)
        for first, last in spans:
            for f in range(first, last + 1):
                vehicles_per_frame[f] += 1
        g1_low = max(f_first + 1, g1_nominal - SEARCH_FRAMES)
        g1_high = min(f_last - 2 * GAP_FRAMES - 2, g1_nominal + SEARCH_FRAMES)
        g2_low = max(g1_low + GAP_FRAMES + 2, g2_nominal - SEARCH_FRAMES)
        g2_high = min(f_last - GAP_FRAMES - 1, g2_nominal + SEARCH_FRAMES)
        g1_range = range(g1_low, g1_high + 1)
        g2_range = range(g2_low, g2_high + 1)

        def boundary_metrics(g):
            cross = sum(1 for first, last in spans if first < g and last > g + GAP_FRAMES - 1)
            rows = sum(rows_per_frame.get(f, 0) for f in range(g, g + GAP_FRAMES))
            active = [vehicles_per_frame.get(f, 0) for f in range(g, g + GAP_FRAMES)]
            return {'cross': cross, 'rows': rows, 'mean': sum(active) / GAP_FRAMES,
                    'max': max(active), 'offset': abs(g - (g1_nominal if g < g2_nominal else g2_nominal))}

        m1 = {g: boundary_metrics(g) for g in g1_range}
        m2 = {g: boundary_metrics(g) for g in g2_range}
        best = None
        feasible = 0
        for g1 in g1_range:
            t1 = m1[g1]
            train_ratio = (g1 - f_first) / usable
            if abs(train_ratio - TARGET_RATIO[0]) > RATIO_TOL + 1e-12:
                continue
            for g2 in g2_range:
                if g2 <= g1 + GAP_FRAMES:
                    continue
                val_ratio = (g2 - g1 - GAP_FRAMES) / usable
                test_ratio = (f_last - g2 - GAP_FRAMES + 1) / usable
                if abs(val_ratio - TARGET_RATIO[1]) > RATIO_TOL + 1e-12:
                    continue
                if abs(test_ratio - TARGET_RATIO[2]) > RATIO_TOL + 1e-12:
                    continue
                feasible += 1
                t2 = m2[g2]
                key = (t1['cross'] + t2['cross'], t1['rows'] + t2['rows'],
                       t1['mean'] + t2['mean'], t1['max'] + t2['max'],
                       t1['offset'] + t2['offset'], g1, g2)
                if best is None or key < best[0]:
                    best = (key, g1, g2)
        per_scene[scene_id] = {
            'first': f_first, 'last': f_last, 'usable': usable,
            'g1_nominal': g1_nominal, 'g2_nominal': g2_nominal,
            'feasible_pairs': feasible, 'selected': (best[1], best[2]) if best else None,
            'key': best[0] if best else None}
    return per_scene


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate the Automatum temporal splits')
    parser.add_argument('--canonical', default='data/automatum_t_crossing/processed/trajectories_10hz.csv')
    parser.add_argument('--splits-dir', default='data/automatum_t_crossing/splits')
    parser.add_argument('--stats', default='reports/data_preprocessing/automatum_split_stats.json')
    parser.add_argument('--manifest', default='data/automatum_t_crossing/splits/split_manifest.json')
    parser.add_argument('--manifest-git-mirror',
                        default='reports/data_preprocessing/automatum_split_manifest.json')
    parser.add_argument('--out', default='reports/data_preprocessing/automatum_split_validation.json')
    args = parser.parse_args()

    canonical = Path(args.canonical).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    stats = json.loads(Path(args.stats).resolve().read_text(encoding='utf-8'))
    manifest = json.loads(Path(args.manifest).resolve().read_text(encoding='utf-8'))
    checks = {}

    def record(name, passed, detail):
        checks[name] = {'pass': bool(passed), 'detail': detail}

    canonical_sha = sha256_file(canonical)
    record('parent_canonical_sha256', canonical_sha == CANONICAL_SHA256,
           {'actual': canonical_sha, 'expected': CANONICAL_SHA256})

    canonical_header, canonical_lines = read_rows(canonical)
    record('canonical_rows_expected', len(canonical_lines) == EXPECTED_ROWS,
           {'rows': len(canonical_lines)})
    canonical_set = set(canonical_lines)
    canonical_rows = [line.split(',')[:3] for line in canonical_lines]
    for line, parts in zip(canonical_lines, canonical_rows):
        frame_of(parts[2])

    headers = {}
    split_lines = {}
    for split in ('train', 'val', 'test'):
        header, lines = read_rows(splits_dir / split / 'trajectories.csv')
        headers[split] = header
        split_lines[split] = lines
    record('schemas_identical_7_columns',
           all(headers[s] == HEADER and headers[s] == canonical_header for s in headers),
           {'headers': headers})

    split_sets = {s: set(split_lines[s]) for s in split_lines}
    record('split_rows_exist_in_canonical',
           all(split_sets[s] <= canonical_set for s in split_sets),
           {s: len(split_sets[s] - canonical_set) for s in split_sets})
    record('split_rows_unique', all(len(split_sets[s]) == len(split_lines[s]) for s in split_lines),
           {s: len(split_lines[s]) - len(split_sets[s]) for s in split_lines})
    record('no_split_row_overlap',
           not (split_sets['train'] & split_sets['val']) and not (split_sets['train'] & split_sets['test'])
           and not (split_sets['val'] & split_sets['test']),
           {'train_val': len(split_sets['train'] & split_sets['val']),
            'train_test': len(split_sets['train'] & split_sets['test']),
            'val_test': len(split_sets['val'] & split_sets['test'])})

    def range_label(scene_id: int, frame: int) -> str:
        scene = manifest['scenes']['scene_%d' % scene_id]
        for label, key in (('train', 'train_frame_range'), ('gap', 'gap1_frame_range'),
                           ('val', 'val_frame_range'), ('gap', 'gap2_frame_range'),
                           ('test', 'test_frame_range')):
            lo, hi = scene[key]
            if lo <= frame <= hi:
                return label
        return 'outside'

    row_split = {}
    for s in split_sets:
        for line in split_sets[s]:
            row_split[line] = s
    label_counts = defaultdict(int)
    mismatch = 0
    outside = 0
    per_scene_frames = defaultdict(lambda: defaultdict(set))
    for line, parts in zip(canonical_lines, canonical_rows):
        scene_id = int(parts[0])
        frame = frame_of(parts[2])
        label = range_label(scene_id, frame)
        label_counts[label] += 1
        if label == 'outside':
            outside += 1
        expected = None if label in ('gap', 'outside') else label
        actual = row_split.get(line)
        if expected != actual:
            mismatch += 1
        if actual is not None:
            per_scene_frames[scene_id][actual].add(frame)
    record('canonical_rows_partitioned',
           mismatch == 0 and outside == 0 and label_counts['gap'] + label_counts['train']
           + label_counts['val'] + label_counts['test'] == len(canonical_lines),
           {'counts': dict(label_counts), 'mismatches': mismatch, 'outside': outside})
    gap_rows_in_splits = sum(1 for line, parts in zip(canonical_lines, canonical_rows)
                             if range_label(int(parts[0]), frame_of(parts[2])) == 'gap'
                             and line in row_split)
    record('gap_rows_not_in_any_split', gap_rows_in_splits == 0,
           {'gap_rows': label_counts['gap'], 'gap_rows_found_in_splits': gap_rows_in_splits})
    record('split_counts_match_stats',
           len(split_lines['train']) == stats['totals']['train_rows']
           and len(split_lines['val']) == stats['totals']['val_rows']
           and len(split_lines['test']) == stats['totals']['test_rows']
           and label_counts['gap'] == stats['totals']['gap_rows_total'],
           {'train': len(split_lines['train']), 'val': len(split_lines['val']),
            'test': len(split_lines['test']), 'gap': label_counts['gap'],
            'stats_gap': stats['totals']['gap_rows_total']})

    cross_time = defaultdict(set)
    for s in split_sets:
        for line in split_sets[s]:
            parts = line.split(',')
            cross_time[(parts[0], parts[2])].add(s)
    violations = sum(1 for splits in cross_time.values() if len(splits) > 1)
    record('timestamp_not_across_splits', violations == 0,
           {'checked_timestamps': len(cross_time), 'violations': violations})

    contiguous = True
    scene_detail = {}
    for scene_id in (0, 1):
        scene = manifest['scenes']['scene_%d' % scene_id]
        ranges = {s: tuple(scene['%s_frame_range' % s]) for s in ('train', 'val', 'test')}
        ranges['gap1'] = tuple(scene['gap1_frame_range'])
        ranges['gap2'] = tuple(scene['gap2_frame_range'])
        ordered = [ranges['train'], ranges['gap1'], ranges['val'], ranges['gap2'], ranges['test']]
        ok = all(ordered[i][1] + 1 == ordered[i + 1][0] for i in range(len(ordered) - 1))
        ok = ok and ordered[0][0] == scene['first_active_frame'] \
            and ordered[-1][1] == scene['last_active_frame']
        gap_sizes = [ranges['gap1'][1] - ranges['gap1'][0] + 1,
                     ranges['gap2'][1] - ranges['gap2'][0] + 1]
        ok = ok and gap_sizes == [GAP_FRAMES, GAP_FRAMES]
        for split in ('train', 'val', 'test'):
            frames = per_scene_frames.get(scene_id, {}).get(split, set())
            lo, hi = ranges[split]
            if not frames or min(frames) < lo or max(frames) > hi:
                ok = False
        contiguous = contiguous and ok
        scene_detail[str(scene_id)] = {'contiguous': ok, 'ranges': {k: list(v) for k, v in ranges.items()},
                                       'gap_sizes': gap_sizes}
    record('scenes_split_as_contiguous_blocks', contiguous, scene_detail)

    radius_ok = True
    boundary_detail = {}
    for scene_id in (0, 1):
        scene = manifest['scenes']['scene_%d' % scene_id]
        total_slots = scene['last_active_frame'] - scene['first_active_frame'] + 1
        usable = total_slots - 2 * GAP_FRAMES
        n_train = int(round(TARGET_RATIO[0] * usable))
        n_val = int(round(TARGET_RATIO[1] * usable))
        g1_nominal = scene['first_active_frame'] + n_train
        g2_nominal = g1_nominal + GAP_FRAMES + n_val
        ok = (scene['nominal_gap1_start_frame'] == g1_nominal
              and scene['nominal_gap2_start_frame'] == g2_nominal
              and abs(scene['selected_gap1_start_frame'] - g1_nominal) <= SEARCH_FRAMES
              and abs(scene['selected_gap2_start_frame'] - g2_nominal) <= SEARCH_FRAMES)
        radius_ok = radius_ok and ok
        boundary_detail[str(scene_id)] = {'nominal': [g1_nominal, g2_nominal],
                                          'selected': [scene['selected_gap1_start_frame'],
                                                       scene['selected_gap2_start_frame']],
                                          'within_search_radius': ok}
    record('selected_boundaries_within_search_radius', radius_ok, boundary_detail)

    reproduced = reselect_boundaries(canonical_rows, manifest)
    reproduction_ok = True
    for scene_id in (0, 1):
        selected = manifest['scenes']['scene_%d' % scene_id]
        expected = reproduced[scene_id]['selected']
        match = expected == (selected['selected_gap1_start_frame'],
                             selected['selected_gap2_start_frame'])
        reproduction_ok = reproduction_ok and match
        boundary_detail[str(scene_id)]['rule_reproduction'] = {
            'reproduced': list(expected) if expected else None,
            'manifest': [selected['selected_gap1_start_frame'], selected['selected_gap2_start_frame']],
            'match': match,
            'feasible_pairs_reproduced': reproduced[scene_id]['feasible_pairs'],
            'feasible_pairs_manifest': selected['feasible_pairs']}
    record('boundary_rule_reproduced', reproduction_ok, boundary_detail)

    ratio_ok = True
    ratio_detail = {}
    for scene_id in (0, 1):
        scene = manifest['scenes']['scene_%d' % scene_id]
        usable = scene['usable_slots']
        ratios = {'train': scene['train_slots'] / usable, 'val': scene['val_slots'] / usable,
                  'test': scene['test_slots'] / usable}
        ok = all(abs(ratios[name] - target) <= RATIO_TOL + 1e-12
                 for name, target in zip(('train', 'val', 'test'), TARGET_RATIO))
        ratio_ok = ratio_ok and ok
        ratio_detail[str(scene_id)] = {k: round(v, 6) for k, v in ratios.items()} | {'within_tol': ok}
    record('ratio_within_2pp', ratio_ok, ratio_detail)

    order_ok = True
    for split in ('train', 'val', 'test'):
        previous = None
        for line in split_lines[split]:
            parts = line.split(',')
            key = (int(parts[0]), float(parts[2]), int(parts[1]))
            if previous is not None and key < previous:
                order_ok = False
                break
            previous = key
        if not order_ok:
            break
    record('csv_sorted_scene_time_vehicle', order_ok, {'sorted': order_ok})

    sizes_ok = True
    sha_ok = True
    size_detail = {}
    for split in ('train', 'val', 'test'):
        name = '%s.csv' % split
        entry = manifest['outputs'][name]
        path = splits_dir / split / 'trajectories.csv'
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        sizes_ok = sizes_ok and entry['rows'] == len(split_lines[split]) and entry['size_bytes'] == actual_size
        sha_ok = sha_ok and entry['sha256'] == actual_sha
        size_detail[name] = {'rows': len(split_lines[split]), 'size_bytes': actual_size,
                             'sha256': actual_sha, 'manifest_sha256': entry['sha256']}
    record('manifest_rows_sizes', sizes_ok, size_detail)
    record('csv_sha256_matches_manifest', sha_ok, {k: v['sha256'] for k, v in size_detail.items()})

    runtime_sha = sha256_file(Path(args.manifest).resolve())
    mirror_sha = sha256_file(Path(args.manifest_git_mirror).resolve())
    record('manifest_mirror_byte_identical', runtime_sha == mirror_sha,
           {'runtime': runtime_sha, 'git_mirror': mirror_sha})
    record('manifest_sha256_matches_stats', runtime_sha == stats['manifest']['sha256'],
           {'manifest': runtime_sha, 'stats': stats['manifest']['sha256']})

    failed = [name for name, entry in checks.items() if not entry['pass']]
    payload = {
        'dataset': 'automatum_t_crossing_splits',
        'validator': 'tools/data_preprocessing/validate_automatum_splits.py',
        'checks': checks,
        'summary': {'checks_total': len(checks), 'checks_passed': len(checks) - len(failed),
                    'checks_failed': len(failed), 'failed_checks': failed,
                    'verdict': 'PASS' if not failed else 'FAIL'},
        'split_row_accounting': {'train': len(split_lines['train']), 'val': len(split_lines['val']),
                                 'test': len(split_lines['test']), 'gap': label_counts['gap'],
                                 'canonical': len(canonical_lines)},
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'checks': payload['summary'],
                      'boundaries': boundary_detail,
                      'ratios': ratio_detail,
                      'accounting': payload['split_row_accounting'],
                      'output': str(out_path)}, indent=1, ensure_ascii=False))
    if failed:
        print('FAILED CHECKS:', failed, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()