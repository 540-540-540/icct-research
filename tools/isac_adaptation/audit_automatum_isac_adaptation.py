#!/usr/bin/env python3
"""AUDIT ONLY: Automatum T-Crossing -> current ISAC frontend adaptation audit.

Read-only with respect to data, configs and research code. Computes
- dataset baseline and scene statistics from the frozen canonical CSV
- sample cohort vs true per-frame scene occupancy from the frozen splits
- candidate 3-BS geometry per T-crossing from the frozen staticWorld.xodr
- static evidence of current code dependencies (file:line hits)
and assembles the migration matrix and open decisions curated in this audit.

Writes reports/isac_adaptation/automatum_isac_adaptation_audit.json.
No training, no sensing experiment, no data modification.

Usage:
    python tools/isac_adaptation/audit_automatum_isac_adaptation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FPS = 29.97
STRIDE = 3
CANONICAL_FPS = SOURCE_FPS / STRIDE
CANONICAL_DT = STRIDE / SOURCE_FPS
SCENE_NAMES = {
    0: 'T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2',
    1: 'T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f',
}
SCENE_LABELS = {0: 'Gaimersheim Stadtweg', 1: 'St2214 Duenzlau Umgehung'}
CANONICAL_REL = 'data/automatum_t_crossing/processed/trajectories_10hz.csv'
SPLIT_SHA256 = {
    'train': 'd75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11',
    'val': 'c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d',
    'test': '6fb3117e95368e2c5b66cca5a28cb45a1ff91691d4afea4fb9b8c1a17eb75371',
}
NPZ_SHA256 = {
    'train': '1b86a6453a53779b9b647f8fa04bf3b40c56bba0dfc26f0d19f6de895a58e0c9',
    'val': '51dc713ba5873ba438267e5c46f97902e4e744cbd3069bf3e289a485d058ff31',
    'test': 'dec95982e48686e03e0ba059fb3a977f095b881342ed8ce9416c4c0ec8424413',
}
RANGE_MIN_COVERAGE = 5.0
COVERAGE_RANGE_MAX = (100.0, 120.0, 150.0)
COVERAGE_FOV_HALF = (60.0, 70.0, 80.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pct(values, ps=(50, 90, 95, 99, 100)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    q = np.percentile(values, ps) if values.size else [float('nan')] * len(ps)
    return {'count': int(values.size), **{'p%g' % p: float(v) for p, v in zip(ps, q)}}


def load_canonical() -> tuple:
    path = ROOT / CANONICAL_REL
    scene, vehicle, ts, x, y, vx, vy = [], [], [], [], [], [], []
    frame = []
    with path.open(encoding='utf-8', newline='') as fh:
        header = fh.readline().rstrip('\n')
        assert header == 'scene_id,vehicle_id,timestamp,x,y,vx,vy'
        for line in fh:
            parts = line.rstrip('\n').split(',')
            scene.append(int(parts[0]))
            vehicle.append(int(parts[1]))
            t = float(parts[2])
            ts.append(t)
            frame.append(int(round(t * SOURCE_FPS)) // STRIDE)
            x.append(float(parts[3]))
            y.append(float(parts[4]))
            vx.append(float(parts[5]))
            vy.append(float(parts[6]))
    return {'sha256': sha256_file(path), 'header': header, 'rows': len(scene),
            'scene': np.array(scene), 'vehicle': np.array(vehicle), 'timestamp': np.array(ts),
            'frame': np.array(frame), 'x': np.array(x), 'y': np.array(y),
            'vx': np.array(vx), 'vy': np.array(vy)}


def scene_stats(canonical: dict) -> dict:
    out = {}
    for scene_id in (0, 1):
        mask = canonical['scene'] == scene_id
        frames = canonical['frame'][mask]
        f_first, f_last = int(frames.min()), int(frames.max())
        counts = np.bincount(frames - f_first, minlength=f_last - f_first + 1)
        speed = np.hypot(canonical['vx'][mask], canonical['vy'][mask])
        out[str(scene_id)] = {
            'scene_id': scene_id, 'label': SCENE_LABELS[scene_id],
            'recording_name': SCENE_NAMES[scene_id],
            'canonical_points': int(mask.sum()),
            'x_range_m': [float(canonical['x'][mask].min()), float(canonical['x'][mask].max())],
            'y_range_m': [float(canonical['y'][mask].min()), float(canonical['y'][mask].max())],
            'vx_range_mps': [float(canonical['vx'][mask].min()), float(canonical['vx'][mask].max())],
            'vy_range_mps': [float(canonical['vy'][mask].min()), float(canonical['vy'][mask].max())],
            'speed_mps': pct(speed),
            'vehicles_per_frame': {**pct(counts),
                                   'frames_total': int(counts.size),
                                   'frames_empty': int((counts == 0).sum())},
            'time_range_s': [float(canonical['timestamp'][mask].min()),
                             float(canonical['timestamp'][mask].max())],
            'static_world_xodr': 'data/automatum_t_crossing/raw/extracted/%s/staticWorld.xodr'
                                 % SCENE_NAMES[scene_id],
        }
    return out


def find_recording_dir(name: str) -> Path:
    raw = ROOT / 'data/automatum_t_crossing/raw'
    for candidate in (raw / name, raw / 'extracted' / name):
        if (candidate / 'dynamicWorld.json').is_file():
            return candidate
    for world in raw.rglob('dynamicWorld.json'):
        if world.parent.name == name:
            return world.parent
    raise FileNotFoundError(name)


def candidate_bs_geometry(canonical: dict) -> dict:
    sys.path.insert(0, str(ROOT / 'tools/data_audit'))
    from audit_automatum_t_crossing import junction_from_xodr, parse_xodr  # noqa: E402
    out = {'placement_rule': 'one BS per approach arm: nearest road point + 30 m along the arm '
                             '+ 8 m lateral offset; boresight points at the junction center',
           'coverage_probe': {'range_min_m': RANGE_MIN_COVERAGE,
                              'range_max_m': list(COVERAGE_RANGE_MAX),
                              'fov_half_angle_deg': list(COVERAGE_FOV_HALF),
                              'height_difference_m': 5.0,
                              'definition': 'a BS sees a point when range_min <= sqrt(rho^2+dh^2) <= '
                                            'range_max and |wrap(bearing-boresight)| <= FOV half'},
           'scenes': {}}
    for scene_id in (0, 1):
        record_dir = find_recording_dir(SCENE_NAMES[scene_id])
        roads = parse_xodr(record_dir / 'staticWorld.xodr')
        junction = junction_from_xodr(roads, None)
        center = np.array(junction['center'], dtype=float)
        stations, boresights = [], []
        for arm in junction['arms']:
            direction = np.array(arm['direction'], dtype=float)
            perp = np.array([-direction[1], direction[0]])
            bs = np.array(arm['point'], dtype=float) + 30.0 * direction + 8.0 * perp
            boresight = math.atan2(center[1] - bs[1], center[0] - bs[0])
            stations.append(bs)
            boresights.append(boresight)
        stations = np.asarray(stations)
        boresights = np.asarray(boresights)
        v1 = stations[1] - stations[0]
        v2 = stations[2] - stations[0]
        area = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        angles = []
        for i in range(3):
            v1 = stations[(i + 1) % 3] - stations[i]
            v2 = stations[(i + 2) % 3] - stations[i]
            cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
        mask = canonical['scene'] == scene_id
        points = np.column_stack([canonical['x'][mask], canonical['y'][mask]])
        delta = points[:, None, :] - stations[None, :, :]
        rho = np.linalg.norm(delta, axis=2)
        r3 = np.sqrt(rho ** 2 + 5.0 ** 2)
        bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0]) - boresights[None, :]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        nearest = rho.min(axis=1)
        coverage = {}
        for range_max in COVERAGE_RANGE_MAX:
            for fov in COVERAGE_FOV_HALF:
                visible = (r3 >= RANGE_MIN_COVERAGE) & (r3 <= range_max) \
                    & (np.abs(bearing) <= math.radians(fov))
                count = visible.sum(axis=1)
                coverage['range_%g_fov_%g' % (range_max, fov)] = {
                    'ge_1_bs': float((count >= 1).mean()), 'ge_2_bs': float((count >= 2).mean()),
                    'all_3_bs': float((count >= 3).mean()),
                    'mean_visible_bs': float(count.mean())}
        recommended = coverage['range_120_fov_70']
        visible = (r3 >= RANGE_MIN_COVERAGE) & (r3 <= 120.0) \
            & (np.abs(bearing) <= math.radians(70.0))
        out['scenes']['scene_%d' % scene_id] = {
            'scene_id': scene_id, 'label': SCENE_LABELS[scene_id],
            'junction_center_xy_m': center.tolist(),
            'junction_center_source': 'staticWorld.xodr (connector-road centrality)',
            'arms': [{'road_id': arm['road_id'], 'axis_deg': arm['axis_deg'],
                      'distance_to_center_m': arm['distance_to_center_m']} for arm in junction['arms']],
            'bs': [{'bs_id': i, 'x_m': float(stations[i, 0]), 'y_m': float(stations[i, 1]),
                    'boresight_deg': math.degrees(boresights[i]),
                    'arm_road_id': junction['arms'][i]['road_id']} for i in range(3)],
            'triangle_area_m2': float(area), 'triangle_min_angle_deg': float(min(angles)),
            'triangle_angles_deg': [float(a) for a in angles],
            'vehicle_to_nearest_bs_m': pct(nearest, (50, 90, 95, 100)),
            'visible_bs_count_per_point': {
                'eq_0': float((visible.sum(axis=1) == 0).mean()),
                'eq_1': float((visible.sum(axis=1) == 1).mean()),
                'eq_2': float((visible.sum(axis=1) == 2).mean()),
                'eq_3': float((visible.sum(axis=1) == 3).mean()),
                'mean': float(visible.sum(axis=1).mean())},
            'per_bs_visibility_fraction': [float(visible[:, i].mean()) for i in range(3)],
            'coverage_matrix': coverage,
            'recommended_candidate': {
                'range_min_m': RANGE_MIN_COVERAGE, 'range_max_m': 150.0, 'fov_half_angle_deg': 70.0,
                'station_height_m': 6.5, 'vehicle_height_m': 1.5, 'height_difference_m': 5.0,
                'candidate_coverage': coverage['range_150_fov_70'], 'status': 'CANDIDATE (not frozen)',
                'rationale': 'max vehicle-to-nearest-BS distance in this scene is %.1f m; range 150 m '
                             'reaches 100%% >=1-BS coverage in both T-crossings (120 m leaves 2.4%% / '
                             '7.3%% of points uncovered) and, unlike 150 m, larger ranges add nothing; '
                             'FOV 70 deg keeps the current value; height difference 5 m keeps the '
                             'existing H_M and simulator constant' % float(nearest.max())},
        }
    return out


def find_trajectory(path: Path) -> dict:
    scene, vehicle, ts, x, y, vx, vy = [], [], [], [], [], [], []
    with path.open(encoding='utf-8', newline='') as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip('\n').split(',')
            scene.append(int(parts[0]))
            vehicle.append(int(parts[1]))
            ts.append(float(parts[2]))
            x.append(float(parts[3]))
            y.append(float(parts[4]))
            vx.append(float(parts[5]))
            vy.append(float(parts[6]))
    return {'scene': np.array(scene), 'vehicle': np.array(vehicle), 'timestamp': np.array(ts),
            'x': np.array(x), 'y': np.array(y), 'vx': np.array(vx), 'vy': np.array(vy)}


def sample_vs_scene_occupancy() -> dict:
    """For every frozen sample: cohort size N vs true per-frame occupancy in the window."""
    result = {'definition': {
        'cohort': 'sample vehicle_ids = vehicles present in all 40 window frames',
        'extra_vehicle': 'a real vehicle present in >=1 of the 40 frames but not in the cohort',
        'history': 'frames 0..19 of the window', 'future': 'frames 20..39',
        'per_sample_metrics': ['N', 'history_max_concurrent', 'history_max_extra',
                               'future_max_concurrent', 'future_max_extra', 'max_concurrent_40',
                               'extra_vehicles_distinct_40', 'history_frames_with_extra']},
        'per_split': {}, 'overall': {}}
    all_metrics = []
    for split in ('train', 'val', 'test'):
        split_dir = ROOT / ('data/automatum_t_crossing/splits/%s' % split)
        trajectory = find_trajectory(split_dir / 'trajectories.csv')
        data = np.load(split_dir / 'samples.npz')
        per_scene_grids = {}
        for scene_id in (0, 1):
            mask = trajectory['scene'] == scene_id
            vehicles = np.unique(trajectory['vehicle'][mask])
            frames = trajectory['timestamp'][mask]
            frame = np.round(frames * SOURCE_FPS).astype(np.int64) // STRIDE
            f_first, f_last = int(frame.min()), int(frame.max())
            grid = np.zeros((len(vehicles), f_last - f_first + 1), dtype=bool)
            vidx = np.searchsorted(vehicles, trajectory['vehicle'][mask])
            seen = np.zeros_like(grid)
            assert not np.any(seen[vidx, frame - f_first]), 'duplicate state'
            seen[vidx, frame - f_first] = True
            grid[vidx, frame - f_first] = True
            per_scene_grids[scene_id] = grid
        metrics = defaultdict(list)
        scene_firsts = {}
        for scene_id in (0, 1):
            mask = trajectory['scene'] == scene_id
            frame = np.round(trajectory['timestamp'][mask] * SOURCE_FPS).astype(np.int64) // STRIDE
            scene_firsts[scene_id] = int(frame.min())
        for i in range(int(data['history'].shape[0])):
            scene_id = int(data['scene_id'][i])
            start = int(data['start_frame'][i])
            n = int(data['num_vehicles'][i])
            grid = per_scene_grids[scene_id]
            offset = start - scene_firsts[scene_id]
            block = grid[:, offset:offset + 40]
            assert block.shape[1] == 40, 'window outside split trajectory'
            counts = block.sum(axis=0)
            distinct = int(block.any(axis=1).sum())
            hist_max_concurrent = int(counts[:20].max())
            fut_max_concurrent = int(counts[20:].max())
            metrics['history_max_concurrent'].append(hist_max_concurrent)
            metrics['history_max_extra'].append(hist_max_concurrent - n)
            metrics['future_max_concurrent'].append(fut_max_concurrent)
            metrics['future_max_extra'].append(fut_max_concurrent - n)
            metrics['max_concurrent_40'].append(int(counts.max()))
            metrics['extra_vehicles_distinct_40'].append(distinct - n)
            metrics['history_frames_with_extra'].append(int((counts[:20] > n).sum()))
            metrics['_scene'].append(scene_id)
            metrics['_N'].append(n)

        def aggregate(arrays: dict) -> dict:
            hist_extra = np.array(arrays['history_max_extra'])
            hist_conc = np.array(arrays['history_max_concurrent'])
            fut_extra = np.array(arrays['future_max_extra'])
            fut_conc = np.array(arrays['future_max_concurrent'])
            extra40 = np.array(arrays['extra_vehicles_distinct_40'])
            max40 = np.array(arrays['max_concurrent_40'])
            return {
                'samples': int(hist_extra.size),
                'zero_extra_in_40_frames_fraction': float((extra40 == 0).mean()),
                'history_with_extra_fraction': float((hist_extra > 0).mean()),
                'future_with_extra_fraction': float((fut_extra > 0).mean()),
                'history_max_extra': pct(hist_extra, (50, 90, 95, 100)),
                'history_max_concurrent': pct(hist_conc, (50, 90, 95, 100)),
                'future_max_extra': pct(fut_extra, (50, 90, 95, 100)),
                'future_max_concurrent': pct(fut_conc, (50, 90, 95, 100)),
                'samples_with_concurrent_gt_8': int((max40 > 8).sum()),
                'samples_with_concurrent_gt_8_fraction': float((max40 > 8).mean()),
                'max_concurrent_observed': int(max40.max()),
            }
        scene_ids = np.array(metrics['_scene'])
        per_scene = {}
        for scene_id in (0, 1):
            selection = scene_ids == scene_id
            per_scene['scene_%d' % scene_id] = aggregate(
                {k: np.asarray(v)[selection] for k, v in metrics.items()
                 if not k.startswith('_') and k not in ('N',)})
        overall = aggregate({k: np.asarray(v) for k, v in metrics.items()
                             if not k.startswith('_')})
        result['per_split'][split] = {'overall': overall, 'per_scene': per_scene}
        all_metrics.append(metrics)
    combined = defaultdict(list)
    for metrics in all_metrics:
        for key, value in metrics.items():
            if not key.startswith('_'):
                combined[key].extend(np.asarray(value).tolist())
    arrays = {k: np.asarray(v) for k, v in combined.items()}
    extra40 = arrays['extra_vehicles_distinct_40']
    hist_extra = arrays['history_max_extra']
    result['overall'] = {
        'samples': int(extra40.size),
        'zero_extra_in_40_frames_fraction': float((extra40 == 0).mean()),
        'history_with_extra_fraction': float((hist_extra > 0).mean()),
        'history_max_extra': pct(hist_extra, (50, 90, 95, 100)),
        'max_concurrent_observed': int(arrays['max_concurrent_40'].max()),
        'samples_with_concurrent_gt_8': int((arrays['max_concurrent_40'] > 8).sum()),
    }
    return result


def code_dependency_scan() -> dict:
    patterns = {
        'legacy_source_data': r'f01_source|source_states\.npy|episodes\.jsonl',
        'legacy_geometry': r'f01a[/\\]geometry\.json|geometry_source',
        'legacy_calibration': r'snr_rebuild_06|f01e|ALLOWED_SNR_DB',
        'snr_reference_definition': r'snr_ref_db|reference_range_m|reference_rcs_m2',
        'fixed_visibility_gate': r'10\.0\s*<=\s*r|10\.0.*300\.0|300\.0|70\.0\)',
        'fixed_dt_0_1': r'dt\s*=\s*0\.1|%100|100_000_000|dt=0\.1',
        'height_constant_5m': r'H_M\s*=\s*5\.0|height_difference_m',
    }
    targets = ['frontend', 'configs', 'scripts']
    hits = defaultdict(list)
    for base in targets:
        for path in sorted((ROOT / base).rglob('*')):
            if path.suffix not in ('.py', '.json', '.yaml'):
                continue
            try:
                lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                for label, pattern in patterns.items():
                    if re.search(pattern, line):
                        hits[label].append({'file': str(path.relative_to(ROOT)).replace('\\', '/'),
                                            'line': number, 'text': line.strip()[:160]})
    return {label: {'count': len(entries), 'hits': entries[:40]}
            for label, entries in sorted(hits.items())}


MIGRATION_MATRIX = [
    {'file': 'frontend/echo_source.py', 'current_role': 'F01-B causal source adapter + streaming '
     'three-station observations; reads data/f01_source/source_states.npy, reports/f01a/episodes.jsonl, '
     'reports/f01a/geometry.json; hard 100 ms deadline grid and 100 ms hold window',
     'verdict': 'refactor/replace', 'changes': 'replace with an Automatum scene adapter over '
     'splits/*/trajectories.csv: canonical frame index f, dt=3/29.97, scene-global phase; keep the '
     'A-domain observation interface (per-BS shared echo + schedule)', 'priority': 'P0'},
    {'file': 'frontend/ofdm_echo.py', 'current_role': 'reference OFDM echo synthesizer with '
     'snr_ref_db and 100 m reference range; Waveform.dt=0.1 validated against 3 CPI',
     'verdict': 'small change', 'changes': 'parameterise dt to 3/29.97 s; remove snr_ref_db reference '
     'range semantics in favour of the new per-BS frame SNR definition (section 9)',
     'priority': 'P1'},
    {'file': 'frontend/sensing/simulator.py', 'current_role': 'formal shared multi-target echo '
     'simulator (A domain); snr_ref_db with 100 m reference, amplitude ~ RCS/10; visibility '
     '10-300 m, 70 deg, requires height_m=5.0',
     'verdict': 'small change', 'changes': 'config-driven visibility (range/FOV/height candidate); '
     'replace snr_ref_db amplitude law with the new SNR_b = 10log10(mean|S_b|^2/mean|W_b|^2) control; '
     'keep per-BS shared echo and single receiver-noise realisation', 'priority': 'P0'},
    {'file': 'frontend/sensing/waveform.py', 'current_role': 'frozen waveform/array/resource + '
     'load_geometry() from reports/f01a/geometry.json',
     'verdict': 'small change', 'changes': 'load_geometry must read the Automatum candidate BS '
     'geometry config (to be frozen); waveform constants unchanged',
     'priority': 'P0'},
    {'file': 'frontend/sensing/coords.py', 'current_role': 'polar/Cartesian conversion, r-u '
     'Jacobian, covariance LUT application (H_M=5.0)',
     'verdict': 'keep', 'changes': 'height from config; LUT recalibration is a detector-side concern',
     'priority': 'P2'},
    {'file': 'frontend/sensing/detector.py', 'current_role': '2D RD CA-CFAR, NMS, AoA; outputs '
     'position + vr_mps + peak_to_noise_db; hard 10-300 m and 70 deg gates; LUT keyed by post-FFT '
     'peak_to_noise_db', 'verdict': 'small change', 'changes': 'config-driven range/FOV gates; '
     'recalibrate covariance LUT under the new SNR definition; keep vr_mps output (needed for '
     'multi-BS velocity fusion)', 'priority': 'P1'},
    {'file': 'frontend/fusion/association.py', 'current_role': 'sequential Hungarian 3-BS position '
     'fusion (x,y only, inverse-covariance)', 'verdict': 'keep / optional extend',
     'changes': 'position fusion is reusable; radial-velocity fusion is handled by '
     'controlled_isac/velocity_fusion.py, so no mandatory change (P2 option: fuse velocity state)',
     'priority': 'P2'},
    {'file': 'frontend/tracking/cv_kf.py', 'current_role': 'anonymous CV Kalman tracker, 8 slots; '
     'assert dt=0.1 s; birth velocity 0 with variance 100; position-only updates',
     'verdict': 'small change', 'changes': 'dt=3/29.97 s (config); recalibrate q_a and birth '
     'inflation for the new dt and speed scale; velocity remains inferred from position unless '
     'radial-velocity fusion is wired in', 'priority': 'P0'},
    {'file': 'frontend/controlled_isac/measurement.py', 'current_role': 'truth polar + calibrated '
     'noise measurement generator bound to reports/f01a/geometry.json; auxiliary controlled study '
     'only, not the formal sensing chain', 'verdict': 'keep as auxiliary',
     'changes': 'rebind geometry/range/FOV/height to the Automatum candidate; replace episode/'
     'deadline seed keys with (scene, canonical frame, BS, vehicle); must stay out of the formal run',
     'priority': 'P1'},
    {'file': 'frontend/controlled_isac/cross_bs_association.py', 'current_role': 'position-only '
     'cross-BS association for controlled measurements', 'verdict': 'keep as auxiliary',
     'changes': 'reuse for smoke tests only; formal association is fusion/association.py',
     'priority': 'P2'},
    {'file': 'frontend/controlled_isac/velocity_fusion.py', 'current_role': 'multi-BS radial '
     'velocity -> vx,vy MAP with a 30 m/s prior (NGSIM scale)',
     'verdict': 'reuse in formal chain', 'changes': 'update velocity prior sigma to the Automatum '
     'speed scale (max ~31 m/s, P99 ~27.6), keep the estimator; wire into the fusion output',
     'priority': 'P1'},
    {'file': 'frontend/controlled_isac/temporal_association.py', 'current_role': 'frame-to-frame '
     'slot linking with dt=0.1 s', 'verdict': 'small change', 'changes': 'dt from config (3/29.97)',
     'priority': 'P1'},
    {'file': 'configs/shared_frontend.json', 'current_role': 'single production config: A01 '
     'geometry_source, visibility 10-300 m / 70 deg / 5 m, power.snr_ref_db with 100 m reference, '
     'detector covariance_lut -> f01e/snr_rebuild_06, calibration -> f01e report, tracker dt=0.1',
     'verdict': 'refactor', 'changes': 'replace geometry_source and power/calibration blocks; add '
     'dataset/scene block (canonical paths, split SHAs, dt, frame phase); move calibration '
     'references to the new SNR calibration output', 'priority': 'P0'},
    {'file': 'frontend/f01e_dataset.py', 'current_role': 'frozen F01-E packed dataset loader '
     '(per-SNR npz under data/f01e)', 'verdict': 'retire (historical chain)',
     'changes': 'superseded by the Automatum split/sample loader', 'priority': 'P3'},
    {'file': 'frontend/symbol_dataset.py', 'current_role': 'F01-E symbol-level prediction inputs '
     'used by QGAT/QGNN training', 'verdict': 'replace with new loader',
     'changes': 'new GNN/QGNN input builder over samples.npz + sensing cache; do not reuse the '
     'F01-E packing', 'priority': 'P2'},
    {'file': 'code/07_shared_frontend/* (probes, calibrations)', 'current_role': 'historical '
     'calibration/probe entry points calling echo_source, simulator, detector, fusion, tracker',
     'verdict': 'experiment-auxiliary', 'changes': 'keep as reference; rerun only after the new '
     'adapter and SNR definition exist', 'priority': 'P3'},
    {'file': 'experiments/* (qgnn/qgat runs, snr smoke)', 'current_role': 'F01-E prediction '
     'training/evaluation entries', 'verdict': 'out of scope this round',
     'changes': 'not to be run on Automatum until the new sample loader and sensing cache exist',
     'priority': 'P3'},
]

OPEN_DECISIONS = [
    {'id': 'D-01', 'question': 'sample-level vs scene-level echo generation',
     'recommendation': 'scene-level cache (route B)', 'status': 'recommend, pending confirmation'},
    {'id': 'D-02', 'question': 'freeze the candidate 3-BS coordinates',
     'recommendation': 'freeze after route B cache smoke test; candidates in candidate_bs_geometry',
     'status': 'CANDIDATE, not frozen'},
    {'id': 'D-03', 'question': 'range/FOV/height candidate',
     'recommendation': 'range_min 5 m, range_max 120 m, FOV half 70 deg, dh 5 m (6.5 m station / '
                       '1.5 m vehicle)', 'status': 'CANDIDATE'},
    {'id': 'D-04', 'question': 'SNR control definition and levels',
     'recommendation': 'per-BS pre-FFT SNR_b; re-derive snr_levels_db and recalibrate CFAR/LUT',
     'status': 'open'},
    {'id': 'D-05', 'question': 'velocity prior sigma for radial-velocity fusion',
     'recommendation': 'replace 30 m/s with Automatum-scale value (~15 m/s P90 of speed spread) '
                       'after observing fused geometry conditioning', 'status': 'open'},
    {'id': 'D-06', 'question': 'anonymous track <-> sample vehicle_id alignment',
     'recommendation': 'C-domain Hungarian alignment on position (+velocity), majority vote per '
                       'track; never fed back into sensing', 'status': 'recommend'},
    {'id': 'D-07', 'question': 'temporary vehicles in shared echo',
     'recommendation': 'include all physically present vehicles per frame (route B); the sample '
                       'cohort only defines prediction slots/labels', 'status': 'recommend'},
]

ROUTES = {
    'route_A_sample_level': {
        'description': 'each sample generates its own 20-frame echo from samples.npz history only',
        'future_leakage': 'none if history only is used',
        'deletes_temporary_vehicles': 'yes (cohort excludes vehicles without 40/40 presence)',
        'shared_echo_fidelity': 'low: echo contains only the cohort, not the physical scene',
        'same_frame_consistency': 'no: overlapping samples would re-simulate the same frame with '
                                  'sample-specific targets and noise',
        'compute': 'lowest (12,394 samples x 20 frames x 3 BS, but highly redundant)',
        'cache_reuse': 'none',
        'gnn_docking': 'direct from samples.npz history',
    },
    'route_B_scene_level': {
        'description': 'per scene/frame sensing cache over trajectories.csv, sliced to samples by '
                       'start_frame',
        'future_leakage': 'none if per-frame echoes are causal (treated state at frame t uses only '
                          'states up to t)',
        'deletes_temporary_vehicles': 'no: every physically present vehicle contributes',
        'shared_echo_fidelity': 'high: one shared echo per BS and frame',
        'same_frame_consistency': 'yes: identical echo for the same (scene, frame) across samples',
        'compute': 'higher upfront (~14k train frames x 3 BS) but one echo per frame',
        'cache_reuse': 'high: reusable by any window/split/stride and by ablations',
        'gnn_docking': 'slice cache by start_frame; sample cohort defines slots/labels only',
    },
    'recommendation': 'route_B_scene_level',
}

ANONYMOUS_IDENTITY = {
    'A_domain': 'simulator truth (positions/velocities/vehicle ids) never leaves the simulator; '
                'produces shared echoes and audit metadata only',
    'B_domain': 'detector/fusion/tracker receive echoes or anonymous measurements only; outputs '
                'anonymous track_key/slot states; no vehicle_id, UUID, future or GT identity',
    'C_domain': 'offline label builder / evaluation may use vehicle_id to score or to slice GT; '
                'must never feed scores, ids or GT states back into B',
    'recommended_matching': {
        'scope': 'per split, per scene, offline',
        'algorithm': 'per frame: Hungarian assignment between track positions (and fused velocity '
                     'when available) and GT sample vehicle states; cost = Mahalanobis position '
                     '(+ optional velocity term); gate at the association chi-square',
        'track_identity': 'each anonymous track_key -> vehicle_id by majority over its frames, '
                          'with vote-margin reported as confidence',
        'outputs': 'track_to_vehicle_mapping per split with per-track confidence; used only by '
                   'evaluation and label alignment code paths',
        'forbidden': 'mapping must not be an input to thresholds, gates, calibration or any '
                     'sensing decision',
    },
}

SNR_MIGRATION = {
    'new_definition': 'per BS b and frame: Y_b = S_b + W_b; SNR_b = 10log10(mean|S_b|^2/mean|W_b|^2), '
                      'pre-FFT, pre-detection, pre-fusion; no per-target SNR as the experiment control',
    'conflicts': [
        {'where': 'configs/shared_frontend.json -> power.snr_ref_db / reference_range_m=100 / '
                  'reference_rcs_m2 / fixed_rcs_m2 / noise_variance',
         'why': 'reference-range per-target SNR control; incompatible with frame-level shared SNR'},
        {'where': 'frontend/ofdm_echo.py -> noise_parameters(snr_db)',
         'why': 'noise variance derived from a 100 m reference gain; per-target SNR semantics'},
        {'where': 'frontend/sensing/simulator.py -> snr_ref_db amplitude law 10^(snr/20)(100/radii)^2',
         'why': 'per-target reference-range amplitude; single-target SNR control'},
        {'where': 'frontend/echo_source.py -> observation(snr_db=20) and pressure at 20 dB only',
         'why': 'episode-level SNR control on the retired F01-B adapter'},
        {'where': 'configs/shared_frontend.json -> snr_levels_db and detector covariance_lut '
                  '(reports/f01e/snr_rebuild_06)',
         'why': 'levels and LUT calibrated against the old SNR definition and post-FFT peak SNR'},
        {'where': 'frontend/f01e_dataset.py -> ALLOWED_SNR_DB per-SNR packed npz',
         'why': 'binds the experiment control to the retired F01-E chain'},
    ],
    'future_change_list': [
        'simulator: accept a frame SNR target and scale the summed clean echo, then add one noise '
        'realisation; report SNR_b per BS/frame',
        'config: replace power block with noise_variance + snr control levels; delete 100 m '
        'reference and RCS-to-SNR scaling from the control path',
        'detector/coords: recalibrate the covariance LUT against the new SNR definition (peak SNR '
        'may still key the LUT entry, but calibration must be produced under SNR_b control)',
        'reports: regenerate calibration summaries under the new definition; retire f01e SNR '
        'reports as calibration sources',
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Automatum ISAC adaptation audit (read-only)')
    parser.add_argument('--out', default='reports/isac_adaptation/automatum_isac_adaptation_audit.json')
    args = parser.parse_args()
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canonical = load_canonical()
    scenes = scene_stats(canonical)
    occupancy = sample_vs_scene_occupancy()
    geometry = candidate_bs_geometry(canonical)
    dependencies = code_dependency_scan()
    for scene_id in (0, 1):
        scenes[str(scene_id)]['junction_center_xy_m'] = \
            geometry['scenes']['scene_%d' % scene_id]['junction_center_xy_m']

    p0 = sum(1 for row in MIGRATION_MATRIX if row['priority'] == 'P0')
    p1 = sum(1 for row in MIGRATION_MATRIX if row['priority'] == 'P1')
    payload = {
        'audit': 'automatum_isac_adaptation',
        'mode': 'AUDIT ONLY (read-only; no data, code, model or split modification)',
        'dataset_baseline': {
            'canonical_path': CANONICAL_REL,
            'canonical_sha256': canonical['sha256'],
            'canonical_rows': canonical['rows'],
            'schema': canonical['header'].split(','),
            'canonical_fps': CANONICAL_FPS, 'canonical_dt_s': CANONICAL_DT,
            'source_fps': SOURCE_FPS, 'stride': STRIDE,
            'velocity_frame': 'world (body-frame rotation already applied in canonical)',
            'splits': {split: {
                'trajectories': 'data/automatum_t_crossing/splits/%s/trajectories.csv' % split,
                'trajectories_sha256': SPLIT_SHA256[split],
                'samples': 'data/automatum_t_crossing/splits/%s/samples.npz' % split,
                'samples_sha256': NPZ_SHA256[split],
                'history': [20, 8, 4],
                'future': [20, 8, 4], 'state': ['x', 'y', 'vx', 'vy'],
                'vehicle_rule': '2 <= N <= 8; cohort = present in all 40 frames'} for split in
                ('train', 'val', 'test')},
            'frozen_commit': 'd017d49d688c0cc92294674b764d3813fa21b327',
        },
        'scene_stats': scenes,
        'sample_vs_scene_occupancy': occupancy,
        'candidate_bs_geometry': geometry,
        'current_code_dependencies': dependencies,
        'migration_matrix': MIGRATION_MATRIX,
        'migration_priority_counts': {'P0': p0, 'P1': p1,
                                      'P2': sum(1 for r in MIGRATION_MATRIX if r['priority'] == 'P2'),
                                      'P3': sum(1 for r in MIGRATION_MATRIX if r['priority'] == 'P3')},
        'routes': ROUTES,
        'anonymous_identity': ANONYMOUS_IDENTITY,
        'snr_migration': SNR_MIGRATION,
        'open_decisions': OPEN_DECISIONS,
    }
    with out_path.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + '\n')
    summary = {
        'output': str(out_path),
        'canonical_sha256_match': canonical['sha256'] ==
        'a41ac4e24a8f21039dde60064868871b21093289a661ca9bef28ee7d1aab3442',
        'occupancy_overall': occupancy['overall'],
        'priorities': payload['migration_priority_counts'],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()