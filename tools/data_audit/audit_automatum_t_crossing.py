#!/usr/bin/env python3
"""AUDIT ONLY: full audit of the Automatum T-Crossing recordings.

Read-only. The script verifies the raw ZIP, loads the two dynamicWorld.json
recordings, and produces the evidence needed for the ICCT migration decision:

- real schema/structure of the files (not the vendor README)
- track continuity, position and velocity quality, physical plausibility
- concurrent vehicle counts, interaction intensity, N<=8 scene supply
- 29.97 Hz -> 10 Hz downsampling comparison (index level, no conversion)
- 2 s history + 2 s future window supply at 10 Hz (index level only)
- staticWorld.xodr map overlays and candidate 3-BS geometry feasibility

Outputs:
- reports/data_audit/automatum_t_crossing_stats.json
- reports/data_audit/automatum_vehicle_stats.csv
- reports/data_audit/automatum_anomalies.csv
- reports/data_audit/figures/automatum/*.png

Usage:
    python tools/data_audit/audit_automatum_t_crossing.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull

FPS = 29.97
DT = 1.0 / FPS
PS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
DURATION_BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
HIST_VEC_KEYS = ['ax_vec', 'ay_vec', 'curvature_vec', 'psi_vec', 'vx_vec', 'vy_vec',
                 'x_vec', 'y_vec', 'jerk_x_vec', 'jerk_y_vec']
AUX_KEYS = ['lane_id_vec', 'road_id_vec', 'road_type_list', 'object_relation_dict_list',
            'dist_along_road_segment', 'lane_change_flag_vec', 'ttc_dict_vec', 'tth_dict_vec',
            'lat_dist_dict_vec', 'long_dist_dict_vec', 'distance_left_lane_marking',
            'distance_right_lane_marking']
JUMP_SPEED_MPS = 45.0
ACCEL_ALERT_MPS2 = 8.0
HEADING_JUMP_RAD = 0.5
HEADING_JUMP_MIN_SPEED = 5.0
WINDOW_S = 4.0
WINDOW_FRAMES = int(round(WINDOW_S / 0.1))  # 40 intervals -> 41 samples
TEN_HZ = 0.1
START = time.time()
FIG_DPI = 150


def log(*args) -> None:
    print('[%6.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def stats(values, ps=PS) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'count': 0}
    q = np.percentile(values, ps)
    out = {'count': int(values.size), 'mean': float(values.mean()),
           'std': float(values.std(ddof=1)) if values.size > 1 else 0.0}
    out.update({'p%g' % p: float(x) for p, x in zip(ps, q)})
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def zip_section(zip_path: Path, extracted: Path) -> dict:
    listing = []
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        for info in zf.infolist():
            listing.append({'name': info.filename, 'size': info.file_size,
                            'compressed': info.compress_size, 'crc': '%08x' % info.CRC})
    recordings = sorted({name.split('/')[0] for name in (e['name'] for e in listing) if '/' in name})
    on_disk = {}
    for rec in recordings:
        for path in sorted((extracted / rec).rglob('*')):
            if path.is_file():
                on_disk[str(path.relative_to(extracted))] = path.stat().st_size
    return {
        'path': str(zip_path), 'size_bytes': zip_path.stat().st_size,
        'sha256': sha256_file(zip_path), 'testzip_corrupt_member': bad,
        'member_count': len(listing), 'members': listing,
        'recordings': recordings, 'extracted_root': str(extracted),
        'extracted_files': on_disk,
        'extracted_total_bytes': int(sum(on_disk.values())),
    }


def load_world(path: Path) -> dict:
    with path.open(encoding='utf-8') as fh:
        return json.load(fh)


def extract_objects(world: dict) -> list:
    objects = []
    for index, raw in enumerate(world['objects']):
        obj = {
            'index': index, 'uuid': raw['UUID'], 'obj_type': raw['objType'],
            'length': float(raw['length']), 'width': float(raw['width']),
            't': np.asarray(raw['time'], dtype=np.float64),
            'x': np.asarray(raw['x_vec'], dtype=np.float64),
            'y': np.asarray(raw['y_vec'], dtype=np.float64),
            'psi': np.asarray(raw['psi_vec'], dtype=np.float64),
            'vx': np.asarray(raw['vx_vec'], dtype=np.float64),
            'vy': np.asarray(raw['vy_vec'], dtype=np.float64),
            'ax': np.asarray(raw['ax_vec'], dtype=np.float64),
            'ay': np.asarray(raw['ay_vec'], dtype=np.float64),
            'jx': np.asarray(raw['jerk_x_vec'], dtype=np.float64),
            'jy': np.asarray(raw['jerk_y_vec'], dtype=np.float64),
            'kappa': np.asarray(raw['curvature_vec'], dtype=np.float64),
            'empty_aux': [k for k in AUX_KEYS if len(raw[k]) == 0],
            'nonempty_aux': {k: len(raw[k]) for k in AUX_KEYS if len(raw[k]) > 0},
            'length_mismatch': [k for k in HIST_VEC_KEYS if len(raw[k]) != len(raw['time'])],
        }
        objects.append(obj)
    return objects


def load_recording(record_dir: Path) -> dict:
    world = load_world(record_dir / 'dynamicWorld.json')
    objects = extract_objects(world)
    xodr_path = record_dir / 'staticWorld.xodr'
    html_files = sorted(record_dir.glob('*.html'))
    x = np.concatenate([o['x'] for o in objects])
    y = np.concatenate([o['y'] for o in objects])
    hull = ConvexHull(np.column_stack([x, y]))
    return {
        'dir': record_dir, 'name': world['RecordingName'], 'uuid': world['UUID'],
        'release': world['Release'], 'utm_ref': world['UTM-ReferencePoint'],
        'wgs84_ref': world['WGS84-ReferencePoint'], 'web': world['Web'],
        'license': world['License'], 'contact': world['Contact'],
        'video': world['videoInfo'], 'object_counts': world['Object_counts'],
        'world': world, 'objects': objects,
        'xodr_path': xodr_path, 'html_files': [p.name for p in html_files],
        'track_bbox': [float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        'track_hull_area_m2': float(hull.volume),
        'dynamic_json_bytes': (record_dir / 'dynamicWorld.json').stat().st_size,
        'xodr_bytes': xodr_path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# continuity / kinematics / physical plausibility
# ---------------------------------------------------------------------------

def continuity_audit(rec: dict) -> dict:
    dts = []
    non_monotonic = duplicate_t = off_grid = mismatch = 0
    for obj in rec['objects']:
        t = obj['t']
        if len(t) > 1:
            d = np.diff(t)
            dts.append(d)
            if np.any(d <= 0):
                non_monotonic += 1
        if len(np.unique(t)) != len(t):
            duplicate_t += 1
        k = t / DT
        if np.max(np.abs(k - np.round(k))) > 1e-3:
            off_grid += 1
        if obj['length_mismatch']:
            mismatch += 1
    dts = np.concatenate(dts)
    return {
        'tracks': len(rec['objects']),
        'track_points': int(sum(len(o['t']) for o in rec['objects'])),
        'dt': stats(dts, [0, 1, 50, 99, 99.9, 100]),
        'dt_equals_nominal_fraction': float((np.abs(dts - DT) <= 1e-9).mean()),
        'tracks_non_monotonic_time': non_monotonic,
        'tracks_duplicate_timestamps': duplicate_t,
        'tracks_with_vector_length_mismatch': mismatch,
        'tracks_not_on_29_97_grid': off_grid,
        'n_objects_with_empty_aux_vectors': int(sum(1 for o in rec['objects'] if o['empty_aux'])),
        'aux_vector_keys_present_but_empty': sorted({k for o in rec['objects'] for k in o['empty_aux']}),
        'aux_vector_keys_populated': sorted({k for o in rec['objects'] for k in o['nonempty_aux']}),
    }


def kinematic_audit(rec: dict) -> dict:
    rows, anomalies = [], []
    fd_mid_all, world_err_all, raw_err_all = [], [], []
    heading_err_moving, accel_all, jerk_all, speed_all = [], [], [], []
    jump_cases, mismatch_cases, hard_accel, heading_jumps = [], [], [], 0
    nonfinite = 0
    for obj in rec['objects']:
        t, x, y, psi = obj['t'], obj['x'], obj['y'], obj['psi']
        vx, vy, ax, ay = obj['vx'], obj['vy'], obj['ax'], obj['ay']
        nonfinite += int(sum(not np.all(np.isfinite(a)) for a in (t, x, y, psi, vx, vy, ax, ay)))
        speed = np.hypot(vx, vy)
        acc = np.hypot(ax, ay)
        jerk = np.hypot(obj['jx'], obj['jy'])
        speed_all.append(speed)
        accel_all.append(acc)
        jerk_all.append(jerk)
        n = len(t)
        if n < 2:
            continue
        dt = np.diff(t)
        dx, dy = np.diff(x), np.diff(y)
        fd = np.hypot(dx, dy) / dt
        fd_mid = fd - 0.5 * (speed[:-1] + speed[1:])
        fd_mid_all.append(fd_mid)
        wx = vx[:-1] * np.cos(psi[:-1]) - vy[:-1] * np.sin(psi[:-1])
        wy = vx[:-1] * np.sin(psi[:-1]) + vy[:-1] * np.cos(psi[:-1])
        world_err = np.hypot(dx / dt - wx, dy / dt - wy)
        raw_err = np.hypot(dx / dt - vx[:-1], dy / dt - vy[:-1])
        world_err_all.append(world_err)
        raw_err_all.append(raw_err)
        moving = speed[:-1] > HEADING_JUMP_MIN_SPEED
        circ_moving = np.array([])
        if moving.any():
            fd_ang = np.arctan2(dy, dx)
            circ = np.abs((psi[:-1] - fd_ang + np.pi) % (2 * np.pi) - np.pi)
            circ_moving = circ[moving]
            heading_err_moving.append(circ_moving)
        dpsi = np.diff(psi)
        step = (np.abs(dpsi) > HEADING_JUMP_RAD) & (speed[1:] > HEADING_JUMP_MIN_SPEED)
        heading_jumps += int(step.sum())
        jump_idx = np.nonzero(fd > JUMP_SPEED_MPS)[0]
        for i in jump_idx:
            jump_cases.append({
                'uuid': obj['uuid'], 'object_index': obj['index'], 'obj_type': obj['obj_type'],
                't_s': float(t[i]), 'frame': int(i), 'step_m': float(np.hypot(dx[i], dy[i])),
                'fd_speed_mps': float(fd[i]), 'official_speed_mps': float(speed[i])})
        mis_idx = np.nonzero(np.abs(fd_mid) > 5.0)[0]
        for i in mis_idx:
            mismatch_cases.append({
                'uuid': obj['uuid'], 'object_index': obj['index'], 't_s': float(t[i]),
                'fd_speed_mps': float(fd[i]), 'mid_speed_mps': float(0.5 * (speed[i] + speed[i + 1])),
                'err_mps': float(fd_mid[i])})
        hard_idx = np.nonzero(acc > ACCEL_ALERT_MPS2)[0]
        for i in hard_idx:
            hard_accel.append({
                'uuid': obj['uuid'], 'object_index': obj['index'], 'obj_type': obj['obj_type'],
                't_s': float(t[i]), 'accel_mps2': float(acc[i]), 'speed_mps': float(speed[i])})
        fd_valid = np.isfinite(fd)
        rows.append({
            'recording': rec['label'], 'object_index': obj['index'], 'uuid': obj['uuid'],
            'obj_type': obj['obj_type'], 'length_m': obj['length'], 'width_m': obj['width'],
            'n_points': n, 't_start_s': float(t[0]), 't_end_s': float(t[-1]),
            'duration_s': float(t[-1] - t[0]),
            'path_length_m': float(np.sum(np.hypot(dx, dy))),
            'net_displacement_m': float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
            'x_min': float(x.min()), 'x_max': float(x.max()),
            'y_min': float(y.min()), 'y_max': float(y.max()),
            'speed_median_mps': float(np.median(speed)), 'speed_p95_mps': float(np.percentile(speed, 95)),
            'speed_max_mps': float(speed.max()),
            'accel_median_mps2': float(np.median(acc)), 'accel_p95_mps2': float(np.percentile(acc, 95)),
            'accel_max_mps2': float(acc.max()), 'jerk_max_mps3': float(jerk.max()),
            'fd_speed_mae_mps': float(np.mean(np.abs(fd_mid[fd_valid]))) if n > 1 else None,
            'fd_speed_p95_abs_mps': float(np.percentile(np.abs(fd_mid[fd_valid]), 95)) if n > 1 else None,
            'fd_speed_max_abs_mps': float(np.max(np.abs(fd_mid[fd_valid]))) if n > 1 else None,
            'heading_err_p95_rad': float(np.percentile(circ_moving, 95)) if circ_moving.size else None,
            'max_step_m': float(np.max(np.hypot(dx, dy))),
            'max_step_speed_mps': float(np.max(fd)),
        })
        for i in np.nonzero(~np.isfinite(fd))[0]:
            anomalies.append({'recording': rec['label'], 'uuid': obj['uuid'], 'object_index': obj['index'],
                              'obj_type': obj['obj_type'], 'anomaly_type': 'nonfinite_step',
                              'time_s': float(t[i]), 'frame': int(i), 'value': None,
                              'detail': 'non-finite position step'})
    fd_mid_all = np.concatenate(fd_mid_all)
    world_err_all = np.concatenate(world_err_all)
    raw_err_all = np.concatenate(raw_err_all)
    heading_err_moving = np.concatenate(heading_err_moving)
    speed_all = np.concatenate(speed_all)
    accel_all = np.concatenate(accel_all)
    jerk_all = np.concatenate(jerk_all)
    err_abs = np.abs(fd_mid_all)
    return {
        'vehicle_rows': rows,
        'anomaly_rows': anomalies,
        'velocity_consistency': {
            'body_frame_check': {
                'fd_vs_official_midpoint': {
                    'mae_mps': float(np.mean(err_abs)), 'rmse_mps': float(np.sqrt(np.mean(fd_mid_all ** 2))),
                    'bias_mps': float(np.mean(fd_mid_all)),
                    'p95_abs_mps': float(np.percentile(err_abs, 95)),
                    'p99_abs_mps': float(np.percentile(err_abs, 99)),
                    'max_abs_mps': float(err_abs.max())},
                'fd_vs_official_left_sample_mae_mps': float(np.mean(np.abs(raw_err_all))),
                'fd_world_after_psi_rotation_mae_mps': float(np.mean(world_err_all)),
                'conclusion': 'vx/vy are body-frame (longitudinal/lateral), x/y world, psi heading; '
                              'rotation of (vx,vy) by psi matches finite differences'
                              if np.mean(world_err_all) < 0.3 and np.mean(err_abs) < 0.3
                              else 'INCONSISTENT - investigate'},
            'pairs_total': int(len(fd_mid_all)),
            'speed_quantization': stats(err_abs, [50, 90, 95, 99, 99.9, 100]),
        },
        'heading_consistency': {
            'moving_points': int(len(heading_err_moving)),
            'psi_vs_fd_direction': stats(heading_err_moving, [50, 90, 95, 99, 99.9, 100]),
            'heading_jump_events_over_0.5rad_moving': heading_jumps,
        },
        'physical': {
            'speed_mps': stats(speed_all, [0, 1, 50, 75, 90, 95, 99, 99.9, 100]),
            'accel_mps2': stats(accel_all, [0, 50, 90, 95, 99, 99.9, 100]),
            'jerk_mps3': stats(jerk_all, [50, 90, 95, 99, 99.9, 100]),
            'speed_kmh_max': float(speed_all.max() * 3.6),
            'events_fd_speed_over_45mps': len(jump_cases),
            'events_accel_over_8mps2': len(hard_accel),
            'jump_cases': jump_cases[:20],
            'hard_accel_cases': hard_accel[:20],
            'speed_mismatch_cases_over_5mps': len(mismatch_cases),
            'speed_mismatch_examples': mismatch_cases[:20],
            'nonfinite_vectors': nonfinite,
        },
    }


# ---------------------------------------------------------------------------
# concurrency / interaction
# ---------------------------------------------------------------------------

def frame_grid(rec: dict) -> int:
    t_end = max(o['t'][-1] for o in rec['objects'])
    return int(round(t_end / DT)) + 1


def active_lists(rec: dict, n_frames: int) -> list:
    active = [[] for _ in range(n_frames)]
    for oi, obj in enumerate(rec['objects']):
        k0 = int(round(obj['t'][0] / DT))
        k1 = int(round(obj['t'][-1] / DT))
        for k in range(k0, k1 + 1):
            active[k].append(oi)
    return active


def concurrency_audit(rec: dict, active: list) -> dict:
    counts = np.array([len(a) for a in active], dtype=np.int64)
    n = len(counts)
    bins = {
        'N=0': int((counts == 0).sum()),
        'N=1': int((counts == 1).sum()),
        '2<=N<=4': int(((counts >= 2) & (counts <= 4)).sum()),
        '5<=N<=8': int(((counts >= 5) & (counts <= 8)).sum()),
        'N>8': int((counts > 8).sum()),
    }
    return {
        'frames': n,
        'active_vehicle_count': stats(counts, [0, 1, 50, 75, 90, 95, 99, 99.9, 100]),
        'bins_frames': bins,
        'bins_fraction': {k: float(v / n) for k, v in bins.items()},
        'fraction_frames_at_least_2': float((counts >= 2).mean()),
        'fraction_frames_at_least_5': float((counts >= 5).mean()),
        'count_series_head': counts[:200].tolist(),
    }


def junction_from_xodr(roads: list, objects: list) -> dict:
    connectors = [r for r in roads if r['junction'] >= 0]
    if not connectors:
        return {'center': None, 'arms': []}
    trees = []
    for road in connectors:
        pts = np.column_stack([road['center_x'], road['center_y']])
        trees.append((pts, cKDTree(pts)))
    best = None
    for pts, _ in trees:
        for p in pts[::4]:
            worst = max(tree.query(p)[0] for _, tree in trees)
            if best is None or worst < best[0]:
                best = (worst, p)
    center = best[1] if best else None
    arms = []
    for road in roads:
        if road['junction'] >= 0:
            continue
        pts = np.column_stack([road['center_x'], road['center_y']])
        if len(pts) < 2:
            continue
        d = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
        i = int(d.argmin())
        hdg = road['center_hdg'][i]
        tangent = np.array([math.cos(hdg), math.sin(hdg)])
        d_start = float(np.hypot(pts[0, 0] - center[0], pts[0, 1] - center[1]))
        d_end = float(np.hypot(pts[-1, 0] - center[0], pts[-1, 1] - center[1]))
        if min(d_start, d_end) < 25.0:
            direction = tangent if d_start < d_end else -tangent
            arms.append({'road_id': road['id'], 'name': road['name'],
                         'point': pts[i].tolist(), 'direction': direction.tolist(),
                         'distance_to_center_m': float(d[i]), 'through_road': False})
        else:
            for direction in (tangent, -tangent):
                arms.append({'road_id': road['id'], 'name': road['name'],
                             'point': pts[i].tolist(), 'direction': direction.tolist(),
                             'distance_to_center_m': float(d[i]), 'through_road': True})
    deduped = []
    for arm in arms:
        angle = math.degrees(math.atan2(arm['direction'][1], arm['direction'][0])) % 360
        if any(abs((angle - math.degrees(math.atan2(a['direction'][1], a['direction'][0])) + 180) % 360
                   - 180) < 15.0 for a in deduped):
            continue
        deduped.append(arm)
    for arm in deduped:
        arm['axis_deg'] = math.degrees(math.atan2(arm['direction'][1], arm['direction'][0])) % 360
    return {'center': center.tolist() if center is not None else None,
            'center_max_distance_to_connector_m': float(best[0]) if best else None,
            'arms': deduped}


def interaction_audit(rec: dict, active: list, junction: dict) -> dict:
    pos = [np.column_stack([o['x'], o['y']]) for o in rec['objects']]
    psi = [o['psi'] for o in rec['objects']]
    speed = [np.hypot(o['vx'], o['vy']) for o in rec['objects']]
    starts = [int(round(o['t'][0] / DT)) for o in rec['objects']]
    nn, n10, n20, n30, hd = [], [], [], [], []
    veh_frames = 0
    frames_with_nn20 = 0
    pair_frames_30 = 0
    pair_following = pair_crossing = pair_oncoming = 0
    frames_crossing = 0
    junction_busy = 0
    junction_multi_arm = 0
    junction_multi_road = 0
    n_active_series = np.zeros(len(active), dtype=np.int64)
    per_frame_max_hd = np.zeros(len(active))
    center = np.array(junction['center']) if junction['center'] else None
    approach_axes = []
    arm_is_side = []
    if center is not None:
        for arm in junction['arms']:
            v = arm['direction']
            axis = math.atan2(v[1], v[0]) % (2 * math.pi)
            approach_axes.append(axis if arm['through_road'] else (axis + math.pi) % (2 * math.pi))
            arm_is_side.append(not arm['through_road'])
    for k, act in enumerate(active):
        n_active_series[k] = len(act)
        if center is not None and approach_axes:
            arms_seen = set()
            for oi in act:
                off = k - starts[oi]
                if speed[oi][off] <= 2.0:
                    continue
                p = pos[oi][off]
                if math.hypot(p[0] - center[0], p[1] - center[1]) > 60.0:
                    continue
                hdg = psi[oi][off] % (2 * math.pi)
                diffs = [abs((hdg - axis + math.pi) % (2 * math.pi) - math.pi) for axis in approach_axes]
                arms_seen.add(int(np.argmin(diffs)))
            if arms_seen:
                junction_busy += 1
                if len(arms_seen) >= 2:
                    junction_multi_arm += 1
                if any(arm_is_side[i] for i in arms_seen) and len(arms_seen) >= 2:
                    junction_multi_road += 1
        if len(act) < 2:
            continue
        pts = np.array([pos[oi][k - starts[oi]] for oi in act])
        ang = np.array([psi[oi][k - starts[oi]] for oi in act])
        veh_frames += len(act)
        dm = np.hypot(pts[:, 0, None] - pts[None, :, 0], pts[:, 1, None] - pts[None, :, 1])
        np.fill_diagonal(dm, np.inf)
        nn.append(dm.min(axis=1))
        n10.append((dm <= 10.0).sum(axis=1))
        n20.append((dm <= 20.0).sum(axis=1))
        n30.append((dm <= 30.0).sum(axis=1))
        if (dm <= 20.0).sum() > 0:
            frames_with_nn20 += 1
        iu = np.triu_indices(len(act), 1)
        d30 = dm[iu]
        pair_frames_30 += int((d30 <= 30.0).sum())
        diff = np.abs((ang[:, None] - ang[None, :] + np.pi) % (2 * np.pi) - np.pi)
        hd_pairs = diff[iu][d30 <= 30.0]
        if hd_pairs.size:
            hd.append(hd_pairs)
            per_frame_max_hd[k] = float(hd_pairs.max())
            pair_following += int((hd_pairs < math.radians(30)).sum())
            crossing = (hd_pairs >= math.radians(30)) & (hd_pairs <= math.radians(150))
            pair_crossing += int(crossing.sum())
            pair_oncoming += int((hd_pairs > math.radians(150)).sum())
            if crossing.any():
                frames_crossing += 1
    nn = np.concatenate(nn) if nn else np.array([])
    n10 = np.concatenate(n10) if n10 else np.array([])
    n20 = np.concatenate(n20) if n20 else np.array([])
    n30 = np.concatenate(n30) if n30 else np.array([])
    hd = np.concatenate(hd) if hd else np.array([])
    neighbors = {
        'nearest_neighbour_distance_m': stats(nn, [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]),
        'neighbours_within_10m': stats(n10, [50, 90, 95, 99, 100]),
        'neighbours_within_20m': stats(n20, [50, 90, 95, 99, 100]),
        'neighbours_within_30m': stats(n30, [50, 90, 95, 99, 100]),
        'heading_diff_within_30m_rad': stats(hd, [25, 50, 75, 90, 95, 100]),
        'vehicle_frames_with_any_neighbour_10m_fraction': float((n10 > 0).mean()) if n10.size else 0.0,
        'vehicle_frames_with_any_neighbour_20m_fraction': float((n20 > 0).mean()) if n20.size else 0.0,
        'vehicle_frames_with_any_neighbour_30m_fraction': float((n30 > 0).mean()) if n30.size else 0.0,
        'veh_frames': veh_frames,
        'frames_with_pair_within_20m_fraction': float(frames_with_nn20 / len(active)),
        'pairs_within_30m': pair_frames_30,
        'pairs_within_30m_following_lt30deg': pair_following,
        'pairs_within_30m_crossing_30_150deg': pair_crossing,
        'pairs_within_30m_oncoming_gt150deg': pair_oncoming,
        'frames_with_crossing_pair_fraction': float(frames_crossing / len(active)),
        'junction_frames_busy': junction_busy,
        'junction_frames_multi_arm': junction_multi_arm,
        'junction_frames_multi_arm_fraction': float(junction_multi_arm / junction_busy) if junction_busy else 0.0,
        'junction_frames_main_plus_side_road': junction_multi_road,
        'junction_frames_main_plus_side_road_fraction': float(junction_multi_road / junction_busy)
        if junction_busy else 0.0,
        'junction_zone_definition': 'moving vehicles (speed > 2 m/s) within 60 m of junction center; '
                                    'arm = nearest approach traffic direction; main+side = both a through-road '
                                    'direction and the side-road direction present',
    }
    reps = representative_windows(rec, per_frame_max_hd, n_active_series, active)
    return {'neighbours': neighbors, 'representative_windows': reps,
            'nearest_neighbour_values': nn}


def representative_windows(rec: dict, per_frame_max_hd: np.ndarray, n_active: np.ndarray,
                           active: list, top: int = 5, window_s: float = 4.0) -> dict:
    w = int(round(window_s / DT))
    score = np.zeros(len(n_active))
    interacting = (n_active >= 2) & (per_frame_max_hd > math.radians(45))
    score[interacting] = n_active[interacting] * (1.0 + per_frame_max_hd[interacting])
    kernel = np.ones(w)
    rolling = np.convolve(score, kernel, mode='valid')
    order = np.argsort(-rolling)
    picked = []
    for start in order:
        if rolling[start] <= 0:
            break
        if any(abs(start - p) < w for p in picked):
            continue
        picked.append(int(start))
        if len(picked) >= top:
            break
    windows = []
    for start in picked:
        frame_ids = range(start, min(start + w, len(n_active)))
        seen = set()
        for k in frame_ids:
            seen.update(active[k])
        examples = []
        for oi in sorted(seen):
            t0 = rec['objects'][oi]['t'][0]
            k0 = int(round(t0 / DT))
            off = max(0, min(start + w // 2 - k0, len(rec['objects'][oi]['t']) - 1))
            examples.append({'uuid': rec['objects'][oi]['uuid'], 'type': rec['objects'][oi]['obj_type'],
                             'x': float(rec['objects'][oi]['x'][off]),
                             'y': float(rec['objects'][oi]['y'][off]),
                             'speed_mps': float(np.hypot(rec['objects'][oi]['vx'][off],
                                                         rec['objects'][oi]['vy'][off]))})
        windows.append({'t_start_s': float(start * DT), 't_end_s': float(min(start + w, len(n_active)) * DT),
                        'max_active_vehicles': int(n_active[start:start + w].max()),
                        'mean_active_vehicles': float(n_active[start:start + w].mean()),
                        'interaction_score': float(rolling[start]),
                        'vehicle_types': {t: sum(1 for e in examples if e['type'] == t)
                                          for t in sorted({e['type'] for e in examples})},
                        'vehicles': examples})
    return {'definition': 'sliding 4 s window ranked by active vehicles x heading diversity',
            'windows': windows}


# ---------------------------------------------------------------------------
# staticWorld.xodr
# ---------------------------------------------------------------------------

def _poly(coeffs, x):
    a, b, c, d = coeffs
    return a + b * x + c * x * x + d * x * x * x


def _width_at(widths, s):
    active = widths[0]
    for entry in widths:
        if entry[0] <= s + 1e-9:
            active = entry
        else:
            break
    return _poly(active[1], s - active[0])


def _sample_geometry(geom: dict, ds: float):
    n = max(2, int(geom['length'] / ds) + 1)
    s_rel = np.linspace(0.0, geom['length'], n)
    x0, y0, hdg0 = geom['x'], geom['y'], geom['hdg']
    if geom['kind'] == 'line':
        x = x0 + s_rel * math.cos(hdg0)
        y = y0 + s_rel * math.sin(hdg0)
        hdg = np.full(n, hdg0)
    elif geom['kind'] == 'paramPoly3':
        p = s_rel if geom['pRange'] == 'arcLength' else s_rel / geom['length']
        aU, bU, cU, dU = geom['aU'], geom['bU'], geom['cU'], geom['dU']
        aV, bV, cV, dV = geom['aV'], geom['bV'], geom['cV'], geom['dV']
        u = aU + bU * p + cU * p ** 2 + dU * p ** 3
        v = aV + bV * p + cV * p ** 2 + dV * p ** 3
        cos, sin = math.cos(hdg0), math.sin(hdg0)
        x = x0 + u * cos - v * sin
        y = y0 + u * sin + v * cos
        du = bU + 2 * cU * p + 3 * dU * p ** 2
        dv = bV + 2 * cV * p + 3 * dV * p ** 2
        if geom['pRange'] != 'arcLength':
            du, dv = du / geom['length'], dv / geom['length']
        hdg = hdg0 + np.arctan2(dv, du)
    else:
        raise NotImplementedError('geometry kind %s' % geom['kind'])
    return s_rel, x, y, hdg


def parse_xodr(path: Path, ds: float = 0.5) -> list:
    text = path.read_text(encoding='utf-8', errors='replace')
    roads = []
    for match in re.finditer(r'<road\b([^>]*)>(.*?)</road>', text, re.S):
        head, body = match.group(1), match.group(2)
        road_id = int(re.search(r'\bid="(-?\d+)"', head).group(1))
        junction = int(re.search(r'\bjunction="(-?\d+)"', head).group(1))
        name = re.search(r'\bname="([^"]*)"', head).group(1)
        length = float(re.search(r'\blength="([^"]+)"', head).group(1))
        geoms = []
        for gm in re.finditer(r'<geometry\b([^>]*)>(.*?)</geometry>', body, re.S):
            attrs, inner = gm.group(1), gm.group(2)
            kind = re.search(r'<(line|arc|spiral|poly3|paramPoly3)\b([^>]*)/?>', inner).group(1)
            geom = {'kind': kind, 's': float(re.search(r'\bs="([^"]+)"', attrs).group(1)),
                    'x': float(re.search(r'\bx="([^"]+)"', attrs).group(1)),
                    'y': float(re.search(r'\by="([^"]+)"', attrs).group(1)),
                    'hdg': float(re.search(r'\bhdg="([^"]+)"', attrs).group(1)),
                    'length': float(re.search(r'\blength="([^"]+)"', attrs).group(1))}
            params = re.search(r'<(line|arc|spiral|poly3|paramPoly3)\b([^>]*)/?>', inner).group(2)
            if kind == 'paramPoly3':
                geom.update({k: float(re.search(r'%s="([^"]+)"' % k, params).group(1))
                             for k in ('aU', 'bU', 'cU', 'dU', 'aV', 'bV', 'cV', 'dV')})
                geom['pRange'] = re.search(r'pRange="([^"]+)"', params).group(1)
            geoms.append(geom)
        offsets = [(float(m.group(1)), (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                        float(m.group(5))))
                   for m in re.finditer(r'<laneOffset s="([^"]+)" a="([^"]+)" b="([^"]+)" c="([^"]+)" d="([^"]+)"',
                                        body)]
        sections = []
        for sec in re.finditer(r'<laneSection s="([^"]+)">(.*?)</laneSection>', body, re.S):
            sec_s = float(sec.group(1))
            lanes = {}
            for lm in re.finditer(r'<lane id="(-?\d+)" type="(\w+)">(.*?)</lane>', sec.group(2), re.S):
                lid, ltype, lbody = int(lm.group(1)), lm.group(2), lm.group(3)
                widths = [(float(w.group(1)), (float(w.group(2)), float(w.group(3)),
                                               float(w.group(4)), float(w.group(5))))
                          for w in re.finditer(r'<width sOffset="([^"]+)" a="([^"]+)" b="([^"]+)" '
                                               r'c="([^"]+)" d="([^"]+)"', lbody)]
                lanes[lid] = {'id': lid, 'type': ltype, 'widths': widths}
            sections.append({'s': sec_s, 'lanes': lanes})
        center_parts = []
        for geom in geoms:
            s_rel, x, y, hdg = _sample_geometry(geom, ds)
            center_parts.append((geom['s'] + s_rel, x, y, hdg))
        s = np.concatenate([p[0] for p in center_parts])
        roads.append({'id': road_id, 'name': name, 'length': length, 'junction': junction,
                      'geoms': geoms, 'lane_offsets': offsets, 'sections': sections,
                      'center_s': s,
                      'center_x': np.concatenate([p[1] for p in center_parts]),
                      'center_y': np.concatenate([p[2] for p in center_parts]),
                      'center_hdg': np.concatenate([p[3] for p in center_parts])})
    return roads


def lane_layout(road: dict, s_values: np.ndarray):
    if not road['sections']:
        return []
    offsets = road['lane_offsets'] or [(0.0, (0.0, 0.0, 0.0, 0.0))]
    base = np.array([_poly(offsets[0][1], s - offsets[0][0]) for s in s_values])
    out = []
    for sec in road['sections']:
        lanes = sec['lanes']
        centers, edges = {}, {}
        off = base.copy()
        edges['left_0'] = off
        for lid in sorted([i for i in lanes if i > 0]):
            w = np.array([_width_at(lanes[lid]['widths'], s) for s in s_values])
            centers['L%d' % lid] = off + 0.5 * w
            off = off + w
            edges['left_%d' % lid] = off
        off = base.copy()
        edges['right_0'] = off
        for lid in sorted([i for i in lanes if i < 0], reverse=True):
            w = np.array([_width_at(lanes[lid]['widths'], s) for s in s_values])
            centers['R%d' % lid] = off - 0.5 * w
            off = off - w
            edges['right_%d' % lid] = off
        out.append({'s': sec['s'], 'centers': centers, 'edges': edges, 'lanes': lanes})
    return out


def lane_centerline_samples(roads: list) -> tuple:
    pts, half_widths = [], []
    for road in roads:
        s = road['center_s']
        for sec in lane_layout(road, s):
            for lid, lane in sec['lanes'].items():
                if lane['type'] != 'driving':
                    continue
                off = sec['centers'].get(('L%d' % lid) if lid > 0 else ('R%d' % lid))
                if off is None:
                    continue
                w = np.array([_width_at(lane['widths'], sv) for sv in s])
                x = road['center_x'] - np.sin(road['center_hdg']) * off
                y = road['center_y'] + np.cos(road['center_hdg']) * off
                pts.append(np.column_stack([x, y]))
                half_widths.append(0.5 * w)
    return np.concatenate(pts), np.concatenate(half_widths)


def on_road_audit(rec: dict, roads: list) -> dict:
    pts, half = lane_centerline_samples(roads)
    tree = cKDTree(pts)
    residuals = []
    worst = []
    for obj in rec['objects']:
        q = np.column_stack([obj['x'], obj['y']])
        d, idx = tree.query(q, workers=-1)
        res = d - half[idx]
        residuals.append(res)
        on = float((res <= 0.5).mean())
        worst.append((on, obj['uuid'], obj['obj_type'], float(np.percentile(res, 95))))
    residuals = np.concatenate(residuals)
    worst.sort()
    return {
        'definition': 'residual = distance(point, nearest driving-lane centerline) - lane half width; '
                      '<=0 means inside the lane surface',
        'points': int(residuals.size),
        'fraction_residual_le_0m': float((residuals <= 0).mean()),
        'fraction_residual_le_0.5m': float((residuals <= 0.5).mean()),
        'fraction_residual_le_1m': float((residuals <= 1.0).mean()),
        'residual_m': stats(residuals, [50, 75, 90, 95, 99, 99.9, 100]),
        'worst_tracks_fraction_on_road': [
            {'uuid': u, 'type': t, 'fraction_le_0.5m': o, 'p95_residual_m': p} for o, u, t, p in worst[:10]],
    }


# ---------------------------------------------------------------------------
# 10 Hz feasibility and 2 s + 2 s windows
# ---------------------------------------------------------------------------

def downsample_audit(rec: dict) -> dict:
    duration = max(o['t'][-1] for o in rec['objects'])
    n10 = int(math.floor(duration / TEN_HZ)) + 1
    k = np.arange(n10)
    idx_a = 3 * k
    idx_b = np.round(k * TEN_HZ / DT).astype(np.int64)
    t_a = idx_a * DT
    t_b = idx_b * DT
    ideal = k * TEN_HZ
    err_a = t_a - ideal
    err_b = t_b - ideal
    stride_b = np.diff(idx_b)
    vals, counts = np.unique(stride_b, return_counts=True)
    pos_delta = []
    speed_delta = []
    for obj in rec['objects']:
        n = len(obj['t'])
        ka = idx_a[idx_a < n]
        kb = idx_b[idx_b < n]
        m = min(len(ka), len(kb))
        if m < 2:
            continue
        pa = np.column_stack([obj['x'][ka[:m]], obj['y'][ka[:m]]])
        pb = np.column_stack([obj['x'][kb[:m]], obj['y'][kb[:m]]])
        pos_delta.append(np.hypot(*(pa - pb).T))
        sa = np.hypot(obj['vx'][ka[:m]], obj['vy'][ka[:m]])
        sb = np.hypot(obj['vx'][kb[:m]], obj['vy'][kb[:m]])
        speed_delta.append(sa - sb)
    pos_delta = np.concatenate(pos_delta)
    speed_delta = np.concatenate(speed_delta)
    return {
        'nominal_source_rate_hz': FPS,
        'source_dt_s': DT,
        'effective_rate_of_every_3rd_frame_hz': 1.0 / (3 * DT),
        'method_A_every_3rd': {
            'dt_s': 3 * DT,
            'dt_is_uniform': True,
            'time_error_vs_0.1s_grid_s': stats(err_a, [0, 50, 99, 100]),
            'max_abs_time_error_s': float(np.abs(err_a).max()),
            'drift_at_end_s': float(err_a[-1]),
        },
        'method_B_nearest_0.1s_grid': {
            'sample_stride_counts': {str(int(v)): int(c) for v, c in zip(vals, counts)},
            'time_error_vs_0.1s_grid_s': stats(err_b, [0, 50, 99, 100]),
            'max_abs_time_error_s': float(np.abs(err_b).max()),
            'drift_at_end_s': float(err_b[-1]),
            'effective_mean_rate_hz': float((n10 - 1) / (t_b[-1] - t_b[0])),
        },
        'position_difference_A_minus_B_m': stats(pos_delta, [50, 90, 95, 99, 99.9, 100]),
        'speed_difference_A_minus_B_mps': stats(speed_delta, [50, 90, 95, 99, 99.9, 100]),
        'method_A_within_one_4s_window_time_error_s': float(3 * DT * WINDOW_FRAMES - WINDOW_S),
        'recommendation': 'method_B_nearest_0.1s_grid',
    }


def window_audit(rec: dict) -> dict:
    n10 = int(np.ceil((max(o['t'][-1] for o in rec['objects']) + 1e-9) / TEN_HZ))
    delta = np.zeros(n10 + 1, dtype=np.int64)
    per_object = []
    for obj in rec['objects']:
        k0 = int(math.ceil(round(obj['t'][0] / TEN_HZ, 6)))
        k1 = int(math.floor(round(obj['t'][-1] / TEN_HZ, 6)))
        anchors = max(0, k1 - k0 - WINDOW_FRAMES + 1)
        per_object.append({'uuid': obj['uuid'], 'type': obj['obj_type'],
                           't0_s': float(obj['t'][0]), 't1_s': float(obj['t'][-1]),
                           'duration_s': float(obj['t'][-1] - obj['t'][0]),
                           'anchors_10hz': int(anchors)})
        if anchors > 0:
            delta[k0] += 1
            delta[k1 - WINDOW_FRAMES + 1] -= 1
    concurrent = np.cumsum(delta[:-1])
    bins = {
        'N=0': int((concurrent == 0).sum()),
        'N=1': int((concurrent == 1).sum()),
        '2<=N<=4': int(((concurrent >= 2) & (concurrent <= 4)).sum()),
        '5<=N<=8': int(((concurrent >= 5) & (concurrent <= 8)).sum()),
        'N>8': int((concurrent > 8).sum()),
    }
    total_anchors = int(concurrent.sum())
    return {
        'definition': '41 consecutive 10 Hz samples (4.0 s) per window; anchor grid 0.1 s; '
                      'window needs t in [k*0.1, k*0.1+4.0] fully inside the track',
        'single_vehicle_windows_total': total_anchors,
        'windows_in_multi_vehicle_scenes': int(concurrent[concurrent >= 2].sum()),
        'windows_in_2_to_8_scenes': int(concurrent[(concurrent >= 2) & (concurrent <= 8)].sum()),
        'windows_in_more_than_8_scenes': int(concurrent[concurrent > 8].sum()),
        'multi_vehicle_anchors': int((concurrent >= 2).sum()),
        'multi_2_to_8_anchors': int(((concurrent >= 2) & (concurrent <= 8)).sum()),
        'anchors_with_more_than_8': int((concurrent > 8).sum()),
        'concurrent_windows': stats(concurrent, [50, 75, 90, 95, 99, 99.9, 100]),
        'concurrent_bins_anchors': bins,
        'per_object': per_object,
    }


# ---------------------------------------------------------------------------
# 3-BS geometry feasibility
# ---------------------------------------------------------------------------

def bs_audit(rec: dict, junction: dict, roads: list) -> dict:
    center = np.array(junction['center'])
    bs_positions = []
    for arm in junction['arms']:
        d = np.array(arm['direction'])
        perp = np.array([-d[1], d[0]])
        p = np.array(arm['point']) + 30.0 * d + 8.0 * perp
        bs_positions.append({'x': float(p[0]), 'y': float(p[1]), 'arm_road_id': arm['road_id'],
                             'arm_name': arm['name'], 'through_road': arm['through_road'],
                             'axis_deg': arm['axis_deg']})
    pts = np.concatenate([np.column_stack([o['x'], o['y']]) for o in rec['objects']])
    dist = np.column_stack([np.hypot(pts[:, 0] - b['x'], pts[:, 1] - b['y']) for b in bs_positions])
    nearest = dist.min(axis=1)
    tri = np.array([[b['x'], b['y']] for b in bs_positions])
    area = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0])) if len(tri) == 3 else 0.0
    sides = [float(np.hypot(*(tri[i] - tri[j]))) for i, j in [(0, 1), (1, 2), (2, 0)]] if len(tri) == 3 else []
    angles = []
    for i in range(len(tri)):
        v1 = tri[(i + 1) % len(tri)] - tri[i]
        v2 = tri[(i + 2) % len(tri)] - tri[i]
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return {
        'candidate_bs_positions': bs_positions,
        'junction_center': junction['center'],
        'junction_center_max_distance_to_connector_m': junction['center_max_distance_to_connector_m'],
        'arms': [{'road_id': a['road_id'], 'name': a['name'], 'through_road': a['through_road'],
                  'axis_deg': a['axis_deg'], 'point': a['point'],
                  'distance_to_center_m': a['distance_to_center_m']} for a in junction['arms']],
        'arm_axes_deg': [a['axis_deg'] for a in junction['arms']],
        'vehicle_to_nearest_bs_m': stats(nearest, [0, 1, 50, 75, 90, 95, 99, 100]),
        'vehicle_to_each_bs_m': {str(b['arm_road_id']): stats(dist[:, i], [50, 95, 100])
                                 for i, b in enumerate(bs_positions)},
        'fraction_vehicles_within_100m': float((nearest <= 100).mean()),
        'fraction_vehicles_within_150m': float((nearest <= 150).mean()),
        'triangle_area_m2': float(area),
        'triangle_sides_m': sides,
        'triangle_angles_deg': angles,
        'min_triangle_angle_deg': float(min(angles)) if angles else None,
        'is_non_collinear': bool(area > 50.0 and min(angles) > 5.0) if angles else False,
        'scene_bbox_m': rec['track_bbox'],
        'scene_hull_area_m2': rec['track_hull_area_m2'],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _lane_polylines(road: dict):
    s = road['center_s']
    cos, sin = np.cos(road['center_hdg']), np.sin(road['center_hdg'])
    for sec in lane_layout(road, s):
        for name, off in sec['edges'].items():
            yield name, road['center_x'] - sin * off, road['center_y'] + cos * off


TYPE_COLORS = {'car': '#1f77b4', 'truck': '#d62728', 'van': '#2ca02c'}


def fig_map(rec: dict, roads: list, junction: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    for ax in axes:
        for road in roads:
            for _, x, y in _lane_polylines(road):
                ax.plot(x, y, color='0.75', lw=0.5, zorder=1)
            ax.plot(road['center_x'], road['center_y'], color='0.45', lw=0.6, ls='--', zorder=1)
        for obj in rec['objects']:
            ax.plot(obj['x'], obj['y'], color=TYPE_COLORS.get(obj['obj_type'], 'k'),
                    lw=0.5, alpha=0.45, zorder=2)
        if junction['center']:
            ax.plot(*junction['center'], marker='x', color='k', ms=8, zorder=4)
        ax.set_aspect('equal')
        ax.set_xlabel('local x [m]')
        ax.set_ylabel('local y [m]')
    xs = [o['x'] for o in rec['objects']]
    ys = [o['y'] for o in rec['objects']]
    axes[0].set_title('%s\nfull scene, tracks by type (blue car / red truck / green van)' % rec['name'])
    axes[0].set_xlim(min(x.min() for x in xs) - 5, max(x.max() for x in xs) + 5)
    axes[0].set_ylim(min(y.min() for y in ys) - 5, max(y.max() for y in ys) + 5)
    if junction['center']:
        cx, cy = junction['center']
        axes[1].set_xlim(cx - 55, cx + 55)
        axes[1].set_ylim(cy - 55, cy + 55)
        axes[1].set_title('junction zoom (50 m radius)')
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_duration(recs: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.38
    labels = ['<2', '2-4', '4-6', '6-10', '10-20', '>20']
    x = np.arange(len(labels))
    for i, rec in enumerate(recs):
        durations = np.array([o['t'][-1] - o['t'][0] for o in rec['objects']])
        counts = [int(((durations >= lo) & (durations < hi)).sum()) for lo, hi in DURATION_BINS]
        axes[0].bar(x + (i - 0.5) * width, counts, width, label=rec['label'])
        axes[1].hist(durations, bins=40, histtype='step', lw=1.8, label=rec['label'])
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel('track duration [s]')
    axes[0].set_ylabel('tracks')
    axes[0].set_title('track duration bins')
    axes[0].legend()
    axes[1].set_xlabel('track duration [s]')
    axes[1].set_ylabel('tracks')
    axes[1].set_yscale('log')
    axes[1].set_title('track duration histogram')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_concurrency(rec: dict, counts: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    t = np.arange(len(counts)) * DT
    axes[0].plot(t, counts, lw=0.5)
    axes[0].set_xlabel('recording time [s]')
    axes[0].set_ylabel('active vehicles')
    axes[0].set_title('%s concurrent vehicles' % rec['label'])
    max_n = int(counts.max())
    bins = np.arange(-0.5, max_n + 1.5)
    axes[1].hist(counts, bins=bins, color='#1f77b4', alpha=0.85)
    axes[1].set_xlabel('active vehicles N')
    axes[1].set_ylabel('frames')
    axes[1].set_yscale('log')
    p90 = np.percentile(counts, 90)
    p99 = np.percentile(counts, 99)
    axes[1].set_title('mean %.2f  p90 %.1f  p99 %.1f  max %d' % (counts.mean(), p90, p99, counts.max()))
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_speed(recs: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rec in recs:
        speeds = np.concatenate([np.hypot(o['vx'], o['vy']) for o in rec['objects']]) * 3.6
        axes[0].hist(speeds, bins=np.arange(0, 140, 2), histtype='step', lw=1.8, label=rec['label'])
        axes[1].hist(speeds, bins=np.arange(0, 140, 2), histtype='step', lw=1.8, log=True,
                     label=rec['label'])
    for ax in axes:
        ax.set_xlabel('speed [km/h]')
        ax.set_ylabel('samples')
        ax.legend()
    axes[0].set_title('speed distribution (official vx/vy)')
    axes[1].set_title('speed distribution, log scale')
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_velocity_consistency(recs: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for rec in recs:
        err = []
        for obj in rec['objects']:
            if len(obj['t']) < 2:
                continue
            dt = np.diff(obj['t'])
            fd = np.hypot(np.diff(obj['x']), np.diff(obj['y'])) / dt
            sp = 0.5 * (np.hypot(obj['vx'], obj['vy'])[:-1] + np.hypot(obj['vx'], obj['vy'])[1:])
            err.append(np.abs(fd - sp))
        err = np.concatenate(err)
        axes[0].hist(err, bins=np.arange(0, 1.01, 0.02), histtype='step', lw=1.8, label=rec['label'])
        axes[1].hist(np.log10(np.maximum(err, 1e-9)), bins=60, histtype='step', lw=1.8, label=rec['label'])
        per_track = []
        for obj in rec['objects']:
            if len(obj['t']) < 2:
                continue
            dt = np.diff(obj['t'])
            fd = np.hypot(np.diff(obj['x']), np.diff(obj['y'])) / dt
            sp = 0.5 * (np.hypot(obj['vx'], obj['vy'])[:-1] + np.hypot(obj['vx'], obj['vy'])[1:])
            per_track.append((float(np.mean(np.abs(fd - sp))), float(np.median(np.hypot(obj['vx'], obj['vy'])))))
        per_track = np.array(per_track)
        axes[2].scatter(per_track[:, 1] * 3.6, per_track[:, 0], s=8, alpha=0.5, label=rec['label'])
    axes[0].set_xlabel('|FD speed - official speed| [m/s]')
    axes[0].set_ylabel('samples')
    axes[0].set_yscale('log')
    axes[0].set_title('finite-difference vs official speed')
    axes[1].set_xlabel('log10 error [m/s]')
    axes[1].set_ylabel('samples')
    axes[1].set_title('error distribution (log)')
    axes[2].set_xlabel('median speed [km/h]')
    axes[2].set_ylabel('per-track MAE [m/s]')
    axes[2].set_title('per-track velocity consistency')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_nearest_neighbour(recs: list, cached: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rec in recs:
        nn = cached[rec['label']]
        axes[0].hist(nn, bins=np.arange(0, 100, 2), histtype='step', lw=1.8, label=rec['label'])
        axes[1].hist(nn, bins=np.arange(0, 100, 2), histtype='step', lw=1.8, log=True,
                     label=rec['label'])
    for ax in axes:
        ax.set_xlabel('nearest neighbour distance [m]')
        ax.set_ylabel('vehicle-frames')
        ax.legend()
    axes[0].set_title('nearest neighbour distance (vehicle-frames with a neighbour)')
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_window_supply(recs: list, windows: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['N=0', 'N=1', '2<=N<=4', '5<=N<=8', 'N>8']
    x = np.arange(len(labels))
    width = 0.38
    for i, rec in enumerate(recs):
        bins = windows[rec['label']]['concurrent_bins_anchors']
        vals = [bins[k] for k in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, label=rec['label'])
    ax.set_xticks(x, labels)
    ax.set_xlabel('vehicles with a full 4 s window at the same anchor')
    ax.set_ylabel('10 Hz anchors')
    ax.set_yscale('log')
    ax.set_title('2 s history + 2 s future window supply at 10 Hz')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_bs_layout(rec: dict, roads: list, junction: dict, bs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for road in roads:
        for _, x, y in _lane_polylines(road):
            ax.plot(x, y, color='0.8', lw=0.6, zorder=1)
    for obj in rec['objects']:
        ax.plot(obj['x'], obj['y'], color='0.55', lw=0.4, alpha=0.5, zorder=2)
    cx, cy = junction['center']
    ax.plot(cx, cy, marker='x', color='k', ms=10, zorder=4, label='junction center')
    for i, b in enumerate(bs['candidate_bs_positions']):
        ax.plot(b['x'], b['y'], marker='^', ms=12, color='#d62728', zorder=5,
                label='candidate BS' if i == 0 else None)
        ax.annotate('BS on road %d' % b['arm_road_id'], (b['x'], b['y']),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)
    for arm in junction['arms']:
        d = np.array(arm['direction'])
        ax.plot([cx, cx + 55 * d[0]], [cy, cy + 55 * d[1]], color='#d62728', lw=1.2, ls=':', zorder=3)
    ax.set_aspect('equal')
    ax.set_xlim(cx - 75, cx + 75)
    ax.set_ylim(cy - 75, cy + 75)
    ax.set_xlabel('local x [m]')
    ax.set_ylabel('local y [m]')
    ax.set_title('candidate 3-BS layout (geometry feasibility only)\n%s' % rec['name'])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Full read-only audit of the Automatum T-Crossing data')
    parser.add_argument('--zip', default='data/automatum_t_crossing/raw/automatum_data_crossing.zip')
    parser.add_argument('--extracted', default='data/automatum_t_crossing/raw/extracted')
    parser.add_argument('--out-dir', default='reports/data_audit')
    parser.add_argument('--fig-dir', default='reports/data_audit/figures/automatum')
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    extracted = Path(args.extracted).resolve()
    out_dir = Path(args.out_dir).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log('verifying zip', zip_path)
    zip_info = zip_section(zip_path, extracted)
    log('zip sha256', zip_info['sha256'], 'testzip', zip_info['testzip_corrupt_member'])

    labels = ['A', 'B']
    recs = []
    dirs = [d for d in sorted(extracted.iterdir()) if d.is_dir()]
    if len(dirs) != 2:
        raise SystemExit('expected 2 extracted recordings in %s, found %d' % (extracted, len(dirs)))
    for label, record_dir in zip(labels, dirs):
        log('loading', record_dir.name)
        rec = load_recording(record_dir)
        rec['label'] = 'recording_%s' % label
        rec['duration_s'] = float(max(o['t'][-1] for o in rec['objects']))
        recs.append(rec)

    for rec in recs:
        log(rec['label'], rec['name'], rec['duration_s'], 's', len(rec['objects']), 'objects')

    report = {
        'audit': 'automatum_t_crossing_full_audit',
        'mode': 'AUDIT ONLY (no raw data modified; no downsampling/cleaning/renumbering performed)',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'zip': zip_info,
        'recordings': {},
        'cross_recording': {},
    }

    checks = {}

    for rec in recs:
        log(rec['label'], 'continuity')
        cont = continuity_audit(rec)
        log(rec['label'], 'kinematics')
        kin = kinematic_audit(rec)
        n_frames = frame_grid(rec)
        rec['n_frames'] = n_frames
        rec['duration_s'] = float((n_frames - 1) * DT)
        active = active_lists(rec, n_frames)
        log(rec['label'], 'concurrency')
        conc = concurrency_audit(rec, active)
        log(rec['label'], 'xodr')
        roads = parse_xodr(rec['xodr_path'])
        junction = junction_from_xodr(roads, rec['objects'])
        log(rec['label'], 'on-road check')
        on_road = on_road_audit(rec, roads)
        log(rec['label'], 'interaction')
        inter = interaction_audit(rec, active, junction)
        rec['_nn'] = inter.pop('nearest_neighbour_values')
        log(rec['label'], '10 Hz feasibility')
        ds = downsample_audit(rec)
        log(rec['label'], '2s+2s windows')
        win = window_audit(rec)
        log(rec['label'], '3-BS geometry')
        bs = bs_audit(rec, junction, roads)
        durations = np.array([o['t'][-1] - o['t'][0] for o in rec['objects']])
        duration_bins = {('%s' % label_): int(((durations >= lo) & (durations < hi)).sum())
                         for (lo, hi), label_ in zip(DURATION_BINS,
                                                     ['<2s', '2-4s', '4-6s', '6-10s', '10-20s', '>20s'])}
        rec_report = {
            'label': rec['label'], 'name': rec['name'], 'uuid': rec['uuid'],
            'source_dir': rec['dir'].name, 'release': rec['release'],
            'utm_reference_point': rec['utm_ref'], 'wgs84_reference_point': rec['wgs84_ref'],
            'video': rec['video'], 'duration_s': rec['duration_s'], 'n_frames': n_frames,
            'video_frame_count': int(rec['video']['frame_count']),
            'video_frame_count_matches_timestamp_grid': bool(int(rec['video']['frame_count']) == n_frames),
            'mean_sampling_rate_hz': float(1.0 / DT),
            'vehicle_type_counts': rec['object_counts'],
            'objects': len(rec['objects']),
            'object_types_in_objects_list': {t: int(sum(1 for o in rec['objects'] if o['obj_type'] == t))
                                             for t in sorted({o['obj_type'] for o in rec['objects']})},
            'schema': {
                'object_keys': sorted(rec['world']['objects'][0].keys()),
                'state_vectors': HIST_VEC_KEYS,
                'aux_vectors_present_but_empty': cont['aux_vector_keys_present_but_empty'],
                'vector_length_mismatch_tracks': cont['tracks_with_vector_length_mismatch'],
                'time_base': 'per-object time vector in seconds from recording start, 1/29.97 s grid',
                'coordinate_base': 'local metric Cartesian frame (UTM reference point given in metadata); '
                                   'same frame as staticWorld.xodr',
                'velocity_frame': 'body frame: vx longitudinal, vy lateral; world velocity = R(psi) @ (vx, vy)',
            },
            'duration_bins': duration_bins,
            'tracks_ge_4s': int((durations >= 4.0).sum()),
            'continuity': cont,
            'kinematics': {k: v for k, v in kin.items() if k != 'vehicle_rows'},
            'concurrency': {k: v for k, v in conc.items() if k != 'count_series_head'},
            'interaction': inter,
            'on_road': on_road,
            'downsample_10hz': ds,
            'windows_2s_2s': {k: v for k, v in win.items() if k != 'per_object'},
            'bs_geometry': bs,
            'files': {'dynamicWorld_json_bytes': rec['dynamic_json_bytes'],
                      'staticWorld_xodr_bytes': rec['xodr_bytes'], 'html': rec['html_files'],
                      'track_bbox': rec['track_bbox'], 'track_hull_area_m2': rec['track_hull_area_m2']},
        }
        report['recordings'][rec['label']] = rec_report
        rec['_kin'] = kin
        rec['_active'] = active
        rec['_counts'] = np.array([len(a) for a in active])
        rec['_roads'] = roads
        rec['_junction'] = junction
        rec['_bs'] = bs

        checks['%s_dt_all_nominal' % rec['label']] = bool(cont['dt_equals_nominal_fraction'] == 1.0)
        checks['%s_tracks_monotonic' % rec['label']] = bool(cont['tracks_non_monotonic_time'] == 0)
        checks['%s_no_duplicate_timestamps' % rec['label']] = bool(cont['tracks_duplicate_timestamps'] == 0)
        checks['%s_vector_lengths_consistent' % rec['label']] = bool(cont['tracks_with_vector_length_mismatch'] == 0)
        checks['%s_velocity_body_frame_matches_fd' % rec['label']] = bool(
            kin['velocity_consistency']['body_frame_check']['fd_world_after_psi_rotation_mae_mps'] < 0.3)
        checks['%s_bs_non_collinear' % rec['label']] = bool(bs['is_non_collinear'])
        checks['%s_tracks_on_road_ge_99pct' % rec['label']] = bool(on_road['fraction_residual_le_0.5m'] >= 0.99)

    a, b = recs
    uuids_a = {o['uuid'] for o in a['objects']}
    uuids_b = {o['uuid'] for o in b['objects']}
    report['cross_recording'] = {
        'uuid_overlap': len(uuids_a & uuids_b),
        'uuid_unique_within_recording': [len(uuids_a) == len(a['objects']), len(uuids_b) == len(b['objects'])],
        'recommended_trace_key': '(recording_id, UUID) or equivalently (recording_name, UUID); '
                                 'UUIDs are globally unique in this release',
        'recommended_final_vehicle_id_rule': 'vehicle_id = stable hash or integer index of '
                                             '(recording_name, UUID); do not rely on the objects list order',
        'total_objects': len(a['objects']) + len(b['objects']),
        'combined_duration_s': a['duration_s'] + b['duration_s'],
    }
    wa = report['recordings']['recording_A']['windows_2s_2s']
    wb = report['recordings']['recording_B']['windows_2s_2s']
    report['cross_recording'].update({
        'single_vehicle_windows_total': wa['single_vehicle_windows_total'] + wb['single_vehicle_windows_total'],
        'multi_vehicle_windows_total': wa['windows_in_multi_vehicle_scenes']
        + wb['windows_in_multi_vehicle_scenes'],
        'windows_in_2_to_8_scenes_total': wa['windows_in_2_to_8_scenes'] + wb['windows_in_2_to_8_scenes'],
        'windows_in_more_than_8_scenes_total': wa['windows_in_more_than_8_scenes']
        + wb['windows_in_more_than_8_scenes'],
    })

    report['checks'] = checks

    log('figures')
    rec_a, rec_b = recs
    fig_map(rec_a, rec_a['_roads'], rec_a['_junction'], fig_dir / 'map_tracks_recording_A.png')
    fig_map(rec_b, rec_b['_roads'], rec_b['_junction'], fig_dir / 'map_tracks_recording_B.png')
    fig_duration(recs, fig_dir / 'track_duration_distribution.png')
    fig_concurrency(rec_a, rec_a['_counts'], fig_dir / 'concurrent_vehicle_count_A.png')
    fig_concurrency(rec_b, rec_b['_counts'], fig_dir / 'concurrent_vehicle_count_B.png')
    fig_speed(recs, fig_dir / 'speed_distribution.png')
    fig_velocity_consistency(recs, fig_dir / 'position_velocity_consistency.png')
    fig_bs_layout(rec_a, rec_a['_roads'], rec_a['_junction'], rec_a['_bs'],
                  fig_dir / 'bs_layout_recording_A.png')
    fig_bs_layout(rec_b, rec_b['_roads'], rec_b['_junction'], rec_b['_bs'],
                  fig_dir / 'bs_layout_recording_B.png')
    nn_cache = {rec['label']: rec['_nn'] for rec in recs}
    fig_nearest_neighbour(recs, nn_cache, fig_dir / 'nearest_neighbour_distance.png')
    windows_cache = {rec['label']: report['recordings'][rec['label']]['windows_2s_2s'] for rec in recs}
    fig_window_supply(recs, windows_cache, fig_dir / 'multi_vehicle_window_supply.png')

    with (out_dir / 'automatum_t_crossing_stats.json').open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=float)

    vehicle_rows = []
    for rec in recs:
        vehicle_rows.extend(rec['_kin']['vehicle_rows'])
    vehicle_fields = list(vehicle_rows[0].keys())
    with (out_dir / 'automatum_vehicle_stats.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=vehicle_fields)
        writer.writeheader()
        writer.writerows(vehicle_rows)

    anomaly_rows = []
    for rec in recs:
        kin = rec['_kin']
        anomaly_rows.extend(kin['anomaly_rows'])
        for case in kin['physical']['jump_cases']:
            anomaly_rows.append({'recording': rec['label'], 'uuid': case['uuid'],
                                 'object_index': case['object_index'], 'obj_type': case['obj_type'],
                                 'anomaly_type': 'position_jump_over_45mps', 'time_s': case['t_s'],
                                 'frame': case['frame'], 'value': case['fd_speed_mps'],
                                 'detail': 'step %.3f m, official speed %.2f m/s'
                                           % (case['step_m'], case['official_speed_mps'])})
        for case in kin['physical']['speed_mismatch_examples']:
            anomaly_rows.append({'recording': rec['label'], 'uuid': case['uuid'],
                                 'object_index': case['object_index'], 'obj_type': None,
                                 'anomaly_type': 'fd_vs_official_speed_over_5mps', 'time_s': case['t_s'],
                                 'frame': None, 'value': case['err_mps'],
                                 'detail': 'fd %.2f, official midpoint %.2f' % (case['fd_speed_mps'],
                                                                                case['mid_speed_mps'])})
        for case in kin['physical']['hard_accel_cases']:
            anomaly_rows.append({'recording': rec['label'], 'uuid': case['uuid'],
                                 'object_index': case['object_index'], 'obj_type': case['obj_type'],
                                 'anomaly_type': 'accel_over_8mps2 (may be real braking)',
                                 'time_s': case['t_s'], 'frame': None, 'value': case['accel_mps2'],
                                 'detail': 'speed %.2f m/s' % case['speed_mps']})
    anomaly_fields = ['recording', 'uuid', 'object_index', 'obj_type', 'anomaly_type',
                      'time_s', 'frame', 'value', 'detail']
    with (out_dir / 'automatum_anomalies.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=anomaly_fields)
        writer.writeheader()
        writer.writerows(anomaly_rows)

    summary = {
        'outputs': [str(out_dir / 'automatum_t_crossing_stats.json'),
                    str(out_dir / 'automatum_vehicle_stats.csv'),
                    str(out_dir / 'automatum_anomalies.csv')],
        'figures': sorted(p.name for p in fig_dir.glob('*.png')),
        'checks': checks,
        'anomalies': len(anomaly_rows),
        'vehicles': len(vehicle_rows),
    }
    log('done')
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()