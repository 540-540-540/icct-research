#!/usr/bin/env python3
"""Build the canonical ICCT trajectory mother dataset from the raw NGSIM CSV.

Deterministic, read-only on the raw file. Rules frozen by the source-session audit
(commit cdf3a28, reports/data_cleaning/source_session_audit.md):

- two source sessions are separated by Frame_ID clock offset:
  session 0 offset 1118935679900 ms, session 1 offset 1118936699900 ms;
- vehicle key during cleaning is (source_session_id, original Vehicle_ID);
- the 4 confirmed cross-session duplicate vehicles are merged with the real session
  boundary as the switch point (session 0 before the boundary, session 1 at/after it);
- timestamp = (Global_Time - global_min_Global_Time) / 1000 (seconds, one timeline);
- x, y = Global_X/Y * 0.3048 (metres, no translation/normalisation);
- vx, vy = causal difference on the canonical x/y (m/s): frame i>0 uses the backward
  difference (x[i]-x[i-1])/0.1; the first frame is kept and uses the forward difference
  (x[1]-x[0])/0.1, so no true trajectory point is deleted; no smoothing;
- vehicle_id = 1..N, ordered by first timestamp, ties by (source_session_id, original
  Vehicle_ID); the mapping table records the provenance.

Outputs:
- trajectories_clean.csv   columns exactly: vehicle_id,timestamp,x,y,vx,vy
- vehicle_id_mapping.csv   columns exactly: vehicle_id,source_session_id,original_vehicle_id
- canonical_cleaning_stats.build.json (row accounting, seam checks, SHA256)

Usage:
    python tools/data_cleaning/build_canonical_trajectories.py \
        --csv /home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv \
        --output /home/dell/YrM/ICCT/data/processed/trajectories_clean.csv \
        --mapping /home/dell/YrM/ICCT/data/processed/vehicle_id_mapping.csv \
        --stats reports/data_cleaning/canonical_cleaning_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEET_TO_M = 0.3048
DT_S = 0.1
EXPECTED_OFFSETS_MS = [1118935679900, 1118936699900]
# confirmed duplicates: (session_0_vehicle_id, session_1_vehicle_id) for the same vehicle
DUPLICATE_PAIRS = [(1418, 2), (1421, 5), (1420, 6), (1422, 7)]
USE_COLUMNS = ['Vehicle_ID', 'Frame_ID', 'Global_Time', 'Global_X', 'Global_Y', 'v_Vel']


def log(*args) -> None:
    print('[%7.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Build canonical trajectories (read-only on raw)')
    parser.add_argument('--csv', default='/home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv')
    parser.add_argument('--output', default='/home/dell/YrM/ICCT/data/processed/trajectories_clean.csv')
    parser.add_argument('--mapping', default='/home/dell/YrM/ICCT/data/processed/vehicle_id_mapping.csv')
    parser.add_argument('--stats', default='reports/data_cleaning/canonical_cleaning_stats.json')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)
    mapping_path = Path(args.mapping)
    stats_path = Path(args.stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    raw_sha256 = sha256(csv_path)
    frame = pd.read_csv(csv_path, usecols=USE_COLUMNS)
    log('raw loaded', len(frame), 'rows', raw_sha256[:12])

    global_time = frame.Global_Time.to_numpy()
    clock_offset = global_time - frame.Frame_ID.to_numpy() * 100
    offsets = np.sort(np.unique(clock_offset))
    if offsets.tolist() != EXPECTED_OFFSETS_MS:
        raise SystemExit(f'unexpected session clock offsets: {offsets.tolist()}')
    session = np.searchsorted(offsets, clock_offset)
    frame['session'] = session
    t_origin = int(global_time.min())
    boundary_ms = int(global_time[session == 1].min())
    log('t_origin', t_origin, 'boundary', boundary_ms)

    # canonical source id: merged pairs share one negative id, regular tracks use session*10^6+vid
    vehicle = frame.Vehicle_ID.to_numpy()
    source_id = session.astype(np.int64) * 1_000_000 + vehicle
    merge_dropped = 0
    pair_rows = []
    for index, (vid0, vid1) in enumerate(DUPLICATE_PAIRS):
        mask0 = (session == 0) & (vehicle == vid0)
        mask1 = (session == 1) & (vehicle == vid1)
        rows0 = int(mask0.sum())
        rows1 = int(mask1.sum())
        if rows0 == 0 or rows1 == 0:
            raise SystemExit(f'expected duplicate pair missing: session0 {vid0} / session1 {vid1}')
        keep0 = mask0 & (global_time < boundary_ms)
        keep1 = mask1 & (global_time >= boundary_ms)
        dropped0 = int((mask0 & ~keep0).sum())
        dropped1 = int((mask1 & ~keep1).sum())
        merge_dropped += dropped0 + dropped1
        source_id[keep0 | keep1] = -(index + 1)
        pair_rows.append({'session0_vehicle_id': vid0, 'session1_vehicle_id': vid1,
                          'session0_rows': rows0, 'session1_rows': rows1,
                          'kept_session0_rows_before_boundary': int(keep0.sum()),
                          'kept_session1_rows_from_boundary': int(keep1.sum()),
                          'dropped_session0_rows_at_or_after_boundary': dropped0,
                          'dropped_session1_rows_before_boundary': dropped1})
    dropped_mask = np.zeros(len(frame), dtype=bool)
    for vid0, vid1 in DUPLICATE_PAIRS:
        merged_keep = ((session == 0) & (vehicle == vid0) & (global_time < boundary_ms)) | \
                      ((session == 1) & (vehicle == vid1) & (global_time >= boundary_ms))
        merged_all = ((session == 0) & (vehicle == vid0)) | ((session == 1) & (vehicle == vid1))
        dropped_mask |= merged_all & ~merged_keep
    frame = frame.loc[~dropped_mask].reset_index(drop=True)
    source_id = source_id[~dropped_mask]
    log('merge dropped', merge_dropped, 'remaining', len(frame))

    frame['source_id'] = source_id
    frame['x'] = frame.Global_X.to_numpy() * FEET_TO_M
    frame['y'] = frame.Global_Y.to_numpy() * FEET_TO_M
    frame = frame.sort_values(['source_id', 'Global_Time'], kind='mergesort').reset_index(drop=True)

    output_rows = []
    first_points_kept = 0
    seam_checks = []
    order_records = []
    for source_value, group in frame.groupby('source_id', sort=False):
        times = group.Global_Time.to_numpy()
        xs = group.x.to_numpy()
        ys = group.y.to_numpy()
        count = len(group)
        if count < 2:
            raise SystemExit(f'trajectory with <2 points: source_id {source_value}')
        if np.any(np.diff(times) != 100):
            raise SystemExit(f'non-100ms gap inside source_id {source_value}')
        vx = np.empty(count)
        vy = np.empty(count)
        vx[1:] = (xs[1:] - xs[:-1]) / DT_S
        vy[1:] = (ys[1:] - ys[:-1]) / DT_S
        vx[0] = (xs[1] - xs[0]) / DT_S
        vy[0] = (ys[1] - ys[0]) / DT_S
        block = pd.DataFrame({
            'source_id': source_value,
            'timestamp': (times - t_origin) / 1000.0,
            'x': xs, 'y': ys, 'vx': vx, 'vy': vy,
        })
        output_rows.append(block)
        first_points_kept += 1
        if source_value < 0:
            pair_index = int(-source_value - 1)
            vid0, vid1 = DUPLICATE_PAIRS[pair_index]
            s0 = group[(group.session == 0)]
            s1 = group[(group.session == 1)]
            last0 = s0.iloc[-1]
            first1 = s1.iloc[0]
            seam = {
                'session0_vehicle_id': vid0, 'session1_vehicle_id': vid1,
                'kept_session0_rows': int(len(s0)), 'kept_session1_rows': int(len(s1)),
                'seam_time_s': float((first1.Global_Time - t_origin) / 1000.0),
                'dt_at_seam_s': float((first1.Global_Time - last0.Global_Time) / 1000.0),
                'dx_m': float(first1.x - last0.x), 'dy_m': float(first1.y - last0.y),
                'jump_m': float(np.hypot(first1.x - last0.x, first1.y - last0.y)),
                'speed_pre_mps': (float(np.hypot(last0.x - s0.iloc[-2].x, last0.y - s0.iloc[-2].y) / DT_S)
                                  if len(s0) >= 2 else None),
                'speed_post_mps': (float(np.hypot(s1.iloc[1].x - s1.iloc[0].x,
                                                  s1.iloc[1].y - s1.iloc[0].y) / DT_S)
                                   if len(s1) >= 2 else None),
            }
            seam['seam_implied_speed_mps'] = (None if seam['dt_at_seam_s'] <= 0
                                              else seam['jump_m'] / seam['dt_at_seam_s'])
            seam_checks.append(seam)
        order_records.append({'source_id': int(source_value),
                              'first_time_ms': int(times[0]),
                              'primary_session': int(group.session.iloc[0]),
                              'primary_vehicle_id': int(group.Vehicle_ID.iloc[0])})

    trajectories = pd.concat(output_rows, ignore_index=True)
    order_table = pd.DataFrame(order_records)
    order_table = order_table.sort_values(['first_time_ms', 'primary_session', 'primary_vehicle_id'],
                                          kind='mergesort').reset_index(drop=True)
    order_table['vehicle_id'] = np.arange(1, len(order_table) + 1, dtype=np.int64)
    id_map = dict(zip(order_table.source_id.to_numpy(), order_table.vehicle_id.to_numpy()))
    trajectories['vehicle_id'] = trajectories.source_id.map(id_map).astype(np.int64)
    trajectories = trajectories[['vehicle_id', 'timestamp', 'x', 'y', 'vx', 'vy']]
    trajectories = trajectories.sort_values(['timestamp', 'vehicle_id'], kind='mergesort').reset_index(drop=True)

    source_to_vehicle = {}
    for _, record in order_table.iterrows():
        sid = int(record.source_id)
        source_to_vehicle[sid] = int(record.vehicle_id)
    filled = []
    for index, (vid0, vid1) in enumerate(DUPLICATE_PAIRS):
        vehicle_id = source_to_vehicle[-(index + 1)]
        filled.append({'vehicle_id': vehicle_id, 'source_session_id': 0, 'original_vehicle_id': vid0})
        filled.append({'vehicle_id': vehicle_id, 'source_session_id': 1, 'original_vehicle_id': vid1})
    regular = []
    for sid, vehicle_id in source_to_vehicle.items():
        if sid > 0:
            regular.append({'vehicle_id': vehicle_id, 'source_session_id': int(sid // 1_000_000),
                            'original_vehicle_id': int(sid % 1_000_000)})
    mapping = pd.DataFrame(regular + filled).sort_values(['vehicle_id', 'source_session_id'],
                                                         kind='mergesort').reset_index(drop=True)

    trajectories.to_csv(output_path, index=False, float_format='%.6f')
    mapping.to_csv(mapping_path, index=False)
    log('written', output_path, mapping_path)

    # v_Vel sanity check on kept rows (raw scalar speed never used as vx/vy)
    raw_speed = frame.set_index(['session', 'Vehicle_ID', 'Global_Time']).v_Vel * FEET_TO_M
    check = trajectories.copy()
    check['Global_Time'] = np.rint(check.timestamp * 1000).astype(np.int64) + t_origin
    check_groups = {int(v): g for v, g in check.groupby('vehicle_id', sort=False)}
    key_map = {}
    for row in mapping.itertuples(index=False):
        key_map.setdefault(row.vehicle_id, []).append((row.source_session_id, row.original_vehicle_id))
    matched_speed = []
    computed_speed = []
    for vehicle_id, sources in key_map.items():
        rows = check_groups[int(vehicle_id)]
        for s, vid in sources:
            if len(sources) == 1:
                subset = rows
            elif s == 0:
                subset = rows[rows.Global_Time < boundary_ms]
            else:
                subset = rows[rows.Global_Time >= boundary_ms]
            for row in subset.itertuples(index=False):
                key = (s, vid, row.Global_Time)
                if key in raw_speed.index:
                    matched_speed.append(float(raw_speed.loc[key]))
                    computed_speed.append(float(np.hypot(row.vx, row.vy)))
    matched_speed = np.asarray(matched_speed)
    computed_speed = np.asarray(computed_speed)
    diff = computed_speed - matched_speed
    v_vel_sanity = {
        'matched_rows': int(len(matched_speed)),
        'total_rows': int(len(trajectories)),
        'matched_fraction': float(len(matched_speed) / max(len(trajectories), 1)),
        'raw_v_vel_mps': {'median': float(np.median(matched_speed)) if len(matched_speed) else None,
                          'p90': float(np.percentile(matched_speed, 90)) if len(matched_speed) else None},
        'computed_speed_mps': {'median': float(np.median(computed_speed)) if len(computed_speed) else None,
                               'p90': float(np.percentile(computed_speed, 90)) if len(computed_speed) else None},
        'difference_mps': {'median': float(np.median(diff)) if len(diff) else None,
                           'p90_abs': float(np.percentile(np.abs(diff), 90)) if len(diff) else None},
        'pearson': float(np.corrcoef(matched_speed, computed_speed)[0, 1]) if len(matched_speed) > 2 else None,
    }
    log('v_vel sanity done', v_vel_sanity['matched_rows'])

    stats = {
        'kind': 'canonical_build',
        'input_csv': str(csv_path), 'input_sha256': raw_sha256,
        'global_time_origin_ms': t_origin, 'session_boundary_ms': boundary_ms,
        'dt_s': DT_S, 'feet_to_m': FEET_TO_M,
        'raw_rows': int(dropped_mask.size),
        'session_clock_offsets_ms': offsets.tolist(),
        'duplicate_pairs': pair_rows,
        'row_accounting': {
            'raw_rows': int(dropped_mask.size),
            'merge_dropped_rows': int(merge_dropped),
            'first_points_kept': int(first_points_kept),
            'first_point_velocity_rule': 'forward difference (x[1]-x[0])/0.1; no point deleted',
            'final_rows': int(len(trajectories)),
            'check_raw_minus_drops': int(dropped_mask.size - merge_dropped),
        },
        'counts': {
            'canonical_vehicles': int(trajectories.vehicle_id.nunique()),
            'mapping_rows': int(len(mapping)),
            'merged_vehicle_ids': [int(source_to_vehicle[-(i + 1)]) for i in range(len(DUPLICATE_PAIRS))],
        },
        'seam_checks': seam_checks,
        'v_vel_sanity': v_vel_sanity,
        'timestamps': {'min_s': float(trajectories.timestamp.min()),
                       'max_s': float(trajectories.timestamp.max())},
        'outputs': {
            str(output_path): {'size': output_path.stat().st_size,
                               'lines': int(len(trajectories)) + 1,
                               'sha256': sha256(output_path)},
            str(mapping_path): {'size': mapping_path.stat().st_size,
                                'lines': int(len(mapping)) + 1,
                                'sha256': sha256(mapping_path)},
        },
        'runtime_s': time.time() - START,
    }
    stats_path.write_text(json.dumps(stats, indent=1, default=float) + '\n', encoding='utf-8')
    log('stats written', stats_path)
    print(json.dumps({'vehicles': stats['counts']['canonical_vehicles'],
                      'final_rows': stats['row_accounting']['final_rows'],
                      'merge_dropped': merge_dropped,
                      'first_points_kept': first_points_kept,
                      'output': str(output_path)}, default=float))


if __name__ == '__main__':
    START = time.time()
    main()