#!/usr/bin/env python3
"""AUDIT ONLY: structure of the two source sessions in the NGSIM Lankershim raw CSV.

This script answers five questions with read-only evidence:

1. How many source recordings/sessions does the CSV contain, and what are their exact boundaries?
2. What is the ~100.7 s Global_Time overlap between them?
3. Do the two sessions contain duplicated vehicle trajectories in that overlap?
4. Can Vehicle_ID be used as a global vehicle key?
5. What is the CSV row ordering, and which time-step anomalies are file-order artifacts?

No data is modified. The script reads the CSV and writes one JSON report.

Usage:
    python tools/data_cleaning/audit_source_sessions.py \
        --csv /home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv \
        --output reports/data_cleaning/source_session_stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEET_TO_M = 0.3048
PS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
RADII_M = [0.5, 1.0, 2.0, 3.0, 5.0]
SHIFTS_S = [0.0, 510.0, 1020.0, 1530.0]


def log(*args) -> None:
    print('[%7.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'count': 0}
    quantiles = np.percentile(values, PS)
    out = {'count': int(values.size), 'mean': float(values.mean()),
           'std': float(values.std(ddof=1)) if values.size > 1 else 0.0}
    out.update({'p%g' % p: float(q) for p, q in zip(PS, quantiles)})
    return out


def sample(values, n=10):
    values = list(values)
    return values[:n]


def leap(value):
    return None if value is None else round(float(value), 6)


def load(csv_path: Path) -> pd.DataFrame:
    columns = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y',
               'Global_X', 'Global_Y', 'v_Vel', 'v_Acc', 'Lane_ID', 'Direction']
    frame = pd.read_csv(csv_path, usecols=columns)
    raw_time = frame.Global_Time.to_numpy()
    frame_offset = raw_time - frame.Frame_ID.to_numpy() * 100
    offsets = np.sort(np.unique(frame_offset))
    frame['session'] = np.searchsorted(offsets, frame_offset)
    return frame, offsets


def session_summary(frame: pd.DataFrame, offsets: np.ndarray) -> dict:
    out = {'global_time_frame_offsets_ms': offsets.tolist(),
           'offset_delta_ms': int(offsets[1] - offsets[0]) if len(offsets) > 1 else None,
           'sessions': {}}
    for session in sorted(frame.session.unique()):
        block = frame[frame.session == session]
        tracks = block.groupby('Vehicle_ID')
        sizes = tracks.size()
        total_frames_match = float((tracks.Total_Frames.first().to_numpy() == sizes.to_numpy()).mean())
        out['sessions'][str(session)] = {
            'rows': int(len(block)),
            'unique_vehicle_id': int(block.Vehicle_ID.nunique()),
            'global_time_min_ms': int(block.Global_Time.min()),
            'global_time_max_ms': int(block.Global_Time.max()),
            'duration_s': float((block.Global_Time.max() - block.Global_Time.min()) / 1000.0),
            'frame_id_min': int(block.Frame_ID.min()),
            'frame_id_max': int(block.Frame_ID.max()),
            'unique_frames': int(block.Global_Time.nunique()),
            'rows_per_frame_mean': float(len(block) / block.Global_Time.nunique()),
            'tracks': int(len(sizes)),
            'track_points': stats(sizes),
            'total_frames_equals_track_points_fraction': total_frames_match,
            'local_x_ft_range': [float(block.Local_X.min()), float(block.Local_X.max())],
            'local_y_ft_range': [float(block.Local_Y.min()), float(block.Local_Y.max())],
            'global_x_ft_range': [float(block.Global_X.min()), float(block.Global_X.max())],
            'global_y_ft_range': [float(block.Global_Y.min()), float(block.Global_Y.max())],
        }
    return out


def file_layout(frame: pd.DataFrame) -> dict:
    session = frame.session.to_numpy()
    vehicle = frame.Vehicle_ID.to_numpy()
    new_run = (vehicle[1:] != vehicle[:-1]) | (session[1:] != session[:-1])
    run_id = np.concatenate([[0], np.cumsum(new_run)])
    frame = frame.assign(run=run_id)
    runs = frame.groupby('run', sort=False)
    run_info = runs.agg(session=('session', 'first'), vehicle=('Vehicle_ID', 'first'),
                        n=('Vehicle_ID', 'size'), t0=('Global_Time', 'min'),
                        t1=('Global_Time', 'max'), row0=('Vehicle_ID', lambda x: x.index[0]))
    key_runs = run_info.groupby(['session', 'vehicle']).size()
    multi = key_runs[key_runs > 1]
    switch_rows = np.nonzero(np.diff(session) != 0)[0]
    within_session_sorted = {}
    for s in (0, 1):
        block = frame[frame.session == s]
        within_session_sorted[str(s)] = bool(np.all(np.diff(block.Vehicle_ID.to_numpy()) >= 0))
    multi_details = []
    for (s, v), _count in multi.items():
        rows = run_info[(run_info.session == s) & (run_info.vehicle == v)]
        multi_details.append({
            'session': int(s), 'vehicle_id': int(v),
            'runs': [{'rows': int(r.n), 'first_frame': int(r.row0), 't0_ms': int(r.t0), 't1_ms': int(r.t1)}
                     for _, r in rows.iterrows()]})
    return {
        'session_switch_rows': switch_rows.tolist(),
        'run_count': int(len(run_info)),
        'unique_session_vehicle_keys': int(frame.groupby(['session', 'Vehicle_ID']).ngroups),
        'vehicle_id_monotone_within_session': within_session_sorted,
        'keys_with_multiple_runs': len(multi),
        'keys_with_multiple_runs_detail': multi_details[:20],
        'first_rows': frame.head(3)[['Vehicle_ID', 'Frame_ID', 'Global_Time', 'session']].to_dict('records'),
        'last_rows': frame.tail(3)[['Vehicle_ID', 'Frame_ID', 'Global_Time', 'session']].to_dict('records'),
        'row_at_boundary': frame.iloc[max(0, int(switch_rows[0]) - 1):
                                      int(switch_rows[0]) + 2][['Vehicle_ID', 'Frame_ID', 'Global_Time',
                                                                'session']].to_dict('records')
        if len(switch_rows) else [],
    }, frame


def overlap_section(frame: pd.DataFrame) -> tuple:
    t0 = frame.groupby('session').Global_Time.min()
    t1 = frame.groupby('session').Global_Time.max()
    lo = int(max(t0.iloc[0], t0.iloc[1]))
    hi = int(min(t1.iloc[0], t1.iloc[1]))
    block = frame[(frame.Global_Time >= lo) & (frame.Global_Time <= hi)]
    per_session = {}
    for s in (0, 1):
        sub = block[block.session == s]
        counts = sub.groupby('Global_Time').size()
        per_session[str(s)] = {
            'rows': int(len(sub)), 'unique_vehicle_id': int(sub.Vehicle_ID.nunique()),
            'frame_id_min': int(sub.Frame_ID.min()), 'frame_id_max': int(sub.Frame_ID.max()),
            'rows_per_frame_mean': float(counts.mean()),
            'rows_per_frame_first_12_frames': counts.head(12).tolist(),
            'rows_per_frame_last_12_frames': counts.tail(12).tolist(),
        }
    return {'interval_ms': [lo, hi], 'duration_s': (hi - lo) / 1000.0,
            'session_time_ranges_ms': {str(s): [int(t0.loc[s]), int(t1.loc[s])] for s in (0, 1)},
            'frame_index_relation': 'session1_frame = session0_frame - 10200 at equal Global_Time',
            'per_session_in_overlap': per_session}, block


def session_cohorts(frame: pd.DataFrame) -> dict:
    out = {}
    for s in (0, 1):
        block = frame[frame.session == s]
        births = block.groupby('Vehicle_ID').Global_Time.min()
        deaths = block.groupby('Vehicle_ID').Global_Time.max()
        t0 = int(block.Global_Time.min())
        edges = np.arange(t0, int(block.Global_Time.max()) + 60001, 60000)
        birth_hist, _ = np.histogram(births.to_numpy(), bins=edges)
        death_hist, _ = np.histogram(deaths.to_numpy(), bins=edges)
        counts = block.groupby('Global_Time').size()
        last_birth = int(births.max())
        out[str(s)] = {
            'tracks': int(len(births)),
            'births_per_minute': birth_hist.tolist(),
            'deaths_per_minute': death_hist.tolist(),
            'rows_per_frame_first_12': counts.head(12).tolist(),
            'rows_per_frame_last_12': counts.tail(12).tolist(),
            'last_birth_ms': last_birth,
            'last_birth_offset_from_session_start_s': (last_birth - t0) / 1000.0,
            'first_death_ms': int(deaths.min()),
            'last_death_ms': int(deaths.max()),
            'birth_spearman_vs_vehicle_id': float(pd.Series(births.index.to_numpy(dtype=float))
                                                  .corr(pd.Series(births.to_numpy(dtype=float)),
                                                        method='spearman')),
        }
    # births of session 0 relative to the session-1 start
    s1_start = int(frame[frame.session == 1].Global_Time.min())
    s0_births = frame[frame.session == 0].groupby('Vehicle_ID').Global_Time.min()
    out['session0_births_at_or_after_session1_start'] = int((s0_births >= s1_start).sum())
    out['session0_last_births'] = [int(x) for x in np.sort(s0_births.to_numpy())[-8:]]
    s1_births = frame[frame.session == 1].groupby('Vehicle_ID').Global_Time.min()
    out['session1_first_births'] = [int(x) for x in np.sort(s1_births.to_numpy())[:8]]
    return out


def frame_index(frame: pd.DataFrame) -> dict:
    index = {}
    for s in (0, 1):
        block = frame[frame.session == s]
        for t, g in block.groupby('Global_Time'):
            index[(int(s), int(t))] = (
                g.Vehicle_ID.to_numpy(), g.Global_X.to_numpy() * FEET_TO_M,
                g.Global_Y.to_numpy() * FEET_TO_M, g.Local_X.to_numpy(),
                g.Local_Y.to_numpy(), g.v_Vel.to_numpy(), g.Lane_ID.to_numpy(),
                g.Direction.to_numpy())
    return index


def nearest_match(index: dict, session_a: int, times, session_b: int, shift_ms: int,
                  radius_m: float) -> tuple:
    hits = 0
    total = 0
    pairs = {}
    for t in times:
        a = index.get((session_a, int(t)))
        b = index.get((session_b, int(t) - shift_ms))
        if a is None or b is None:
            continue
        total += len(a[0])
        dx = a[1][:, None] - b[1][None, :]
        dy = a[2][:, None] - b[2][None, :]
        distance = np.sqrt(dx * dx + dy * dy)
        nearest = distance.argmin(1)
        dmin = distance[np.arange(len(a[0])), nearest]
        hits += int((dmin <= radius_m).sum())
        for i in np.nonzero(dmin <= radius_m)[0]:
            key = (int(a[0][i]), int(b[0][nearest[i]]))
            entry = pairs.setdefault(key, [0, []])
            entry[0] += 1
            entry[1].append(float(dmin[i]))
    return total, hits, pairs


def content_matching(frame: pd.DataFrame, index: dict, overlap: dict) -> dict:
    lo, hi = overlap['interval_ms']
    s1_times = np.sort(frame[frame.session == 1].Global_Time.unique())
    overlap_times = s1_times[(s1_times >= lo) & (s1_times <= hi)]
    placedo_times = s1_times
    true_counts = {}
    true_pairs = {}
    for radius in RADII_M:
        total, hits, pairs = nearest_match(index, 1, overlap_times, 0, 0, radius)
        true_counts[radius] = {'points': total, 'hits': hits, 'fraction': hits / max(total, 1)}
        if radius == 2.0:
            true_pairs = pairs
    placebo_total, placebo_hits, _ = nearest_match(index, 1, overlap_times, 0, 10000, 2.0)
    # track-level matched pairs
    matched_pairs = []
    for (v1, v0), (n_hits, errors) in sorted(true_pairs.items(), key=lambda kv: -kv[1][0]):
        if n_hits < 5:
            continue
        matched_pairs.append({'session1_vehicle_id': v1, 'session0_vehicle_id': v0,
                              'matched_frames': n_hits,
                              'median_error_m': float(np.median(errors)),
                              'p90_error_m': float(np.percentile(errors, 90))})
    # identity metrics for the strongest pairs
    top_pairs = []
    for entry in matched_pairs[:8]:
        v1 = entry['session1_vehicle_id']
        v0 = entry['session0_vehicle_id']
        a = frame[(frame.session == 1) & (frame.Vehicle_ID == v1)].set_index('Global_Time')
        b = frame[(frame.session == 0) & (frame.Vehicle_ID == v0)].set_index('Global_Time')
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a2, b2 = a.loc[common], b.loc[common]
        detail = dict(entry)
        detail.update({
            'common_frames': int(len(common)),
            'session1_frames': [int(a.Frame_ID.min()), int(a.Frame_ID.max())],
            'session0_frames': [int(b.Frame_ID.min()), int(b.Frame_ID.max())],
            'v_vel_correlation': float(np.corrcoef(a2.v_Vel, b2.v_Vel)[0, 1]),
            'same_lane_fraction': float((a2.Lane_ID.to_numpy() == b2.Lane_ID.to_numpy()).mean()),
            'same_direction_fraction': float((a2.Direction.to_numpy() == b2.Direction.to_numpy()).mean()),
            'local_dy_ft_mean_std': [float((a2.Local_Y - b2.Local_Y).mean()),
                                     float((a2.Local_Y - b2.Local_Y).std())],
            'global_dxy_m_mean_std': [float(np.hypot(a2.Global_X - b2.Global_X,
                                                     a2.Global_Y - b2.Global_Y).mean() * FEET_TO_M),
                                      float(np.hypot(a2.Global_X - b2.Global_X,
                                                     a2.Global_Y - b2.Global_Y).std() * FEET_TO_M)],
        })
        top_pairs.append(detail)
    # fixed-shift search at sampled S1 times
    sample_times = s1_times[::20]
    shift_results = {}
    for shift_s in SHIFTS_S:
        total, hits2, _ = nearest_match(index, 1, sample_times, 0, int(shift_s * 1000), 2.0)
        shift_results[str(shift_s)] = {'sampled_points': total, 'hits_2m': hits2,
                                       'fraction': hits2 / max(total, 1)}
    # scene distribution comparison (per-frame sorted Global_Y quantiles)
    def quantiles(g):
        return np.quantile(g[2], [.1, .25, .5, .75, .9])

    def scene_distance(lag_ms: int) -> list:
        distances = []
        for t in range(lo, hi + 1, 2000):
            a = index.get((1, int(t)))
            b = index.get((0, int(t) + lag_ms))
            if a is None or b is None or len(a[0]) < 3 or len(b[0]) < 3:
                continue
            distances.append(np.abs(quantiles(a) - quantiles(b)))
        if not distances:
            return []
        arr = np.asarray(distances)
        return {'mean_percentile_gap_m': [float(x) for x in arr.mean(0)],
                'overall_mean_m': float(arr.mean()), 'frames': int(len(arr))}
    scene = {'true_same_time': scene_distance(0)}
    for lag in (-60000, -30000, 30000, 60000):
        scene['lag_%+ds' % (lag // 1000)] = scene_distance(lag)
    return {
        'same_time_match_by_radius': {str(r): v for r, v in true_counts.items()},
        'same_time_placebo_plus10s_2m': {'points': placebo_total, 'hits': placebo_hits,
                                         'fraction': placebo_hits / max(placebo_total, 1)},
        'matched_track_pairs_min5_frames': matched_pairs[:20],
        'matched_track_pairs_total': len(matched_pairs),
        'top_pair_identity': top_pairs,
        'fixed_shift_search_2m': shift_results,
        'scene_distribution_gap': scene,
        'overlap_times_compared': int(len(overlap_times)),
    }


def id_namespace(frame: pd.DataFrame) -> dict:
    ids0 = set(frame[frame.session == 0].Vehicle_ID.unique())
    ids1 = set(frame[frame.session == 1].Vehicle_ID.unique())
    both = sorted(ids0 & ids1)
    rows = []
    for v in both:
        g0 = frame[(frame.session == 0) & (frame.Vehicle_ID == v)]
        g1 = frame[(frame.session == 1) & (frame.Vehicle_ID == v)]
        end_time = int(g0.Global_Time.max())
        start_time = int(g1.Global_Time.min())
        end_x, end_y = g0.Global_X.iloc[-1] * FEET_TO_M, g0.Global_Y.iloc[-1] * FEET_TO_M
        start_x, start_y = g1.Global_X.iloc[0] * FEET_TO_M, g1.Global_Y.iloc[0] * FEET_TO_M
        gap_s = (start_time - end_time) / 1000.0
        jump_m = float(math.hypot(start_x - end_x, start_y - end_y))
        implied = jump_m / gap_s if gap_s > 0 else None
        rows.append({'vehicle_id': int(v), 'gap_s': gap_s, 'jump_m': jump_m,
                     'implied_speed_mps': implied, 'session0_rows': int(len(g0)),
                     'session1_rows': int(len(g1))})
    jumps = np.asarray([r['jump_m'] for r in rows])
    gaps = np.asarray([r['gap_s'] for r in rows])
    examples = sorted(rows, key=lambda r: r['jump_m'])
    return {
        'session0_unique_ids': len(ids0), 'session1_unique_ids': len(ids1),
        'ids_in_both_sessions': len(both),
        'ids_only_session0': len(ids0 - ids1), 'ids_only_session1': len(ids1 - ids0),
        'session0_vehicle_id_is_global_key': len(ids0 & ids1) == 0,
        'cross_session_gap_s': stats(gaps),
        'cross_session_end_start_jump_m': stats(jumps),
        'gap_s_le_0': int((gaps <= 0).sum()),
        'jump_m_gt_100': int((jumps > 100).sum()),
        'jump_m_le_10': int((jumps <= 10).sum()),
        'examples_smallest_jump': examples[:8],
        'examples_largest_jump': examples[-8:],
    }


def ordering_anomalies(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(['session', 'Vehicle_ID'], sort=False)
    transitions = []
    negative_rows = []
    for (s, v), g in grouped:
        t = g.Global_Time.to_numpy()
        if len(t) < 2:
            continue
        d = np.diff(t)
        transitions.append(d)
        for index in np.nonzero(d < 0)[0]:
            negative_rows.append({'session': int(s), 'vehicle_id': int(v),
                                  'row': int(g.index[index + 1]),
                                  'prev_time_ms': int(t[index]), 'time_ms': int(t[index + 1]),
                                  'next_time_ms': int(t[index + 2]) if index + 2 < len(t) else None,
                                  'next_delta_ms': int(t[index + 2] - t[index + 1]) if index + 2 < len(t) else None})
    transitions = np.concatenate(transitions)
    frame_key = frame.duplicated(['session', 'Vehicle_ID', 'Frame_ID'], keep=False)
    frame_time_key = frame.duplicated(['session', 'Vehicle_ID', 'Global_Time'], keep=False)
    # same gaps after sorting each track by Global_Time: separates file-order artifacts from real dropouts
    sorted_gaps = []
    for (_s, _v), g in frame.groupby(['session', 'Vehicle_ID'], sort=False):
        t = np.sort(g.Global_Time.to_numpy())
        if len(t) > 1:
            sorted_gaps.append(np.diff(t))
    sorted_gaps = np.concatenate(sorted_gaps)
    return {
        'non100_gaps_after_time_sort': int((sorted_gaps != 100).sum()),
        'non100_after_time_sort_200ms': int((sorted_gaps == 200).sum()),
        'non100_after_time_sort_100_to_1000ms': int(((sorted_gaps > 100) & (sorted_gaps < 1000)).sum()),
        'non100_after_time_sort_ge_1000ms': int((sorted_gaps >= 1000).sum()),
        'file_order_transitions_total': int(len(transitions)),
        'dt_eq_100ms': int((transitions == 100).sum()),
        'dt_eq_200ms': int((transitions == 200).sum()),
        'dt_100_to_1000ms': int(((transitions > 100) & (transitions < 1000)).sum()),
        'dt_ge_1000ms': int((transitions >= 1000).sum()),
        'dt_lt_0_negative': int((transitions < 0).sum()),
        'dt_min_ms': float(transitions.min()), 'dt_max_ms': float(transitions.max()),
        'duplicate_session_vehicle_frame_rows': int(frame_key.sum()),
        'duplicate_session_vehicle_time_rows': int(frame_time_key.sum()),
        'negative_step_examples': negative_rows[:10],
        'negative_steps_total': len(negative_rows),
        'negative_steps_with_sequence_resume': int(sum(1 for r in negative_rows
                                                       if r['next_delta_ms'] == 100)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit NGSIM Lankershim source sessions (read-only)')
    parser.add_argument('--csv', default='/home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv')
    parser.add_argument('--output', default='reports/data_cleaning/source_session_stats.json')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    frame, offsets = load(csv_path)
    log('loaded', len(frame), 'rows')

    report = {
        'audit': 'source_session_structure',
        'mode': 'AUDIT ONLY (no data modified)',
        'csv': str(csv_path),
        'rows': int(len(frame)),
        'columns_used': list(frame.columns),
        'session_summary': session_summary(frame, offsets),
    }
    layout, frame = file_layout(frame)
    report['file_layout'] = layout
    log('layout done')

    overlap, overlap_block = overlap_section(frame)
    report['overlap'] = overlap
    report['session_cohort_structure'] = session_cohorts(frame)
    log('overlap/cohorts done')

    index = frame_index(frame)
    report['content_matching'] = content_matching(frame, index, overlap)
    log('content matching done')

    report['vehicle_id_namespace'] = id_namespace(frame)
    report['ordering_anomalies'] = ordering_anomalies(frame)
    log('id/ordering done')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf-8')
    log('wrote', output)
    print(json.dumps({'output': str(output), 'rows': int(len(frame)),
                      'sessions': list(report['session_summary']['sessions'].keys()),
                      'overlap_s': report['overlap']['duration_s'],
                      'matched_pairs_min5': len(report['content_matching']['matched_track_pairs_min5_frames']),
                      'ids_in_both_sessions': report['vehicle_id_namespace']['ids_in_both_sessions'],
                      'negative_steps': report['ordering_anomalies']['negative_steps_total']},
                     default=float))


if __name__ == '__main__':
    START = time.time()
    main()