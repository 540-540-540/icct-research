"""CPU-only, outcome-blind V_select strata for original QGNN versus historical GNN."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/qgnn_cases'
SNRS = (5, 10, 15, 20)
PATHS = {
    'metadata': 'data/f01d/metadata/V_select.json',
    'inputs': 'data/f01d/inputs/V_select_snr_20.npz',
    'labels': 'data/f01d/labels/V_select.npz',
    'Q': 'results/qgnn_readout/seed2026/original/validation_epoch_75.json',
    'G': 'results/qgat_extend100/seed2026/gnn/validation_epoch_97.json',
}
DEFINITIONS = {
    'active_tracks_last': 'Number of existing tracks in last history frame; not a count of all road vehicles.',
    'mean_neighbor_degree_last': 'Mean other-track degree at last frame, Euclidean radius <=45m, excluding self edges.',
    'nearest_pair_distance_m': 'Minimum last-frame distance among existing pairs, including pairs beyond graph radius.',
    'max_closing_speed_mps': 'Maximum max(0,-dot(delta_position,delta_velocity)/distance) among pairs within 45m; 0 when none.',
    'mean_speed_last_mps': 'Mean norm of estimated last-frame velocity among existing tracks.',
    'history_abs_acceleration_mean_mps2': 'Per-track change of mean speed between first and last five history frames divided by mean-time difference; mean absolute change, >=3 existing frames in each end.',
    'history_max_deceleration_mps2': 'Largest max(0,-acceleration) from the same first/last-five-frame calculation.',
    'history_heading_change_mean_deg': 'Mean absolute angle between mean first/last-five-frame velocities, each with >=3 existing frames and speed >=1m/s; heading proxy, not a lane-change label.',
    'not_detected_given_exists_fraction': '20dB history fraction existing & not detected divided by existing; absence of variation cannot diagnose missed detection.',
    'unavailable_slot_fraction': 'History fraction of the eight allocated slots without an existing track; not a missed-detection rate.',
    'posthoc_future_cv_ADE_m': 'POSTHOC LABEL ONLY: constant-velocity extrapolation from last observed state, scored with official eligible/label masks; departure includes maneuver and observation-estimation errors, not a pure interaction label.',
}


def read_json(name):
    return json.loads((ROOT / PATHS[name]).read_text())


def make_features():
    metadata = read_json('metadata')
    assert metadata['split'] == 'V_select'
    samples = metadata['samples']
    with np.load(ROOT / PATHS['inputs'], allow_pickle=False) as z:
        assert set(z.files) == {'state_hat', 'track_exists', 'detected', 'timestamp'}
        state, exists, detected, times = [z[k].copy() for k in ('state_hat', 'track_exists', 'detected', 'timestamp')]
    with np.load(ROOT / PATHS['labels'], allow_pickle=False) as z:
        assert set(z.files) == {'future_position', 'label_valid'}
        future, valid = z['future_position'].copy(), z['label_valid'].copy()
    n = len(samples)
    assert n == 400 and state.shape == (n, 20, 8, 4) and future.shape == (n, 20, 8, 2)
    assert exists.dtype == detected.dtype == valid.dtype == bool
    assert exists.shape == detected.shape == valid.shape == (n, 20, 8)
    assert np.isfinite(state).all() and np.isfinite(future[valid]).all()
    assert not np.any(detected & ~exists) and np.allclose(np.diff(times), .1, atol=1e-6, rtol=0)
    assert [s['sample_index'] for s in samples] == list(range(n))
    eligible = exists[:, -1] & (exists.sum(1) >= 3)
    supervision = valid & eligible[:, None]
    ade_targets = supervision.any(1).sum(1)
    fde_targets = supervision[:, -1].sum(1)
    rows = []
    for i, meta in enumerate(samples):
        x, mask = state[i].astype(float), exists[i]
        live = np.flatnonzero(mask[-1])
        assert len(live) > 0
        pos, vel = x[-1, live, :2], x[-1, live, 2:]
        dp, dv = pos[None] - pos[:, None], vel[None] - vel[:, None]
        distance = np.linalg.norm(dp, axis=-1)
        pairs = np.triu(np.ones(distance.shape, bool), 1)
        edges = pairs & (distance <= 45.)
        radial = (dp * dv).sum(-1) / np.maximum(distance, 1e-12)
        accelerations, turns = [], []
        for k in range(8):
            first, last = np.flatnonzero(mask[:5, k]), np.flatnonzero(mask[-5:, k]) + 15
            if len(first) < 3 or len(last) < 3:
                continue
            va, vb = x[first, k, 2:], x[last, k, 2:]
            span = times[i, last].mean() - times[i, first].mean()
            accelerations.append((np.linalg.norm(vb, axis=-1).mean() - np.linalg.norm(va, axis=-1).mean()) / span)
            a, b = va.mean(0), vb.mean(0)
            if min(np.linalg.norm(a), np.linalg.norm(b)) >= 1.:
                turns.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))))
        cv = x[-1, None, :, :2] + np.arange(1, 21)[:, None, None] * .1 * x[-1, None, :, 2:]
        scored = supervision[i]
        counts = scored.sum(0)
        err = np.linalg.norm(np.where(scored[..., None], cv - future[i], 0), axis=-1)
        cv_ade = np.mean(err.sum(0)[counts > 0] / counts[counts > 0])
        row = {k: meta[k] for k in ('sample_index', 'episode_index', 'episode_id', 'origin_ms', 'source_block', 'source_keys', 'source_groups', 'track_keys')}
        row['origin'] = i
        row.update(active_tracks_last=len(live), mean_neighbor_degree_last=2.*int(edges.sum())/len(live),
                   nearest_pair_distance_m=float(distance[pairs].min()) if pairs.any() else None,
                   max_closing_speed_mps=float(np.maximum(-radial[edges], 0).max()) if edges.any() else 0.,
                   mean_speed_last_mps=float(np.linalg.norm(vel, axis=-1).mean()),
                   history_abs_acceleration_mean_mps2=float(np.mean(np.abs(accelerations))) if accelerations else None,
                   history_max_deceleration_mps2=float(max(0., -min(accelerations))) if accelerations else None,
                   history_heading_change_mean_deg=float(np.mean(turns)) if turns else None,
                   history_acceleration_tracks=len(accelerations), history_heading_tracks=len(turns),
                   not_detected_given_exists_fraction=float((mask & ~detected[i]).sum()/mask.sum()),
                   unavailable_slot_fraction=float((~mask).mean()), posthoc_future_cv_ADE_m=float(cv_ade),
                   ade_targets=int(ade_targets[i]), fde_targets=int(fde_targets[i]))
        rows.append(row)
    return rows


def assign_groups(rows):
    episodes = sorted(set(r['episode_id'] for r in rows))
    sources = {e: {g for r in rows if r['episode_id'] == e for g in r['source_groups']} for e in episodes}
    unseen = set(episodes)
    components = []
    while unseen:
        group = {min(unseen)}
        while True:
            vehicles = set().union(*(sources[e] for e in group))
            joined = {e for e in unseen if vehicles & sources[e]}
            if joined <= group:
                break
            group |= joined
        unseen -= group
        components.append(sorted(group))
    for i, episodes in enumerate(components):
        for row in rows:
            if row['episode_id'] in episodes:
                row['source_component'] = i
    return components


def define_bins(rows):
    # Fixed semantic thresholds or input-only tertiles, all computed before model outcomes are loaded.
    fixed = {'active_tracks_last': [5.5, 7.5], 'nearest_pair_distance_m': [5., 15.],
             'max_closing_speed_mps': [.5, 2.], 'mean_speed_last_mps': [1., 5.],
             'history_max_deceleration_mps2': [.5, 1.5], 'history_heading_change_mean_deg': [5., 15.],
             'not_detected_given_exists_fraction': [1e-9, .1], 'unavailable_slot_fraction': [1e-9, .25]}
    bins = {}
    for feature in DEFINITIONS:
        values = [r[feature] for r in rows if r[feature] is not None]
        if len(set(values)) < 2:
            bins[feature] = {'skipped': 'No variation in available values.', 'value': values[0] if values else None}
            continue
        edges = fixed.get(feature, np.unique(np.quantile(values, [1/3, 2/3])).tolist())
        source = 'fixed_semantic' if feature in fixed else ('future_label_tertiles_posthoc' if feature.startswith('posthoc_') else 'input_tertiles')
        groups = {f'bin_{i}': [] for i in range(len(edges)+1)}
        groups['unavailable'] = []
        for row in rows:
            value = row[feature]
            label = 'unavailable' if value is None else f'bin_{int(np.searchsorted(edges, value, side="right"))}'
            row['bin_' + feature] = label
            groups[label].append(row['origin'])
        bins[feature] = {'threshold_source': source, 'edges': edges, 'interval_rule': '[-inf,e0), [e0,e1), [e1,+inf); missing separately', 'members': groups}
    return bins


def add_errors(rows):
    metrics = {}
    for arm in ('Q', 'G'):
        pack = read_json(arm)
        assert pack['split'] == 'V_select' and len(pack['per_scene']) == 1600
        lookup = {(p['origin'], p['snr_db']): p for p in pack['per_scene']}
        assert set(lookup) == {(i, snr) for i in range(400) for snr in SNRS}
        for row in rows:
            records = [lookup[(row['origin'], snr)] for snr in SNRS]
            assert all(p['ade_targets'] == row['ade_targets'] and p['fde_targets'] == row['fde_targets'] for p in records)
            assert all(np.isfinite(p[m]) for p in records for m in ('ADE', 'FDE'))
            for m in ('ADE', 'FDE'):
                row[arm + '_' + m] = float(np.mean([p[m] for p in records]))
                row[arm + '_' + m + '_snr_range'] = float(np.ptp([p[m] for p in records]))
        for m in ('ADE', 'FDE'):
            assert np.isclose(np.mean([r[arm + '_' + m] for r in rows]), pack[m], atol=1e-9, rtol=0)
        metrics[arm] = {k: pack[k] for k in ('ADE', 'FDE', 'J', 'by_snr')}
    for row in rows:
        row['delta_ADE'] = row['Q_ADE'] - row['G_ADE']
        row['delta_FDE'] = row['Q_FDE'] - row['G_FDE']
        row['case_rank_score'] = row['delta_ADE'] + .5*row['delta_FDE']
    return metrics


def summarize(rows):
    if not rows:
        return {'origins': 0}
    episodes = sorted(set(r['episode_id'] for r in rows))
    result = {'origins': len(rows), 'episodes': len(episodes), 'source_blocks': len(set(r['source_block'] for r in rows)),
              'source_components': len(set(r['source_component'] for r in rows)),
              'Q_better_both_origins': sum(r['delta_ADE'] < 0 and r['delta_FDE'] < 0 for r in rows),
              'Q_worse_both_origins': sum(r['delta_ADE'] > 0 and r['delta_FDE'] > 0 for r in rows)}
    for field in ('source_block', 'source_component'):
        result['by_' + field] = {}
        for group in sorted(set(r[field] for r in rows)):
            part = [r for r in rows if r[field] == group]
            result['by_' + field][str(group)] = {'origins': len(part), 'episodes': len(set(r['episode_id'] for r in part)),
                'delta_ADE': float(np.mean([r['delta_ADE'] for r in part])),
                'delta_FDE': float(np.mean([r['delta_FDE'] for r in part]))}
    for m in ('ADE', 'FDE'):
        for arm in ('Q', 'G'):
            result[arm + '_' + m] = float(np.mean([r[arm + '_' + m] for r in rows]))
        values = np.array([r['delta_' + m] for r in rows])
        by_ep = [np.array([r['delta_' + m] for r in rows if r['episode_id'] == e]) for e in episodes]
        loo = [(values.sum()-v.sum())/(len(values)-len(v)) for v in by_ep if len(v)<len(values)]
        result['delta_' + m] = {'mean': float(values.mean()), 'median': float(np.median(values)),
            'Q_better_origins': int((values < 0).sum()), 'equal_episode_mean': float(np.mean([v.mean() for v in by_ep])),
            'Q_better_episodes': int(sum(v.mean()<0 for v in by_ep)), 'Q_worse_episodes': int(sum(v.mean()>0 for v in by_ep)),
            'leave_one_episode_out_min': float(min(loo)) if loo else None, 'leave_one_episode_out_max': float(max(loo)) if loo else None,
            'single_episode_removal_can_flip_sign': bool(any(v*values.mean()<0 for v in loo)),
            'largest_absolute_episode_contribution_fraction': float(max(abs(v.sum()) for v in by_ep)/max(sum(abs(v.sum()) for v in by_ep),1e-30))}
    return result


def candidates(rows, worst):
    seen, selected = set(), []
    for row in sorted(rows, key=lambda r: r['case_rank_score'], reverse=worst):
        if row['episode_id'] in seen or (row['case_rank_score']>0) != worst:
            continue
        seen.add(row['episode_id'])
        selected.append(row)
        if len(selected) == 6:
            break
    return selected


def main():
    rows = make_features()
    components = assign_groups(rows)
    bins = define_bins(rows)
    metrics = add_errors(rows)
    strata = {}
    for feature, spec in bins.items():
        strata[feature] = {k: v for k, v in spec.items() if k != 'members'}
        if 'members' in spec:
            strata[feature]['bins'] = {label: summarize([rows[i] for i in ids]) for label, ids in spec['members'].items()}
    episodes = []
    for episode in sorted(set(r['episode_id'] for r in rows)):
        part = [r for r in rows if r['episode_id'] == episode]
        episodes.append({'episode_id': episode, 'episode_index': part[0]['episode_index'], 'source_component': part[0]['source_component'],
                         'source_block': part[0]['source_block'], 'origin_ids': [r['origin'] for r in part],
                         'source_groups': sorted({g for r in part for g in r['source_groups']}), 'metrics': summarize(part),
                         'feature_means': {f: float(np.mean([r[f] for r in part if r[f] is not None])) if any(r[f] is not None for r in part) else None for f in DEFINITIONS}})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'sources': PATHS,
        'scope': 'CPU-only descriptive V_select diagnosis. No model execution, checkpoint loading, training, GPU, V_confirm, or test access.',
        'comparison': 'Original QGNN best-J epoch75 versus historical GNN best-J epoch97. GNN first40 epochs had different physical batching; not a fully paired superiority test.',
        'units': 'ADE/FDE and distances in meters; positive delta=QGNN worse. Four SNR rows averaged within each origin before any grouping.',
        'identity_validation': {'origin_snr_pairs': 1600, 'unique_origins': 400, 'exact_origin_snr_keys': True,
                                'Q_G_denominators_match': True, 'denominators_recomputed_from_eligibility_and_labels': True,
                                'aggregate_reproduction_atol': 1e-9},
        'dependence': '25 episodes contain16 adjacent/overlapping origins each; shared source_groups link episodes into conservative connected components. Blocks also correlate. No independence or significance claim.',
        'feature_scope': 'Input features use only SNR20 observed history; posthoc_future_cv_ADE_m additionally uses V_select future labels, never model inputs.',
        'omissions': ['No lane or interaction ground-truth labels are available; no invented lane-change/occlusion categories.',
                      'Heading differences require moving endpoint mean velocities >=1m/s; unavailable cases retained separately.',
                      'Estimated history acceleration/heading are observation proxies, not exact physical ground truth.',
                      'SNR20 has no missed detections among existing tracks; unavailable slots are a different phenomenon.'],
        'feature_definitions': DEFINITIONS, 'overall': summarize(rows), 'original_metrics': metrics, 'stratification': strata,
        'episodes': episodes,
        'source_components': [{'component': i, 'episode_ids': e, 'metrics': summarize([r for r in rows if r['source_component']==i])} for i,e in enumerate(components)],
        'source_blocks': {str(b): summarize([r for r in rows if r['source_block']==b]) for b in sorted(set(r['source_block'] for r in rows))},
        'candidate_rule': 'Rank individual origins by delta_ADE+0.5delta_FDE for inspection only; keep at most one per episode. Different episodes can share source_component and are not independent cases.',
        'worst_Q_candidates': candidates(rows, True), 'best_Q_candidates': candidates(rows, False)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'stratification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (OUT/'scene_features.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in rows)
    assert len(list(csv.DictReader((OUT/'scene_features.csv').open()))) == 400
    print(json.dumps({'overall': report['overall'], 'episode_count': len(episodes), 'component_count': len(components),
                      'worst_origins': [r['origin'] for r in report['worst_Q_candidates']],
                      'best_origins': [r['origin'] for r in report['best_Q_candidates']]}))


if __name__ == '__main__':
    main()
