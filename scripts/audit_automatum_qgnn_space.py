"""Automatum QGNN task-space audit using validation-only strong-local and pairwise controls."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'reports/q0/g1r_0db_local_seed2026/val_diagnostics.json'
PAIR=ROOT/'reports/q0/g1r_0db_pairwise_seed2026/val_diagnostics.json'
OUT_JSON=ROOT/'reports/q0/automatum_qgnn_task_space_audit_0db.json'
OUT_MD=ROOT/'reports/q0/AUTOMATUM_QGNN_TASK_SPACE_AUDIT_0DB.md'

local=json.loads(LOCAL.read_text())['rows']
pair=json.loads(PAIR.read_text())['rows']
assert [r['dataset_index'] for r in local]==[r['dataset_index'] for r in pair]

strata={
 'N>=4': lambda r:r['vehicle_count']>=4,
 'N>=5': lambda r:r['vehicle_count']>=5,
 'N>=6': lambda r:r['vehicle_count']>=6,
 'close20>=2': lambda r:r['close_pairs_20m']>=2,
 'close20>=3': lambda r:r['close_pairs_20m']>=3,
 'close30>=3': lambda r:r['close_pairs_30m']>=3,
 'closing30>=1': lambda r:r['closing_pairs_30m_gt_0p5']>=1,
 'closing30>=2': lambda r:r['closing_pairs_30m_gt_0p5']>=2,
 'CPA<5m': lambda r:r['min_cpa_distance_m']<5,
 'CPA<10m': lambda r:r['min_cpa_distance_m']<10,
 'complex_combo': lambda r:(r['vehicle_count']>=4 and r['close_pairs_20m']>=2),
 'closing_combo': lambda r:(r['vehicle_count']>=4 and r['closing_pairs_30m_gt_0p5']>=2),
}

tot_ade=sum(r['ADE'] for r in local); tot_fde=sum(r['FDE'] for r in local)
rows=[]
for name,pred in strata.items():
    ids=[i for i,r in enumerate(local) if pred(r)]
    if not ids: continue
    la=sum(local[i]['ADE'] for i in ids)/len(ids)
    lf=sum(local[i]['FDE'] for i in ids)/len(ids)
    pa=sum(pair[i]['ADE'] for i in ids)/len(ids)
    pf=sum(pair[i]['FDE'] for i in ids)/len(ids)
    share_a=sum(local[i]['ADE'] for i in ids)/tot_ade
    share_f=sum(local[i]['FDE'] for i in ids)/tot_fde
    rows.append({
      'stratum':name,'scenes':len(ids),'scene_share':len(ids)/len(local),
      'local_ADE':la,'pair_ADE':pa,'pair_gain_ADE':1-pa/la,
      'local_FDE':lf,'pair_FDE':pf,'pair_gain_FDE':1-pf/lf,
      'local_ADE_error_share':share_a,'local_FDE_error_share':share_f,
      'required_internal_ADE_gain_for_global_5pct':0.05/share_a,
      'required_internal_FDE_gain_for_global_5pct':0.05/share_f,
    })

# Per-scene oracle between local and pairwise.
oracle_ade=sum(min(a['ADE'],b['ADE']) for a,b in zip(local,pair))/len(local)
oracle_fde=sum(min(a['FDE'],b['FDE']) for a,b in zip(local,pair))/len(local)
base_ade=sum(r['ADE'] for r in local)/len(local)
base_fde=sum(r['FDE'] for r in local)/len(local)
beat_ade=[i for i,(a,b) in enumerate(zip(local,pair)) if b['ADE']<a['ADE']]
beat_fde=[i for i,(a,b) in enumerate(zip(local,pair)) if b['FDE']<a['FDE']]
beat_both=[i for i,(a,b) in enumerate(zip(local,pair)) if b['ADE']<a['ADE'] and b['FDE']<a['FDE']]

summary={
 'reference':'strong_local_residual',
 'comparison':'pairwise_residual',
 'snr_db':0,
 'scenes':len(local),
 'local_ADE':base_ade,'local_FDE':base_fde,
 'pair_ADE':sum(r['ADE'] for r in pair)/len(pair),
 'pair_FDE':sum(r['FDE'] for r in pair)/len(pair),
 'pair_global_ADE_gain':1-(sum(r['ADE'] for r in pair)/len(pair))/base_ade,
 'pair_global_FDE_gain':1-(sum(r['FDE'] for r in pair)/len(pair))/base_fde,
 'pair_beats_local_scene_fraction_ADE':len(beat_ade)/len(local),
 'pair_beats_local_scene_fraction_FDE':len(beat_fde)/len(local),
 'pair_beats_local_scene_fraction_both':len(beat_both)/len(local),
 'pair_better_both_local_ADE_error_share':sum(local[i]['ADE'] for i in beat_both)/tot_ade,
 'pair_better_both_local_FDE_error_share':sum(local[i]['FDE'] for i in beat_both)/tot_fde,
 'local_pair_oracle_ADE':oracle_ade,
 'local_pair_oracle_FDE':oracle_fde,
 'oracle_ADE_gain_vs_local':1-oracle_ade/base_ade,
 'oracle_FDE_gain_vs_local':1-oracle_fde/base_fde,
 'test_set_used':False,
}
payload={'summary':summary,'strata':rows}
OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')

md=[
'# Automatum QGNN Task-Space Audit — 0 dB validation pilot','',
'Reference: frozen strong Local residual; comparator: frozen-base Pairwise residual. Validation only; no test use.','',
'## Global','',
f"- Strong Local: ADE {base_ade:.6f}, FDE {base_fde:.6f}",
f"- Pairwise: ADE {summary['pair_ADE']:.6f}, FDE {summary['pair_FDE']:.6f}",
f"- Pairwise gain vs Local: ADE {100*summary['pair_global_ADE_gain']:+.2f}%, FDE {100*summary['pair_global_FDE_gain']:+.2f}%",
f"- Pairwise beats Local on both metrics in {100*summary['pair_beats_local_scene_fraction_both']:.1f}% of validation scenes.",
f"- Per-scene Local/Pairwise oracle gain: ADE {100*summary['oracle_ADE_gain_vs_local']:.2f}%, FDE {100*summary['oracle_FDE_gain_vs_local']:.2f}%.",'',
'## Predefined interaction strata','',
'| Stratum | Scenes | Local error share ADE/FDE | Pairwise gain ADE/FDE | Required internal gain for 5% global ADE/FDE |',
'|---|---:|---:|---:|---:|'
]
for r in rows:
 md.append(f"| {r['stratum']} | {r['scenes']} | {100*r['local_ADE_error_share']:.1f}% / {100*r['local_FDE_error_share']:.1f}% | {100*r['pair_gain_ADE']:+.1f}% / {100*r['pair_gain_FDE']:+.1f}% | {100*r['required_internal_ADE_gain_for_global_5pct']:.1f}% / {100*r['required_internal_FDE_gain_for_global_5pct']:.1f}% |")
md += ['','## Interpretation','',
'- A stratum with error share alpha can mathematically support a 5% global improvement only if its internal error can be reduced by at least 0.05/alpha, assuming all other scenes are unchanged.',
'- The current Pairwise residual does **not** establish marginal interaction value over the Strong Local control; it is worse globally and in most predefined high-interaction strata.',
'- However, high-interaction strata carry a large fraction of Strong-Local error, so Automatum still has mathematical room for an interaction-specialized mechanism if a better interaction model can exploit it.',
'- The Local/Pairwise per-scene oracle quantifies heterogeneity: some scenes do benefit from Pairwise, but current gating/interaction learning does not identify and exploit them reliably.','']
OUT_MD.write_text('\n'.join(md))
print(json.dumps(payload,indent=2))

