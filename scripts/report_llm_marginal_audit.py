"""Create the validation-only LLM marginal-effect diagnostic report."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(path): return json.loads((ROOT/path).read_text())['best_validation']
def summary(path): return json.loads((ROOT/path).read_text())

gpt_n=summary(Path('reports/q0/g1_0db_nograph_seed2026/summary.json'))
gpt_m=summary(Path('reports/q0/g1_0db_mpnn_seed2026/summary.json'))
s20_n=summary(Path('reports/q0/llm_factorial_0db_nograph_simple_seed2026/summary.json'))
s20_m=summary(Path('reports/q0/llm_factorial_0db_mpnn_simple_seed2026/summary.json'))
s40_n=summary(Path('reports/q0/llm_factorial_0db_nograph_simple40_seed2026/summary.json'))
s40_m=summary(Path('reports/q0/llm_factorial_0db_mpnn_simple40_seed2026/summary.json'))

def gain(base,graph,metric):
    return 1-graph['best_validation'][metric]/base['best_validation'][metric]
def rel(a,b,metric):
    return 1-b['best_validation'][metric]/a['best_validation'][metric]

equal20={
 'simple':{
   'nograph':s20_n['best_validation'],'mpnn':s20_m['best_validation'],
   'graph_gain_ADE':gain(s20_n,s20_m,'ADE'),'graph_gain_FDE':gain(s20_n,s20_m,'FDE'),
 },
 'gpt2':{
   'nograph':gpt_n['best_validation'],'mpnn':gpt_m['best_validation'],
   'graph_gain_ADE':gain(gpt_n,gpt_m,'ADE'),'graph_gain_FDE':gain(gpt_n,gpt_m,'FDE'),
 },
}
equal20['difference_in_graph_gain_percentage_points']={
 'ADE':100*(equal20['gpt2']['graph_gain_ADE']-equal20['simple']['graph_gain_ADE']),
 'FDE':100*(equal20['gpt2']['graph_gain_FDE']-equal20['simple']['graph_gain_FDE']),
}
equal20['simple_vs_gpt2_absolute']={
 'NoGraph_ADE_gain':rel(gpt_n,s20_n,'ADE'),
 'NoGraph_FDE_gain':rel(gpt_n,s20_n,'FDE'),
 'MPNN_ADE_gain':rel(gpt_m,s20_m,'ADE'),
 'MPNN_FDE_gain':rel(gpt_m,s20_m,'FDE'),
}
payload={
 'snr_db':0,'seed':2026,'split':'val','test_set_used':False,
 'equal_20_epoch_factorial':equal20,
 'simple_40_epoch_extension':{
   'nograph':s40_n['best_validation'],'nograph_best_epoch':s40_n['best_epoch'],
   'mpnn':s40_m['best_validation'],'mpnn_best_epoch':s40_m['best_epoch'],
   'graph_gain_ADE':gain(s40_n,s40_m,'ADE'),'graph_gain_FDE':gain(s40_n,s40_m,'FDE'),
 },
 'parameter_counts':{
   'simple_nograph':s20_n['parameters'],'simple_mpnn':s20_m['parameters'],
   'gpt2_nograph':gpt_n['parameters'],'gpt2_mpnn':gpt_m['parameters'],
 },
 'interpretation':[
   'The 20-epoch matched-exposure pilot does not support the hypothesis that GPT-2 is so strong that it compresses Graph marginal value.',
   'Graph is strongly harmful under the Simple-GRU decoder but slightly helpful for ADE and neutral/slightly harmful for FDE under GPT-2.',
   'Absolute Simple-GRU validation errors are substantially lower than GPT-2+LoRA at the same 20-epoch exposure and improve further by 40 epochs.',
   'Therefore the current GPT-2 module is not demonstrated to be an accuracy-improving component on Automatum under this harness; optimization/interface effects remain a plausible bottleneck.',
   'These are single-seed, 0 dB validation diagnostics, not final model comparisons.'
 ]
}
(ROOT/'reports/q0/llm_marginal_effect_audit_0db.json').write_text(json.dumps(payload,indent=2)+'\n')
md=[
'# LLM Marginal-Effect Audit — Automatum 0 dB validation','',
'Single-seed diagnostic, validation only; test remains closed.','',
'## Matched 20-epoch exposure','',
'| Downstream | NoGraph ADE/FDE | MPNN ADE/FDE | Graph gain ADE/FDE |',
'|---|---:|---:|---:|',
f"| Simple-GRU | {s20_n['best_validation']['ADE']:.6f} / {s20_n['best_validation']['FDE']:.6f} | {s20_m['best_validation']['ADE']:.6f} / {s20_m['best_validation']['FDE']:.6f} | {100*equal20['simple']['graph_gain_ADE']:+.2f}% / {100*equal20['simple']['graph_gain_FDE']:+.2f}% |",
f"| GPT-2+LoRA | {gpt_n['best_validation']['ADE']:.6f} / {gpt_n['best_validation']['FDE']:.6f} | {gpt_m['best_validation']['ADE']:.6f} / {gpt_m['best_validation']['FDE']:.6f} | {100*equal20['gpt2']['graph_gain_ADE']:+.2f}% / {100*equal20['gpt2']['graph_gain_FDE']:+.2f}% |",
'',
f"- GPT-2 changes Graph marginal gain by {equal20['difference_in_graph_gain_percentage_points']['ADE']:+.2f} percentage points in ADE and {equal20['difference_in_graph_gain_percentage_points']['FDE']:+.2f} points in FDE relative to Simple-GRU.",
f"- Simple-GRU vs GPT-2 absolute gain at 20 epochs: NoGraph ADE {100*equal20['simple_vs_gpt2_absolute']['NoGraph_ADE_gain']:+.2f}%, FDE {100*equal20['simple_vs_gpt2_absolute']['NoGraph_FDE_gain']:+.2f}%; MPNN ADE {100*equal20['simple_vs_gpt2_absolute']['MPNN_ADE_gain']:+.2f}%, FDE {100*equal20['simple_vs_gpt2_absolute']['MPNN_FDE_gain']:+.2f}%.",
'',
'## Simple-GRU 40-epoch convergence extension','',
f"- NoGraph best: ADE {s40_n['best_validation']['ADE']:.6f}, FDE {s40_n['best_validation']['FDE']:.6f}, epoch {s40_n['best_epoch']}.",
f"- MPNN best: ADE {s40_m['best_validation']['ADE']:.6f}, FDE {s40_m['best_validation']['FDE']:.6f}, epoch {s40_m['best_epoch']}.",
f"- Graph gain remains negative: ADE {100*gain(s40_n,s40_m,'ADE'):+.2f}%, FDE {100*gain(s40_n,s40_m,'FDE'):+.2f}%.",
'',
'## Interpretation','',
'- The current evidence does **not** support “GPT-2 is already so strong that it creates diminishing returns for Graph.”',
'- In the matched 20-epoch diagnostic, Graph is much less harmful under GPT-2 than under Simple-GRU; GPT-2 therefore does not appear to be suppressing Graph marginal utility.',
'- More importantly, the Simple-GRU downstream is substantially better in absolute ADE/FDE than the current GPT-2+LoRA pipeline. The LLM module itself is therefore not yet demonstrated to provide an accuracy benefit on Automatum.',
'- This points to a possible LLM/interface/optimization bottleneck rather than a saturation effect. The project can still retain LLM as a core module, but its role and training recipe need to be separated from claims about Graph/QGNN gains.',
'- All conclusions are 0 dB, one seed, validation-only diagnostics.',''
]
(ROOT/'reports/q0/LLM_MARGINAL_EFFECT_AUDIT_0DB.md').write_text('\n'.join(md))
print(json.dumps(payload,indent=2))

