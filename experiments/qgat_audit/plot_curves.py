"""Render original formal training curves without smoothing or rescaling metrics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'reports/qgat_formal/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':1.2,'legend.frameon':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey='row')
rows=[]
for col,seed in enumerate([2023,2024,2025]):
    for name,color in [('qgat','#0F4D92'),('gnn','#B64342')]:
        h=r['tasks'][f'{name}-{seed}']['training']['history']
        x=[v['epoch'] for v in h]
        axes[0,col].plot(x,[v['loss'] for v in h],color=color,lw=2,label=name.upper())
        axes[1,col].plot(x,[v['J'] for v in h],color=color,lw=2,label=name.upper())
        best=min(h,key=lambda v:v['J'])
        axes[1,col].scatter(best['epoch'],best['J'],s=45,color=color,edgecolor='white',zorder=3)
        tail=h[-6:]
        rows.append(dict(job=f'{name}-{seed}',epochs=len(h),best_epoch=best['epoch'],
          tail_epochs=[v['epoch'] for v in tail],tail_validation_J=[v['J'] for v in tail],
          tail_train_loss=[v['loss'] for v in tail],
          tail_J_relative_change_pct=100*(tail[-1]['J']/tail[0]['J']-1),
          validation_plateau_triggered=r['tasks'][f'{name}-{seed}']['training']['stopped_early']))
    axes[0,col].set_title(f'Seed {seed}')
    axes[1,col].set_xlabel('Epoch')
    for ax in axes[:,col]:
        ax.set_xticks([1,5,10,15,20]);ax.grid(axis='y',alpha=.18,lw=.6);ax.set_xlim(.5,20.5)
axes[0,0].set_ylabel('Training loss')
axes[1,0].set_ylabel('V_select J (lower is better)')
axes[0,2].legend(loc='upper right')
fig.suptitle('Formal run: training continues, validation behavior differs',fontsize=15,y=1.01)
fig.text(.5,-.015,'All four SNRs; original epoch values. Dots mark selected checkpoints. No test data.',ha='center',fontsize=10)
fig.tight_layout(pad=1.5)
out=ROOT/'reports/qgat_audit'
for suffix in ('png','pdf','svg'):fig.savefig(out/f'learning_curves.{suffix}',dpi=300,bbox_inches='tight',facecolor='white')
(out/'convergence.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
