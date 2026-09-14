"""Render the completed development comparison without selecting a winner."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/qgat_factorial'
s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
assert s['status'] == 'complete' and s['same_final_epoch'] == 8
names = {'A': '5 qubits, all', 'B': '7 qubits, all', 'C': '5 qubits, 45 m', 'D': '7 qubits, 45 m'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for c in 'ABCD':
    h = s['jobs'][c]['training']['history']
    for ax, key, title in zip(axes, ['J','ADE','FDE'], ['Validation J (lower is better)','Validation ADE (m)','Validation FDE (m)']):
        ax.plot([r['epoch'] for r in h], [r[key] for r in h],
                label=f'{c}: {names[c]}', color='#0F4D92' if c in 'AC' else '#B64342',
                linestyle='-' if c in 'AB' else '--', marker='o' if c in 'AB' else 's', ms=4, lw=1.8)
        ax.set(xlabel='Epoch',title=title,xticks=range(1,9))
        ax.grid(axis='y',alpha=.18)
axes[0].legend(fontsize=9)
fig.suptitle('QGAT development comparison | seed 2026 | 512 train / 128 validation origins',fontsize=12)
for ext in ('png','pdf'):
    fig.savefig(OUT / f'learning_curves.{ext}',dpi=300)
plt.close(fig)

lines = ['# QGAT 四组开发对照结果', '',
         '四组均完成固定 8 轮训练。每组 512 个训练起点、128 个 V_select 起点，种子 2026；验证覆盖四档 SNR。',
         '5 比特 = 3 索引 + 1 注意力 + 1 值；7 比特 = 3 索引 + 2 注意力 + 2 值。', '',
         '## 同为第 8 轮的结果', '', '| 组别 | 结构与邻居 | ADE (m) | FDE (m) | J |', '|---|---|---:|---:|---:|']
cn = {'A':'5 比特，全部有效车辆','B':'7 比特，全部有效车辆','C':'5 比特，45 米','D':'7 比特，45 米'}
for c in 'ABCD':
    r = s['final_epoch_metrics'][c]
    lines.append(f"| {c} | {cn[c]} | {r['ADE']:.6f} | {r['FDE']:.6f} | {r['J']:.6f} |")
lines += ['', 'J = ADE + 0.5 × FDE，均为越低越好。', '', '## 各组最佳验证轮次', '',
          '| 组别 | 最佳轮次 | ADE (m) | FDE (m) | J |', '|---|---:|---:|---:|---:|']
for c in 'ABCD':
    r = s['best_epoch_metrics'][c]
    e = s['jobs'][c]['training']['best_metrics']['epoch']
    lines.append(f"| {c} | {e} | {r['ADE']:.6f} | {r['FDE']:.6f} | {r['J']:.6f} |")
lines += ['', '## 第 8 轮因素差值', '', '负数表示前者误差更低。', '',
          '| 比较 | ADE 差 | FDE 差 | J 差 |', '|---|---:|---:|---:|']
labels = {'B_minus_A':'B−A：全邻居下扩展表示','D_minus_C':'D−C：45 米下扩展表示',
          'C_minus_A':'C−A：5 比特下限制邻居','D_minus_B':'D−B：7 比特下限制邻居',
          'interaction_D_minus_B_minus_C_plus_A':'D−B−C+A：交互项'}
for k, r in s['final_epoch_contrasts']['macro'].items():
    lines.append(f"| {labels[k]} | {r['ADE']:+.6f} | {r['FDE']:+.6f} | {r['J']:+.6f} |")
lines += ['', '## 最后一轮趋势与检查', '', '| 组别 | 第 8 轮 J − 第 7 轮 J | 最后一轮是否最佳 | 量子角更新数 |', '|---|---:|---|---:|']
for c in 'ABCD':
    t = s['training_trends'][c]
    lines.append(f"| {c} | {t['last_J_change']:+.6f} | {'是' if t['final_epoch_is_best'] else '否'} | {s['jobs'][c]['theta_best_changed_elements']} |")
lines += ['', '四组时序模块初始参数完全一致，A/C 与 B/D 图初始参数分别一致；评估起点和有效目标分母一致。',
          '所有组数值与梯度门禁通过。完整线路等价性、置换、掩码、边界、Pauli 读出及真实四维输入 Jacobian 秩检查见 graph_checks.json。',
          '未使用 V_confirm 或测试集，未修改既有 GNN 与 A07 结果。', '', '## 解释边界', '',
          '本轮为单种子、固定短程训练的开发比较，不据此宣称统计显著或正式性能优势。',
          '两比特扩展同时改变编码、量子参数和读出投影容量，因此差异不能单独归因于比特数。',
          '不按速度选择或淘汰模型；最佳轮次与同轮次结果分别报告。是否充分收敛仍未建立。',
          '', '![训练曲线](learning_curves.png)', '', '完整逐 SNR、逐场景结果及执行记录：summary.json。']
(OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote RESULTS.md and learning_curves.png/pdf')
