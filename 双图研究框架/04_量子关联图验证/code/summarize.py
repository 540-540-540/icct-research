import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]
d=json.loads((BASE/'development/completed.json').read_text())
arms=('hungarian','original_gnn','plain','mlp','quantum')
summary={}
for arm in arms:
    values=[x[arm]['aggregate'] for x in d['results'].values()]
    summary[arm]=dict(mean_association_f1=float(np.mean([v['association_f1'] for v in values])),
       mean_position_rmse_m=float(np.mean([v['position_rmse_m'] for v in values])),
       wrong_associations=sum(v['wrong_associations'] for v in values),owner_transitions=sum(v['owner_transitions'] for v in values))
comparisons={}
for a in ('original_gnn','plain','mlp'):
    q=summary['quantum']; b=summary[a]
    comparisons[a]=dict(rmse_reduction_percent=100*(b['mean_position_rmse_m']-q['mean_position_rmse_m'])/b['mean_position_rmse_m'],
      f1_difference_percentage_points=100*(q['mean_association_f1']-b['mean_association_f1']),
      rmse_win_tie_loss=[sum(int(np.sign(v[a]['aggregate']['position_rmse_m']-v['quantum']['aggregate']['position_rmse_m']))==sign for v in d['results'].values()) for sign in (1,0,-1)])
payload=dict(summary=summary,comparisons=comparisons,training=d['training'])
(BASE/'development/aggregate_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
path=BASE/'development/结果与分析.md'; text=path.read_text(encoding='utf-8')
if '## 汇总结论' not in text:
    text+='\n## 汇总结论\n\n12 个条件的 F1/RMSE 为条件等权平均，计数为所有条件累计；同一批场景重复加噪，不是 288 个独立场景。\n\n'
    text+='| 方法 | 平均关联 F1 | 平均位置 RMSE (m) | 错误关联累计 |\n|---|---:|---:|---:|\n'
    for a,v in summary.items(): text+='| %s | %.6f | %.6f | %d |\n'%(a,v['mean_association_f1'],v['mean_position_rmse_m'],v['wrong_associations'])
    for a,v in comparisons.items(): text+='\n量子版相对 %s：位置 RMSE 降低 %.5f%%，关联 F1 差异 %.5f 个百分点；12 条件 RMSE 胜/平/负=%s。\n'%(a,v['rmse_reduction_percent'],v['f1_difference_percentage_points'],v['rmse_win_tie_loss'])
    text+='\n本轮量子电路已经通过数值和梯度检查，但工程可运行不等同于性能增益。由于指标接近饱和且样本及微调预算有限，本轮只评价该具体结构，不推断所有量子关联方法均无效。\n'
    path.write_text(text,encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))
