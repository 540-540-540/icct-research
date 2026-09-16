"""Render completed remote results; no training and no test data."""
import json, hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;D=B/'development'
done=json.loads((D/'completed.json').read_text());s=done['summaries']
a=json.loads((D/'association_downstream.json').read_text())
lines=['# 新版核心消息 QGNN＋LLM：单种子实验结果','','状态：已完成；训练种子2026。所有数据为开发证据，未读取原测试数组。','',
'## 完整991场景开发集，四档SNR宏平均','','| 模型 | 图ADE | 图FDE | 含LLM ADE | 含LLM FDE | 图/LLM所选轮次 |','|---|---:|---:|---:|---:|---|']
for arm,row in s.items():
    g=row['graph']['aggregate'];l=row['llm']['aggregate']
    lines.append(f"| {arm} | {g['ade_m']:.6f} | {g['fde_m']:.6f} | {l['ade_m']:.6f} | {l['fde_m']:.6f} | {row['graph_selected_epoch']}/{row['llm_selected_epoch']} |")
lines += ['','## 量子组相对对照的误差降低百分比','','正数表示量子组更好。图阶段和含LLM阶段分开比较；不与历史其他划分结果直接比绝对误差。','']
reductions={}
for phase in ('graph','llm'):
    q=s['quantum'][phase]['aggregate']
    for ref in ('plain','classical'):
        r=s[ref][phase]['aggregate'];v={k:100*(r[k]-q[k])/r[k] for k in q};reductions[phase+'_'+ref]=v
        lines.append(f"- {phase} 相对 {ref}：ADE {v['ade_m']:.4f}%，FDE {v['fde_m']:.4f}%。")
lines += ['','## 64场景未知身份经典关联＋图＋LLM，四档SNR宏平均','','| 模型 | ADE | FDE |','|---|---:|---:|']
assoc={}
for arm,row in a['results'].items():
    v={k:float(np.mean([x[k] for x in row.values()])) for k in ('ade_m','fde_m')};assoc[arm]=v
    lines.append(f"| {arm} | {v['ade_m']:.6f} | {v['fde_m']:.6f} |")
lines += ['','## 训练曲线与耗时','','| 模型 | 阶段 | 首轮训练损失 | 末轮训练损失 | 所选轮次 | 累计训练及逐轮评价秒数 | 零梯度参数张量数 |','|---|---|---|---:|---:|---:|']
timing={}
for arm in s:
    timing[arm]={}
    for phase in ('graph','llm'):
        z=json.loads((D/(arm+'_'+phase+'_audit.json')).read_text()); rec=z['records'][1:]
        seconds=sum(x['seconds'] for x in rec);timing[arm][phase]=seconds
        lines.append(f"| {arm} | {phase} | {rec[0]['train_loss']:.6f} | {rec[-1]['train_loss']:.6f} | {z['selected_epoch']} | {seconds:.1f} | {len(z['never_received_gradient'])} |")
lines += ['','## 限制','','- 原图从头训练12轮，含LLM联合训练6轮，不保证充分收敛；所选轮次和曲线应共同判断。',
'- 已知身份含噪历史开发集与未知身份关联历史链路结果分别列出，不能互相替代。',
'- 同接口经典核心84参数、量子54参数，其余公共结构相同；不宣称精确参数或运行时间匹配。',
'- 全部991场景中包含256个选模场景，且此前属于开发范围；不主张独立测试或统计显著。',
'- 原经典关联模型复用且可能见过开发场景；三组输入相同，仍只作开发完整链路检查。',
'- 固定成员集合，带噪真实首帧初始化；解析态矢量仿真，不支持量子加速结论。',
'- 前一次40步残差适配器诊断和本次核心消息替换是不同实验，不混用结果。']
(B/'实验结果与分析.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(D/'comparison_summary.json').write_text(json.dumps(dict(reductions_percent=reductions,association_macro=assoc,timing_seconds=timing),indent=2),encoding='utf8')
print(json.dumps(dict(reductions_percent=reductions,association_macro=assoc,timing_seconds=timing),indent=2))
