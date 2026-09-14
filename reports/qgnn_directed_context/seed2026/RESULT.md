# 可学习有向上下文残差 QGNN 配对结果

seed2026；仅 V_select；上限 150 轮，patience=20。按最低 J 的同一检查点分别报告 ADE、FDE。

| 模型 | 实际轮数 | 最佳轮次 | ADE | FDE | 早停 |
|---|---:|---:|---:|---:|---|
| original | 95 | 75 | 0.542767 | 1.095422 | True |
| directed_context | 74 | 54 | 0.542487 | 1.096786 | True |

上下文版相对原版：ADE 改善 0.052%，FDE 退步 0.125%；未通过 ADE、FDE 同时改善的开发门槛。

这是 seed2026、V_select 单种子开发结果，不进入多种子确认。强 GNN 仅保留历史参照；本轮未训练 QGAT，未读取 V_confirm 或 test，也不构成量子优势证据。
