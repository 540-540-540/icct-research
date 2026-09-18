# 行驶方向关系编码与强 GNN 对照结果

seed2026；仅 V_select；上限 150 轮，patience=20。按最低 J 的同一检查点分别报告 ADE、FDE。

| 模型 | 实际轮数 | 最佳轮次 | ADE | FDE | 早停 |
|---|---:|---:|---:|---:|---|
| motion_frame | 73 | 53 | 0.551578 | 1.114769 | True |
| gnn | 102 | 82 | 0.550223 | 1.118283 | True |
| original_reused | 95 | 75 | 0.542767 | 1.095422 | True |

motion_vs_fresh_gnn：ADE 改善 -0.246%，FDE 改善 0.314%（负值表示退步）。
motion_vs_original：ADE 改善 -1.623%，FDE 改善 -1.766%（负值表示退步）。
original_vs_fresh_gnn：ADE 改善 1.355%，FDE 改善 2.044%（负值表示退步）。

这是单种子开发结果。若达到 150 轮上限，须另行检查曲线判断训练是否充分；不能仅凭进程完成宣称收敛。
原 QGNN 复用已完成且来源、数据、训练规则完全匹配的 150/patience20 结果；强 GNN 本轮从头训练。
此强 GNN 对照用于预测性能比较，不单独隔离量子计算必要性。未训练 QGAT，未读取 V_confirm 或 test。
