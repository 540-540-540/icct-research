# 候选 A 当前状态与后续讨论交接

日期：2026-09-14。状态：正式配对开发实验已完成；候选 A 未通过预设开发门槛，尚未授权任何后续实验。

## 本轮做了什么

在师兄双层 QGNN 的两层关系编码中，各增加零初始化的接收者/发送者上下文矩阵，仅作用于非自环边，共新增 2,560 个经典参数。六比特三层 PQC、12 维 Z/ZZ 读出、score、gate、经典 value、45 m 邻域、公共 GPT-2/LoRA 时间预测器和数据规则均保持不变。

原始 QGNN 与 `directed_context` 使用相同 seed2026、5,549 个 train 起点、400 个 V_select 起点、四档 SNR、batch/micro-batch 16，从头并行训练；最多 150 轮，`patience=20`，按最低 `J=ADE+0.5*FDE` 选择一个检查点并分别报告 ADE、FDE。

## 正式结果

| 模型 | 实际轮数 | 最佳轮次 | ADE | FDE | J |
|---|---:|---:|---:|---:|---:|
| 原始 QGNN | 95 | 75 | 0.542767 | 1.095422 | 1.090478 |
| 有向上下文版 | 74 | 54 | 0.542487 | 1.096786 | 1.090880 |

候选相对原版：ADE 改善 0.000280（0.052%），FDE 退步 0.001364（0.125%），J 增加 0.000402。结果为一好一坏，`development_gate_passed=false`，按预先规则不进入多种子确认。总运行约 1 小时 41 分钟；两臂均正常早停。

## 已验证边界

- 零初始化时完整模型输出、损失和共享梯度与原版一致；四个新增矩阵的 2,560 个元素在真实 train 损失下均获得有限非零梯度。
- CPU 结构/优化器检查、D0 和双 GPU smoke/恢复检查均通过；正式运行共同分母一致。
- 本轮只访问 train 与 V_select；未访问 V_confirm/test，未训练 QGAT，不形成稳定性、统计显著性或量子优势结论。
- `experiments/qgnn_hzy/` 是 PPT 电路展示材料，不属于本候选，也未纳入本次提交。

## 文件与运行资产

- 结构与入口：`experiments/qgnn_directed_context/graph.py`、`run.py`
- 冻结配置：`configs/qgnn_directed_context.json`
- 协议与检查：本目录 `PROTOCOL.md`、`graph_checks.json`、`training_checks.json`、`d0.json`、`smoke_checks.json`
- 正式结果：`seed2026/RESULT.md`、`seed2026/summary.json`、`seed2026/coordinator.json`
- 大检查点继续保存在服务器 `/home/dell/YrM/ICCT/results/qgnn_directed_context/seed2026/`，不进入 Git。

## 下一步仅供讨论

需要先决定：候选 A 就此归档，还是先按场景/SNR/轨迹长度分析 ADE 微增而 FDE 微降的来源；以及是否按原审计方案独立评估候选 B。以上均未授权实施、训练或访问确认集。
