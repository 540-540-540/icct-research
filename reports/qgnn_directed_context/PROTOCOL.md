# 可学习有向上下文残差 QGNN 协议

2026-09-14。当前阶段只实施候选 A；不实现候选 B，不组合 A+B，不改动原 QGNN、强 GNN、公共时间模型、数据或历史结果。

## 唯一结构变化

原量子关系输入保留七维物理边特征和五维对称节点统计。每个图层新增两个零初始化矩阵：

\[
U_r,U_s\in\mathbb R^{5\times128},\qquad
c^+_{ij}=c_{ij}+\mathbb 1_{i\ne j}(U_ru_i+U_su_j).
\]

物理层和上下文层仍分别使用原 `context_gain=0.35/1.0`，之后的角度编码、六比特三层 PQC、十二维 Z/ZZ 读出、score、gate、经典 value、45 m 邻域、mask 和时间预测器保持不变。每层新增 1,280 个经典参数，两层共 2,560 个；自环不变。

该结构为接收者和发送者加入可学习角色条件，不宣称等同于强 GNN 的双线性 Q/K 比较。

## D0 与 S0

- D0 仅读取 `train`：统计未来有效长度、训练/评估分母差异；在真实轨迹损失上记录原 Q 与零初始化候选的输出、共享梯度、新参数梯度、量子梯度和门控质量。D0 不选择模型指标，不读取 V_confirm/test。
- S0 CPU 检查要求共享权重、前向、输入梯度和共享参数梯度一致；新增参数全零且真实可达；训练态 dropout/RNG、mask、空帧、置换、padding、自环和有向性通过。
- S0 GPU smoke 使用 17 个 train 与 17 个 V_select 起点，验证 16+1 尾批、首次验证前中断恢复、两次更新、四档验证、有限数值、最佳权重恢复和完成后零额外更新。工程检查不作为 ADE/FDE 证据。

## A1 开发实验

| 项目 | 固定规则 |
|---|---|
| 比较 | `original` 与 `directed_context` 都从头训练，使用配对 seed2026 |
| 数据 | 5,549 个 train；400 个 V_select；训练四档 SNR 平衡轮换，验证四档全覆盖 |
| 预算 | 最多 150 epoch；patience=20；min_delta=0 |
| 批量 | batch/micro-batch/validation micro-batch 均为 16，保留尾批 |
| 优化器 | 图 lr=3e-4；LoRA=3e-5；其他时间模块=1e-4；量子参数 wd=0，普通矩阵 wd=0.01 |
| 选模 | 最小 `J=ADE+0.5*FDE` 的同一检查点，分别报告 ADE、FDE |
| 开发门槛 | 候选 ADE、FDE 都优于原 Q 才进入多种子确认；一好一坏记为混合结果 |
| 数据边界 | 不读取 V_confirm/test；不训练 QGAT；不宣称量子优势 |

正式运行前必须通过当前源码绑定的 CPU 检查、D0 和完整 GPU smoke。正式 A1 由用户亲自启动。

服务器：`/home/dell/YrM/ICCT`；环境：`/home/dell/YrM/envs/ICCT`。

```text
/home/dell/YrM/envs/ICCT/bin/python -m experiments.qgnn_directed_context.check_graph --output reports/qgnn_directed_context/graph_checks.json
/home/dell/YrM/envs/ICCT/bin/python -m experiments.qgnn_directed_context.check_training
/home/dell/YrM/envs/ICCT/bin/python -m experiments.qgnn_directed_context.diagnose
/home/dell/YrM/envs/ICCT/bin/python -m experiments.qgnn_directed_context.run --smoke
/home/dell/YrM/envs/ICCT/bin/python -m experiments.qgnn_directed_context.run --smoke --resume
```

以上检查完成后，由用户在本机 PowerShell 亲自执行：

```powershell
ssh -t YrM_TwYhB "cd /home/dell/YrM/ICCT && /home/dell/YrM/envs/ICCT/bin/python -u -m experiments.qgnn_directed_context.run --run"
```

当前终端会实时显示带有 `[original]` 和 `[directed_context]` 标签的逐轮训练进度，同时保留两臂独立日志。若进程被中断，只能由用户显式将末尾改为 `--run --resume` 恢复；不得再次使用无 `--resume` 的命令覆盖已有正式运行。
