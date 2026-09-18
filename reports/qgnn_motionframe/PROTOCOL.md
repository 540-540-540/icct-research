# 行驶方向关系编码 QGNN 与从头训练的强 GNN

2026-09-14。用户在误差诊断与单点改动方案后要求“开始吧”。本轮完成案例诊断、实现、检查和正式启动；沿用最多 150 轮、patience=20 的已确认预算，并在正式更新确认后停止交互工作，训练由服务器后台继续。

## 本轮仅改变什么

`MotionFrameQGNNGraph` 保留原双层 QGNN 全部参数。每个接收车辆用当帧估计速度构成单位前向 f=(vx,vy)/||v||，右向 r=(fy,−fx)；量子分支前四个物理通道从 (dx,dy,dvx,dvy) 改为 (Δp·r,Δp·f,Δv·r,Δv·f)。旋转在原有 tanh 之前完成，原距离和速度归一化尺度不变。

速度低于 1 m/s 时回退到 f=(0,1)、r=(1,0)。此阈值预先固定，没有验证集调参；阈值处不光滑，不对速度方向作 detach。经典 `edge_value` 仍使用原始边特征；距离、接近速度、TTC、5 个上下文特征、45 m 加自环邻域、mask、6 比特×3 层电路、12 个 Z/ZZ 读出、消息门控和时间模型均保持。

新增参数 **0**。图参数 219,278，其中量子旋转参数 108；完整可训练参数 825,014。低速回退或 +Y 行驶时的前向与原图一致；这不代表完整模型具有旋转不变性。

假设依据、反例和限制见 [案例诊断报告](../qgnn_cases/REPORT.md)。该假设尚未经过本轮预测误差验证。

## 三个比较对象

| 对象 | 本轮执行 | 用途 |
|---|---|---|
| motion_frame | GPU0 从头训练 | 唯一新 QGNN 改动 |
| gnn | GPU1 从头训练 `prediction.classical.GNNGraph` | 补齐共同训练规则的强经典性能对照 |
| original_reused | 复用已完成原 QGNN 的 150/patience20 结果 | 隔离坐标编码变化，不重复相同训练 |

原版参照为 `results/qgnn_readout/seed2026/original`：实际 95 轮，第 75 轮最佳，ADE=0.5427669769、FDE=1.0954215576。复用前逐项校验其完整源码记录、资产修改记录、依赖版本、数据标识、起点范围、seed、初始化偏移、学习率、批量、验证批量、训练上限、早停规则和选模规则。来源变化即拒绝复用。

新 GNN 与历史强 GNN 使用同一图类、初始化种子和公共时间模型；图参数 319,756、完整可训练参数 925,492。历史 GNN 的 40+60 训练结果仍保留，但本轮最终比较使用新的全程一致训练结果。强 GNN 用于性能比较，不能单独证明量子计算必要性。

## 共同训练规则

| 项目 | 规则 |
|---|---|
| seed / 初始化 | 2026；图 seed+200000，时间模型 seed+100000；复用冻结 GPT-2 基座，不载入任何历史训练权重初始化新组 |
| 数据 | 全部 5,549 个 train 起点；全部 400 个 V_select 起点 |
| SNR | 训练按既有平衡轮换；每轮验证 5/10/15/20 dB 全部四档 |
| 批量 | batch=16、micro_batch=16、validation_micro_batch=16；保留尾批 |
| 预算 | 最多 150 轮；J 连续 20 轮未严格改善即早停；min_delta=0 |
| 选模 | 最低 V_select J=ADE+0.5×FDE；同一检查点分别报告 ADE、FDE |
| AdamW | 图 lr=0.0003，LoRA lr=0.00003，其余可训练时间模块 lr=0.0001；普通矩阵 wd=0.01，标量/向量 wd=0；量子旋转参数 wd=0 |
| 数值与资源 | 两张 GPU 各一个独立工作进程；FP32 模型，QGNN 复数双精度状态；关闭 TF32 |
| 保存 / 恢复 | 每次更新保存 latest，改善时保存 best；显式恢复、严格合同校验；拒绝无断点时静默从头开始 |
| 范围 | 不读取 V_confirm/test，不新增 QGAT 训练 |

## 必要验证与结果位置

CPU 检查覆盖图结构、前向与梯度、物理编码、原始经典边路径、全部权重与 RNG 配对、GNN 原类复现、优化器覆盖及分组。GPU 检查使用 17 个 train 和 17 个 V_select 起点：16+1 尾批、首次验证前中断恢复、两次更新、四档验证、有限数值、最佳权重恢复、完成后恢复 0 次额外更新。检查产物只作工程证据。

服务器：`/home/dell/YrM/ICCT`；环境：`/home/dell/YrM/envs/ICCT`。

- 配置：`configs/qgnn_motionframe.json`；入口：`experiments/qgnn_motionframe/run.py`。
- 检查：`reports/qgnn_motionframe/{graph_checks,training_checks,smoke_first_pass,smoke_checks}.json`。
- 正式检查点和逐轮指标：`results/qgnn_motionframe/seed2026/{motion_frame,gnn}/`。
- 状态和日志：`reports/qgnn_motionframe/seed2026/coordinator.json`、`jobs/`、`logs/`。
- 完成后自动生成：`reports/qgnn_motionframe/seed2026/summary.json` 和 `RESULT.md`。
- 启动记录：`reports/qgnn_motionframe/LAUNCH.json`。

最终检查三者最佳检查点的起点、SNR、ADE/FDE 分母一致，并分别报告方向编码相对原版、方向编码相对新强 GNN、原版相对新强 GNN 的变化。本轮为单种子开发证据；达到训练上限不自动等于收敛，也不据此宣称稳定量子优势。
