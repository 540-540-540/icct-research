# 师兄双层 QGNN：当前接口下的 100 epoch 对照

**当前状态（2026-09-14）：用户已将本协议对应的师兄双层QGNN确定为主候选，QGAT暂停训练投入。本轮100 epoch已完成，结果见`comparison.json`；主候选身份不代表稳定优势已确认。本次不启动新训练。**

协议制定日期：2026-09-13。当时主方案为关系旋转QGAT（relation_D），本轮以师兄QGNN图模块作为迁移对照，由用户手动启动，GNN未重新训练。下文保留该次复跑的实际条件与历史准备记录。

## 本轮移植范围

直接复用 `code/02_pennylane_qgnn_core/physics_aligned_dual_model.py` 的两层 `PhysicsAlignedMessageLayer`，分别为 physical/contextual。保留每条边的 12 维结构化关系、6 比特、3 层 RY/RZ 重上传、RZ–RY–RZ 可训练旋转、环形 CNOT、6 个 Z 与 6 个 ZZ 读出，以及原注意力、消息门控、经典 value 和残差更新。两层各 54 个量子旋转参数，合计 108 个；不代表整个模型只有 108 个参数。

输入采用当前 GNN 的逐帧六维接口：标准化位置速度四维加 detected/track_exists，经相同规格的 MLP 映射为 128 维。邻接沿用当前 45m 邻域与有效 self 边，SI 物理边特征不变。每帧两层 QGNN 输出每车 128 维特征，接当前公共 GPT-2/LoRA、数值 token 适配器、预测头和估计起点恒速项。

本轮是师兄图模块在当前系统下的迁移对照，不是复现师兄原先的 GRU、未来词分支及三阶段信任融合整套模型。归档源文件不修改。

## 固定训练设置

| 项目 | 设置 |
|---|---|
| 训练 / 验证 | 当前 f01d 全部 train 5549 / V_select 400 origins |
| 时域 / 车辆数 | 历史 20 帧，未来 20 帧，最多 8 辆车 |
| 种子 | 2026；图 seed 202026，公共时间模块 seed 102026 |
| 训练 | 从头联合训练 100 epoch，patience 101，不加载历史已训练参数 |
| batch | 全局 16，双进程 DDP；两卡各 8，尾批两卡 7+6；每卡验证 batch 16 |
| 样本曝光 | 每轮每 origin 一档 SNR，5/10/15/20 dB 平衡轮换；每档 25 次 |
| 优化器 | 当前 AdamW；图 3e-4，LoRA 3e-5，其他可训练时间模块参数 1e-4 |
| 权重衰减 | 普通矩阵 .01，一维参数 0，量子旋转参数单独 0 |
| 精度 | 经典 float32、量子解析模拟 complex128；AMP/TF32/compile 关闭 |
| 梯度 / 断点 | clip norm 1；每 update 保存可恢复断点；预期共 34700 updates |
| 选模 | 每轮全部 V_select 四 SNR 宏平均；J = ADE + 0.5 FDE 最低 |
| 禁止访问 | V_confirm、test；训练入口只接受 train/V_select |

量子核在同设备显式准备零态，以适配当前 PennyLane GPU 初始态行为；其数值输出和梯度应通过原电路等价检查。完整 batch 经真实输入前向/反向检查，不改变全局 batch。准备检查不执行 optimizer.step。

训练使用两个独立进程驱动 GPU0/GPU1，合并梯度更新同一份模型，不是训练两个种子。尾批 13 按实际样本数加权，避免 7+6 被错误地等权处理。验证也分配到两卡，各计算 200 个 origin 的四档 SNR，再按原指标的误差总和及有效场景数汇总。检查点保存两个进程的完整随机状态，恢复时继续相同训练轨迹。

## 已完成准备检查（无参数更新）

服务器检查通过，`optimizer_steps_executed=0`，参数逐值未改变，未访问 V_confirm/test，未创建正式训练断点。包括原电路输出/梯度等价、空帧与掩码、置换、公共时间模块初始化、全部参数学习率/衰减、双卡梯度、并行验证、随机状态恢复及零更新断点保存/读取。

- 全局 16 的双卡/单卡梯度最大绝对差：9.54e-7。
- 尾批 13 的梯度最大绝对差：7.15e-7。
- 两进程训练模式随机状态恢复后，输出逐值一致。
- GPU0/GPU1 检查峰值已分配显存约 2.38 / 2.77 GB。

相同工作量、预热后的计时（本次服务器测量）：

| 项目 | 单卡 | 双卡 | 解释 |
|---|---:|---:|---|
| 全局 batch16 训练前向+反向 | 0.1290秒 | 0.1269秒 | 约1.7%提速，训练收益很小 |
| 64个验证origin × 四档SNR | 1.0708秒 | 0.5341秒 | 约2.00倍，收益主要来自验证并行 |

这是初始化模型的运行时间检查；前反向计时不包括数据装载、优化器更新或断点写入，不能据此声称整个100轮流程有两倍加速。实际总时长待用户启动后观察。

## 复用 GNN 结果

服务器：`results/qgat_extend100/seed2026/gnn/`。启动前检查完成状态、100 轮历史、34700 更新、相同数据/标签/公共代码及 GPT-2 来源。与 baseline 源协议比对使用既有实验溯源校验。

| 取点 | GNN epoch | ADE | FDE | J |
|---|---:|---:|---:|---:|
| 最佳 J | 97 | 0.551612872696 | 1.121638645004 | 1.112432195197 |
| 共同第 100 轮 | 100 | 0.562582582091 | 1.157240945892 | 1.141203055037 |

完成后自动生成 `comparison.json`，并列两种口径，核对逐场景 origin/SNR/有效目标数覆盖。正百分比表示 QGNN 误差更低。不拼接不同轮次的 ADE/FDE 最小值。

现有 GNN 是先 40 轮、再保留模型/AdamW/RNG 续到 100 轮，其物理分批中途调整。本轮 QGNN 从头 100 轮，训练超参和样本曝光相同，但不声称 dropout 随机轨迹逐位相同。结果为单 seed 的 V_select 比较，不是多种子或独立测试证据。

## 用户手动启动

在服务器终端运行：

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u experiments/qgnn_inherited/run.py --run
```

程序在当前终端显示进度；中断后追加 `--resume` 恢复：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgnn_inherited/run.py --run --resume
```

无参数执行只检查准备状态；`--self-check` 检查接口、梯度、公共时间模块初始化和参数分组，不执行参数更新。只有显式 `--run` 进入训练；已有断点须显式 `--resume`，同一结果目录具有训练互斥锁。

## 输出

- `reports/qgnn_inherited100/checks.json`：准备验收及零参数更新记录。
- `reports/qgnn_inherited100/protocol.json`：手动启动时保存的协议。
- `results/qgnn_inherited100/seed2026/qgnn/`：独立断点、逐轮历史、验证明细和汇总。
- `reports/qgnn_inherited100/comparison.json`：完成后的 QGNN / 已有 GNN 比较。

不会覆盖既有 QGAT 或 GNN 结果。
