# ICCT QGNN 研究交接（2026-09-20）

## 0. 给师兄的最短说明

这个仓库研究三基站 ISAC 场景下的多车辆轨迹预测。当前真正卡住的不是代码能否运行，而是：**怎样从任务瓶颈出发，找到最合适、可公平验证、能够形成论文贡献的 QGNN 路线。**

我们不希望你只在现有版本中选择。TRC-QGNN、TO-JQGNN、Raj-style、RC-HQGNN、Scene-Adaptive RC-HQGNN 等都是已有探索和证据材料，不是封闭候选集合。希望你帮助判断：QGNN 是否适合介入、应该介入哪个环节，以及最值得正式验证的具体结构；答案可以沿用、组合或修改既有方案，也可以是此前完全不存在的新方案。

本分支冻结到服务器提交 `bdd4aa0`：两种最新提出的 TRC-QGNN 与 TO-JQGNN 已完成工程预检，但尚未启动 15,802-window 正式训练，预测 test 仍关闭。

## 1. 一句话任务与当前链路

```text
SinD 真实轨迹
→ 三基站受控 ISAC 感知
→ 20 帧 sensed history [x_hat, y_hat, vx_hat, vy_hat]
→ 多车交互模型（QGNN 或 matched classical control）
→ 连续 interaction representation / tokens
→ 固定 GPT-2 backbone + LoRA / Motion Token 接口
→ 20 帧未来轨迹
→ ADE、FDE
```

当前固定边界：

- 数据：SinD public Changchun + Xi'an；不是完整 SinD 2.0。
- 任务：20 history → 20 future，四轮车辆，history-only 因果选择。
- 感知：正式输入来自冻结的 3-BS controlled-ISAC sensing cache，不允许模型读取 GT history。
- 交互：最多 8 个 model-facing vehicles 是当前缓存/工程接口，不是理论上限。
- 评价：ADE、FDE 分别报告；`J=ADE+0.5*FDE` 只用于验证集 checkpoint 选择。
- 数据权限：模型研究阶段只使用 train/validation；prediction test 尚未打开。
- 公平性：量子和经典模型必须匹配数据、split、seed、训练曝光、checkpoint 规则和 downstream。

数据迁移的权威说明见 [`docs/dataset_migration/SIND_MIGRATION_HANDOFF.md`](docs/dataset_migration/SIND_MIGRATION_HANDOFF.md)。

## 2. 为什么现在仍是开放研究问题

tag1 组会后，我们不是沿着一条固定 QGNN 不断调参，而是先后改变或审计了：

1. 感知前端与输入标签；
2. 数据集（Lankershim → Automatum → SinD）；
3. NoGraph、普通 pairwise graph、physics-routed graph 和 higher-order graph；
4. LLM / Motion Token 接口；
5. 多种量子交互机制与 matched classical controls；
6. 训练规模、初始化、反馈、readout 和 checkpoint 规则。

因此，历史 ADE/FDE 不能直接排成一个模型排行榜。某次变好可能来自数据、输入、训练预算、经典对照或 downstream，而不一定来自 QGNN；某次失败也只能约束当时的结构和协议，不能证明整类路线无效。

完整时间线见 [`docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md`](docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md)。

## 3. tag1 后已经得到的主要研究事实

### 3.1 数据与任务空间

- Automatum 上，Graph + Motion-Token LLM 的全局边际收益较小，但不同交互强度场景间存在明显异质性；这促使我们重新检查数据是否提供足够多的多车交互。
- SinD 迁移后，history-only 高交互筛选得到 19,768 个 model-facing windows；正式 train/validation 分别为 15,802 / 1,880。
- SinD Q0 诊断中，普通两车、自适应两车、两车+三车关系相对 NoGraph 都出现开发集改善；pair+triplet 的 0 dB、seed 2026、20-epoch 诊断约改善 ADE 4.95%、FDE 5.56%。这是任务空间证据，不是量子优势。

### 3.2 RC-HQGNN（Round 2）

- 使用 graph-conditioned joint ZZ/ZZZ evolution、连续物理关系输入和 relation-carrying cumulant readout。
- 4096-window pilots 经 v1/v2/v3 两次结构修正后仍未超过强 matched classical higher-order control。
- full 15,802-window、0 dB、seed 2026、20 epoch：量子 ADE/FDE `0.51094460/1.12605076`，matched classical `0.49146503/1.07928883`；量子全局更差约 3.96%/4.33%。
- closing>=15 的高动态子集曾出现正收益，但不能替代全体结果。

权威记录：[`docs/qgnn/ROUND2_EXECUTION_LOG_20260919.md`](docs/qgnn/ROUND2_EXECUTION_LOG_20260919.md)。

### 3.3 Scene-Adaptive RC-HQGNN（Round 3）

- 尝试 scene-conditioned coupling、K2/K3 跨轮反馈、signed phase、basis feedback 和 neutral initialization。
- 三版结构与一次中性初始化都没有通过预注册 full-train gate，因此没有启动 full train、多 SNR 或多 seed。
- 预注册的反馈 on/off 从头训练表明，联合反馈在一次有限预算下给量子模型带来约 0.40% ADE、0.88% FDE 改善，但量子仍比 matched classical 差约 2.58%/2.49%。
- 这说明反馈有弱正贡献，但不能证明 K3 单独因果有效，也没有解决全局经典差距。

权威记录：[`docs/qgnn/ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md`](docs/qgnn/ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md) 和 [`docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md`](docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md)。

### 3.4 Raj-style subset QGNN（Round 4）

- 将 pair/triplet subset identity、history-only 连续物理关系和 weighted Johnson quantum mixing 结合，matched control 为 multi-j JohnsonGIN。
- 4096-window、0 dB、20-epoch 诊断在 seed 2026/2027 均出现量子两指标正信号，两 seed 平均相对增益约 ADE 2.70%、FDE 2.96%。
- full 15,802-window、seed 2026 后几乎打平：量子 `0.481258665/1.051172996`，Johnson `0.482314337/1.049728396`；量子 ADE 略好 0.219%，FDE 略差 0.138%。不能称为稳定量子优势。
- paper-math fidelity、feature builder、row-local reupload、adjacency operator 等隔离实验提供了机制信息，但没有把单一论文复现支线升级为已确定主方案。

权威记录：[`docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md`](docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md) 和 [`docs/qgnn/ROUND4_RAJ_NEW_CHAT_HANDOFF_20260919.md`](docs/qgnn/ROUND4_RAJ_NEW_CHAT_HANDOFF_20260919.md)。

### 3.5 SQM-GNN 路线审计

- 论文层面的 node/edge register、`U_MSG/U_UPD` 和 star-subgraph message passing 与本项目方向有契合点。
- 没有找到可验证的官方实现；论文没有给出足以逐门恢复 `U_MSG/U_UPD` 的完整 gate-level ansatz。
- 该路线值得作为设计素材，但尚未形成可公平训练的冻结实现。

权威记录：[`docs/qgnn/QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md`](docs/qgnn/QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md)。

### 3.6 2026-09-20 新增的两种候选

- **TRC-QGNN**：rooted pair / ordered rooted-triplet、风险条件 configuration Hamiltonian、35D Pauli/correlator readout；matched control 为 RTCN-128。
- **TO-JQGNN**：最多 6 车的 joint register、四阶段 noncommuting Hamiltonian families、31D root-RDM/correlator readout；matched control 为 TR-TGN。
- 四个 arms 已通过梯度、排列、padding、float64、checkpoint save/load、暂停恢复和实际数据 B32 smoke。
- 尚未启动 15,802-window 正式训练，因此没有 ADE/FDE 效果结论。

代码目录中的 `prediction/qgnn_finalists/` 和文档标题中的 `finalists` 是开发期内部命名，不表示这两种方案已经经过全部历史路线统一比较并被正式选中。

权威记录：[`docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md`](docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md)。

## 4. 运行方式会不会影响实验效果

会，而且需要区分“改变实验定义的因素”和“工程上本应等价、但可能造成数值漂移的因素”。

### 4.1 会实质改变结果或比较含义

- 数据集、数据版本、清洗和高交互筛选规则；
- train/validation split、窗口数量、采样顺序和重叠结构；
- SNR、sensing cache 版本和是否错误使用 GT history；
- seed、初始化和是否加载旧 checkpoint；
- batch size、学习率、optimizer、scheduler、epoch、early stopping；
- loss 组成、权重和 correction cap；
- checkpoint 选择规则及 ADE/FDE 聚合方式；
- FP32/FP64/AMP、单卡/DDP、梯度累积；
- padding、mask、邻居选择、时间锚点和 invalid-edge 规则；
- resume 是否恢复 optimizer、scheduler 和 Python/NumPy/Torch/CUDA RNG。

### 4.2 理论上不应改变科学结论，但可能造成小漂移

- GPU 0/1、CUDA 非确定性 kernel；
- dataloader worker 数量；
- PyTorch/CUDA/PennyLane 版本；
- 同机并发和资源争用；
- 日志、checkpoint I/O 时机。

这些漂移通常不能解释稳定的大幅收益，但当模型只相差 0.1%～1% 时，足以改变最佳 epoch 或胜负方向。Round 3 已观察到同 seed 经典重复训练不逐数一致，因此不能把“同 seed”写成“数学上完全相同”。

### 4.3 历史结果的可比性规则

只有同时匹配以下字段的实验，才允许把差异主要归给模型：

```text
dataset + data hash + split + sensing/SNR
train indices/order + seed + initialization
downstream + loss + optimizer/scheduler
batch size + epochs + checkpoint rule
metric aggregation + test-access status
```

不同可比组之间只能用于形成假设，不能直接排行。特别是：

- Lankershim、Automatum、SinD 的绝对 ADE/FDE 不直接比较；
- 4096-window pilot 与 15,802-window full train 不直接比较；
- 单 seed、两 seed、多 seed 的证据强度不同；
- implementation/preflight、资源 profile、validation ADE/FDE、test ADE/FDE 是四类不同证据。

## 5. 希望师兄帮助回答的问题

请不要受现有实现数量或名称限制。我们更希望得到一条任务驱动的研究路线，而不是在当前版本中二选一。

1. 当前任务中，真正需要超越普通 pairwise GNN 的瓶颈是什么：高阶交互、关系身份保持、全局联合态、动态风险条件化，还是别的问题？
2. QGNN 应该介入图交互、时序建模、readout，还是只作为某种可检验的关系算子？
3. 哪些已有负结果能够排除一类设计，哪些只是训练预算、readout 或数据协议造成的局部失败？
4. Raj-style 小样本正信号在 full train 中消失，最值得优先验证的解释是什么？
5. 现有 TRC/TO 设计是否真正对应任务瓶颈，还是仍然过于“先设计量子线路，再寻找任务解释”？
6. 最合适的 classical controls 应怎样匹配，才能避免通过弱 baseline 制造优势？
7. 下一轮最小但有辨识力的实验应该是什么？需要哪些机制消融，什么条件下停止？
8. 如果已有路线都不理想，最值得新设计的 QGNN 结构是什么？

## 6. 建议阅读顺序

如果只有 30 分钟：

1. 本文；
2. [`docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md`](docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md)；
3. [`docs/dataset_migration/SIND_MIGRATION_HANDOFF.md`](docs/dataset_migration/SIND_MIGRATION_HANDOFF.md) 的 Executive decision、Model-facing contract、Remaining risks；
4. [`docs/qgnn/ROUND2_EXECUTION_LOG_20260919.md`](docs/qgnn/ROUND2_EXECUTION_LOG_20260919.md) 的 Final acceptance；
5. [`docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md`](docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md) 的结论与四组结果；
6. [`docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md`](docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md)；
7. [`docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md`](docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md)。

如果要看代码，建议按以下入口：

| 主题 | 入口 |
|---|---|
| SinD 模型输入 | [`frontend/sind_prediction_dataset.py`](frontend/sind_prediction_dataset.py) |
| 三基站受控感知 | [`frontend/controlled_isac/sind_frontend.py`](frontend/controlled_isac/sind_frontend.py) |
| Q0 Graph / Motion-Token | [`prediction/q0/`](prediction/q0/) |
| RC-HQGNN | [`prediction/qgnn_final/`](prediction/qgnn_final/) |
| Scene-Adaptive RC-HQGNN | [`prediction/qgnn_adaptive/`](prediction/qgnn_adaptive/) |
| Raj-style 路线 | [`prediction/qgnn_paper_native/`](prediction/qgnn_paper_native/) |
| 最新 TRC/TO 路线 | [`prediction/qgnn_finalists/`](prediction/qgnn_finalists/) |
| 最新训练入口 | [`scripts/train_qgnn_finalists.py`](scripts/train_qgnn_finalists.py) |
| 最新工程检查 | [`scripts/check_qgnn_finalists.py`](scripts/check_qgnn_finalists.py) |

## 7. 冻结点与证据边界

- 分支：`handoff-20260920`
- 基础提交：`bdd4aa0a21889c9a0e987ece96db620a19b0a323`
- tag1：`stage1-meeting-freeze-20260915` / `8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7`
- 服务器运行权威：`/home/dell/YrM/ICCT`
- 服务器 Python：`/home/dell/YrM/envs/ICCT/bin/python`
- 正式训练进程：冻结时无正在运行的 QGNN 正式训练
- 最新两种方案：只完成工程预检，不存在正式 ADE/FDE
- prediction test：关闭

原始数据、sensing cache、checkpoint、逐窗口大结果和完整日志不进入 Git。仓库保留代码、配置、可复核的小型结果、研究报告以及产生这些结果的 Git 历史。
