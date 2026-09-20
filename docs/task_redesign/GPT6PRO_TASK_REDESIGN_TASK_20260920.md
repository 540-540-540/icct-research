# ICCT 研究任务重设计专项 — GPT-6 Pro 深度研究任务书（2026-09-20）

## 0. 任务定位

你现在接手 ICCT 项目中一个比 QGNN 架构本身更上游、更关键的问题：

当前 trajectory prediction task 是否真正需要多车交互信息？如果当前 2 s history → 2 s future 主要可以被单车惯性/平滑运动模型解决，应如何重新设计一个科学合理、可复现、非 cherry-pick 的预测任务，使 LLM、自车动力学建模、多车交互图建模和 QGNN 的研究叙事自然成立？

本轮角色：

独立任务设计研究员 + 轨迹预测审稿人 + benchmark 设计者 + 叙事逻辑审查者

最终必须给出：
1. 一个唯一主任务方案；
2. 最多一个辅助 benchmark；
3. 选择理由、物理依据、文献依据、数据可行性、实验判据；
4. 明确否决其余候选方案的原因。

不要只给候选列表，不要回避最终决策。

## 1. 项目权威

服务器正式项目：/home/dell/YrM/ICCT
正式分支：qgnn

服务器当前状态优先于本地与 GitHub。开始时确认 branch、HEAD、workspace status。不得 reset 到 stale remote，不得删除历史实验。

本轮原则上以研究、审计、任务设计为主。可以创建研究报告、只读分析脚本和 machine-readable summary，但不要修改正式 QGNN、ISAC、LLM 训练实现。

建议产物：
docs/task_redesign/
reports/task_redesign/

## 2. 当前已知问题

当前冻结预测任务：
- SinD
- history = 20 frames ≈ 2 s
- future = 20 frames ≈ 2 s
- ≈10 Hz
- 输入来自冻结 3-BS ISAC sensing estimate
- 每目标状态 [x_hat, y_hat, vx_hat, vy_hat]
- N<=8 为当前工程接口

用户最新独立尝试发现：

不输入任何多车图信息，只针对单目标自身历史，用经典 LSTM / TCN / 普通 Transformer，ADE 已经约 0.4 m，FDE 约 0.9 m。

这些数字当前视为待服务器结果文件核验的最新实验事实。如果服务器有对应结果，必须读取核实；如果尚未沉淀，明确标注为 user-reported observation，不要伪造精确表格。

核心风险：

当前任务可能主要是短时惯性/局部运动平滑外推，而不是 interaction-dependent trajectory prediction。若成立，无论 GNN 还是 QGNN，理论提升空间都会被大量 easy/self-motion samples 稀释。

## 3. 希望建立但不能预设成立的论文逻辑

### 3.1 LLM 建模单车时序先验

GPT-2 + LoRA / Motion Token 分支用于学习目标车辆自身运动历史的时间规律。

希望检验：
GPT-2 + LoRA > LSTM / TCN / ordinary Transformer

但禁止通过反复修改 benchmark 阈值直到 GPT-2+LoRA 赢。

如果 GPT-2+LoRA 在科学合理候选任务上始终无法胜过这些 self-only baseline，必须明确指出，并提出 LLM 模块本身需要怎样调整，而不是篡改 benchmark。

### 3.2 密集交通下 self-only 不够

在道路密集、冲突关系显著、未来动作受周边车辆约束时，应检验：
best self-only model < multi-agent interaction model

且差距应具有实际意义，不只是 seed noise。

### 3.3 图建模自然引出 QGNN

只有在多车 interaction information 被证明真正有额外价值后，才有科学理由比较：
Classical GNN vs QGNN

QGNN 不得因为 benchmark 设计而被预设为赢家。

本轮只需保证最终 task 有足够 interaction headroom，让 GNN/QGNN 比较本身有意义。

## 4. 科学硬约束：禁止 cherry-pick

### 4.1 benchmark inclusion 只能用预测时可获得信息

允许：
- history trajectory
- history sensing estimate
- 当前/历史相对位置
- 当前/历史相对速度
- closing speed
- TTC
- DCPA
- history-only risk
- active vehicle count
- local density
- 静态道路/路口语义，如数据正式提供
- conflict-zone / lane / intersection geometry，如可可靠冻结

禁止用于 sample selection：
- future trajectory
- future ADE/FDE
- 某模型是否预测失败
- 某模型是否从 graph 受益
- future collision / future maneuver outcome
- QGNN 是否优于 GNN

### 4.2 Threshold 不得用 formal val/test 反复调

优先采用：
- 物理可解释阈值
- 文献标准
- train-only 分布分位数
- train 内部 design split / cross-validation

formal test 必须关闭。
formal validation 不能被用来挑“最有利故事”的 benchmark threshold。

## 5. 必须研究的任务设计维度

### A. Forecast horizon

至少研究：
- 2 → 2 s
- 2 → 3 s
- 2 → 4 s

若文献和数据支持，可研究 3→3、3→4、3→5 等。

核心判据不是 self-only ADE 是否变大，而是：
Graph Gain = Error(best self-only) - Error(strong graph)
是否随 horizon 或 interaction 强度显著扩大。

### B. Dense / interaction-critical 定义

必须研究 history-only 交互指标：
- active vehicle count
- radius 内 neighbor count
- minimum inter-vehicle distance
- closing speed
- TTC
- DCPA
- history-only collision/interaction risk
- simultaneous interacting neighbor count
- pairwise / triplet conflict density
- intersection conflict region，如数据支持
- following / crossing / merging / turning 等历史可辨识结构

不能只用“车辆数 >= X”作为唯一交互定义，除非证明它与真实 interaction necessity 高度相关。

### C. Prediction target

比较：
1. absolute future trajectory
2. local-frame / displacement trajectory
3. residual-to-kinematic baseline，例如 ΔY = Y_GT - Y_CV
4. residual-to-self model，例如 ΔY = Y_GT - Y_self

重点判断：
让 graph/QGNN 专门预测 interaction-induced deviation，是否比让它重新预测整个未来更符合论文问题。

### D. Target / neighbor formulation

研究：
- all-agent simultaneous prediction
- target-agent prediction with neighbor context
- ego-centered subgraph prediction
- conflict-centered target selection

判断哪种最适合：
LLM self-motion + QGNN interaction + future trajectory

### E. Interaction difficulty / modality

评估：
- acceleration/deceleration response
- crossing conflict
- merging/yielding
- lane-change / lateral motion
- turning
- multi-neighbor/triplet interaction
- multimodal future

不要为了“更难”添加与 SinD 不匹配的任务。

## 6. 必须进行文献检索

使用 Web / arXiv / 官方论文页面严格检索。

重点：
1. 近三年 motion forecasting 常见 observation/prediction horizons；
2. dense traffic / intersection / interaction-aware trajectory prediction；
3. interaction-critical / collision-risk / social-interaction benchmark 定义；
4. self-motion vs interaction information 消融；
5. LLM / pretrained Transformer 用于轨迹、时间序列、motion-token prediction；
6. GPT-2 / LoRA / frozen pretrained backbone 为什么可能优于从零训练 LSTM/TCN/Transformer；
7. residual motion forecasting / kinematic residual prediction；
8. SinD 原论文及后续 trajectory-prediction 使用方式。

优先近三年；经典基础文献可以补充。

所有关键结论必须有可靠来源，并区分：
- 文献事实
- 项目实验事实
- 数据审计事实
- 推断/建议

## 7. 必须读取的项目资料

先搜索服务器最新文件，至少覆盖：

### 当前数据/任务
- SinD 数据处理、split、sample contract
- trajectory dataset/cache
- ISAC sensing cache contract
- 2s→2s 设置相关 config

### self/graph 历史证据
搜索：
- no-graph
- MPNN
- pairwise
- pair+triplet
- routed graph
- motion LLM
- LSTM
- TCN
- Transformer
- GPT2 / LoRA
- ADE / FDE / J

历史上下文曾有：
- no graph ADE ~0.524
- pairwise ~0.502
- pair+triplet ~0.498

必须以服务器正式结果为准。

### QGNN 最新状态
只需理解方向，不要继续改：
- docs/qgnn/RAJ_PENNYLANE_RESIDUAL_MASTER_TASK_20260920.md 若实际路径存在请核对
- docs/qgnn/RAJ_PENNYLANE_P1_1_REPAIR_AUDIT_20260920.md
- reports/qgnn/raj_pennylane_p1_1/repair_preflight_20260920.json

## 8. 数据审计要求

允许写只读分析脚本。

至少回答：
1. SinD train/val/test 窗口数；
2. active vehicle count 分布；
3. history-only interaction metrics 分布；
4. 高 interaction 条件剩余多少 train/val 样本；
5. 不同 forecast horizon 可构造多少窗口；
6. horizon 增加是否严重损失样本；
7. interaction-critical samples 是否集中于少数 scene，是否有 scene leakage 风险；
8. train/val interaction distribution 是否一致；
9. 当前 3-BS ISAC cache 对更长 horizon 的支持方式；
10. future GT target 是否能从原始轨迹重构。

不要打开 formal test future labels 做 benchmark selection。

## 9. 候选任务评价框架

必须预先定义，不得按 QGNN 输赢打分。

### Interaction necessity
best self-only vs strong classical graph 的 Graph Gain 是否足够大且多 seed 稳定？

### LLM usefulness
公平 self-only setting：
GPT2+LoRA vs LSTM / TCN / ordinary Transformer

必须控制：
- identical input
- identical train data
- identical checkpoint-selection protocol
- comparable tuning budget
- no test use
- disclose parameter counts

### Sample sufficiency
不能只剩极少 extreme samples。你负责定义最低样本量并给理由。

### Physical legitimacy
任务必须对应真实交通预测问题。

### Benchmark stability
阈值小幅变化不能导致结论完全翻转。

### QGNN headroom
任务需要 graph，但不能预设 quantum advantage。

### Computational feasibility
2×4090 和当前项目周期内可完成。

## 10. 核心 interaction 信息问题

概念上研究：

I(Y_future ; H_neighbors | H_ego)

不要求精确估计 mutual information，但必须用可操作实验近似。

推荐：
Interaction Gain = Error(best self-only) - Error(strong graph)

按 horizon 和 interaction level 分层。

如果 Interaction Gain 在某候选任务明显扩大，说明该任务更适合研究 multi-agent graph reasoning。

## 11. GPT-2 + LoRA 专项审查

这是本轮核心之一。

### 为什么 GPT-2？
必须具体解释：
- frozen pretrained blocks 的价值
- LoRA 学什么
- Motion Token / continuous adapter 做什么
- pretrained text Transformer 向 motion sequence 迁移的合理依据
- 如果 input embedding 重建，预训练知识还保留在哪里

### 为什么可能胜普通 Transformer？
若 GPT2+LoRA 赢，需要分析来源：
- pretraining initialization
- frozen sequence prior
- parameter-efficient adaptation
- low-data generalization
- tokenized motion regularization
- longer-horizon pattern modeling

不能只归因于“参数更多”。

### 公平 baseline
至少审查：
- LSTM
- TCN
- scratch Transformer
- GPT-2 frozen
- GPT-2 + LoRA

判断是否还需要：
- same-width Transformer
- same-trainable-param Transformer
- pretrained Transformer control

最终给出最小但足够强的 baseline 组合。

## 12. 必须比较的叙事候选

### Narrative A — Long-horizon interaction
LLM 做 self-motion prior；随着 prediction horizon 增加，interaction becomes necessary。

### Narrative B — Interaction-critical benchmark
保留 full benchmark，但以预先定义的 dense/interaction-critical benchmark 作为主要 interaction 分析任务。

### Narrative C — Interaction residual forecasting
LLM 预测 intrinsic/self trajectory；QGNN 预测 neighbors 导致的 deviation。

可以提出 Narrative D，但必须和 A/B/C 对比。

## 13. 最终主任务必须精确到可直接实现

最终推荐必须明确：
- history seconds / frames
- future seconds / frames
- sampling rate
- target state
- coordinate system
- target-agent definition
- neighbor graph definition
- interaction-critical inclusion formula
- thresholds
- threshold 来源
- minimum active cars
- local radius
- TTC/DCPA/closing 条件
- 是否要求至少两个 interacting neighbors
- train/val sample count 预估
- split policy
- 是否保留 full benchmark
- primary metrics
- secondary metrics
- self-only baseline
- graph baseline
- QGNN comparison protocol
- LLM comparison protocol
- checkpoint metric
- test opening condition

要达到下一轮执行 agent 可以直接实现的程度。

## 14. 三类 Gate

### Gate S — Self-model gate
GPT-2+LoRA 对 LSTM / TCN / ordinary Transformer 的优势。

你负责定义：
- 比较指标
- multi-seed 稳定性
- 有论文意义的最小效果幅度

### Gate G — Graph-necessity gate
Strong classical graph 对 best self-only 的优势。

这是本轮最重要 gate。

### Gate Q — QGNN eligibility gate
只有 Gate G 通过，才值得继续 QGNN vs GNN。

Gate Q 此处不要求 QGNN 一定赢，只要求 task 确实存在 graph reasoning headroom。

## 15. 本轮禁止

- 改正式 QGNN core
- 改 Raj-PennyLane P1.1
- 重跑 full QGNN
- 打开 test
- 用未来结果筛样本
- 以“QGNN 赢”为 benchmark 选择标准
- 以“GPT-2 赢”为 benchmark 选择标准
- 删除旧实验
- 覆盖正式 split
- 大规模无计划训练

允许：
- Web / literature research
- 数据统计
- 读取已有结果
- cheap re-evaluation
- 小型 diagnostic / proxy experiment
- 设计正式实验计划

## 16. 最终交付

### 主报告
docs/task_redesign/GPT6PRO_OPTIMAL_TRAJECTORY_TASK_SELECTION_20260920.md

至少包含：
1. Executive conclusion
2. Why 2s→2s is problematic
3. Literature evidence
4. SinD/data audit
5. Candidate task space
6. Candidate comparison
7. GPT2+LoRA role audit
8. Interaction necessity analysis
9. Final selected primary task
10. Optional secondary benchmark
11. Exact implementation contract
12. Experiment matrix
13. Scientific risks
14. Rejected alternatives
15. Final paper narrative

### Machine-readable summary
reports/task_redesign/optimal_task_selection_20260920.json

至少包含：
- recommended_task
- auxiliary_task
- thresholds
- expected_sample_counts
- literature_sources
- self_baselines
- graph_baselines
- gate_S
- gate_G
- gate_Q
- rejected_candidates
- unresolved_questions

### 下一阶段执行计划
明确：
- data rebuild
- baseline verification
- LLM verification
- graph-necessity verification
- QGNN reopening condition

## 17. 最终必须决策

不要以“建议试 2→3 和 2→4 再看看”结束。

可以设计小型验证矩阵，但最终必须基于：
- 文献
- SinD 数据
- 项目已有结果
- 理论逻辑
- 计算成本

明确给出：

我推荐 ICCT 正式主任务采用 X。

并说明为什么不是 Y/Z。

若证据仍不足以完全冻结，也必须给：

主推荐 X；只有当预注册 Gate G 在 train-design split 失败时才切换备用 Y。

## 18. 核心论文故事目标

目标是找到最合理、最公平、最可能被实验真正支持的任务设置，使以下叙事能够被验证：

1. 单车短期运动具有强时间规律；
2. GPT-2+LoRA 学习 self-motion prior；
3. 低交互场景 self-motion 已足够；
4. 密集/冲突/更长 horizon 下邻车历史提供显著条件增益；
5. 因此需要显式 multi-agent graph reasoning；
6. higher-order multi-vehicle interaction 进一步自然引出 Raj-style QGNN；
7. 最终职责分离：
   - LLM：self-motion temporal prior
   - QGNN：interaction-induced correction / multi-agent reasoning

你的任务不是证明这段话，而是找出一个最合理、最公平、最可能被实验真正支持的 task，并指出任何不能成立的环节。

## 一句话任务定义

重新设计 ICCT trajectory prediction task，使研究问题从“短时惯性外推”转化为“LLM 学习单车运动先验、图模型解决密集交通中的额外交互不确定性、QGNN承担 higher-order interaction reasoning”，同时严格避免 future-based filtering、validation cherry-picking 和预设模型胜负。

