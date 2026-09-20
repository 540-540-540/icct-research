# Tag1 组会后研究尝试台账

## 1. 范围与使用方法

本台账覆盖：

```text
stage1-meeting-freeze-20260915
8adc24c  Add BDX-01 QGNN bottleneck diagnostics
        ↓
bdd4aa0  Complete QGNN finalist engineering preflight
```

“完整”指全部有独立科学问题、模型机制、数据决策或对照意义的尝试；一次方案内部的修复、文档补充、checkpoint 增强和向量化等工程提交合并记录，不把它们伪装成新模型。

状态词仅描述该次尝试，不表示永久淘汰或最终选中：

- **诊断**：用于回答问题，不是正式性能结论；
- **工程通过**：实现/梯度/资源检查通过，不等于 ADE/FDE 有效；
- **有限正信号**：在指定开发协议下为正，不外推到其他规模/seed/test；
- **负/中性**：对应协议下未通过门禁，不否定所有相关设计；
- **未正式训练**：只有设计和工程证据。

## 2. 总时间线

| 阶段 | 日期 | 研究问题或尝试 | 主要提交 | 状态 |
|---|---|---|---|---|
| 感知重建 | 09-15～09-16 | 3-BS 多目标 ISAC、融合、跟踪、B64/F01E 数据接口 | `0cac17b`～`0e59236` | 前端与数据工程闭环；不是QGNN效果实验 |
| 数据适配 | 09-16～09-18 | SNR敏感性、Automatum受控ISAC、Lankershim/Automatum审计 | `e35f905`～`0ffe779` | 暴露任务空间与数据适配问题 |
| Q0诊断 | 09-18 | NoGraph/MPNN/GATv2、Simple-GRU/GPT-2、Motion Token、physics routing | `12d9a4a`,`5fce328` | 0dB单seed诊断；图边际收益小且异质 |
| SinD迁移 | 09-18～09-19 | 高交互数据迁移、3-BS cache、模型接口、pair/triplet任务空间 | `0d541aa`,`ddbcc40`,`be3dc2a`,`7edde41`,`213ff9c` | 迁移接受；形成当前正式数据主线 |
| Round2 | 09-19 | Joint Hypergraph QGNN v1/v2/v3 → RC-HQGNN | `269544d`,`53844e5`,`2569745`,`591030c` | full-train整体负；高动态子集正 |
| Round3 | 09-19 | Scene-Adaptive RC-HQGNN、反馈、phase/basis、neutral初始化 | `dbb27b8`～`28cfaf0` | 三版及反馈消融完成；整体仍负 |
| Round4 | 09-19 | Raj-style subset QGNN、paper-native controls、fidelity/机制隔离 | `ae1adf6`～`bee1974` | 4096两seed正；full-train近似打平 |
| SQM审计 | 09-19 | SQM-GNN node/edge register message passing | `9ad7c1a`,`094def5` | 有研究契合点；论文复现信息不足 |
| 最新新增 | 09-20 | TRC-QGNN、TO-JQGNN及RTCN-128/TR-TGN | `c8afb07`,`bdd4aa0` | 工程预检通过；正式训练未开始 |

## 3. 非QGNN但影响选型的尝试

### 3.1 3-BS ISAC 与数据标签链

尝试内容：

- 重建共享 ISAC sensing core、echo geometry、multi-BS fusion 和 CV tracking；
- practical SNR、contiguous burst、B64 production frontend；
- F01E label attribution、origin-safe label、identity switch 审计；
- sensing cache 和 model-agnostic prediction dataset。

为什么与QGNN选型有关：如果输入标签、身份连续性或 sensing 误差发生变化，后续模型的 ADE/FDE 变化不能归因于图结构。

主要历史范围：`0cac17b`～`0ffe779`。当前主线后来迁移到 SinD，Automatum/F01E 仍作为数据工程与失败经验保存。

### 3.2 Q0：Graph 与 downstream 的边际作用

Automatum 0 dB / seed 2026 / validation-only 诊断包括：

- NoGraph + simplified GPT-2；
- Edge-MPNN + simplified GPT-2；
- NoGraph + Simple-GRU；
- MPNN + Simple-GRU；
- NoGraph / Plain MPNN / Physics-Routed MPNN + Motion-Token LLM；
- GATv2、local/pairwise、higher-order task-space diagnostics。

已知结果边界：Graph 的全局平均收益较小，某些高交互子集更有利；GPT-2 并没有简单地“吃掉所有图收益”，但不同 downstream 会改变图模块的表现。不能把任一组合当作全局正式基线。

证据：

- [`Q0_LLM_AND_TASK_SPACE_DIAGNOSTIC_20260918.md`](Q0_LLM_AND_TASK_SPACE_DIAGNOSTIC_20260918.md)
- [`Q0_DUAL_DIAGNOSTIC_FINDINGS_20260918.md`](Q0_DUAL_DIAGNOSTIC_FINDINGS_20260918.md)
- [`../../reports/q0/llm_marginal_effect_audit_0db.json`](../../reports/q0/llm_marginal_effect_audit_0db.json)

### 3.3 SinD Q0 pairwise / routed / higher-order诊断

统一为 SinD、0 dB、seed 2026、20 epoch、train/validation only：

- NoGraph；
- MPNN；
- GATv2；
- gated/routed MPNN；
- pairwise；
- pair+triplet higher-order。

pair+triplet 相对 NoGraph 的开发集改善约 ADE 4.95%、FDE 5.56%，但这是经典任务空间诊断，不能作为QGNN优势。其作用是说明显式多车和高阶关系值得被后续模型认真建模。

证据：[`SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md`](SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md) 和 [`../../reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`](../../reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json)。

## 4. QGNN结构尝试

### 4.1 Round2：Joint Hypergraph QGNN / RC-HQGNN

#### v1

- graph-conditioned joint ZZ/ZZZ evolution；
- 无量子线路前的 classical message passing；
- 最多8 qubits，N>8使用history-only ego patches；
- matched adaptive pair+triplet classical core；
- 4096 train / 1880 val / 12 epoch / seed 2026 / 0 dB。

结果：量子 `0.601817/1.255976`，adaptive classical `0.583580/1.205876`；无量子胜利。

#### v2

- 四个独立条件化8-qubit channels；
- weighted phase-energy normalization；
- 调整 pair/triple readout scaling；
- 同时增强经典对照为四头、三层pair+triplet attention。

结果：量子 `0.608360/1.265798`，经典 `0.581917/1.198834`；仍无优势。机制审计显示模型确实依赖entangler/ZZZ，但“被使用”不等于“改善预测”。

#### v3 / RC-HQGNN freeze

- relation-carrying cumulant readout，避免只做per-node mean/RMS；
- shared output correction cap 从4修正到16，量子和经典同时应用；
- full 15,802 train / 1880 val / 20 epoch / seed 2026 / 0 dB。

结果：量子 `0.51094460/1.12605076`，matched classical `0.49146503/1.07928883`，全局更差约3.96%/4.33%；closing>=15子集改善约4.29%/6.35%。

代码：[`../../prediction/qgnn_final/`](../../prediction/qgnn_final/)。证据：[`ROUND2_EXECUTION_LOG_20260919.md`](ROUND2_EXECUTION_LOG_20260919.md)。

### 4.2 Round3：Scene-Adaptive RC-HQGNN

共同目标：让量子耦合随场景、relation和前一轮K2/K3响应变化，同时给经典侧同等控制器和反馈能力。

#### v1：positive multiplier feedback

- scene/history-conditioned ZZ/ZZZ multiplier；
- K2/K3 relation feedback；
- 4096/1880/12 epoch/0 dB/seed 2026。

结果：量子 `0.609190/1.256607`，经典 `0.588472/1.214436`；full gate失败。

#### v2：phase feedback

- 将 `lambda*base` 改为有界相位增量，允许弱base relation被激活或改变符号。

结果：量子 `0.604980/1.261906`，经典 `0.587954/1.216045`；仍失败。

#### v3：basis feedback

- 恢复正倍率coupling；
- 前一轮node K2/K3 moments调制下一轮noncommuting RX mixing；
- 经典侧增加同样32个控制权重。

结果：量子 `0.609224/1.264438`，经典 `0.589355/1.217505`；仍失败。该版本只作为机制备份。

#### neutral initialization与反馈消融

- 同一v2 graph只改变控制器初始化，不加载旧R2 checkpoint；
- neutral量子 `0.601021648/1.241815257`，经典约 `0.587082428/1.212963395`；门禁仍失败；
- 反馈on/off四组从头训练：量子反馈带来约0.397% ADE、0.876% FDE改善，但有反馈量子仍输matched classical约2.58%/2.49%。

代码：[`../../prediction/qgnn_adaptive/`](../../prediction/qgnn_adaptive/)。证据：[`ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md`](ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md) 和 [`ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md`](ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md)。

### 4.3 Round4：Raj-style / paper-native subset路线

#### Weighted multi-j Raj-style QGNN

- j=2 pair subset + j=3 triplet subset；
- history-only physics-conditioned weighted Johnson quantum mixing；
- fixed-particle-number embedding / compound-RBS-style dynamics；
- continuous per-agent fusion；
- downstream沿用Motion-Token/GPT-2。

matched主对照：multi-j JohnsonGIN；辅助对照还包括GIN、PPGN-style higher-order、RC-HQGNN和历史强经典模型。

4096-window、20-epoch结果：

| Seed | Quantum ADE/FDE | Johnson ADE/FDE | Quantum相对变化 |
|---:|---:|---:|---:|
| 2026 | 0.516458 / 1.102781 | 0.525952 / 1.129782 | +1.81% / +2.39% |
| 2027 | 0.530895 / 1.148234 | 0.550423 / 1.189891 | +3.55% / +3.50% |

full 15,802-window、seed 2026：量子 `0.481258665/1.051172996`，Johnson `0.482314337/1.049728396`；ADE略好0.219%，FDE略差0.138%，属于近似打平，不是稳定量子优势。

#### Fidelity / paper-math / mechanism支线

为理解正信号来源，继续进行了：

- Raj paper-fidelity reconstruction；
- joint mixer数学与PennyLane核验；
- single-j / multi-j；
- 4096 transfer；
- weighted classical control；
- intermediate weighted match；
- feature-builder isolation；
- row-local reupload isolation；
- adjacency-operator isolation；
- representation-bottleneck audit。

这些支线用于区分subset identity、feature builder、adjacency composition、quantum evolution和readout的贡献。它们没有产生一个已经完成正式选型的新主方案，也不能把论文j-WL理论保证直接转写成连续轨迹ADE/FDE优势。

代码：[`../../prediction/qgnn_paper_native/`](../../prediction/qgnn_paper_native/)。证据：[`QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md`](QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md) 及全部 `ROUND4_*` findings/prereg文件。

### 4.4 SQM-GNN式quantum message passing

- node register + edge register；
- `U_MSG` 生成邻居消息，`U_UPD` 顺序更新中心节点；
- star-subgraph采样，具有适配本项目连续node/edge特征的潜力。

当前只有论文与实现可行性审计。没有找到可验证官方代码，论文没有完整给出gate sequence、observables、parameter sharing等细节，因此没有进入正式训练。

证据：[`QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md`](QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md) 和 [`SQM_GNN_UMSG_SOURCE_BOUNDARY_20260919.md`](SQM_GNN_UMSG_SOURCE_BOUNDARY_20260919.md)。

### 4.5 2026-09-20新增TRC-QGNN与TO-JQGNN

#### TRC-QGNN vs RTCN-128

- rooted pair `Omega1`、ordered rooted-triplet `Omega2`；
- 四个5-frame temporal blocks；
- 42D role descriptions及六种有向物理关系；
- risk-conditioned configuration Hamiltonian；
- 35D Pauli/correlator readout；
- matched RTCN-128共享输入、角色/边特征、风险势、token reader、GPT-2、loss、训练顺序和checkpoint规则。

#### TO-JQGNN vs TR-TGN

- root + 最多5个history-selected neighbors；
- 四阶段持续joint register，不reset、不feedback；
- 全部target-neighbor和neighbor-neighbor关系进入六类noncommuting Hamiltonian families；
- 31D root-RDM/correlator readout；
- matched TR-TGN共享selector、输入、四阶段/四轮、reader、GPT-2、loss与训练协议。

四个arms的工程预检、actual-data B32 smoke、checkpoint save/load和TO暂停恢复均通过。**没有正式15,802-window ADE/FDE。** `finalists`只是内部目录名。

代码：[`../../prediction/qgnn_finalists/`](../../prediction/qgnn_finalists/)。证据：[`finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md`](finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md)。

## 5. 运行协议与可比组

### A组：Automatum Q0诊断

- 数据：Automatum；
- 目的：LLM边际、图边际和任务空间；
- 结论只用于解释为何迁移数据和重新设计交互模型。

### B组：SinD Q0经典诊断

- 数据：SinD；0 dB；seed 2026；20 epoch；
- 目的：确认pairwise/higher-order interaction有可利用空间；
- 不与量子结构作直接优势结论。

### C组：Round2/3 4096-window pilots

- SinD；4096 train；1880 val；0 dB；seed 2026；12 epoch；B32；
- cap4与cap16结果必须分开；
- v1/v2/v3仅在matched exposure内比较。

### D组：Round2 full train

- SinD；15,802 train；1880 val；0 dB；seed 2026；20 epoch；B32；
- RC-HQGNN与其matched higher-order classical可直接比较。

### E组：Round4 4096-window paper-native

- SinD；4096 train；1880 val；0 dB；seed 2026/2027；20 epoch；B32；
- Weighted multi-j Quantum与multi-j JohnsonGIN在同seed内比较。

### F组：Round4 full train

- SinD；15,802 train；1880 val；0 dB；seed 2026；20 epoch；B32；
- 结论是近似打平，不能用4096-window结果替换。

### G组：TRC/TO工程预检

- 只有合成/小样本smoke、correctness、resume和profile；
- 没有正式效果，不能和A～F组按ADE/FDE排行。

## 6. 当前开放边界

1. 以上路线不是封闭候选集合；可以提出全新QGNN。
2. “负结果”只约束对应实现与协议，不自动淘汰整个理论方向。
3. “有限正信号”必须通过matched full-train、多seed和最终sealed test逐级验证。
4. QGNN价值必须从实际任务瓶颈出发，并与有竞争力的经典模型匹配；不能通过减少经典容量、训练轮数或输入信息制造优势。
5. 当前最需要的是更好的研究问题和可证伪实验，而不是继续无边界堆叠qubit、layer或controller。

## 7. Git回溯

查看tag1后完整提交历史：

```bash
git log --reverse --oneline stage1-meeting-freeze-20260915..handoff-20260920
```

查看某次尝试当时的完整代码：

```bash
git show <commit>:<path>
```

当前工作树保留截至 `bdd4aa0` 的完整最新实现；历史版本通过不可变commit回溯，不在工作树中重复复制。
