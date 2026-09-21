# ICCT 论文主线重构专项任务书（2026-09-21）

## 0. 任务定位

本任务是 ICCT 项目下一阶段的主线重构任务。

当前正式仓库：

`/home/dell/YrM/ICCT`

正式分支：

`qgnn`

当前已冻结 QGNN 主方案：

`Raj Weighted Multi-J QGNN (j=2+j=3)`

冻结说明：

`docs/qgnn/RAJ_QGNN_MAIN_SOLUTION_FREEZE_20260921.md`

本轮不再以“继续调 QGNN”作为优先级，而是补齐论文证据链：

```text
Task / Interaction Necessity
→ Classical Graph Necessity
→ Raj QGNN Advantage
→ LLM Utility
```

核心原则：

> 先证明问题存在，再证明经典图模型有必要，再证明 QGNN 相对经典图模型有增益，最后单独解决 GPT-2/LoRA 的效用问题。

---

## 1. 当前已确认事实

### 1.1 旧 2s→2s 任务过于容易

旧 SinD self-only 结果：

| 模型 | ADE / m | FDE / m |
|---|---:|---:|
| TCN | 0.376311 | 0.816700 |
| LSTM | 0.377409 | 0.822467 |
| Transformer | 0.388013 | 0.829766 |
| GPT-2 + LoRA | 0.604277 | 1.313128 |

因此不能叙述为“单车预测做不好，所以需要多车交互”。

正式论文问题应改为：

> 大量短时平滑轨迹可由目标车自身运动惯性解释，导致全场景平均 ADE/FDE 掩盖了多车交互建模的边际价值。

### 1.2 Raj QGNN 已经获得可用的 matched-classical 优势

4096-sample exact matched setting，frozen test 2086 scenes，两 seed：

- ADE: 0.502408 vs 0.517977，提升 3.006%
- FDE: 1.069034 vs 1.105594，提升 3.307%
- J: 提升 3.161%
- graph 参数量差约 0.67%

因此 Raj QGNN 当前不是最薄弱环节。

但该证据来自旧 2s→2s legacy benchmark。

本轮必须补齐：

```text
Self
→ Strong Classical Graph
→ Raj QGNN
```

而不是直接从 QGNN 结果倒推“交互必要”。

### 1.3 当前 Graph-Necessity 证据仍不稳定

现有未提交 q0 pilot 中：

- Automatum Simple-GRU: NoGraph 0.3795/0.6433，MPNN 0.4515/0.8370，Graph 更差
- Automatum Strong Local: 0.4566/0.7577，Pairwise 0.4991/0.8586，Graph 更差
- SinD GPT-2 交互诊断：NoGraph 0.5242/1.0638，Pair+Triplet 0.4982/1.0046，约 4.95%/5.56% 正向
- 某些 stronger-closing 子群可出现约 8.7% ADE / 10.2% FDE 增益

因此当前结论只能是：

> 交互信息可能有场景依赖的价值，但尚未建立稳定、可复现、可作为论文主线的 Graph > Self 证据。

### 1.4 GPT-2 + LoRA 当前没有 accuracy utility 证据

旧 SinD：

- TCN 0.3763/0.8167
- GPT-2+LoRA 0.6043/1.3131

Automatum q0：

- Simple-GRU NoGraph 0.3795/0.6433
- GPT-2+LoRA NoGraph 0.5440/0.9504

因此论文不能写“LLM 提升了轨迹预测精度”。

GPT-2/LoRA 必须独立诊断，不能与 Graph/QGNN 归因混在一起。

---

## 2. 本轮最终目标

建立一条可以直接支撑论文 Introduction → Method → Experiments 的证据链：

```text
普通短时任务掩盖交互
→ 构造 interaction-critical task
→ 证明 neighbors 带来稳定增益
→ 证明 classical graph 优于 self/capacity controls
→ 在同一新任务上验证 Raj QGNN > matched classical graph
→ 单独验证 GPT-2/LoRA 的预训练/适配效用
```

本轮优先级：

`Task Definition > Graph Necessity > GPT-2 Repair > Raj Migration`

---

## 3. Gate A：Task / Interaction Necessity

### 3.1 目标

回答：

> 在严格 history-only 定义的 interaction-critical trajectory prediction task 中，邻车信息是否能够稳定改善预测？

### 3.2 主候选任务

保留：

`SinD-IC4: history≈2s → future≈4s`

要求：

- target-centered
- history-only membership
- target slot 固定
- self-only 只看 target history
- graph 额外看 neighbors
- IC 与 Full 使用相同训练群，只作为 evaluation strata
- 禁止根据 future error / future maneuver / model performance 重新挑阈值
- 保留 Full2 / IC2 / Full4 / IC4
- 主任务为 IC4，Full4 为完整场景控制

### 3.3 必须先解决的数据因果性问题

必须完成 source-causality audit：

- history source
- future label source
- velocity derivation
- RTS smoothing 是否引入 future information
- 当前 benchmark 是否可称 strict online forecasting
- 若不能，明确定位为 offline latent-state simulation

P0 未明确前，不得做“严格在线预测”表述。

### 3.4 最小模型组

第一轮只用强经典模型，不使用 GPT-2，不使用 QGNN。

必须至少：

1. Strong Self
2. Own-only Capacity Control
3. Context Pooling Control
4. Strong Classical Graph
5. All-neighbor Strong Graph

共享：

- 同一 temporal backbone
- 同一 coordinate system
- 同一 decoder
- 同一 loss
- 同一训练/选点协议
- 同一 horizon
- 唯一区别应能明确归因到 neighbor information / interaction modeling

首选 temporal backbone：TCN。

### 3.5 Gate A 通过条件

Design pilot：seed 2026/2027。

至少满足：

- IC4 上 Graph 的 ADE/FDE 都优于 Strong Self
- 两 seed 方向一致
- Own-only capacity control 不能解释主要增益
- all-neighbor control 不显示 N<=8 丢失大部分收益
- GraphGain(IC4) > GraphGain(Full4)
- GraphGain(4s) > GraphGain(2s) 为优先支持方向，但若未成立，不强行解释

若 Graph 收益小、不稳定或反向：

- 先查训练、mask、neighbor contract、坐标、选点是否正确
- 不允许修改 IC 阈值追求漂亮结果
- 必须允许“当前任务不支持 Graph necessity”成为有效结论

---

## 4. Gate B：Classical Graph → Raj QGNN

只有 Gate A 通过后才能启动。

目标：

> 在同一 interaction-critical 新任务、同一 graph information contract 下，验证 Raj QGNN 是否稳定优于 matched classical graph。

要求：

- 不改 Raj 核心架构，除非迁移接口必要
- classical comparator 使用 Matched Multi-J Johnson core
- 公平匹配 graph core 参数量、输入、temporal encoder、decoder、loss、训练预算
- 保留 strong classical graph 和 all-neighbor graph 作为外部强基线
- 不因 QGNN N<=8 而削弱 classical baseline
- test 不参与设计

输出必须分别报告：

- ADE
- FDE
- J
- parameter count
- seed-wise paired gain
- IC / Full 分层
- 高交互 strata

目标不是要求必须达到旧任务 3%，而是检验旧优势是否迁移。

---

## 5. Gate C：GPT-2 / LoRA Utility

Graph necessity 与 LLM utility 分开研究。

### 5.1 目标

回答：

1. GPT-2 预训练是否在匹配条件下有正迁移？
2. 当前弱势是否主要来自冻结/LoRA 适配限制？
3. GPT-2 是否能达到或超过 strong classical temporal baseline？
4. 若不能，LLM 在论文中应承担什么角色？

### 5.2 优先比较

已有诊断协议基础上至少比较：

- pretrained + LoRA
- pretrained + full adaptation
- random init + full training
- strong TCN / GRU reference

控制：

- 同一 heading / coordinate information
- 同一 continuous motion input
- 同一 output representation
- 同一训练数据与预算
- 不把 random+full 与 pretrained+LoRA 直接拿来归因预训练作用

### 5.3 决策原则

- GPT-2 赢：可以保留为 accuracy contributor
- GPT-2 持平：只能保留为 representation / downstream interface，需谨慎表述
- GPT-2 明显输：不得硬写 accuracy advantage；要么继续修复，要么调整其论文角色

不以“GPT 必须赢”为实验成功标准。

---

## 6. Codex 本轮施工范围

Codex 第一阶段只负责 Gate A。

禁止一开始同时施工 Gate B/C。

必须按：

```text
AUDIT
→ DATA CONTRACT
→ PREFLIGHT
→ SMALL PILOT
→ DIAGNOSIS
→ SECOND-SEED CONFIRMATION
→ GATE DECISION
```

推进。

第一阶段主要交付：

1. 当前 q0 / task_redesign 结果独立审计
2. source-causality audit
3. IC4 数据 contract 代码与统计
4. Strong Self / Own / Pool / Graph / All-neighbor 实现
5. unit test / smoke test / preflight
6. seed2026 design pilot
7. 若有正信号，seed2027
8. Gate A 报告
9. 下一步是否允许进入 Gate B 的明确结论

---

## 7. 禁止事项

- 不修改已冻结 Raj QGNN 主结果来追求更高分
- 不根据验证结果重挑 IC 阈值
- 不削弱 classical baseline
- 不 cherry-pick seed
- 不把 validation 当 test
- 不读取或使用 test label 进行设计
- 不把相关性描述成交通因果
- 不把 Graph gain 描述成“GNN 不可替代”
- 不把 QGNN gain 描述成量子硬件优势
- 不把当前 state-vector simulation 描述成真实量子硬件结果
- 不因为 GPT-2 当前弱就降低 TCN/GRU baseline
- 不覆盖旧任务、旧结果和现有冻结证据

---

## 8. 工作区与 Git 规则

服务器正式项目：

`/home/dell/YrM/ICCT`

正式分支：

`qgnn`

当前工作区已经存在其他未提交 / 未跟踪结果。

Codex 必须：

- 开始先只读检查 branch / HEAD / status
- 不清理、不移动、不 stash、不 commit 其他 agent 的未提交内容
- 只提交本任务明确创建或修改的文件
- 大规模正式训练前先完成 smoke/preflight
- 正式长实验仍由用户启动
- 小型诊断和短 pilot 可由 Codex 直接运行
- 每个阶段 commit + push qgnn
- 报告中记录 commit SHA、配置、seed、结果路径

---

## 9. Gate A 最终报告必须回答的七个问题

1. 旧 2→2 为什么不足以支撑 interaction claim？
2. 新 IC4 membership 是否严格 history-only？
3. history source 是否 causal？若不是，论文如何定位？
4. Strong Graph 是否稳定优于 Strong Self？
5. Own-only capacity control 是否解释了 Graph gain？
6. interaction-critical strata 是否比 Full 更能体现 neighbor utility？
7. 是否有足够证据允许进入 Raj QGNN migration？

最终只能给出：

- PASS：允许进入 Gate B
- CONDITIONAL PASS：存在正信号，但需补指定验证
- FAIL：当前任务/模型未建立 Graph necessity

不得为了继续 QGNN 而默认 PASS。

---

## 10. 论文目标叙事冻结

本阶段目标论文叙事为：

> 短时普通轨迹预测中，大量轨迹主要由自车运动惯性解释，使全场景平均指标难以揭示多车交互价值。我们首先构建严格 history-only 的 interaction-critical 预测任务，并验证邻车信息相对强 self-only 与容量控制的边际效用。在此基础上，再比较经典图建模与 Raj Weighted Multi-J QGNN，以检验量子图表示是否能在相同信息与近似参数预算下进一步降低预测误差。LLM 时序模块作为独立问题，通过匹配输入与适配自由度的消融验证其预训练与参数高效适配价值。

该叙事在 Gate A/B/C 结果未全部完成前属于“目标叙事”，不是已证实结论。
