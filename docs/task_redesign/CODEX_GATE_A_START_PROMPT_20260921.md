你现在接手 ICCT 项目的“论文主线重构：Task / Graph Necessity Gate A”专项施工。

服务器正式项目：

/home/dell/YrM/ICCT

正式分支：

qgnn

第一件事，完整阅读：

docs/task_redesign/ICCT_PAPER_NARRATIVE_REBUILD_TASK_20260921.md

然后按其中要求工作。

==================================================
一、当前背景
==================================================

当前 Raj Weighted Multi-J QGNN 已经在旧 2s→2s legacy benchmark 上获得较稳定的 matched-classical 优势：

- 两 seed frozen test
- ADE 提升约 3.006%
- FDE 提升约 3.307%
- J 提升约 3.161%
- graph 参数量差约 0.67%

冻结文件：

docs/qgnn/RAJ_QGNN_MAIN_SOLUTION_FREEZE_20260921.md

因此本轮不要继续调 Raj QGNN。

当前论文真正缺失的是：

Self
→ Classical Graph
→ Raj QGNN

中第一段“为什么需要 Graph / 多车交互”的可靠证据。

旧 2s→2s self-only 任务过于容易：

TCN 0.376311 / 0.816700
LSTM 0.377409 / 0.822467
Transformer 0.388013 / 0.829766
GPT-2+LoRA 0.604277 / 1.313128

因此论文不能写“单车预测做不好，所以需要交互”。

正确研究问题是：

大量短时平滑轨迹可由自车惯性解释，导致全场景平均 ADE/FDE 掩盖多车交互的边际价值。我们需要构造严格 history-only 的 interaction-critical 任务，并验证 Graph 是否真的优于 strong Self。

==================================================
二、你本轮只做 Gate A
==================================================

只做：

Task / Interaction Necessity

不要开始：

- Raj QGNN migration
- QGNN 新训练
- GPT-2 大规模修复
- PennyLane 化
- 新量子架构探索

本轮顺序固定：

AUDIT
→ DATA CONTRACT
→ PREFLIGHT
→ SMALL PILOT
→ DIAGNOSIS
→ SECOND-SEED CONFIRMATION
→ GATE DECISION

==================================================
三、工作区安全规则
==================================================

正式工作区当前已经存在其他 agent / 历史任务留下的未提交和未跟踪文件。

开始后先执行并记录：

cd /home/dell/YrM/ICCT
git branch --show-current
git rev-parse HEAD
git status --short

禁止：

- git clean
- git reset --hard
- stash 其他人的修改
- 删除或移动其他 agent 的未提交结果
- 把无关未提交文件纳入你的 commit

只允许提交你本轮明确创建或修改的文件。

服务器工作区优先于旧聊天上下文。

==================================================
四、必须优先阅读
==================================================

除任务书外，至少完整读取：

1. docs/task_redesign/CODEX_TRAJECTORY_TASK_REDESIGN_FREEZE_20260920.md
2. docs/task_redesign/GPT6PRO_OPTIMAL_TRAJECTORY_TASK_SELECTION_20260920.md
3. docs/task_redesign/GPT6PRO_REVIEW_AND_NEXT_PILOT_20260920.md
4. docs/qgnn/RAJ_QGNN_MAIN_SOLUTION_FREEZE_20260921.md
5. docs/qgnn/GPT2_SELF_DIAGNOSTIC_PROTOCOL_20260920.md
6. reports/q0/AUTOMATUM_QGNN_TASK_SPACE_AUDIT_0DB.md
7. reports/q0/LLM_MARGINAL_EFFECT_AUDIT_0DB.md
8. reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json
9. 当前 q0 相关 summary/config/training/val_diagnostics
10. task_redesign 相关 JSON/CSV 审计产物

不要只看 markdown 结论，必须核实际 config、代码和结果。

==================================================
五、第一阶段：独立审计
==================================================

先回答：

1. 当前 q0 pilot 使用的到底是什么 dataset / task / horizon / split？
2. Automatum 上 Graph 为什么出现 Graph < Self？
3. SinD 上 Pair/Triplet 为什么出现约 4–5% 正向？
4. 两批结果是否可直接比较？如果不能，差异在哪里？
5. 当前 IC4 候选的数据管线是否已经真正实现？
6. history source 的 RTS smoothing / causal status 到底是什么？
7. 当前是否存在 future information 进入 history input 的风险？

输出一份新的 Gate A audit。

不要先改模型。

==================================================
六、第二阶段：IC4 数据合同
==================================================

目标：

SinD-IC4
history ≈ 2s
future ≈ 4s

要求：

- target-centered
- history-only membership
- target slot 固定
- Self 只看 target history
- Graph 看 target + neighbors
- Full2 / IC2 / Full4 / IC4
- IC membership 在 2s/4s horizon 间保持同一 history-only 定义
- future 缺失只产生 future_mask
- 不允许 future availability 反向影响 membership
- 不允许用 future maneuver / future error / model performance 选样本

必须生成可复现的数据统计和 contract 验证。

如果当前 history source 不能证明 causal：

不要阻塞所有诊断，但必须在报告中明确：

offline latent-state simulation

不能写 strict online forecasting。

==================================================
七、第三阶段：Graph-Necessity 最小模型组
==================================================

首轮不要用 GPT-2。

统一 temporal backbone 首选 TCN。

至少实现：

A. Strong Self
B. Own-only Capacity Control
C. Context Pooling Control
D. Strong Classical Graph
E. All-neighbor Strong Graph

要求：

- temporal encoder 尽量共享
- decoder 相同
- loss 相同
- coordinate system 相同
- 同一数据
- 同一选点协议
- 同一 horizon
- 只让 neighbor information / interaction mechanism 成为主要差异

不得因为 QGNN 最多 N=8 就限制 E。

==================================================
八、Pilot 设计
==================================================

先 seed 2026。

四个 evaluation cells：

- Full2
- IC2
- Full4
- IC4

如果 seed2026 没有可靠正信号：

先诊断，不要直接跑大量 seeds。

如果有正信号：

再跑 seed2027 复核。

核心指标：

- ADE
- FDE
- J = ADE + 0.5 FDE
- absolute gain
- relative gain
- Graph vs Own
- Graph vs Pool
- Full vs IC
- 2s vs 4s
- k / closing / CPA strata

==================================================
九、Gate A 判断
==================================================

目标证据：

Graph > Self

而且优先希望：

GraphGain(IC4) > GraphGain(Full4)

以及：

GraphGain(4s) > GraphGain(2s)

但第二个 horizon 条件如果不成立，不得硬解释。

必须检查：

- Own-only 是否解释主要收益
- Context Pooling 是否已经足够
- All-neighbor 是否显著优于 N<=8
- Graph gain 是否跨 seed 同向

最终只能输出：

PASS
CONDITIONAL PASS
FAIL

PASS 才允许下一轮 Raj QGNN migration。

==================================================
十、科研约束
==================================================

严禁：

- 为了 Graph/QGNN 赢而改筛选阈值
- cherry-pick seed
- 削弱 baseline
- test-set tuning
- 把 validation 当 test
- 把相关性写成交通因果
- 把 Graph gain 写成 GNN 不可替代
- 把 state-vector simulation 写成真实量子硬件
- 修改冻结 Raj 结果来迎合新叙事

实验失败也是有效科研结论。

==================================================
十一、施工与提交
==================================================

短测试、小型 pilot 你可以直接运行。

正式长时间训练：

准备到只差用户一条命令，不要擅自启动大规模正式 sweep。

每一阶段：

- unit test / smoke test / preflight
- 生成报告
- git diff 自查
- 只 commit 本任务文件
- push origin qgnn
- 报告 commit SHA

最终请交付：

1. Gate A audit
2. source-causality audit
3. IC4 data contract
4. model implementation
5. tests / preflight
6. seed2026 pilot
7. seed2027 confirmation（仅在有正信号时）
8. Gate A final report
9. 是否允许进入 Raj migration 的明确决定
10. 如需用户启动正式实验，只给一条最终启动命令

现在开始。先不要改代码，先完成工作区检查和完整审计。
