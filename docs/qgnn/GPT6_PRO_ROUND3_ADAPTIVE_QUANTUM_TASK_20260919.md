# GPT-6 Pro 第三轮任务书：场景自适应量子交互专项攻关

更新时间：2026-09-19
正式分支：qgnn
服务器正式目录：/home/dell/YrM/ICCT
当前服务器 HEAD：62b7494（本任务书提交前）
硬件：2 × RTX 4090 24GB
Python：/home/dell/YrM/envs/ICCT/bin/python

## 1. 本轮任务定位

第二轮已经完成 RC-HQGNN v3 的架构、实现、工程验证和 0 dB 公平 pilot。
第三轮不重新探索整个 QGNN 空间，也不推翻“量子作为核心多车交互模块”的项目定义。

本轮只解决一个核心问题：

> QGNN 全程使用量子，但如何根据当前 SinD 历史交互状态，连续、自适应地改变量子内部的二体、三体、关系、通道和轮次交互模式，从而保留高动态复杂场景的量子优势，同时消除普通场景的整体退化。

最高目标仍然是：
QGNN + same GPT-2 在公平比较下真正优于功能匹配的经典 GNN + same GPT-2。

正式长期目标约为 5% 级 ADE/FDE 改善。
本轮先要求 0 dB 全量 validation 的量子结果至少翻正，再考虑多 seed 和五 SNR。
## 2. 必须使用 Remote Desktop Commander

本轮必须通过 Remote Desktop Commander 直接操作服务器：
/home/dell/YrM/ICCT

不能只做理论讨论。
必须读取当前代码、冻结文档、最新诊断、运行实际 prototype/pilot，并把有效结果沉淀到仓库。

任何长训练/评估必须后台运行，记录 PID、GPU、命令、日志、输出目录和 heartbeat；使用短轮询，不得用一个长阻塞调用一直等待。

## 3. 不可修改的项目硬约束

1. SinD 是正式主数据集；Automatum 只作为历史失败经验。
2. 正式 history 使用 ISAC sensing [x_hat,y_hat,vx_hat,vy_hat]，不能偷用 GT history。
3. 20 帧 history → 20 帧 future。
4. 3-BS controlled ISAC 上游本轮不重做。
5. QGNN 输入连续物理/运动信息，不吃离散 Token。
6. QGNN 必须承担核心跨车辆交互推理。
7. QGNN 输出连续交互表示，再 Token 化进入 GPT-2。
8. GPT-2 backbone 固定，不换其他 LLM。
9. classical / quantum 主比较使用相同数据权限、GPT-2、Tokenizer、监督、合理训练预算。
10. test split 本轮保持关闭。
11. N=8 只是当前正式缓存接口，不是理论硬约束。
12. 所有场景都使用量子 QGNN，不允许设计 classical/quantum fallback switch。
## 4. 第二轮当前事实：不能回避

最终统一接口，SinD 0 dB，seed2026，15802 train / 1880 val，20 epoch：

RC-HQGNN：
- ADE 0.51094460
- FDE 1.12605076
- J 1.07396998
- 墙钟 19.11 min

Adaptive Classical：
- ADE 0.49146503
- FDE 1.07928883
- J 1.03110945
- 墙钟 14.86 min

当前量子整体相对经典：
- ADE 更差约 3.96%
- FDE 更差约 4.33%

所以全局量子优势尚未建立。

但预先定义的高动态子集 closing pairs >=15（旧正式 evaluator 的 final-history 定义，140 个 validation 窗口）：
- 量子 ADE 相对经典改善约 +4.29%
- 量子 FDE 相对经典改善约 +6.35%

同时：
- 删除全部 ZZ/ZZZ：ADE 0.6234
- 删除 ZZZ：ADE 0.5235
- 清空 graph context：ADE 1.0649

这些只证明当前模型依赖量子交互路径，不证明量子优于经典。
## 5. 第三轮前新诊断：必须完整读取

必读：
docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md

紧凑机器结果：
reports/qgnn/history_only_quantum_mode_summary_20260919.json

服务器另有逐场景完整结果：
reports/qgnn/history_only_quantum_mode_diagnostic_20260919.json

诊断脚本：
scripts/diagnose_qgnn_history_modes.py

关键结果：

1. 当前逐场景 J 上，RC-HQGNN 约 44.2% 场景优于经典。
2. 逐场景 oracle 相对经典上限约：
   - ADE +5.31%
   - FDE +6.67%
   该 oracle 使用未来误差，只能诊断，绝不能进入模型。
3. 高动态、高阶交互越强，量子相对收益越好。
4. 新诊断中的 recent-5-frame closing>=15 子集为 115 个场景，量子 J 胜率约 68.7%。注意它与旧 evaluator 的 final-history 140 窗口定义不同，不能混写。
5. active wedges>=35：量子 J 胜率约 57.0%。
6. active triangles>=6：量子 J 胜率约 57.8%。
## 6. 最关键新证据：三体量子响应 K3

冻结 RC-HQGNN 内部信号与相对收益 ΔJ=J_classical-J_quantum：

- Spearman(K2, ΔJ) ≈ +0.112
- Spearman(K3, ΔJ) ≈ +0.223

K3 最高 20%：
- N=376
- 平均 ΔJ≈+0.0356
- 量子 J 胜率≈59.3%

为了排除“只是车辆更多/更密”的解释，按 30m 近车数量分 5 个密度层；每层内部比较 K3 高低两半。
五层的 ΔJ(high-K3)-ΔJ(low-K3) 全部为正：
+0.0161 / +0.0287 / +0.1107 / +0.0161 / +0.0793。

按层样本数加权约 +0.0505。
描述性分层 bootstrap 95% 区间约 [+0.0325,+0.0683]。

注意：validation 窗口高度重叠，因此该 bootstrap 只作结构稳定性诊断，不是正式独立样本显著性结论。

两个 SinD scene 分开看，K3 与 ΔJ 的 Spearman 均为正（约 +0.182 / +0.311）。
## 7. 另一个关键反证：不要做简单外部 router

用 history-only 物理特征预测“当前冻结量子是否赢经典”，做 5 个时间块交叉验证，并在 held-out 边界加 39-frame embargo：

physics-only：
- Logistic AUC≈0.504
- HGB AUC≈0.503
- ΔJ 回归 Spearman≈-0.008

physics + frozen quantum internal signals：
- Logistic AUC≈0.493
- HGB AUC≈0.501
- ΔJ 回归 Spearman≈+0.012

因此没有证据支持：
- “closing>=阈值就切换模式”的死规则；
- 外部 history classifier 决定 classical vs quantum；
- 离散 hard router 直接预测“量子会不会赢”。

本轮必须坚持：量子全程使用，内部进行连续、自适应交互。

## 8. 当前 post-readout gate 不够

当前 v3 gate 位于量子演化之后，只调最终 quantum residual。

新诊断：
- recent-5 closing<10：gate mean≈0.854
- recent-5 closing>=15：gate mean≈0.739
- Spearman(gate, ΔJ)≈-0.082
- Spearman(gate × readout RMS, ΔJ)≈+0.008

所以当前 gate 并没有实现有效的量子模式适配。
不要把下一轮主要工作继续放在“量子结果出来以后乘一个更复杂 gate”上。
## 9. 本轮优先研究的量子内部自适应机制

以下是高优先级设计空间，不是必须逐条采用。GPT-6 Pro 可以提出更好的同类机制，但必须满足硬约束。

### A. 量子阶数自适应

例如：

H_s^(l,c) =
H_node
+ λ2^(l,c)(s) H_ZZ
+ λ3^(l,c)(s) H_ZZZ

λ2、λ3 只由 history 可获得信息生成。

目标：
- 简单场景仍然运行量子，但偏 pair-like quantum interaction；
- 高动态联合场景加强 higher-order quantum interaction。

### B. 关系级量子耦合自适应

让场景上下文与局部物理关系共同决定：
- 哪些 ZZ_ij 应该更强；
- 哪些 ZZZ_ijk 应该更强。

可以在当前 beta_ij / gamma_ijk 基础上增加 scene-conditioned modulation，但不能让一个强 classical GNN 先完成主要关系推理。

### C. Quantum channel specialization

当前 4 个 channel 基本同构。
研究是否让 channel 学到不同的交互阶数、时间尺度、风险模式或频率响应，并用连续 history context 调节 channel contribution。
所有 channel 仍是量子线路，不是 classical expert。
### D. Quantum round / depth 内部自适应

当前 D=3。
优先考虑每轮不同的 λ2^(l)、λ3^(l)、channel mix 或 residual strength，而不是盲目增加 D。

### E. Quantum-to-quantum adaptive interaction

K3 与量子相对收益存在明确诊断信号。
可以研究前一轮量子相关性如何影响后一轮量子交互。

但必须严格处理物理实现解释：
- simulator 可以从保存的 statevector 直接算 K2/K3；
- 真实量子硬件读取中间期望通常需要额外线路/重复制备，并可能涉及测量坍缩；
- 除非设计 coherent control / ancilla 等真正量子机制，否则不得把“免费读取 K3 后继续同一线路”描述成硬件原生过程。

若该方案只作为 simulator-side differentiable architecture，也必须诚实写清成本和硬件迁移边界。

## 10. 经典计算不能抢走核心量子职责

允许：
- per-agent temporal encoder
- normalization
- 物理关系计算
- 轻量 scene summary/controller
- measurement 后 readout

不允许：
- classical GNN / attention 先做完整跨车推理，再生成 λ2/λ3 给量子层装饰；
- 外部 classical router 直接决定哪种预测器工作；
- 用未来误差/oracle 标签训练推理时不可获得的 selector。

如果使用 scene controller，必须证明它只是调制量子计算，而不是代替量子计算完成主要多车关系建模。
## 11. 公平经典对照

第三轮不能只拿第二轮旧 Adaptive Classical 当固定靶子，如果量子新增的 scene-adaptive controller 也给模型带来了额外经典容量。

至少需要：
1. Frozen Round-2 Adaptive Classical：保持历史连续性；
2. Functional-matched Adaptive Classical：允许同样的 history context、类似参数预算和 pair/triplet 自适应能力，但核心交互仍是经典计算。

若量子使用 K2/K3 等量子内部状态做后续自适应，经典不需要获得“不存在的量子 K3”，但应允许其使用功能对应的自身内部 relation summary / hidden-state feedback。
最终必须解释为什么比较公平。

核心变量仍然应尽可能收敛到：
Adaptive Classical Interaction vs Adaptive Quantum Interaction。

## 12. 训练成本硬约束

2 × RTX4090 24GB。

当前 C4/D3 RC-HQGNN 完整 0dB 单 run 约19.11min，资源预算充足，但不能无上限增加复杂度。

每个候选必须报告：
- qubit 数
- channels
- rounds/depth
- circuit calls
- 是否增加 intermediate measurement/evaluation
- 参数量
- B32 step time
- VRAM
- 单 epoch / 完整 run 实测或估计
- N=8 以及更大 N 时 scaling

正式单 run 理想<=2h；2–4h需说明；>4h原则上不作为主方案。
## 13. 第三轮实验流程

### Phase A：设计与机制审查
- 完整读取第二轮 freeze、round2 log、新 history-only 诊断和代码。
- 最多提出 2–3 个“场景自适应量子交互”候选。
- 明确每个候选解决哪条已有实验失败。
- 做理论、梯度、置换、mask、成本和硬件边界审查。
- 最终只选 1 个进入主 prototype，最多保留 1 个备用。

### Phase B：小规模 paired pilot
优先复用第二轮协议：
- SinD 0 dB
- seed2026
- 4096 train
- 1880 val
- 12 epoch
- same GPT-2 / Tokenizer / loss
- test closed

同时训练：
- 新 adaptive quantum
- functional-matched adaptive classical
- 必要时 frozen RC-HQGNN 作历史参考

重点同时看：
- overall ADE/FDE/J
- closing<10
- closing 10–14
- closing>=15
- high wedge / triangle
- K3 分层
候选进入 full-train 的优先条件：
- 相比 frozen RC-HQGNN，overall J 明显改善；
- 量子/经典全局差距显著缩小，最好已经翻正；
- 高动态场景的量子正优势不能被抹掉；
- 普通场景退化明显减轻；
- 不是靠额外 classical capacity 单独带来的收益。

### Phase C：全量 0 dB pilot
只有 Phase B 有正信号才运行：
- 15802 train
- 1880 val
- 20 epoch
- seed2026
- paired quantum/classical
- 相同训练曝光和调参机会
- test closed

本轮成功的最低模型目标：
新 adaptive QGNN 在 0 dB 全量 validation 上 ADE 和 FDE 均不差于 matched classical。

本轮强成功：
ADE 和 FDE 都稳定翻正，并出现接近 5% 的改善信号。

只有整体翻正后，才建议继续 seed2027/2028 与五 SNR。
## 14. 架构搜索与停止规则

本轮最多允许 2 次有明确证据的一级结构修正。
不要无限循环“再加一点 qubit / depth / channel”。

若两次结构修正后仍整体明显输经典：
- 如实记录；
- 判断当前 adaptive hypothesis 是否被证伪或只是优化不足；
- 给出下一步最小可证伪实验；
- 不允许削弱 baseline 或 cherry-pick 子集制造胜利。

可调但不视为一级改架构：
- LR
- modest hidden width
- loss weight
- regularization
- λ 初始化/温度
- 小范围 channel/depth 数值

不应反复推翻：
- QGNN 连续输入
- quantum interaction core 地位
- GPT-2 backbone
- QGNN→Token→GPT-2 主接口
- 全程量子原则

## 15. 长任务防卡死

沿用第二轮规范：
- >几分钟任务用独立进程；
- job.json / heartbeat / summary / checkpoint；
- stdout/stderr 写日志；
- 短轮询；
- 预算到时保存并可 resume；
- 及时杀掉明显失效实验；
- 不允许一个训练把整个会话锁死。
## 16. 本轮必须交付

P0 必须：
- 一套最终 Scene-Adaptive RC-HQGNN 结构定义；
- 为什么它针对当前失败而设计；
- 完整数学公式；
- controller / λ2 / λ3 / relation / channel / round 的精确定义；
- 经典计算与量子计算职责边界；
- matched classical comparator；
- 复杂度与训练成本；
- 量子硬件/模拟边界。

P1 必须：
- 可运行代码；
- forward/backward/gradient/mask/permutation sanity；
- step time / VRAM；
- 0 dB 小 pilot。

P2 强烈希望：
- full 15802 train / 1880 val / 20epoch paired pilot；
- overall + predefined subgroup + K3 strata 结果；
- 判断是否已翻正。

P3 超额：
- 在不超过两次一级结构修正的前提下，0 dB ADE/FDE 都出现接近 5% 的量子优势信号。

最终文档建议：
docs/qgnn/ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md

关键结果：
reports/qgnn/round3_*。
## 17. 启动阅读顺序

第一优先：
1. docs/qgnn/GPT6_PRO_ROUND3_ADAPTIVE_QUANTUM_TASK_20260919.md
2. docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md
3. reports/qgnn/history_only_quantum_mode_summary_20260919.json
4. docs/qgnn/FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md
5. docs/qgnn/ROUND2_EXECUTION_LOG_20260919.md

第二优先：
6. prediction/qgnn_final/
7. scripts/train_qgnn_final.py
8. scripts/diagnose_qgnn_history_modes.py
9. configs/qgnn_final_architecture.json
10. configs/qgnn_final_tokens.json

第三优先：
11. docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md
12. docs/dataset_migration/SIND_MIGRATION_HANDOFF.md

## 18. 最后一句话

第三轮不是“再造一个新的量子模型”。

第三轮要做的是：

> 保留已经成立的 RC-HQGNN 量子图核心，把当前固定、同质的高阶量子交互升级为真正面向 SinD 场景异质性的连续自适应量子交互，并用公平 matched classical 对照证明这种升级能否把高动态场景中的量子正信号转化成全局优势。
