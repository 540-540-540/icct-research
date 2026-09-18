# GPT-6 Pro 第二轮：最终 QGNN 架构冻结与原型推进任务书

更新时间：2026-09-19
正式分支：qgnn
仓库：https://github.com/540-540-540/icct-research
服务器正式目录：/home/dell/YrM/ICCT
Python：/home/dell/YrM/envs/ICCT/bin/python
硬件：2 × RTX 4090 24GB

这是一个全新 GPT-6 Pro 对话的主交接文件。
不要假设自己知道此前聊天上下文。开始后必须先使用 Remote Desktop Commander 进入服务器，完整阅读本文件及引用文档、代码和实验结果，再开始研究与执行。
本文件优先级高于旧文档中与当前决策冲突的 Automatum、N≤8、候选路线等旧假设。

## 1. 本轮最高目标

本轮不是“再列几个 QGNN 候选”，而是：
设计、论证、实现并尽可能验证一套最终 QGNN 主架构，使其具备在公平比较下真正赢经典 GNN 的最大现实可能性。

P0 必须完成：最终 QGNN 一级架构冻结，之后不再反复推翻“QGNN 到底怎么搭”。
P1 强烈希望：完整原型实现并跑通 forward/backward、梯度、显存和速度检查。
P2 最好完成：SinD 0 dB 上量子 vs 功能匹配经典的第一轮真实 pilot。
P3 超额完成：基于 pilot 做 1–2 轮有证据的结构迭代，并看到接近目标量级的量子优势。

最终最多保留 1 个主方案 + 1 个明确备用方案。

## 2. 强制执行方式

必须使用 Remote Desktop Commander 直接操作授权电脑/服务器。
主要目录：/home/dell/YrM/ICCT
禁止只基于 GitHub 页面、模型记忆或交接文档空谈。

建议执行链：
读仓库 → 读 SinD / ISAC / Q0 证据 → 审查现有实现 → Web/arXiv/GitHub 理论研究 → 最终 QGNN 设计 → 代码原型 → 工程验证 → 必要 pilot → 分析与有限迭代 → 冻结文档。

可以严格检索近 3 年 QGNN、equivariant quantum graph、higher-order quantum graph、quantum message passing、wireless/trajectory quantum graph 文献与代码。
外部方案必须最终落到本项目真实接口、经典对照和训练成本上。

## 3. Git / 目录规则

正式开发分支固定为 qgnn。
服务器唯一正式工程目录固定为 /home/dell/YrM/ICCT。
不要再长期创建 ICCT_qgnn、ICCT_sind* 等平行工程目录。
需要隔离实验可用临时 worktree，但结束前清理。
不要提交大型 checkpoint。
小型 JSON / summary / config / source / 诊断文档应沉淀到仓库。
不要使用 git clean 或激进 reset 误删用户已有实验结果。
不要把大量时间浪费在无关仓库治理和防御性工程。

## 4. 正式项目链路

SinD 真实车辆轨迹
→ 3 个固定 ISAC 基站
→ 多基站感知状态估计
→ QGNN 多车辆交互推理
→ 连续交互表示 Token 化
→ GPT-2
→ 未来车辆轨迹。

上游 ISAC 目标只是“实验可信、稳定、不会成为下游主要瓶颈”，不是本轮创新重点。

正式预测任务：
- history：20 帧
- future：20 帧
- 状态：[x,y,vx,vy]
- 正式 history：ISAC sensing 的 [x_hat,y_hat,vx_hat,vy_hat]
- future supervision：真实未来轨迹
- 采样间隔约 0.1001001001 s
- 3 个固定 BS
- SNR：[-10,-5,0,5,10] dB

禁止正式模型用 GT history 偷换 sensing history。

## 5. 数据集冻结

SinD 是唯一正式主数据集。
Automatum T-Crossing 已冻结为历史探索，不再作为正式数据集继续开发。
完整数据交接：docs/dataset_migration/SIND_MIGRATION_HANDOFF.md当前 split：
- train 15,802
- val 1,880
- test 2,086
- total 19,768

当前工程接口：
- history_state [20,8,4]
- future_state [20,8,4]
- vehicle_mask [8]
- history_timestamp [20]

N=8 不是科研硬约束，只是当前工程预处理和训练成本控制下的 model-facing 设置。
约 90.73% 最终样本来自原始 N>8 窗口再选成局部 8 车图。

GPT-6 Pro 可以重新研究动态 N、更多车辆、经典轻量筛选、物理筛选、soft weighting、sparse graph、hierarchical local/global、QGNN 自己学习重要性或 classical+quantum hybrid routing。
但必须控制训练成本和 circuit-call scaling。

SinD 高交互统计：
- N>=4 99.94%，N>=5 99.25%，N>=6 98.16%
- 10m / 20m / 30m 内车辆对：5.726 / 12.230 / 18.779 每窗口
- closing rate >0.5m/s：15.141 pairs/window
- DCPA<=5m：74.52%；DCPA<=10m：93.56%
- active neighbors >=2 / >=3 / >=4：60.22% / 38.83% / 23.96%
- active component >=3 / >=4 / >=5 vehicles：100% / 93.83% / 87.12%

完整统计：reports/sind/sind_interaction_audit.json

## 6. 3-BS ISAC 当前状态

SinD 已完成 3-BS controlled sensing 适配。
held-out test sensing RMSE：
- -10 dB：position 1.175 m，velocity 1.176 m/s
- -5 dB：0.660 m，0.661 m/s
- 0 dB：0.371 m，0.372 m/s
- 5 dB：0.208 m，0.209 m/s
- 10 dB：0.117 m，0.118 m/s

当前误差量级足以用于下游，不要重新把主要精力转向 ISAC。

关键文件：
configs/sind_controlled_isac.json
configs/sind_prediction.json
frontend/sind_prediction_dataset.py
frontend/controlled_isac/sind_frontend.py

## 7. 最新经典交互预诊断：必须先读

优先阅读：docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md
机器可读：reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json

协议：SinD、0 dB、seed 2026、20 epoch、train/validation only、test 未使用、同一 Motion-Token GPT-2，仅经典交互模块不同。四个方案全部是经典模型，不是量子模型。

最佳 validation：
- 不看其他车：ADE 0.524150，FDE 1.063804，J 1.056052
- 普通两车：ADE 0.502264，FDE 1.030864，J 1.017696
- 自适应两车：ADE 0.501781，FDE 1.020530，J 1.012046
- 两车+三车联合：ADE 0.498218，FDE 1.004633，J 1.000534

三车联合相对不看其他车：ADE +4.95%，FDE +5.56%。
三车联合相对普通两车：ADE +0.81%，FDE +2.55%。

在至少 15 对“30m 内且 closing rate >0.5m/s”的 140 个高动态交互场景：
- 三车联合相对不看其他车：ADE +8.66%，FDE +10.22%
- 三车联合相对普通两车：ADE +4.33%，FDE +6.31%

关键解释：
三车联合价值主要来自动态多车接近/冲突，而不是单纯车辆数多。

逐场景 oracle 若从普通两车 / 自适应两车 / 三车联合中选最优，相对不看其他车：
- ADE +13.22%
- FDE +15.27%

这说明不同场景需要不同交互机制，仍存在显著自适应建模空间。

## 8. Automatum 历史结论的地位

Automatum 曾发现图收益较弱、动态路由有限、显式三车关系弱、逐场景 oracle 仍有空间、旧简化 GPT2 接口较弱。
这些现在只能作为历史失败经验和风险提示，不能约束 SinD。

尤其“higher-order 价值弱”已被 SinD 结果推翻。

参考：
docs/qgnn/Q0_LLM_AND_TASK_SPACE_DIAGNOSTIC_20260918.md
docs/qgnn/Q0_DUAL_DIAGNOSTIC_FINDINGS_20260918.md
docs/qgnn/G2_HIGHER_ORDER_DIAGNOSTIC_20260918.md

## 9. QGNN 与 GPT-2 职责：硬冻结

QGNN 输入必须是连续物理/运动信息，而不是离散 Token。
可使用 x,y,vx,vy、相对位置、相对速度、distance、closing rate、TCPA/DCPA、轻量时间编码等仅由 history 计算的连续特征。

QGNN 必须真正承担核心多车交互推理。
不接受 Classical GNN 已做完主要图推理，再接一个 VQC 小修正然后称为 QGNN。
允许 normalization、physics features、light encoder 等经典 preprocessing。
但 node / edge / subset / graph structure 必须直接控制核心 quantum encoding、interaction、message、aggregation、higher-order evolution 或 global quantum dynamics。

QGNN 输出应是连续交互表示，再经过 Token 化送入 GPT-2。QGNN 最终输出形式必须冻结并写清数学定义和 tensor shape，例如：
- per-agent h_i^Q
- pair relation r_ij^Q
- higher-order subset representation
- global / multi-scale quantum interaction representation

## 10. GPT-2 backbone：硬冻结

正式 LLM backbone 固定使用 GPT-2。
不要换成 Llama、Qwen、GRU 或其他 backbone。

概念链路：
Historical Motion Tokens
+ QGNN Interaction Tokens
+ Future Query / Control Tokens
→ GPT-2
→ 未来 20 帧轨迹。

可以重新设计 Motion Token、Interaction Token、QGNN→Token projection/quantization、Token ordering、agent/relation/time embedding、future query token、LoRA/freeze policy 和 loss。
但 GPT-2 backbone 本身固定。

正式 classical / quantum 主比较应尽量使用同一 GPT-2 和同一 Token 接口原则。

## 11. 正式 Token 设计仍开放

当前 0 dB 预实验只用了临时 SinD Token 范围，不能直接作为五档 SNR 正式词表。

train-only 五档统计已经确认：
- 高 SNR 正常运动分布较窄
- -10 dB sensing history 有明显长尾
- 0 dB 范围直接用于 -10 dB 会大量边界截断
- 简单把统一 31×31 均匀范围拉宽又会损失正常运动分辨率

正式设计需同时解决“正常运动高分辨率 + 低 SNR 长尾鲁棒性”。

可以探索 non-uniform bins、中央精细/尾部粗粒度、hierarchical tokens、learnable codebook、noise-aware tokenization、continuous embedding + discrete semantic token hybrid 等。

## 12. 车辆重要性 / 动态 N

当车辆多时必须解决哪些车辆和关系值得重点计算，但不冻结具体实现。
可以经典轻量 scorer、物理粗筛、QGNN importance、soft weighting、hybrid routing、sparse graph、hierarchical interaction。

要求：
- circuit calls 不随 N、|E|、history length 无控制爆炸
- 不应提前丢掉真正关键的高阶冲突车辆
- 必须给出复杂度随 N 的公式
- 讨论 N=8、12、16、20 的可行性

## 13. 训练成本：一级硬约束

硬件：2 × RTX 4090 24GB。
正式 Python：/home/dell/YrM/envs/ICCT/bin/python
实现前直接检查当前 PennyLane / lightning-gpu / cuStateVec 等实际版本，不要只信旧文档。

对一个正式模型 / SNR / seed：
- 理想 ≤2h
- 2–4h 可接受
- >4h 必须有充分的性能或理论理由
- 约 12h 基本不可接受

2×4090 可用于实验并行，但不能用“无限并行”掩盖单模型不可扩展。

冻结最终 QGNN 前必须明确：
1. qubit 数 Q(N)
2. circuit depth
3. 每 sample circuit calls
4. 每 batch circuit calls
5. 随 N / E / subset 数的 scaling
6. quantum trainable params
7. classical params
8. batch size 约束
9. VRAM 实测
10. forward/backward step time
11. 单 epoch 和完整训练时间估计

尤其避免 O(T|E|) 或更高数量级的昂贵 circuit calls，除非真实 GPU 实测证明仍在预算内。优先考虑一次/少数 quantum evolution 处理多个关系、batched simulation、shared circuit、可控 subset 数、sparse quantum interaction、cheap classical routing + expensive quantum reasoning。

理论漂亮但在 2×4090 上一天跑不完的方案不能作为最终主架构。

## 14. 长任务防卡死 / 防阻塞执行规范：硬约束

此前已出现过训练、评估或其他长任务占用工具调用过久，看起来“卡死”的情况。本轮必须主动规避。

任何预计超过几分钟的训练、评估、数据处理或搜索任务：
- 不要用一个超长阻塞调用一直等待
- 后台启动或独立进程执行
- 明确记录 PID、完整命令、GPU、输出目录、日志路径
- stdout/stderr 持续写入日志
- 每个 epoch 或关键阶段写可机器读取的 summary / checkpoint
- 用短轮询检查进度，而不是单次调用等到结束
- 设置合理单次工具调用 timeout
- 长任务应能判断“正常慢 / 无输出 / 死锁 / OOM / 进程已退出”
- 训练脚本尽可能支持从 checkpoint 续跑
- 长研究任务分阶段保存结论，不要等最后才写文档
- 一旦某条实验明显失效，及时停掉，不要空耗数小时

不要让单个实验、网页研究或远程命令把整轮 GPT-6 Pro 会话锁死。

## 15. 正式量子 vs 经典比较原则

项目目标明确：量子要赢经典。
但比较必须公平。

最终至少有一个合理、训练充分、功能匹配的经典主对照。
保持相同 SinD/ISAC 输入、history/future、GPT-2、Token 接口原则、监督和合理接近的训练预算。
如果车辆筛选不是量子创新本身，也应尽量保持相同筛选规则。

核心变量：
Classical interaction core vs Quantum interaction core。

如果最终 QGNN 主机制是高阶多车联合，就不能只拿普通 pairwise MPNN 当唯一主 baseline。
至少需要一个合理 classical higher-order / adaptive counterpart。

不要求把整个轨迹预测领域所有最新 SOTA 都塞进核心 baseline。
原则是“合理、功能匹配、不故意弱化”。

## 16. 性能目标

希望 QGNN + same GPT-2 相对 functional-matched Classical GNN + same GPT-2 达到约 5% 级 ADE/FDE 改善。
优先希望在高动态、多车联合冲突、高 active-neighbor、交互模式异质场景出现更明显优势，再尽可能转化为全局优势。如果第一版 QGNN 没赢：
1. 不要立即放弃
2. 定位是表达力、优化、接口、噪声、circuit bottleneck 还是经典前处理吃掉了量子空间
3. 允许 1–2 次有理论/实验依据的结构迭代
4. 不允许通过削弱 baseline“解决”
5. 如果最终仍无法赢，要如实记录，但仍需冻结当前最合理主架构并解释失败条件

## 17. “真正 QGNN”边界

不接受：
- Classical GNN → PQC → GPT-2
- MPNN / attention 已完成主要交互后用 PQC 压缩
- 单纯把 Linear / MLP 换成 VQC
- 与 graph structure 无关的 quantum layer
- 量子层仅作 embedding decoration

优先研究但不强制：
- higher-order subset quantum interaction
- physics-conditioned quantum graph evolution
- quantum-native joint multi-vehicle state
- permutation-aware / equivariant quantum graph computation
- dynamic quantum interaction routing
- graph-conditioned global quantum dynamics
- 能统一表达 pairwise + higher-order + scene-dependent interaction mode 的其他量子机制

关键问题：量子计算到底承担了哪个经典图核心本来承担的关键推理职责？

## 18. 本轮必须产出的最终架构内容

最终冻结文档至少写清：
1. 输入张量与全部特征定义
2. 车辆/关系选择机制
3. 节点、边、高阶关系的数学表示
4. qubit 数和每个 qubit 的语义
5. data encoding
6. graph-conditioned quantum interaction
7. circuit ansatz、depth、trainable params
8. measurement / readout
9. QGNN 输出 tensor
10. QGNN→Interaction Token
11. Motion Token + Interaction Token + GPT-2 序列
12. GPT-2 训练/LoRA/freeze 策略
13. loss
14. classical matched baseline
15. 参数量与复杂度
16. circuit-call scaling
17. 2×4090 训练预算
18. permutation / vehicle-order 处理
19. 动态 N 处理
20. 多 SNR / sensing noise 处理
21. ablation
22. 失败条件和备用方案

数学必须完整到可以直接按文档编码，而不是停在“用纠缠捕获高阶关系”这种概念描述。

## 19. 研究方法要求

必须搜索和引用真实、可核查的近年论文/代码。
严格区分：
- 已有论文证明的理论
- 从论文迁移到本项目的合理推断
- 本项目自创设计
- 实验假设

不要把 quantum expressivity 的一般性结果直接包装成“本项目 ADE 一定提升”。
不要把经典 higher-order improvement 当成 quantum advantage。

要主动寻找能够反驳最终量子设计的强经典替代，并检查量子机制是否仍有独立价值。

## 20. 启动后建议阅读顺序

第一优先：
- 本任务书
- docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md
- reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json
- docs/dataset_migration/SIND_MIGRATION_HANDOFF.md

第二优先：
- prediction/q0/
- scripts/train_q0_motion_llm.py
- scripts/evaluate_q0_motion_llm_val.py
- frontend/sind_prediction_dataset.py
- configs/sind_prediction.json

第三优先（历史风险）：
- docs/qgnn/QGNN_DESIGN_CONTRACT_AND_PRO_HANDOFF.md
- docs/qgnn/Q0_LLM_AND_TASK_SPACE_DIAGNOSTIC_20260918.md
- docs/qgnn/Q0_DUAL_DIAGNOSTIC_FINDINGS_20260918.md
- docs/qgnn/G2_HIGHER_ORDER_DIAGNOSTIC_20260918.md

## 21. 最终交付与验收

P0 必须：
- 一个明确最终 QGNN 主架构
- 完整数学定义
- 输入 / 输出 / 量子线路 / 经典辅助 / GPT-2 接口全部冻结
- 复杂度与训练成本可行
- matched classical baseline 明确
- 不再留下“还需下一轮决定 QGNN 怎么搭”的一级问题

P1 强烈希望：
- 实际代码原型
- forward / backward 可跑
- 梯度正常
- 输出 shape 正确
- 显存与 step time 实测
- permutation / vehicle-order 基本 sanity check
- 小规模训练 loss 能下降

P2 最好：
- SinD 0 dB quantum vs classical pilot
- 只用 train/val，不提前消耗 test
- 同时分析整体和高交互子集

P3 超额：
- 根据 pilot 做 1–2 轮有证据的改进
- 已观察到接近约 5% 的量子优势信号

最终在 docs/qgnn/ 下写一份 FINAL_QGNN_ARCHITECTURE_FREEZE 文档，并把关键小型实验结果写入 reports/qgnn/ 或等价目录。

## 22. 最后提醒

本轮最重要的不是“做出一个量子模型”，而是：

让 QGNN 的结构直接针对 SinD 已经暴露出的“高阶动态交互 + 场景异质性”问题，
同时保持 GPT-2 固定、量子 vs 经典公平、训练成本现实，
最终冻结一套最有希望真正赢经典 GNN 的 QGNN。

如果正式架构和现有临时经典探路模块不同，这是允许的。
现有 Pairwise / Routed / Pair+Triplet 的作用是提供任务证据和经典参考，不是把最终 QGNN 限死在这些结构里。

请把“性能潜力、理论合理性、量子真实性、工程可行性、训练成本”同时作为主方案选择标准。
不要因为某条路线理论新颖就忽略 2×4090 上的实际可运行性，也不要因为当前代码方便就牺牲最终论文核心创新。

本轮结束时，应当让下一阶段主要变成“正式训练和实验”，而不是继续争论“QGNN 到底怎么搭”。