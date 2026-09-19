# GPT-6 Pro 第三轮开场 Prompt

你现在接手我的 ICCT 科研项目第三轮 QGNN 专项攻关。

仓库：
https://github.com/540-540-540/icct-research

正式分支：
qgnn

服务器正式目录：
/home/dell/YrM/ICCT

这是一个新对话。不要假设你知道此前聊天上下文，也不要只根据 GitHub 页面或模型记忆给方案。

你必须使用 Remote Desktop Commander 直接进入服务器工作。

开始后第一件事，请完整读取：

docs/qgnn/GPT6_PRO_ROUND3_ADAPTIVE_QUANTUM_TASK_20260919.md

然后严格按任务书中的阅读顺序继续读最新诊断、第二轮 freeze、代码与实验结果。
本轮不是重新探索整个 QGNN 空间。

第二轮已经得到一个真正承担多车交互的 RC-HQGNN：
连续 sensing history
→ 单车轻量时间编码
→ graph-conditioned ZZ/ZZZ 联合量子演化
→ K2/K3 与 relation-carrying readout
→ 每车连续 interaction representation
→ Token
→ 固定 GPT-2
→ future trajectory。

当前问题不是“要不要使用量子”。

所有场景都必须使用 QGNN，禁止设计 classical/quantum fallback switch。

第三轮唯一核心问题是：

**在不同 SinD 历史交互场景中，应该使用什么样的量子交互结构与强度。**

重点研究：
- 二体 ZZ 与三体 ZZZ 的 scene-conditioned 自适应；
- relation-level quantum coupling；
- quantum channel specialization；
- quantum round/depth 内部自适应；
- 如合理，研究 K2/K3 驱动的 quantum-to-quantum adaptive interaction。

不要退化成“强 classical GNN 先把关系推理做完，再把几个系数喂给量子层”。
你必须特别理解最新 history-only 诊断：

docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md

关键事实包括：

1. 当前 RC-HQGNN 全局仍输 matched classical：
   - ADE 约差 3.96%
   - FDE 约差 4.33%

2. 但高动态复杂场景里量子存在明确正信号。
   旧正式 closing>=15 子集中：
   - ADE 约 +4.29%
   - FDE 约 +6.35%

3. 冻结 QGNN 内部：
   - Spearman(K2, ΔJ)≈+0.112
   - Spearman(K3, ΔJ)≈+0.223

4. 在控制 30m 车辆密度后，高 K3 组相对低 K3 组的 ΔJ 仍稳定更高。
   五个密度层全部同方向，加权差约 +0.0505。

5. 简单 history-only 外部 classifier/router 的严格时间块 CV AUC≈0.50。
   因此不要设计“closing 超阈值就切模式”的死规则，也不要做经典/量子开关。

6. 当前量子计算之后的 output gate 与量子相对收益基本不相关，说明调节发生得太晚、粒度太粗。
项目硬约束继续保持：

- SinD 是正式主数据集。
- 3-BS controlled ISAC 不重做。
- 正式 history 来自 ISAC sensing，不使用 GT history。
- 20 history → 20 future。
- QGNN 输入连续物理/运动信息，不吃 Token。
- QGNN 必须承担核心跨车交互推理。
- QGNN 输出连续交互表示，再 Token 化。
- GPT-2 backbone 固定。
- classical / quantum 使用相同数据权限、GPT-2、Tokenizer、监督和合理训练预算。
- test split 本轮保持关闭。
- N=8 只是当前工程缓存接口，不是理论硬约束。
- 所有场景全程量子，不做 classical fallback。
- 训练成本必须现实，单个正式 run 理想 <=2h。

经典主对照必须继续合理、功能匹配。
如果量子新增 scene-adaptive controller，经典也必须获得公平的 history-conditioned adaptive 能力，不能靠额外 classical capacity 让量子“赢”。
执行方式：

1. 先完整审计任务书、freeze、新诊断和现有 prediction/qgnn_final 实现。
2. 最多提出 2–3 个真正针对当前失败的 adaptive quantum 候选。
3. 只选 1 个进入主 prototype，最多留 1 个备用。
4. 先做 forward/backward、梯度、mask、permutation、复杂度和 GPU 成本检查。
5. 先跑 0 dB / seed2026 / 4096 train / 1880 val / 12epoch paired pilot。
6. 只有量子/经典差距明显缩小、最好翻正，并且高动态量子优势保住，才进入 full 15802 train / 1880 val / 20epoch。
7. 0 dB overall 还没翻正之前，不扩展五 SNR × 多 seed。
8. 本轮最多允许 2 次有明确证据的一级结构修正，禁止无限搜索。

如果想让中间 K2/K3 调节下一轮量子演化，必须明确 simulator 与真实量子硬件的差异。
不能把 statevector 中“免费读取期望值”包装成真实硬件上的免费 coherent feedback。
长任务必须防卡死：

- >几分钟的训练/评估用独立后台进程；
- 写 job.json、heartbeat、summary、checkpoint；
- 记录 PID、命令、GPU、日志和输出目录；
- 短轮询进度；
- 支持 resume；
- 明显失败及时停止；
- 不要让一个训练或搜索把整个会话锁死。

除非遇到真正无法自主解决的阻塞，否则不要中途反复问我确认。

请自行做研究决策，并把每次关键取舍写入执行日志。

本轮最低验收：

**冻结一套 Scene-Adaptive RC-HQGNN，并完成真实 0 dB paired pilot。**

强验收：

**在全量 0 dB validation 上，adaptive QGNN 的 ADE 和 FDE 都开始优于 functional-matched adaptive classical，并保留/扩大高动态子集优势。**

最终把架构、实验、失败/成功证据沉淀到 docs/qgnn/ 与 reports/qgnn/，并提交 qgnn 分支。

现在开始。先使用 Remote Desktop Commander 进入 /home/dell/YrM/ICCT，完整读取第三轮任务书，然后直接推进。
