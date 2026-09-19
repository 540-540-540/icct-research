# 新对话开场 Prompt — ICCT Round4 Raj 主线续接

你现在接手我的 ICCT 科研项目 QGNN 主线。

项目正式服务器目录：

`/home/dell/YrM/ICCT`

正式分支：

`qgnn`

这是一个新对话。不要假设你知道此前聊天上下文，也不要只根据 GitHub 页面或模型记忆工作。

**你必须使用 Remote Desktop Commander 直接进入服务器。**

当前服务器状态比 GitHub 远端更新很多，因此：
- 以服务器 `/home/dell/YrM/ICCT` 为唯一 canonical source；
- 不要因为 `origin/qgnn` 落后而 reset / checkout / pull 覆盖服务器；
- 开始后先读取服务器真实 HEAD、git log 和 status。
开始后第一件事，请完整读取：

`docs/qgnn/ROUND4_RAJ_NEW_CHAT_HANDOFF_20260919.md`

随后严格按该文档“启动阅读顺序”继续读取最新 findings、two-seed JSON、baseline policy、prereg、代码和训练入口。

不要重新从 Round2 / Round3 开始探索。

当前主线已经发生关键变化：

**Raj paper-native route 已成为正式 QGNN 主线。**

当前主候选：

**Weighted multi-j Raj-style QGNN**
- j=2 + j=3 两条量子 subset 分支；
- pair/triplet subset identity 保留；
- SinD history-only physical relation 控制 graph-conditioned weighted Johnson quantum mixing；
- fixed-particle-number embedding evolution / compound-RBS dynamics；
- same Motion-Token / GPT-2 downstream。

当前主 baseline：

**multi-j JohnsonGIN**

Baseline policy 已冻结为：

**paper-native primary baseline first。**

GIN / PPGN 属于 paper-native 辅助 baselines。

ICCT 以前的 Adaptive Classical / Pairwise / Routed / Pair+Triplet 已降级为 secondary strong references，不再作为 Raj 路线的第一主门槛。
当前关键实验已经完成：

统一协议：
- SinD
- 0 dB
- 4096 train windows
- 1880 validation windows
- 20 epochs
- batch32
- same GPT-2 / Tokenizer / loss
- test closed

Seed2026：

Quantum：
- ADE 0.516457940
- FDE 1.102781152
- J 1.067848516

JohnsonGIN：
- ADE 0.525952385
- FDE 1.129781810
- J 1.090843291

Quantum gain：
- ADE +1.805%
- FDE +2.390%
- J +2.108%

Seed2027：

Quantum：
- ADE 0.530895042
- FDE 1.148233937
- J 1.105012010

JohnsonGIN：
- ADE 0.550422603
- FDE 1.189891210
- J 1.145368208

Quantum gain：
- ADE +3.548%
- FDE +3.501%
- J +3.523%

两 seed 平均优势约：
- ADE +2.696%
- FDE +2.960%
- J +2.833%

因此 Raj route 当前已经获得重复的 paper-native positive signal。
Seed2026 另外的 paper-native baselines：

PPGN-style：
- ADE 0.579594
- FDE 1.151496
- J 1.155342

GIN：
- ADE 0.590572
- FDE 1.191985
- J 1.186565

当前 Quantum 同时明显优于 GIN / PPGN。

但还没有：
- full 15,802 train-window paired confirmation；
- full-train 多 seed；
- five-SNR；
- test；
- 原始约5%量级稳定优势。

所以当前阶段已经不是“继续寻找 QGNN 架构”，而是：

**增强现有 Raj 主线的证据强度。**
你的下一项主任务：

## Full-train 0 dB paired confirmation

在真正启动前，先写一份 prereg：

`docs/qgnn/ROUND4_RAJ_FULLTRAIN_PREREG_20260919.md`

冻结：
- SinD
- SNR = 0 dB
- seed = 2026
- full train = 15,802
- val = 1,880
- epochs = 20
- batch = 32
- LR = 3e-4
- depth = 3
- correction cap = 16
- same GPT-2 / Tokenizer / loss / optimizer
- Quantum kind = `raj_weighted_multij_quantum`
- Classical kind = `raj_multij_johnson`
- test closed
- 两张 RTX4090 各跑一边
- 不修改架构 / LR / loss / Token / baseline strength

Primary gate：

**Quantum ADE 和 FDE 都低于 JohnsonGIN。**

Strong gate：
- 两项都赢；
- J relative gain >= 1%。

如果通过，再决定 full seed2027/2028 和 five-SNR。
如果任一指标不通过，不要立刻调参或换弱 baseline；先分析为什么 4096-train 的两 seed positive signal 没扩展到 full train。
长任务必须防卡死：

- 训练/评估用独立后台进程；
- 写 job.json；
- 记录 PID / GPU / exact command / log；
- heartbeat.json；
- stdout/stderr 持续写 run.log；
- 每25 steps 左右 heartbeat；
- 定期 checkpoint；
- 支持 resume；
- 短轮询；
- 不使用一个长 blocking 调用等待整个训练；
- OOM / deadlock / process exit 要明确区分；
- 明显失败及时停止；
- 不能让一个实验锁死整个会话。

服务器有 2×RTX4090 24GB，可并行：
- GPU0 Quantum
- GPU1 JohnsonGIN。
特别注意：

1. 当前正式主代码：
   `prediction/qgnn_paper_native/raj_paper.py`

2. 当前正式模型入口：
   `prediction/qgnn_paper_native/model.py`

3. 当前训练入口：
   `scripts/train_qgnn_paper_native.py`

4. `prediction/qgnn_paper_native/raj_fidelity_a.py`
   现在已提交，用于保留历史成功 run 的 source hash 和 Fidelity-A 研究记录；
   **不要把它误当成当前主模型。**

5. 当前服务器最新 HEAD 在你开始时必须自行核对。
   交接生成时最新为：
   `e924943`

6. GitHub origin 交接时仍落后在：
   `28cfaf0`
   所以绝不能从远端覆盖服务器。

7. test split 继续关闭。

8. 不重新启动 RC-HQGNN controller / K3 router / lambda2-lambda3 自适应搜索。

9. 如果 full-train Raj route 最终失效，再考虑 SQM-GNN U_MSG/U_UPD 作为下一替代路线；现在不要抢跑。
除非遇到真正无法自主解决的阻塞，否则不要中途反复问我确认。

请自行完成：
- 服务器审计；
- prereg；
- exact paired command；
- 后台启动；
- heartbeat 监控；
- 结果收集；
- paired ADE/FDE/J 分析；
- 与 two-seed 4096-train 结果对照；
- 写 Round4 full-train findings；
- 保存机器 JSON；
- 提交 qgnn 分支。

不要为了“量子赢”放宽预注册 gate。

现在开始。

第一步：用 Remote Desktop Commander 进入：

`/home/dell/YrM/ICCT`

完整读取：

`docs/qgnn/ROUND4_RAJ_NEW_CHAT_HANDOFF_20260919.md`

然后直接推进。
