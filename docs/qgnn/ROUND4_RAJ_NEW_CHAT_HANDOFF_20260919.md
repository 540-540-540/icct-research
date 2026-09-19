# ICCT QGNN Round4 新对话完整交接 — Raj Paper-Native 主线

日期：2026-09-19
用途：新 ChatGPT 对话无缝接手当前 ICCT QGNN 主线，不依赖旧聊天上下文。

## 1. 当前最高结论

Round2 / Round3 的自研 RC-HQGNN 与 Scene-Adaptive RC-HQGNN 没有建立全局量子优势。

Round4 已切换为 **paper-native first** 策略，并以 Raj et al. 2026 的 higher-order subset / message-passing QGNN 思路为主线。

当前已冻结的最强候选：

**Weighted multi-j Raj-style QGNN**
- 所有场景同时运行 j=2 与 j=3 两条量子 subset 分支；
- j=2 保留 pair-subset identity；
- j=3 保留 triplet-subset identity；
- SinD history-only 连续物理关系直接控制 weighted Johnson quantum mixing；
- 各分支使用固定粒子数 embedding register 与 compound/RBS evolution；
- 每车融合 j=2/j=3 的量子表示；
- 输出连续 interaction representation；
- 后接项目统一 Motion-Token / GPT-2；
- graph module trainable params = 126,535。

当前主 baseline：

**multi-j JohnsonGIN**
- 相同 j=2+j=3 subset lift；
- 相同 downstream GPT-2 / Tokenizer / loss / data protocol；
- graph params = 179,201；
- 不故意减弱经典模型。

当前量子模型在两个独立 seed 上都已同时赢主 baseline 的 ADE 与 FDE。
## 2. 服务器与 Git 状态

正式服务器目录：

`/home/dell/YrM/ICCT`

正式分支：

`qgnn`

当前服务器最新 HEAD（交接时）：

`e924943 Preserve_Round4_FidelityA_source_for_reproducibility`

前一核心结果 commit：

`fcce4d4 Freeze_Round4_Raj_paper_native_two_seed_results`

重要：GitHub `origin/qgnn` 当前仍明显落后，交接时停在：

`28cfaf0`

因此：

**服务器 /home/dell/YrM/ICCT 是当前唯一 canonical source。**

新对话不能因为 GitHub 远端较旧而 reset / checkout / pull 覆盖服务器状态。

必须使用 Remote Desktop Commander 直接读取服务器。

当前没有正在运行的 Round4 训练任务。

大量 `reports/qgnn/` 历史 run 目录仍为 untracked，包含 checkpoints、逐窗口结果、日志与 heartbeat。不要做全仓库 clean，不要删除它们。
## 3. 项目固定链路与硬约束

正式链路：

SinD real trajectories
→ 3-BS controlled ISAC
→ sensing history [x_hat,y_hat,vx_hat,vy_hat]
→ QGNN multi-vehicle interaction
→ continuous interaction representation
→ Motion / Interaction Token interface
→ fixed GPT-2 backbone
→ 20-frame future trajectory.

硬约束：
- SinD 是正式主数据集；
- 20 history → 20 future；
- 正式 history 使用 ISAC sensing，不偷用 GT history；
- 0 dB 当前用于 QGNN 主线筛选；
- test split 继续关闭，直到正式协议明确授权；
- QGNN 输入连续物理/运动信息，不吃离散 Token；
- QGNN 必须承担核心多车交互；
- GPT-2 backbone 固定；
- QGNN 输出连续 representation 后再进入 Token/GPT-2；
- N=8 是当前缓存/工程接口，不是理论研究硬约束；
- 所有正式 quantum/classical 对比保持同一 downstream、训练曝光与数据权限；
- 不允许靠削弱 baseline 或 cherry-pick 子集制造量子优势。
## 4. Baseline policy 已冻结：Paper-native first

正式政策：

`docs/qgnn/ROUND4_BASELINE_POLICY_20260919.md`

如果采用某篇论文路线，主 baseline 优先采用论文原生 classical comparator family。

对 Raj 路线：

### Tier A — 主 baseline
1. multi-j JohnsonGIN — 最重要 matched baseline；
2. GIN — original graph / 1-WL baseline；
3. PPGN-style higher-order / 3-WL reference。

### Tier B — quantum baselines / ablations
- RC-HQGNN；
- Scene-Adaptive RC-HQGNN；
- Edge-Local QGCN / Skolik EQC 等，如后续需要。

### Tier C — ICCT historical strong classical references
- Adaptive Classical Higher-Order；
- Pairwise / Routed / Pair+Triplet。

Tier C 仍要诚实报告，但已经降为 secondary strong reference，不再是 Raj 路线的第一道准入门槛。
## 5. 当前两 seed 关键结果

完整 findings：

`docs/qgnn/ROUND4_RAJ_PAPER_NATIVE_FINDINGS_20260919.md`

机器结果：

`reports/qgnn/ROUND4_RAJ_TWO_SEED_RESULTS_20260919.json`

统一协议：
- SinD
- SNR = 0 dB
- train = 4096 windows / seed
- val = all 1880
- epoch = 20
- batch = 32
- correction cap = 16
- same GPT-2 / Tokenizer / loss
- test closed

### Seed 2026

Weighted multi-j Quantum:
- ADE = 0.516457940
- FDE = 1.102781152
- J = 1.067848516
- best epoch = 19

multi-j JohnsonGIN:
- ADE = 0.525952385
- FDE = 1.129781810
- J = 1.090843291
- best epoch = 20

Quantum relative gain:
- ADE +1.805%
- FDE +2.390%
- J +2.108%

### Seed 2027

Weighted multi-j Quantum:
- ADE = 0.530895042
- FDE = 1.148233937
- J = 1.105012010
- best epoch = 20

multi-j JohnsonGIN:
- ADE = 0.550422603
- FDE = 1.189891210
- J = 1.145368208
- best epoch = 20

Quantum relative gain:
- ADE +3.548%
- FDE +3.501%
- J +3.523%

### Two-seed mean

Quantum:
- ADE 0.523676491
- FDE 1.125507544
- J 1.086430263

JohnsonGIN:
- ADE 0.538187494
- FDE 1.159836510
- J 1.118105749

two-seed mean relative gain:
- ADE +2.696%
- FDE +2.960%
- J +2.833%

这已经是重复的 paper-native positive signal，但仍未达到最初希望的约5%量级。
## 6. 其他 paper-native baseline 结果

Seed2026 / 4096 train / 1880 val / 20 epoch：

Weighted multi-j Quantum：
- ADE 0.516458
- FDE 1.102781
- J 1.067849

PPGN-style higher-order：
- ADE 0.579594
- FDE 1.151496
- J 1.155342

GIN：
- ADE 0.590572
- FDE 1.191985
- J 1.186565

Quantum vs PPGN：
- ADE +10.893%
- FDE +4.231%
- J +7.573%

Quantum vs GIN：
- ADE +12.550%
- FDE +7.484%
- J +10.005%

因此当前最难、最重要的 paper-native baseline 是 multi-j JohnsonGIN。
## 7. 当前主模型到底是什么

主要代码：

`prediction/qgnn_paper_native/raj_paper.py`

入口：

`prediction/qgnn_paper_native/model.py`

训练：

`scripts/train_qgnn_paper_native.py`

当前 accepted quantum kind：

`raj_weighted_multij_quantum`

当前主 classical kind：

`raj_multij_johnson`

当前 Weighted multi-j Quantum 不是 Raj 论文逐字逐门的完整复现。

它是 ICCT 对 Raj 论文思想的任务适配，已包含：
- j-subset representation；
- weighted / graph-conditioned Johnson quantum mixing；
- continuous SinD physical subset features；
- fixed-particle-number embedding evolution；
- compound/RBS-style feature dynamics；
- data re-uploading；
- j=2 + j=3 全程量子分支；
- per-agent local fusion。

它没有声称完整复现论文最终 cross-register joint mixer M(theta_M)。

这一边界必须继续诚实保留。
## 8. raj_fidelity_a.py 的特殊说明

文件：

`prediction/qgnn_paper_native/raj_fidelity_a.py`

它此前处于 untracked 状态，但 seed2026/2027 成功实验的 config source hash 都包含它：

SHA256:
`78fd0b588ac28bb366a0464d5fcfb11d136f62db142c57de5d0a1d2baf857cc2`

交接时已确认服务器文件 hash 完全一致，并在 commit `e924943` 中补提交。

作用：
- 保证历史已通过实验的 source-hash 可复现；
- 它是 Fidelity-A 中间实现 / 研究文件；
- **当前正式主模型不是它，而是 raj_paper.py 中的 RajWeightedMultiJQGNNCore。**

新对话不要把 raj_fidelity_a.py 误当成新的主线，也不要随意删除，因为它属于成功实验的 source snapshot。
## 9. Representation bottleneck audit

正式结果：

`reports/qgnn/round4_representation_bottleneck_audit_20260919.json`

history-only probe target：
冻结 strong-classical graph interaction residual。

Validation R²：
- raw pair/triplet physical representation ≈ 0.853
- beta/gamma / quantum control angles ≈ 0.863
- rich X/Y/Z/K2/K3 observable ≈ 0.836
- relation-carrying pre-readout ~500D ≈ 0.922
- compressed quantum interaction 64D ≈ 0.824
- local-only 64D ≈ 0.872
- final local+quantum 64D ≈ 0.919

解释：
- 没有证据表明原始 beta/gamma encoding 存在灾难性信息损失；
- 500D→64D 单独 quantum correction 会丢信息，但 final local+quantum 64D 保留了大部分可恢复信息；
- 早期 RC-HQGNN 的失败不能单独归因于 encoding 或 readout；
- 改成 subset identity + graph-conditioned subset quantum evolution 后，模型性能显著提高；
- 当前更值得继续验证的是 Raj-style subset quantum route，而不是回去继续调 RC-HQGNN gate/controller。
## 10. Round2 / Round3 只作为历史，不要回退

Round2 RC-HQGNN full 0 dB：
- Quantum ADE 0.51094460 / FDE 1.12605076
- matched classical 0.49146503 / 1.07928883
- Quantum 全局更差约3.96% / 4.33%
- 高动态 closing>=15 子集曾有正优势。

Round3 Scene-Adaptive：
- adaptive / K2-K3 feedback 有小幅正贡献；
- 但 4096-train paired 仍整体输 classical ~2.5%
- full-train gate失败，因此停止。

Round4 已经提供比这两轮更强、重复的 paper-native positive signal。

因此新对话不要重新启动：
- RC-HQGNN controller search
- lambda2/lambda3 adaptive search
- K3 router search
- “什么时候用量子”的 classical/quantum switch。

这些都不是当前主线。
## 11. 当前最重要的未完成验证

当前还没有：
- full 15,802 train-window paired confirmation；
- full-train 多 seed；
- five-SNR；
- final test；
- 完整 Raj cross-register joint-mixer 复现。

由于 4096-train 下 seed2026/2027 均通过 ADE+FDE 双指标 gate，当前下一步应该优先 **增强证据强度，而不是再改架构**。

### 推荐的下一步：Full-train 0 dB paired confirmation

先写 prereg：

`docs/qgnn/ROUND4_RAJ_FULLTRAIN_PREREG_20260919.md`

建议冻结：
- dataset: SinD
- SNR: 0 dB
- seed: 2026
- train: full 15,802 windows
- val: all 1,880
- epoch: 20
- batch: 32
- lr: 3e-4
- correction cap: 16
- depth: 3
- same GPT-2 / Tokenizer / loss / optimizer
- Quantum kind: raj_weighted_multij_quantum
- Classical kind: raj_multij_johnson
- test closed
- 两张 RTX4090 各跑一边
- 不做 architecture / LR / loss / Token 调参

Primary gate：
**Quantum ADE 与 FDE 都必须低于 JohnsonGIN。**

Strong gate：
- 两项都赢；
- J gain >= 1%。

如果 full-train seed2026 仍通过：
- 再决定 full-train seed2027 / seed2028；
- 然后才进入五 SNR；
- test 最后统一授权使用。

如果 full-train 任一指标不赢：
- 不立即改 baseline 或 cherry-pick；
- 先分析为何 4096 positive signal 未扩展到 full train；
- 再决定是否追加一个 full seed 或转入 SQM-GNN 路线。
## 12. 可复用的已通过训练参数

历史成功 Quantum run config：
`reports/qgnn/round4_raj_weighted_multij_quantum_20e_2026/config.json`

核心：
- kind=raj_weighted_multij_quantum
- seed=2026
- snr=0
- epochs=20
- batch_size=32
- train_limit=4096
- depth=3
- correction_cap=16
- lr=3e-4

历史成功 Johnson run config：
`reports/qgnn/round4_raj_multij_johnson_20ep_2026/config.json`

核心相同，只是：
- kind=raj_multij_johnson

full-train 时最关键变化：
- 不再设置 train_limit=4096，或显式用完整数据语义；
- 使用新 run-dir；
- max_seconds 要给足，并支持 resume。

不要复用旧 checkpoint 冒充 full-train。
## 13. 长任务防卡死规则

这是用户非常在意的硬约束。

任何 >几分钟的训练/评估：
- 后台独立进程；
- 记录 PID；
- 记录 CUDA_VISIBLE_DEVICES；
- 保存完整 command；
- job.json；
- heartbeat.json；
- stdout/stderr → run.log；
- 至少每25 steps heartbeat；
- 定期 checkpoint；
- 支持 resume；
- 短轮询，不用一个 blocking call 等20-40分钟；
- OOM / deadlock / exited / normal-slow 要区分；
- 明显失败及时停；
- 不让一次训练锁死整个 ChatGPT 会话。

2×RTX4090 24GB 可并行跑 Quantum / Johnson。
## 14. 新对话启动阅读顺序

必须按以下顺序：

1. `docs/qgnn/ROUND4_RAJ_NEW_CHAT_HANDOFF_20260919.md`
2. `docs/qgnn/ROUND4_RAJ_PAPER_NATIVE_FINDINGS_20260919.md`
3. `reports/qgnn/ROUND4_RAJ_TWO_SEED_RESULTS_20260919.json`
4. `docs/qgnn/ROUND4_BASELINE_POLICY_20260919.md`
5. `docs/qgnn/ROUND4_RAJ_CONVERGENCE_PREREG_20260919.md`
6. `docs/qgnn/ROUND4_RAJ_SEED2027_CONFIRMATION_PREREG_20260919.md`
7. `prediction/qgnn_paper_native/raj_paper.py`
8. `prediction/qgnn_paper_native/model.py`
9. `scripts/train_qgnn_paper_native.py`
10. `reports/qgnn/round4_representation_bottleneck_audit_20260919.json`

需要追溯历史时再读：
- ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md
- FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md

不要把历史 Round2/Round3 的失败架构重新当成主候选。
## 15. 论文与代码锚点

主论文：

Snehal Raj et al.,
“Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler–Leman Hierarchy”
arXiv:2606.26873 (2026)

官方代码：
https://github.com/SnehalRaj/mp-qgnns

当前 ICCT 模型是 Raj-inspired adaptation，不等同于论文逐字复现。

此前审计的备用路线：
- SQM-GNN 2026：quantum U_MSG + U_UPD，无已确认官方完整仓库；若 Raj full confirmation失败，再考虑。
- Edge-Local QGCN 2026：官方 QGCNlib，适合 quantum baseline/comparator，但 pairwise + classical aggregation，不是当前主路线。
- Skolik EQC 2023：graph-weighted equivariant quantum ansatz，主要作为历史理论/量子 baseline。
## 16. 当前科学状态一句话

当前可以说：

> 在 SinD 0 dB、4096训练窗口、1880验证窗口、20 epoch 下，Raj-inspired Weighted multi-j QGNN 已在 seed2026 和 seed2027 两次独立 paired run 中同时优于 paper-native 主 baseline multi-j JohnsonGIN 的 ADE 与 FDE，两 seed 平均优势约 2.7%–3.0%。

当前不能说：
- 已达到5%；
- 已在 full train 验证；
- 已在 test 验证；
- 已在五SNR验证；
- 已全面优于所有经典方法；
- 已完整复现 Raj 原论文全部 joint-register architecture。

## 17. 新对话的任务原则

新对话接手后：
- 不重新讨论“QGNN 到底要不要用 Raj”；
- 不重新讨论 baseline policy；
- 不从 GitHub 旧远端覆盖服务器；
- 先做服务器审计和 prereg；
- 优先执行 full-train paired confirmation；
- 除非发现明确代码正确性问题，否则不要在 full confirmation 前继续架构搜索；
- 结果无论正负都完整保留并提交。

当前交接目标是把研究从“架构探索”推进到“证据确认”阶段。
