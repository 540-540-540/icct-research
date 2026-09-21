# ICCT SinD 正式数据集决策与主线恢复（2026-09-21）

## 0. 权威性

本文件冻结 ICCT 项目当前正式数据集政策，并覆盖此前把 SinD 官方 smoothing 提升为 strict-causal 阻塞项的临时判断。

如与以下历史文件冲突，以本文件为准：

- `docs/task_redesign/ICCT_PAPER_NARRATIVE_REBUILD_TASK_20260921.md`
- `docs/task_redesign/CODEX_GATE_A_INDEPENDENT_AUDIT_20260921.md`
- `docs/task_redesign/CODEX_GATE_A_FINAL_REPORT_20260921.md`
- `reports/task_redesign/GATE_A_FINAL_SUMMARY_20260921.json`
- `configs/gate_b_raj_migration.json`

## 1. 正式主数据集冻结

ICCT 论文正式主数据集保持为：

`SinD public Changchun + Xi'an`

正式 source 为当前仓库已经 pin 住并完成 SHA256 校验的官方：

- Changchun `Veh_smoothed_tracks.csv`
- Xi'an `Veh_smoothed_tracks.csv`

这些文件视为 SinD 官方交付的 canonical trajectory data。

## 2. 对官方 smoothing 的最终处理原则

SinD 官方提供的是 smoothed trajectory，官方处理链包含 trajectory smoothing / state estimation。

本项目不再把“官方 smoothing”本身定义为需要中断主线的项目级 data leakage。

原因：

1. smoothing 是数据集官方预处理的一部分，不是本项目为了提高预测成绩额外引入；
2. Self / classical Graph / QGNN / LLM 在正式比较中必须共享相同官方数据合同；
3. 本论文核心是 ISAC sensing + multi-agent interaction + GNN/QGNN + LLM，而不是从原始无人机视频重新实现 detector/tracker/filter；
4. 使用官方处理后的 trajectory 可以减少与论文核心无关的轨迹去噪和运动状态清洗工作。

论文中应诚实披露：

> Experiments use the official smoothed trajectories released by SinD as the canonical trajectory states.

不得额外声称：

> 本项目从原始视频实现了严格在线 causal state reconstruction。

## 3. 什么仍然属于真正需要修复的问题

接受官方 smoothing 不代表放松我们自己的 benchmark 设计。

以下项目自定义行为仍必须严格修正：

- sample membership 不得由 future availability 决定；
- neighbor inclusion / interaction score 必须只使用 history-side state；
- target-centered sample key 必须先冻结，再挂 future label；
- future 缺失只产生 `future_mask`；
- train / val / test physical trajectory 不得交叉污染；
- test 不得用于模型设计或 checkpoint selection；
- Self / Own / Pool / Graph / QGNN 必须使用匹配的数据、坐标、temporal encoder / decoder 和训练预算；
- FDE2/FDE4 必须使用固定 2s/4s endpoint，不能用“最后一个可见 future point”替代。

## 4. Lankershim 的正式地位

提交 `0a18050` 的 Lankershim strict-causal Gate A 实验保留。

它的正式地位为：

`auxiliary causal sanity experiment`

它可以说明 Graph-Necessity methodology 在一个严格 causal 的真实轨迹数据上能够检测到 neighbor utility。

但它：

- 不是正式主数据集；
- 不能替代 SinD；
- 不能作为 SinD Gate A PASS；
- 不能授权正式 Raj Gate B。

因此当前正式状态为：

`SinD Gate A: NOT YET EVALUATED ON THE REDESIGNED TASK`

## 5. 当前主线

从现在起只推进：

```text
Official SinD canonical trajectories
→ redesigned target-centered SinD-IC4
→ Graph-Necessity Gate A
→ Raj QGNN Gate B
→ GPT-2 / LLM Gate C
```

不再因为官方 smoothing 去：

- 回退 Lankershim；
- 更换数据集；
- 重建原始视频 detector / tracker；
- 搜索 pre-smoothing position 作为主线前置条件。

只有未来实验出现结构性异常，并且通过独立诊断证明异常确实由 SinD 官方 preprocessing / scene distribution 导致，才重新评估数据集。

## 6. SinD-IC4 正式任务合同

主候选：

`history ≈ 2s (20 frames) → future ≈ 4s (40 frames)`

同时保留：

- Full2
- IC2
- Full4
- IC4

要求：

- target-centered；
- target 为固定语义 slot 0；
- sample inclusion 只由 20-frame history 决定；
- IC membership 只由 history-side interaction rule 决定；
- IC2 / IC4 使用同一 membership；
- Full / IC 是同一训练 population 的 evaluation strata；
- future availability 只决定 mask；
- 不用 future maneuver / future error / future completeness 选择样本；
- history 使用官方 SinD canonical state，经同一 3-BS controlled ISAC 后输入模型；
- future label 使用官方 SinD canonical trajectory。

## 7. Gate A 科学问题

Gate A 只回答：

> 在新的 SinD-IC4 任务上，history-side neighbor information 是否在公平控制容量和训练条件后，稳定改善 target trajectory prediction？

最小比较：

1. Strong Self
2. Own-only Capacity Control
3. Context Pooling
4. Strong Graph N<=8
5. All-neighbor Strong Graph

先证明 neighbor utility，再讨论 Raj QGNN。

## 8. Gate B 解锁条件

只有 SinD Gate A 获得 PASS 或足够强的 CONDITIONAL PASS 后，才把冻结的 Raj Weighted Multi-J QGNN 迁移到同一个 SinD-IC4 benchmark。

Raj 核心架构不因本次任务重设计重新搜索。

其 matched comparator 仍应是参数和信息条件尽可能匹配的 classical Johnson core。

## 9. 项目自主权边界

后续 agent / Codex 可以自主调整：

- 数据实现方式；
- loader / mask / batching；
- causal-free 的 benchmark engineering；
- classical control 实现；
- training / convergence diagnostics。

但不得自主更换：

- 正式主数据集 SinD；
- 正式主研究问题；
- IC4 的 history-only membership 原则；
- 冻结 Raj 主架构。

“大步推进”仍然是默认：一次 Codex 开工应尽量完成一个科研 Gate，而不是只完成 Gate 中的一次审计。
