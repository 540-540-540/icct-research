# Gate A 独立审计：Task / Interaction Necessity

日期：2026-09-21
状态：`AUDIT COMPLETE / P0 SOURCE BLOCKED / GATE A NOT YET EVALUABLE`
Material Passport：academic-research-suite / experiment-agent validate；证据状态 `ANALYZED`，未执行复现实验。

## 1. 工作区起点与审计边界

正式目录：`/home/dell/YrM/ICCT`
分支：`qgnn`
起点提交：`f935223aa8f954a689336521b5e665e5dc05de8c`

开工时 `git status --short` 只有既有未跟踪文件：一份 task-redesign 审阅、若干 q0 结果目录/JSON，以及 `results/qgnn/`。本审计没有清理、移动、stash 或纳入这些外部产物；未修改模型，未运行训练，未读取 test label。

审计覆盖实际代码、config、summary、逐 epoch training、validation diagnostics 和数据 manifest；Markdown 只用于定位与交叉核对。Automatum 已弃用，因此只保留解释旧反例与不可比性所需的最小审计。

## 2. 审计结论摘要

1. 旧 Automatum 与旧 SinD q0 都是约 `20 history -> 20 future`，不是 IC4。
2. Automatum 的 Graph < Self 与 SinD 的 Pair/Triplet 正增益来自不同数据、membership、模型、训练协议和参数量，不能直接比较，更不能解释成数据集层面的因果差异。
3. SinD Pair+Triplet 的 `+4.947% ADE / +5.562% FDE` 可从 1,880 行 validation diagnostics 精确复算，是可信的单 seed 正信号；但它不是 Strong Graph vs Strong Self 的 Gate A 证据。
4. 当前 IC4 只完成阈值预登记、history-indexed membership 和未来可见性元数据统计；20->40 数值标签、future mask、cache/dataloader、内部 split 及五模型组均未实现。
5. 当前 history 的底层 `x,y,vx,vy` 来自 `Veh_smoothed_tracks.csv`。项目没有从 pre-smoothing observations 重建 causal positions/velocities；逐帧 3-BS sensing 不能消除上游 RTS smoothing 的未来信息风险。
6. 当前 benchmark **不能称 strict online forecasting**。在获得/重建 causal source 前，只能在另行明确接受适用范围后称为 `offline latent-state controlled-sensing simulation`。
7. Raj QGNN 结果应继续冻结；GPT-2 utility 与 Gate A 分离。下一步不是调 Raj/GPT-2，而是先解决 P0 source scope，再实现 IC4 数据合同。

## 3. 当前 q0 到底用了什么任务

### 3.1 Automatum（历史反例，低优先级）

实际合同：

- 20 帧 history + 20 帧 future；29.97 Hz 源数据每 3 帧取样，约 9.99 Hz，即约 2.002 s -> 2.002 s。
- 样本 stride=1；车辆必须完整存在于全部 40 帧；只保留 `2<=N<=8`，`N>8` 整窗拒绝。
- history 输入来自 frozen Route-B `state_hat=[x_hat,y_hat,vx_hat,vy_hat]`；future 是 GT `[x,y,vx,vy]`。
- 两个 scene 分别按连续时间轴约 8:1:1 划分，并留 40 帧 gap；train/val/test 为 10,366/893/1,135。
- 训练入口只实例化 train/val；summary 声明 `test_set_used=false`。

最直接结果：同一 Simple-GRU downstream、0 dB、seed 2026、40 epochs 下：

| 模型 | ADE / m | FDE / m | J |
|---|---:|---:|---:|
| NoGraph | 0.379483 | 0.643283 | 0.701125 |
| plain MPNN | 0.451491 | 0.837012 | 0.869997 |

MPNN 分别恶化 18.98%、30.12%、24.09%。但 plain MPNN 对所有有效非自身 pair 聚合，没有 interaction-critical membership 或半径截断，也不是 NoGraph 的严格嵌套零初始化扩展；参数为 684,840 vs 532,776，且只有一个 seed。

更有解释力的是 frozen-base residual 对照：Pairwise 虽把原 base 改善到 0.499086/0.858638，但参数近似的 Own/Local capacity control 达到 0.456623/0.757666（trainable 213,697 vs 211,969）。因此当前最稳妥解释是：在普通短时 Automatum 窗口中，neighbor-specific branch 的边际价值没有超过额外自车容量；不能推出邻居绝对无用。

### 3.2 SinD q0（旧正信号）

实际合同：

- 20 history + 20 future，约 2s->2s，N<=8，world-frame 状态。
- 每个城市录像内 chronological 8:1:1，两个 40-frame gap，并清除跨 gap 车辆；train/val 为 15,802/1,880 scene windows。
- 不是逐 target：最后历史帧只要至少一辆车 active-degree>=2 就保留；N>8 时选一个 focal+7 neighbors；focal 不固定 slot，所选全部车辆均参与监督。
- 候选车辆必须完整存在到 future 第 20 帧，所以整个 node membership 依赖 future persistence；只有 interaction ranking 使用最后历史帧。
- history 来自 0 dB frozen sensing cache，future 为同源 smoothed trajectory GT；训练只用 train/val。

实际结果：

| 模型 | ADE / m | FDE / m | J | best epoch |
|---|---:|---:|---:|---:|
| NoGraph | 0.524150 | 1.063804 | 1.056052 | 17 |
| Pair MPNN | 0.502264 | 1.030864 | 1.017696 | 20 |
| Pair+Triplet | 0.498218 | 1.004633 | 1.000534 | 17 |

Pair+Triplet 相对 NoGraph 为 `+4.947% ADE / +5.562% FDE`。逐行重算确认四份 diagnostics 的 1,880 个 dataset index/scene/start frame 完全对齐，聚合值匹配；NoGraph 与 Pair+Triplet 又恰好都选 epoch 17，epoch 20 方向也保持。

`closing30_ge15` 的 140 个 validation windows 上，Pair+Triplet 为 `+8.656% ADE / +10.217% FDE`。该分层由 history state 计算，但没有发现事先登记证据，且与 N≈8/密度强耦合，只能作为探索性 validation 诊断。

这个正信号仍不能通过 Gate A：

- 单 seed，validation 同时用于 checkpoint selection 与报告；
- Pair+Triplet 比 NoGraph 多 238,978 个 trainable parameters；
- 没有 Own-only、Pooling、all-neighbor controls；
- temporal backbone 是当前明显弱于 TCN 的 MotionToken GPT-2；
- 旧数据不是 target-centered，也不是 strict history-only membership；
- 没有第二 seed 或新 IC4 结果。

## 4. 为什么两批结果不可直接比较

共同点只有约 2s->2s、0 dB、seed 2026 和 scene-macro ADE/FDE/J。关键差异包括：

| 轴 | Automatum 负结果 | SinD 正结果 |
|---|---|---|
| 数据 | 普通完整多车窗口 | 旧 high-interaction 预筛窗口 |
| temporal | Simple-GRU | MotionToken GPT-2+LoRA |
| graph | plain all-pair MPNN | nested-safe Pair+Triplet |
| exposure | 40 epochs | 20 epochs |
| supervision | 全窗口车辆 | 全局部图车辆 |
| controls | 有近似容量 Local 对照 | 无 Own-only capacity control |
| 参数/优化 | 非严格匹配 | 非严格匹配 |

因此符号反转只说明 task/model/protocol dependent。它提出“SinD 的历史交互筛选可能更有用”的假设，不证明 SinD 天生需要 Graph，也不证明 Automatum 不需要 neighbors。

## 5. IC4 当前实现到什么程度

已经存在：

- 固定候选任务 `20 history -> 40 future`、stride 10、target-centered、Full4/IC4 同训练群；
- radius=50 m、closing>=0.5 m/s、0<TCPA<=40dt、DCPA<=5 m，以及 following proxy 的阈值预登记；
- 使用连续 20 帧历史与最后历史时刻 0 dB sensed state 计算 k；
- 只统计 future frame availability/endpoint 的 membership manifest；
- synthetic selector tests、跨 horizon k 一致性检查和元数据重计数。

尚不存在：

- `data/task_redesign/IC4_v1`；
- 20->40 数值 `future_xy/future_mask/future_ts`；
- 完整 20 帧 IC4 sensing cache 与 same-key parity 验收；
- fit/dev/design 内层 split 的实际构建和跨界 ID purge；
- ego-local transform、ragged all-neighbor loader、40-step decoder；
- Strong Self / Own / Pool / Graph / All-neighbor 五模型实现与 preflight；
- 任一 seed 2026 IC4 pilot。

当前正式树中的 `target_views_v1` 是旧 20->20 数据，不是 IC4。现有 `audit_history_only.py` 还读取临时 worktree 才有的 `data/task_redesign/raw`；正式 qgnn 只有 `data/sind/raw`。`verify_task_selection.py` 又硬限制旧临时 worktree 路径，因此当前正式树不能原样端到端复现已有 metadata audit。

现有 smoothed-state 统计可作为规模参考：Full4 train/val 为 36,778/3,855 targets；IC4 为 12,186/1,149；IC4 endpoint coverage 为 89.03%/86.34%。这些数值在 causal source 改变后必须用原冻结阈值重算。

## 6. Source-causality 审计

实际数据流是：

```text
Veh_smoothed_tracks.csv 的 x,y,vx,vy
-> 单帧 synthetic 3-BS range/bearing/radial-velocity
-> 单帧融合 state_hat=[x_hat,y_hat,vx_hat,vy_hat]
-> 20 帧 history
```

项目直接读取 CSV 提供的 `vx/vy`，没有从截至 t0 的 position prefix 因果估计速度。3-BS frontend 每帧独立，不含 temporal Kalman/tracker，但其 radial velocity 由 upstream smoothed velocity 生成，所以不能消除上游泄漏。

SinD 原论文 III-D 明确说明用 RTS smoothing refine trajectory，并同时估计 velocity/acceleration。RTS 是 backward smoothing 方法，方法层面会使用后续观测修正过去状态。当前 pinned Changchun/Xi'an release 没有随项目提供 pre-smoothing observations 或逐文件生成代码，故：

- 已证实：使用的是名为并实际发布为 `Veh_smoothed_tracks.csv` 的平滑 state；原 SinD pipeline 使用 RTS；当前 history 没有通过 prefix-invariance。
- 未证实：两个 pinned 文件具体使用了哪版实现、未来影响幅度多大。
- 不能写：泄漏幅度已经量化，或所有结果因此失效。
- 必须写：strict prediction-time causality 尚未建立。

所谓“history-only membership”目前只表示 selector 不显式读取 future coordinates/labels；不表示其 history 数值在 t0 可在线获得。对平滑位置重新做 past-only velocity 也无法消除 position 本身可能携带的后缀信息。

## 7. Raj 与 GPT-2 的边界

Raj Weighted Multi-J QGNN 的两 seed exact-match frozen-test 证据成立：ADE 0.502408 vs 0.517977，FDE 1.069034 vs 1.105594，J gain 3.161%，graph 参数 126,535 vs 125,697。但它严格属于旧 20->20、4096-train-sample、N<=8 benchmark，回答的是 Raj vs matched Johnson，不回答 Graph vs Strong Self。本轮继续优化 Raj 只会混淆 Gate A，故保持冻结。

注意：`RAJ_QGNN_EXACT_TWO_SEED_AGGREGATE_20260921.json` 是 validation aggregate，J gain 约 1.95%；论文中的 3.161% 必须引用 frozen-test 文件，不能混用。

最新 GPT-2 Self 诊断实际完成五组：TCN-ego 0.392270/0.862640；pretrained+LoRA 0.599186/1.300303；pretrained+full 0.545506/1.209782；random+LoRA 0.632950/1.392921；random+full 0.503522/1.120886。当前证据说明 full adaptation 优于 LoRA，但 full 条件下 random 优于 pretrained，尚无预训练正迁移证据。该 factorial 在代码上禁止 interaction core，是纯 Self 诊断，不能承担 Gate A。

## 8. 阶段判决与下一步

本轮完成的是 `AUDIT`，不是最终 Gate A 性能判决。当前不得输出 PASS/CONDITIONAL PASS/FAIL，因为 Strong Self/Own/Pool/Graph/All-neighbor 在 IC4 上尚未运行。

进入 `DATA CONTRACT` 前有一个实质 P0 决策：

1. **严格在线路线**：获得当前 release 的 pre-smoothing observations/causal filtered states，或从可核验原始观测重建 causal position、velocity、identity/quality pipeline，并做 prefix-invariance；随后固定现有 IC 阈值重算 membership/counts。
2. **离线状态路线**：明确将任务限定为 `offline latent-state controlled-sensing simulation`，不作 strict online forecasting 声明；这需要作为论文适用范围的明确决定，不能由实现者自动豁免。

在该范围决定前，不应进入 IC4 PREFLIGHT 或 seed 2026 pilot。无论选择哪条路线，都不修改 IC 阈值，不打开 test，不启动 Raj migration、GPT-2 修复或 QGNN 训练。

## 9. 关键证据路径

- `docs/task_redesign/ICCT_PAPER_NARRATIVE_REBUILD_TASK_20260921.md`
- `docs/task_redesign/CODEX_TRAJECTORY_TASK_REDESIGN_FREEZE_20260920.md`
- `scripts/task_redesign/audit_history_only.py`
- `reports/task_redesign/history_only_audit_20260920.json`
- `reports/task_redesign/audit_supplement_20260920.json`
- `reports/task_redesign/revalidation_20260920.json`
- `tools/data_preprocessing/build_sind_high_interaction.py`
- `frontend/sind_prediction_dataset.py`
- `reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`
- `reports/q0/automatum_qgnn_task_space_audit_0db.json`
- `reports/qgnn/RAJ_QGNN_FROZEN_TEST_0DB_TWO_SEED_20260921.json`
- `results/qgnn/gpt2_self_diagnostic_v1/*/{config,summary,training}.json`

外部 primary source：SinD paper `https://arxiv.org/pdf/2209.02297`；官方格式说明 `https://github.com/SOTIF-AVLab/SinD/blob/main/Format.md`。
