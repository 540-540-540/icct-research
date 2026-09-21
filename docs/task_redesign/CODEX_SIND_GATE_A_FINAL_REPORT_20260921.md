# SinD Gate A 最终报告：Task / Interaction Necessity

日期：2026-09-21
判决：`PASS`
证据等级：two-seed sufficiently-trained validation pilot；sealed test 未打开。

## 1. 结论

在修复后的 SinD-IC4 上，history-side neighbor information 相对 strong target-only TCN 和 own-only capacity control 提供了可复现的预测增益。两粒种子中，N<=8 Graph 与 AllGraph 都同时改善 IC4 ADE/FDE，origin-cluster paired bootstrap 的 Self-Graph 95% CI 均严格大于 0。因此 Gate A 的问题得到肯定回答。

这个 PASS 只支持“neighbors 有稳定边际预测价值”。它不支持“GNN 不可替代”“message passing 必不可少”或交通因果陈述；Pooling 已获得大部分增益，Graph-Pool 的不确定性并非两粒种子、两指标全部排除 0。AllGraph 与 N<=8 Graph 也没有稳定显著差距，因此本轮没有建立 N=8 information-cap loss。

## 2. 核心 IC4 结果

| seed | Self ADE/FDE | Own ADE/FDE | Pool ADE/FDE | Graph ADE/FDE | AllGraph ADE/FDE |
|---|---|---|---|---|---|
| 2026 | 1.2810 / 3.4408 | 1.2969 / 3.5180 | 1.1747 / 3.0321 | 1.1347 / 2.9154 | 1.1198 / 2.8614 |
| 2027 | 1.2894 / 3.4690 | 1.2706 / 3.4482 | 1.1654 / 3.0483 | 1.1282 / 2.8918 | 1.1324 / 2.8831 |

Graph 相对 Self：

- seed 2026：ADE -0.1463 m（-11.42%），FDE4 -0.5254 m（-15.27%）。
- seed 2027：ADE -0.1611 m（-12.50%），FDE4 -0.5772 m（-16.64%）。

Graph 相对 Own：两粒种子 ADE 分别改善 12.50% / 11.20%，FDE4 改善 17.13% / 16.14%，排除了“只是参数更多”的解释。CV IC4 为 2.0496 / 5.4257，五个已训练模型的 absolute accuracy 均明显优于 CV。

## 3. 不确定性与机制边界

5,000 次 prediction-origin cluster paired bootstrap：

- seed 2026 Self-Graph：ADE +0.1561 m，95% CI [0.0808, 0.2275]；FDE +0.5254 m，[0.2912, 0.7472]。
- seed 2027 Self-Graph：ADE +0.1710 m，[0.1114, 0.2307]；FDE +0.5772 m，[0.3839, 0.7648]。

这里正值表示前者误差更大。AllGraph-Self 的四个对应 CI 同样全为正。

Pooling 相对 Self 已改善 IC4 ADE 8.30%/9.62%、FDE 11.88%/12.13%。Graph 再比 Pool 改善约 3.2-3.4% ADE 与 3.8-5.1% FDE，但 IC4 bootstrap 中 seed 2026 的 Pool-Graph ADE/FDE CI 均跨 0，seed 2027 ADE 也跨 0。因此最稳健机制结论是 neighbor information utility，而不是 edge-aware message passing indispensability。

AllGraph 对 Graph：seed 2026 两指标小幅更好；seed 2027 ADE 略差、FDE 略好，paired CI 跨 0。不能声称 N<=8 cap 已造成显著信息损失。

## 4. 分层与限制

- 2026 的 IC4 相对百分比增益没有同时强于 Full4，但 IC4 的 absolute Self-Graph gain 更大；2027 的 IC4 gain concentration 成立。
- Changchun 两粒种子方向一致。Xi'an IC4 origin 只有 36（FDE origin 35）；2026 Graph 的 Xi'an ADE 改善但 FDE 略差，2027 两项均改善。不能写成 city-uniform effect。
- 结果来自 validation，虽然两 seed、共同 checkpoint criterion、paired uncertainty 和 convergence diagnosis 均支持 PASS，但 test 仍保持 sealed；正式最终泛化数值需在论文冻结后一次性评估。
- 官方 SinD smoothed trajectories 是 canonical state；不宣称 raw-video online reconstruction。

## 5. 历史审计问题的最终回答

1. q0 Automatum 与旧 SinD 都是约 2s->2s，不是 IC4。Automatum 是完整窗口、N<=8、约 8:1:1 chronological split；旧 SinD 是 high-interaction 预筛、focal local graph、future-persistence-dependent membership。Automatum 已弃用，只保留历史反例的最低优先级审计。
2. Automatum Graph<Self 主要说明当时 all-pair MPNN 的 neighbor branch 没有超过 strong own-capacity control；它不是 neighbor 无用的普遍结论。
3. 旧 SinD Pair+Triplet 的约 4.95%/5.56% 正增益来自预筛 high-interaction windows 与 nested-safe higher-order core，是可信单-seed hypothesis signal，但不是 Gate A 证明。
4. 两批结果在数据 population、membership、temporal model、graph core、参数、训练 exposure 和 supervision 上均不匹配，不能直接比较。
5. IC4 现在已完成数值 train/validation cache、future mask、3-BS 补齐、五 controls、固定 endpoint、preflight、两 seed 与 paired bootstrap；test 没有 materialize。
6. history 使用官方 smoothed canonical states。官方 smoothing 可能包含双向估计语义，但已由最高优先级数据政策接受为共享数据合同；未来 label 绝未进入本项目定义的 history/membership/ranking。
7. 可以称 canonical-state 条件下 strict history-only forecasting task；不能称 raw-video strict online state reconstruction。

## 6. Gate B readiness

Gate A PASS 后只完成迁移就绪，没有启动 QGNN 正式训练：

- 冻结 `RajWeightedMultiJQGNNCore (j=2+j=3)`；
- matched comparator 为 `RajMultiJJohnsonCore`；
- target+前 7 个 history-ranked neighbors、20->40 supervision、同 sensed history/mask/ego contract；
- 共用 Gate A TCN、CV anchor、interaction projection、40-step decoder、loss 和 checkpoint rule；
- CUDA shape/mask/padded-node invariance/finite-gradient/共同初始化/参数审计全部通过；
- AllGraph 保留为 information-cap reference。

Gate B 参数量：Raj 420,521；matched Johnson 473,187；二者非-core 参数完全一致。正式长训练入口为 `scripts/run_sind_gate_b.sh`，本轮未执行。

## 7. 可复现产物

- `reports/task_redesign/SIND_GATE_A_FINAL_SUMMARY_20260921.json`
- `reports/task_redesign/SIND_GATE_A_PAIRED_BOOTSTRAP_20260921.json`
- `reports/task_redesign/sind_gate_a_seed2026/summary.json`
- `reports/task_redesign/sind_gate_a_seed2027/summary.json`
- `reports/task_redesign/SIND_GATE_B_READINESS_20260921.json`
- `configs/gate_b_raj_migration.json`
