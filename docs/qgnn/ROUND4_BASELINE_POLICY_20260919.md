# Round 4 Baseline Policy — Paper-Native First

日期：2026-09-19。

## 1. 新的 baseline 原则

如果 ICCT 最终主 QGNN 明确采用某篇论文的量子图计算路线，则正式主 baseline 优先沿用该论文自己的 classical comparator / ablation family，而不是要求新量子路线首先击败此前 ICCT 自研的最强 classical interaction core。

原因：
- 论文原生 baseline 与其量子模型通常共享图表示层级、输入权限和任务接口，最能解释“量子计算本身是否带来增益”；
- 避免拿一个为 ICCT 专门反复强化的经典结构去压制刚迁移的论文方法，从而把“方法迁移失败”和“量子机制失败”混在一起；
- 仍保留强项目级 classical reference，避免只赢弱 baseline 后夸大量子优势。

## 2. Baseline 分层

### Tier A — Paper-native primary baselines

若采用 Raj et al. 2026 路线：
- GIN：original graph / 1-WL baseline；
- JohnsonGIN：同一 j-subset / Johnson lift 上的经典 message passing，最关键 matched baseline；
- PPGN / 3-WL higher-order reference（资源允许时）；
- 对连续回归任务，再参考论文 QM9 中使用的 classical message-passing network。

若采用 Edge-Local QGCN：
- HybridQGCN：相同 quantum feature extraction + classical aggregation，是最关键原生 ablation；
- classical GCN/DGI reference。

若采用 SQM-GNN：
- 论文 classical GNN；
- WMMSE 仅是其无线优化任务的算法基准，迁到轨迹预测不直接适用；
- QSGCN 作为 quantum comparator。

### Tier B — Quantum baselines / ablations

- RC-HQGNN v3；
- Scene-Adaptive RC-HQGNN；
- Edge-Local QGCN；
- Skolik EQC / symmetry ablations（如工程允许）。

### Tier C — ICCT strong classical references

- Adaptive Classical Higher-Order core；
- Pairwise / Routed / Pair+Triplet historical models。

Tier C 降级为 secondary strong reference：仍然报告，但不再作为选择 paper-native QGNN 路线的第一道准入门槛。

## 3. Raj 路线的正式核心比较

第一优先：

Raj-QGNN(j=2/3 or multi-j) + same GPT-2
vs
JohnsonGIN(j=2/3 or multi-j) + same GPT-2

必须保持：
- 相同 SinD / ISAC history；
- 相同 subset features；
- 相同 j-subset topology / Johnson lift；
- 相同 GPT-2、Token、loss、train split、seed 和训练曝光；
- 不给量子侧额外未来信息；
- classical baseline 不故意减宽或减层。

第二优先：
- GIN；
- PPGN / 3-WL；
- 论文对应 continuous-task classical GNN。

第三优先：
- ICCT Adaptive Classical 作为项目级强参考。

## 4. 当前 Round4 简化 Raj pilot 的地位

当前 `prediction/qgnn_paper_native/` 中的 Raj prototype 是论文思想迁移验证，不是最终忠实实现。

已完成 0 dB / seed2026 / 4096 train / 1880 val / 12 epoch：
- j=2 Raj-QGNN：ADE 0.608321 / FDE 1.246530 / J 1.231586；
- j=2 JohnsonGIN：0.604584 / 1.220608 / 1.214888；
- j=3 Raj-QGNN：0.609641 / 1.253344 / 1.236313；
- j=3 JohnsonGIN：0.610803 / 1.233711 / 1.227659；
- multi-j Quantum：0.571660 / 1.233083 / 1.188201；
- multi-j Johnson：0.552304 / 1.151958 / 1.128283。

说明：
- subset representation 本身对 SinD 很有价值；
- 简化 quantum Johnson mixing 尚未超过 matched Johnson classical；
- multi-j quantum 相对此前 RC-HQGNN 的 ADE 已明显提升，但 Johnson classical 从同一表示空间获益更大；
- 因而下一步重点不是继续强化旧 Adaptive Classical，而是补齐 Raj 论文更完整的 quantum feature dynamics / readout，再重新打 paper-native baseline。

## 5. 决策边界

如果更完整 Raj-style quantum implementation 仍稳定输 JohnsonGIN / PPGN，则停止把 Raj 作为主量子路线，转向 SQM-GNN-style U_MSG / U_UPD。

如果 Raj-style 能赢 paper-native baseline，但仍输 ICCT Adaptive Classical：
- 可以继续作为论文主量子路线候选；
- Adaptive Classical 结果必须诚实保留为 secondary strong reference；
- 不能把“赢 paper-native baseline”写成“全面优于所有经典方法”。

