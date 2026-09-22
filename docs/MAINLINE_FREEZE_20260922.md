# ICCT 主线冻结：Senior Stack + Raj QGNN + Motion-Token GPT-2（2026-09-22）

## 1. 主线决策

自 2026-09-22 起，ICCT 正式 `qgnn` 主线升级为：

`Lankershim multi-target scenes -> senior sensing/noise protocol -> Raj-enhanced interaction backbone -> Motion-Token GPT-2/LoRA -> future trajectory`

此前 SinD-IC4 / Gate A / Gate B / Gate B+ 路线不再作为当前施工主线，已进入历史归档；其 Git 历史和服务器运行产物均保留，可随时回溯。

## 2. R0：师兄原系统完整复现

本轮先原样复现师兄多目标实验，而不是直接加入 QGNN。

冻结任务：
- 数据：NGSIM Lankershim 共时多车场景；
- history / future：20 -> 20，`dt=0.1 s`；
- `N <= 8`，邻域半径 45 m；
- train / val / test：6000 / 1200 / 1200；
- 沿用师兄原 sensing/noise、baseline、Target Interaction GNN、Motion Token、GPT-2/LoRA、decoder、训练与评测协议。

R0 test 复现：
- Independent GRU：ADE `0.70036555`，FDE `1.47034609`；
- Target Interaction GNN：ADE `0.61963260`，FDE `1.25282386`；
- Graph + Motion-Token GPT-2：ADE `0.59115313`，FDE `1.19507305`。

结果与师兄历史结果高度一致，R0 复现成立。

## 3. R1：当前正式方法

当前最终模型不是“纯 Raj 完全替代经典 GNN”，而是量子-经典混合交互骨干：

1. 师兄原 self/history encoder 形成每车节点状态；
2. `RajWeightedMultiJQGNNCore(rounds=3)` 建模 multi-order joint interaction；
3. `quantum_first = true`，Raj interaction 先以残差形式注入节点表示；
4. `quantum_scale = 0.05`；
5. 随后保留 `classical_layers = 2` 的经典局部图传播；
6. 轨迹粗预测与图表示继续接入师兄原 Motion-Token GPT-2/LoRA；
7. GPT-2 基础权重冻结，LoRA 与预测头按冻结协议训练。

因此论文方法应表述为 **Raj-enhanced hybrid QGNN interaction backbone**，其中 Raj 负责 multi-order quantum interaction encoding，经典图层负责稳定的局部传播，LLM 负责后续时序预测/refinement。

## 4. 冻结性能结果

### 4.1 Validation 三 seed

最终 metric-aligned quantum refinement：
- ADE `0.52086908 ± 0.00148451`；
- FDE `1.04870809 ± 0.00228177`。

Matched classical GNN+LLM 三 seed mean：
- ADE `0.54784824`；
- FDE `1.10349883`。

Validation 上 Raj 方案相对 matched classical：
- ADE 改善 `4.9246%`；
- FDE 改善 `4.9652%`。

### 4.2 Frozen test 三 seed

最终 frozen test：
- ADE `0.56045337 ± 0.00160149`；
- FDE `1.13526806 ± 0.00168084`。

师兄原 Graph + GPT-2 test：
- ADE `0.59115313`；
- FDE `1.19507305`。

最终方案相对师兄原完整方案：
- ADE 改善 `5.1932%`；
- FDE 改善 `5.0043%`。

关闭 Raj quantum contribution 后，test 三 seed mean：
- ADE `0.63607839`；
- FDE `1.32119616`。

启用 Raj 后相对 quantum-off：
- ADE 改善 `11.8893%`；
- FDE 改善 `14.0727%`。

该 test 仅在协议和 checkpoint 冻结后打开一次；打开后没有再更改模型选择、参数或 checkpoint。后续任何新结构/调参不得继续使用该 test 进行选择，应使用新的 holdout 或外部确认集。

## 5. 当前结论边界

已证实：
- 师兄原系统可稳定复现；
- 当前 Raj-enhanced QGNN+LLM 在 frozen test 上相对师兄原 Graph+GPT-2 同时达到超过 5% 的 ADE/FDE 改善；
- 三 seed 波动很小；
- Raj quantum branch 对当前模型输出具有显著实际贡献。

仍需补齐：
- matched classical counterpart 的同等三 seed frozen-test / 新 holdout confirmation，进一步收紧“quantum mechanism vs matched classical mechanism”的因果对照；
- multi-SNR 正式主实验；
- j=2 / j=3 / Multi-J、量子顺序与必要机制消融；
- 最终论文图、方法公式与统计报告。

## 6. Active runtime assets

服务器正式项目：`/home/dell/YrM/ICCT`

当前需继续保留的本地/服务器运行资产：
- `reports/senior_r0_reproduction/`
- `results/senior_raj_metric_aligned_quantum_refine/`
- `results/senior_raj_llm_validation/`
- `results/senior_raj_initialization_matched/`
- `data/multitarget_lankershim_v1.npz`

其中大型 checkpoint / runtime 结果不上传 GitHub；GitHub 保留代码、协议、关键 JSON 报告和冻结文档。

## 7. 历史归档

旧 SinD/IC4 主线：
- branch：`archive/sind-ic4-mainline-20260922`
- tag：`archive-sind-ic4-mainline-20260922`
- commit：`dad2b5e79920b57c77cef9600882408b6c899c7c`

Senior R0/R1 开发冻结：
- branch：`archive/senior-r0-r1-development-20260922`
- tag：`senior-raj-r1-freeze-20260922`
- commit：`31ea9d2b3426f08a2e70a7b02301c2c7dfd07e83`

服务器物理归档位于正式项目 `archive/` 下，不上传 GitHub。
