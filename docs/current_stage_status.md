# ICCT 当前阶段状态与施工边界（2026-09-22）

## 1. 当前唯一正式主线

正式服务器目录：`/home/dell/YrM/ICCT`
正式分支：`qgnn`

当前主线：

`Lankershim multi-target -> senior sensing/noise -> Raj-enhanced hybrid QGNN -> Motion-Token GPT-2/LoRA -> trajectory`

完整冻结说明见：`docs/MAINLINE_FREEZE_20260922.md`。

## 2. 已完成

- 师兄原多目标 R0 系统完整复现：**PASS**；
- Target Interaction GNN 相对 independent baseline 的交互收益复现：**PASS**；
- 师兄 Graph + Motion-Token GPT-2 正增益复现：**PASS**；
- Raj QGNN 集成与三 seed 训练：**PASS**；
- frozen test 三 seed：ADE `0.56045337 ± 0.00160149`，FDE `1.13526806 ± 0.00168084`；
- 相对师兄原 Graph+GPT-2：ADE `+5.1932%`，FDE `+5.0043%`；
- quantum-off 消融显示 Raj branch 对最终预测存在显著实际贡献；
- 旧 SinD/IC4 主线已转入 archive branch/tag 和服务器物理归档；
- 服务器临时 `ICCT-senior-r0-reproduction` worktree 已移除，仅保留一个正式 ICCT 项目目录。

## 3. 当前方法身份

当前方法不是纯量子网络，也不是简单在经典 GNN 后挂一个量子适配器。

正式结构为：

`self/history encoder -> Raj Weighted Multi-J quantum interaction -> 2-layer classical local graph propagation -> trajectory decoder -> Motion-Token GPT-2/LoRA`

Raj 先注入，经典图层后传播。论文中必须按此结构描述和归因。

## 4. 当前施工优先级

1. 补 matched classical 的三 seed formal confirmation（避免继续使用已打开 test 做选择）；
2. 五档/多档 SNR 主实验；
3. j=2 / j=3 / Multi-J 与关键量子机制消融；
4. 最终统计、图表和论文 Method/Experiment 证据链；
5. 完成后再决定是否保留 SinD-IC4 作为 appendix/generalization 研究，不再阻塞主线。

## 5. 数据与结果管理

Active runtime：
- `reports/senior_r0_reproduction/`
- `results/senior_raj_metric_aligned_quantum_refine/`
- `results/senior_raj_llm_validation/`
- `results/senior_raj_initialization_matched/`
- `data/multitarget_lankershim_v1.npz`

大型 runtime/checkpoint 仅保留服务器与本地同步镜像，不上传 GitHub。
