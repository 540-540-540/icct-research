# SinD Gate B 预注册判决与运行合同

日期：2026-09-21

本文件在正式 Gate B 结果生成前冻结。主比较为同一 SinD-IC4、同一训练子集、同一 batch order、同一非-core 初始化、同一 TCN/CV/decoder/loss/checkpoint rule 下：

`RajMultiJJohnsonCore - RajWeightedMultiJQGNNCore`

正值表示 Raj 误差更低。Primary view 为 IC4；ADE、fixed-endpoint FDE4、J 全部报告，J 只用于 checkpoint selection。

## Decision rule

- PASS：2026/2027 两个 seed 的 Raj 均改善 IC4 ADE、FDE4、J；六个对应 origin-cluster paired-bootstrap 95% CI 下界全部大于 0；无 convergence warning。
- CONDITIONAL PASS：两 seed 平均在三项指标均为正，且至少一个 seed 三项全正，但仍有一个明确有限的 bootstrap 或收敛确认缺口。
- FAIL：其余情况。

Strong Graph 与 AllGraph 是绝对性能参考，不是 matched quantum-classical comparator。只有 Raj 同时优于 Johnson、Strong Graph 和 AllGraph 时，才允许写“超过全部已测经典参考”；仅优于 Johnson 时只能写 matched advantage。

## Symmetric convergence rule

两模型先按相同规则运行至最多 30 epochs。若任一模型在 base phase 中 best epoch>=29 且最后三个 validation IC4 J 连续下降，则两模型同时扩展到最多 45 epochs；不能按胜负单独延长。

正式 test 保持 sealed。冻结 Raj core、j=2+j=3、数据 population、IC membership 和 checkpoint rule均不因结果修改。
