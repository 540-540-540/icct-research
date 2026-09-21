# SinD Gate B 最终报告：冻结 Raj QGNN 正式验证

判决：`FAIL`

## Primary IC4 table

| seed | Self | Pool | Strong Graph | AllGraph | Matched Johnson | Raj QGNN |
|---|---|---|---|---|---|---|
| 2026 | 1.2810 / 3.4408 / 3.0014 | 1.1747 / 3.0321 / 2.6907 | 1.1347 / 2.9154 / 2.5924 | 1.1198 / 2.8614 / 2.5505 | 1.1074 / 3.0022 / 2.6085 | 1.1359 / 3.0320 / 2.6519 |
| 2027 | 1.2894 / 3.4690 / 3.0239 | 1.1654 / 3.0483 / 2.6895 | 1.1282 / 2.8918 / 2.5741 | 1.1324 / 2.8831 / 2.5739 | 1.1544 / 3.0266 / 2.6677 | 1.1207 / 2.9984 / 2.6199 |

表中依次为 ADE / FDE4 / J；J 仅用于 checkpoint selection。

## Raj vs matched Johnson

- seed 2026: ADE -2.575%、FDE4 -0.992%、J -1.664%。
  Paired 95% CI: ADE [-0.0597164409229232, 0.0006288511092324864]，FDE4 [-0.12853587570819688, 0.0637309940334289]，J [-0.12025594360459432, 0.02995745890797571]。
- seed 2027: ADE +2.923%、FDE4 +0.932%、J +1.794%。
  Paired 95% CI: ADE [0.0043667073912017, 0.0646214373484994]，FDE4 [-0.06477470957140506, 0.12438500934398435]，J [-0.025411294826081288, 0.1270763523860241]。

Two-seed mean matched gain (Johnson - Raj): ADE +0.002617 m，FDE4 -0.000788 m，J +0.002223 m。
FDE4 均值仍为负，因此不满足 CONDITIONAL PASS 的三指标均正条件。

## Absolute-reference finding

- Raj 未在两个 seed 上同时超过 Strong Graph 和 AllGraph。
- seed 2026：Raj 的 ADE/FDE4/J 均劣于 AllGraph，FDE4/J 也劣于 Strong Graph。
- seed 2027：Raj ADE 优于两个外部图基线，但 FDE4/J 仍劣于两者。
- 因此本轮不能声称 QGNN top1、量子优势或不可替代性。

## Statistical validity scan

- Selection: 两个预登记 seed 全部报告，无 seed/cherry-pick。
- Checkpoint: 两模型共用 validation IC4 J 选择规则，未访问 test。
- Multiplicity: Gate 按预登记的 ADE/FDE4/J 联合条件判定，未从子组中挑显著结果。
- Dependence: 95% CI 在 prediction-origin 层面聚类 bootstrap，不把同场景样本当独立。
- Causality: 只报告预测误差差异，不声称交通因果。
- Comparator: matched Johnson 是主对照，Strong Graph/AllGraph 只是绝对参照。

## Eight required answers

1. Raj 是否稳定优于 Johnson：FAIL；以两 seed 三指标与 paired CI 的预登记规则为准。
2. ADE 提升：见上方 seed-wise relative gain。
3. FDE4 提升：见上方 seed-wise relative gain。
4. 两 seed 一致性：见 machine summary conditions。
5. Paired uncertainty：5,000 次 prediction-origin cluster bootstrap。
6. Strong Graph / AllGraph 位置：主表将其作为 absolute reference，不冒充 matched comparison。
7. 参数：Raj 420,521，Johnson 473,187；Johnson 更多。
8. Legacy 2s->2s 约 3% gain 与 IC4 的变化：只作历史比较，不并入新判决。

## Boundaries

- test 保持 sealed。
- Raj core 未改架构、未改 j、未改数据 population 或 checkpoint rule。
- 只有当 Raj 同时胜过 matched Johnson 和外部 Strong Graph/AllGraph，才可写成超过全部已测经典参考。
- Gate C 仅在 Gate B PASS 时 ready；本报告不启动 GPT-2。
