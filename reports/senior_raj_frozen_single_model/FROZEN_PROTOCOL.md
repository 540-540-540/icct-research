# Raj QGNN + Motion-Token GPT-2 单模型冻结方案

冻结日期：2026-09-22

选择依据：仅验证集完成方案选择；冻结后按用户授权一次性打开最终测试集。

冻结范围：不使用蒸馏、不使用模型集成、不使用经典/量子路由。

## 冻结模型

每个随机种子保留一个独立的 Raj QGNN + Motion-Token GPT-2 模型：

| seed | ADE (m) | FDE (m) | 图骨干检查点 | 最终检查点 |
|---:|---:|---:|---|---|
| 2026 | 0.522679 | 1.051946 | `results/senior_raj_frozen_single_model/seed2026/raj_qgnn.pt` | `results/senior_raj_frozen_single_model/seed2026/graph_motion_token_gpt2.pt` |
| 2027 | 0.526549 | 1.059038 | `results/senior_raj_frozen_single_model/seed2027/raj_qgnn.pt` | `results/senior_raj_frozen_single_model/seed2027/graph_motion_token_gpt2.pt` |
| 2028 | 0.521567 | 1.054483 | `results/senior_raj_frozen_single_model/seed2028/raj_qgnn.pt` | `results/senior_raj_frozen_single_model/seed2028/graph_motion_token_gpt2.pt` |

三种子均值（样本标准差）：

- ADE：`0.523598 ± 0.002615 m`
- FDE：`1.055155 ± 0.003593 m`

师兄经典 GNN + LLM 三种子均值为 ADE `0.547848 m`、FDE `1.103499 m`。冻结方案相对提升：

- ADE：`4.4264%`
- FDE：`4.3809%`

目标 5% 尚未达到；冻结值不得写成 5% 或量子优势已被最终证明。

师兄最初复现实验中的 Graph + GPT-2 在 validation 集为 ADE `0.547997 m`、FDE `1.104733 m`；当前冻结方案三种子均值相对它改善 ADE `4.4524%`、FDE `4.4877%`。同一师兄模型的 `0.591153 / 1.195073` 是 test 集结果，因此只能与下节冻结后得到的 QGNN test 指标比较，不能与 QGNN validation 指标混算。

## 冻结后最终测试

方案冻结后，按用户明确授权一次性评估 test 集；不根据 test 结果调参或更换模型。三种子 test 结果为：

| seed | ADE (m) | FDE (m) |
|---:|---:|---:|
| 2026 | 0.567060 | 1.148811 |
| 2027 | 0.564863 | 1.146117 |
| 2028 | 0.564021 | 1.142935 |

test 三种子均值（样本标准差）：

- ADE：`0.565314 ± 0.001569 m`
- FDE：`1.145954 ± 0.002941 m`

与师兄原 Graph + GPT-2 的同 test 指标 `0.591153 / 1.195073` 相比，ADE 改善 `4.3709%`，FDE 改善 `4.1101%`。test 量子关闭消融均值为 `0.621737 / 1.277461`；启用量子核心相对改善 ADE `9.0751%`、FDE `10.2944%`。

## 量子核心核验

将同一最终模型的 `quantum_scale` 置零后，三种子均值为：

- ADE：`0.582534 m`
- FDE：`1.194435 m`

启用量子核心相对量子消融改善 ADE `10.1172%`、FDE `11.6607%`。该核验说明冻结模型的输出实质依赖 Raj 量子核心，但不替代与匹配经典基线的比较。

## 固定配置

- Raj：加权 `j=2 + j=3` 子集核心，3 轮量子演化，`quantum_scale=0.05`
- 图结构：`classical_layers=2`，量子交互在经典图层之前注入
- 投影：hidden `128`，depth `1`
- LLM：本地 GPT-2，4 层，LoRA rank `8`
- GPT-2 基座冻结，图骨干在最终 LLM 训练中冻结
- 使用 64 维量子坐标特征
- LLM 适配层从头训练 32 轮，学习率 `3e-4`
- metric-aligned loss，token weight `0.035`
- 输入噪声：位置 `0.35`、速度 `0.20`

图骨干训练沿用逐段验证集选优。seed 2026 和 2028 的最终骨干包含第三段 `60` 轮、学习率 `5e-5` 的续训；seed 2027 在同段未刷新验证最佳点，因此冻结上一段检查点。三个种子均执行了相同的候选协议和检查点选择规则。

## 排除项

下列实验不属于冻结方案：蒸馏、双模型输出平均、权重平均、`j=4` 分支、4/5 轮量子深度、transition coherence、可学习量子尺度、量子辅助未来头、独立量子残差头、量子上下文投影及推理期尺度校准。
