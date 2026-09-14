# 第二研究点：多 SNR 对比、消融与轨迹预测实验记录

## 1. 实验目的与状态

本轮实验用于验证“不确定性感知双图多目标 ISAC 轨迹预测”方案在不同信噪比条件下的精度、鲁棒性与各模块贡献。全部实验已在远程服务器 `/home/js_cn/sensing` 的 NVIDIA GeForce RTX 4090 上完成，未使用本机 CPU/GPU 执行项目代码。

实验状态：

- 多 SNR 对比实验：完成；
- 六组严格消融训练：完成；
- 多 SNR 消融评测：完成；
- 代表性四目标轨迹导出：完成；
- PDF、SVG、600 dpi PNG：完成；
- 自动版面检查、人工读图、灰度检查和 PDF 字体嵌入检查：全部通过。

## 2. SNR 取值与观测噪声映射

沿用上一篇论文的四个信噪比条件：5、10、15、20 dB。为避免直接假设抽象的高斯噪声强度，本实验从第一研究点对应 SNR 的 RF 定位缓存统计位置误差，并把每轴位置 RMSE 作为第二研究点的位置观测噪声标准差。速度噪声以 20 dB 下 0.20 m/s 为基准，按照位置 RMSE 成比例缩放。

| SNR (dB) | Position noise std. (m) | Velocity noise std. (m/s) |
|---:|---:|---:|
| 5 | 0.7557 | 0.4457 |
| 10 | 0.5123 | 0.3021 |
| 15 | 0.4824 | 0.2845 |
| 20 | 0.3391 | 0.2000 |

所有模型在同一 SNR 下使用完全相同的标准正态噪声样本，测试噪声种子为 4026，从而保证逐模型比较公平。所有 checkpoint 只在接近经验标定 20 dB 的条件下训练一次，再直接测试四档 SNR，考察模型对未见噪声强度的泛化能力。

需要在论文中明确说明：位置噪声来自第一研究点 RF 定位结果的经验统计；速度噪声是按位置误差比例构造的工程近似，并非由 RF 速度估计器直接标定。

## 3. 输入、模型与输出

### 3.1 输入

每个场景输入为多个目标的带噪历史状态序列及有效目标掩码：

1. 历史位置与速度状态；
2. 目标有效性 mask，用于处理不同场景中目标数变化；
3. 多目标空间关系，由 Target Interaction GNN 构图并进行消息传递；
4. 由运动状态产生的软运动 token 分布及不确定性温度。

### 3.2 完整模型

完整模型由以下部分组成：

1. Target Interaction GNN：建模目标间空间交互，输出基础多目标预测与图节点上下文；
2. Uncertainty-aware soft motion tokenization：根据观测不确定性调节 token 温度，避免低 SNR 时过早做硬离散决策；
3. GPT-2 motion reasoning module：接收软运动 token 与图上下文，建模长时运动模式；
4. LoRA adaptation：以参数高效方式适配 GPT-2；
5. Coordinate prediction head：输出未来二维位置；
6. Auxiliary future-token loss：联合坐标损失训练，增强运动语义约束。

完整模型训练参数量为 2,505,958，总参数量为 70,643,348。

### 3.3 输出与指标

模型输出每个有效目标未来预测时域内的二维轨迹。定量指标包括：

- ADE：整个预测时域的平均位移误差；
- FDE：预测终点位移误差；
- Interaction ADE/FDE：交互场景子集上的误差；
- Collision rate 与 excess collision rate：预测轨迹的碰撞统计。

本文主图使用 ADE 和 FDE；碰撞指标保留在 JSON/CSV 中用于补充分析。

## 4. 多 SNR 对比实验

### 4.1 对比模型

- Constant Velocity；
- Independent GRU；
- LSTM；
- TCN；
- Transformer；
- Target Interaction GNN；
- Proposed Graph+LLM。

主论文折线图保留上一篇论文的 LSTM、TCN、Transformer，并加入 Target GNN 和 Proposed。CV 与 GRU 的完整结果保留在数据文件中，可用于表格或补充材料，避免主图七条曲线过度拥挤。

### 4.2 ADE/FDE 结果

表中每个单元格为 `ADE / FDE`，单位为 m。

| Model | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---:|---:|---:|---:|
| Constant Velocity | 1.5538 / 2.5660 | 1.2511 / 2.2433 | 1.2154 / 2.2059 | 1.0497 / 2.0350 |
| Independent GRU | 0.9776 / 1.8577 | 0.7970 / 1.6011 | 0.7785 / 1.5757 | 0.7015 / 1.4722 |
| LSTM | 0.9321 / 1.7505 | 0.7210 / 1.4266 | 0.7011 / 1.3983 | 0.6244 / 1.2972 |
| TCN | 0.9683 / 1.7821 | 0.7453 / 1.4609 | 0.7245 / 1.4333 | 0.6429 / 1.3310 |
| Transformer | 1.0338 / 1.8678 | 0.7879 / 1.5306 | 0.7645 / 1.5003 | 0.6743 / 1.3866 |
| Target GNN | 0.8699 / 1.5902 | 0.7081 / 1.3693 | 0.6917 / 1.3477 | 0.6229 / 1.2612 |
| **Proposed** | **0.8348 / 1.5274** | **0.6683 / 1.2899** | **0.6526 / 1.2696** | **0.5900 / 1.1935** |

Proposed 在四档 SNR 的 ADE 和 FDE 上均为最优。相对 Target GNN 的误差降低为：

| SNR (dB) | ADE reduction | FDE reduction |
|---:|---:|---:|
| 5 | 4.03% | 3.95% |
| 10 | 5.63% | 5.80% |
| 15 | 5.64% | 5.80% |
| 20 | 5.28% | 5.36% |

结果表明：GNN 首先通过目标交互建模显著优于无交互序列模型；在此基础上，软运动 token 与 LLM 时序推理进一步带来稳定的约 4%–6% 误差下降，并且这一优势覆盖全部测试 SNR。

## 5. 消融实验

### 5.1 公平训练协议

六组模型均使用相同的数据划分、随机种子、初始化流程、16 个训练 epoch、20 dB 经验标定噪声和测试噪声样本。完整模型也在该协议下重新训练，而不是直接复用旧 checkpoint。

消融项如下：

- Full model：完整模型；
- w/o uncertainty：学习式不确定性温度替换为固定温度 0.22；
- w/o soft token：软 token 分布替换为 argmax 硬 token；
- w/o graph context：去除送入 LLM token 序列和修正头的图节点上下文；
- w/o token loss：去除未来运动 token 辅助交叉熵；
- w/o LoRA：冻结 GPT-2 并去除全部 LoRA adapter。

### 5.2 多 SNR 消融结果

表中每个单元格为 `ADE / FDE`，单位为 m。

| Variant | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---:|---:|---:|---:|
| **Full model** | **0.8348 / 1.5274** | **0.6683 / 1.2899** | **0.6526 / 1.2696** | **0.5900 / 1.1935** |
| w/o uncertainty | 0.8411 / 1.5436 | 0.6708 / 1.2969 | 0.6548 / 1.2756 | 0.5913 / 1.1977 |
| w/o soft token | 0.8467 / 1.5560 | 0.6810 / 1.3188 | 0.6649 / 1.2974 | 0.5994 / 1.2164 |
| w/o graph context | 0.8690 / 1.5888 | 0.7025 / 1.3563 | 0.6856 / 1.3336 | 0.6176 / 1.2493 |
| w/o token loss | 0.8500 / 1.5683 | 0.6865 / 1.3378 | 0.6702 / 1.3161 | 0.6039 / 1.2325 |
| w/o LoRA | 0.8356 / 1.5289 | 0.6708 / 1.2966 | 0.6549 / 1.2757 | 0.5915 / 1.1986 |

### 5.3 贡献分析

1. 图上下文贡献最大。去除图上下文后，四档 SNR 的 ADE/FDE 分别恶化约 4.02%–5.15%，说明多目标交互信息是第二研究点的核心新增价值。
2. 辅助 token loss 排名第二。去除后 ADE 恶化 1.81%–2.73%，FDE 恶化 2.68%–3.72%，说明未来运动语义监督尤其有助于终点预测。
3. 软 token 的贡献稳定。硬 token 在四档 SNR 下均更差，ADE/FDE 恶化约 1.42%–2.24%，支持“保留运动模式不确定性”的设计。
4. 不确定性温度在低 SNR 更重要。5 dB 下去除该模块使 ADE/FDE 恶化 0.76%/1.06%，而 20 dB 下为 0.22%/0.35%，符合噪声越大越需要自适应软化的理论动机。
5. LoRA 的精度增益较小但方向一致。四档 SNR 下 ADE/FDE 均略优于 w/o LoRA，但最大提升不足 0.6%。论文中不宜把 LoRA 描述为主要精度来源，更合适的定位是参数高效适配机制；若版面有限，可将该消融放入补充材料。

## 6. 轨迹预测图

最终代表性测试场景为 scene index 870，选取四个相互作用目标：1401、1435、1396、1452。该场景不是按最低误差人工挑选，而是从 325 个四目标交互候选中按以下预先定义的规则自动选择：

1. Proposed 在 5 和 20 dB 的场景平均 ADE、FDE 均优于 Target GNN；
2. Proposed 的四项误差均位于全部候选场景的 25%--75% 分位区间；
3. 在满足条件的 26 个场景中，选择最接近候选中位数且目标间误差较均衡的场景。

该场景下各方法的场景平均误差如下（单位：m）：

| SNR (dB) | Method | ADE | FDE |
|---:|---|---:|---:|
| 5 | LSTM | 2.2119 | 4.6860 |
| 5 | Transformer | 2.1097 | 4.3698 |
| 5 | Target GNN | 1.2089 | 2.4315 |
| 5 | **Proposed** | **1.0275** | **1.8067** |
| 20 | LSTM | 1.8162 | 4.3128 |
| 20 | Transformer | 1.9872 | 4.5985 |
| 20 | Target GNN | 1.0463 | 2.2233 |
| 20 | **Proposed** | **0.8804** | **1.8458** |

Proposed 相对 Target GNN 的 ADE/FDE 改善为 15.01%/25.70%（5 dB）和 15.86%/16.98%（20 dB）。图中同时显示：

- Clean history：来自轨迹数据集的干净历史轨迹；分目标对比图仅保留预测起点前 6 个历史点，以浅灰细线表示；
- Ground truth：真实未来轨迹，以实线表示；
- LSTM：循环网络基线，以灰色点线和稀疏倒三角表示；
- Transformer：注意力基线，以橙色长短复合虚线和稀疏菱形表示；
- Target GNN：目标交互图基线，以绿色虚线和稀疏方形表示；
- Proposed：完整 Graph+LLM，以蓝色加粗点划线和稀疏加号形标记突出显示。

最终分目标对比图采用 `2 × 4` 小面板布局：第一行对应 5 dB，第二行对应 20 dB；四列从左到右对应同一场景中的四个目标。每个面板均以该目标的最后历史位置作为局部坐标原点，因此横纵轴表示相对位移，并且同一目标在两档 SNR 下使用相同坐标尺度。这种布局将原来每个全局面板约 24 条轨迹拆分为每个面板 6 条轨迹，使方法差异和预测终点更易辨识。

另外生成全局场景概览图，仅保留 Clean history、Ground truth 和 Proposed，用于展示四目标之间的空间关系。模型评测仍使用对应 SNR 的带噪历史输入，但两张最终轨迹图均不绘制 Noisy observations，也不显示车辆编号。TCN 与 LSTM 在该场景的空间轨迹高度重叠，因此 TCN 保留在全测试集定量折线图中，不加入定性轨迹图。分目标图使用颜色、线型和稀疏标记对方法进行冗余编码，灰度打印时仍可辨识。Proposed 在两档 SNR 下的场景平均 ADE/FDE 均为四种方法中最低。轨迹图属于按客观规则选取的典型定性示例，不能替代全测试集 ADE/FDE 的定量结论。

## 7. 图件与建议英文图注

### 7.1 多 SNR 对比图

文件：`figures/second_point_snr/fig_snr_model_comparison.pdf`

建议图注：

> Comparison of ADE and FDE under different SNR conditions. The proposed Graph+LLM model consistently achieves the lowest displacement errors across all SNR levels.

### 7.2 多 SNR 消融图

文件：`figures/second_point_snr/fig_snr_ablation.pdf`

建议图注：

> Ablation results under different SNR conditions. Removing graph context causes the largest degradation, while uncertainty-aware soft tokenization provides increasing benefits under noisier observations.

### 7.3 轨迹预测图

文件：`figures/second_point_snr/fig_snr_trajectory_prediction_comparison.pdf`

建议图注：

> Per-target trajectory comparisons at 5 and 20 dB. Rows denote SNR conditions and columns denote the four targets in the same interaction scene. Each trajectory is expressed relative to its prediction origin. Colors, line styles, and sparse markers distinguish LSTM, Transformer, Target GNN, and the proposed model; the proposed trajectories are highlighted by thicker dash-dot lines.

### 7.4 多目标场景概览图

文件：`figures/second_point_snr/fig_snr_trajectory_scene_overview.pdf`

建议图注：

> Global overview of the representative multi-target scene at 5 and 20 dB. Clean historical trajectories, future ground truth, and predictions from the proposed model are shown in the common spatial coordinate system.

所有图内文字均为英文，无图内标题，字体为 Times New Roman。主文件同时提供 PDF、SVG、600 dpi PNG 与灰度预览。

## 8. 可复现文件索引

- `comparison_snr_results.json`：全部对比模型的四档 SNR 原始结果；
- `ablation_snr_results.json`：六组消融的四档 SNR 原始结果与相对退化率；
- `snr_calibration.json`：第一研究点 RF 定位结果到第二研究点噪声的映射；
- `trajectory_examples_comparison.npz`：按客观规则选择的四目标场景，包括干净历史、真实未来，以及 LSTM、Transformer、Target GNN 与 Proposed 在 5/20 dB 下的预测轨迹；
- `trajectory_example_model_metrics.csv` 和 `trajectory_example_model_metrics.json`：代表性场景四种方法的 ADE/FDE；
- `trajectory_selection_candidates.csv`：325 个候选交互场景的误差、增益和选择分数；
- `trajectory_selection_report.json`：最终场景选择规则、候选数量和场景指标；
- `figures/second_point_snr/comparison_snr_metrics.csv`：对比长表；
- `figures/second_point_snr/ablation_snr_metrics.csv`：消融长表；
- `figures/second_point_snr/proposed_gain_vs_gnn.csv`：Proposed 相对 Target GNN 的增益；
- `prepare_multitarget_snr_figure_data.py`：绘图数据整理脚本；
- `plot_multitarget_snr_results.py`：学术图生成和视觉检查脚本；
- `export_representative_scene_baselines.py`：为固定代表性场景导出各对比方法轨迹及场景级指标；
- `run_multitarget_snr_evaluation.py`：多 SNR 对比与轨迹导出脚本；
- `run_multitarget_ablation.py`：严格消融训练脚本；
- `evaluate_multitarget_ablation_snr.py`：多 SNR 消融评测脚本。

## 9. 论文表述边界与下一步

当前可以严谨表述：在固定测试划分与统一噪声样本下，Proposed 在 5–20 dB 的 ADE/FDE 均优于所有对比模型，完整模型也优于全部消融版本。

当前不能表述为统计显著优于，因为本轮是单随机种子点估计，没有跨种子均值、标准差和显著性检验。投稿前建议增加 3–5 个随机种子，报告 mean ± std，并对 Proposed 与最强基线进行配对统计检验；届时再为折线图添加真实误差带。
