# History-Only Quantum Mode Diagnostic — 2026-09-19

状态：第三轮 QGNN 设计前诊断，不是正式论文结果。

## 1. 目的

本诊断不回答“什么时候用量子、什么时候不用量子”。

项目约束仍然是：QGNN 全程作为核心多车交互模块。

真正要回答的是：

> 在不同 SinD 历史交互场景中，当前 RC-HQGNN 为什么有时赢经典、有时输经典；这些差异是否能支持“什么时候采用什么样的量子交互结构/强度”的下一轮设计。

因此本轮不改 QGNN、不重新训练，只分析冻结的 0 dB / seed2026 / validation 结果。

## 2. 数据与泄漏边界

- SinD validation：1880 个场景。
- 输入特征全部来自 0 dB sensing history。
- 量子内部 q_* 特征也是冻结 RC-HQGNN 对 history 的确定性响应。
- future GT 只用于离线定义“当前量子相对经典的收益标签”。
- test split 完全未读取。
定义：

[
\Delta J = J_{classical}-J_{quantum},
\qquad J=ADE+0.5FDE.
]

所以：
- \(\Delta J>0\)：当前 RC-HQGNN 在该场景优于匹配经典；
- \(\Delta J<0\)：当前 RC-HQGNN 在该场景更差。

候选 history-only 特征包括：
- 车辆密度：10/15/20/30 m 近邻对；
- 动态冲突：closing、TCPA、DCPA、active edges；
- 高阶图结构：active degree、wedge、triangle、连通分量；
- 当前 QGNN 物理风险权重与三体风险；
- 当前量子内部 pair/triple phase；
- 二体连通相关 \(K_2\) 与三体连通累积量 \(K_3\)；
- 当前 readout gate。

## 3. 全局事实

当前冻结 RC-HQGNN 的逐场景 J 胜率：

[
44.20\%.
]

平均：

[
\Delta J=-0.04286,
]

所以全局仍然输给经典。

但逐场景 oracle 说明量子与经典存在互补：
- ADE oracle 相对经典上限约 +5.31%；
- FDE oracle 相对经典上限约 +6.67%。

这些 oracle 只用于诊断，不可作为部署策略或论文量子优势。
## 4. 最重要的历史结构信号

### 4.1 高动态交互越强，当前量子越容易赢

按 30 m 内 closing>0.5 m/s 的车辆对数分层：

| 子集 | N | 平均 ΔJ | 量子 J 胜率 |
|---|---:|---:|---:|
| closing < 10 | 1130 | -0.0678 | 41.3% |
| closing 10–14 | 635 | -0.0185 | 44.9% |
| closing >= 15 | 115 | **+0.0674** | **68.7%** |

这进一步确认：当前量子正信号不是一般密度，而是集中在强动态多车交互。

### 4.2 高阶图结构也呈同方向

- active wedges >=35：N=416，平均 ΔJ=+0.0246，量子胜率 57.0%；
- active triangles >=6：N=391，平均 ΔJ=+0.0296，量子胜率 57.8%。

因此“多个关系围绕同一目标形成联合结构”比简单 pair count 更接近量子有利区域。
## 5. 关键新证据：K3 不是单纯的“车多效应”

在全部场景中：

- Spearman(\(K_2\), ΔJ) = +0.112；
- Spearman(\(K_3\), ΔJ) = **+0.223**。

\(K_3\) 与量子收益的联系明显强于 \(K_2\)。

\(K_3\) 最高 20% 的 376 个场景：
- 平均 ΔJ = **+0.0356**；
- 量子 J 胜率 = **59.3%**。

更重要的是，为排除“场景只是更拥挤”的解释，将场景按 30 m 内近车数量分成 5 个密度层，在每一层内部再比较 \(K_3\) 高/低两半。

5 个密度层中，高 \(K_3\) 组相对低 \(K_3\) 组的 ΔJ 差分别为：

- +0.0161
- +0.0287
- +0.1107
- +0.0161
- +0.0793

五层全部为正。

按各层样本数加权：

[
\boxed{\Delta J_{high-K3}-\Delta J_{low-K3}=+0.0505}
]

分层 bootstrap 95% 区间：

[
\boxed{[+0.0325,+0.0683]}
]

因此，在当前 validation 诊断中，“更强的三体量子响应对应更好的量子相对收益”不能仅由车辆密度解释。
## 6. 这种关系不是单一城市偶然

两个 SinD scene 分开计算：

- scene 0：Spearman(\(K_3\), ΔJ)=+0.182；
- scene 1：Spearman(\(K_3\), ΔJ)=+0.311。

方向一致。

这仍不是独立测试结论，但至少说明当前关联没有完全被某一个录制场景单独驱动。

## 7. 重要反证：简单 history router 并不好用

为了检查“能不能直接用若干历史物理量预测哪一场量子会赢”，做了严格的 5 个时间块交叉验证，并在 held-out 时间块边界增加 39-frame embargo。

结果：

### physics-only
- Logistic AUC = 0.504
- 非线性 HGB AUC = 0.503
- ΔJ 回归 Spearman = -0.008

### physics + frozen quantum internal signals
- Logistic AUC = 0.493
- 非线性 HGB AUC = 0.501
- ΔJ 回归 Spearman = +0.012

即：**没有证据支持用一个外部简单分类器/阈值稳定预测“这一场量子会不会赢”。**

这与项目方向一致：下一步不应做“经典/量子开关”，也不应写死“closing>=15 就开 ZZZ”。
时间块相关性仍有明显非平稳性：某些 block 中 closing 或 K3 与 ΔJ 的关系更强，另一些 block 变弱甚至局部反向。

合理解释包括：
- 当前 RC-HQGNN 的相对优势标签本身噪声较大；
- 不同道路片段的交互语义不同；
- 单个统计量无法描述完整场景；
- 当前固定量子结构并未形成稳定的内部模式分工。

所以应学习连续的量子内部自适应机制，而不是外部离散规则。

## 8. 当前 gate 没有解决这个问题

当前 v3 readout gate 是量子结果出来后的输出门，不是量子线路内部的交互模式选择器。

诊断：

- closing<10：gate mean = **0.854**
- closing>=15：gate mean = **0.739**
- Spearman(gate, ΔJ) = -0.082
- Spearman(gate × readout RMS, ΔJ) ≈ **0.008**

也就是说，当前 gate 并没有把“量子在复杂高阶场景更有价值”转化成稳定的有效贡献强度。

更关键的是：它发生在量子演化之后，粒度过粗，无法决定量子内部到底采用多少 pair / triplet / channel / depth。
## 9. 对下一轮 QGNN 设计的约束

当前证据最支持的方向不是“继续堆 qubit / depth / channel”，而是：

[
\boxed{\text{固定高阶量子图} \rightarrow \text{场景自适应的量子交互模式}}
]

但量子始终使用，不做 classical/quantum switch。

下一轮值得让 GPT-6 Pro 重点研究：

1. **量子阶数自适应**
   - history-conditioned \(\lambda_2(s)\) 控制 ZZ；
   - history-conditioned \(\lambda_3(s)\) 控制 ZZZ；
   - 目标是让简单场景偏 pair-like quantum interaction，复杂场景加强 higher-order quantum interaction。

2. **关系级自适应**
   - 不是所有 pair / triple 使用相同重要性机制；
   - 让物理关系和场景上下文共同决定哪些 \(ZZ_{ij}\)、\(ZZZ_{ijk}\) 值得更强量子耦合。

3. **channel specialization**
   - 当前 4 个 quantum channel 基本同构；
   - 可研究让不同 channel 分工承担不同交互阶数、时间尺度或风险模式，再由 history 连续调节。

4. **layer/depth 内部自适应**
   - 不必简单增加 D；
   - 可以让不同 quantum round 的 pair/triple 比例随场景变化。

5. **利用 K3 作为内部状态，而不是事后 oracle**
   - K3 与量子收益存在密度控制后的正关联；
   - 可研究前一轮量子相关性如何调节后一轮量子演化，实现 quantum-to-quantum adaptive interaction。
## 10. 公平比较要求保持不变

如果 classical counterpart 也具备 scene-adaptive pair/triplet weighting，则必须保留。

最终核心仍然是：

[
\text{Adaptive Classical Higher-Order Interaction}
\quad vs \quad
\text{Adaptive Quantum Higher-Order Interaction}.
]

不能通过给量子模型独占额外未来信息或削弱 classical baseline 获胜。

## 11. 当前结论

本诊断支持以下判断：

1. 当前 RC-HQGNN 全局仍输经典，不能宣称量子优势。
2. 量子正信号集中在高动态、高阶联合交互场景。
3. 三体连通量子响应 K3 与相对量子收益存在比 K2 更强的关系。
4. K3 的正关系在控制车辆密度后仍存在，并在两个 SinD scene 中同方向。
5. 当前输出 gate 不是有效的量子模式调节器。
6. 简单 history-only 外部 router 不能稳定预测量子胜负，因此不应走“规则切换”路线。
7. 下一轮最合理问题是：
   **QGNN 全程使用量子，但根据历史场景连续、自适应地改变二体/三体/通道/轮次的量子交互模式。**

这足以作为第三轮 GPT-6 Pro 的设计起点，但还不足以证明任何具体 adaptive quantum architecture 一定能获得全局优势。

## 12. 机器可读结果

- 完整逐场景诊断：reports/qgnn/history_only_quantum_mode_diagnostic_20260919.json
- 紧凑摘要：reports/qgnn/history_only_quantum_mode_summary_20260919.json
- 诊断脚本：scripts/diagnose_qgnn_history_modes.py

test_set_used = false。

## 13. 统计解释注意

SinD validation 窗口存在明显时间重叠，因此本文中的 Spearman、分层均值和 bootstrap 区间只用于结构诊断与方向筛选。
尤其 bootstrap 95% 区间不能被解释为基于独立样本假设的正式统计显著性结论。正式论文若需要显著性分析，应按时间块/轨迹组进行重采样或使用独立 test 协议。
