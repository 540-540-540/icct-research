# Q0 双诊断阶段结论：LLM 边际效应与 Automatum QGNN 任务空间

日期：2026-09-18
状态：0 dB、单 seed、validation-only 阶段性诊断；不是正式 test 结论。

## 1. 为什么做这两个诊断

当前项目同时出现两个疑问：

1. GPT-2+LoRA 是否已经吸收了大部分单车时序规律，从而压缩 GNN/QGNN 的边际收益？
2. Automatum 是否本身缺少足够强的多车交互，使 QGNN 无法获得全局 >=5% ADE/FDE 的任务空间？

本轮使用统一 Q0 Automatum harness，test 始终封闭。

## 2. LLM 边际效应：2x2 factorial pilot

匹配 20 epoch 数据曝光：

| Downstream | NoGraph ADE/FDE | Edge-MPNN ADE/FDE | Graph gain ADE/FDE |
|---|---:|---:|---:|
| Simple-GRU | 0.434671 / 0.725519 | 0.466372 / 0.843173 | -7.29% / -16.22% |
| GPT-2+LoRA | 0.543987 / 0.950401 | 0.529046 / 0.953208 | +2.75% / -0.30% |

Simple-GRU 与 GPT-2 侧的 trainable parameter 数量接近：
- Simple NoGraph: 532,776
- GPT2 NoGraph: 586,280
- Simple MPNN: 684,840
- GPT2 MPNN: 738,344

但 GPT-2 额外包含约 124M frozen pretrained parameters。

Simple-GRU 40-epoch 延伸：
- NoGraph best: ADE 0.379483 / FDE 0.643283，epoch 39
- MPNN best: ADE 0.451491 / FDE 0.837012，epoch 31

因此，当前证据不支持“GPT-2 太强导致 Graph 边际收益被压缩”。

恰好相反：
- Graph 在 Simple-GRU 下显著有害；
- Graph 在 GPT-2 下的 ADE 由负作用变为小幅正作用；
- 但 GPT-2+LoRA 自身的绝对 ADE/FDE 明显弱于 Simple-GRU。

阶段解释：
**当前 LLM 更像是尚未优化好的 downstream/interface，而不是性能已饱和的强组件。**

这不否定 LLM 作为项目核心模块的角色，但后续不能把“用了 GPT-2”默认当成性能优势。

## 3. Automatum QGNN 任务空间：Strong-Local 参考

为了避免弱 NoGraph 造成假 Graph gain，本轮以已经训练出的 Strong-Local residual 作为参考：
- Local ADE 0.456623
- Local FDE 0.757666

当前 Pairwise residual：
- ADE 0.499086
- FDE 0.858638

Pairwise 当前全局明显劣于 Strong-Local，因此尚未证明 cross-vehicle interaction 的独立边际收益。

## 4. 高交互子集的误差份额

但 Automatum 并不是数学上没有交互空间。

典型子集：

| 子集 | Strong-Local ADE/FDE 总误差份额 | 若只改善该子集，达到全局5%所需内部改善 |
|---|---:|---:|
| N>=4 | 53.1% / 54.1% | 9.4% / 9.2% |
| close20>=2 | 46.4% / 47.5% | 10.8% / 10.5% |
| close30>=3 | 50.3% / 51.4% | 9.9% / 9.7% |
| CPA<10m | 55.4% / 55.4% | 9.0% / 9.0% |
| N>=4 & close20>=2 | 43.8% / 45.0% | 11.4% / 11.1% |

所以高交互子集贡献了约 44%–55% 的 Strong-Local 总误差。

这意味着：
**只要一个 interaction-specialized mechanism 能在这些子集内部稳定降低约 9%–11% 的误差，而不破坏其他场景，理论上就足以推动全局约 5%。**

## 5. 当前 Pairwise 为什么仍不够

Pairwise residual 在上述高交互子集相对 Strong-Local仍普遍更差约 5%–8%。

但 Pairwise 在 893 个 validation scenes 中：
- ADE 优于 Local：32.7%
- FDE 优于 Local：33.4%
- ADE/FDE 同时优于 Local：27.0%

这 27.0% 场景贡献：
- 28.9% 的 Local ADE 总误差
- 29.5% 的 Local FDE 总误差

在这些 Pairwise 真正适合的场景里，它目前已经取得：
- ADE 内部改善 10.25%
- FDE 内部改善 15.71%

但若只靠这批场景推动全局 5%，需要内部约：
- ADE 17.29%
- FDE 16.94%

因此当前 Pairwise 已接近 FDE 所需强度，但 ADE 仍不足，而且模型不会可靠识别“什么时候该用 interaction”。

## 6. Local/Pairwise oracle

若允许逐场景 oracle 选择 Local 或 Pairwise 中更好的输出：
- ADE 可改善约 3.20%
- FDE 可改善约 5.09%

这说明：
1. interaction 的价值具有明显场景异质性；
2. Automatum 并非完全没有 interaction signal；
3. 当前主要缺陷之一可能是 **interaction gating / specialization**，而不是简单“再堆更多 message passing”。

## 7. 当前阶段判断

### 关于 LLM

暂不支持：
“LLM 已经非常强，因此出现边际效应，Graph/QGNN没有空间。”

更符合当前数据的是：
“当前 GPT-2+LoRA 不是最强 downstream，甚至明显弱于更简单的 GRU；LLM 接口或训练方式需要独立审视。”

### 关于 Automatum

暂不支持：
“Automatum 完全不适合 QGNN。”

更符合当前数据的是：
“Automatum 的高交互子集有足够大的误差质量支持全局5%的数学空间，但当前 classical pairwise interaction 没能在 Strong-Local 之上稳定利用这块空间。”

## 8. 下一步门禁

1. 不用 test。
2. 不立即冻结 QGNN 最终架构。
3. 新数据集分支并行评估更高交互载体。
4. Automatum 主线下一步应优先研究：
   - Strong-Local + matched Local-control
   - Strong-Local + gated/specialized Pairwise
   - Strong-Local + explicit Higher-order
   用来确定 interaction 的真正增量价值。
5. 若 classical higher-order 在 Strong-Local 上获得稳定、可解释增益，再决定 A-SUBSPACE 是否进入。
6. 若场景条件化的全图 interaction 更重要，则 C-GLOBAL 仍是首要 QGNN 原型候选。
7. LLM 在正式 QGNN 比较前需要单独冻结一个更可信的训练/接口协议，否则会掩盖 graph core 的真实差异。

## 9. 证据边界

所有数字均为 0 dB、单 seed、validation-only diagnostic。
不能据此声称：
- QGNN 必然达到5%；
- Automatum 最终一定保留；
- GPT-2 无价值；
- Graph interaction 无价值。

正式结论仍需多 seed、更多 SNR 和冻结后的 test 评价。

