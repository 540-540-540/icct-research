# Q0 双诊断结果：LLM 核心与 Automatum QGNN 任务空间

日期：2026-09-18  
状态：0 dB / seed 2026 validation-only diagnostic，不是最终多 SNR / 多 seed 论文结果。  
Test split：未使用。

## 1. 硬约束

- QGNN 和 LLM 都是项目核心模块，不能降级为辅助模块。
- 最终正式比较应为：
  - Classical Graph Core + shared LLM Core
  - QGNN Core + shared LLM Core
- Graph/QGNN 负责核心多车关系推理。
- LLM 负责核心时序/运动 token 推理并直接决定未来轨迹。

## 2. 诊断 A：LLM 是否压缩 Graph 边际收益

### 2.1 简化 GPT-2 Q0 接口

20 epoch pilot：
- NoGraph + simplified GPT2: ADE 0.543987 / FDE 0.950401
- Edge-MPNN + simplified GPT2: ADE 0.529046 / FDE 0.953208

Pairwise 整体 ADE 略优，但 FDE 无改善。

### 2.2 无 LLM 的 Simple-GRU 对照

40 epoch：
- NoGraph + Simple-GRU: ADE 0.379483 / FDE 0.643283
- MPNN + Simple-GRU: ADE 0.451491 / FDE 0.837012

结论：
- “GPT-2 太强，已经把 Graph 的边际收益吃完”不受该 2x2 诊断支持。
- 反而当前 Simple-GRU 时序模型显著强于简化 GPT-2。
- 因此问题不是 LLM 已经过强，而是当前简化 LLM 接口没有充分发挥 LLM。

## 3. 恢复 LLM 核心地位

审计 Stage0 师兄原版后，保留以下核心机制：
- agent-centric motion tokens
- soft motion-token embedding
- graph context token
- GPT-2 future queries
- future token supervision
- GPT-2 hidden state直接决定未来轨迹

历史 Stage0 结果表明：
- Graph Motion-Token GPT-2 相对已接受 GNN 约提升 4.47% ADE / 4.57% FDE。
- 去掉 graph context 退化明显。
- 去掉 future token loss、soft token 也有稳定退化。
- LoRA 和 uncertainty temperature 的边际贡献相对小。

### Automatum Motion-Token LLM

Automatum train split agent-centric transition 统计：
- forward median ≈ 1.084 m / 0.1001 s
- forward 0.1%–99.9% ≈ [-0.844, 3.384] m
- lateral 0.1%–99.9% ≈ [-1.784, 1.773] m

采用：
- forward range [-1.0, 3.5], 31 bins
- lateral range [-1.8, 1.8], 31 bins
- vocab = 961
- 4-layer GPT-2
- LoRA rank 8
- CV 无参数基线 + GPT-2 future-query correction
- expected motion token 直接进入 coordinate head

这保证 LLM 仍是核心可学习预测器；CV 只是无参数物理起点。

## 4. Motion-Token LLM 下的 Graph 边际收益

0 dB / seed2026 / validation：

| Model | ADE | FDE | J |
|---|---:|---:|---:|
| NoGraph + Motion-Token LLM | 0.539409 | 1.089912 | 1.084365 |
| Plain MPNN + Motion-Token LLM | 0.538697 | 1.071487 | 1.074441 |
| Physics-Routed MPNN + Motion-Token LLM | 0.535965 | 1.062347 | 1.067138 |

相对 NoGraph：
- Plain MPNN: ADE +0.13%, FDE +1.69%
- Physics-Routed MPNN: ADE +0.64%, FDE +2.53%

结论：
- 更合理的 LLM Core 下，Graph 仍只有很小的全局边际收益。
- 因此 Automatum 的全局 Graph gain 小，不只是“简化 GPT-2 接口”导致。

## 5. Automatum 是否有 QGNN 可利用的任务空间

### 5.1 Plain MPNN + Motion-Token LLM 分层

相对 NoGraph + Motion-Token LLM：

- N<=3: ADE -3.32%, FDE -4.26%
- N>=4: ADE +3.09%, FDE +6.67%
- N>=5: ADE +2.75%, FDE +5.51%
- close20 = 0: ADE -8.15%, FDE -12.08%
- close20 >= 2: ADE +3.12%, FDE +6.73%
- close20 >= 3: ADE +3.02%, FDE +5.47%
- closing30 >= 2: ADE +2.73%, FDE +4.08%
- CPA < 5m: ADE +4.16%, FDE +8.37%
- CPA >= 10m: ADE -3.78%, FDE -4.72%

解释：
- Graph interaction 的价值集中在 interaction-active 场景。
- 在低交互场景，强制 message passing 明显造成负迁移。

### 5.2 Oracle 空间

逐 validation scene 在 NoGraph+LLM 与 Plain-MPNN+LLM 中取更优：
- ADE oracle gain ≈ 7.88%
- FDE oracle gain ≈ 10.49%

Plain MPNN 同时改善 ADE/FDE 的场景约 45.35%。

结论：
- Automatum 并非数学上没有 >=5% 的任务空间。
- 核心瓶颈是“模型无法可靠判断何时应使用跨车 interaction”，而不是完全没有 interaction 信息。

## 6. Physics-Conditioned Routing 诊断

加入 permutation-invariant scene-level gate：
- 输入只来自 N 和物理 edge summary
- local bypass
- graph residual
- epoch0 与 NoGraph+LLM prediction/token logits 完全一致

训练后 gate：
- global mean ≈ 0.261
- close20=0: ≈0.217
- close20>=3: ≈0.288
- CPA<5m: ≈0.282
- CPA>=10m: ≈0.238
- range ≈ [0.093, 0.340]

说明 gate 确实学到了“交互越强，更多使用 Graph”。

分层表现相对 NoGraph：
- N<=3: -1.53% / -1.68%
- N>=4: +2.50% / +6.05%
- close20=0: -1.30% / -3.11%
- close20>=3: +4.06% / +7.61%
- CPA>=10m: +0.37% / +0.50%

Routing 显著减少低交互场景的负迁移，但尚未把 oracle 空间完全转化为全局收益。

## 7. 当前科学判断

### 已得到支持

1. Automatum 存在真实的多车 interaction prediction space。
2. 该空间高度异质：高交互场景明显受益，低交互场景可能被 Graph 伤害。
3. 全局 5% 并非数学上不可能，oracle 已超过 5%。
4. 当前普通 pairwise MPNN 无法稳定利用这块空间。
5. Physics-conditioned routing 是必要的 baseline 能力之一。
6. LLM 必须使用更有任务结构的 motion-token / graph-context 接口；简化数值投影 GPT-2 不够强。

### 尚未得到支持

1. 尚无证据表明 QGNN 能超过强 classical routed baseline >=5%。
2. 尚无证据表明 higher-order interaction 是当前关键缺口。
3. 尚不能确定 Automatum 是最终最佳数据集。
4. 当前结果仅 0 dB / one seed，不能作为论文最终结论。

## 8. 对 QGNN 最终设计的直接约束

最终 QGNN 需要同时做到：

- graph structure 真正进入 quantum computation
- physics-conditioned / interaction-adaptive
- permutation equivariant
- 尽量整图 few-circuit，而非 per-edge PQC
- 在低交互场景能够抑制无益 quantum interaction
- 在高交互场景能够利用 joint multi-vehicle relations
- 与相同 Motion-Token LLM Core 对接
- 与 strong classical routed pairwise / classical higher-order baseline 公平比较

因此 C-GLOBAL（physics-conditioned global quantum graph dynamics）仍然非常契合当前诊断；
A-SUBSPACE 是否启用，仍需 classical higher-order diagnostic 决定。

## 9. 下一阶段

建议顺序：
1. 冻结本轮 Q0 诊断；
2. 等新数据集迁移对话给出候选/新分支；
3. 在 Automatum 上继续做 classical higher-order diagnostic：
   - strong routed pairwise
   - rooted triplet / PPGN-style control
4. 同时把 Motion-Token LLM Core 固定成所有正式 Graph/QGNN 共享下游；
5. higher-order 若有稳定增益，再保留 A-SUBSPACE；
6. 无论 higher-order 是否成立，C-GLOBAL 原型都值得进入下一轮，因为当前数据明确显示 physics-conditioned adaptive interaction 是关键。

