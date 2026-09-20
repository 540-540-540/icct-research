# QGNN 历史实验统一审计与主线决策 — 2026-09-20

## 0. Executive decision

本审计统一重读 ICCT 项目当前服务器 `/home/dell/YrM/ICCT` 的 QGNN / Classical 历史实验、正式 full-train 结果、4096-window pilots、机制消融和分层结果。

核心结论：

1. **2026-09-20 的 TRC/TO finalists 不是项目第一次 full-train。**
   在它们之前，RC-HQGNN 与 Raj-style Candidate A 已经在 SinD full 15,802 train / 1,880 validation / 0 dB / seed2026 / 20 epochs 上完成正式 paired full-train。
2. **截至本审计，没有任何 QGNN 在 SinD full-data 强 classical 对照下建立稳定的 overall quantum advantage。**
3. **Raj-style Weighted Multi-j 是目前唯一 full-data 上达到“与强 classical 基本打平”的 Quantum 路线。**
   RC-HQGNN 稳定落后约 4%；TRC-QGNN 和 TO-JQGNN 分别出现约 15% 和 10% 的 J 劣势。
4. **Raj 在 4096-window、两个 seed 上出现重复的约 2–3% 正信号，但该优势在 full 15,802 seed2026 中缩小到近零。**
   因而低数据量 / inductive-bias / sample-efficiency 是仍值得检验的假设，但不能写成已建立结论。
5. **SinD 的 higher-order interaction 需求本身是成立的，但它不是 quantum-exclusive bottleneck。**
   Q0 的经典 pair+triplet 已经稳定利用该信号并显著优于 no-graph / pairwise。
6. **“Quantum mechanism 被模型使用”不等于“Quantum 比 Classical 更好”。**
   RC-HQGNN 删除 ZZ/ZZZ 或 graph context 会显著退化，但 matched Classical 仍在 overall 上更优。
7. **高动态场景的局部量子正信号是历史上最一致的量子特征之一，但不是所有架构都能复现。**
   RC 和 Raj 在高 closing 子集有局部正信号；TRC/TO 在同一分层下仍全部落后。
8. **当前不建议立即再设计第三个新的 QGNN。**
   应先把 Raj-style 作为唯一 full-data survivor 做一个最终多 seed / data-size 决策门槛；若仍不能建立稳定优势，再决定论文是否转向低数据量归纳偏置或接受 quantum-specific advantage 未建立。

---

## 1. 证据层级

### Tier A1 — 当前最高价值：SinD full-data matched paired experiments

共同基础：

- SinD；
- 0 dB；
- seed2026；
- train 15,802；
- validation 1,880；
- 20 epochs；
- batch 32；
- prediction test closed。

其中 RC-HQGNN 和 Raj full 使用相同 train-index SHA：

`18eeffab9a311866f3cb67c352e9882d86e7068fb5527d82cd2d15d100ef73ca`

TRC/TO finalists 也使用该 full-train index SHA。

注意：不同架构家族的 QGNN→GPT-2 reader / Token interface 不完全相同，因此**最强因果证据是每一对 Q vs matched C 的内部比较**，跨家族绝对 ADE/FDE 排名只能作项目级参考。

### Tier A2 — SinD full-data task diagnostic

Q0 的 no-graph / pairwise / routed pairwise / pair+triplet 也是完整 15,802 / 20 epochs。

但该轮使用 0 dB 专用临时 Motion Token 刻度和更早的 LLM interface，因此适合证明：

- surrounding vehicles 有价值；
- pairwise interaction 有价值；
- triplet/higher-order 有额外价值；

不应与后续 frozen Token 架构直接做绝对数值排行。

### Tier B — 4096 / small paired pilots

适合：

- 快速筛选 architecture family；
- 检验正负信号是否重复；
- 研究 data-limited inductive bias。

不能替代 full-data 结论。

### Tier C — ablation / subset / mechanism diagnostics

适合解释：

- 模型是否真的依赖某个 quantum path；
- 哪类场景存在局部信号；
- feedback / routing / readout 是否有贡献。

不能单独证明 Quantum > Classical。

### Tier D — Automatum / f01d / old-QGNN historical evidence

只用于：

- 工程历史；
- 编码、qubit、readout、优化等机制经验。

不进入当前 SinD 性能排名。

---

## 2. Full-data matched pair 总表

正的 Quantum gain 表示 Quantum 更好。

| Family | Quantum | Classical | Q ADE | C ADE | Q FDE | C FDE | Q J | C J | Q gain ADE | Q gain FDE | Q gain J |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RC-HQGNN | RC-HQGNN | Adaptive Classical | 0.510945 | 0.491465 | 1.126051 | 1.079289 | 1.073970 | 1.031109 | -3.964% | -4.333% | -4.157% |
| Raj | Weighted Multi-j Quantum | Multi-j JohnsonGIN | 0.481259 | 0.482314 | 1.051173 | 1.049728 | 1.006845 | 1.007179 | +0.219% | -0.138% | +0.033% |
| Finalist A | TRC-QGNN | RTCN-128 | 0.576768 | 0.498817 | 1.180421 | 1.035420 | 1.166978 | 1.016527 | -15.627% | -14.004% | -14.801% |
| Finalist B | TO-JQGNN | TR-TGN | 0.613655 | 0.547933 | 1.260698 | 1.167918 | 1.244004 | 1.131892 | -11.995% | -7.944% | -9.905% |

### 直接结论

- Raj 是唯一一个 full-data overall 近中性的 Quantum family。
- RC 是明确但中等程度的负结果。
- TRC/TO 是明显负结果，当前没有理由仅靠增加 seed 把它们升级为主线。
- “更 quantum-native / 更严格可测量 / 更物理”的设计没有自动带来更好的预测结果。

---

## 3. Q0：higher-order task signal 是真的，但不是 Quantum 专属

完整 Q0 / SinD / 0 dB / seed2026：

| Classical interaction | ADE | FDE | J |
|---|---:|---:|---:|
| No graph | 0.524150 | 1.063804 | 1.056052 |
| Pairwise MPNN | 0.502264 | 1.030864 | 1.017696 |
| Routed pairwise | 0.501781 | 1.020530 | 1.012046 |
| Pair + triplet | **0.498218** | **1.004633** | **1.000534** |

相对 no-graph：

- pairwise ADE/FDE 改善约 4.18% / 3.10%；
- pair+triplet ADE/FDE 改善约 4.95% / 5.56%。

相对普通 pairwise，pair+triplet 仍额外改善：

- ADE ~0.81%；
- FDE ~2.55%。

高 closing 场景增益更大。

因此：

> ICCT 的核心任务确实需要 multi-neighbor / higher-order reasoning，但现有经典模型已经能够直接利用该信号。不能把“triplet 有价值”推导成“Quantum 必须有优势”。

---

## 4. Raj：唯一重复正信号与 full-data collapse

### 4096 train

Seed2026：

- Q ADE/FDE/J = 0.516458 / 1.102781 / 1.067849
- C ADE/FDE/J = 0.525952 / 1.129782 / 1.090843
- Q gain J = +2.108%

Seed2027：

- Q = 0.530895 / 1.148234 / 1.105012
- C = 0.550423 / 1.189891 / 1.145368
- Q gain J = +3.523%

两 seed 平均约：

- ADE +2.70%
- FDE +2.96%
- J +2.83%

### Full 15,802

Seed2026：

- Q J = 1.006845
- C J = 1.007179
- Q gain J = +0.033%

ADE 略胜，FDE 略负，primary gate 失败。

### 不能简单解释成“Classical 多训练了更多 updates”

4096 / B32 / 20 epochs 共 2560 updates。

Full train 每 epoch 494 steps；epoch5 约 2470 updates，与 small 20 epoch 的 update 数近似：

- Full Raj-Q epoch5: J = 1.042177
- Full Raj-C epoch5: J = 1.035046

即在接近相同 update count 时，full-data Classical 已略领先。

所以：

- “full classical 只是靠 9880 updates 才追上”不是充分解释；
- 仍可能存在 data coverage、subset variance、regularization/inductive bias、optimization trajectory 的交互；
- 必须用 nested subset + 固定 model seed / dataset seed + matched-update 设计才能真正检验 sample-efficiency 假设。

---

## 5. 高动态分层：不同 Quantum family 的行为并不相同

统一按 validation 中 `closing_pairs_30m_gt_0p5` 分层。

### RC-HQGNN

历史正式报告：

| Stratum | Quantum gain ADE | Quantum gain FDE |
|---|---:|---:|
| closing <10 | -6.29% | -7.25% |
| closing ≥10 | -0.73% | -0.19% |
| closing ≥15 (140) | **+4.29%** | **+6.35%** |

说明 RC 的 quantum inductive bias 可能更适合极高动态多车冲突，但在占多数的普通场景明显伤害整体性能。

Round3 feedback 后，高动态 final>=15 仍有约 +1.90% ADE / +4.08% FDE 的 Q over C 正信号，但 overall gap 仍未关闭。

history-only router 对“何时 Quantum 赢”的预测：

- Logistic AUC ~0.50
- HGB AUC ~0.50

因此不能简单做一个可部署的 Q/C hard switch 来兑现事后 oracle。

### Raj full

逐窗口 paired full-data：

| Stratum | n | Q gain ADE | Q gain FDE | Q win-rate ADE | Q win-rate FDE |
|---|---:|---:|---:|---:|---:|
| all | 1880 | +0.219% | -0.138% | 49.8% | 49.8% |
| <10 | 1054 | +0.360% | +0.604% | 52.3% | 53.8% |
| 10–14 | 686 | -0.295% | -1.918% | 45.5% | 43.3% |
| ≥15 | 140 | **+1.362%** | **+1.710%** | 52.9% | 51.4% |

Raj 的局部优势比 RC 更温和，也更接近整体中性。

### TRC finalist

| Stratum | Q gain ADE | Q gain FDE |
|---|---:|---:|
| all | -15.63% | -14.00% |
| <10 | -17.83% | -17.51% |
| 10–14 | -11.50% | -8.04% |
| ≥15 | -16.63% | -12.55% |

### TO finalist

| Stratum | Q gain ADE | Q gain FDE |
|---|---:|---:|
| all | -12.00% | -7.94% |
| <10 | -12.57% | -9.68% |
| 10–14 | -10.51% | -4.87% |
| ≥15 | -14.09% | -7.79% |

因此：

> TRC/TO 的失败不是“overall 被容易场景稀释，但高动态其实赢”。它们在高 closing 也明显落后，与 RC/Raj 的局部量子正信号属于不同失败模式。

---

## 6. Optimization / capacity：不是一个统一原因

| Pair | Q core params | C core params | Q/C ratio | Q best epoch | C best epoch | Q last train loss | C last train loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| RC | 67,357 | 323,138 | 0.208 | 17 | 13 | 0.3569 | 0.3539 |
| Raj | 126,535 | 179,201 | 0.706 | 9 | 9 | 0.3381 | 0.3429 |
| TRC | 7,052 | 203,140 | 0.035 | 16 | 20 | 0.5468 | 0.4330 |
| TO | 6,722 | 30,167 | 0.223 | 19 | 18 | 0.7104 | 0.5393 |

### RC

- Quantum best epoch 更晚；
- 末期 train loss 已和 Classical 很接近；
- validation 仍有约 4% gap。

所以 RC 不能简单归因于“没优化到”。

### Raj

- 两方 best epoch 相同；
- Quantum 最终 train loss 甚至略低；
- validation 近乎完全打平。

这是最典型的“函数族/归纳偏置差异很小，full-data classical 可以追平”的结果。

### TRC / TO

- Quantum 最优都很晚；
- 最终 train loss 仍明显高于 matched Classical；
- core 参数量极小，尤其 TRC 只有 RTCN 的约 3.5%。

这更像：

- 过强的结构约束；
- 信息/readout bottleneck；
- optimization / effective capacity 不足；

而不是“Quantum 已经拟合训练集但泛化输掉”。

因此不能把四条路线的失败都归到一个统一的 barren plateau / overfit / underfit 标签。

---

## 7. Mechanism-level common lessons

### 7.1 Higher-order interaction 是任务需求，不是 Quantum advantage

Q0 classical pair+triplet 已经证明 strong classical 能直接利用 multi-neighbor joint interaction。

因此后续 Quantum 方案必须超过：

- strong higher-order Classical；
- functional matched Classical；

而不是只超过 pairwise GNN。

### 7.2 Quantum path 被使用 ≠ Quantum path 有比较优势

RC 的 post-hoc：

- 去全部 ZZ/ZZZ：显著退化；
- 去 ZZZ：退化；
- 清 graph context：严重退化。

说明 model 确实依赖 quantum interaction path。

但 matched Classical overall 仍更好。

所以：

> “去掉量子门会掉点”只能证明该模型使用了量子路径，不能证明量子计算是更好的实现方式。

### 7.3 低数据量 Quantum inductive bias 是目前最值得保留的假设

Raj 是唯一出现：

- 4096 two-seed repeated positive；
- full-data near-tie；

的 family。

这与“受约束函数族在少数据时正则化更强、full-data classical flexibility 追上”的解释一致，但还没有被严格隔离验证。

### 7.4 局部 hard-scene advantage 存在，但不是普适 Quantum 性质

- RC：高 closing 正信号强；
- Raj：高 closing 小幅正信号；
- TRC/TO：高 closing 也明显负。

因此不能写成：

> “Quantum 天然适合复杂高动态场景。”

更准确的是：

> 某些具体 quantum inductive bias 在高动态场景出现过可重复的局部信号。

### 7.5 更“quantum-pure”的设计没有带来更强 prediction

项目中表现最好的 full-data Quantum 仍是工程化 Raj-style Candidate A，而更严格采用 measurable readout、合法 unitary / joint-register / configuration construction 的 TRC/TO 明显更差。

这**不能证明**非物理操作本身带来收益，但它提醒：

- direct amplitude-like freedom；
- richer classicalized readout/update；
- less constrained latent representation；

可能对实际 trajectory prediction 很重要。

不能假设“更贴近硬件量子电路”会自动提高 ML 性能。

### 7.6 当前 small-register Quantum 没有计算加速论据

这些线路在 4090 上可精确 statevector 模拟，强 classical counterpart 也并不昂贵。

当前唯一合理的优势主张空间是：

- statistical inductive bias；
- sample efficiency；
- task-specific representation；

而不是 computational speedup。

---

## 8. 哪些历史结果仍然有参考价值

### 可以直接作为当前主结论证据

- RC full pair；
- Raj full pair；
- TRC/RTCN formal pair；
- TO/TR-TGN formal pair。

### 可以作为 task/mechanism evidence

- Q0 full no-graph / pair / triplet；
- RC / Round3 high-closing strata；
- Raj 4096 two-seed；
- Raj fidelity / adjacency / reupload isolation；
- representation bottleneck diagnostics。

### 只能作为历史工程经验

- Automatum / f01d early QGNN；
- QGAT；
- 旧 SNR / 旧 frontend 的绝对 ADE/FDE。

---

## 9. 当前 strongest evidence 排序

### Full-data Quantum

按“是否超过自己的 matched Classical”而不是绝对 ADE：

1. **Raj Weighted Multi-j**：near-tie，唯一 survivor；
2. RC-HQGNN：约 -4%；
3. TO-JQGNN：约 -10%；
4. TRC-QGNN：约 -15%。

### Positive evidence

1. Raj 4096 two-seed +2–3%；
2. RC high-closing subset +4–6%；
3. Raj full high-closing +1–2%。

### Strongest Classical task evidence

Q0 pair+triplet：

- ADE 0.498218
- FDE 1.004633
- J 1.000534

其协议使用早期临时 Token interface，因此不与后续绝对排行混用；但它强烈证明 higher-order signal 并不要求 Quantum 才能利用。

---

## 10. 主线决策

### 不建议

- 立即设计第三个新 QGNN；
- 给 TRC/TO 继续补 2027/2028 seed；
- 再加 router/controller/feedback 试图救整体；
- 弱化 Classical baseline；
- 因为 Quantum 高 closing 局部赢就绕开 overall。

### 建议把 Raj-style 设为唯一 survivor，进入最终决策门槛

原因：

- 4096 two-seed 重复正信号；
- full-data 至少达到 near-tie；
- high-closing full-data 仍保留小幅正信号；
- 训练/验证曲线不支持“只是优化失败”；
- 它是目前所有 Quantum family 中唯一同时具备 low-data positive + full-data neutral 的路线。

---

## 11. 下一步终局实验建议

### Gate A — Raj full-data multi-seed confirmation

补 full 15,802：

- seed2027；
- seed2028；

Quantum 和 JohnsonGIN paired、相同协议。

建议预注册：

- macro mean ADE 和 FDE 都不得明显退化；
- J mean gain 至少 +1% 才支持 full-data quantum-advantage narrative；
- 至少 2/3 seed Quantum J 为正；
- 同时报告 high-closing strata，但不能替代 overall gate。

如果三 seed full-data 仍 near-zero / mixed-sign：

> 停止继续搜索 full-data Quantum superiority。

### Gate B — 若 Gate A 不支持 full-data advantage，再检验 low-data sample-efficiency

采用 nested train subsets：

- 1k / 2k / 4k / 8k / 15.8k；
- 同一 nested 数据；
- dataset seed 与 model seed 分离；
- Q/C 相同 initialization policy；
- 同时报告 matched-epoch 和 matched-update。

目标不是重新调架构，而是回答：

> Raj 的 4096 正信号到底是稳定 sample-efficiency inductive bias，还是 subset/seed/optimization 偶然。

如果该曲线显示 Quantum 只在低数据量稳定领先，可以把论文主张收缩为：

> data-limited quantum inductive bias / sample efficiency

而不是 full-data universal superiority。

### Gate C — 若 low-data 也不稳定

接受当前项目结论：

> 在当前 SinD + ISAC + GPT-2 任务上，尚未验证 quantum-specific predictive advantage；higher-order interaction 本身有价值，但 strong classical models 已能充分利用。

这比继续无限改 QGNN 更有科学可信度。

---

## 12. 当前状态一句话总结

> **项目历史已经从“还没找到正确 QGNN”转变为“多类 Quantum inductive bias 已被 full-data 公平检验；Raj 是唯一未被明显否定的 survivor，但其优势尚未超过 full-data near-tie”。**

下一步应做 Raj 的终局稳定性检验，而不是继续扩大架构搜索空间。

---

## 13. Primary evidence paths

- `docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md`
- `docs/qgnn/FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md`
- `docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md`
- `docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md`
- `docs/qgnn/ROUND4_RAJ_FULLTRAIN_FINDINGS_20260919.md`
- `docs/qgnn/finalists/QGNN_FINALISTS_FORMAL_RESULTS_20260920.md`
- `reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`
- `reports/qgnn/r2_final_full_quantum_0db_seed2026/summary.json`
- `reports/qgnn/r2_final_full_classical_0db_seed2026/summary.json`
- `reports/qgnn/round4_raj_fulltrain_quantum_0db_seed2026/summary.json`
- `reports/qgnn/round4_raj_fulltrain_johnson_0db_seed2026/summary.json`
- `reports/qgnn/finalists/formal_0db_seed2026_*/summary.json`
