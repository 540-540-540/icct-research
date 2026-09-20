# ICCT Raj-PennyLane + Residual 主线总任务书（2026-09-20）

## 0. 文档用途

本文件是 ICCT 项目 QGNN 主线从 2026-09-20 起的执行权威。执行 agent（当前为 GPT-5.6 Luna）必须先完整读取本文件，再读取指定代码和历史证据。

服务器正式项目：

`/home/dell/YrM/ICCT`

正式分支：

`qgnn`

服务器状态优先于本地与 GitHub。禁止用 stale remote 覆盖服务器。

当前执行门：

> **P1：Raj-PennyLane-native interaction core 实现与 correctness/preflight。**

P1 完成后必须停止，等待核心审查；不得自行进入 Residual LLM、4096/full training、multi-seed、multi-SNR 或 prediction test。

---

## 1. 当前科学结论与主线收缩

历史统一审计：

`docs/qgnn/QGNN_HISTORICAL_EXPERIMENT_UNIFIED_AUDIT_20260920.md`

当前 full-data 证据：

| Family | Quantum J | Matched Classical J | Quantum 相对 J |
|---|---:|---:|---:|
| Raj Weighted Multi-j | 1.006845 | 1.007179 | +0.033%（近乎打平） |
| RC-HQGNN | 1.073970 | 1.031109 | -4.157% |
| TO-JQGNN | 1.244004 | 1.131892 | -9.905% |
| TRC-QGNN | 1.166978 | 1.016527 | -14.801% |

Raj 4096-window 证据：

- seed2026：Quantum J gain +2.108%
- seed2027：Quantum J gain +3.523%
- 两 seed 平均约 +2.833%

因此当前唯一保留的 QGNN 主线是：

`Raj Weighted Multi-j`

TRC / TO / RC 保留为历史证据，不作为本轮新主线。

---

## 2. 三阶段总路线

### P1 — Raj higher-order → PennyLane-native Quantum core

必须保留 Raj 最有证据的 inductive bias：

- `j=2 + j=3` higher-order subset structure
- weighted Johnson / occupation-transfer interaction
- history-only continuous physical conditioning
- shared parameters
- multi-j fusion
- per-agent interaction representation

同时将正式 Quantum core 改为真正由 PennyLane 表达。

### P2 — Residual QGNN–LLM interface（P1 通过后才能执行）

最终结构：

`Y_hat = Y_CV + ΔY_self(LLM) + g × ΔY_interaction(Raj-QGNN)`

要求：

- Self LLM 只看目标车自身历史；
- Raj-QGNN 独占多车辆交互信息；
- 20 个 future query 分别读取 interaction representation；
- 不允许 LLM 通过旁路访问其他车辆关系；
- 支持 Self → Interaction → Joint 三阶段训练。

### P3 — 公平实验与终局 Gate（P2 通过后）

按顺序：

1. correctness / profile / smoke
2. 4096 screening
3. full 15,802
4. multi-seed
5. 最后才 multi-SNR / prediction test

P3 不属于当前执行范围。

---

# P1 当前执行任务

## 3. 开始前必须按顺序读取

### 3.1 当前主线证据

1. `docs/qgnn/QGNN_HISTORICAL_EXPERIMENT_UNIFIED_AUDIT_20260920.md`
2. `docs/qgnn/ROUND4_RAJ_FULLTRAIN_FINDINGS_20260919.md`
3. `docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md`
4. `docs/qgnn/ROUND4_RAJ_PAPER_FIDELITY_AUDIT_20260919.md`

### 3.2 当前 Raj 实现

5. `prediction/qgnn_paper_native/raj_paper.py`
6. `prediction/qgnn_paper_native/raj_subset.py`
7. `prediction/qgnn_paper_native/raj_joint.py`
8. `prediction/qgnn_paper_native/raj_paper_math.py`
9. `prediction/qgnn_paper_native/model.py`

### 3.3 项目已有真正 PennyLane 实现

10. `prediction/quantum.py`
11. `prediction/check_pennylane.py`
12. `ENVIRONMENT.md`
13. `requirements.txt`
14. `requirements-lock.txt`

不要仅根据本任务书猜实现；必须先读代码。

---

## 4. 当前 full Raj 的身份必须明确

历史最好 Raj：

`RajWeightedMultiJQGNNCore`

核心结构：

- j=2 branch
- j=3 branch
- continuous subset physics features
- graph-conditioned weighted Johnson mixing
- D=6, k=3 embedding idea
- repeated interaction/evolution/re-upload
- per-agent incidence readout
- MultiJ fusion

历史 full-data：

Quantum：
- ADE 0.481258665
- FDE 1.051172996
- J 1.006845163

JohnsonGIN：
- ADE 0.482314337
- FDE 1.049728396
- J 1.007178535

这些结果保留为 architecture-search / quantum-inspired reference。

**不得把这些历史结果改写成 PennyLane-native 正式 QGNN 结果。**

---

## 5. 为什么需要重构

当前 Raj / RC / TRC / TO 后期主线大量使用 PyTorch 手写 quantum mathematics，例如：

- `torch.matrix_exp(1j * H)`
- complex tensors
- manual statevector evolution
- additive state re-upload
- direct real/imag amplitude readout

这些可以作为数学模拟 / reference，但不再接受为最终论文正式 Quantum core。

正式 Raj-QGNN 必须由 Quantum SDK 直接表达。

本项目统一选择：

`PennyLane`

服务器环境已确认：

- PennyLane 0.45.1
- `lightning.qubit` 可创建
- `lightning.gpu` 可创建

具体 operator / batching / differentiation 能力仍需 P1 preflight 实测，不得假设。

---

## 6. Quantum SDK 硬约束

正式 Quantum interaction 路径必须存在于：

`qml.QNode`

Quantum interaction / entanglement / evolution / measurement 不得由 PyTorch 手写 statevector 替代。

允许 PyTorch 做：

- history-only feature extraction
- physical feature construction
- gate-angle generation
- post-measurement readout MLP
- per-agent fusion
- loss / optimizer
- downstream LLM（后续 P2）

不允许正式 Quantum core：

- 用 `torch.matrix_exp` 完成全部 quantum evolution
- 手写 complex statevector 更新作为正式路径
- 仅把旧 dense unitary 包成 `qml.QubitUnitary` 就宣布完成
- 直接读取 statevector real/imag 作为正式 downstream feature

Torch statevector 实现只能放在 reference / equivalence test 中。

---

## 7. Raj higher-order 语义必须保留

### 7.1 强制保留

- j=2 与 j=3 两个 branch
- fixed-cardinality subset / occupation semantics
- weighted Johnson-style transition
- physical interaction conditioning
- both branches participate in every valid scene
- per-agent readout
- MultiJ fusion

不得将模型退化成普通 pairwise QGNN 后仍称为 Raj mainline。

### 7.2 首选 PennyLane 映射

优先评估：

- vehicle/node register：N=8 qubits
- j=2 branch：Hamming-weight-2 subspace
- j=3 branch：Hamming-weight-3 subspace

Johnson 邻接中“一个成员被另一个成员替换”对应 occupation transfer：

`|10> <-> |01>`

优先使用 particle-number-preserving gate，例如：

`qml.SingleExcitation`

或数学等价、明确可解释的 PennyLane gate sequence。

历史 Raj 的 D=6, k=3 embedding semantics 应尽量保留。

### 7.3 首选两寄存器结构

优先设计：

- 8 vehicle/subset qubits
- 6 embedding qubits
- embedding register 使用固定 Hamming-weight k=3 或明确等价表示
- 总计 14 qubits / branch

如果 14-qubit 两寄存器在当前 PennyLane/GPU 路径上存在不可接受的技术障碍，可以提出更小映射，但必须：

1. 明确说明丢失哪些 Raj semantics；
2. 不得自行退化成普通 pairwise QGNN；
3. 不得未经核心审查就把降级版本定义为正式主线。

---

## 8. Graph-conditioned Johnson interaction

目标不是照抄旧的 dense `U = exp(iH)`。

应把 weighted Johnson interaction 映射成可解释 gate sequence。

推荐逻辑：

1. 从 history-only physical relation 生成 pair angle；
2. 对 vehicle qubit pair 执行 particle-number-preserving transition；
3. 多个 pair gate composition 构成一轮 graph-conditioned Johnson mixing。

示意：

`theta_ab^(l) = f_l(e_ab)`

`U_graph^(l) = product_over_pairs U_SE(theta_ab^(l))`

具体 pair ordering 必须冻结，并验证 permutation consistency。

禁止使用：

- future trajectory
- future error
- target ID
- slot ID semantic key

来生成 gate angles。

---

## 9. Data encoding / re-upload

历史 Raj 的：

`state <- state + feature; normalize(state)`

不再接受为正式 Quantum data re-upload。

P1 必须改成 unitary data encoding / re-upload。

允许候选：

- RY / RZ
- PauliRot
- MultiRZ
- controlled rotations
- excitation-preserving encodings
- 其他明确可分解、可解释的 unitary feature map

原则：

- feature -> gate angle
- state evolution stays inside quantum circuit
- 不直接向 statevector 加向量

如果使用 `qml.StatePrep`：

- 只允许固定/明确初态或 diagnostic；
- 不应把任意 learned continuous amplitude injection 作为正式 repeated re-upload 主机制。

---

## 10. Measurement / readout

正式 Quantum output 必须来自 measurable observables。

禁止：

- `qml.state()` real/imag 直接作为 downstream feature
- PyTorch complex amplitude real/imag 作为正式 readout

允许：

- single-qubit Pauli expectations
- pair correlators
- occupation-related observables
- XX / YY / ZZ correlations
- vehicle ↔ embedding cross-register correlations
- 其他明确 Hermitian observable

对于 fixed-weight circuit，优先使用 occupation / coherence 有明确意义的 observables。

P1 必须记录：

- observable list
- observable count
- 每个 agent 如何获得自己的 interaction feature
- quantum measurement 如何映射成最终 64D per-agent interaction representation

---

## 11. Mask / padding / permutation

正式数据当前工程接口 N<=8，但 N=8 不得包装成科研硬限制。

P1 必须正确处理：

- active vehicles
- padded vehicles
- active count < j
- vehicle permutation

### Padding invariance

加入 inactive padded slots 后，active-agent output 在数值容差内不变。

### Permutation consistency

对 active vehicle slots permutation，输出 inverse-permute 后应与原输出一致。

任何依赖 slot ID / absolute index 产生语义的设计都不允许。

如果 gate scheduling 使用 canonical history-only order，必须证明 permutation consistency。

---

## 12. 参数容量公平规则

历史 QGNN core 参数量差异很大：

- TRC ~7k
- TO ~6.7k
- RC ~67k
- Raj ~126.5k

这会混淆 architecture 与 capacity。

从 P1 起必须拆分记录：

1. history / feature pre-encoder params
2. quantum-specific circuit params
3. post-measurement readout params
4. multi-j fusion params
5. total interaction-core params

### P1 目标预算

PennyLane Raj total interaction-core target：

`100k–150k trainable parameters`

这是 total interaction-core，不要求 quantum gate 参数本身达到 100k。

禁止为了凑参数：

- 无意义增加 gate depth
- 重复无物理意义 parameter block
- 加入和 interaction 无关的大 MLP

若合理架构自然落在预算外，必须报告原因，不得偷偷扩缩。

### 后续 matched Classical

P1 只准备接口和参数统计；正式 matched Classical comparison 后续执行。

未来 matched Classical 应：

- share same history features
- share same input/output contract
- share same per-agent output dimension
- total interaction-core params 尽量 within ±20%

同时必须保留 strong classical baseline，不能只留下弱的 parameter-matched baseline。

---

## 13. P1 输出接口

Quantum core 输出固定：

`[B, N, 64]`

64D 是 interaction representation，不是未来轨迹。

要求：

- per-agent
- padding masked
- permutation equivariant
- j=2/j=3 都实际进入表示
- P1 不做 direct coordinate prediction

Residual future-query interface 属于 P2，本轮不实现。

---

## 14. 推荐代码结构

新增独立目录，不覆盖历史 Raj：

`prediction/qgnn_raj_pennylane/`

建议：

- `__init__.py`
- `features.py`
- `quantum.py`
- `torch_reference.py`
- `classical.py`（P1 可只保留 future matched-control scaffold）
- `model.py`

检查脚本：

`scripts/check_raj_pennylane.py`

preflight 结果：

`reports/qgnn/raj_pennylane_preflight/`

不得修改历史 Raj result directories。

---

## 15. Torch reference 的角色

`torch_reference.py` 只用于：

- primitive equivalence
- angle convention
- gate ordering
- observable cross-check
- debugging

它不是正式训练模型。

对于能够严格对应的 primitive，要求检查：

`max_abs(PennyLane - Torch reference)`

并给出容差。

如果新 PennyLane-native mechanism 没有旧 Torch 等价物，应明确标注：

`new circuit-native mechanism`

不要制造伪等价测试。

---

## 16. P1 correctness / preflight 硬门槛

### 16.1 QNode identity

报告必须记录：

- PennyLane version
- device
- interface
- differentiation method
- qubit count
- shots=None / analytic simulator
- gate families

### 16.2 CPU / GPU consistency

至少比较：

- `default.qubit` CPU
- `lightning.gpu` GPU

同 input / parameters 下 observable output 应在合理容差一致。

若某 differentiation method 在 lightning.gpu 不支持，允许工程上调整，但必须记录。

### 16.3 Gradient

所有应训练的 Quantum circuit parameters：

- finite gradient
- 在非退化 synthetic case 上 nonzero gradient

所有 feature / readout trainable blocks也必须有 finite nonzero gradient。

不得只检查总 loss 有梯度。

### 16.4 j=2 / j=3 participation

分别关闭 j=2、j=3：

- output 必须发生非零变化
- 本轮不要求性能变化

### 16.5 Interaction-off

关闭 graph-conditioned interaction：

- output 必须发生非零变化

### 16.6 Johnson / entangling-off

关闭主要 Johnson / excitation interaction：

- output 必须发生非零变化

### 16.7 Padding

active-agent max abs effect 接近 0。

### 16.8 Permutation

slot permutation 后 inverse-permute output，与原输出接近。

### 16.9 Finite

forward / backward 不得 NaN / Inf。

### 16.10 Actual-data micro-smoke

允许 train split 1–2 sample forward/backward。

禁止：

- 多 epoch
- 4096
- full 15,802
- validation tuning
- prediction test

---

## 17. Performance preflight

P1 允许短 profile：

- B=1
- B=2
- 必要时 B=4

只用于判断：

- lightning.gpu 是否实际可训练
- 单 step 速度数量级
- peak VRAM
- 是否存在明显 Python / QNode bottleneck

不要为了速度牺牲 correctness。

如果 14-qubit branch 异常慢，先 profile 并分析 bottleneck，再由核心审查决定是否改变 mapping。

不得自行退回手写 Torch statevector。

---

## 18. P1 明确禁止

本轮禁止：

- 正式 training
- 4096 training
- full 15,802 training
- multi-seed
- five-SNR
- prediction test
- 修改 SinD dataset
- 修改 ISAC frontend
- 修改 GPT-2 / LoRA
- 接入 Residual LLM
- 重新选择 TRC / TO / RC
- 弱化 baseline
- validation-driven architecture tuning
- 删除历史结果
- 新建 worktree / clone
- 从 GitHub stale remote reset server

---

## 19. 文件与临时目录规则

服务器正式项目：

`/home/dell/YrM/ICCT`

本地唯一允许修改的 ICCT 项目：

`E:\NJUPT\ICCT会议`

如果使用本地临时文件，只允许：

`E:\NJUPT\ICCT会议\.codex-work`

禁止：

- Desktop 临时源码
- `E:\NJUPT\ICCT_*`
- 额外 clone
- 额外 worktree

优先直接在服务器正式项目工作。

---

## 20. Git 规则

服务器 workspace 存在与本任务无关的历史 untracked reports。

Luna：

- 不清理
- 不提交
- 不修改这些无关文件

完成后先检查：

`git diff`

只提交本轮：

- `prediction/qgnn_raj_pennylane/**`
- P1 check script
- P1 preflight report
- P1 implementation handoff
- 必要的极小配置变更

commit message 建议：

`Implement PennyLane-native Raj QGNN core`

**不要自动 push**，除非用户明确要求。

---

## 21. P1 交付物

### Code

- PennyLane-native Raj j=2+j=3 core
- 64D per-agent output
- Torch reference（只作 check）
- check script

### Machine-readable preflight

建议：

`reports/qgnn/raj_pennylane_preflight/preflight_20260920.json`

至少包含：

- environment
- qubits
- gates
- observable count
- parameter breakdown
- CPU/GPU numerical check
- gradient checks
- j2/j3 ablations
- interaction-off
- padding
- permutation
- actual-data micro-smoke
- short profile

### Handoff

建议：

`docs/qgnn/RAJ_PENNYLANE_P1_IMPLEMENTATION_HANDOFF_20260920.md`

---

## 22. Luna 完成时必须汇报

1. 实际 circuit architecture
2. 为什么仍属于 Raj higher-order family
3. j=2/j=3 如何编码
4. vehicle / embedding register 实际 qubit mapping
5. gate list
6. data encoding / re-upload
7. Johnson interaction 实现
8. observables
9. per-agent 64D readout 形成方式
10. quantum-specific params
11. feature / readout / fusion params
12. total core params
13. circuit depth / gate count（至少代表性 N=8 active scene）
14. CPU/GPU difference
15. gradient result
16. permutation / padding result
17. short profile
18. known limitations
19. changed files
20. commit hash

---

# P1 STOP GATE

完成上述代码与 preflight 后：

**立即停止。**

不要开始 P2。

核心审查将决定：

- Quantum circuit 是否真实成立；
- Raj higher-order semantics 是否保留；
- 参数预算是否合理；
- measurement/readout 是否可信；
- GPU 路径是否可训练；
- 是否允许进入 Residual QGNN–LLM。

---

# 后续 P2 预冻结方向（仅供理解，不执行）

若 P1 通过，下一步实现：

`Y_hat = Y_CV + ΔY_self + g × ΔY_interaction`

职责隔离：

- Self LLM：target-own history only
- Raj QGNN：multi-agent interaction only
- 20 future queries：分别读取 interaction representation
- gate：控制每个未来时刻 interaction residual

训练阶段：

1. Self stage
2. Interaction stage
3. Joint stage

具体接口与 loss 在 P1 验收后重新冻结。

---

## 一句话任务定义

> **本轮不是追求 ADE/FDE，而是把当前最有希望的 Raj higher-order inductive bias，重构成真正 PennyLane-native、可微、可测、可公平比较、可继续接 Residual LLM 的正式 Quantum interaction core。**

