# Raj-PennyLane P1.1 修复与独立验收（2026-09-20）

## 1. 背景

Luna 在 commit 98863aa 中完成第一版 PennyLane-native Raj core。独立验收确认其 Quantum SDK 路径真实存在，但发现三个不能进入正式训练的结构性问题：

1. 每个 scene 根据 trainable node feature 生成不同非对易 gate order，并动态创建/缓存 QNode；
2. B=1/2/4 forward+backward 约为 9.17/18.07/34.65 s，几乎逐 scene 线性增长；
3. continuous subset feature 在进入 Quantum circuit 前被明显 pooling，历史最好 Raj 的 subset-conditioned 语义保留不足。

此外，Luna 越过 P1 STOP GATE 提前新增 Residual/Model prototype。本轮不删除这些文件，但从 package public API 撤下，等待 P2 重新冻结。

本轮不运行任何 trajectory training epoch。

## 2. 修复后的 Raj-PennyLane

### 2.1 Higher-order branch

继续保留：
- j=2 fixed-Hamming-weight vehicle branch；
- j=3 fixed-Hamming-weight vehicle branch；
- D=6,k=3 embedding register；
- 8 vehicle qubits + 6 embedding qubits = 14 qubits / branch；
- 3 rounds；
- MultiJ per-agent fusion。

### 2.2 显式 subset-conditioned encoding

P1.1 将每个 j-subset 自身的连续 history/physics feature 映射为 subset-conditioned phase gate：

- j=2：28 个 ControlledPhaseShift / round；
- j=3：56 个 two-control PhaseShift / round。

独立消融 subset_phase_off 导致 output max change = 0.03436，确认该 higher-order 通道实际参与输出。

### 2.3 Fixed topology

删除：
- scene-specific pair ordering；
- trainable feature -> discrete gate orientation；
- per-scene QNode cache。

所有 scene 共享固定 circuit topology，仅 gate angles 随 history 改变。

### 2.4 Canonical packing

active vehicles 根据原始 observed history 做 deterministic lexicographic packing，key 只使用：
- first frame [x,y,vx,vy]；
- middle frame；
- last frame。

不使用任何 trainable representation，因此模型参数不再控制离散 circuit topology。

8 组随机 vehicle permutation 的 inverse-permute max error = 0.0。
active count 3/5/7 的 padding active/inactive error 均为 0.0。

## 3. Formal observable 与训练加速

每个 vehicle 使用 37 个 measurable features：

1. vehicle occupancy projector：1；
2. conditional embedding occupation：6；
3. 15 个 embedding pair 的 real/imag coherence：30。

合计 37D/vehicle。

正式数学定义仍然是 Hermitian observable expectation。

训练快路径：
1. PennyLane QNode 执行完整 circuit；
2. QNode 返回 exact qml.state；
3. PyTorch 只通过 Pauli expectation 公式计算上述 37 个 observable；
4. state real/imag 从不直接作为 downstream feature。

数值等价：
- j2 fast-statevector vs direct qml.expval：max abs 0.0；
- j3：max abs 0.0；
- default.qubit vs lightning.gpu formal qml.expval：
  - j2 max abs 5.96e-8；
  - j3 max abs 5.96e-8。
- fast observable path vs direct qml.expval angle-gradient：
  - j2 max abs 0.0；
  - j3 max abs 0.0。

因此训练快路径与 formal measurable model 数值等价。

## 4. Batch 执行修复

Luna P1：
- 100 synthetic scenes -> 100 unique QNode schedules；
- B=1/2/4 约 9.17/18.07/34.65 s。

P1.1：
- fixed topology；
- batch 只改变 angles；
- active-count 相同 scene 一次 broadcast；
- 不同 active-count 仅分少量 group。

SinD train active-count：
- 8 cars: 14900 / 15802 = 94.29%
- 7: 290
- 6: 328
- 5: 171
- 4: 102
- 3: 11

最终 full-core forward+backward：

| Batch | Seconds | Peak CUDA allocated |
|---:|---:|---:|
| 1 | 1.783 | 0.218 GiB |
| 4 | 3.149 | 0.808 GiB |
| 32 | 4.175 | 6.314 GiB |

B=32 已满足冻结的 engineering gate <20 s。

## 5. 参数量与 future baseline fairness

新 Quantum interaction core：
- j2 feature/angle generation: 51,030
- j2 quantum-specific: 84
- j2 readout: 10,058
- j2 total: 61,172
- j3 feature/angle generation: 51,030
- j3 quantum-specific: 84
- j3 readout: 10,058
- j3 total: 61,172
- MultiJ fusion: 16,705
- total: 139,049

当前参考 classical：
- matched Johnson: 112,390，较 Quantum -19.17%
- strong Johnson: 162,496，较 Quantum +16.86%

Quantum / matched / strong 均为：
- interaction_tokens = 2
- readout_dim = 64

因此参数量和未来 Residual token interface 已回到可公平比较范围。

## 6. P1.1 最终 preflight

机器报告：

reports/qgnn/raj_pennylane_p1_1/repair_preflight_20260920.json

最终 14/14 checks passed：

1. parameter budget
2. fixed circuit topology
3. formal observable contract
4. CUDA forward/backward + all gradient groups
5. statevector vs qml.expval exact forward equivalence
6. default.qubit vs lightning.gpu formal backend consistency
7. statevector-observable fast path vs direct qml.expval angle-gradient equivalence
8. mechanism participation
9. permutation + padding
10. mixed active-count batch
11. j=2/j=3 fixed Hamming weight
12. logical resource accounting
13. actual SinD train micro-smoke
14. B=32 training-speed gate

关键机制消融 max output change：
- j2 off: 1.9996
- j3 off: 2.9238
- interaction off: 0.6256
- Johnson off: 0.6496
- entangling off: 0.1010
- subset phase off: 0.03436

Actual SinD train 0 dB micro-smoke：
- sample [0,1]
- active count [8,8]
- labels unopened
- all four gradient groups finite/nonzero

## 7. Logical circuit resources

qml.specs logical resources；StatePrep 未分解，因此不能当作 physical elementary-gate count。

j=2：
- 14 wires
- 358 logical operations
- depth 73
- StatePrep 1
- ControlledPhaseShift 84
- SingleExcitation 129
- IsingZZ 144

j=3：
- 14 wires
- 442 logical operations
- depth 189
- StatePrep 1
- two-control PhaseShift 168
- SingleExcitation 129
- IsingZZ 144

论文报告时必须明确是 logical count with undecomposed fixed StatePrep。

## 8. Residual prototype 状态

Luna 提前新增的 residual.py、model.py、classical.py 保留为未批准 P2 prototype。

本轮：
- 未训练；
- 未作为正式 P2；
- 已从 prediction.qgnn_raj_pennylane 的 public __all__ 撤下。

当前正式 public API 只有：
- PennyLaneRajBranch
- PennyLaneRajMultiJCore

## 9. P1.1 结论

当前 Raj-PennyLane core 通过 P1/P1.1 gate。

相较 Luna 原 P1，四个 blocker 已解决：
- dynamic per-scene QNode -> fixed QNode
- trainable discrete gate order -> raw-history canonical packing
- weak subset semantics -> explicit subset-conditioned phase encoding
- unusable speed -> B32 ~4.3 s full-core forward/backward

下一阶段可以进入 P2 架构工作，但仍应先冻结 Residual interface、phase training policy 和 baseline fairness，不直接启动 full training。
