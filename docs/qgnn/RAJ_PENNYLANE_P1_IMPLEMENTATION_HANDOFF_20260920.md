# Raj-PennyLane P1 实现交接

日期：2026-09-20
范围：P1 correctness / preflight
状态：已完成，等待核心审查；本交接不启动 P2。

## 1. 数据与执行边界

本轮实际数据口径为当前正式 SinD 主数据集，不使用旧 f01d/f01e 数据线：

- train sample contract：`data/sind/splits/train/samples.npz`
- frozen ISAC history cache：`data/sind/isac/train/sensing_cache.npz`
- actual-data micro-smoke：nominal `0 dB`，样本索引 `[0, 1]`，active count `[8, 8]`
- 只读取 train history 所需字段；`future`、validation、test 和 prediction test 均未打开
- 没有 optimizer step、epoch、4096/full training、multi-seed、multi-SNR 或 prediction test

服务器环境已实测：Python 3.11.15，Torch 2.13.0+cu130，PennyLane 0.45.1；`default.qubit`、`lightning.qubit`、`lightning.gpu` 均可构造，正式 GPU profile 使用 `lightning.gpu`。

## 2. Circuit architecture

正式 interaction core 是两个并行的 PennyLane-native Raj branch：

- branch A：vehicle fixed-Hamming-weight `j=2`
- branch B：vehicle fixed-Hamming-weight `j=3`
- 每个 branch 使用同一拓扑的 `8 vehicle qubits + 6 embedding qubits = 14 qubits`
- embedding register 固定为 `D=6, k=3` occupation sector
- 初态是 mask 约束下的合法 vehicle j-subset 与 embedding k-subset 的固定均匀叠加；它只承担明确的 fixed initial state，不承担连续 learned amplitude injection
- 每个 branch 为 3 个 re-upload/evolution round
- 两个 branch 的 measurable readout 各为 `[B,N,64]`，随后由 per-agent `MultiJFusion` 融合为唯一正式输出 `[B,N,64]`

每个 round 的 gate 顺序为：

1. vehicle `RZ` history/feature phase encoding；
2. vehicle-register `SingleExcitation` weighted Johnson occupation transfer；
3. embedding-register `SingleExcitation` weight-k evolution；
4. vehicle-embedding `IsingZZ` entangling conditioning。

正式 Quantum output 只来自 QNode 内的 `qml.expval` observable；formal model readout 不读取 `qml.state()` 的 real/imag。`qml.state()` 只出现在 Torch reference primitive equivalence check 中，不属于正式 readout。

## 3. 为什么仍属于 Raj higher-order family

本实现保留了 Raj Weighted Multi-j 的结构身份，而不是将其替换为普通 pairwise GNN：

- j=2 与 j=3 是显式并行 higher-order subset branches；
- vehicle register 的固定 Hamming weight 直接表示合法 j-subset occupation；
- `SingleExcitation` 在两个 vehicle wires 间转移 occupation，连接两个只差一个 occupied vehicle 的 j-subset，等价于 Johnson graph 的 occupation-transfer edge；
- graph-conditioned angle 由连续 history/physical edge feature 生成，因此保留 weighted Johnson mixing；
- `SubsetFeatureBuilder` 保留 subset-level continuous physics feature 与 incidence pooling；
- 每个 agent 的 branch readout 来自包含该 agent 的 subset incidence pooling；
- j=2/j=3 的 per-agent representation 最后只做本车 fusion，没有新增跨车 classical message passing。

这是一份 PennyLane-native ICCT adaptation，不宣称对 Raj 论文层级加载器和 gate 顺序的 gate-for-gate reproduction。

## 4. Qubit mapping 与 gate list

| register | wires | semantics |
|---|---:|---|
| vehicle | 0--7 | 8 个 vehicle slots；j=2 或 j=3 fixed-weight sector |
| embedding | 8--13 | D=6 embedding occupation；k=3 fixed-weight sector |

在全 active 的代表性 branch、3 rounds、28 个 vehicle pair 下，`qml.specs` 实测：

- qubits：14；shots：`None`；
- total gates：298；logical depth：76；
- `StatePrep`：1 次，14 wires；
- `RZ`：24 次（8 vehicle wires × 3 rounds）；
- `SingleExcitation`：129 次，其中 vehicle Johnson transfer `28×3=84`，embedding pair evolution `15×3=45`；
- `IsingZZ`：144 次（8×6×3）；
- gate size：1 个 14-wire state preparation、24 个 1-wire gate、273 个 2-wire gates。

每个 scene 的 vehicle pair schedule 根据 history-only symmetric physical edge key 与 node feature key canonicalize；QNode 按 schedule cache。为了不以 slot index 破坏 permutation consistency，非对易 `SingleExcitation` 的方向也由 node key 决定。

## 5. Data encoding / re-upload

每轮连续 feature 只转换为 QNode gate angle：

- node history/subset feature -> `RZ` angle；
- directed physical pair feature -> weighted Johnson `SingleExcitation` angle；
- subset scene feature -> embedding `SingleExcitation` angles；
- per-agent feature strength 与 trainable circuit basis/bias -> vehicle-embedding `IsingZZ` angles。

因此 state evolution 完全留在 PennyLane unitary circuit 中，没有 `state += feature`、normalize、`torch.matrix_exp` 或手写 complex statevector 作为正式 Quantum core。

## 6. Johnson interaction 与 formal readout

### Johnson interaction

vehicle register 的 28 个 unordered vehicle pairs 都可成为 occupation-transfer edge。每个 pair 的 graph angle 来自 directed physical edge feature 的 shared MLP；`SingleExcitation` 保持 vehicle Hamming weight，因此 j=2/j=3 sector 内的转移仍在 Johnson-style subset graph 中进行。`johnson_scale=0` 的实测 output change 为 `3.474327802658081`。

### Observable list

每个 vehicle wire `i` 的 67 个 observable 为：

- `Z_i`：1 个；
- `Z_i Z_e`：6 个，`e=0..5`；
- 对 15 个 embedding pair `(p,q)`，`Z_i XX_{p,q}`、`Z_i YY_{p,q}`、`Z_i XY_{p,q}`、`Z_i YX_{p,q}`：60 个。

因此每个 branch 为 `8×67=536` 个 expval，先组成 branch raw observable representation，再经 post-measurement readout 映射到每 agent 64D。

## 7. 64D output contract

正式返回字典的主字段为：

- `readout`: `[B,N,64]`，唯一正式 interaction representation；
- `interaction_mask`: `[B,N]`；
- `branch_readout`: `[B,N,2,64]`，j=2/j=3 diagnostic stack；
- `branch_mask`: `[B,N,2]`。

j=2 与 j=3 各自产生 `[B,N,64]`，`MultiJFusion` 对同一个 agent 的两个 branch 做 gate + delta + LayerNorm，输出固定 `[B,N,64]`。没有将两个 branch 暴露为 downstream 的两个 interaction token。

## 8. Parameter budget

参数统计为整个 interaction core，而不是只数 quantum gate bias：

| block | parameters |
|---|---:|
| j=2 history/feature/angle generation | 51,030 |
| j=2 quantum-specific circuit parameters | 84 |
| j=2 post-measurement readout | 12,998 |
| j=2 branch total | 64,112 |
| j=3 history/feature/angle generation | 51,030 |
| j=3 quantum-specific circuit parameters | 84 |
| j=3 post-measurement readout | 12,998 |
| j=3 branch total | 64,112 |
| multi-j fusion | 16,705 |
| **total interaction core** | **144,929** |

quantum-specific 84/branch 由 trainable node bias、15 个 embedding bias、6 个 cross bias 和 6 个 cross basis 在 3 rounds 的共享拓扑参数组成；没有为了凑预算无意义堆深 circuit 或 MLP。总预算位于目标 `100k--150k`。

## 9. Correctness / preflight evidence

最终机器报告：`reports/qgnn/raj_pennylane_preflight/preflight_20260920.json`。

所有 22 项 check 通过：

- Torch reference 的 `SingleExcitation + IsingZZ` primitive equivalence：max abs error `0.0`；
- QNode forward output：`[1,8,64]`，branch diagnostic `[1,8,2,64]`；
- synthetic finite forward/backward：通过；
- gradient groups：feature/angle `112/112` finite nonzero，quantum-specific `8/8`，post-readout `16/16`，fusion `10/10`；
- j=2 off max change：`2.43174409866333`；
- j=3 off max change：`2.2133543491363525`；
- interaction off max change：`3.510582685470581`；
- Johnson off max change：`3.474327802658081`；
- entangling off max change：`0.2167741060256958`；
- padding active output error：`0.0`；padded output error：`0.0`；
- vehicle permutation inverse-permute max error：`9.5367431640625e-07`，threshold `2e-5`；
- initial j=2 fixed-weight state：node weight `{2}`、embedding weight `{3}`、560 nonzero basis states；
- CPU/GPU same input/state/parameters：`default.qubit + backprop` 对 `lightning.gpu + adjoint`，readout 与 branch max abs error 都为 `0.0`，tolerance `5e-5`；
- actual SinD train micro-smoke：2 samples，`0 dB`，readout `[2,8,64]`，finite loss，146/146 nonzero finite gradients，labels unopened；
- B=1/B=2/B=4 profile：全部 finite。

## 10. Short runtime / VRAM profile

在 `lightning.gpu + adjoint` 上，forward + backward 单 step：

| batch | time (s) | GPU0 peak / 24564 MiB | GPU1 peak / 24564 MiB |
|---:|---:|---:|---:|
| 1 | 9.1720317341 | 785 MiB | 123 MiB |
| 2 | 18.0720269438 | 821 MiB | 123 MiB |
| 4 | 34.6476875059 | 799 MiB | 123 MiB |

外部显存峰值由 `nvidia-smi` 100 ms 采样得到；`torch_peak_allocated/reserved` 为 0，是因为当前 QNode 的 Torch angle/state tensor 保持 CPU，GPU 内存由 PennyLane Lightning 管理。profile 没有 optimizer step。

B=1/2/4 近似线性增长，说明当前主要瓶颈是每个 scene 的 Python/QNode 执行，而不是单个 14-qubit state 的 VRAM 容量。正式实现没有为了该 profile 退回 Torch statevector。

## 11. Known limitations

1. 为保持当前 permutation consistency，scene-specific pair schedule 是 Python-side canonicalization，并按 schedule cache QNode；当前不是单个 fully broadcasted QNode。
2. 如果 physical edge key 与 node key 出现完全相同的 tie，非对易 gate 没有 index-free 的唯一顺序/方向，当前会跳过该 tie group；需要后续审查决定是否引入显式对称 tie-breaking。
3. 当前正式工程 cache 是 8 slots；代码保留已有 history-only patch interface，N>8 的路径仍是 8-node ego patch，不把 N=8 宣称成理论硬限制。
4. `qml.evolve` 没有作为正式依赖；本实现选择了在当前环境已实测可反向传播的基础 PennyLane gates。`lightning.gpu` 的 adjoint profile 已通过。
5. 本轮只有 correctness/preflight，没有任何性能、泛化、ADE/FDE 或量子优势结论。

## 12. Changed files and stop gate

本轮应提交的文件：

- `prediction/qgnn_raj_pennylane/common.py`
- `prediction/qgnn_raj_pennylane/quantum.py`
- `prediction/qgnn_raj_pennylane/torch_reference.py`
- `scripts/check_raj_pennylane.py`
- `reports/qgnn/raj_pennylane_preflight/preflight_20260920.json`
- `docs/qgnn/RAJ_PENNYLANE_P1_IMPLEMENTATION_HANDOFF_20260920.md`

同目录下原有 `model.py`、`residual.py` 等文件没有为本轮 P1 修改；Residual QGNN--LLM 属于 P2，本轮不执行。

最终 commit hash 以本交接文件、check、preflight report 和上述 P1 代码共同提交后的服务器 `qgnn` HEAD 为准；不 push。
