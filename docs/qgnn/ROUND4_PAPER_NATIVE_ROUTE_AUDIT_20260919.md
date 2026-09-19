# Round 4 前置：论文原生 QGNN 路线与 Baseline 审计

日期：2026-09-19
状态：Round2 / Round3 整体量子优势未建立后的换路线前审计。

## 1. 为什么现在重新回到论文原生路线

当前 RC-HQGNN 已经验证：
- 图结构真正进入量子核心；
- ZZ/ZZZ 路径与 K2/K3 确实被模型使用；
- 高动态多车场景存在量子正信号；
- scene-adaptive / K2K3 feedback 有小幅真实收益。

但：
- Round2 full 0dB 仍整体输强经典约4%；
- Round3 adaptive 小实验仍输约2.5%；
- feedback from-scratch 只带来约0.4% ADE / 0.9% FDE 改善；
- 因而继续在同一 ZZ/ZZZ + cumulant readout 范式内微调的边际价值已经下降。

下一步必须同时做两件事：
1. 审计当前 RC-HQGNN 的 encoding / evolution / measurement bottleneck；
2. 重新引入此前筛选出的论文原生 QGNN，尤其沿用论文自身 classical baseline，避免继续完全自创。
## 2. Route P1 — Raj et al. 2026：Higher-Order Subset Message-Passing QGNN

论文：
Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler–Leman Hierarchy
arXiv:2606.26873

官方代码：
https://github.com/SnehalRaj/mp-qgnns

### 原生量子机制

核心不是“每条边生成一个相位”，而是把 j-车辆子集本身作为 node register 的量子表示单位。

N个节点、particle number j：
- node register 位于 Hamming-weight-j 子空间；
- basis 对应 C(N,j) 个无序 j-subset；
- Johnson graph J(N,j) 连接只相差一个节点的两个 subset；
- quantum message passing 使用 exp(i alpha A_J) 在 subset 空间中传播；
- embedding register 使用固定粒子数子空间与 RBS / compound mixing；
- permutation equivariance 由构造保证；
- j 决定 set-j-WL 表达层级。

官方代码同时提供：
- EquivariantQGNN
- JohnsonGIN
- GIN
- QM9QGNN
- TSPQGNN
- PennyLane 与 reduced-basis PyTorch 后端。
### 为什么和 SinD 很匹配

N=8 时：
- j=2：28 个 pair subset；
- j=3：56 个 triplet subset。

这正好对应 SinD 已确认的：
- pair interaction 有价值；
- triplet interaction 有额外价值；
- 高动态联合场景中量子正信号更明显。

相比当前 RC-HQGNN：
当前方法把连续 pair/triple relation 压进 beta_ij / gamma_ijk 标量相位，再依赖有限 observables 读出。

Raj-style 路线可以：
- 让每个 pair/triplet 保持独立 subset identity；
- 在 Johnson subset graph 上做真正的跨-subset message passing；
- 减少“多维物理关系压成一个相位标量”的表示瓶颈风险。

### SinD 迁移草案

保持每车 history encoder。

对 j=3 subset {i,j,k}：
- node history summary；
- e_ij/e_ik/e_jk；
- closing/TCPA/DCPA/risk；
- recent-vs-early dynamics；
构造成 permutation-symmetric continuous subset feature。

subset feature → embedding register loader。

Johnson quantum mixing 在 56 个三车 subset 之间传播。

最后对所有包含车辆 i 的 subset embedding 做 permutation-equivariant incidence readout，得到 per-agent 64维连续 interaction representation，再进入同一 Token + GPT-2。
### 原生 classical baselines

优先沿用论文对应体系：

1. GIN：original graph 1-WL baseline。
2. JohnsonGIN：在同一个 J(N,j) subset lift 上做经典 message passing。
3. PPGN / 3-WL：作为更强 higher-order 参考。
4. 项目原 Adaptive Classical：继续作为最终强项目 baseline。

其中 JohnsonGIN 是最关键公平对照：
- 和量子模型使用相同 j-subset 表示；
- 相同 subset features；
- 相同 Johnson topology；
- 区别主要集中在 quantum unitary mixing vs classical nonlinear message passing。

### 风险

- WL 理论针对图结构可分辨能力，不直接保证连续轨迹回归 ADE/FDE。
- 论文 QM9 中提高 j 会降低量子误差，但一个参数量约大320倍的 classical GNN 仍能更低误差。
- 需要把 graph-level pool 改成 per-agent incidence readout。
- 必须防止 classical subset encoder 本身提前完成主要多车推理。

### 当前优先级

**最高。若 RC-HQGNN bottleneck 审计确认 relation compression / readout 是主问题，优先切换到该路线。**
## 3. Route P2 — SQM-GNN 2026：Quantum MSG + Quantum UPD

论文：
Scalable Quantum Message Passing Graph Neural Networks for Next-Generation Wireless Communications
arXiv:2601.18198

### 原生机制

它保留经典 GNN 的 message-passing 语义，但把核心过程放进 PQC：
- node feature quantum encoding；
- edge attribute quantum encoding；
- U_MSG 对 neighbor-edge pair 生成量子 message；
- U_UPD 将多个 neighbor message 依次汇入 center-node state；
- 多层 QGCL 后测量 node embedding；
- fixed-size star subgraph + shared PQC 控制 qubit 数。

论文任务本身就是无线 D2D power control，连续 node/edge 属性与本项目物理图比一般分子/分类任务更接近。

### 论文原生 baseline

- classical GNN：Graph Neural Networks for Scalable Radio Resource Management (2021)；
- WMMSE：优化算法基准；
- QSGCN：另一种量子图结构参考。

论文报告的设置中，SQM-GNN(k=6)使用13 qubits、5925 trainable params，classical GNN约67073参数；论文报告 SQM-GNN 在 D2D 测试和跨网络规模泛化中优于对应 classical GNN / WMMSE。
### SinD 迁移价值

这是最适合借鉴“真正 quantum message / update”的论文。

可以将每个目标车作为 center：
- neighbors = history-only 风险图中的邻车；
- node register 编码本车/邻车状态；
- edge register 编码 distance/closing/TCPA/DCPA/relative velocity；
- U_MSG 产生 neighbor-edge quantum message；
- U_UPD 在量子域内聚合到中心车辆；
- 输出 per-agent quantum interaction representation。

### 风险

- 目前未找到官方 GitHub 仓库，不能把论文公式重构称为官方复现。
- 论文主要是 pairwise star message passing，没有显式 triplet subset。
- 若逐 neighbor-edge PQC 调用，会重新引入 circuit-call 成本。
- 必须检查 U_MSG/U_UPD 是否能自然扩展到我们已证实重要的三车联合关系。

### 当前优先级

**第二。非常适合作为“无线任务原生 quantum message passing”路线，但实现与 higher-order 扩展风险高于 Raj。**
## 4. Route P3 — Edge-Local QGCN 2026

论文：
Edge-Local and Qubit-Efficient Quantum Graph Learning for the NISQ Era
arXiv:2602.16018

官方代码：
https://github.com/ArminAhmadkhaniha/QGCNlib

### 官方代码真实结构

Quantum feature extraction：
- amplitude embedding；
- StronglyEntanglingLayers；
- Z expectation。

Edge message：
- concatenate h_i/h_j；
- RY angle encoding；
- feature-wise CNOT-RZ-CNOT interaction；
- RX mixers；
- Z expectations。

官方 PyG QGCNConv / NISQQGCNConv 最终仍通过 MessagePassing.propagate(..., aggr=add) 做邻居求和。

HybridQGCNConv：
- 保留同一个 quantum feature extractor；
- 用 classical message aggregation 替换 quantum edge message。

因此其最干净的对照是：
Fully/edge-quantum message vs same quantum feature extractor + classical aggregation。
### 对本项目的定位

优点：
- 官方代码完整；
- PennyLane + PyG；
- pairwise quantum message 非常清楚；
- qubit-efficient；
- hybrid ablation 天然公平。

缺点：
- 主要是 pairwise；
- 无显式三体高阶；
- 代码中邻居聚合仍然是 classical add；
- 不满足我们“量子核心必须承担高阶多车联合推理”的最终主路线要求。

### 当前优先级

**量子 baseline / comparator，而不是最终主模型。**

可用来回答：
“即使采用论文原生 edge-local quantum message，是否能超过对应 hybrid classical aggregation？”
## 5. Route P4 — Skolik et al. 2023 EQC

论文：
Equivariant quantum circuits for learning on weighted graphs
npj Quantum Information 9,47 (2023)
arXiv:2205.06109

### 原生比较

EQC：graph-structured、permutation-equivariant ansatz。

论文量子 ablation：
- NEQC：相同基本结构但打破参数共享/等变；
- HWETE：hardware-efficient + trainable graph embedding；
- HWE：hardware-efficient，graph embedding固定。

另外使用 nearest-neighbor / Christofides 等 TSP classical heuristic 作为参考。

### 对本项目的定位

RC-HQGNN 已经继承了它最核心的思想：
- graph weights 直接控制 quantum evolution；
- 参数共享；
- permutation equivariance。

因此继续“更像 Skolik”预计不会解决当前核心瓶颈。

### 当前优先级

**历史量子 baseline / symmetry ablation。**
## 6. 不推荐作为主路线：Quantum Positional Encoding

Quantum Positional Encodings for Graph Neural Networks (ICML 2024) 的主要价值是量子计算图位置/结构编码，再交给 classical GNN。

这类路线适合作为辅助特征或 ablation，但与项目硬约束冲突：
核心多车交互不能主要由 classical GNN 完成后，只把 quantum representation 当额外 embedding。

因此不作为主 QGNN。

## 7. 新的实验比较矩阵

后续不再只有：
RC-HQGNN vs Adaptive Classical。

建议形成两级矩阵。

### A. Paper-native validity

Raj-QGNN(j=2/3)
vs
JohnsonGIN(j=2/3)
vs
GIN
vs
PPGN/k-IGN（资源允许时）。

Edge-Local QGCN
vs
HybridQGCN（same quantum feature extractor + classical aggregation）。

SQM-GNN adaptation
vs
论文风格 classical GNN。

### B. Project-level strong comparison

每条有希望的 quantum 路线最终都还必须挑战：
Project Adaptive Classical + same GPT-2。

这样既不故意挑弱 baseline，也能判断论文机制迁移是否成功。
## 8. 下一步顺序

### Step 1 — 先完成 representation bottleneck audit
判断当前 RC-HQGNN 主要失败在：
- relation encoding/compression；
- quantum evolution；
- measurement/readout。

### Step 2 — 同时做 Raj-style 最小原型
原因：
- 官方代码完整；
- subset高阶结构最贴合SinD证据；
- N=8,j=3仅56个subset，工程成本低；
- JohnsonGIN baseline天然存在；
- 可以直接验证“保持triplet identity而不是压成gamma标量”是否解决当前问题。

原型只做：
- forward/backward；
- permutation；
- mask；
- B32 step time；
- 512/4096 train小pilot。
不立即替换正式主模型。

### Step 3 — 若 Raj-style 明显优于对应 JohnsonGIN 或至少显著缩小差距
再进入 full 0dB + same GPT-2。

### Step 4 — 若 Raj-style 仍无优势
再转 SQM-GNN U_MSG/U_UPD 路线。

### Step 5 — Edge-Local 保留为论文原生 quantum baseline
不作为最终高阶主模型。

## 9. 当前决策

当前不继续在 Scene-Adaptive RC-HQGNN 上增加第三种 controller。

优先策略变为：

**“瓶颈定位 + 论文原生路线最小原型并行推进”。**

第一替代主路线候选：
**Raj-style Higher-Order Subset Message-Passing QGNN。**

第二替代主路线候选：
**SQM-GNN-style Quantum MSG/UPD。**
