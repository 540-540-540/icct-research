# 外部机制研究与候选淘汰

日期：2026-09-20。研究范围：2023—2026 年量子图学习、量子聚合、图条件演化、子集/超图和时空模型；少量更早文献用于基本机制与反论证。不是穷尽全部论文的系统综述。项目事实来自服务器 `qgnn@094def5b38fa21ccc09cda2328fe78ee8cc694c1`，不是将外部 benchmark 当成 ICCT 结果。

## 1. 文献证据地图

| ID | 原始来源 | 可借鉴的机制 | 不能迁移成 ICCT 结论的部分 | 源码/阅读边界 |
|---|---|---|---|---|
| L1 | Raj et al., Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler–Leman Hierarchy, arXiv:2606.26873v1 (2026) | 子集寄存器、特征寄存器、子空间保持演化、条件 RDM 读出 | WL 区分力不是连续轨迹预测优势；56 qubits 是子空间模拟而非完整 statevector；与同阶经典模型没有普适分离 | 已读正文/补充材料与官方 README，官方代码固定在 `851537589d61bcce96130b055e5724291e7ea318`；服务器 fidelity 审计与实际适配代码交叉核对 |
| L2 | Giang et al., SQM-GNN, arXiv:2601.18198v1 (2026) | 局部 star，node/edge 独立量子寄存器，U_MSG/U_UPD | D2D 标量控制不是多车时序预测；共享参数不足以保证不对易顺序更新的置换对称；一次 CFE 不等于一次门操作/shot | 正文数学与架构已读；尚未核实可直接复用的官方完整代码。原文关于 amplitude encoding 一概不可微的说法不能作为一般定理 |
| L3 | Skolik et al., Equivariant quantum circuits for learning on weighted graphs, npj QI 9,47 (2023) | 图权重进入 Hamiltonian，节点重标号协变，局部门与图演化交替 | TSP/对称性实验不证明优于强经典时序高阶模型 | 论文提供 `askolik/eqc_for_nco` 链接；本轮 connector README 读取返回404，不能声称现已复现其代码 |
| L4 | Schatzki et al., Theoretical guarantees for permutation-equivariant QNNs, npj QI 10,12 (2024), arXiv:2210.09974 | 将任务对称性编码入模型；有条件的可训练性/泛化理论 | 对固定 S_n 对称 ansatz、输入与损失的理论不能原封不动套给数据依赖图演化和整个 GPT-2 联合网络 | 原始摘要/出版信息与已读正文相关结论；不声称本项目自动无 barren plateau |
| L5 | Das & Caruso, Permutation-equivariant QCNNs, QST 10,015030 (2025), published2024 | 对所有排列的 QCNN 做等概率平均，构造等变集成 | 全排列平均/随机近似有额外 calls；并不自动保留有方向的动态交通关系 | 原始摘要与数据可用性声明；不能视为可直接运行基底 |
| L6 | Faria et al., Inductive Graph Representation Learning with QGNNs, arXiv:2503.24111 (2025) | GraphSAGE 式可变邻域、量子卷积/池化聚合器 | 需要逐处检查哪些聚合在经典域完成；QM9 小实验和梯度数值观察不是全任务可训练性定理 | 正文方法与实验阅读；无 ICCT 适配实验 |
| L7 | Faria et al., QGAT: Trainable Quantum Encoders for Inductive Graph Learning, arXiv:2509.11390v1 (2025) | 输入依赖编码、QCNN 聚合与可学习邻车权重 | 原文 Eq8 是张量积输入，不能误说成经典混合态；Eq13—14 的通道分解及固定输入规模说法需要实现澄清；没有直接解决 e_jk 的显式动态关系 | 正文核查，不将名字叫 QGAT 的不同论文/本项目旧实现混为一谈 |
| L8 | Shankar & Towsley, Bosonic Random Walk Networks for Graph Learning, arXiv:2101.00082 (2020/2021) | 多粒子配置空间、相互作用势、干涉决定扩散 | 最后用概率 P 做经典 PX；实验没有读取高阶配置特征；不是本报告的全量子联合特征核心 | 原始 PDF 方法3.1—3.2已读；不搬运其指标 |
| L9 | Li et al., CTQWformer, arXiv:2605.09486v1 (2026) | 特征/拓扑条件 Hamiltonian，多个演化时刻的读出 | 量子 walk 时间不是20帧真实交通时间；主体含经典 Transformer/BiGRU，不能直接当本项目 Primary | 正文方法阅读 |
| L10 | A Spatio-Temporal Hybrid Quantum-Classical GCN for Urban Taxi Destination Prediction, arXiv:2512.13745 (2025) | 时空表示和量子特征结合 | classical GCN/diffpool 先处理主要空间关系；预测单出租车目的地而非多目标20步轨迹 | 原始摘要与 Exa 抽取方法，未作完整代码复现 |
| L11 | Innan et al., A2QTGN, arXiv:2605.21916v2 (2026-07-26) | 按历史变化更新量子编码 | 主干是经典 TGN；adaptive node embedding并不是本项目要求的主要多车联合推理 | 已读最新检索到的v2方法，不将v1/v2混用 |
| L12 | Li et al., Learning Socio-Temporal Graphs for Multi-Agent Trajectory Prediction, arXiv:2312.14373 (2023) | 显式跨时间/跨目标关系；经典时空 attention 反事实 | 行人数据而非当前 SinD，不构成实际 ICCT baseline 得分 | 原始全文方法与消融阅读 |
| L13 | Schuld, Sweke & Meyer, Effect of data encoding on expressive power, PRA103,032430 (2021) | 数据重上传改变可访问的 Fourier 频谱 | 更丰富频谱不等于在带噪数据上更好；小网络也可用经典 Fourier/tensor 特征逼近 | 原始摘要与机制，作为理论依据而非性能证据 |
| L14 | Caro et al., Generalization in QML from few training data, NatCommun13,4919 (2022), arXiv:2111.05292 | 泛化与有效可训练门数/训练变化相关的上界 | 不能忽略共享 GRU、adapter、LoRA、数据依赖编码参数与重叠窗口；也不能从上界推出优于经典模型 | 原始 arXiv 摘要、作者机构出版页；不使用缺失根号的网页转录公式 |
| L15 | Sweke et al., Potential and limitations of RFF for dequantizing QML, Quantum (2025) | 量子函数可否被随机 Fourier 特征代理取决于谱/分布条件 | 不是所有 QNN 都可低代价 RFF 模拟，也不是所有不可简单 RFF 化的模型有实用优势 | 作者 IBM Research 原始出版页 |
| L16 | Sahebi et al., On Dequantization of Supervised QML via RFF, arXiv:2505.15902 (2025) | 分类/回归的去量子化比较框架 | 不自动覆盖本报告的数据条件矩阵指数结构；可作为强经典代理设计依据 | 原始摘要 |
| L17 | Li, Nagano & Terashi, Enforcing exact permutation and rotational symmetries on point clouds, PRResearch6,043028 (2024) | 相对几何、置换与旋转对称的结构设计 | 交通绝对道路方向/信号可能有意义，不能强制整个 predictor 旋转不变；点云分类非轨迹预测 | 原始论文摘要 |

研究发现：直接叫 quantum hypergraph 的检索结果中，部分实际是“经典超图网络学习量子物理/纠错数据”；不满足 quantum core 条件，不列为可用算法。显式 ZZZ phase/量子超图态可归入 F3，但“是超图态”“有纠缠”没有自动的 ADE/FDE 解释。

PDF 阅读说明：对 Raj/SQM/Bosonic PDF 均调用了 web screenshot，但服务返回 Internal Error/Cache miss；文本、公式、HTML 与原始代码继续交叉核对。未据不可见图片推断电路连线。PDF 下载备用通道也失败，因此不宣称已完成这些图的视觉核查。

## 2. 六个本质不同的架构族

### F1 — 独立 node/edge PQC + 经典聚合
量子态表示单车或单条边；通常 q=4—8，角度或幅值编码；PQC后用 Pauli 读出，然后经典求和、attention或MPNN。每场景 calls=O(LN) 或 O(LE)，state=2^q，参数可共享 O(Lq)。适合局部特征映射而非邻车间联合状态。输出向量可接单个softtoken。ICCT早期VQC装饰路线无法作为主要多车推理。对照应为同输入MLP/RFF+相同聚合。**淘汰：违反本轮核心定位**；不是因为所有此类模型无用。

### F2 — Star quantum message/update（SQM型）
量子态包含root、k邻居及k条root-edge；原始scalar版本Q=2k+1。L层逐目标处理 calls约LN；state2^(2k+1)，参数共享不随N但输入/测量/梯度成本不能忽略。多邻居可通过center发生联合影响，不应误称始终只是独立pair求和；问题在e_jk缺失、时序/高维feature编码和顺序对称性未充分定义。接center多Pauli→多个softtokens可改造。对照shared star-MPNN/attention＋同级高阶扩展。**淘汰当前原版**：需要同时重建encoding、j-k关系、U_UPD对称性，源码可复用性尚未核实；修复后更接近F3而非原文直接迁移。否证条件：置换失败、center读出无法保留joint必要信息、matchedstarC追平且更廉价。

### F3 — 节点寄存器上的图条件联合自旋演化（RC/等变Hamiltonian型）
每车q_v量子比特，Q=q_vK；全图ZZ/ZZZ或有向角色Pauli耦合＋非对易局部旋转，沿真实时间重上传。每scene可一次全图，或N个boundedego；state2^(q_vK)，gate约O(L(K^2+K^3))，若不用显式ZZZ则K^2。参数可独立于N。所有邻车确有联合态，直接多体测量；输出rootRDM＋root-neighbor矩→time softtokens。最强对照是同physics/time的tensor/higher-orderMPNN。历史RC总体输、部分高closing赢，说明机制有场景信号但不能继续无约束加ZZZ/feedback。**保留唯一Backup：TES-QGNN**，两比特/车、四时间块、原生Pauli耦合；无反馈控制器。主要风险是指数state和仍然有限的局部测量。

### F4 — 无根全局 fixed-weight subset 两寄存器（Raj型）
子集S∈C(N,j)表示高阶对象；feature寄存器维C(D,k)，联合/受限空间必须按实际M定义。调用通常每scene每j一条；适配版受限空间C(N,j)C(D,k)，全M可进入C(N+D,j+k)，不可混报；dense expH常见O(C(N,j)^3)。适配输出实虚振幅池化，原论文定义条件1RDM并付projection代价。其classicalcounterpart就是weightedJohnsonGIN/更强subsetTensor，并非普通GIN。**不直接选择**：最强适配full-train近打平且含权重匹配/振幅读出边界；paper-fidelity失败后不能假装原论文已经复现到当前任务。保留“子集身份＋结构化酉混合”，不保留其全部接口。否证：物理加权/预算匹配后收益消失，或RDM版本无法保留有效信息。

### F5 — Quantum walk/kernel 作为经典全局模型的结构偏置
单/多walker graph-dependent dynamics，readout probabilities或kernel；再用classicalPX/Transformer/GNN推理。单walkerbinaryQ=ceil(log2N)+coin，多walkerQ约rlogN，dense矩阵指数/全n起点/readout并不免费。量子calls随初态、时间、target而变；模型若多数推理在经典Transformer则不符合本轮定位。强对照classicaldiffusion、complex/unitarywalk、learnedgraphattention。**淘汰完整架构**，保留graph-conditionedHamiltonian与干涉机制作为F6的构件。否证：不用量子概率偏置不掉点、经典walk代理等价，或主要收益完全来自经典后端。

### F6 — 目标条件、时序保持的邻车配置空间 QGNN（项目定制）
root i显式固定；Ω1={j}、Ω2={(j,k),j≠k}。量子态为configurationindex×root/neighborsfeature寄存器；不同配置通过有物理权重的HermitianHamiltonian相干混合，再做configuration条件的多角色酉编码。四个历史块顺序演化，读出8个可测量、保留阶次/时间的token，不用振幅直接作feature，也不在量子后做跨车GNN。
Q1=ceil(log2(K−1))+4；Q2=2ceil(log2(K−1))+6；K8时7/12qubits，模拟有效态112/2688。每target两个state trajectories，不是每个tuple单独PQC后经典池化。S=4时间快照在模拟中可复用；真实shots必须重复制备。参数与N无关，denseconfigurationhop O((K−1)^6)但K固定8时M2仅42。对照sameΩ/samephysicalweights/sametime/sametokens的强classicalconfigurationMPNN/attention，以及complexunitary精确模拟/去相干机制控制。
**选择Primary：TRC-QGNN**。选择理由是任务匹配、可检验的单一机制和能修复历史物理/读出边界，而不是已证明ADE/FDE优势。最大风险是index-trace读出仍丢信息、少参过度约束、条件门硬件深度、以及强经典配置网络自然追平。

## 3. 淘汰原则与唯一创新中心

本报告不是将所有论文模块堆叠。Primary只有一个创新中心：**在目标条件的邻车配置空间中，保留真实历史顺序进行图条件相干消息传递，然后测量得到有时间/阶次结构的交互表示**。boundedcontext、tokenadapter、共享GRU是支撑模块，不另算量子创新。Backup只有一个，采用物理节点联合寄存器，以检验“配置空间限制是否不适合任务”这一不同失败模式。

最强反论证预先承认：本项目小配置空间可在经典GPU上精确模拟；若一个complexunitary classical程序实现相同数学函数，它不可能在相同参数下被Primary“量子胜出”。科学命题只能是该量子可实现的归纳偏置是否优于预先选择的、功能匹配的强经典学习家族。该命题必须由新原型证实，当前没有此证据。

## 4. 文献入口

L1 https://arxiv.org/abs/2606.26873 ; https://github.com/SnehalRaj/mp-qgnns/tree/851537589d61bcce96130b055e5724291e7ea318
L2 https://arxiv.org/abs/2601.18198
L3 https://doi.org/10.1038/s41534-023-00710-y
L4 https://doi.org/10.1038/s41534-024-00804-1 ; https://arxiv.org/abs/2210.09974
L5 https://doi.org/10.1088/2058-9565/ad8e80
L6 https://arxiv.org/abs/2503.24111
L7 https://arxiv.org/abs/2509.11390
L8 https://arxiv.org/abs/2101.00082
L9 https://arxiv.org/abs/2605.09486
L10 https://arxiv.org/abs/2512.13745
L11 https://arxiv.org/abs/2605.21916v2
L12 https://arxiv.org/abs/2312.14373
L13 https://doi.org/10.1103/PhysRevA.103.032430
L14 https://doi.org/10.1038/s41467-022-32550-3 ; https://arxiv.org/abs/2111.05292
L15 https://research.ibm.com/publications/potential-and-limitations-of-random-fourier-features-for-dequantizing-quantum-machine-learning
L16 https://arxiv.org/abs/2505.15902
L17 https://doi.org/10.1103/PhysRevResearch.6.043028
