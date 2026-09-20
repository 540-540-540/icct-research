# 历史 QGNN 审计：结果、机制、证据边界

版本：20260920_v1。源快照：qgnn@094def5b38fa21ccc09cda2328fe78ee8cc694c1。
本轮没有训练、推理或重算历史指标；下列数值来自已有报告。CODE/RESULT表示直接读取，ANALYSIS表示本轮推论。所有预测比较均为validation，不是论文最终test成绩。

## 1. 结论

目前没有已验证的“完整量子机制 + 全量SinD + 同信息强经典对照 + ADE/FDE同时稳定改善”候选。Raj-style Candidate A是重要数值参照，但它的高阶子集表示、复数动力学与完整可执行量子算法应分开评价。SQM是值得研究的量子聚合架构模板，尚不是已有可复现ICCT方案。

## 2. 历轮结果不能混成一张排名表 [RESULT]

| 轮次及协议 | Quantum ADE/FDE | 同轮 Classical ADE/FDE | 含义 |
|---|---|---|---|
| R2 RC，15802 train/1880 val，20ep，0dB，seed2026 | .51094460 / 1.12605076 | .49146503 / 1.07928883 | 整体仍落后；高动态子集获益不能覆盖整体损失 |
| R3最终neutral，4096 train，12ep，0dB，seed2026 | .60102165 / 1.24181526 | .58708243 / 1.21296340 | 小规模门禁未过；未进入R3正式全量训练 |
| R3重训练反馈对照：feedback on | .601021648 / 1.241815257 | .585932840 / 1.211676313 | 使用该重训练实验自己的classical结果，不混用上一行 |
| R3重训练反馈对照：feedback off | .603416829 / 1.252787244 | .586512270 / 1.210704665 | feedback对Q的J改善约.641%，但不足反超 |
| Raj weighted multi-j，4096 train，20ep，seed2026 | .516457940 / 1.102781152 | .525952385 / 1.129781810 | 小样本有正收益 |
| 同上，seed2027 | .530895042 / 1.148233937 | .550422603 / 1.189891210 | 第二seed小样本正收益 |
| Raj weighted multi-j，15802 train，20ep，seed2026 | .4812586647 / 1.0511729960 | .4823143366 / 1.0497283959 | ADE约+.219%，FDE约-.138%，接近打平 |

最后一行Q/C最佳checkpoint均epoch9；Q/C训练时间1121.75/1204.08秒，图核心参数126535/179201；共同下游non-LoRA=4052375、LoRA=98304。不要据单次墙钟推导普遍量子加速。全部均为经典GPU模拟。

## 3. RC-HQGNN究竟量子化了哪里 [CODE]

C=4、D=3、每通道每车一个qubit。单车GRU32和物理边/三元关系生成角度；RY/RZ/RX、ZZ/ZZZ联合演化；读出X/Y/Z及Z二/三体相关量。
量子输出之后仍有经典pair/triplet value网络，把邻车历史表示与关系特征变成16维payload，再用量子二/三体相关量加权聚合。因此不能简单表述为“所有跨车消息生成和聚合都在量子域完成”。它是量子相关性驱动的混合交互网络。
二/三体Z cumulants不是通用纠缠见证；不能由其非零直接证明量子资源必要。
移除ZZ、ZZZ或整个graph后变差，说明训练好的网络依赖这些通道，但事后输入/算子删除可能造成分布偏移。移除整个graph还会删除本车GRU/local路径，因此不能解释成纯粹的邻车信息贡献。

## 4. Scene-Adaptive路线：依赖不等于优势 [RESULT/ANALYSIS]

固定历史模式与误差有相关性，但已有39-frame embargo的blocked-CV路由诊断接近随机，未证明history-only稳定选择器。该结果只约束已试特征与学习器，不能证明所有自适应方法不可能有效。
反馈从头重训练的factorial比post-hoc关闭更强。它支持既有K2/K3反馈有有限收益，不支持其足以解决整体性能差距，也不能单独归因于K3。
测量反馈后的同一未知量子态不能直接无扰动继续使用；真正硬件方案须交代prefix重放、多副本或其他反馈协议。模拟器读取中间状态的低成本不能作为硬件成本。

## 5. Raj Candidate A的科学价值和缺口 [CODE/ANALYSIS]

实现：prediction/qgnn_paper_native/raj_paper.py:291-326，raj_subset.py:54-97。
对j=2和j=3分别形成子集特征；图条件实对称Hamiltonian的exp(iH)混合子集；D=6、k=3的20维embedding子空间作compound演化；每轮添加新特征并逐子集行归一化；把状态实部/虚部直接作为经典特征，再按子集-节点incidence平均，叠加本车local表示并接共同GPT-2。

### 5.1 数值有效性和量子算法完整性分开

已有训练结果对上述计算函数有效，不因本审计而作废。但完整量子主张还有两处必须交代：
- raw real/imag amplitudes不是普通observable期望值；需要明确相位参照、状态制备访问、测量/层析协议和成本，不能直接把模拟器数组索引称为测量。
- x <- normalize_row(x + f)一般不是对未知保留态施加的确定性unitary。可以研究测量重制备、辅助比特/后选择等实现，但成本和计算模型会改变。
数学反例见03。这里不声称相关函数在任何量子访问模型下都不可实现，也不把“可经典模拟”本身当作QGNN失格条件。

### 5.2 作者公开代码核验

通过GitHub连接直接读取SnehalRaj/mp-qgnns，pin=851537589d61bcce96130b055e5724291e7ea318，src/mp_qgnns/models/cfi.py，blob SHA=6ea2183d64a7231472dbab4881d8e11fcd01c228。
其中EquivariantQGNN采用同类Johnson exp(i alpha A)、添加encoder输出、逐行归一化、实虚部sum-pool。因此该区别不仅来自ICCT迁移，也存在于作者特定reduced-basis实验实现与论文更完整的两寄存器量子叙事之间。不能由这个文件否定整篇论文的所有定理或其他实验实现。
论文：arXiv:2606.26873v1，2026-06-25。官方入口：https://arxiv.org/abs/2606.26873
代码：https://github.com/SnehalRaj/mp-qgnns/blob/851537589d61bcce96130b055e5724291e7ea318/src/mp_qgnns/models/cfi.py

### 5.3 Baseline匹配

Candidate A的quantum mixer是带权图条件演化，而成功小样本/全量比较的Johnson core并非完全相同的带权动态算子。它仍是有价值的强对照，但不构成只改变quantum/classical substrate的严格隔离。必须补充信息权限与带权关系匹配的classical counterpart；不得削弱原baseline以制造优势。
固定j=2/3、D=6,k=3时，子集态维度为20*C(N,j)。N=8时分别560和1120个复振幅，可经典有效处理。与完整2^(N+D) statevector不同；56-qubit子空间模拟不能解释成任意56-qubit线路模拟或指数计算优势。

## 6. Paper-fidelity隔离实验：不是一个开关解释全部差距 [RESULT]

协议：4096 train/1880 val，20ep，0dB/seed2026；以下是关联但不同结构的原始实验，不作跨结构因果推断。

| 实验 | ADE | FDE | J |
|---|---:|---:|---:|
| Full paper-math量子链路 | .810536438 | 1.707538617 | 1.664305747 |
| 带权、无local旁路的classical control | .768920247 | 1.568731510 | 1.553286002 |
| Intermediate量子：保留additive/row-normalized路径、observable读出 | .754217271 | 1.542620952 | 1.525527747 |
| Full paper-math仅改rowlocal reupload | .810507801 | 1.708257703 | 1.664636653 |
| Full paper-math仅将adjacency组合改为exp(iH) | .783533251 | 1.619350527 | 1.593208514 |

最后一项原始summary已COMPLETED、最佳epoch5，wall436.52s；较原full math有部分恢复，但没有追上带权classical。早期hand-off中若仍把它列为待完成，应以此服务器结果为准。
更换classical feature builder的隔离J变化约.081%，不足独立解释大幅性能差距。
上述full-math路径也改变了local旁路、读出与中间规范化等结构。不能据A好于full-math就断言“物理量子机制一定更差”，也不能把所有损失归结为单一最终M或条件投影。

## 7. 对“64维压缩瓶颈”的证据修正 [RESULT/ANALYSIS]

以当前主JSON为准，而非较早rerun.log：
reports/qgnn/round4_representation_bottleneck_audit_20260919.json。
协议：冻结R2 Q/C；4096 train/1880 val，0dB；train-only scaler/PCA，common probe width128；probe目标是strong classical core的full-local表示，不是future trajectory。报告的val_r2分母以train目标均值为基准，应按其定义解释。

| 被探测表示 | 维数 | reported val_r2 |
|---|---:|---:|
| current_readout_input | 500 | .921636263 |
| compressed_interaction64 | 64 | .824465985 |
| local_only64 | 64 | .871815089 |
| final_graph64=local+interaction | 64 | .918706656 |

单独interaction64看起来损失较大，但真正交给下游的final64接近500维输入；local-only已能预测不少teacher proxy。因此尚不能认定“64维就是主瓶颈”。该探针也不证明final64保留了所有未来相关信息；拟合teacher并不等于ADE/FDE改善或完整邻车因果推理。

## 8. SQM边界 [CODE/PAPER/ANALYSIS]

当前没有ICCT SQM模型或训练结果。论文2601.18198v1的邻车-边U_MSG不触碰中心，随后U_UPD顺序融合到中心；星形采样、共享参数和量子聚合值得保留。
但是明确的门级U_MSG/U_UPD、邻车顺序不变性、多维输入编码和读出宽度仍要补齐。共享参数不自动保证共享中心上的非交换更新顺序无关；数学反例见03。本次GitHub按SQM-GNN检索未返回可确认作者仓库，不能据此声称代码不存在。
不把D2D功率控制的参数量与性能直接迁移成SinD轨迹预测的优势证据。

## 9. 决策影响

保留Raj A作为历史数值参照，而不是默认Primary；保留其subset建模思想，但合法可测量版本须独立定义。RC与Scene-Adaptive作为失败/边界证据，不直接再次堆控制器或通道数。SQM按机制模板研究，不补写作者未说明门组后声称论文复现。
后续比较要同时回答：目标上下文是否完整、邻车联合计算在哪里发生、读出是否可执行、相对于同信息强经典机制为何可能改善最终轨迹。

## 10. 主要本地来源

- docs/qgnn/FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md:196-276
- docs/qgnn/ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md
- docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md
- docs/qgnn/ROUND3_INDEPENDENT_ACCEPTANCE_20260919.md
- docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md
- docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md
- docs/qgnn/ROUND4_RAJ_FULLTRAIN_FINDINGS_20260919.md
- docs/qgnn/ROUND4_RAJ_PAPER_FIDELITY_AUDIT_20260919.md
- docs/qgnn/ROUND4_RAJ_PAPER_MATH_WEIGHTED_CLASSICAL_FINDINGS_20260919.md
- docs/qgnn/ROUND4_RAJ_INTERMEDIATE_WEIGHTED_MATCH_FINDINGS_20260919.md
- docs/qgnn/ROUND4_RAJ_FEATURE_BUILDER_ISOLATION_FINDINGS_20260919.md
- docs/qgnn/ROUND4_RAJ_ROWLOCAL_REUPLOAD_FINDINGS_20260919.md
- prediction/qgnn_final/quantum.py; relational.py; common.py
- prediction/qgnn_paper_native/raj_subset.py:54-97; raj_paper.py:291-326; raj_paper_math.py; raj_mechanism.py:144-224
- scripts/audit_qgnn_representation_bottleneck.py:87-134
- reports/qgnn/round4_raj_fulltrain_quantum_0db_seed2026/summary.json
- reports/qgnn/round4_raj_fulltrain_johnson_0db_seed2026/summary.json
- reports/qgnn/round4_raj_paper_math_expadj_quantum_4096_seed2026/summary.json
- reports/qgnn/round4_representation_bottleneck_audit_20260919.json
