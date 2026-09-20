# Mechanism landscape and references

 外部机制与六类候选的淘汰

|Family|量子状态/联合处理|成本与主要风险|决定|
|---|---|---|---|
|独立edge PQC/QGAT|一pair编码，测量后经典聚合|Q固定但calls O(TKR)，联合推理多数在经典端|淘汰|
|SQM star|node+edge量子消息进入center|标量Q=2K+1；每call仍K门及shots；富特征/邻车间边/顺序等变未解决|淘汰|
|时间有序联合register|M车同一态，node+全部局部pair分段上传；族间非对易|Q=2M，O(TSR(M+E)2^(2M))；训练缓存与readout风险|Primary|
|rooted subset walk|r1/r2邻车子集，受控编码后weighted hopping|Q=K+2，reduced4*C(K,r)；多控加载硬件成本高|唯一Backup|
|显式ZZZ/hypergraph/QHA|高阶Pauli相位，后续旋转使其可测|完整高阶枚举O(M^3)或更高；RC总体负；parity理论不覆盖强tensor|淘汰|
|walk/TFIM特征+强经典图网络|量子统计作GraphTransformer输入|单粒子walk可能O(N^3)经典模拟；joint TFIM指数；经典主导推理|不作主方案|

F1控制为同projection/value的edge MLP/GATv2；F2为同sampled K的Gated-MPNN；F5为同三元物理信息的tensor/hypergraph；F6为同backbone的heat-kernel/random-walk/complex-dynamics替换。不得把Q采样与C全图比较宣称加速。

文献重点：Skolik/EQGC提供图条件幺正和等变构造；Raj提供subset思路，不提供同j Johnson下的ADE保证；QHA的有限预算单层attention/parity与局部trainability条件不能外推强temporal tensor；HyQuRP提供qubit block对称思路，不为本项目强加SU(2)；TFIM/CTQW的量子演化时间不是车辆历史时间。Fourier去量子化、频谱系数限制与BP/可模拟性文献构成最强反证框架。[L1-L16]



 来源索引（自包含）

服务器P路径均相对项目根。
P1 docs/dataset_migration/SIND_MIGRATION_HANDOFF.md; configs/sind_prediction.json。
P2 frontend/sind_prediction_dataset.py; prediction/q0/motion_token_llm.py:122-249; prediction/qgnn_final/model.py:13-91; prediction/q0/temporal.py:14-54; configs/qgnn_final_tokens.json。
P3 tools/data_preprocessing/build_sind_high_interaction.py:170-295; frontend/controlled_isac/{sind_frontend,automatum_frontend,automatum_measurement}.py; reports/sind/sind_isac_validation.json:4071-4093。
P4 docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md; reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json。
P5 prediction/quantum.py; experiments/qgat_value/graph.py; reports/qgat_formal/summary.json; reports/qgat_factorial_full/seed2026/。
P6 docs/qgnn/FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md; prediction/qgnn_final/{quantum,relational,common}.py。
P7 docs/qgnn/ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md; HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md。
P8 reports/qgnn/round4_raj_weighted_multij_{quantum,johnson}_20e_2026/{summary,training}.json;对应20e_seed2027/summary.json; round4_raj_fulltrain_{quantum,johnson}_0db_seed2026/{summary,training}.json。
P9 prediction/qgnn_paper_native/raj_subset.py; raj_paper.py:123-147,291-326。
P10 docs/qgnn/ROUND4_RAJ_PAPER_FIDELITY_AUDIT_20260919.md; reports/qgnn/round4_raj_paper_math_expadj_quantum_4096_seed2026/summary.json。
P11 docs/qgnn/QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md; SQM_GNN_UMSG_SOURCE_BOUNDARY_20260919.md。
P12 reports/qgnn/round4_representation_bottleneck_audit_20260919.json。

L1 Skolik等，Equivariant quantum circuits for learning on weighted graphs，npjQI9,47(2023): https://www.nature.com/articles/s41534-023-00710-y 。正文结构/定理/trainability阅读；官方 https://github.com/askolik/eqc_for_nco README.MD核验，非运行。
L2 Mernyei等，Equivariant quantum graph circuits: constructions for universal approximation over graphs(2023): https://doi.org/10.1007/s42484-022-00086-w 。正文构造；官方notebook目录 https://github.com/pmernyei/eqgc-experiments 。
L3 Raj等，Scalable Message-Passing QGNNs in the Weisfeiler-Leman Hierarchy: https://arxiv.org/abs/2606.26873v1 。主文/关键supplement及官方CFI代码； https://github.com/SnehalRaj/mp-qgnns commit851537589d61bcce96130b055e5724291e7ea318, src/mp_qgnns/models/cfi.py, blob6ea2183d64a7231472dbab4881d8e11fcd01c228。
L4 Scalable Quantum Message Passing GNNs for Next-Generation Wireless Communications: Architectures, Use Cases, and Future Directions: https://arxiv.org/abs/2601.18198v1 。III节/复杂度阅读，未核实完整官方gate代码。
L5 Higher-Order Token Interactions via Quantum Attention: https://arxiv.org/abs/2606.11673v1 。架构/定理条件/梯度讨论，单层受限attention非本项目强基线。
L6 HyQuRP: https://arxiv.org/abs/2602.06381 。block对称/方法阅读；作者代码 https://github.com/YonseiQC/equivariant_QML 本轮未逐文件审计。
L7 On the Expressive Power of the Transverse-Field Ising Model for Graph Learning: https://arxiv.org/abs/2608.17750 。方法和feature control。
L8 CTQWformer: https://arxiv.org/abs/2605.09486 。方法筛选，walk与classical transformer区别。
L9 A2QTGN: https://arxiv.org/abs/2605.21916v2 。temporal方法筛选，经典memory边界。
L10 Quantum Graph Attention Network: A Novel Quantum Multi-Head Attention Mechanism for Graph Learning: https://arxiv.org/abs/2508.17630v3 。方法/元数据筛选。
L11 Quantum Graph Attention Networks: Trainable Quantum Encoders for Inductive Graph Learning: https://arxiv.org/abs/2509.11390 。架构筛选。
L12 Sweke等，Potential and limitations of random Fourier features for dequantizing quantum machine learning，Quantum9,1640(2025): https://quantum-journal.org/papers/q-2025-02-20-1640/ 。摘要/定理范围，不当作普遍去量子化定理。
L13 Mhiri等，Constrained and Vanishing Expressivity of Quantum Fourier Models，Quantum9,1847(2025): https://quantum-journal.org/papers/q-2025-09-03-1847/ 。摘要/理论总结。
L14 Cerezo等，Does provable absence of barren plateaus imply classical simulability?，NatureComm16,7907(2025): https://www.nature.com/articles/s41467-025-63099-6 。论证/例外，不是全体QML否定定理。
L15 Caro等，Generalization in quantum machine learning from few training data(2022): https://pubmed.ncbi.nlm.nih.gov/35995777/ 。bibliographic/abstract核验，publisher全文本次未取得。
L16 I2XTraj: Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything: https://arxiv.org/abs/2501.13461 。输入/方法阅读，额外map/light与多模态指标不作同权限排行。

文献筛选非穷尽systematic review，阅读深度不同，不宣称逐篇复现。PDF截图服务对Raj/SQM/QHA失败，采用取得的正文/HTML/代码，不声称已目视图中未解析内容。没有从这些primary来源取得当前SinD+ISAC+同GPT2下量子严格胜强经典的现成定理。

## 最终声明

TO-JQGNN是最值得进入下一轮工程原型的Primary，T-SQW-GNN是唯一Backup。核心可证伪问题：受约束、时间有序的联合量子表示能否在相同信息与预算下，比强temporal higher-order classical更有效地改善未来轨迹。任务已完成理论选型，不预支实验结论。
