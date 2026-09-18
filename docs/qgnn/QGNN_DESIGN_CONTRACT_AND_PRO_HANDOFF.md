# QGNN 设计合同与 GPT-6 Pro 交接说明

更新时间：2026-09-18
分支：qgnn
状态：用于下一阶段“候选路线空间探索”，不是最终架构冻结稿。

## 1. 项目任务
Automatum T-Crossing → 3 个固定 ISAC 基站 → 冻结 sensing cache → 每车历史状态 [x_hat, y_hat, vx_hat, vy_hat] → 图模型（Classical GNN / QGNN）→ 共享 GPT-2 + LoRA → 未来 20 步轨迹 → ADE / FDE。

冻结接口：
- history = 20 frames
- future = 20 frames
- 2 <= N <= 8
- state_dim = 4
- SNR = {-10,-5,0,5,10} dB
- history 来自 ISAC sensing cache；future 来自 Automatum GT
- 当前没有 lane ID / route / HD map / traffic light / maneuver label / driver intent

## 2. 最终目标
主目标不是“做出一个量子模型”，而是 QGNN 在强、合理、经典 baseline 下，对 ADE 和 FDE 取得有实际量级的稳定优势。

一级目标：尽量在每个 SNR 上都达到 ADE >= 5%、FDE >= 5% 相对提升。
二级可接受目标：若无法做到每档 SNR >= 5%，则五档 SNR 宏平均 ADE >= 5%、FDE >= 5%。

要求：不能依赖单一 seed 偶然结果；不能挑最弱 baseline；不能通过给 QGNN 额外信息、削弱 baseline 或 cherry-pick 制造优势。

## 3. Baseline 原则（已冻结）
Baseline 必须全部是经典方法。至少覆盖两类：
1. strong pairwise classical GNN：候选 Edge-MPNN / GIN / GAT 等，后续筛选。
2. classical higher-order GNN：候选 PPGN / k-IGN / 其他适合小图高阶交互的经典方法。

量子模型（如 Edge-Local QGNN）只能作为 quantum ablation / quantum comparator，不能称为 baseline。
最终主 QGNN 要和“最强合理经典 baseline”比较。

## 4. QGNN 定义边界（硬约束）
不接受：Classical GNN → 把某个 MLP / Attention / Linear 换成 VQC → 命名 QGNN。

要求至少满足：graph structure directly controls quantum computation。

即 node / edge / subset relation 必须直接参与 quantum encoding / interaction / message / aggregation / higher-order quantum evolution 中的核心图计算环节。

允许 classical preprocessing：
[x,y,vx,vy] → normalization / projection / physics-derived features / light temporal encoder → quantum graph core。

但不能由 classical GNN 先完成主要图推理，再让 PQC 做后处理。

## 5. 参数量约束
目标 P_QGNN <= P_strong_classical，最好 P_QGNN << P_strong_classical。
“参数更少”是本项目主动追求的 parameter-efficiency 目标，不预设为一般量子理论定律。

## 6. LLM 模块定位
GPT-2 + LoRA 是原项目已有核心模块，Stage0 / Stage1 已实现并验证。
当前研究不重新设计 LLM。
所有正式 GNN / QGNN 主比较使用相同 downstream LLM / temporal predictor，主要变量集中在 graph module。## 7. 训练与硬件约束
服务器：2 x RTX 4090 24GB。
Python 环境：/home/dell/YrM/envs/ICCT。
PennyLane 0.45.1；已安装 pennylane-lightning-gpu 0.45.0 + cuStateVec。

训练预算：
- 单次正式训练理想 <= 2h
- 2-4h 可接受
- >4h 需要很强理由
- ~12h 基本不接受

qubit 数不提前冻结。
正确流程：候选架构 → 推导 qubit / circuit-call scaling → GPU 实测 → 判断能否满足约 2h 训练预算。
尤其避免 circuit-call 数随 T*|E| 无控制增长。

## 8. SNR 策略
允许每个 SNR 单独训练一个 QGNN。
是否最终一模型覆盖五档 SNR，还是五档分别训练，由性能与总训练成本决定。

## 9. 任务定制原则
允许且鼓励针对 Automatum + ISAC + 多车轨迹预测专门设计 QGNN，不要求忠实复现某篇论文。
文献用于提供理论依据、quantum graph mechanism、baseline 参考、代码、失败经验、对称性/表达能力/复杂度证据。
最终模型可融合多篇思想，但必须有清晰主机制。

## 10. 当前高价值文献/机制线
- Raj et al. 2026：higher-order subset representation、permutation equivariance、set-j-WL、官方完整代码；经典 baseline 可参考 GIN / k-IGN / PPGN。
- SQM-GNN 2026：wireless graph、node+edge quantum encoding、U_MSG quantum message、U_UPD quantum aggregation/update。
- Edge-Local QGCN 2026：官方 PennyLane+PyG，工程参考 / quantum comparator，不作为经典 baseline。
- Skolik / equivariant quantum graph circuits：graph weights 直接控制 quantum evolution，permutation-equivariant design。## 11. 当前候选主路线
A. Higher-Order Subset QGNN
- 主机制：显式建模 {i,j,k} 等多车联合关系
- 理论锚点：Raj 2026
- 风险：迁移到连续物理 edge / 动态轨迹后，原 WL theorem 不能直接照搬

B. Full Quantum Message + Aggregation
- 主机制：message generation 与 neighbor aggregation 都在量子态中完成
- 结构锚点：SQM-GNN
- 风险：若 edge-by-edge circuit call，会违反训练时间约束

C. Physics-Conditioned Global Quantum Graph Dynamics
- 主机制：车辆物理关系直接控制整图 quantum interaction / Hamiltonian / evolution
- 目标：尽量一次 quantum evolution 处理完整 <=8 车小图
- 优点：有机会减少 circuit calls，并保留全图联合关系
- 风险：需要更多任务定制与理论验证

允许 A/B/C 合理融合，但最终论文最好只有一个清晰“主机制”。

## 12. 尚待验证的核心科学问题
1. Automatum 是否存在普通 pairwise MPNN 难以利用、而 explicit higher-order interaction 能改善的瓶颈？
2. 如果 higher-order 有效，经典 higher-order GNN 是否已经足够？
3. 量子机制能否在不增加参数量的情况下进一步带来 >=5% ADE/FDE 收益？
4. 哪种 quantum graph representation 能在 <=8 节点、20 帧历史、2x4090、约2h训练预算下实现？
5. 如何最大限度降低 circuit calls，同时让一次 quantum evolution 承担尽可能多的 graph interaction？

## 13. GPT-6 Pro 的任务
请以“独立研究顾问 + 审稿人”身份，从上述任务、成功标准和硬约束出发：
- 系统检索 2023-2026 QGNN / equivariant quantum graph / higher-order quantum graph / wireless quantum graph 文献与代码；
- 补全、拆分 A/B/C 或提出更优新路线；
- 对每条路线给出：核心机制、>=5% 的潜在依据、最强经典 baseline、是否真正属于 QGNN、qubit scaling、circuit-call scaling、参数量趋势、2x4090 训练成本风险、GPT-2+LoRA 接口、最大理论/实验风险；
- 特别寻找强经典替代，避免把 higher-order improvement 错当 quantum advantage；
- 最终输出 1-3 条最值得进入原型验证的路线，但不要强行给结论。

证据强度必须与结论强度匹配；不能预设 QGNN 一定优于经典 GNN。
