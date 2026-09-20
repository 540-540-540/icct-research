# ICCT 轨迹预测主任务选择报告

日期：2026-09-20
状态：**主设计已选择；数据纳入统计已核验；新任务 Gate S/G 尚未获得实验通过证据；Gate Q 关闭。**

## 1. Executive conclusion：唯一主推荐

**我推荐 ICCT 正式主任务采用 SinD-IC4：约 2 秒历史 → 4 秒未来、逐目标 ego-centered 邻域、history-only interaction-critical 纳入规则，以及 Self-motion + interaction-conditioned residual 预测。**

具体为 20 个历史状态、40 个未来位置、原始 9.99 Hz 时间轴；主评测目标至少具有 1 个当前可识别的潜在冲突/跟驰邻居。至少 2 个交互邻居仅作为预注册的高阶分层，不作为主任务门槛。唯一辅助 benchmark 为同一时域、同一划分的完整历史合格目标集 SinD-Full4。

这个决定不是“4 秒已经实测最优”，更不是“4 秒一定让 GPT-2 或 QGNN 赢”。它是根据物理目的、已提交项目证据、文献和一次不看新预测误差的数据审计选择的可执行主设计。**本轮没有训练新的 self/strong-graph 模型，不能证明 Graph Gain 随 horizon 扩大。** 下一轮应先验证 Gate S/G，而不是直接训练 QGNN。

若预注册 Gate G 在 train-design 失败：不改 DCPA/TTC/密度阈值追求显著性，不迁移到“恰好能赢”的子集；备用 Y 就是已经保留的 Full4 普通轨迹预测任务，并撤回“交互必要性”与 QGNN 主线资格。不是再开放一串候选 benchmark。

### 本轮实际完成范围

只使用正式 qgnn 已提交快照 `e94d61989333f0b93b1ddf25f4423c08defb49e9`，新分支 `task-redesign-gpt6pro-20260920`，独立 worktree `/home/dell/YrM/ICCT_task_redesign_gpt6pro`。正式目录中其他 agent 的未提交产物未读取为本轮证据，也未修改、移动、stash、commit 或清理。未 push/merge/cherry-pick。

已完成：任务书/代码/历史结果审计、原始数据独立下载并核对两个 SHA256、文献研究、阈值事先登记、20/30/40 帧历史纳入与未来可见性统计、因果纳入函数单元测试、精确设计与 Gate 冻结。未完成：新预测目标 tensor、完整新 sensing cache、强模型训练、多 seed 性能验证、formal test。研究设计结论与性能结论严格分开。

## 2. 为什么旧 2→2 任务存在问题，而不只是“太容易”

### 2.1 能核实的历史实验

来源 P1：`reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`。这是旧 SinD、0 dB、seed2026、20 epochs、1880 个 validation 场景的 Motion-Token LLM 控制实验。

| 交互模块 | ADE / m | FDE / m | 选中 epoch |
|---|---:|---:|---:|
| None | 0.524150 | 1.063804 | 17 |
| Pair | 0.502264 | 1.030864 | 20 |
| Routed pair | 0.501781 | 1.020530 | 17 |
| Pair + triplet | 0.498218 | 1.004633 | 17 |

Pair+triplet 相对 None 的绝对改善约 0.02593 m / 0.05917 m，相对改善 4.95% / 5.56%。相对 Pair 仅改善约 0.81% / 2.54%。它支持“邻车信息存在一些价值”，不能支持“强 classical self 明显不足”“高阶机制有稳定优势”或“量子优势”。单 seed、旧裁剪目标、不同模型独立选 checkpoint，使这些值不能替代新任务的配对实验。

用户报告的 LSTM/TCN/普通 Transformer ADE≈0.4 m、FDE≈0.9 m：**在本轮允许读取的已提交快照中未核验到对应正式训练结果，保留为 user-reported observation**。正式目录的 untracked `results/qgnn/` 可能含正在生成的结果，但本轮不绕过 worktree 隔离去读取它们，也不把旧 NGSIM/Automatum 日志冒充新 SinD 结果。

P2：`reports/qgnn/sind_target_self_repair/preflight.json` 的新 target-view CV 为 origin-macro ADE 0.784092 / FDE 1.629495；其中短 smoke 不是训练完成的模型能力证据。其 train/val 是 319855/31211 个 target windows，不能与旧 15802/1880 scene windows 直接比较绝对指标。

P3：`reports/q0/llm_marginal_effect_audit_0db.json` 的 Automatum 历史比较中，20-epoch simple self 为 0.434671/0.725519，GPT-2 self 为 0.543987/0.950401。它提示 LLM 接口/训练/归纳偏置可能有问题，但不构成当前 SinD、4-layer GPT-2 的 Gate S 判决。

### 2.2 构造代码暴露了比 horizon 更重要的问题

`tools/data_preprocessing/build_sind_high_interaction.py` 先要求每个候选车辆整个 history+future 完整，再对候选车辆按历史 GT 状态选图。因此“边的计算只看历史”不等于“context 集合不依赖未来”。20→40 若只修改 TOTAL，会随未来长度改变邻居集合，污染 horizon 比较。

旧 selection 是 `distance<=30 AND (closing>0.5 OR (0<TCPA<=4 AND DCPA<=10))`。接近但横向错开很远的车辆也可通过 OR 分支，车多、近、接近均不能单独证明行为依赖。旧窗口也只选择一个 focal 局部图，却监督其中多辆车，每辆被监督车辆并不一定拥有自己的关键邻居。

全轨迹 position/velocity consistency 过滤还使用了预测时点之后的轨迹。本报告的新纳入规则不继承这一动态整轨过滤，而按历史可用性纳入，保留“历史质量/标签缺失”审计标记。原始文件本身已经平滑，其是否使用非因果平滑仍未知，见风险章节。

`prediction/qgnn_raj_pennylane/residual.py` 的 ego_v1 已修复 Self 坐标方向接口，不能继续把历史旧接口的问题当成未修复缺陷。然而旧头仍是 20 future queries 和 16 m tanh 残差上限；新 40 帧任务不能直接重用其饱和边界。

## 3. Literature evidence：支持什么，不支持什么

以下文献均核验 primary source；完整 URL 见末尾 L1–L10 与 `literature_sources_20260920.json`。

| 来源 | 对本任务的实际作用 | 不可外推的结论 |
|---|---|---|
| L1 SinD 原论文，2022 | 数据系列与路口行为背景 | 原 Tianjin 数据量不是本项目两个公开录像的数据量 |
| L2 SinD2.0，2026 | 多城市信号交叉口；其 benchmark 示例包含 QCNet 50历史/60未来、Diffuser 31历史/52预测帧，并明确不是完全标准化排行榜 | 不可把其多模态/地图/不同历史长度结果与本项目 ADE 直接排序；不能据此声称 2→4 是标准答案 |
| L3 InterHub，2024 | 从行为依赖、潜在路径冲突理解交互，而非只看密度；采用运动外推等方法识别交互事件 | 完整事件起终点/回顾性片段标签不能原样作为在线筛选器；不提供“DCPA5m必然最优”证明 |
| L4 SMART，2024 | 时间与 agent 交互分解、motion token、自车局部坐标；方法中采用 50m 邻域 | GPT-style motion Transformer 不等于文本预训练 GPT-2；多模态模拟能力不能直接证明本项目确定性轨迹优势 |
| L5 One Fits All，2023 | 冻结预训练 attention/FFN 迁移到若干时序任务的正向证据，小数据设置尤其值得检验 | 不能保证短历史车辆预测比强 LSTM/TCN 好；其方法不是本项目接口的逐字复现 |
| L6 Are Language Models Actually Useful…，2024 | 多个受审时序方法中去掉/替换 LLM 或随机初始化仍可持平/改善，是必要反证 | 不能反向断言所有预训练迁移均无效，应在相同接口下检验 |
| L7 LoRA，2021/ICLR2022 | 冻结 W，以低秩 BA 适配，降低训练参数开销 | 不是轨迹准确率或小数据泛化的保证 |
| L8 KA-MGAT，2024 | 运动学与图网络残差结合已有方法依据 | 残差不自动具有“纯交互因果效应”含义 |
| L9 FHWA SSAM，2008 | TTC/PET 与冲突严重度的定义；例如 1.5s TTC 的冲突筛选语境 | TCPA≠TTC，不能把 4s TCPA 与 1.5s 安全冲突阈值混用 |
| L10 Argoverse 官方 guide | target/context 区分、局部轨迹建模与多秒场景设置 | 不同数据集的输入、任务时域、样本筛选不可混用 |

文献整体支持“中等时域 + 历史冲突结构 + 明确 temporal/context 分工”的问题设计；**没有文献能替本项目证明 GPT-2、GNN 或 QGNN 必然获胜**。50m 有直接文献先例；5m DCPA、0.5m/s closing 与跟驰几何规则是本轮可解释的预先工程选择，不能包装成行业标准。

## 4. SinD/data audit：新统计与旧统计分开

### 4.1 数据来源、时基、ISAC 身份

在本 worktree 独立下载 pinned SinD commit `930e4dea78d924c6e9a58ff8e378331f93bba8ec` 的 Changchun / Xi'an 两个 `Veh_smoothed_tracks.csv`，SHA256 与冻结构造报告逐个一致。没有复制正式目录的未提交 cache。

原始时间步为约 `100/999 = 0.1001001001 s`，即 **9.99 Hz**；原流程未重采样。20 个状态从第一点到最后一点跨 19dt≈1.901902s，通常所谓“2s历史”指 20-bin 的约2.002002s窗口。40 个未来点对应 t0+dt…t0+40dt，终点 **4.004004s**。后续论文必须写清约数，不能把实际时钟强制改成0.1s。

冻结 Route-B `sense_vehicle` 是针对每辆已知身份车辆生成 3-BS 几何测量并融合的受控感知估计器；其源码明确**假设 detection/identity association 成功，不是真正共享回波多目标检测系统**。本轮沿用、不修改、不把其身份与多目标 detection 混淆。个别状态仅2个基站可见，这不意味着配置少了一个基站。

本轮用同一 frontend、同一 calibration、同一历史状态 key 规则重建 0dB 最后历史观测来计算选择指标；没有导入正式 cache 做逐位对照，故“同协议重建”不写成“新旧完整 cache 已逐位验证”。历史输入完整20帧 cache 的复用/补缺属于下一阶段。

### 4.2 阈值登记与审计边界

`design_preregistration_20260920.json` 在新统计运行前写入。radius、DCPA、closing、TCPA、following 规则未根据统计或模型表现修改。train-only sensitivity 只用于评估采样敏感性，不反向选择阈值。

审计读取 train/val 历史来源状态，统计 future frame 是否存在，但**不构造未来坐标目标、不计算新 ADE/FDE、不训练模型**。原始下载文件整体含 test 行；扫描时只解析 test 的类别/ID/frame 元数据以核对分区与身份跨界，未转换其位置速度数值，未打开 test 预测标签。旧构造报告中已存在的 test count=2086 仅作为归档元数据引用。

### 4.3 三个 horizon 的可行性

每10帧创建一个 origin，target 是每辆历史完整车，不要求其未来完整。风险判据统一向前看40dt，与当前比较的预测头20/30/40帧无关；因此不会为较长 horizon 专门找更有利的目标。

| 预测帧数 | Full train targets / origins | IC train targets / origins | Full val targets / origins | IC val targets / origins | IC train/val终点可见目标 |
|---|---:|---:|---:|---:|---:|
| 20 | 36803 / 1896 | 12189 / 1527 | 3871 / 255 | 1149 / 180 | 11823 / 1102 |
| 30 | 36791 / 1894 | 12188 / 1526 | 3864 / 253 | 1149 / 180 | 11375 / 1047 |
| 40 | **36778 / 1892** | **12186 / 1525** | **3855 / 251** | **1149 / 180** | **10849 / 992** |

旧的 train/val/test scene windows 为15802/1880/2086；新 counts 与之不同，原因是 origin stride、逐目标定义、纳入规则和60帧保护带都改变，不是丢失文件。

由20变40，主 train 历史合格目标仅减少3个；主要代价是标签在视野中退出：IC4终点可见率 train **89.03%**、val **86.34%**。Full4相应为86.75%/85.21%。IC4中 train9、val2 个目标没有任何未来可见帧，仍保留在纳入清单，单独报告无法计分；有至少一帧可监督目标为12177/1147。不能把“完整40帧未来”改成纳入条件来让数据看起来更整齐。

上述未来可见数基于 frame 元数据；正式标签构建还需核验数值有限性，不能用元数据存在冒充已完成 GT tensor 验收。

主任务可计算ADE的origin数为train1525、val180；可计算40步FDE的origin数为train1492、val177。ADE与FDE的origin分母不同，所有模型必须使用同一套各指标mask和分母，不能以缺失终点时的最后可见位置代替40步FDE。

### 4.4 密度、交互分布、场景集中性

IC4占完整目标集 train33.13%、val29.81%。主 train 为长春10752、西安1434；主 val为长春1063、西安86。至少2交互邻居的 train/val为3829/273，其中 val为长春268、西安仅5。**不采用k>=2作为主门槛，避免把主任务进一步退化为单一录像。**

完整目标的50m邻居超过7个的比例 train80.79%、val68.92%。但本次审计没有任何目标的风险邻居数k超过7，因此“先保留所有风险邻居、再补近邻”的8-slot规则可以保留全部当前规则识别的直接风险边。它不保证保留所有预测信息，必须保留 all-neighbor classical 对照。

按 target 定义的 pair density为k/n50，wedge density为C(k,2)/C(n50,2)，分母为0时置0。Full train/val的均值分别约0.03655/0.03705与0.002499/0.002247；潜在target-centered三元wedge数总和7342/518。这是历史几何机会，不是高阶不可约交互的实证。

详细 n_active、n50、距离、closing、TCPA、DCPA、TTC 分位数与每个城市分布保存在 `history_only_audit_20260920.json`，不把“附近节点数多”当成风险标签。

主 val共有133辆不同目标车、180个 origins，但只有约10个非空30秒时间块，长春6、西安4。高频窗口及同一车辆大量重复，禁止以1149作为独立统计样本数。train/val总体比例接近也不代表城市/行为分布相同；需按城市分别报告。

### 4.5 阈值敏感性（只看 train、无模型误差）

| 改动 | IC4 train targets | k>=2 targets |
|---|---:|---:|
| 冻结 D=5,R=50,c=0.5 | 12186 | 3829 |
| D=4 | 9762 | 2301 |
| D=6 | 14223 | 5163 |
| R=40 | 12062 | 3660 |
| R=60 | 12226 | 3914 |
| c=0.3 | 12650 | 3920 |
| c=0.7 | 11905 | 3762 |

radius附近变化影响较小；DCPA敏感，不能声称 benchmark 已有“性能结论稳定性”。后续须在train-design对同一组固定模型做D=4/5/6的预注册敏感性报告，**不得因哪档结果最好而更换D=5**。

工程性样本下限冻结为主train>=5000 targets/500origins、主val>=500targets/100origins及>=50不同目标车、每个城市>=20不同目标车。这是本报告设定的可运行性下限，不是功效分析，不谎称在看counts前已计算统计功效。当前通过这些描述性下限，但少录像/少时间块仍限制显著性和泛化叙事。

## 5. Candidate task space：不是只改一个未来长度

候选轴分别为：H=20或30；F=20/30/40/50；纳入full/density-only/history-conflict/k>=2；global/local/CV residual/self residual；all-agent/target-centered/conflict-focal；确定性或多模态。

本轮优先隔离“未来长度”这一变化，保留20帧历史与原ISAC历史接口。2→4兼顾响应时间与视野可见性；不同时增加历史长度、引入地图、改感知噪声、改标签模态来制造优势。加速、减速、交叉冲突、跟驰、历史转弯都属于本数据合理的可研究行为；future lane-change/yielding/turning标签不能用于选样。

### 三种叙事的比较

A 长时域：合理，但只看到更大self误差不够，必须测 GraphGain40−GraphGain20。B 交互子集：使问题集中且可复现，但要防单录像集中、阈值敏感和删掉低交互完整基准。C 残差：职责清晰，但残差不等于纯因果交互。**最终采用A+B+C的受限组合**，每一段叙事各有独立实验条件，不让其中一段替其他段背书。

## 6. Candidate comparison 与取舍

| 候选 | 物理合理性/样本支持 | 核心问题 | 决策 |
|---|---|---|---|
| 20→20 Full | 稳定，旧流程可用 | 短时惯性+旧目标裁剪，已有增益小且非强self比较 | 否决为主任务，只作train-design horizon诊断 |
| 20→30 IC | 可构建，样本与40帧近似 | 没有显著样本收益，提前截掉更晚反应；尚无实测证据说它优于40 | 否决为主任务，不作为第二主benchmark |
| **20→40 IC4** | 纳入数量稳定；终点可见约86%–89%；对应中等时域反应 | 必须验证G，处理视野截尾、噪声累积与多模态 | **唯一主推荐** |
| 30→30/40/50 | 文献中更长历史/未来并不少见 | 同时改变两轴，重做history/cache，项目收益未知 | 本轮否决，不能无限延长直到某模型赢 |
| 20→40、k>=2硬筛 | 表面契合triplet | 西安val仅5目标，主结果单场景化 | 否决；仅固定分层 |
| 20→40、N大/距离近硬筛 | 容易实现 | 密度≠交互；静止车队/平行流也很密 | 否决 |
| future-maneuver/error筛选 | 常可制造大gap | 违反预测时可用性与公平性 | 禁止 |
| 多模态minADE_K替代主任务 | 能表达分叉 | 同时改变目标、模型、评测；K不同可制造优势 | 本轮否决，主K=1 |

这里没有伪造“候选总分”或新horizon性能表。2→2的旧图增益属于旧协议；2→3/4的强图与强self误差均未知。选择4秒的理由是可解释的研究设计，不是未运行实验的排行榜。

## 7. GPT-2 + LoRA role audit

### 7.1 当前代码实际做的事

`MotionTokenGPT2Core` 从本地预训练 checkpoint 加载 GPT-2，冻结 backbone；`FinalMotionGPT2` 只用4层、宽度768，attention fused-QKV 上 LoRA rank8、alpha16。运动 embedding、continuous adapter、history adapter、future queries、token head、coordinate/self head仍需训练。LoRA更新为

\[
W'=W+\frac{\alpha}{r}BA.
\]

只计算4层768→2304 fused-QKV的低秩参数，约为 `4×8×(768+2304)=98304`，与P2 preflight一致。旧ego_v1 smoke的总trainable约4.087M，远不止LoRA；**这不是新40帧模型已经测得的参数量**，新模型需重新分列 frozen backbone、LoRA、tokenizer/adapter、temporal head、interaction core 和readout。

运动词表从分散的GPT-2词embedding行初始化，并不赋予这些motion tokens自然语言交通语义。跨域可迁移部分主要在已学的attention/FFN/归一化表示与优化先验；新adapter必须把运动信号映射到这些计算结构能利用的空间。代码中已有连续速度/局部位置通道与ego_v1修复，不能只以“token没带速度”解释所有结果。

普通20帧历史也很短；未来查询由20扩为40并不自动制造语言模型的长上下文优势。4秒的运动复杂性、时间序列长度、语言预训练能力是三件不同的事。

### 7.2 为什么可能有效，以及如何证伪

L5支持冻结预训练表示/较少有效自由度可能改善某些小数据时序任务；L6提醒这种效果依赖模型、接口、基线与数据。LoRA可能正则化优化，但限制更新秩也可能限制跨域适配。Motion Token可能提供合理运动量化先验，也可能因量化/词表覆盖或辅助CE主导造成性能损失。参数规模、输入编码、冻结策略、预训练内容、数据量必须分开比较。

冻结最小主比较为：LSTM、TCN、scratch Transformer、GPT-2 frozen（adapter/head可训、无LoRA）、GPT-2+LoRA。补充两项不可省的容量控制：同4层768宽度、同因果mask的scratch GPT-2架构；与GPT-2总trainable参数相差<=10%的scratch Transformer，宽度从可合法分head的候选按参数数目选，不能按哪个最弱选。

同接口归因矩阵：pretrained/random backbone × LoRA有/无；同motion-token/continuous输入；辅助CE权重0/0.035；训练标签10%/50%/100% physical-track分组。低数据只是训练条件/分层实验，不是另开有利benchmark，更不能在100%数据失败后只报告10%。所有对应模型拥有相同历史状态信息，使用同坐标预处理和可比调参预算。ordinary Transformer不得被偷偷降宽、降层、少训练来让GPT-2胜出。

如果Gate S不通过，调整对象是LLM接口/优化/预训练策略或论文定位，而不是阈值。先审计motion-token覆盖、CE权重、head饱和、梯度尺度与冻结层数；旧ego_v1已修复的问题不得反复当作未做工作。经过这些公平检验仍无优势，应明确“文本预训练不是本任务已验证的准确率贡献”，允许best classical self接替，而非把GPT-2的存在本身当贡献。

## 8. Interaction necessity analysis：真正需要验证的量

令 H 为目标自身历史，N为邻车历史，Y为未来轨迹。在平方误差与理想Bayes条件均值下，

\[
m_S=\mathbb E[Y\mid H],\quad m_G=\mathbb E[Y\mid H,N],
\]
\[
\mathcal R_S^*-\mathcal R_G^*=\mathbb E\|m_G-m_S\|^2\ge0.
\]

这是条件期望的正交投影分解：`Y-m_S=(Y-m_G)+(m_G-m_S)`，交叉项期望为0。它解释邻车的条件信息为什么**可能**有用；不保证有限训练模型达到Bayes最优，不保证ADE/FDE下精确沿用该恒等式，也不等于数值估计了互信息 `I(Y;N|H)`。

连续运动可写作
\[
p(T)=p_0+v_0T+\int_0^T(T-s)a(s)ds.
\]
若邻车信息改变未来加速度的条件分布，较长horizon会对稍晚响应积累更大位移影响；如果邻车与未来运动在给定本车历史后近乎独立，延长horizon只会增加所有模型误差，Graph Gain并不会自然增加。未知意图、信号灯、地图、不可观测驾驶员动作也会导致误差增长，不能全部归给交互。

操作性检验：用同一批历史合格目标、相同标签mask，比较best self-only与strong classical graph的ADE/FDE配对差。比较20/30/40时不按模型错误选样；同时报告共同40-frame可见mask的配对分析和各horizon原生覆盖，避免截尾变化伪造增益。20/30只作train-design诊断，不升级为第二、第三正式benchmark。

额外的own-only residual MLP控制可检验“增加head容量”是否足够；DeepSets/context pooling检验不做message passing也能否利用邻车；邻居屏蔽/置乱只作为干预敏感性诊断，置乱可能产生分布外输入，不能当因果效应估计。

**对self的增益支持邻车信息有用，不逻辑推出某种GNN架构不可替代。** Pairwise多层网络也可能表示高阶关系；k>=2只是有三个参与者的几何机会，不证明不可约triplet，更不证明量子优越性。

### residual的正确解释

采用
\[
\hat Y=Y_{CV}+R_i\{f_S(H_i)+g(H_i,N_i)f_I(H_i,N_i)\}.
\]

`Y_GT−Y_CV`包含加速、转弯、感知噪声、交互和动力学误差；`Y_GT−Y_self`还包含self模型偏差。两者都不是经过因果识别的纯交互标签。报告和论文应称 **interaction-conditioned correction**，而不是“已分离的邻车因果位移”。

首轮使用冻结Self，让所有graph/own-only残差控制共享同一Self checkpoint。训练interaction residual优先使用train内按时间/车辆purge的3-fold out-of-fold self预测，避免用过拟合self在自身训练样本上的残差估计交互难度；推理换为全train self后需报告这种stacking分布差异。Graph与own-only residual控制都使用相同OOF过程。Joint微调只能作为后续消融，不能替代冻结Self下的贡献归因。

## 9. Final selected primary task：SinD-IC4

### 9.1 纳入公式

对最后一个历史感知时刻的车辆i、j，令
\[
r_{ij}=\hat p_j-\hat p_i,\ u_{ij}=\hat v_j-\hat v_i,\ d_{ij}=\|r_{ij}\|,
\quad c_{ij}=-\frac{r_{ij}^{\mathsf T}u_{ij}}{\max(d_{ij},10^{-6})}.
\]
\[
\tau_{CPA}=-\frac{r_{ij}^{\mathsf T}u_{ij}}{\|u_{ij}\|^2},\qquad
D_{CPA}=\|r_{ij}+u_{ij}\tau_{CPA}\|.
\]

若相对速度平方<=1e-8，令TCPA=∞、CPA条件false。对有限相对速度，
\[
C_{ij}=\mathbf1[c_{ij}\ge0.5]\,\mathbf1[0<\tau_{CPA}\le40dt]\,\mathbf1[D_{CPA}\le5].
\]

跟驰proxy：车辆速度均>=0.5m/s、速度方向夹角<=30°，j在i前方，纵向中心距离除i车速<=2s，绝对横向距离<=2.5m，记为F_ij。它是历史几何近似，不宣称已经识别同一车道或真实让行。

\[
A_{ij}=\mathbf1[i\ne j]\mathbf1[d_{ij}\le50]\mathbf1[C_{ij}\lor F_{ij}],
\quad k_i=\sum_j A_{ij},\quad \mathcal I_i=\mathbf1[k_i\ge1].
\]

主任务min active=2，不要求整个场景N>=8或每车k>=2。k>=2是固定子层，不能在看误差后另选更高k。所有量从同一0dB预测输入所对应的历史感知状态计算；不用GT当前位置、future maneuver、未来collision或模型误差选样。

TTC只作诊断。为明确它不等于TCPA，5m中心包络下解
\[
a\tau^2+2b\tau+c_0=0,
\quad a=\|u\|^2,\ b=r^Tu,\ c_0=d^2-5^2.
\]
若当前已在包络内，TTCproxy=0；否则仅在a>0,b<0,判别式非负时取首个非负根，其他为∞。这个5m包络不是实际车身碰撞判定，也没有采用“TTC<=4s”作为主门槛；CPA预期接近和紧急安全冲突不是同一概念。

### 9.2 目标、坐标和图

每个 `(scene_id,t0,original_track_id)` 最多一条目标样本。target slot=0；仅监督slot0的40×2位置，其余车辆只提供历史。ID、slot ID和scene ID不作为模型语义特征，ID仅用于数据索引与完全相同物理key的最终排序tie-break。

坐标原点取target最后感知位置，方向规则复用ego_v1：最近速度范数>=0.2m/s；没有时用最近可用历史位移，再用非零速度；全无方向时保存has_direction=false与有限缺省轴，不使用未来yaw。所有状态与预测残差只在明确的一次局部/世界变换下流转，避免Self局部输出与interaction全局输出误加。

邻居候选必须有完整20帧历史，不要求未来存在。半径50m。共同工程接口为target+最多7邻居，优先当前交互边，再按TTCproxy、有效TCPA范围内的DCPA、closing降序、distance升序、历史物理key排序；不用future预测/误差。选择前先算全半径内k，再裁剪context，不能以裁剪后k重定义benchmark。

Strong classical必须另保留50m全邻居版本，所有目标仍保留；若它比8-slot版本明显更强，不能为了QGNN接口只报告受限baseline。Raj内部feature/QNode不在本轮修改，未来匹配Johnson使用同feature函数与同decoder；强classical允许从相同原始历史构造更强物理特征，但须披露。

### 9.3 输出与loss

主输出为确定性未来二维位置，不是文本，不用oracle best-of-K。GT记录可以保留[x,y,vx,vy]，但主监督固定为[x,y]；速度如由预测位置差分求出，仅作为次要诊断，不能伪称已单独训练速度头。

新horizon-aware decoder统一使用无16m硬上限的线性coordinate输出、相同零初始化CV起点和梯度裁剪；旧模型代码保持不动。位置/速度归一化只使用train统计，完整原始时间戳保留；CV使用真实dt。为防把感知噪声外推误差误认成交互，除last-state CV外还需causal OLS-CV/CA诊断与强self历史滤噪能力。

所有模型使用相同Full4 train历史目标集合，不做依据误差的抽样，也不通过削减经典模型训练样本制造LLM优势。训练基本loss为全train origin-macro的ADE+0.5FDE；辅助token CE在匹配输入控制中对称配置并做0/0.035消融。一个origin含多个目标时先平均目标再平均origin，不能让车辆更多的origin自动得到更多权重。

## 10. 唯一辅助 benchmark：SinD-Full4

与IC4使用相同20→40时域、相同源数据/时间边界/目标/坐标/cache。唯一差异是取消k>=1，允许仅target自己存在。train36778、val3855目标，分别1892/251 origins。

IC4与Full4训练来源相同，评测口径不同。Full4显示方法在一般场景是否退化，k=0/1/>=2显示适用范围。禁止再增加一个“挑得更容易让GPT2赢”的辅助子集；following、crossing、history-turning、不同SNR仅作为本任务的预先定义诊断条件，不是多开主benchmark。

主SNR固定0dB。后续五SNR鲁棒性优先在Full4的同一历史目标群上比较；每个SNR的interaction分层只能按该SNR可见历史重新计算，并披露membership变化。不能拿0dB较好观测为-10dB模型选择“在线可识别”目标却不说明用了不可见信息。

## 11. Exact implementation contract 与数据重建

### 11.1 划分

| 录像 | outer train | guard | outer val |
|---|---|---|---|
| Changchun | [0,11998] | [11999,12058] | [12059,13628] |
| Xi'an | [0,7034] | [7035,7094] | [7095,8155] |

保持原已提交划分的主要cutoff，只把保护带扩为60帧，足以隔离20+40时域。跨outer split身份按frame元数据purge，重新构造邻域。不得随机拆重叠窗口。新test尚未构建；未来边界按相同规则，从g2+60开始，只有获得开启授权才生成其标签。

train内部设计集固定：Changchun fit[0,9598]、guard[9599,9658]、design[9659,11998]；Xi'an fit[0,5627]、guard[5628,5687]、design[5688,7034]。fit内部再切最后15%作为stopping-dev，floor规则得到Changchun subtrain[0,8158]、guard[8159,8218]、dev[8219,9598]；Xi'an subtrain[0,4782]、guard[4783,4842]、dev[4843,5627]。每次切分都按身份purge并重算候选邻居，不能直接复用全train的k。

本轮没有把这些内层dataset/cache完整构造出来，因此不虚构内层样本数。outer counts是已实际审计值，内层counts须由执行agent按上述确定规则生成并验收。

### 11.2 接口

```
input.history_state: float32 [B,20,8,4]  # same sensing contract
input.vehicle_mask:  bool    [B,8]
input.history_ts:    float64 [B,20]
input.target_slot:   0
meta.target_key:     (scene_id,t0,original_track_id)
meta.interaction_k:  int     # full50m historical candidate pool
label.future_xy:    float32 [B,40,2]
label.future_mask:   bool    [B,40]
label.future_ts:     float64 [B,40]
output.pred_xy:      float32 [B,40,2]
```

Full-neighbor classical版本使用相同语义的ragged/padded维度，不强制8。Raj保持 `[B,20,N,4]→[B,N,64]`；未来40步只要求新外围decoder接收其64D表示，不需要把40步塞进QNode。N=2、inactive j=3与padding必须重新preflight；不能通过删N=2样本逃避core兼容性问题。

### 11.3 label与cache规则

先固定history membership，再按frame/time key取GT label。future存在性只决定每个时间点的监督mask，不反过来改变target或neighbor集合。不插值填造丢失的未来，不把padding零当真实状态。无标签target保留于manifest、单列不可计分；FDE只对真实40th终点可见的目标计分并报告分母。推理失败/NaN算模型失败，不得从指标里删除。

更长future不要求对未来做ISAC；只扩展GT索引与query/head。新origins、新context、恢复的历史合格车辆仍可能产生原cache没有的历史keys。执行agent应只在独立新data namespace中复用已验证same-key cache、对缺失keys调用同一冻结frontend并做逐位抽样核验，不覆盖旧cache或旧split。raw source的未来frame存在说明GT可按key重建，但不等于40-step数据管线已运行完成。

历史转弯可定义为观测期首/末可用速度方向差>=15°（两端speed>=0.5），crossing proxy为CPA且相对方向在[30°,150°]。这些是预先描述层，不作纳入条件；merging/yielding/lane-change缺少冻结车道/优先权语义，不给伪真值标签。静态路口区域可作为后续辅助信息，但目前不靠手绘future轨迹挑conflict-zone。

### 11.4 checkpoint与测试封存

一个模型一个选中checkpoint，分数 `J=origin_macro_ADE+0.5*origin_macro_FDE`，在train内stopping-dev的IC4口径上选择，并报告同checkpoint两个指标。formal val不用于调阈值、epoch或多轮追故事。

配置/epoch由train内部确定后，允许按固定epoch数在outer train重训，formal val只作一次冻结确认。固定epoch取该配置内层重复中选中epoch的中位数，不能看formal val再增加训练。必须保存配置、manifest、checkpoint hash以及一次确认ledger。

formal test开启条件：源历史可用性/平滑问题有明确处理；data/cache与mask验收；S/G结果和最终方法身份写明；如保留Q则Gate Q及预注册Q-vs-classical评估完成；全部配置/checkpoint/metrics冻结；**用户明确授权**。test一旦评估不再回流调阈值或选checkpoint。本轮未开启。

## 12. Experiment matrix 与三个 Gate

### 12.1 最小分阶段矩阵

| 阶段 | 数据/模型 | 目的 | 本轮状态 |
|---|---|---|---|
| D0 | pinned raw + history metrics + metadata masks | 纳入合法性、样本数量、阈值敏感性 | 已运行；完整cache/labels未重建 |
| D1 | 20→20/30/40；CV、OLS-CV、LSTM、TCN、scratch Transformer、strong classical graph；同一历史群 | 比较实际Graph Gain曲线，不以self变差替代 | 未运行新训练 |
| S | frozenGPT、GPT+LoRA、经典self及容量/输入控制 | 判断预训练、LoRA、token和容量来源 | 新任务未建立 |
| G | bestself + own-onlyhead / DeepSets / strongMPNN8 / all-neighborMPNN / pair+triplet | 邻居额外信息、图结构、高阶机制分别归因 | 新任务未建立 |
| Q | 固定Self与decoder，Raj-PennyLane vs matchedJohnson + strongJohnson | 仅在G之后检验量子候选 | 关闭；本轮不训练 |

每个具体架构允许最多6组optimizer配置、同最大40 data passes；先train内部2 seeds筛配置，再冻结champions做5 seeds `2026..2030`。小模型、同宽scratch与LLM按各自合理学习率范围搜索，预算披露，不强制用明显不适配某类模型的单一学习率。建议经典lr={1e-4,3e-4,1e-3}、预训练适配/同宽深模型lr={3e-5,1e-4,3e-4}，均×weight_decay={0,0.01}；batch256 target rows为起点，显存不足按相同effective batch累积，global grad clip=1.0。新40-frame主比较的实际参数与速度必须另测，不引用旧profile当新实测。

量子core历史P1.1的B32前反向profile约4.175s且通过14项工程检查，是可执行性参考，不是本轮新40步训练耗时。self bypass QNode；在G/S不成立时不支付Q训练成本。40-query较39-token旧Self序列变为59 tokens，attention矩阵元素约增到2.29倍，FFN token数约1.51倍；这只是复杂度，不是壁钟时间承诺。N8 quantum core历史维度仍20帧，未来长度主要影响外围decoder。

### 12.2 Gate S — Self-model

Comparator是在train-dev按J选定的一种最强经典self，不是逐样本oracle。GPT-2+LoRA须在IC4同时达到：相对ADE>=3%、FDE>=5%；绝对ADE>=0.02m、FDE>=0.05m；5个配对seed至少4个两指标均改善；paired cluster bootstrap的两个差值95%CI下界>0；Full4任一主指标退化不超过2%。

这些是本项目预注册的最小值得报告效应，不是文献定律。S通过仍需同宽、同trainable参数和input/token对照解释来源。S不通过，不许调benchmark直到它赢；改变的是LLM贡献声明。本轮S状态 **NOT_ESTABLISHED**。

### 12.3 Gate G — 最重要的 Graph-necessity gate

Comparator是在相同train内部选择的best self-only（若GPT-2获胜则包含它）。strong graph须在IC4同时达到：相对ADE/FDE均>=10%；绝对ADE>=0.05m、FDE>=0.10m；5个配对seed至少4个两指标均改善；两个paired cluster95%CI下界>0；各城市平均方向不反向，且Full4任一指标退化不超过2%。同容量own-only residual不能解释全部增益。上述最低幅度也是研究准入标准，不是“必定会达到”的数值。

CI按录像分层、30s时间块配对重采样，并另做physical-track cluster敏感性；不能把窗口当独立样本，也不能把seed当新的交通样本。只有2个录像、val10个时间块，区间解释仅限这些录像的条件性证据，不能声称跨城市普遍优势。城市小样本不够时明确标示，不通过改阈值挽救。

“更长horizon扩大交互价值”另需 `GraphGain40−GraphGain20` 配对95%CI>0。即使4s上的G通过，这个差值未通过也只能写“4秒任务存在邻车条件增益”，不能写成horizon因果规律。

本轮G状态 **NOT_RUN_ON_NEW_TASK**。无新模型误差，所以本报告不写“已证明明显headroom”。若失败，停Q并退到Full4普通预测定位，不变更风险阈值。

### 12.4 Gate Q — eligibility，不是量子胜利门

G在train-design通过且固定formalval得到确认之后，才可考虑重启Q。还要求8-slot强图保留all-neighbor强图ADE/FDE增益的至少80%，否则当前Q工程接口不足，不能假装已匹配完整信息。新40步decoder、坐标、labelmask、N=2/j3不可用状态、padding、permutation、gradient和cache必须验收。

Q比较共享Self checkpoint、train数据/样本、future decoder、损失、checkpoint规则；同时保留参数matched Johnson与强Johnson，不只保留弱的matched模型。Raj core保持P1.1结构，本轮不动它。物理风险k>=2并不为quantum胜率背书；若要声称高阶必要性，另需strong pair+triplet对strong pair的贡献证据。未证实前只能作为待检验机制。

当前P1/P2通过代表QNode/梯度/接口工程检查；旧Raj weighted multi-j full-data与Johnson几乎打平的结果不是新PennyLane优势。**本轮Gate Q=CLOSED。**

## 13. Scientific risks 与正式采用阻塞项

**历史源平滑的因果性未证实。** 新selector只读history且通过接口测试，不等于`Veh_smoothed_tracks.csv`的上游平滑只用了过去帧。新增过去窗口再做一次因果滤波，不能抹掉原文件可能已经引入的未来信息。正式声称prediction-time available前必须核验生成流程或以原始未平滑观测重建causal history；否则应明确限定为“offline smoothed-state sensing simulation”，并让核心审查决定是否接受。当前不能签发严格在线benchmark已合规的结论。

**路口意图与静态语义缺失。** Graph可能学习邻车作为红绿灯/道路几何的代理，而不一定识别真实驾驶交互。多车信息胜过self不能排除这种解释。不能把所有误差都标成interaction-induced；后续若增加静态地图/当前信号，self/graph必须对称享有。

**视野截尾。** 4秒endpoint缺失不是随机事件，可能集中于高速离场车辆。相同mask能保证模型间公平，但不能证明评测群代表所有未来；必须报告覆盖、每时刻误差与被截尾目标数量。

**密度与高阶分层集中。** k>=2的Xi'an val只有5目标，不支持跨城高阶优势结论；长春数量多也不等于场景数量多。不能用bootstrap生成不存在的新场景。

**感知假设。** 当前controlled frontend没有真正多目标检测、漏检、关联错误；本轮不修改ISAC，也不作这些能力声明。0dB选择阈值对更低SNR不自动稳定。

**模型归因。** 新head、不饱和输出、token/连续编码、总trainable差异和容量都可能改变排名。公平同接口消融优先于叙事。未经新训练，S/G/Q的准确率或增益均未知。

## 14. 明确否决与失败策略

否决旧2→2作为正式主任务；否决2→3与2→4并列主线；否决靠增加历史长度/更长未来无限搜索；否决N多即交互强；否决k>=2硬门槛；否决用future完整性裁掉邻居；否决future轨迹/未来动作/某模型残差挑样本；否决绝对全局坐标作为主target；否决将CV residual或self residual直接命名为已识别纯交互；否决把多模态oracle指标切换当作准确率提升；否决只给量子保留弱baseline。

备用只允许已有的Full4普通预测定位：若G失败，报告失败，停止Q比较；若G通过而S失败，保留interaction任务但撤回GPT-2优越性声明，允许best classical self。若G通过而8-slot信息不足，先解决容量/接口问题而不是筛掉拥挤样本。若source smoothing无法满足可用性要求，不能仅靠换阈值升级为在线任务。

## 15. Final paper narrative 与交接计划

目前可写的事实：采用受控多基站历史状态估计，在预先定义的中等预测时域与历史潜在冲突群上，研究本车运动规律和邻车条件信息的不同作用；使用可解释的Self+context residual接口进行公平对照。既不预设文本预训练有优势，也不预设量子机制优于强classical。

只有S通过，才写“GPT-2+LoRA较经典self有稳定实证收益”；只有G通过，才写“邻车历史在所研究数据/条件下提供显著预测增益”；只有horizon差分通过，才写“长horizon放大该增益”；只有高阶对照通过，才写“更高阶关系在该分层上有额外价值”；只有Q最终公平实验支持，才写相应量子模型收益，且不能将仿真器准确率收益写成量子计算加速或普遍quantum advantage。

下一阶段顺序：先source/ISAC作用范围审查与新data namespace重建；验证history membership、未来mask、cache keys、坐标与40-query头；补齐并核验经典self及LLM；在train内部执行预注册horizon/S/G矩阵；固定formalval确认；满足Q资格才批准Raj-QGNN比较。不得自动启动正式Q训练，不自动开放test。

本轮交接建议：**采纳IC4/Full4这一唯一设计作为下一轮实施合同；不批准把S/G写为已通过，不批准直接进入QGNN full training。** 研究主报告与机器摘要、原始审计脚本及可重复metadata manifest一并提交临时分支，保持正式qgnn工作目录不动，等待核心审查。

## 16. 可重复产物与来源索引

运行仅作用于本worktree的命令：

```bash
python tools/data_preprocessing/download_sind_public.py --root data/task_redesign/raw
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dell/YrM/envs/ICCT/bin/python scripts/task_redesign/audit_history_only.py
/home/dell/YrM/envs/ICCT/bin/python scripts/task_redesign/complete_audit.py
```

分析脚本不实例化训练模型，不打开test动态状态，不写现有model/dataset/cache目录。`history_only_audit_20260920.json`保存所有horizon/城市分布；`audit_supplement_20260920.json`保存13项测试、密度/截尾统计与项目证据SHA256；`history_only_membership_20260920.csv.gz`保存可复现目标纳入与可见性元数据，不含未来位置/误差。CSV未压缩版本及运行log留在worktree但不提交，raw文件保持data目录忽略状态。

关键项目证据：P1=`reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`；P2=`reports/qgnn/sind_target_self_repair/preflight.json`；P3=`reports/q0/llm_marginal_effect_audit_0db.json`；数据构造=`reports/sind/sind_dataset_build.json`与`tools/data_preprocessing/build_sind_high_interaction.py`；Q工程状态=`reports/qgnn/raj_pennylane_p1_1/repair_preflight_20260920.json`、`reports/qgnn/raj_pennylane_p2/preflight_20260920.json`。这些是快照证据，不宣称包括其他agent当前尚未提交的结果。

### 文献链接

[L1] SinD: A Drone Dataset at Signalized Intersection in China (2022). https://arxiv.org/abs/2209.02297
[L2] SinD 2.0: A Multi-City UAV Dataset with Semantic Risk Annotations for SOTIF-Oriented Safety Validation at Signalized Intersections (2026). https://arxiv.org/html/2607.16943v1
[L3] InterHub: A Naturalistic Trajectory Dataset with Dense Interaction for Autonomous Driving (2024). https://arxiv.org/html/2411.18302v2
[L4] SMART: Scalable Multi-agent Real-time Simulation via Next-token Prediction (2024). https://arxiv.org/html/2405.15677v3
[L5] One Fits All: Power General Time Series Analysis by Pretrained LM (2023). https://arxiv.org/html/2302.11939v6
[L6] Are Language Models Actually Useful for Time Series Forecasting? (2024). https://arxiv.org/html/2406.16964v2
[L7] LoRA: Low-Rank Adaptation of Large Language Models (2021). https://arxiv.org/abs/2106.09685
[L8] Kinematics-Aware Multigraph Attention Network with Residual Learning for Heterogeneous Trajectory Prediction (2024). https://ieeexplore.ieee.org/document/10586904/
[L9] FHWA SSAM Software User Manual (2008). https://www.fhwa.dot.gov/publications/research/safety/08050/
[L10] Argoverse Motion Forecasting User Guide (未标注发布日期；访问2026-09-20). https://argoverse.github.io/user-guide/tasks/motion_forecasting.html
