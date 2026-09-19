# Round 3 Scene-Adaptive RC-HQGNN

日期：2026-09-19。服务器 /home/dell/YrM/ICCT，qgnn。设计起点 dc5daca，原型 dbb27b8。
状态：一级计算结构冻结为 v2 phase_feedback。工程通过，三轮结构pilot的性能门禁均未通过；额外同结构初始化检验已完成，最终初始化冻结为neutral；它仍未通过full-train门禁。不得将冻结解读为量子优势成立。

## 1. 范围、证据与候选选择
保留 R2 连续 sensing → 逐车编码 → ZZ/ZZZ 联合量子核心 → relation-carrying readout → 连续 Token → GPT-2。SinD、3-BS controlled sensing、20→20、缓存8车、GPT-2前4block/LoRA8/41x41码本/cap16 全部不变。输入只来自 sensing history；scene/vehicle ID不作为特征；future仅监督；test关闭。所有有效场景均执行四条量子线路，不存在经典/量子开关。

依据新 history-only 诊断：K3与量子相对收益有关，但不是因果证明；时间块router AUC约0.5，不支持分类量子胜负。输出gate调节太晚是待检验解释，而不是已证实唯一原因。

候选A：历史场景/关系调制ZZ、ZZZ及通道轮次，零中间反馈，硬件简单，但不能直接利用量子内部状态。
候选B：A加前轮K2/K3驱动下一轮关系耦合，选为主原型；仍C4/D3/每寄存器最多8qubit。
候选C：仅scene-global阶数标量，不进入原型，过于粗糙。候选A未进入主训练。最终仅保留v3 basis_feedback作为唯一量子备用，用于独立复核非对易转换假设；不按场景、阈值或预测误差在线切换。

## 2. 与R2相同的连续接口
X[B,20,N,4]、M[B,N]、timestamps[B,20]。本车编码h[B,N,32]、物理e[B,N,N,16]、风险w[B,N,N]、三车描述f[B,H,32]及w3[B,H]严格复用R2定义。Q=min(N,8)，P(N)=1 (N<=8)，否则N个8车ego patch。N>8的选择、padding、自身旁路、64维输出、Token与LLM均继承R2冻结文件，不重做上游。

## 3. 历史场景摘要：只统计，不做经典图推理
无序有效pair集合E与triple集合T。stat(v)=concat(masked_mean(v),sqrt(masked_population_variance(v)+1e-6))，空集合均值/标准差为0。
S=concat(stat(e_ij),stat(w_ij),stat(w_ijk))∈R36；场景统计的范围是当前模型图/ego patch，不包含patch外车辆。s=SiLU(Ws*asinh(S)+bs)∈R16。
没有MPNN、attention、学习后的邻居聚合；S只供量子耦合编译，绝不作为额外context token或预测头输入。S包含场景几何与关系分布的非学习统计，承认这属于经典预处理，而非称为量子操作。

## 4. 关系、阶数、通道、轮次条件化
索引o∈{2,3}，c=1..4，l=1..3。
B_s=reshape(Whead*s+bhead,[2,C,D])。q_o(s)为Linear16→16的两个8维分块。
t2_ij=<tanh(Wkey2*e_ij),q_2(s)>/sqrt(8)。
t3_ijk=<tanh(Wkey3*f_ijk),q_3(s)>/sqrt(8)。
关系键分别Linear16→8与32→8，无bias。关系调节系数a[o,c,l]可训练；最终neutral初始化为0。初版specialized实验初始化0.1。

最终初始化neutral：scene_head的weight/bias、feedback_global、relation_gain、feedback_local全部置0；其余scene encoder、key、query及R2原有参数按独立种子初始化。初始lambda=1，因此整个模型精确等于同seed新建的R2模型，但没有加载任何已训练checkpoint。训练后这些参数允许自由更新。量子四通道本来就有不同node/pair/triple编译输出，不要求以人工阶数偏置制造通道差异。
历史specialized实验的pair初始目标为linspace(1.10,0.80,C)，triple为linspace(0.75,1.20,C)*linspace(0.9,1.1,D)，对应atanh((prior-1)/0.8)作为head bias。该方案已完成实验但不是最终初始化。

## 5. 前轮量子反馈的精确定义
第l轮后计算K2、K3（定义同R2）；z2=tanh(K2/0.05)，z3=tanh(K3/0.02)。使用有界响应限制反馈极端值，不把K3强度直接当作量子胜负标签。
对每个通道，计算w加权的 mean(z2)、mean(z2²)、mean(z3)、mean(z3²)，拼接为u∈R(4C)。F=reshape(Wfb*u,[2,C,D])；Wfb无bias、初始0。
局部反馈为b[o,c,l,0]*z_o + b[o,c,l,1]*z_o²，最终两个b初始均0；历史specialized实验为0.05。第一轮没有反馈，F和局部反馈均0。

L2[c,l,ij]=B_s[2,c,l]+a[2,c,l]*t2_ij+F[2,c,l]+b[2,c,l,0]*z2_prev+b[2,c,l,1]*z2_prev²。
L3同理。lambda_o=1+0.8*tanh(L_o)，严格在(0.2,1.8)。
主架构v2只调量子门内相位：beta_tilde=beta_R2+1.5*(lambda2−1)*w_ij/s2，gamma_tilde=gamma_R2+1.5*(lambda3−1)*w_ijk/s3。其中s2=sqrt(max(1,sum(w_ij²)/n_active))，s3同理。lambda在经典侧是正倍率，在量子主方案中只是有界相位调节的坐标，并不表示最后耦合必须为正。节点RY、RZ与RX采用R2定义。每轮 U_l=RX * exp(-i[sum alpha Z + sum beta_tilde ZZ + sum gamma_tilde ZZZ]/2) * RY。

S、局部关系、量子反馈均不直接进入readout。关闭全部ZZ/ZZZ时，跨车梯度降到浮点误差；这用于确认控制器没有绕过量子交互成为隐蔽的经典图网络。所有场景执行完整量子演化，不根据lambda跳过线路。学习的有效相位可以为零；不声称每个输入都必然产生非零纠缠。

### 5.1 最终neutral初始化覆盖规则

上文0.1/0.05与通道prior描述的是已存档specialized实验的初值；最终主入口的controller_init=neutral覆盖它们。精确规则是：将scene_head的weight和bias、feedback_global.weight、relation_gain与feedback_local全部置零；保留scene_encoder、query、pair_key、triple_key及所有原R2参数的同seed随机初始化。控制器各参数仍可训练。最终主方案没有basis_weights，备用v3保留其独立记录的初始化，不据场景切换。

由此初始L2=L3=0、lambda2=lambda3=1，最终有界相位增量恰为0，两侧分别恢复各自原R2核心；之后用同一训练损失学习调制。不是加载训练好的R2参数，不是量子线路关闭，也不是强制未来全部lambda恒为1。该规则适用于全部场景。

## 6. 连续输出与训练
各轮state保留R2的7维单/二/三体统计与32维relation-carrying消息；4通道3轮共468维，经同样readout/gate与本车旁路得到[B,N,64]。原输出gate保留但不作为本轮主要创新。
预测头、码本、历史19transition+1interaction+20query、GPT-2权重与LoRA、16*tanh(raw/4)校正不变。loss=scene-macro ADE+0.5FDE+0.035CE；AdamW非LoRA3e-4/LoRA7.5e-5、wd2e-4、cosine、clip3；按同一validation J选择checkpoint。

## 7. 功能匹配经典对照
保留R2三层、hidden64、四头pair和rooted-triplet attention、原有mix/update/readout。新增相同2112参数CouplingController、完全相同S/关系键/正倍率/通道初值。
每层attention logits加log(lambda)，并将每head聚合输出乘风险加权mean(lambda)，避免scene-global倍率被softmax抵消。lambda=1时恢复旧经典交互（数值误差除外）。
经典反馈不是量子K3：前一层pair score取两方向平均再tanh；triple score取三种root平均再tanh，形成相同形状的关系内部响应。以相同加权统计/反馈参数调节下一经典层。经典原本具有更大的消息网络，未减宽/减层。
两组controller初始state完全相同；逐车编码与下游初始化规则不变。额外经典控制容量不是量子独占。

## 8. 置换、mask与梯度
S对车辆置换不变，关系特征对无序端点/三元组对称；K2/K3随qubit置换等变。归纳每轮Hamiltonian和量子态保持重标号等变，最终读出亦然。所有集合统计分母按有效mask计数，不按padding宽度。N>8沿用同一history-only patch规则。
反馈是可微的跨轮计算图，训练没有detach K2/K3；这不是把oracle预测误差反传到controller。梯度经下一轮角度、前轮相关期望回到前轮门参数。
不声称旋转/平移等变，也不把K2/K3非零当纠缠见证。

## 9. 模拟与真实硬件的边界
模拟：每sample C*P(N)条statevector轨迹，每条3轮，额外2次用于反馈的中间K2/K3评估；最后一轮K仍为读出必需。不中断statevector，不增加qubit。反向使用Torch autograd。
真实硬件：不能免费读出期望后继续同一条未坍缩态。必须先在独立制备的前缀副本上估计Z/ZZ/ZZZ，再计算下一轮lambda；为下一轮和读出重新制备并重放前缀。这是ensemble expectation feedback，不是coherent control，也不是单shot conditional gate。
R2本就要读三轮X/Y/Z，因此最多同样的3*C*D*P=36P种前缀/基设置，但它们现在存在串行依赖，必须计入shots、重放和经典延迟。单次各设置1shot的门轮数合计3*C*sum(l)=72P；不能以4P个模拟调用代替。有限shots累积量误差会影响后续所有轮，0.02/0.05反馈刻度和0.005读出刻度需额外评测；本轮不承诺硬件优势。

## 10. 参数、复杂度、成本
新增controller=2112参数，双方相同。Quantum core69469，classical325250；总可训练4220148与4475929，冻结GPT-2参数67736832。
Q=8时状态维256。模拟轨迹C*P(N)，各轮pair/triple枚举分别28/56，不是每edge调用PQC，不乘20history。计算含O(B*C*P*D*2^Q*(Q+Q²+Q³))及经典描述/读出；全局选邻居还含O(N²logN)，不宣称全程序O(N)。
主架构工程v2 RTX4090：B32 Q约0.11153秒/步，allocated3.641GiB；C约0.08447秒/步，3.666GiB。实际精确数值以engineering_v2 JSON为准。20epoch每epoch494步；由step推算全训练计算约18.4分钟，不含验证/保存/数据加载，本轮未据此启动全量训练。小pilot实际wall-clock另报。qubit/量子门数与R2相同：D3共324个抽象旋转/通道，基本分解约840 CNOT/通道（不含拓扑路由）。
N8/12/16/20的模拟轨迹数为4/48/64/80。对应v2 B16 core-only实测见下面资源表；这是随机输入工程测试，不是大N预测性能。

## 11. 研究依据与迁移边界
[1] Skolik etal, Equivariant quantum circuits for learning on weighted graphs, npj Quantum Information9,47(2023), doi10.1038/s41534-023-00710-y。只援引图条件量子动力学与置换思想，不援引SinD精度保证。
[2] PennyLane0.45.1 qp.measure: https://docs.pennylane.ai/en/stable/code/api/pennylane.measure.html 。中间测量返回测量值/改变后测量状态，不能等价为免费精确期望读取。
[3] Koh etal, Readout Error Mitigation for Mid-Circuit Measurements and Feedforward, arXiv:2406.07611(2024)。其硬件feedforward误差问题支持成本谨慎解释，不证明本方案有效。
[4] Mhiri etal, Constrained and Vanishing Expressivity of Quantum Fourier Models, Quantum9,1847(2025), doi10.22331/q-2025-09-03-1847。用于约束“加频率/深度必胜”的说法。
本轮2112参数控制器、局部累积量反馈与公平经典对应均为项目设计假设，非上述论文已有结论。

## 12. 全部配对实验与准入结论

所有下表均为同一4096train/1880val/12epoch/seed2026/0dB/B32/cap16。训练15,802窗口的20epoch第二轮结果不是本表的直接可比对象。各自checkpoint按同一validation J最小选取。

| 实验 | Quantum ADE/FDE | Classical ADE/FDE | Q的J | 相对匹配经典 ADE/FDE改善 |
|---|---:|---:|---:|---:|
| 冻结R2，同协议新训参考 | 0.602135 / 1.246336 | 0.585355 / 1.208482 | 1.225303 | -2.87% / -3.13% |
| v1 正倍率反馈 | 0.609190 / 1.256607 | 0.588472 / 1.214436 | 1.237493 | -3.52% / -3.47% |
| v2 有界相位增量 | 0.604980 / 1.261906 | 0.587954 / 1.216045 | 1.235933 | -2.90% / -3.77% |
| v3 非对易旋转反馈 | 0.609224 / 1.264438 | 0.589355 / 1.217505 | 1.241443 | -3.37% / -3.85% |
| v2 同结构neutral初始化（最终） | 0.601022 / 1.241815 | 0.587082 / 1.212963 | 1.221929 | -2.37% / -2.38% |

正值表示量子误差较低。本轮没有全局量子胜利。最终neutral方案相对同协议冻结Q仅改善0.185% ADE、0.363% FDE；相对最强保留的冻结经典仍差2.676%/2.758%，不能通过只看较弱新经典掩盖这一点。初始化和结构都由validation选择，不是独立确认。v2与v1很小的J差异也不代表稳定排序。

先验full门禁：Q overall J相对冻结Q至少改善1%；Q/C相对J差距至少缩小30%或翻正；closing<10的QJ改善；高140窗口QJ退化不超过1%；高140的Q/C差距不恶化。最终neutral只通过后三项：overall J改善0.275%，差距缩小约20.8%。故未启动15,802训练集的full pilot，未启动五SNR或多seed。所有旧结果/中间结构保留，一级结构修正次数为2，额外初始化校准为1，之后停止搜索。

### 12.1 最终方案的完整预定义分层

下列closing前缀final指最后历史帧，recent5指最后5帧平均状态；140与115不是同一子集。K3分位边界来自原冻结模型历史响应，不来自本轮未来误差。

| 子集 | 窗口 | Q对匹配经典ADE改善 | FDE改善 | J改善 |
|---|---:|---:|---:|---:|
| all | 1880 | -2.37% | -2.38% | -2.38% |
| final_closing_lt10 | 1054 | -2.74% | -3.14% | -2.94% |
| final_closing_10_14 | 686 | -2.88% | -2.76% | -2.82% |
| final_closing_ge15 | 140 | +2.59% | +4.78% | +3.72% |
| recent5_closing_lt10 | 1130 | -2.71% | -3.07% | -2.89% |
| recent5_closing_10_14 | 635 | -2.87% | -2.55% | -2.71% |
| recent5_closing_ge15 | 115 | +3.15% | +4.69% | +3.95% |
| recent5_wedges_ge35 | 416 | +0.52% | +1.58% | +1.06% |
| recent5_triangles_ge6 | 391 | -0.25% | +0.30% | +0.03% |
| frozen_K3_quintile_1 | 376 | -4.23% | -5.20% | -4.72% |
| frozen_K3_quintile_2 | 376 | -2.95% | -3.12% | -3.04% |
| frozen_K3_quintile_3 | 376 | -1.92% | -1.68% | -1.79% |
| frozen_K3_quintile_4 | 376 | -2.07% | -2.01% | -2.04% |
| frozen_K3_quintile_5 | 376 | -0.14% | +1.14% | +0.50% |

窗口高度重叠且只使用seed2026；所有分层是描述性诊断，不报告伪独立样本显著性。最高K3组FDE略优但ADE未翻正，不能合称“两项均胜”。v3高140子集3.52%/5.68%的结果在其独立报告保留，不替换最终neutral的2.59%/4.78%。

## 13. 结构修正、失败解释与唯一备用

v1正倍率→v2相位增量是第一次修正：当原base角接近0，乘法调制的梯度按base角缩小，无法有效激活或反转耦合；改用有界相位增量解决这一表示限制，但实验未因此胜出。
v2→v3是第二次、最后一次修正：Z型相位与Z测量可交换，必须借助非对易旋转把相位关系转化为可观测相关。v3回到正倍率耦合，增加上一轮节点关联反馈4向量 [weighted_mean(z2),weighted_mean(z2²),weighted_mean(z3),weighted_mean(z3²)]，令下一轮 rho[c,l,i]=rho_R2[c,l]+0.6*tanh(A[c,l]·node_feedback_i)。A形状[C,D−1,4]，共32权重，初始化[0.1,0.25,0.1,0.25]。经典侧增加同样32权重，用1+0.6*tanh(...)缩放下一层各head节点特征，不减弱原网络。该方案仅作为全量子备用，不在线按场景选择。

最终主结构选择v2后，只做了一次双方对等neutral初始化校准，没有增加参数/门数/数据/损失或加载训练好的R2权重。初始整模型prediction/token_logits/graph_features与同seed新建R2逐数一致，证明它是优化起点检验，不是借用额外训练。

结论：本轮证据未支持“只要把自适应提前进入量子计算就能解决整体劣势”的强假设。K3和相对收益的关联不等于操控K3会因果改善误差。三个specialized原型删除反馈的J变化只有约0.061%/0.142%/0.362%；这些分布外干预只说明训练后依赖程度，不是重训练因果消融。neutral改善了整体优化结果，但未弥补经典差距。
同时，独立参数包含性检查证明新Q/C结构能恢复各自R2函数；因此不能断言函数类“天生必然更弱”。当前失败可来自优化、控制信号选择、训练预算下的归纳偏置或读出限制；本轮不能唯一归因，也不能据单seed否定所有adaptive quantum。未运行full就不推测其full结果，保留原科学不确定性。

## 14. 完整资源记录与硬件限制

| 实测项目 | 主量子v2 | 对应经典 |
|---|---:|---:|
| B32 step秒 | 0.111532 | 0.084472 |
| allocated GiB | 3.640848 | 3.666331 |
| reserved GiB | 3.972656 | 4.039062 |
| neutral 12epoch墙钟秒 | 217.792 | 168.303 |

上述为GPU精确量子线路模拟，不构成量子硬件加速。15,802样本20epoch的训练计算估计约18.4分钟，不含完整验证/IO，不是本轮full-run实测。正式预算7200秒到时保存并暂停；性能门禁失败，不因成本便宜而绕过。

| N | 每scene轨迹 | B16 core前向反向秒 | allocated GiB |
|---|---:|---:|---:|
| 8 | 4 | 0.047320 | 0.035780 |
| 12 | 48 | 0.049985 | 0.247336 |
| 16 | 64 | 0.052362 | 0.322718 |
| 20 | 80 | 0.051742 | 0.399831 |

B32、N8逻辑轨迹为128；N12/16/20为1536/2048/2560，均批处理模拟。大N测试为随机输入工程检查，不能推出真实轨迹精度。

有限shots反馈检查：16个训练窗口，256/1024/8192/65536 shots时，输出相对精确反馈的向量RMS偏差约0.0100/0.0050/0.0019/0.00075m。该检查使用specialized v2，最终readout精确且无真实门噪声，不是neutral模型的硬件结果。详见round3_finite_shot_feedback.json。

## 15. 验收、复现与交付边界

机器汇总：reports/qgnn/ROUND3_FINAL_RESULTS_20260919.json，生成器scripts/collect_qgnn_round3_results.py。四份配对summary、每个run的config/training/best_validation_rows、源代码哈希均保存。原始R2核心、Tokenizer/GPT-2配置与上游未改动。

round3_release_acceptance.json从原1880逐窗口结果重新计算指标与分层，核验训练曝光、哈希及门禁，并重载最终neutral两侧checkpoint验证输出；结果PASS，性能状态仍FAILED_FULL_TRAIN_GATE。round3_release_resume_regression.json实际比较连续4steps与暂停1step再恢复，不是仅检查resume参数。工程测试覆盖N1/2/4/8/12/16/20、mask、padding、置换、前后向、量子/控制器梯度、PennyLane前缀重放的独立状态/梯度核验。

所有长作业独立启动，job.json记录PID/GPU/命令/日志，25steps heartbeat、100steps与每epoch checkpoint，summary原子保存，预算到时暂停。参数、优化器、scheduler、随机状态、epoch/batch位置均可恢复。源码或码本哈希改变时拒绝混用断点；复现实验必须使用对应commit，不能拿旧checkpoint冒充新训练。大checkpoint只留服务器，不提交Git。

以下命令仅供复现已经结束的小pilot，不是继续大规模实验的建议。使用新run目录；经典把kind与CUDA设备对应替换，其余参数完全相同：

```bash
cd /home/dell/YrM/ICCT
CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_adaptive.py --kind quantum --adaptive-mode phase_feedback --controller-init neutral --snr 0 --seed 2026 --epochs 12 --train-limit 4096 --batch-size 32 --correction-cap 16 --run-dir reports/qgnn/round3_reproduce_quantum_2026 --max-seconds 1200
```

冻结的是完整可复现的Scene-Adaptive RC-HQGNN，不是性能晋级。minimum验收（结构、代码、真实paired pilot）完成；strong验收（全训练后overall量子两指标翻正）未完成，而且本轮门禁不允许启动full。后续不应把0.185%/0.363%的小改善写成达到5%目标，或把高动态子集当作整体结论。

## 16. 数学接口补充与中间张量

R2基础连续定义以FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md第3–5节为完整引用，以下给出本轮反馈所需精确接口。
psi初态为每通道|0>^Q；node Linear32→24经1.2*tanh，分为每通道每轮RY/RZ角，第一轮RY加pi/2；共享RX角rho[C,D]初始化0.4。base beta=1.5*tanh(MLP16→16→12(e)+0.5)*w2/s2，base gamma=1.5*tanh(MLP32→16→12(f)+0.5)*w3/s3，MLP中间SiLU。每层state形状[B*P*C,2^Q]，complex64训练、complex128核验。

```text
m_i = <Z_i>
K2_ij = <Z_i Z_j> - m_i*m_j
K3_ijk = <Z_i Z_j Z_k> - m_i*<Z_j Z_k>
         - m_j*<Z_i Z_k> - m_k*<Z_i Z_j> + 2*m_i*m_j*m_k
```

控制器K2输入[B*P,C,Q(Q−1)/2]，K3输入[B*P,C,Q(Q−1)(Q−2)/6]；局部反馈与边/三元组身份严格对齐，不按车辆ID编译参数。空关系集合的均值/反馈为0。K3不等同纠缠见证。

连续读出保留每车每通道每轮X/Y/Z及加权K2/K3均值、RMS；二体asinh刻度0.02，三体0.005。邻车value为MLP48→32→16([h_j,e_ij])，三车value为MLP112→32→16([h_j+h_k,|h_j−h_k|,e_ij+e_ik,|e_ij−e_ik|,e_jk])。分别乘w*asinh(K/scale)再按总w归一化；不经量子相关不允许value绕行预测。最终468维经MLP468→64→64，sigmoid门MLP500→32→1与本车local64相加得到G[B,N,64]。

Interaction Token为G逐车LayerNorm64→Linear768→GELU软Token，不是让量子输入离散ID。其后与本车19运动Token和20未来query组成长度40的GPT-2序列；原输出模块从future hidden、G与期望运动Token生成连续残差。最终仅输出[B,20,N,2]位置；future_state中的GT仅监督，不作为控制输入。

## 17. 最终中性初始化反馈依赖补查

补充报告ROUND3_SELECTED_FEEDBACK_AUDIT_20260919.md重新核验最终Q/C各5种设置的1880窗口。最终Q去全部内部反馈的整体J下降约0.057%，但高140窗口J增加0.164%；不能称为反馈的稳定独立收益，也不据此切换模型。该补充的训练端梯度探针因辅助脚本GRU模式错误未完成，已如实保留，不能与既有工程PASS混写。本轮主架构、初始化、原始pilot指标与FAILED_FULL_TRAIN_GATE结论均不改变。

## 18. 续跑完成：辅助修复及反馈重训练消融

第17节的辅助梯度失败已在后续51b832a修复并验证；旧失败记录保留。最新两侧均完成512训练场景统计、32场景有限梯度及全部1880验证重放，原模型与数据未改。完整过程见ROUND3_SELECTED_FEEDBACK_AUDIT_20260919.md第6节。

另完成预注册量子/经典×反馈on/off四组从头训练，协议4096train/1880val/12epoch/0dB/seed2026，与此前small pilot一致。量子on为0.601021648/1.241815257，off为0.603416829/1.252787244；反馈带来0.3969%ADE、0.8758%FDE、0.6408%J改善，但on仍输本次匹配Con2.5752%/2.4874%。本次Con为0.585932840/1.211676313，较原neutral经典run不同，故高140增益为1.9002%/4.0820%，不是原表2.5923%/4.7764%；两套配对结果分别保留，不能混搭。全部14分层、重复训练差异、checkpoint逐窗口重放与资源开销见ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md及round3_feedback_retraining_results_20260919.json。

这项固定消融表明联合跨轮反馈在一次训练协议下有小幅正贡献，不等于K3单独因果作用，也没有解决整体经典差距。主结构、neutral初始化、唯一备用、两次结构修正上限及FAILED_FULL_TRAIN_GATE均不改变。不把off消融选为新部署模型，不启动full训练、五SNR、多seed或test。
