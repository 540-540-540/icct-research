# A01 修订：确认集保留与三站几何覆盖

日期：2026-09-12。授权：用户明确要求“解决这两个问题”。
当前方案书及ROADMAP版本1.1。备份：reports/amendment_A01_before/。
所有实践在服务器 /home/dell/YrM/ICCT 完成。未运行Q/G训练或查看模型测试成绩；原始数据曝光状态仍为unknown_or_previously_exposed。

## 1. 确认集不足的根因与修订

旧规则把所有原始行跨时间段的312个源组全局排除，实际候选队列跨集合冲突只有28个。部分车辆虽在原始记录中跨界，但并未进入另一集合的实际实验队列；旧规则仍删除其场景。

A01保留原60/10/10/20时间划分、各边界5秒隔离、20秒片段、60秒统计块、45米邻域和最多8车。先生成完整候选队列，再按train→V_select→V_confirm→test处理；同集合按起点/锚点排序。任一源组已归属其他集合即拒绝整个episode；否则保留并归属。拒绝不占组，不删邻车，不根据模型分数或运动难度筛选。原始行跨界仅作诊断。最终实际输入来源互斥，未降低源身份隔离要求。

| 集合 | 旧场景 | A01场景 | A01不同20秒起点 | A01 60秒块 | 候选预测网格 |
|---|---:|---:|---:|---:|---:|
| train |168|179|60|21|5549|
| V_select |6|25|9|3|400|
| V_confirm |1|22|8|4|352|
| test |49|54|19|7|864|

新规则分别拒绝0/2/5/4条候选，拒绝原因和原归属完整保留。确认22条队列来自8个时间起点、4个块，仍是小样本先导，不能称22或352个独立交通样本。此处网格仍须F01-D用实际系统航迹筛选。

训练几何仅拟合实际保留训练队列中193898条唯一位置行，避免被排除/非训练身份通过尺度或站位间接参与。重复锚点不重复计入拟合。

## 2. 覆盖不足的根因与修订

旧外侧基站向中心斜照，使长道路两端位于全部扇区之外。A01采用横穿道路的固定0°/180°/0°朝向；只有中央BS2的横向道路边缘余量从50m增为100m。外侧站位公式、三站数量、±70°视场、10—300m斜距、6m/1m高度及波形不变。配置选择只依据训练几何；重建后验证/测试仅作固定规则覆盖审计，未反向选择配置。

新W=28.46745132m，L=427.62714576m：
- BS1=(-64.23372566,-106.90678644)m
- BS2=(114.23372566,0)m
- BS3=(-64.23372566,106.90678644)m

在同一批193894条去重活跃训练源位置行比较（旧位置按旧原点正确换算）：

| 几何指标 | 旧方案 | A01 |
|---|---:|---:|
| 至少一站 |77.9921%|100.0000%|
| 至少两站 |66.6880%|99.9288%|
| 三站同时 |50.3976%|19.4993%|

不能只报告提升：三站同时重叠明显减少，中央站距离变远可能降低部分回波SNR。当前解决的是道路几何盲区和至少两站可见性，不能保证检测、关联、速度或预测精度；F01-C仍需原工程验收，不将几何指标替代检测覆盖。

原始训练段全部owned行、各集合实际队列的覆盖另见geometry.json/scene_quality.json，分母不得混用。旧训练几何三方案选择证据保留于reports/geometry_revision/。

## 3. 验收与范围

- 独立重新构建每个保留队列，车辆及顺序与原始选择完全一致。
- 所有集合源组两两互斥；拒绝不抢占身份测试通过。
- 所有episode支撑仍在原时间集合、5秒隔离及60秒块内。
- 因果速度、新生车延迟激活、真实轨迹前缀不变检查通过。
- 训练几何193898行独立复核一致，非训练源组交叉为0。
- 共享几何函数自检通过，生成与绘图使用同一函数。
- 预处理与图片在服务器生成；抽样轨迹和覆盖图已查看。

跨记录局部坐标约2米差异仍作为源标注局限保留；不把跨段轨迹拼成连续同车，不声称真实地图坐标标定。源数据侧状态不能作为state_hat，F01-B/C/D尚未执行。

## 4. 当前入口

frontend/scene_manifest.py：构建源场景；
frontend/station_geometry.py：唯一共同几何规则；
frontend/review_source.py：数量/几何复核及训练图；
configs/f01a.json：A01阶段配置。
阶段结果见reports/f01a/F01A_REPORT.md。下一阶段F01-B先做少量叠加回波与时间一致性校核，不启动正式模型训练。


## F01-C执行记录（2026-09-12，保留A02模型协议）

初版检测追踪已完成12训练与25个V_select开发场景验收但覆盖失败。C01依据训练诊断修正角度CFAR保护区和NMS；C02试用既定保守R；C03启用方案预声明的局部JPDA。每版q_a仅取既定1/4/9，不改变波形、源队列、三站、确认/删除周期、5m绑定及全部确认池虚假口径。所有有限候选均未同时满足门槛，未封存任何通过配置。C03仅完成固定12训练场景，未追加开发集、确认集或测试集。详细失败矩阵、噪声/因果/协方差检查见reports/f01c/F01C_REPORT.md。F01-C为failed，F01-D为blocked；后续需针对重复确认航迹与协方差/关联模型形成明确修订后再验收，不能以选择八槽后的低虚假比例替代全部确认池口径。


## C06：共同前端通过验收（2026-09-12）

**C06执行修订（2026-09-12）：** 用户要求继续解决F01-C失败。本轮采用训练残差拟合的双高斯角度误差模型：窄分量保留常规观测精度，宽分量固定5°描述少量较大误差；按分量似然参与JPDA，每个Joseph更新分支及均值散布共同形成后验。R公开字段为两分量先验加权协方差，不能把它当作单高斯更新的实际全部模型。采用既定q_a=9，连续第7个漏检周期删除（前6个保留），2/3周期确认不变。保留期限在训练检出间断分位给出的5—8周期范围内比较，6/7在训练均通过后按覆盖选择7，再验证开发集。阈值、全部确认池口径、5m一对一绑定、8槽和源车辆未改变。V_select经过自适应开发，不能称盲测；V_confirm/test未使用。所有失败候选、选择顺序、因果与数值证据见reports/f01c/F01C_REPORT.md。协方差一致性仍有偏乐观迹象，工程通过不等于后验概率完全校准。

C06共同前端在固定12个训练及全部25个V_select场景通过标称20dB工程验收；开发覆盖91.43%、位置/速度RMSE 1.091m/1.257m/s、全确认池虚假比例4.04%。F01-D尚未执行，未训练预测模型。训练侧覆盖95.37%、全确认池虚假比例4.44%。源代码、配置与完整比较在reports/f01c/；A02模型设计未改。


# A03：已知分离/对应的逐车符号级感知（2026-09-12）

用户明确选择以Wei2024符号级多基站论文为前端主依据，逐车独立感知，假设目标已分离且对应已知，再组成最多8车的Q/G共同输入。根文档升为1.3；修改前四份根文档备份于`reports/amendment_A03_before/`。

当前规则完整替换旧盲CFAR、JPDA、5m绑定及虚假确认门槛；C06代码/数据/失败与通过记录保留为历史，不作为新前端成绩。A01源划分、队列与共同几何继续复用，开发曝光记录不撤销。A02 Q24/G128数学、LoRA、模型训练和比较矩阵不变。

A03采用二维三站、项目K256、B93.1MHz、fc24GHz、T12.375μs、N_s256；论文K128仅作小区域校核。原始符号SNR{−20,−15,−10,−5}dB、标称−5dB，与旧加窗输出SNR区分。位置搜索由训练min/max±5m固定、最大角点376.37m小于约412m距离范围，速度分量各±40m/s。逐车固定接收SNR和独立虚拟通道不主张多用户共享资源或固定发射功率预算。

ID/key仅作独立元数据路由，禁止作为模型数值；exists/detected重定义为因果估计历史/当帧更新可用，保留A02六维输入。未来标签不能改变mask，不以真值状态初始化或挑选估计结果。

37episode×3快照×4SNR真实验证已完成：每档训练233、V_select543目标快照全部输出，见`reports/symbol_level/real_scene_quality.json`。综合验收现已完成并通过：两配置各15项核心校核、共512次随机试验零异常；每SNR776个真实目标快照零搜索失败、零全局边界命中；两片段100帧498个目标状态删除5秒后源数据重放逐项一致。完整证据见`reports/symbol_level/A03_REPORT.md`及`provenance.json`。归一化为固定复噪声方差1、信号幅度10^(SNR_dB/20)，不是固定信号再调噪声。F01-D完整199帧历史缓存仍未执行，没有Q/G训练结果；小规模前端通过不等于完整缓存验收。


## F01-D启动与契约补齐（2026-09-12）
用户授权继续完整时序打包。A03核心/真实快照/因果验收已通过，F01-D现为执行中，尚未验收通过。根文档改前备份于`reports/f01d/before_docs/`。清除当前规则残留的FoV出生、旧CPI中心及D2称谓，统一采用A03 deadline逐车快照。产物为`data/f01d/inputs/`、`labels/`、`metadata/`、`sequences/`，统一loader为`frontend/symbol_dataset.py`；标准化按train四档SNR有效缓存状态项同权拟合，保留重叠窗口项并记数，验证/确认/测试不参与。测试缓存可用固定前端生成并封存，不评估模型预测或用于选择。A02模型数学与训练矩阵不变。


## A03 SNR修订：沿用师兄正档位（2026-09-12）

用户明确要求沿用师兄的5/10/15/20dB档位，标称20dB；当前修订A03-symbol-v2-positive-snr。保留A03原始接收符号SNR定义和逐车已知分离/对应模型。正档位配置已确定；全量生成已停止，待CPU/GPU加速讨论，新任务未启动，旧负档位完整生成已停止；F01-D尚未验收，未训练预测模型。

只修改SNR档位及相应生产版本，A03符号级观测、已知分离/对应假设、A01划分、A02数学与训练矩阵不变。固定复噪声方差1，信号幅度10^(SNR_dB/20)。可选压力仍指额外降低功率20dB，不是负SNR主档位。旧负档位报告数值及旧C06均为历史证据，不替换档位标签冒充新成绩。本文此前的负档位“通过”和F01-D“执行中”均为当时状态，由本条更新。修改前五份文本保存在`reports/snr_positive_before/docs/`；代码、配置、数据归档由主线程负责，本次仅更新文本。


## A03 GPU实施与过程文件清理（2026-09-12）

用户已授权GPU运行及清理旧过程文件。当前修订A03-symbol-v3-cuda-positive-snr，SNR为5/10/15/20dB、标称20dB，接收符号定义不变。GPU实现开发校验中，全量GPU生成尚未启动，F01-D未验收；未开始预测训练。

当前生产改用`frontend/generate_symbol_gpu.py`及`frontend/symbol_level_gpu.py`，双4090批量执行。`frontend/pack_symbol_dataset.py pack`负责打包，`frontend/symbol_dataset.py normalize`负责CUDA数值标准化及统一数据入口。旧CPU全量generate入口禁用，CPU仅源读取、路由和写盘。

主线程已清理未完成负档位缓存`data/f01d_negative_snr_archive`（874个文件，有效数据约6.4MB）及本地f01d-packing临时目录；原始数据、验证报告、文档备份保留。此处记录已完成清理，不代表本轮文档任务另行删除文件。历史验证数值不改写为GPU成绩，GPU尚无加速倍数和全量完成结论；A02数学与训练协议保持不变。


## GPU验收与用户手动启动边界（2026-09-12）

当前A03-symbol-v3-cuda-positive-snr，SNR为5/10/15/20dB、标称20dB。GPU独立验收已通过，episode0/95整合试跑已完成；正式全量未启动、F01-D未验收。用户将在服务器终端手动启动并监视，助手不得代为启动正式全量。未开始预测训练。

证据为`reports/symbol_gpu/validation.json`和`reports/f01d/gpu_smoke_generation.json`。正式入口`scripts/run_f01d_gpu.py`按生成→pack→normalize→validate顺序运行，进度前台显示，写run_status.json与F01D_REPORT.md。用户最新决定覆盖此前助手启动授权：正式全量只能由用户在服务器终端手动启动。当前停止助手侧启动及持续监视；历史验证数值与A02训练协议不变。


## F01-D v4重跑与F02新任务交接（2026-09-12）

当前为A03-symbol-v4-cuda-stable-solver，SNR5/10/15/20dB、标称20dB。用户已手动启动修复后的F01-D双GPU重建；交接核验时run_status.json为running/generate，不能提前判定全量通过。F02转交新任务，尚未实现或验收；未进行预测训练。正式实验仍由用户手动启动。

粗定位求解器改为稳定计算目标函数差值，未放宽阈值。183个同回波训练样本旧35失败降为0，GPU校核通过；旧缓存仍不能被视为v4结果。用户确认已重跑，不再改为增量修复。F02旧对话的委派已中断，未发现prediction实现产物；按用户要求另开GPT-5.6 Sol/High任务接续。详见docs/F02_新任务交接.md。

### 2026-09-12 F02执行完成（不更改A02结构）
F01D v4全量通过，F02由原任务接回并完成A–H共64项验收；量子float64/complex128，GPT-2 float32。交付模型、独立数学参考、真实缓存与标签侧车接口验收，未启动训练。详见reports/F02_REPORT.md和reports/theory_validation.json。跨车连接与24维充分性问题登记为后续独立研究问题，未借本次实现改动主模型。

### 2026-09-12 取消新增隔离实验，追加PennyLane校核
用户取消新增连接方式与读出宽度隔离实验，现有Q24实现保持。追加同电路PennyLane标准门/态矢量/读出/输入与参数梯度交叉验证，不启动训练。

PennyLane补充校核完成：23组404项通过，最大输出误差6.83897e-14、梯度误差2.54463e-13。默认8车3层全有效结构亦通过；现有模型未修改、无训练。结果见reports/f02/pennylane_crosscheck.json。


# A04 修订：正式后端与主要baseline（2026-09-12）

版本1.4，status=passed。用户要求PennyLane为正式实现，主要baseline使用师兄Plain的图核心，展示名仅QGNN/GNN；前端沿用A03v4，Q数学沿用A02。

此前将用户原意误解为整套无LLM的G预测器并改写共同时间规则，现已撤回该误解，直接更正权威正文。

正式quantum.py使用PennyLane与Snapshot单次演化浅/深态；quantum_torch_reference.py仅作校核。GNN保留师兄2层4头128维DenseEdgeGraphAttention、七维build_edge_features、45m+self、门控/归一化/FFN；当前逐帧4维标准化状态加detected/exists共6维经MLP6→128→128进入图，输出[B,T,8,128]接共同TrajectoryPredictor(128)。原GRU、132维节点projection、24维时间embedding及decoder不纳入主模型，故只称图核心适配，不继承原完整模型历史成绩。

两侧共同GPT-2、LoRA、预测头、CV、loss和3+2+15训练阶段；Q24/G128经各自投影形成同规格768维时间输入。H1比较共同时间条件下图分支的净收益，不能忽略图内经典特征和维数差异而单独归因量子门。F04用共同小头，Q原A/B/C、G固定两层两档lr；确认/灰区恢复对称共同时间规则。主张LLM作用时恢复Q/G各3seed时间模块对照。

当前图核心/共同时间接口汇总70项passed：量子48项、接口16项、源G图4组、主入口2项。GNNGraph源图层、输入及50参数梯度校核误差为0；两主入口均使用GPT-2 12层/12头/768、LoRA294912参数，实际train20帧3车完整loss反向通过。G图分支319756参数；G/Q完整总参数分别125365300/124965720，含冻结权重，不能当作可训练参数量。此前误解整套G版本的71项不作为当前结果。PennyLane后端及历史数学报告保留各自范围。主入口prediction/model.py build_model仅接受QGNN/GNN，主配置configs/prediction.json。F03尚未执行；正式训练仍用户终端启动，新增连接与表示宽度隔离实验持续取消。


## A05: equivalent execution acceleration and training readiness (2026-09-12)

User authorized progression through prerequisites until F05 can be launched, applying feasible speedups. Both GPUs are available continuously without a hard deadline; F05 itself remains a manual user launch. No model mathematics, data permissions, candidate family, seed count, or gate threshold changed.

- PennyLane history execution groups frames by active node count, broadcasts up to 20 frames, restores original frame/slot order, and uses checkpoint recomputation only when the call contains more than 20 frames. Production remains float64/complex128. Output/input/parameter-gradient checks pass against original serial PennyLane; maximum error 4.63e-14.
- Frozen evaluation deduplicates exact complete state+mask frame bytes within one evaluation call; no cross-weight cache, label or ID key. Real validation graph evaluation 23.90s to 11.33s; full prediction metric equivalence checked.
- Same effective batch16/micro1 full QGNN short-probe median 198.328s to 20.302s (9.77x), peak8.24GiB. Q21 comparator and both F04 small heads separately profiled. Source reports are in reports/f03/optimized_* and reports/quantum_speed.
- Training retains scene-point-normalized loss; evaluation uses per-target, per-scene, per-SNR macro means. Resumption retains optimizer/RNG and checks source provenance. F04 selects only on V_select; V_confirm is used for fixed-model confirmation. No locked test training/evaluation is performed.
- Budget estimate268.02 GPUh, reserved413 GPUh including compulsory old-Q and mechanism work; F04 reserved48 GPUh including9 GPUh gray-review reserve. Derived planning horizon13 days on two GPUs gives624 GPUh capacity, not a user deadline. See budget JSON for assumptions, conservative proxies and unmeasured quantities.
- F04 must still produce PASS before selected_protocol and F05 readiness. No hypothetical training outcome is recorded as passed.


# A06：旧路线用户中止与5-qubit QGAT候选试跑（2026-09-13）

**当前交接状态（2026-09-13，A06路线变更）：用户已取消旧16-qubit QGNN路线并要求停止旧训练。旧F04状态为用户中止（USER_ABORTED），不是算法STOP、数值失败或优势判定；旧路线F05禁止启动。旧F04协调器、两个worker及tracker已由主线程终止，并核验旧进程退出；自动化保持PAUSED。当前仅推进独立5-qubit PennyLane QGAT候选与同GNN的有界试跑，首轮有界试跑已完成，详见`reports/qgat_candidate/RESULTS.md`；该结果不是正式优势结论，也不是旧F04 PASS。**

用户最新授权覆盖A05的继续执行指令。旧F04协调器、两个worker及tracker已由主线程停止并确认退出；旧训练记录保留，旧F05不得启动，自动化保持PAUSED。新候选独立在`experiments/qgat_candidate/`实现，沿用共同前端与GPT-2/LoRA；128 train、64 V_select、20dB、3epochs，与同GNN试跑。架构参考Li等DOI 10.1088/2632-2153/ae32dd，采用5-qubit、6→3角度encoder、节点间共享且Q/K/V独立的U3，以及3+4+1共8维可测不变读出后接Linear(8,128)+LayerNorm；不称原论文flatten复刻。首轮有界试跑已完成，实测见`reports/qgat_candidate/RESULTS.md`，仍不是正式优势或旧F04 PASS。实际执行为PennyLane单比特Q/K/V子线路加精确张量收缩；与完整五比特线路最大输出误差6.66e−16、梯度误差1.42e−14，等价验收通过，不代表量子硬件优势。路线边界见`reports/qgat_candidate/ROUTE_CHANGE.md`。


# A07：新QGAT路线正式实验授权（2026-09-13）

**当前交接状态（2026-09-13，A07正式实验）：用户已授权启动新QGAT路线正式实验。最终主审与check-only已通过，当前为READY，等待用户亲自在服务器终端启动，助手不启动正式训练；执行依据为`reports/qgat_formal/PROTOCOL.md`。旧16-qubit路线永久停止，旧F04为USER_ABORTED而非算法STOP，旧F05禁止启动；自动化仍为PAUSED。A06首轮有界试跑已完成，其结果不代表正式优势或旧F04 PASS。**

沿用A06图结构，QGAT/GNN两侧同规格GPT-2/LoRA；seeds2023/2024/2025，同seed共享完整可变时间模块初始权重含图投影，全train四SNR平衡轮换，直接联合最多20epoch、耐心4，batch16/micro1。六任务双GPU动态独立进程调度；每epoch完整V_select四档J选best，六best全部锁定后统一V_confirm四档最终评价，不据其调本批配置。当前入口不读test，后续锁定best后另行最终评价。无需旧warmup/adapt/F04门禁，旧路线永久停，自动化PAUSED。具体优化器、恢复和证据边界以`reports/qgat_formal/PROTOCOL.md`为准；当前实查train5549起点（5439含可预测目标）、V_select400起点（400含可预测目标）、V_confirm352起点（346含可预测目标）；不剔除不可预测起点。用户最新要求停在亲手启动阶段，正式运行等待用户在服务器终端手动启动，助手不代启。

# 2026-09-14：师兄双层QGNN为主候选，QGAT暂停训练投入

用户明确决定：“把师兄的QGNN作为主候选，我们的QGAT暂时不打算投入训练资源了。” 本条覆盖历史QGAT主线及后续训练安排。

主候选为`experiments/qgnn_inherited/graph.py`的`InheritedQGNNGraph`，复用师兄两层量子关系/消息模块并接当前公共GPT-2/LoRA预测器。保留强GNN作主要对照。这不是恢复旧16比特A02路线，也不是完整复现师兄原GRU、未来词及信任融合架构。

已有QGNN100轮结果见`reports/qgnn_inherited100/comparison.json`：最佳J在epoch54，ADE/FDE=0.5475210414/1.0955790387；GNN最佳J在epoch97，ADE/FDE=0.5516128727/1.1216386450，分别改善0.742%/2.323%。共同epoch100时QGNN反而落后。结果限于seed2026、V_select；历史GNN续训和物理分批差异仍须披露，尚无该QGNN候选的多种子或独立确认结论。

2026-09-14 00:06北京时间只读核查服务器：最新`results/qgat_value/seed2026/{relation_D,relation_value}/summary.json`均为complete、34700次更新；完整运行各100轮。对应最佳ADE/FDE分别为0.5560028313/1.1104653807（epoch97）和0.5697787055/1.1465322471（epoch69）。进程表未见QGAT/QGNN训练进程，GPU计算进程为空，无需中断进程。训练完成不据此推断充分收敛。

QGAT代码、配置、结果、日志和检查点全部保留，暂停所有QGAT新增试跑、重训、加训及自动恢复；不因主候选变更恢复历史自动化。此次只更新候选和资源决策，不启动新的QGNN训练，不访问V_confirm/test。后续正式训练仍由用户手动启动，优先围绕QGNN优势稳定性及量子关系核心作用准备最小验证方案。

## 2026-09-14：授权 QGNN X 读出配对试跑

用户授权 Codex 直接准备并启动原 QGNN 与 X 读出增量分支的两组单种子试跑，接受最多150轮、patience=20；正式启动并确认实际更新后停止代理工作，由服务器后台独立完成。该次授权覆盖本轮此前用户手动启动安排。两组共享初始化、全量train/V_select及训练和选模规则；新增分支仅增加96个可训练参数。强GNN保留为历史参照；QGAT继续暂停，无V_confirm/test访问。实际状态与完成结果以服务器 `reports/qgnn_readout/` 为准，协议见 `reports/qgnn_readout/PROTOCOL.md`。

## 2026-09-14：方向编码 QGNN 与统一规则的强 GNN

X读出试跑于02:34完成。原版实际95轮、最佳75轮ADE/FDE=0.5427669769/1.0954215576；X版实际74轮、最佳54轮=0.5465124407/1.1039378626。X版未改善，后续保留原Q并暂停X扩展。

用户在诊断和下一步方案后要求“开始吧”。已完成全400个V_select起点的分组分析、六个不同片段案例及固定权重关系干预。保持注意力加权门控总强度后，取消邻居间门控差异仅使ADE/FDE退步0.53%/0.62%；不能把gate=1的大幅退步全归因于邻居选择。后方邻居也有明显有益案例，不采用简单前车/距离硬筛选。详见 reports/qgnn_cases/REPORT.md。

本轮仅改变量子关系分支前四个物理通道的坐标，以接收车行驶方向作为前向、其右侧作为横向，低于1 m/s回退到原坐标；新增参数0。其他图机制和时间模型保持，收益有待训练验证。原QGNN已完成的150/p20同协议结果经完整来源、资产、数据范围与训练规则校验后复用；方向编码QGNN和强GNN各在一张GPU从头训练，seed2026、全train5549/V_select400、batch/micro/valid16、四SNR、最多150轮、patience20、min_delta0。

完整模型/优化器CPU检查、17+17真实数据GPU检查、首次验证前恢复和完成后零额外更新均通过。2026-09-14 09:50发起服务器后台运行；确认两组实际更新后停止交互工作。运行状态以 reports/qgnn_motionframe/seed2026 为准，协议见 reports/qgnn_motionframe/PROTOCOL.md。QGAT继续暂停，不读取V_confirm/test，不恢复历史自动化。此轮仅提供单种子开发证据，强GNN作为性能对照不单独证明量子必要性。

## 2026-09-14：两轮改进完成，主方案保持师兄QGNN

用户确认当前主方案依然是师兄方案中的双层QGNN；已完成的两次改进效果不佳，不据此替换原QGNN，也不恢复QGAT训练投入。

第一轮X读出最佳ADE/FDE为0.546512/1.103938，较同轮原QGNN退步0.690%/0.777%。第二轮行驶方向关系编码最佳ADE/FDE为0.551578/1.114769，较复用的同协议原QGNN退步1.623%/1.766%；相对本轮从头训练强GNN则ADE退步0.246%、FDE改善0.314%，未形成ADE和FDE同时改善。两轮均限于seed2026、V_select开发证据，未读取V_confirm/test。当前路线因此保持师兄双层QGNN为主方案、强GNN为主要对照，两个改进分支只保留为负结果记录。

## 2026-09-14：可学习有向上下文残差候选完成

候选 A 在师兄双层 QGNN 的两层关系编码中分别加入零初始化接收者/发送者上下文矩阵，仅作用于非自环边，共新增2,560个经典参数；原六比特三层PQC、12维读出、消息路径、公共时间模型及数据规则保持。D0、CPU检查和双GPU smoke/恢复均通过。

正式配对开发实验由用户亲自启动。原QGNN运行95轮、最佳第75轮ADE/FDE/J为0.542767/1.095422/1.090478；候选运行74轮、最佳第54轮为0.542487/1.096786/1.090880。候选ADE改善0.052%，FDE退步0.125%，未通过两项同时改善门槛，不进入多种子确认。仅使用train与V_select，未读取V_confirm/test；下一步待讨论，未授权新实验。详见`reports/qgnn_directed_context/STATUS.md`。
