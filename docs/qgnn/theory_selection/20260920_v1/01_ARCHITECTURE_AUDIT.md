# ICCT 全链路架构审计：任务瓶颈先于量子选型

研究版本：20260920_v1。事实快照：qgnn@094def5b38fa21ccc09cda2328fe78ee8cc694c1。
证据标记：CODE=直接读取实现；RESULT=已有实验文件；INFERENCE=本轮推论；HYPOTHESIS=待验证。
本报告不修改已有模型、数据、ISAC、训练配置，不运行新的训练或推理。

## 1. 当前信息流 [CODE]
SinD公开长春/西安两段连续轨迹 -> 完整40帧窗口资格 -> GT历史高交互筛选及focal八车选择 -> 三BS受控极坐标量测 -> 位置/速度融合 -> sensing history [B,20,8,4] -> 单车编码和交互核心 -> 每车64维g -> graph token/本车19个Motion Token -> GPT-2前4block+QKV LoRA -> 20个future query -> token概率和连续坐标修正 -> 全部有效目标的scene-macro ADE/FDE。
GPT-2按车辆独立执行；跨车信息必须通过交互核心。g也直接进入coordinate head，并非所有交互收益都必须经过GPT-2。

## 2. 数据与任务定义 [CODE/RESULT]
正式四轮车辆car/bus/truck；不是完整SinD 2.0，也不包含全部道路参与者。Train/val/test=15802/1880/2086。源dt=0.1001001001s，20点历史跨度约1.9019s，未来第20点约2.0020s。
高交互门：末历史帧至少一车有>=2 active neighbors；active=距离<=30m且(closing>0.5m/s 或 0<TCPA<=4s且DCPA<=10m)。
N>8每全局窗口只发一个图；先按active degree等选focal，再选与focal最相关七车；评价却覆盖所有选中车辆。90.73%窗口源于N>8候选集合。
重要限定：original_num_vehicles是完整窗口合格且未被guard排除的车辆数，不是现场全部车辆数。候选资格要求轨迹覆盖40帧；未来位置未参与邻车排序，但未来标签可用性参与离线样本资格。
选车和高交互门读的是GT历史。五SNR共用这个干净历史选定的集合；正式模型history仍来自sensing。这里是离线benchmark协议，不是已验证的完全观测驱动在线routing。

## 3. 感知定义 [CODE]
Route B: 检测与身份关联假设成功。按GT车逐BS生成range/bearing/radial-velocity高斯噪声，再按协方差融合位置、加弱先验恢复速度。没有真实detector、身份混淆、漏检、Kalman或时序tracker。
测量sigma_q(gamma)=sqrt(floor_q^2+a_q^2*10^(-gamma/10))。基础随机量按scene/frame/BS/vehicle/channel键独立生成；SNR不进入随机种子。各SNR复用标准噪声，再缩放。
已有test sensing RMSE，位置/速度依次：-10dB 1.174900m/1.175739mps；-5dB 0.659561/0.661126；0dB 0.370687/0.371775；5dB 0.208422/0.209075；10dB 0.117216/0.117595。
这些已有test感知汇总仅用于理解冻结上游，不用于架构调参；本轮不打开test轨迹或预测标签。
[INFERENCE] 最后速度误差经2秒CV外推可产生约2倍误差项；这不是最终FDE或下界，历史编码可修正它。仅凭位置RMSE不能宣称所有SNR都不受感知限制。
[INFERENCE] Delta p_hat_t=Delta p_t+epsilon_t-epsilon_(t-1)，噪声差分与token表示耦合。量子/经典必须共享同质量单车历史编码，避免把去噪差异算成量子交互收益。

## 4. 当前Token与LLM [CODE]
实际FinalMotionGPT2使用configs/qgnn_final_tokens.json的41x41=1681码本。每轴31个中心点、两侧各5个尾点；前向中心[-1.2,2.2]步长0.113333m；横向[-1.3,1.3]步长0.086667m。五SNR train历史用于尾点拟合。
历史位移转本车heading坐标，低速使用已观察位移方向；四角双线性embedding加[f,l,overflow_f,overflow_l]连续adapter，不是硬离散历史。
输入序列1 graph+19 history+20 queries=40；g由LN64->Linear768->GELU投影。19个历史token都加入graph token并过history adapter。
GPT-2使用本地预训练权重前4block、hidden768；冻结基础参数；每block fused c_attn注入rank8/alpha16 LoRA。共享LoRA参数98304；另有约4052375个非LoRA可训练下游参数。
输出不是自回归离散token积分。forward未调用soft_decode；trajectory=CV+cap*tanh(4*z/cap)，默认cap16，因此CV+16*tanh(z/4)。CV的dt由FinalModel用源timestamps校正。
future hidden、g和expected motion共同输入坐标头。损失=ADE+0.5FDE+0.035 TokenCE；checkpoint按validation J选择。
[INFERENCE] 码本步长不是当前最终坐标分辨率下界；只改细codebook不能保证ADE/FDE下降。
[HYPOTHESIS] 单个64维g可能丢失目标相关的邻车身份/时间结构，但未证实容量不足。投影到768维不能恢复已丢信息。

## 5. 已有交互证据 [RESULT]
SinD Q0: 0dB/seed2026/20epoch/validation/旧临时0dB tokenizer。
NoGraph ADE/FDE=0.524150116/1.063803667；pair=0.502264256/1.030863958；routed=0.501780581/1.020529961；pair+triplet=0.498217953/1.004632843。
Triplet相对pair整体收益ADE约0.806%、FDE约2.545%；动态closing>=15分层140窗口收益约4.33%/6.31%；仅按近距离密度的优势不稳定。
这些结果支持联合动态关系研究，不证明经典已经到上限，更不证明量子必要。triplet与pair存在结构/参数差异；pair最佳epoch20仍在训练边界。
按真实误差逐窗口选模型的oracle是错误互补性诊断，不是可部署收益承诺。窗口高度重叠，不能按1880独立样本做显著性解释。

## 6. 瓶颈分解
A. [HYPOTHESIS/CODE RISK] focal选车与全节点评价错位，非focal目标可能缺关键邻车。必须区分上下文覆盖与模型能力。
B. [RESULT-SUPPORTED RESEARCH TARGET] 对i同时考虑j,k以及j-k关系，理解动态多车接近下的联合影响；不是只给独立边评分。
C. [HYPOTHESIS] 当前物理图摘要以early5/recent5均值及差为主，可能压缩冲突形成先后和瞬态信息；GRU仍保留部分历史，不能声称时间信息全部丢失。
D. [HYPOTHESIS] 每车单向量g及g->head旁路可能决定交互信息的可用性/收益归因。
不是目前已证实瓶颈：长距离消息传播、1-WL不可分图、LoRA位置、码本精度。

## 7. 候选机制入口
将N_scene、K_context、Q_register分开。更大场景上下文不要求指数增加qubit；现有BoundedCore有大N的逐目标patch路径，但正式loader仍硬编码8槽。
所有候选必须把跨车联合计算放在量子核心，并明确进入/离开量子态的信息。允许共享逐车temporal encoder和物理特征，不允许强GNN先完成联合推理再挂VQC。
公平比较保留同一上下文、历史编码、物理权限、接口、GPT-2、token、LoRA和选择规则；还需强classical higher-order与功能匹配的substrate controls。

## 8. 可核验源位置
- docs/dataset_migration/SIND_MIGRATION_HANDOFF.md:49-208,285-405
- tools/data_preprocessing/build_sind_high_interaction.py:170-295
- frontend/sind_prediction_dataset.py:33-109
- frontend/controlled_isac/automatum_measurement.py:55-101
- frontend/controlled_isac/automatum_frontend.py:29-70
- frontend/controlled_isac/velocity_fusion.py:17-49
- reports/sind/sind_isac_validation.json:4071-4093
- prediction/qgnn_final/common.py:7-82
- prediction/qgnn_final/model.py:13-64,79-91
- prediction/q0/motion_token_llm.py:141-249
- prediction/q0/temporal.py:14-25
- prediction/q0/graph.py:55-128,277-416
- prediction/q0/features.py:9-55; prediction/q0/metrics.py:9-36
- scripts/train_qgnn_paper_native.py:22-31,100-104
- reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json

本报告定位研究问题，不预先选Raj、SQM或旧RC路线。
