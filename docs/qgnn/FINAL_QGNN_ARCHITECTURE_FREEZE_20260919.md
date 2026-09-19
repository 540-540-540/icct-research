# FINAL QGNN ARCHITECTURE FREEZE — RC-HQGNN

日期：2026-09-19。分支：qgnn。服务器：/home/dell/YrM/ICCT。

## 0. 冻结决定

主架构：**RC-HQGNN，Relation-Carrying Physics-Conditioned Joint Hypergraph Quantum Graph Neural Network（保留关系身份的物理条件联合超图量子网络）**。

连续感知历史 → 逐车轻量时间编码 → 四通道联合ZZ/ZZZ量子图演化 → 单体/两体/三体观测与关系携带读出 → 每车64维连续交互表示 → 交互软Token与运动Token → 同一GPT-2 → 20步未来位置。

一级结构固定：C=4，D=3，register_cap=8，message_width=16，output_dim=64。本轮v1→v2→v3只有两次量子结构修正，v3为最后一次；后续进入训练与消融，不再继续列一级候选。唯一备用为同一架构C=2成本版，仅在更大N资源门禁失败时启用，不按validation结果暗中切换。

**架构冻结不等于量子优势成立。** v1、v2均未超过匹配经典，v3和最终共同接口结果见本文件实验节及reports/qgnn。不得将参数较少、纠缠存在、经典三车收益、oracle空间包装成量子优势。当前在RTX4090上精确模拟量子线路，不声称量子硬件加速。

事实来源：主任务书、SinD迁移、SinD经典诊断、实际代码和运行结果。旧Automatum的低交互结论不再约束SinD。上游继续使用3-BS controlled sensing；当前没有真实检测、关联失败或跟踪器，论文不得扩大其实现边界。

## 1. 任务证据与待检验瓶颈

来源：docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md及对应JSON。SinD/0dB/seed2026/full-train/20epoch：经典三车联合相对普通两车整体ADE/FDE改善0.81%/2.55%，高动态接近的140个validation场景改善4.33%/6.31%。这证明三车动态有任务价值，**不证明强经典不会建模高阶关系**。

不同场景最优经典机制不同。事后按未来误差挑选的oracle不是可部署路由，也不是量子模型的保底指标。本文假设是：共享少量参数的联合量子演化可学习由整个邻域状态决定的两体/三体响应，并结合关系身份保留连续运动语义；其统计归纳偏置可能改善泛化，但必须公平实测。

v1的量子相关偏弱。v2增强相位与通道后，去除全部纠缠门使ADE从0.608360恶化到0.658697；去除ZZZ恶化到0.640063，但完整v2仍输给经典。这是依赖性检查，不是重训练消融。v3不盲目继续加qubit，而是解决相关性只做均值/RMS池化时丢失关系身份的问题。该瓶颈是结构诊断与实验假设，不宣称已证实是唯一失败原因。

## 2. 输入与信息权限

输入X[B,20,N,4]为ISAC估计x/y/vx/vy，M[B,N]为bool mask，timestamps[B,20]取真实源时间戳。输出未来位置[B,20,N,2]；future GT[B,20,N,4]只进入监督/评价。scene_id、vehicle_ids、focal_vehicle_id、selection_mode不作为模型特征。padding先清零，无效输出为零。

正式缓存仍为N=8；train15802、val1880。本轮test保持关闭。动态N测试仅证明模型工程接口，不代表已重建或评测N12/16/20的SinD预测数据。90.73%原始全局N>8窗口被预选成局部8车，不能当作大N性能证据。

定义P(N)=1（N≤8），否则P(N)=N。N≤8整图进入一个寄存器；N>8每辆目标建立一个8车ego patch：自身和风险最高的7个有效邻居，输出只取根节点。风险并列时按完整感知历史字典序排序，不按ID或输入槽号；完全相同历史的节点可交换。经典主对照使用完全相同patch。数据原有8车预选择不在本轮改动。

## 3. 连续编码与物理关系（精确实现定义）

每车每帧6维输入为[(p_t-p_last)/10, v_t/10, p_last/50]。逐车Linear6→32、SiLU、GRU32、LN32得到h_i；没有跨车操作。自身旁路为Linear32→64、SiLU、LN64。

对前5帧与后5帧各取平均状态。r_ij=p_j-p_i，u_ij=v_j-v_i；d=sqrt(max(r·r,1e-6))，c=−r·u/d，tau=clip(−r·u/max(u·u,1e-4),0,4)，dCPA=sqrt(max(||r+tau*u||²,1e-6))，speed=sqrt(max(u·u,1e-6))。每快照e8=[d/30,c/10,tau/4,dCPA/10,speed/10,v_i·v_j/100,abs(r_x)/30,abs(r_y)/30]；实际e16=[e_recent,e_recent−e_early]。

权重w_ij=M_i*M_j*1(i≠j)*exp(−d/30)*(0.25+sigmoid(c/2))*exp(−dCPA/20)，使用recent快照。该TCPA已截断，不要与审计脚本的未截断事件TCPA混淆。

三体特征f_ijk=[mean(e_ij,e_ik,e_jk),sqrt(var(e_ij,e_ik,e_jk)+1e-6)]为32维，var使用总体方差。三体w_ijk=(w_ij*w_ik+w_ij*w_jk+w_ik*w_jk)/3再乘有效mask。两条强边即可形成强三体，不要求三车两两都近。Q=8时枚举28个无序pair、56个无序triple。

## 4. 联合量子演化

C=4个独立通道，每通道Q=min(N,8)个qubit，每个qubit对应一辆车。四条8-qubit线路不是32-qubit纠缠寄存器。初态为|0>的Q重张量积，D=3轮重上传。

节点编译器Linear32→24，经1.2*tanh，重排成[C,D,2]分别为RY角eta与RZ角alpha；第一轮RY加pi/2。padding角为0。pair编译器F2:16→16 SiLU→12；triple编译器F3:32→16 SiLU→12。设n_a=max(有效车数,1)：

```text
s2 = sqrt(max(1, sum_{i<j} w_ij² / n_a))
s3 = sqrt(max(1, sum_{i<j<k} w_ijk² / n_a))
beta[c,l,i,j] = 1.5*tanh(F2(e_ij)[c,l]+0.5)*w_ij/s2
gamma[c,l,i,j,k] = 1.5*tanh(F3(f_ijk)[c,l]+0.5)*w_ijk/s3
H[c,l] = sum_i alpha_i Z_i + sum_{i<j} beta_ij Z_i Z_j
         + sum_{i<j<k} gamma_ijk Z_i Z_j Z_k
psi[c,l] = (tensor_i RX(rho[c,l]*M_i))
           exp(−i H[c,l]/2)
           (tensor_i RY(eta[c,l,i])) psi[c,l−1]
```

rho[C,D]是12个直接训练角，初始化0.4、跨车辆共享。Z型图相位与每轮RY/RX不对易；同一轮的所有Z/ZZ/ZZZ项可交换，因此模拟时可精确合成一次对角相位，不需要Trotter。每轮保存快照。

所有节点和图描述都先进入线路控制，**不存在经典MPNN/attention先完成主要图推理再接VQC**。当前不是声称全部聚合均在量子硬件内完成的纯量子message passing：量子负责联合图动力学与关系响应，经典部分负责逐车编码和测量后的连续读出。

实现：prediction/qgnn_final/quantum.py中的tables、rotate、evolve和circuit_inputs。Torch complex64状态向量用于训练，complex128用于独立PennyLane核验；autograd反传不是parameter-shift。

## 5. 测量、关系携带读出与输出形状

对每个通道、每轮快照计算m_i=<Z_i>、x_i=<X_i>、y_i=<Y_i>。Z测量的概率为|psi|²；X/Y由对应qubit bit-flip的相干项精确计算。

```text
K2_ij = <Z_i Z_j> − m_i*m_j
K3_ijk = <Z_i Z_j Z_k> − m_i*<Z_j Z_k>
         − m_j*<Z_i Z_k> − m_k*<Z_i Z_j> + 2*m_i*m_j*m_k
```

K2/K3是所选Z测量基下的连通相关/累积量，**不等于纠缠度量或纠缠见证**。乘积纯态的K2/K3为零，但经典相关混态也可能有非零K2/K3。本轮单qubit purity只用于已知纯态模拟的辅助诊断。

基础读出每车每轮7维：[X,Y,Z, weighted_mean(K2), weighted_RMS(K2), weighted_mean(K3), weighted_RMS(K3)]。pair与triple分别用w_ij和w_ijk加权，分母均为对应权重之和并设1e-6下界；没有关系时输出0。RMS实现为sqrt(weighted_mean(K²)+1e-12)，减去数值底噪1e-6后截非负。均值/RMS经asinh(x/0.02)处理两体、asinh(x/0.005)处理三体。跨4通道3轮得到84维。

最后一次结构修正保留关系身份：

```text
V2_ij = MLP48→32→16([h_j, e_ij])
V3_i;jk = MLP112→32→16([h_j+h_k, abs(h_j−h_k),
                       e_ij+e_ik, abs(e_ij−e_ik), e_jk])
M2_i[c,l] = sum_j w_ij*asinh(K2_ij[c,l]/0.02)*V2_ij / max(sum_j w_ij,1e-6)
M3_i[c,l] = sum_{j<k;j,k≠i} w_ijk*asinh(K3_ijk[c,l]/0.005)*V3_i;jk
            / max(sum_{j<k;j,k≠i} w_ijk,1e-6)
```

value网络中间激活均为SiLU。value不执行邻域聚合、不生成attention，也不直接绕过量子进入下游；**全部跨车value只有乘上量子连通相关后才能形成消息**。乘积态时该路径为零（浮点残差除外）。这区别于先做经典GNN再用VQC装饰。每轮每通道32维关系消息，跨轮/通道共384维；与基础84维拼接为q_i∈R^468。

```text
r_i = Linear64(SiLU(Linear468→64(q_i)))
a_i = sigmoid(Linear32→1(SiLU(Linear500→32([h_i,q_i]))))
g_i = M_i * (local64(h_i) + a_i*r_i)
```

gate最后bias初始化−1。QGNN最终输出G[B,N,64]，不输出离散词ID，也不直接输出未来轨迹。给旧GPT2接口扩展为[B,20,N,64]只为兼容，**不重复运行20次量子线路**。实现prediction/qgnn_final/relational.py。

## 6. 理论论证及其不能证明的事

### 6.1 联合机制而非独立边打分

固定一轮，对目标i将H写成Z_i*A_i + H_not_i，其中A_i=alpha_i+Σ_j beta_ij*Z_j+Σ_jk gamma_ijk*Z_jZ_k。令sigma_i_plus=(X_i+iY_i)/2，D=exp(−iH/2)，因为所有Z项可交换且[Z_i,sigma_i_plus]=2*sigma_i_plus，有：

```text
D† sigma_i_plus D = sigma_i_plus * exp(i A_i)
```

若入态在目标i与其余寄存器间可分解，上式的期望可分解出邻域联合相位的特征函数；一般纠缠入态不允许这一因式分解，但仍是包含目标相干与邻域联合算子的期望，而不是简单逐边独立MLP求和。后续非对易旋转及重复编码让这种联合响应继续耦合；ZZZ显式引入两名邻居共同作用于目标的相位。该等式是本线路的代数事实，**不蕴含相对任意经典神经网络的表达分离或ADE优势**。

### 6.2 置换与padding

共享节点/边/三体编译器、对称物理特征以及对所有无序集合求和，使H(PX)=U_P H(X) U_P†；局部门在不同qubit上可交换且共享参数，初态置换不变。故量子态、局部观测随车辆重标号同步变换。根化value对j/k对称，求和保持等变；同一history排序patch规则也保持目标置换等变。于是F(PX,PM)=P F(X,M)。

padding的本地门和所有关联边/三体权重为零，有效车辆数参与归一化，不受额外空槽影响。动态N多patch是局部模型，不保证patch间完整长程推理。模型**不宣称旋转/平移等变**：本车编码含世界坐标，边中也有轴向绝对值。

### 6.3 最强反驳与检验

8-qubit、浅层、精确读出的线路可以在经典GPU上便宜模拟；连续输入下WL的离散同构瓶颈不能直接认定为本任务瓶颈；经典高阶attention、张量/傅里叶交互模型可能学习同样有用的响应。局部Z累积量可能漏掉其他基下的关联，asinh还会放大有限shots噪声。不存在“有纠缠即泛化更好”的定理。量子优势只能由同数据/下游/合理预算下的多seed指标支持，并通过去除ZZZ、乘积态、读出与经典相关路由对照检验归因。

## 7. Token与GPT-2共同接口（冻结）

QGNN始终吃连续物理状态，不吃离散Token。其输出G[B,N,64]经LN64、Linear64→768、GELU形成每车一个连续soft graph token；本车历史另形成19个motion transition token。graph token加到历史token并经共同history adapter，最后拼接20个future query，总序列40。各车独立经过GPT-2，唯一跨车输入来自interaction core。

**具体backbone是当前仓库的预训练GPT-2前4个Transformer block，hidden768，不是完整12层。** 固定使用models/gpt2；冻结预训练权重，每层QKV注入LoRA rank8、alpha16。不换LLM，也不从零训练同名网络。

码本configs/qgnn_final_tokens.json为41×41=1681项；每轴中心31点、两侧各5个尾点。forward中心[-1.2,2.2]步长0.113333m，lateral中心[-1.3,1.3]步长0.086667m。包含尾点的完整范围分别[-5.281850,6.109736]和[-8.231545,6.851280]。全部尾点由15802个train场景的五SNR历史transition拟合，未用val/test。

位移先转本车heading坐标；低速<0.2m/s使用已观察位移方向补充。四角双线性权重插值embedding，并加入Linear4→768的[f,l,overflow_f,overflow_l]/4连续adapter，避免码本越界时丢失幅值。五SNR审计见interface_audit.json；旧中心范围在−10dB约38.6%历史transition越界，不能作为全部SNR唯一尺度。

GPT-2 future hidden、预期motion token和G进入共同coordinate head：LN834→256→GELU→2。未来GT仅用于token CE监督，而非输入。最终输出为 p_hat_last+k*dt*v_hat_last+16*tanh(z_k/4)，dt来自源timestamps。最后层零初始化，初始预测仍为CV，对raw z在0处导数仍为4。

16m是双方共同长尾保护，不是量子改进：train-only审计中，4m上限在−10dB导致至少0.153357m的FDE下界，16m覆盖现有train坐标残差。它不保证未见数据永不越界。历史v1/v2/v3小pilot均用cap4；恢复旧checkpoint须显式cap4，最终full pilot及正式配置用cap16。

## 8. 经典对照与公平性

主经典AdaptiveClassicalCore：相同逐车GRU32、自身旁路64、物理e16/risk和patch，内部为3层hidden64、4-head的pair与rooted-triplet attention。pair网络144→128→68，triple网络240→128→68；64维value和4维score，score加log risk后mask-softmax。local/pair/triple融合，残差LN，三层拼接后读出64维。

附加经典LegacyHigherOrder保留原有64宽逐帧pair+triplet。正式结论以预先选定的强经典为参考，当前没有穷尽经典最优。经典相关路由、傅里叶或张量交互仍是有价值的反驳对照。

共享缓存、数据划分、历史权限、dt、选车、Tokenizer、GPT-2、LoRA、预测头、loss、种子、曝光和checkpoint规则。三组GPT-2/Token可训练参数初始SHA256相同。经典core参数更多仍保留。主比较同epoch，附墙钟；后续调LR使用双方同一网格与试验数。本轮单LR是受控pilot，不代表双方已充分调优。

## 9. 冻结训练与评价协议

五SNR分别训练，保持同一个五SNR码本和架构；SNR为−10/−5/0/5/10dB。正式seed集合2026/2027/2028，本轮pilot只有2026。B32，20epoch，AdamW；非LoRA LR3e-4，LoRA LR7.5e-5，wd2e-4；epoch cosine的eta_min=2.4e-5；梯度裁剪3。

每scene先在有效车辆和20步上平均欧氏距离得到ADE，最后一步在车辆上平均得到FDE，再作scene宏平均。loss=ADE+0.5FDE+0.035*tokenCE；CE在有效车与未来token上平均。根据validation J=ADE+0.5FDE选择同一个checkpoint并同时报告ADE/FDE。固定曝光，不按一方暂时领先决定结束。

正式目标是在强经典对照上，多seed、五SNR宏平均ADE和FDE均争取5%改善，同时报告每seed、每档、子集、整体和墙钟。本轮不读取test，test只在后续全部协议锁定后评价。重叠窗口的不确定性按时间块/轨迹组处理，不采用独立窗口假设。

本轮两次量子结构修正后停止架构搜索。后续为冻结结构训练与预注册消融：去ZZZ重训练、去纠缠门重训练、统计池化读出、固定量子参数或经典相关路由。post-hoc删除仅表示OOD依赖，不能替代重训练因果消融。

## 10. 资源预算：线路、硬件与模拟分开

令P(N)=1（N≤8）或N（N>8），Q=min(N,8)，C=4，D=3。精确模拟每scene为C*P(N)条量子状态演化，整个batch为B*C*P(N)。不乘20历史帧，也不为每条edge单独调用一次VQC。每条线路保存三个中间快照；状态维度为2^Q，Q8时256个complex64振幅，每通道单状态仅2048字节，但autograd、角度表、MLP与GPT-2显存另计。

每通道每轮有3Q个局部RY/RZ/RX及C(Q,2)个ZZ、C(Q,3)个ZZZ；Q8、D3合计324个抽象旋转门。若ZZ分解为2CNOT+1RZ、ZZZ为4CNOT+1RZ，单通道三轮共840个CNOT，另有324个单qubit旋转；无路由、无并行时顺序门深度上界1164。实际拓扑路由可能更贵。因此D=3是宏观交互轮数，绝不等于硬件原生深度3。

硬件不能免费复用已测量快照。每轮分别估计全X、全Y、全Z三个测量设置；全Z样本可同时估计Z/ZZ/ZZZ。四通道三轮需36*P(N)个不同线路前缀/测量设置，每个设置还需shots。asinh三体刻度0.005会放大小相关测量噪声，单个有界量达到约0.005标准误差的最坏量级已约4万shots，累积量组合还需误差控制。本项目不承诺此硬件成本可行。

训练用状态向量autograd，不用parameter-shift。硬件求导还涉及输入相关旋转出现次数，不能只按12个共享RX参数估算。线路编译、状态演化和所有经典辅助算子的成本均计入端到端step time。

## 11. 参数、显存与时间实测

来源：reports/qgnn/engineering_checks_final.json，RTX4090，Torch2.13.0、PennyLane0.45.1、transformers4.57.6，cap16。数值是GPU实测，不由参数量推测。

| 项目 | RC-HQGNN | 主经典 | 原有高阶经典 |
|---|---:|---:|---:|
| interaction core参数 | 67,357 | 323,138 | 268,674 |
| 共同LoRA参数 | 98,304 | 98,304 | 98,304 |
| 共同其他可训练参数 | 4,052,375 | 4,052,375 | 4,052,375 |
| 全部可训练参数 | 4,218,036 | 4,473,817 | 4,419,353 |
| 冻结参数 | 67,736,832 | 67,736,832 | 67,736,832 |
| B32训练step中位数/s | 0.103810 | 0.081396 | 0.081157 |
| 峰值allocated/GiB | 3.6380 | 3.6693 | 4.0715 |
| 峰值reserved/GiB | 4.6758 | 4.6816 | 5.0117 |

Q core的67,357参数包含12个直接RX角、2,000个角度编译器参数、65,345个逐车编码/连续value/gate/readout参数。不能将全模型参数说成12。相对主经典，graph参数少约79.2%，端到端可训练参数只少约5.7%；单步反而约慢27.5%，没有计算加速结论。

full15802、B32每epoch494步。按孤立微基准，仅20epoch训练计算约0.285小时；还须加入数据加载、validation、checkpoint及系统开销。本轮完整train/val pilot的实际总墙钟在实验节报告。正式单run预算冻结为7200秒；超时保存并暂停，需要依据日志决定续跑，而不是自动无限延长。五SNR、三seed、多个对照的总成本不能用单run成本代替。

动态N core-only、B16、forward+backward实测：N8/12/16/20约0.0338/0.0377/0.0390/0.0400秒，allocated约0.069/0.273/0.347/0.421GiB；每scene线路轨迹数4/48/64/80。这不含GPT-2，不是大N完整训练时间。大N数据迁移后仍需用真实batch测端到端预算，再决定是否启用C2备用。

## 12. 文献来源与迁移边界

[1] Skolik等，Equivariant quantum circuits for learning on weighted graphs，npj Quantum Information 9,47 (2023)，DOI:10.1038/s41534-023-00710-y。采用其图权重直接进入量子演化与车辆置换对称的基础思想；本文件自己的三体、读出与时序设计不归为原文结论。
[2] Raj等，Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler–Leman Hierarchy，arXiv:2606.26873v1，2026-06-25，官方代码SnehalRaj/mp-qgnns。其子空间、双寄存器与set-j-WL构造只作为高阶和独立后端验证参考；当前ZZ/ZZZ线路不满足相同构造前提，不继承其WL定理或56-qubit实验结论。
[3] Thabet等，Quantum Positional Encodings for Graph Neural Networks，ICML2024，PMLR235:47965–47996。支持研究图量子相关表示，但量子PE再交给经典GNN不是本项目主架构。
[4] Mhiri等，Constrained and Vanishing Expressivity of Quantum Fourier Models，Quantum9,1847 (2025)，DOI:10.22331/q-2025-09-03-1847。作为反面边界：更多频率、qubit或线路并不自动提供有用表达和任务精度。

以上通过出版社/arXiv HTML及官方GitHub核查。具体SinD效果、3轮/4通道选择、参数预算与代数推导来自本项目代码、实验和推断，不是这些文献已证明的结论。

## 13. 本轮真实实验结果与冻结验收

机器汇总：`reports/qgnn/final_round2_results_20260919.json`，生成脚本`collect_qgnn_final_results.py`。所有数据为train/validation，未打开test。

### 13.1 小规模结构迭代（4096 train、1880 val、12epoch、cap4）

| 版本 | Quantum ADE/FDE | Matched Classical ADE/FDE | 结论 |
|---|---|---|---|
| v1 | 0.601817 / 1.255976 | 0.583580 / 1.205876 | 量子未胜出 |
| v2 | 0.608360 / 1.265798 | 0.581917 / 1.198834 | 量子相关增强不等于指标改善 |
| v3最终结构 | 0.600826 / 1.245823 | 0.581917 / 1.198834 | 比v2改善，但仍输经典 |

v1还有未削弱的原有高阶经典：0.610172/1.138172，J1.179258。不能只看ADE忽略它更好的FDE。它未完成最终cap16全训练集重训，因此本轮最终表不是“所有经典中最强者”的穷尽评测。

### 13.2 最终统一接口（15802 train、1880 val、20epoch、cap16）

| 模型 | ADE/m | FDE/m | J | 选中epoch | 完整训练墙钟 |
|---|---:|---:|---:|---:|---:|
| RC-HQGNN + same GPT-2 | 0.51094460 | 1.12605076 | 1.07396998 | 17 | 1146.55s / 19.11min |
| Adaptive Classical + same GPT-2 | 0.49146503 | 1.07928883 | 1.03110945 | 13 | 891.65s / 14.86min |

双方均20epoch、9880steps、seed2026、0dB、B32，共同参数初始SHA和训练顺序SHA完全一致。完整训练源commit为`e2f617a6e0711d55eeda899de71d9d0604ffcfae`。量子相对经典ADE差3.9636%、FDE差4.3327%，没有达成全局5%优势。全训练集、共用cap16与小pilot的cap4/少样本协议不同，不能将两者差额归因给量子结构。

### 13.3 预先定义的交互分层

| validation子集 | 窗口数 | 量子ADE相对改善 | 量子FDE相对改善 |
|---|---:|---:|---:|
| 全部 | 1880 | −3.96% | −4.33% |
| closing pairs ≥10 | 826 | −0.73% | −0.19% |
| closing pairs ≥15 | 140 | **+4.29%** | **+6.35%** |
| closing pairs <10 | 1054 | −6.29% | −7.25% |

closing采用感知历史定义的30m内且closing>0.5m/s的车辆对数，阈值10/15来自先前诊断，不按本轮预测误差挑选。高动态140窗口中Q的ADE/FDE为0.527534/1.152910，经典为0.551162/1.231053。这是任务匹配的局部正信号，不抵消整体负结果；该子集不是独立测试集，窗口重叠、单seed，不作显著性结论。

### 13.4 最终checkpoint机制诊断

正常：0.510945/1.126051；删除全部ZZ与ZZZ：0.623443/1.436022；只删除ZZZ：0.523522/1.162552；清空graph context：1.064923/2.432902。说明训练后模型依赖交互路径，但这些是post-hoc分布外扰动，不是重训练消融，更不是经典不可能替代的证据。删除ZZZ只删除显式三体相位，不保证连通三体相关K3为零。

### 13.5 当前验收结论

一级架构、数学接口、资源预算：FROZEN。实现、梯度、GPU、0dB实际pilot：PASS。公平配置与test封闭：PASS。全局量子优势：NOT ESTABLISHED，当前受控pilot整体为阴性。冻结的是可实现、可复现、成本可接受且在目标高动态子集出现正信号的一套研究架构，不声称数学上已找到所有路线中的全局最优。

## 14. 最终工程修复、复现与停止边界

### 14.1 边界修复与独立核验

更严格检查发现：N>8的张量包含不足7个有效邻居时，−inf filler可能再次选中root并误当有效车辆，导致4车与填充到20槽输出相差0.06696。`8305cf4`已修复：neighbor mask来自候选有效性，无效patch历史清零，并在FinalModel入口清理padding。N≤8原本路径未变，正式缓存始终8槽；修复后最终checkpoint重载的1880窗口正常ADE/FDE逐数一致。

`additional_checks_and_provenance.json`验证真实已安装代码，而非临时mock：量子跨patch阈值padding误差4.44e−16，经典同样通过；重复history车辆置换通过；全mask输出0；独立PennyLane的X/Y/Z及二三体读出最大误差2.22e−16。训练/验证文件、感知缓存、GPT-2权重和码本的SHA256均已保存，没有读取test数据。`padding_boundary_regression.json`保留修复前后证据。

`resume_regression.json`真实执行“连续4steps”与“第1step预算暂停、恢复到4steps”：模型参数最大差异0，validation ADE/FDE差异0；不是仅检查存在resume参数。

### 14.2 恢复与运行规则

所有小实验使用独立进程、固定GPU、job.json记录PID/命令/日志，每25steps heartbeat，每100steps与每epoch保存atomic checkpoint。last.pt包含模型、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、epoch/batch/global step、累计loss和best记录。summary.json持续更新；异常写failure.json。正式run默认硬预算7200s，预算到时保存并标记PAUSED_BUDGET，使用原命令加--resume恢复；修改源码或码本会触发兼容性检查，应使用新run目录而非混用旧checkpoint。已完成的本轮pilot不需要再恢复。

### 14.3 冻结结构内的下一步与失败条件

不继续本轮第三次结构改造，不立即扩展成五SNR×多seed大扫参。优先用同架构做0dB的配对seed2027/2028复核，以及双方对等的小LR网格；检查高动态增益是否重复、较低交互窗口的退化能否在不损害强经典的条件下减轻。若整体仍稳定更差，保留阴性结果，停止宣称该冻结方案已具备全局量子优势。硬件有限shots鲁棒性、真实大N轨迹质量、五SNR正式训练及最终test均未完成。

唯一成本备用C2为44,287个core参数，不按验证指标临时切换；若C4单run超过2h，先审计数据/GPU与step瓶颈，2–4h须说明，超过4h需特别论证，不接受无解释的约12h方案。当前N8实际19.11分钟无需降配。

虽然量子轨迹调用数C*P(N)在N>8时线性增长，整个程序并非O(N)：全局风险构建O(N²)，所有目标邻居排序约O(N²logN)，逐patch有O(Q³)三体描述，状态演化含2^Q因子；这里只把Q上界固定为8。GPT-2还随有效车辆数增长，大N完整端到端成本应重新实测。

仅供后续独立配对复核的启动例（本轮没有启动）：

```bash
cd /home/dell/YrM/ICCT
CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python scripts/train_qgnn_final.py --kind quantum --seed 2027 --snr 0 --epochs 20 --batch-size 32 --depth 3 --channels 4 --quantum-version 3 --correction-cap 16 --run-dir reports/qgnn/frozen_quantum_0db_seed2027 --max-seconds 7200
CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python scripts/train_qgnn_final.py --kind classical --seed 2027 --snr 0 --epochs 20 --batch-size 32 --depth 3 --channels 4 --quantum-version 3 --correction-cap 16 --run-dir reports/qgnn/frozen_classical_0db_seed2027 --max-seconds 7200
```

主要交付：本冻结文档、configs/qgnn_final_architecture.json、prediction/qgnn_final/、训练/工程检查/汇总脚本、reports/qgnn下的小型summary/training/config/机制与资源报告。大checkpoint仅留服务器原run目录，不提交Git。

### 可核查的原始文献入口

- https://www.nature.com/articles/s41534-023-00710-y
- https://arxiv.org/abs/2606.26873 与 https://github.com/SnehalRaj/mp-qgnns
- https://proceedings.mlr.press/v235/thabet24a.html
- https://quantum-journal.org/papers/q-2025-09-03-1847/
