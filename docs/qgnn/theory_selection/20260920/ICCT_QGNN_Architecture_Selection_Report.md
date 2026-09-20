# ICCT QGNN Architecture Selection Report

**日期：2026-09-20｜状态：理论选型完成，允许进入原型；尚无新架构的 ICCT 性能验证。**

服务器事实源：`/home/dell/YrM/ICCT`，`qgnn` 分支，`094def5b38fa21ccc09cda2328fe78ee8cc694c1`。本轮未修改正式代码、数据、感知、模型、配置或历史实验；没有新训练。新产物独立存放于 `docs/qgnn/theory_selection/20260920/`。

## 0. 最终选型

**Primary：TRC-QGNN — Temporal Rooted-Configuration Quantum Graph Neural Network，时序目标条件配置空间量子图网络。**

唯一主创新是：**将“目标车与哪些邻车共同作用”作为量子配置索引，在保留真实历史顺序的条件下做相干的配置消息传递，再由可测量的角色相关矩形成时间/阶次交互 token。**

**Backup：TES-QGNN — Temporal Equivariant Spin-Graph Quantum Neural Network，时序等变联合自旋图网络。**

它在每车两个量子比特的有界 ego 图上直接做图条件联合演化；不用配置索引、不依赖后选择，也不引入 feedback/controller。

本报告选择的是**最值得下一轮检验的机制**，不是已经证明的最优 predictor。Primary 是项目定制设计，不冒充 Raj 原版、SQM 原版或已发表算法。文献支持其构件和数学可行性，ICCT 历史支持研究方向；ADE/FDE 优势仍是假设，存在明确的失败出口。

不选择“直接继续最强 Raj”：full-train 近乎打平，且最强适配版包含物理邻接权重匹配、直接振幅读出与非酉重上传问题。也不选择“直接照搬 SQM”：其标量编码、star 信息边界与顺序更新对称性尚不足以支撑本任务。

---

## 1. 当前任务的真实瓶颈

保留已完成的系统审计，不重新选数据/ISAC/GPT-2。正式链路是 SinD 两段公开四轮车辆数据 → 受控三基站状态感知 → 20帧连续带噪历史 → 多车交互 → Token-based GPT-2+LoRA →20帧未来位置。当前训练/验证/测试为15,802/1,880/2,086窗口。

需要区分四种误差来源：

1. **上下文覆盖**：围绕一个focal选8辆却评价全部8辆，非focal目标可能缺自己的关键邻车。90.73%的窗口来自原本>8辆的候选集合。
2. **多邻车联合关系**：B对A的影响可能取决于C及B–C关系，不是独立A–B/A–C打分的简单相加。
3. **关系的时间形成过程**：同时接近、先后减速、风险强度变化，比静态车辆密度更贴近已有SinD分层证据。
4. **预测端可用性**：交互信息能否通过readout/softtoken被GPT-2和坐标头使用。

不是已证实的瓶颈：1-WL无法区分某类无属性图、长距离消息传不过去、量子比特不够、Token只是粗速度分桶、LoRA位置不合适。当前小图近乎全连接，连续属性丰富；不能借这些泛化标签倒推任务需要量子。

### 1.1 不可忽略的系统边界

- sensing是逐车带噪极坐标量测与融合，假设检测/身份关联成功，不是漏检/错关联任务。
- 当前离线选车用GT历史，正式模型用感知历史；五SNR共用理想化选车集合。未来更大context/routing必须使用感知历史，不暗中使用GT关系。
- 同一窗口车辆必须有未来标签，是离线评测资格；不能把“有未来标签”与“使用未来位置选邻居”混为一谈。新增context-only邻车只要求历史可见，不强制未来完整。
- 当前GPT-2是预训练前4个block，hidden768，QKV LoRA rank8/alpha16；并非完整12层，也非重新训练同名小Transformer。
- 当前41×41非均匀位移码本使用四角插值加连续overflow adapter；输出是CV＋连续修正，而非逐个离散Token采样后积分。
- 同一连续片段的重叠窗口不是独立样本；泛化统计必须按时间块/车辆关系簇而不只按窗口重采样。

## 2. 历史成败机制：哪些证据可以复用

### 2.1 早期QGNN/QGAT：不要误标成SinD实验

早期全图电路使用每车P/V两种角色比特和物理ZX耦合；QGAT后续使用索引、query/key重叠、value态与Pauli读出，也尝试物理edge进入score/value。若主要value/message/update仍在经典网络，局部PQC增强不等于主要交互量子化。

对应数据是旧 `f01d` 流程，训练5,549 origins，SNR5/10/15/20，不能与当前SinD0dB直接比较。三seed的QGAT总体输于经典对照；factorial中增加state qubit本身无收益，增加局部性和qubit的组合有改善，但编码和读出同时变化，不能归因为纯qubit数效应。

旧bottleneck报告中的高Q/C错误相关、后期训练/验证分离值得保留；“SNR改变却大部分输入不变”属于旧前端，不能套到当前连续sensing cache。

### 2.2 RC-HQGNN：联合态存在，但相关性不等于有效预测信息

C=4、每通道8qubits、3轮联合Z/ZZ/ZZZ与非对易局部旋转。单车GRU32；pair16、triplet32进入耦合；X/Y/Z、K2/K3及关系携带读出汇成64维。

同轮SinD full-train、0dB、seed2026：

| 模型 | ADE | FDE | J=ADE+0.5FDE |
|---|---:|---:|---:|
| RC-HQGNN |0.51094460|1.12605076|1.07396998|
| Adaptive Classical |0.49146503|1.07928883|1.03110945|

高closing的140窗口子集有正信号，但低closing更差。去耦合后变差说明模型使用了耦合，不证明同功能经典模型做不到。增大phase/correlation没有修复整体差距。K3是测量分布的连通累积量，不是纠缠见证。

另一个预测经典残差的representation probe中，丰富读出R²约0.922，最终local＋quantum64约0.919；不能把“所有失败都来自64维/角度压缩”写成已证实事实。该probe也不证明future信息已充分保留。

### 2.3 Scene-Adaptive/feedback：有小贡献，但没有改变总体排序

历史统计/关系key控制耦合，前一层K2/K3反馈调下一层phase/basis。中性初始化能够恢复旧族，故不能由优化失败直接推导函数类更弱。

真正重训练on/off的后续报告给出：Qon J=1.221929276，Qoff=1.229810451，反馈约贡献0.64%；Con=1.191770997，经典仍明显更好。固定checkpoint置零与从头训练删除回答不同问题。history-only winner routing的block-held-out AUC约0.5，不支持把逐场景oracle空间当作可实现开关。

**禁区：继续以增大K3、加controller或自动量子/经典切换作为主创新。** 硬件反馈还需测量、重复制备和串行执行，不能免费读取未坍缩的状态。

### 2.4 Weighted Multi-j Raj-style：最强已有信号，但归因尚不完整

j=2/3两个子集分支、embedding D=6/k=3即20振幅；物理加权Johnson子集Hamiltonian的 `exp(iH)`，Cayley/compound特征变换，四个深度快照，最终按子集归属池化到每车。

实际代码有三个不可忽略的差异：

- 量子侧学习物理加权hopping，主要Johnson对照使用固定二值Johnson邻接。
- 最强候选读取 `z.real/z.imag`，这些不是量子可观测量；全局相位约定会改变这样的特征。
- 将新的特征向量相加后逐行归一化不是一般未知量子态上的确定性酉通道。

这些不使GPU算法得分失效，但限制“量子机制单独带来优势”和“可直接硬件运行”的论证。

| 数据/seed | Q ADE/FDE | Johnson ADE/FDE | 结论 |
|---|---|---|---|
|4096 /2026|0.516457940 /1.102781152|0.525952385 /1.129781810|正信号|
|4096 /2027|0.530895042 /1.148233937|0.550422603 /1.189891210|重复正信号|
|15802 /2026|0.481258665 /1.051172996|0.482314337 /1.049728396|ADE约+0.219%，FDE约−0.138%，J约+0.033%：近乎打平|

两seed小数据均值比的改善约2.70%ADE/2.96%FDE。**为什么full-data追上？** 一个合理假设是：范数保持、参数共享与固定结构形成较强归纳偏置，少量观测时更易泛化；更灵活的经典高阶模型在数据充足后追上。但现有结果还不能单独支持这个解释：

- B32/20epoch意味着small只有2560 updates，full有9880；small最佳19/20epoch，full两方最佳9epoch约4446updates。数据量与优化曝光混杂。
- seed同时改变4096子集和初始化，不能分离data-selection方差与optimization方差。
- physical weighting、local bypass、振幅读出、共享decoder限制和验证集相关性都可能影响差异。

后续需嵌套子集、分离dataset seed/model seed，同时报告matched-update与matched-epoch，并补齐**同物理加权**的经典counterpart。Caro等泛化上界[L14]可解释为何结构复杂度值得关注，不能证明这个已有2–3%就是量子样本优势。

### 2.5 Paper-native/fidelity：应按真实后续隔离结果更新认识

原始论文的V/A/W/M/条件RDM结构与最强工程适配不相同[L1]。单j、删除local、改变readout等同时变化，不能将两个模型差值归为单一M或编码因素。

后续已完成隔离：替换classical feature builder只改变J约0.081%；row-local重上传未救回full paper-math版本。最新服务器已有但叙述freeze未充分覆盖的exp-adjacency结果，J由约1.66431改善到1.59321（约4.27%），但仍远不能解释/恢复最强CandidateA表现。

因此，不再把global重上传或某个最后M列为“已找到的唯一失败原因”。post-hoc小变化也不能否定训练时作用。

### 2.6 SQM Candidate B：只是来源审计，没有ICCT得分

论文的node/edge量子寄存器、U_MSG/U_UPD和boundedstar是可借鉴构件[L2]。但center只有单标量embedding、star缺邻车—邻车关系、顺序update的确切对称条件、可运行官方代码，都仍需补齐。不能拿D2D功控结果替代当前轨迹验证。

## 3. 文献机制与候选淘汰

六族归纳如下。完整来源、代码状态、读到的范围和未核实处列于附录A。

| 架构族 | 联合信息在哪里形成 | 主要问题 | 最终处理 |
|---|---|---|---|
|独立node/edge PQC＋经典聚合|单边/单节点；主要邻域推理经典|不满足本轮主推理定位|淘汰|
|SQM型star message/update|邻车/edge/center量子寄存器|高维时间编码、e_jk缺失、顺序对称性、代码边界|淘汰原版|
|节点寄存器联合自旋演化|全部选中车辆联合态|state随每车qubit指数增加；RC已有失败需实质改变|TES为唯一Backup|
|无根fixed-weight subset Raj型|子集寄存器与feature寄存器|目标角色/时间/readout与实际适配物理性问题，full近打平|不直接继续，保留机制证据|
|walk/kernel→强经典Transformer|walk提供结构偏置；主要推理在后端|会将核心任务交回强经典网络|淘汰整套架构，保留构件|
|目标条件时序配置空间联合演化|配置之间相干混合＋配置内多角色量子关系|新设计待验证，readout和去量子化风险|TRC为Primary|

量子消息传递、图条件演化的基础来自[L1–L3]，多配置/多粒子walk来自[L8–L9]，时空任务依据与强经典反例来自[L12]。**本报告并不把这些不同论文的性能优势相乘或累加。** 对称理论[L4–L5,L17]与重上传理论[L13]只支持特定性质，不保证本项目无barren plateau或优于经典。

---

## 4. Primary 的定义与默认规模

记B为batch场景数，N_t为预测目标数，N_c为该窗口可用的历史候选车辆数，K_i为给目标i保留的context总数（含i）。必须区分：

\[
N_c\quad\ne\quad K_i\quad\ne\quad Q_i.
\]

**原型默认：保持当前8辆目标/输入不变，K_i≤8；S=4个历史块，每块5帧；每角色2个feature qubits；每块4个编码子轮。** 较大context在后续独立factorial中比较N_c=12/16、K=8或12，不把扩充输入的收益计为量子收益。

设目标i的邻车集合 \(\mathcal N_i\) 大小n=K_i−1。定义：

\[
\Omega_i^{(1)}=\{(j):j\in\mathcal N_i\},\quad M_1=n,
\]
\[
\Omega_i^{(2)}=\{(j,k):j,k\in\mathcal N_i,j\ne k\},\quad M_2=n(n-1).
\]

第一分支是**目标＋一辆邻车**，第二分支是**目标＋两辆邻车**。分支编号b不是Raj论文的j-WL阶。第二分支同时保留(j,k)和(k,j)，以有序角色槽保存“哪辆车对应哪条边”，最后读出对邻居角色做对称化；它不是将车辆ID作为数值特征。

不采用先对三车特征作mean/std、再用一个scalar控制整个三车电路。不同组合之间也不是独立PQC后在经典域求和；它们共享同一configuration量子态并发生hopping。

## 5. 单车时间表示 \(H_i\to h_i^m\)

输入 \(H_a\in\mathbb R^{20\times4}\) 仅含sensing估计。时间块 \(B_m=\{5m-4,\ldots,5m\}\)，m=1…4。沿用轻量shared GRU，但在新原型明确采用可逐前缀计算的六维输入：

\[
x_{a,t}=\left[(\hat p_{a,t}-\hat p_{a,1})/10,\ \hat v_{a,t}/10,\ \hat p_{a,1}/50\right]\in\mathbb R^6,
\]
\[
u_{a,t}=\operatorname{SiLU}(W_xx_{a,t}+b_x),\quad
h_a^m=\operatorname{LN}(\operatorname{GRU}_{32}(u_{a,1:5m})_{5m}).
\]

历史GRU在每车独立运行；没有跨车attention/GNN。first-observation原点而非last-observation原点，是为了让每个prefix不引用更晚的观测。它是相对旧shared encoder的设计修改，必须与新classical counterpart同时应用；旧模型得分不能直接拼接为此变体得分。

每块另提取均值位置 \(\bar p_a^m\)、速度 \(\bar v_a^m\)、块均值时间 \(\bar t^m\)。m>1时 \(\bar a_a^m=(\bar v_a^m-\bar v_a^{m-1})/(\bar t^m-\bar t^{m-1})\)；m=1用首块内速度对时间的最小二乘斜率。block内均值只是显式physics，GRU仍看全部20点。

给目标i下角色a定义：

\[
\chi_{a|i}^m=[(\bar p_a^m-\bar p_i^m)/30,
(\bar v_a^m-\bar v_i^m)/10,\bar v_a^m/10,
\bar a_a^m/10,\mathbf1_{a=i},\mathbf1_{a\ valid}]\in\mathbb R^{10}.
\]

完整角色描述 \(d_{a|i}^m=[h_a^m,\chi_{a|i}^m]\in\mathbb R^{42}\)。核心采用一致world-frame相对向量，不强称整个网络SE(2)不变；自己的绝对初始位置保留了可能有用的道路位置。Motion Token仍使用原局部heading规则。

此分工保留“单车运动可预处理”，同时把每个时间块的相对状态、接近关系变化送入量子演化。不是只处理最后一帧静态图。4块是工程假设，不能声称已达到最优时间分辨率。

## 6. Pair 与 higher-order 物理信息

对于有向角色a→b，令r=\(\bar p_b^m-\bar p_a^m\)，u=\(\bar v_b^m-\bar v_a^m\)，d=\(\|r\|\)，c=\(-r^Tu/\max(d,\epsilon)\)，

\[
\tau=\operatorname{clip}_{[0,4]}[-r^Tu/\max(\|u\|^2,\epsilon)],\qquad D=\|r+\tau u\|.
\]

\[
e_{ab}^m=[r_x/30,r_y/30,u_x/10,u_y/10,d/30,c/10,
\tau/4,D/10,(\bar a_b^m-\bar a_a^m)_x/10,
(\bar a_b^m-\bar a_a^m)_y/10,\Delta c/10,\Delta D/10]\in\mathbb R^{12}.
\]

m=1时最后两个差分为0。所有量用源timestamps；分母epsilon固定，padding不参与。TCPA在这里是clipped closest-approach描述，不等同于“真实碰撞时间”。

对三车配置(i,j,k)，保留三个角色描述和**六条有向边** \(e_{ij},e_{ji},e_{ik},e_{ki},e_{jk},e_{kj}\)。不给高维信息先做不可解释的全局MLP压缩。

用于配置图的确定性风险权重取：

\[
w_{ab}^m=\frac{e^{-d_{ab}/30}\,[0.25+\operatorname{sigmoid}(c_{ab}/2)]\,e^{-D_{ab}/20}}{1.25}\in[0,1].
\]

它对a/b交换对称。定义共时接近势：
\[
\eta_{ijk}^m=w_{ij}^mw_{ik}^mw_{jk}^m
\exp[-|\tau_{ij}^m-\tau_{ik}^m|/(1\ \mathrm s)].
\]

\(\eta\)只是额外的物理归纳偏置，不是唯一三车编码；完整有向边仍直接进入feature gates。其时间尺度1s为预注册初值，双方对照一致，不用test调整。

## 7. 量子初态与寄存器

每个配置分支具有index寄存器及feature寄存器。b=1时feature角色为(root,A)，q_f=4；b=2时(root,A,B)，q_f=6。令c=ceil(log2 n)，使用b个c-qubit二进制index槽：

\[
Q_1=c+4,\qquad Q_2=2c+6.
\]

有效初态为：

\[
|\psi_{i,0}^{(b)}\rangle=\frac1{\sqrt{M_b}}
\sum_{s\in\Omega_i^{(b)}}|s\rangle_I\otimes|0\rangle_F^{\otimes q_f}.
\]

二进制非法index或重复(j=j)是零振幅；所有算子在非法子空间延拓为identity。有效子空间模拟与完整二进制电路应在下一轮做等价验证。初态依赖有效mask，不依赖GT或未来，不假设qRAM。

n=0跳过两个分支，只使用本车表示；n=1仅启用b=1；缺失分支的token通过attention mask排除，不用虚构的车辆填充量子相互作用。全padding样本输出严格为0。

均匀合法配置制备是已知有限集合的状态准备问题，不是免费的一次Hadamard就完成；硬件可用受控旋转编译，保守计O(M_bc)门量级，不宣称指数加载加速。

## 8. 结构化数据编码 \(V_m\)：角色和边都可追踪

每个角色的24个单比特旋转角按顺序组装为：

\[
\theta_{a|i}^m=
[\pi\tanh(W_hh_a^m+b_h),\ \pi\tanh(\chi_{a|i}^m),\ 0,0]\in\mathbb R^{24},
\quad W_h\in\mathbb R^{12\times32}.
\]

它由12个学习的历史投影、10个**直接保留的物理/角色分量**和2个零角组成。按四个子轮分成每轮6角，在每角色两qubit上依次施加RX/RY/RZ。所有角色、目标、时间、两阶分支共享W_h。保留12维历史投影仍是压缩，不声称无损；未来可用更高投影宽度消融，但不能掩盖这一边界。

每条有向边12个分量分别控制四子轮×三个Pauli关系门，不先混成一个角：

\[
\phi_{ab,r,\mu}^m=\pi\tanh(s_{r\mu}e_{ab,3(r-1)+\mu}^m+b_{r\mu}),
\quad r=1..4,\mu=1..3.
\]

选 \((P_\mu,Q_\mu)=(Z,X),(X,Z),(Y,Y)\)，定义：

\[
U_{E,r,\mu}^{s,m}
=\exp\!\left[-\frac i2\sum_{a\ne b\in\{i,s\}}
\phi_{ab,r,\mu}^m P_\mu^{(a,0)}Q_\mu^{(b,1)}\right].
\]

每个固定μ中的项两两对易：每角色的0号qubit只承载Pμ、1号只承载Qμ。因此该子块不依赖车辆遍历次序；不同μ子块、node旋转和intra-role门通常不对易。每子轮再给每角色施加共享角 \(\lambda_r\) 的 \(e^{-i\lambda_r Z_{a,0}Z_{a,1}/2}\)。

**一个子轮的确定执行顺序**：所有node RX→RY→RZ；所有intra-role ZZ；edge μ=1→2→3。重复r=1…4，得到配置内特征酉算子 \(F_{i,s}^m\)。

\[
V_i^{(b),m}=\sum_{s\in\Omega_i^{(b)}}|s\rangle\langle s|\otimes F_{i,s}^m+I_{\rm invalid}.
\]

这是block-diagonal controlled unitary，不是逐tuple测量后经典pool。b=2电路中e_jk直接影响邻车角色，两位邻车又通过非对易块共同影响root；不存在强经典GNN先做主要推理。

## 9. 图条件配置消息传递 \(U_G\)

在n个邻车上建立对称矩阵：
\[
A_{jl}^m=\mathbf1_{j\ne l}\sqrt{w_{ij}^mw_{il}^m}\,w_{jl}^m.
\]

单邻车配置基础算子是A。双邻车配置使用：
\[
A_{\rm cfg}^{(2),m}=P_{\ne}\,(A^m\otimes I+I\otimes A^m)P_{\ne},
\]

其中P_≠选出j≠k的有效行列；即一次只替换一个邻居，但保留另一个邻居和root。双配置并非两个独立单配置，因为排除了相同车辆占据，且存在配置联合势。

设 \(\mathcal N(A)=A/\max(1,\max_s\sum_t|A_{st}|)\)，
\[
H_i^{(1),m}=\alpha_1\mathcal N(A^m)+\beta_1\operatorname{diag}(w_{ij}^m),
\]
\[
H_i^{(2),m}=\alpha_2\mathcal N(A_{\rm cfg}^{(2),m})
+\beta_2\operatorname{diag}((w_{ij}^m+w_{ik}^m+\eta_{ijk}^m)/3).
\]

\(\alpha_b=0.5\tanh a_b\)、\(\beta_b=0.5\tanh b_b\)，四个标量跨时间、目标共享，建议初值α=.2、β=.1。H实对称，故
\[
U_{G,i}^{(b),m}=e^{-iH_i^{(b),m}}
\]
严格酉。\(\|H\|\le |\alpha|+|\beta|\le1\)。这些系数是latent演化参数，**不是把量子演化时间等同于真实秒数**；真实时序体现在数据块顺序与源时间特征中。

**完整时间递推：**
\[
|\psi_{i,m}^{(b)}\rangle=
V_i^{(b),m}(U_{G,i}^{(b),m}\otimes I_F)|\psi_{i,m-1}^{(b)}\rangle,
\quad m=1,2,3,4.
\]

m个block后的状态保留前m个block的演化历史。没有幅值相加、逐行归一化、未知态复制或量子反馈控制器。

### 9.1 一个必须保留的设计约束

若只在最后做 \(U_G\otimes I\)，然后仅测量feature：
\[
\operatorname{Tr}_I[(U_G\otimes I)\rho(U_G^\dagger\otimes I)]
=\operatorname{Tr}_I\rho.
\]

这种末端hopping完全不可见。故本设计将hopping放在**后续配置条件V之前**，通过不同配置内操作把index相干性转成feature可观测差异。工程实现若把这两个块随意交换，可能让“量子消息传递”变成无效装饰。

## 10. 为什么不是独立pair message的求和

将某一步前状态写成 \(\sum_t|t\rangle|\varphi_t\rangle\)。经过U与配置条件F后，对feature观测O：

\[
z_O=\sum_{s,t,u} U_{st}U^*_{su}
\langle\varphi_u|F_s^\dagger O F_s|\varphi_t\rangle.
\]

t≠u是不同配置之间的相干交叉项。U的邻接、F_s的角色/有向边以及前序真实时间块共同决定它们。这与“分别算每个邻居message再固定相加”不同。

但区别不意味着经典无法表达：非线性DeepSets、tensor/高阶MPNN、complexunitary网络可表示同样的乘积或交叉项。式子证明的是**信息处理形式不同及非可分性可能存在**，不是对所有强经典模型的表示能力分离。

不宣称得到某个j-WL级别；本设计修改了rooting、encoding、readout与邻接空间，不能继承Raj定理。

## 11. 可测量读出：不直接读取振幅

每个阶次b、每个时间prefixm，提取feature上35维：

1. root两qubit完整Pauli坐标15项：\(\langle P_{i,0}Q_{i,1}\rangle\)，P,Q∈{I,X,Y,Z}且非II。
2. root–neighbor同轴相关12项：rootqubit a=0/1、neighborqubit c=0/1、P=X/Y/Z，观测 \(b^{-1}\sum_{\ell=1}^bP_{i,a}P_{\ell,c}\)。
3. b=2时的8项三角色Z矩：\(\frac12\langle Z_{i,a}(Z_{A,c}Z_{B,d}+Z_{A,d}Z_{B,c})\rangle\)，a,c,d∈{0,1}；b=1置0并用阶次标识。部分项冗余是可接受的固定布局，不宣称35个独立自由度。

所有观测都是有界Hermitian算子的期望，均对整体相位不敏感。选择raw moments而不是放大小量K3，避免把“更大高阶相关”当优化目标。需要K2/K3时仅作辅助诊断。

\[
Z^Q\in\mathbb R^{B\times N_t\times2\times4\times35},\qquad
M^Q\in\{0,1\}^{B\times N_t\times2\times4}.
\]

没有root子集后选择、无1/P_proj归一化，因而避免对应小概率不稳定。但trace掉index、仅保留局部低阶矩**仍然可能丢失重要配置内容**。本设计并没有“解决所有measurement bottleneck”；这是Primary的主要kill风险之一。

9组Pauli测量设置足够覆盖这些观测：root两qubit的3×3基组合；相同root轴时所有neighbor测同轴，包含全X/Y/Z，从而覆盖12种两体同轴矩与ZZZ。纯statevector中可直接计算期望；硬件每个prefix/设置均需重复制备，不能在同一个坍缩态上免费拿全部前缀。

## 12. 置换、padding、时间与稳定性

### 12.1 节点重标号性质

在给定相同物理context集合下，对邻车置换π，配置索引变为P_π。物理关系满足A′=PAPᵀ；normalized operator和diag势也协变。于是U′=PUP†，V′=(P⊗I)V(P†⊗I)，均匀合法初态按相同置换映射。最后I_index⊗O读出不变；对所有目标一起重标号，输出按目标重标号等变。

这个论证不依赖对非对易边门用车辆ID排序。history-based selector的tie-break也必须只用物理历史的canonical键；完整相同历史的车辆在当前输入权限下应可交换。浮点排序边界附近的扰动稳定性另作测试，不能由置换证明替代。

无效index算子identity，初态无非法振幅；因此增加padding不应影响有效root。n=0/1按上节分支mask处理。

### 12.2 时间顺序

通常 \([V_m(U_m\otimes I),V_{m'}(U_{m'}\otimes I)]\ne0\)。故交换形成关系的先后顺序可改变读出；这为“先后减速”提供了结构通路，不证明模型必学到正确时间因果关系。

### 12.3 稳定性不是自动鲁棒性

对两个输入x,x′，酉乘积的telescoping界给出：
\[
\|\psi(x)-\psi(x')\|\le
\sum_m\|U_m(x)-U_m(x')\|+\sum_m\|V_m(x)-V_m(x')\|,
\]
\[
|\langle O\rangle_x-\langle O\rangle_{x'}|
\le2\|O\|\,\|\psi(x)-\psi(x')\|.
\]

其中图选择不跳变且前处理Lipschitz常数受控时，酉传播不产生逐层norm爆炸；但编码角、GRU、TCPA近零分母、hard routing和decoder都可能放大噪声。这个界也适用于等价经典酉实现，不是量子特有的鲁棒性定理。

### 12.4 已执行的小型代数检查

`algebra_sanity.py/json`在辅助CPU容器运行，使用4邻车/12个有序配置/2featurequbits的简化实例，不读取ICCT数据，不训练，不是完整Primary实现。

- norm误差0；conditional unitary误差6.66e−16。
- 重标号读出差2.50e−16；整体相位差1.11e−16。
- 末端index-only hopping的读出差1.11e−16。
- 反转时间顺序的最大读出差0.43856。
- 去除配置相干后的最大读出差0.17967。

这些只验证恒等式与“机制确实能产生不同输出”的存在性，不验证可训练性、样本效率、实际编码保真或ADE/FDE。

## 13. QGNN → GPT-2 interface

量子输出不需先压成一个64维向量。每个35维readout通过共享adapter：
\[
a_{i,b,m}=\operatorname{GELU}(W_1\operatorname{LN}_{35}(z_{i,b,m})+b_1)\in\mathbb R^{128},
\]
\[
q_{i,b,m}=\operatorname{GELU}(W_2a_{i,b,m}+b_2)+e_b+e_m\in\mathbb R^{768}.
\]

W1:35→128，W2:128→768；e_b两个阶次embedding、e_m四个时间embedding。只逐token变换，无跨车经典GNN。再由本车h_i^4形成一个own token。

每车输入按以下48位置排列：

\[
[\mathrm{Own}_i;\ q_{i,1,1},q_{i,2,1},\ldots,q_{i,1,4},q_{i,2,4};\
19\ \mathrm{MotionTokens}_i;\20\ \mathrm{FutureQueries}].
\]

历史motion embeddings可加有效q的均值，再经过原history adapter；不重复运行量子电路20次。未来query可对不同时间/阶次的prefix采取不同attention，形成随prediction horizon变化的利用方式，而不是额外运行20套QGNN。

为兼容当前坐标头，令有效a的均值与本车h拼接为160维，LN160→Linear160→64→GELU得到g_i^{head}。共享坐标头继续用[futurehidden,g_i^{head},expectedmotion]。g→GPT与g→head两通路都保留，基线完全相同；后续用干预检查收益到底经哪条通路传递，不宣称量子增益必由LLM贡献。

**重要公平隔离：**第一原型同时提供将8个z池化成旧40位置接口的选项；对Q/C做相同40/48位置factorial。如果双方主要靠更好的48位置接口提升，应归于interface，而非量子核心。

## 14. Motion Token与LoRA决定

**首轮保留现有41×41码本、四角插值、连续delta/overflow adapter和QKV LoRA8。** 不引入额外learnablecodebook、不是转回整数速度桶。保留TokenCE与连续预测头，准确称为“Token-conditioned continuous trajectory prediction”。

理由是当前硬量化损失并非已证实主瓶颈。低SNR相邻位置差分噪声和低速heading不稳可能更重要；已有连续adapter已缓解码本溢出，改codebook会引入新的归因混杂。

后续只有出现train/val量化覆盖或heading稳定性异常时，才测试：相同train-only非均匀中心的多分辨率输入，或位置/速度一致性的单车连续支路；Q/C共享。未来Token标签第一步使用带噪last-position与GT future的差，要审计其噪声敏感性；不能靠test重新拟合码本。

LoRA优先保留每层fusedQKV。若两方都不能利用交互prefix，再同预算比较QKV与QKV+attention-output `c_proj`，不是直接给Q多一个adapter。暂不扩大到MLP LoRA或增加GPT-2层数。

## 15. 车辆规模与routing：不把丢失的车辆交给量子“补全”

### 15.1 当前可直接实施的第一步

先在相同已选8车输入/目标上检验核心；这不修复context缺失，但能保证新核心与现有结果具有可比较入口。每个目标都用该8车集合，角色root随目标改变。

### 15.2 N_c=12/16与dynamic N的独立研究

预测目标集合、window及评价权重固定，只增加可用历史context。先从同一SinD正式split内取得历史存在的车辆，再用**同一已冻结sensing算法**生成额外context估计，存新的派生cache；这属于下一轮授权工程，不在本轮改正式cache。

对目标i，按四块中最大历史风险 \(\max_m w_{ij}^m\) 排序，保留K−1邻居；相同风险时按最近距离和完整观测历史canonical键稳定打破平局。基础版不使用会随query改变的learnedrouter，不用未来挑车。

比较N_c∈{8,12,16/dynamic},K∈{8,12}时，同时报告：按sensing物理风险定义的累计风险覆盖率、非focal目标的覆盖差异、邻车集合随噪声的稳定性，以及ADE/FDE。风险覆盖是诊断代理，不是未来因果重要性的真值。

**设计默认先扩大candidate而不扩大register。** K8→12只有当同target/context增加存在稳定信息收益时再启用。报告给出K16算术用于上界分析，不建议首轮直接K16。更大N_t会增加GPT-2调用数；候选N_c增长与目标N_t增长必须分别测成本。

---

## 16. Matched strong classical baseline

### 16.1 主对照 RTCN：Rooted Temporal Configuration Network

与Primary共享H/GRU32、Ω1/Ω2、全部node42与directededge12、相同A_cfg/η、时间四块、context/mask、readout输出宽度/8token/GPT-2/LoRA/head/loss/初始化规则。

每个配置的classical输入为按角色拼接：
\[
x_s^m=\operatorname{concat}(d_i^m,d_{s_1}^m,[d_{s_2}^m],\{e_{ab}^m:a\ne b\}).
\]

b1输入2×42+2×12=108维；b2为3×42+6×12=198维。分别Linear→d→SiLU，d取64/128，给经典方两种容量。

其四时间块更新定义为带物理结构bias的configuration attention：
\[
t_s^m=u_s^{m-1}+f_d(x_s^m),\qquad
\ell_{st}^m=\frac{(W_Qt_s^m)^{T}(W_Kt_t^m)}{\sqrt d}
+\gamma\log(\epsilon+(\widetilde A_{cfg})_{st}),
\quad t\in\{s\}\cup\mathcal N_{cfg}(s),
\]
\[
a_s^m=\sum_t\operatorname{softmax}_t(\ell_{st}^m)W_Vt_t^m,
\quad u_s^m=\operatorname{LN}(t_s^m+\mathrm{MLP}_{2d\to2d\to d}([t_s^m,a_s^m])).
\]

u^0=0；定义\(\widetilde A_{cfg}\)的off-diagonal等于A_cfg，diagonal为1加量子侧使用的同一物理势v_s（不含可学习β），从而classical也获得η。W_Q/W_K/W_V不加bias，MLP用SiLU。读出用mean与max拼接、Linear2d→35，每个阶次/时间生成一个向量。允许classic的物理边bias可学习，不能刻意只给固定二值邻接。输出35维不是把经典信息限制到Q幅值：两方同接口，并另报更宽classic接口敏感性。

近似参数量（四时间步共享、b间两套input，其余共享）：
\[
P_{RTCN}\approx(108d+d)+(198d+d)+3d^2+(6d^2+3d)+2d+(2d\cdot35+35),
\]

即约9d²+383d+O(1)（d64约61,400，d128约196,500），另加共享GRU、adapter/GPT。具体实现如attention biases、独立层权重须用实际 `numel`重新核算，不伪装精确实例化计数。

更强全configuration attention可作为同预算容量控制；不能只与最弱pairwise比较。

### 16.2 必须区分三种“经典反事实”

- **功能匹配学习模型**：上述RTCN与weightedJohnson/强时空attention，检验量子归纳偏置的任务价值。
- **量子机制删除**：同配置输入下去index相干、去e_jk、只保留最后时间块，各自重训练；固定checkpoint干预只作辅助。
- **精确经典酉模拟器**：用complex张量执行完全相同U/V/readout，在相同参数下必须数值一致。它是验证与成本对照，不可能被相同数学函数“量子击败”。可再训练一个放宽酉约束的complex configuration网络作为capacity/control，但不是精确同模型。

去相干控制可用ρ=Σ_s|s><s|⊗ρ_s，hopping后ρ′_s=Σ_t|U_st|²ρ_t，再F_sρ′_sF_s†。它删除配置间相干，同时保留配置内量子关系；不等同于一个廉价全经典baseline，不能用其更高density-matrix模拟耗时宣称量子速度优势。

### 16.3 Project-level references

保留既有AdaptiveClassical、GatedPairTriplet，以及按相同物理权重强化的Multi-j Johnson。历史得分只作参考，新输入/encoder/interface下需重训。强baseline不因其参数更多而移除，不将当前弱pairwise作为唯一对照。

## 17. Primary复杂度、qubits、calls和显存

### 17.1 三种不同的成本计数

1. **有效state轨迹数**：两阶×目标数；模拟一次演化链可保存四prefix。
2. **实际算子工作量**：configuration受控编码、矩阵指数/矩阵乘、观测计算，随M增加。
3. **硬件circuit executions**：prefix×measurement setting×shots，加上参数梯度的重复制备。

三者不能互换。“每target两条线路”不等于O(1)门数。

| K含root | n | M1/M2 | Q1/Q2 | 完整state维度 | 有效state维度 | 两分支complex64有效态/target |
|---:|---:|---:|---:|---|---|---:|
|8|7|7/42|7/12|128 /4096|112 /2688|21.875 KiB|
|12|11|11/110|8/14|256 /16384|176 /7040|56.375 KiB|
|16|15|15/210|8/14|256 /16384|240 /13440|106.875 KiB|

K12与K16的binaryqubit数相同，但有效配置、门数和矩阵成本显著不同。不存在“qubits没增加所以免费扩大context”。完整两分支state/target是33/130/130KiB。

模拟calls/scene=2N_t，calls/batch=2BN_t；K8、B32、N_t8为512条有效state轨迹，原始有效state约5.47MiB，四prefix约21.88MiB。N_t12/16、K固定8分别约8.20/10.94MiB（不计autograd）。这些不是训练总显存。

每个配置每块的Pauli旋转数：b1为80，b2为156（含node、intra、directededge）。四块保守硬件受控旋转计数：
\[
G_{enc}/target=4(80M_1+156M_2),
\]

K8/12/16对应28,448/72,160/135,840个带index条件的Pauli旋转，**还没有展开多控门为原生门**。因此“12qubits”不等于近端设备浅电路。模拟可按角色/μ融合对易门并并行处理index行，不需创建全矩阵V。

U_G在有效配置空间dense matrix_exp为O(SΣ_b M_b³)，作用到feature为O(SΣ_b M_b²2^{q_f,b})；featuregates约O(SRΣ_b M_b 2^{q_f,b}[q_f,b+k_b²])。总体乘BN_t。K固定时对N_t线性；若K也随N_t增长，双配置dense指数出现约O(K^6)，绝不声称整个模型线性扩展。

硬件U_G可在小二进制index空间通用unitary合成，保守O((2^{bc})²)级原生门编译；或用稀疏Hamiltonian/productformula，但需显式误差/门数分析，不假设免费oracle。matrix_exp与任何gate-product近似必须验证到目标容差，且排序不能破坏置换性质。

### 17.2 参数

按本报告明确的层定义：单车encoder6,624；quantum-data projection396，edge尺度/偏置24，intra4，hop/potential4，共**428个编码/演化参数**，整个交互前处理7,052。428不是“全部模型参数”，也不是独立可训练gate occurrence数；同一参数作用于大量数据相关受控旋转。

全部新模型预期可训练参数的层级算术合计**4,252,073**：包含1,291,008 motionembedding、1,182,720 historyadapter、1,294,225 tokenhead、98,304LoRA和其余head/adapters。尚未实例化，不是实测 `numel`。完整分项在 `complexity_tables.json`。

### 17.3 4090训练可行性

默认complex64 reduced-state、合并对易Pauli块、按时间块checkpoint；不得为每条受控门保存一份完整state。B32应作为待profile目标而非保证，必要时microbatch8/16＋累积达到effective32。classical对照有效batch、数据顺序和loss归一化相同。

GPT-2输入40→48，attention位置对数约×1.44，tokenwise部分约×1.20；整体不会简单等于其中一个倍率。两张4090默认各运行一个对照/seed，不能把显存相加成48GB单卡。DDP会有通信与有效batch变化，未测不声称2倍提速。

已有Raj full20epoch耗时约18.7/20.1分钟只是历史参考；本架构更多root/feature操作，不能直接沿用该耗时。20epoch/B32共有9880updates。下表是**吞吐假设换算，不是profile结果**：

| 实测若达到的step时间 | 纯train时间，不含val/启动/checkpoint |
|---:|---:|
|0.25s|41.2min|
|0.50s|82.3min|
|0.70s|115.3min|
|1.00s|164.7min|

接近2小时总预算通常需有效step约0.5–0.6s并控制validation开销；实际由第一原型短profile决定。若到1s，不自动否定科学方案，但必须说明超过理想预算；若优化后>2s、占满24GB仍无法effectiveB32，则优先降低K/门轮数并重新声明模型版本，不无限训练调参。

### 17.4 测量与梯度成本

四prefix×两阶×9设置=72种完整线路设置/target；N_t8每scene576种，B32为18,432种，再乘shots。S_shots=1000时每target72,000执行、每scene576,000；这是硬件推理量级，非当前GPU一次forward。

单个±1Pauli估计标准差≤1/√shots；1000shots约0.0316。联合R个观测，Hoeffding保守保证可取 \(S\ge2\log(2R/\delta)/\epsilon^2\)，而非默认精确期望免费。此界较保守，也不直接给ADE/FDE界。

模拟训练使用可微矩阵指数/门操作与反向传播。真实参数shift对共享参数的多次出现不能简单说成“每个428参数只多两条线路”；应展开gate-occurrence或使用适用的广义规则，还要计数据编码参数的链式梯度。Primary定位是**量子可实现机制的精确模拟研究**，不在本轮承诺NISQ训练或量子加速。

## 18. Backup：TES-QGNN的完整边界

保留共享GRU32与四历史块；每个选中车辆两比特，初态|0>^{2K}。用与Primary同风格的24角node编码和12角directededge编码，但不使用tupleindex：在同一个K车联合寄存器直接施加所有有向 \(P_{a,0}Q_{b,1}\) 子块及intra/node旋转。每μ内部对易、μ间固定顺序非对易。无显式ZZZ必需项，多邻车效应通过跨轮非对易相互作用形成。

全图N_t=N_c≤8时，可共享一条联合state，读出每车的root15＋neighbor平均同轴12＋其他两车对称三体Z8，四时间块各35维；接4个interactiontokens＋own＋19motion＋20query=44位置。N>8则为每个目标构造K≤8ego，再仅取root读出。各车node编码必须是**自身参照world-frame**，不能因不同root而改变共享整图编码；建议[h_a^m,(pmean−pfirst)/10,vmean/10,amean/10,pfirst/50,valid,1]，共享线性投影/固定physics角。

\[
|\psi_m\rangle=\prod_{r=1}^4(U_{E,r,3}U_{E,r,2}U_{E,r,1}U_{intra,r}U_{node,r})|\psi_{m-1}\rangle.
\]

全图节点置换为qubit成对交换；固定μ内对易和参数共享保证结构等变。读出三体项使用不同邻居j≠k，平均分母显式按有效数；不足2邻居时置0。matchedcontrol使用同context/time/physics的高阶MPNN或tensorgraphnetwork＋相同44位置decoder。

Q=2K，complex64单state K8为16qubits/0.5MiB，K12为24qubits/128MiB，K16为32qubits/32GiB（单态已经超过24GB）。**Backup默认K封顶8，不用K16。** 每block4子轮抽象Pauli门数约4[7K+3K(K−1)]，四block即16[7K+3K(K−1)]；K8约3584。调用数N≤8为1state/scene，大图ego为N_t；batch乘B。训练参数约GRU6624＋同node/edge/intra424，其余adapter/head约百万级；实际计数随4token接口重新实例化核查。

读每车完整2qubitRDM时不能错误宣称9设置覆盖任意全图所有root：为每个root做至多9设置，四prefix最保守36N_t设置/scene；共用全X/Y/Z可减少，但不得默认免费。B32/N_t8全图裸state16MiB；大图逐target补丁增加为B N_t×0.5MiB，autograd和phase表显著更大，需checkpoint/fusedphase/microbatch。其模拟训练可能比Primary慢，也可能受实现差异影响，尚无时间实测；沿用第17节吞吐换算框架，不伪造分钟数。

**仅当Primary的配置trace/readout或conditional编译路径被证实限制任务，而该联合节点态能在相同时间/输入条件下给出更好的可测读出时启用。** 不因为某个seed碰巧更高就随意切换。它是不同失败模式的唯一备用，不是又开一轮无边界方案搜索。

---

## 19. 理论优势链与最强经典反驳

### 19.1 本项目可以检验的链条

SinD Q0：多邻车共同接近的分层更受益于triplet → 任务需要保留root相关联合关系及形成顺序 → 普通独立pair message有不同的参数化归纳偏置，但强经典高阶模型可以解决 → TRC用角色保持编码、配置相干交叉项和时间有序酉演化形成一种受约束的联合特征族 → 多time/order token提供预测端可区分的证据 → 假设在相同context/decoder/训练预算下，尤其高closing的FDE能改善，同时overall不回退。

这条链中的“普通pair参数化可能不合适”不等于“所有经典方法不行”。Primary的潜在价值是**任务适配的归纳偏置**，不是Hilbert空间维度或量子平行性口号。

### 19.2 什么变化才真正降低ADE/FDE

设经典预测误差 \(e=p_C-p_{GT}\)，新模型相对它的修正为Δ。每个点的L2误差变化精确等于：
\[
\|e+\Delta\|-\|e\|
=\frac{2e^T\Delta+\|\Delta\|^2}{\|e+\Delta\|+\|e\|},
\]
分母为0时两者都零。只有修正与原误差足够反向，收益才出现。高阶特征或纠缠更强都不能保证该条件成立；decoder忽略quantum token时Δ也可能≈0。FDE是最后一步的该项，ADE是所有步/目标的平均。

### 19.3 最强反驳必须进入实验

1. **经典精确模拟**：K8有效state不过数千振幅，一个经典complexunitary程序可完全复制，因此本任务没有已证明计算复杂度优势。
2. **经典高阶表达**：RTCN/tensor/attention自然表达联合邻车和时间乘积，nonseparable项不属于量子独占。
3. **归因混杂**：rooting、额外context、4块时间、更多tokens、稳定去噪都可能带来收益；必须两方同步改变。
4. **少参数也可能欠拟合**：428个core编码参数不是证据，可能不足；大LLM/adapters也可能掩盖核心作用。
5. **Born读出损失**：摒弃直接振幅后，可能再次失去CandidateA最有效的数值自由度。若强classical保持更丰富配置特征，Q没有优势也合理。
6. **NISQ代价**：binaryqubit少但controlled门/measurement多；没有硬件噪声与编译证据就不能宣称实用量子部署。

RFF/去量子化文献[L13,L15,L16]提醒检查函数是否可被廉价经典代理；但不能反过来宣称所有QNN总可被同预算RFF替代。本设计的小空间**精确可模拟性**本身已经足够否定当前计算加速宣传。

## 20. 已证明、已有证据与待验证假设

| 级别 | 结论 |
|---|---|
|本报告可直接推导|H Hermitian→U unitary；合法子空间/全局相位性质；固定context下置换等变；terminal index-only hop对feature trace无效；复杂度计数；Δ与L2误差的关系|
|已执行代数检查|简化实例满足上述数值恒等式；顺序与相干性可以改变读出；不是全原型测试|
|文献证据|子集/图条件量子演化、重上传Fourier结构、特定对称ansatz的理论、量子聚合/多walker构件|
|ICCT证据|邻车/三车关系诊断、RC分层信号但global失败、Raj小数据重复正信号但full近打平、后续隔离结果|
|合理推断|目标相关context与角色保持应更贴合当前任务；固定boundedregister符合资源条件；多时间token可能改善信息传递|
|仍是假设|TRC能更好利用动态joint信息；小数据归纳偏置有用；可测量readout够用；full-data在强classic下仍有稳定ADE/FDE优势；2h训练能达到|
|明确不声称|量子必胜、量子计算加速、普适samplecomplexity优势、自动无barrenplateau、真实硬件已运行、全SinD泛化、量子state等于真实车辆未来分布|

## 21. 最小工程/实验ROADMAP与kill criteria

### R0：原型正确性，不做性能选优

新目录实现TRC、RTCN、toy/reference，不改正式上游。检查mask0/1/2邻居、padding、重复历史tie、20→4时间块对齐、permutation、norm、globalphase、有限梯度、完整binary与有效态的低N等价；H/U/feature同参数精确classic模拟对齐。readout不能读实虚振幅，Hamiltonian最终必须对称。

**立即失败：** equiv/phase/mask误差在complex128参考中>1e−8，未来信息泄漏，真实graph-dependent路径无梯度，或通过经典跨车GNN绕过量子核心。先修实现，不用训练指标掩盖。

### R1：资源与信号通路短profile

B∈{8,16,32}，K8，单卡完整forward/backward，预热后统计step/峰值显存；可先用合成张量，不读test。核对own-only、neighbor扰动、e_jk扰动、timeorder扰动能够改变合法输出。强classical以相同输入一起profile。

**资源失败条件：**按合理融合/checkpoint后仍无法effectiveB32或>2s/step，则退出当前默认规模，最多一次预注册降阶/降轮数修订；不依赖长训练继续选型。接近0.5–0.6s/step才有2h总训练的现实希望。

### R2：最小机制筛选，0dB，train/val

固定4096嵌套train子集，dataset seed与modelseed分离。先比较Q与RTCN：同GRU、同physics、同48tokendecoder；补一个own-only或已有strongprojectreference。短预算仅淘汰明显失败，不宣称最终提升。

之后最必要三种重训练消融：去configuration coherence；去e_jk（保持节点/参数接口）；四块改为仅最后块/时间均值（参数预算尽量匹配）。独立做40/48token的Q/C小factorial，只在资源允许下完整展开。后验高分配置不得偷偷重新定义Primary。

**机制失败条件：**无neighbor变化影响；去相干/去e_jk/去时序在多个seed训练后完全不影响模型且没有表示诊断支持核心被使用，或所有收益都能由相同接口/预处理的classic复制。单个消融不掉点不自动证明无用，但若全部joint/time/coherence主张都无证据，就不能保留该主创新叙事。

### R3：full-data主门槛，先只0dB

Q与最强预选matchedC各3seed（2026/2027/2028），full15802，同epochs/早停/初始化hash、相同LR搜索次数。不得只因经典更大而排除。固定标准训练schedule；另补matched-update学习曲线，将4096/full分别与相同updates对比，判断此前Raj信号是否只是训练曝光。

**建议预注册GO门槛（不是领域公认标准）：**full-data median配对J改善≥1%，至少2/3seed为正，ADE和FDE总体均无>0.5%退步，按时间块的配对置信区间支持J改善；报告200/400帧block敏感性和seed方差。分层highclosing改善只能辅助，不能替代overall。

**默认淘汰：**通过合理等预算优化后full依然近打平/负收益，或只一个seed/事后子集有效。若small优势可重复、full不提升，可留下“有限数据量子可实现归纳偏置”研究结果，但**不继续宣称满足本项目full-data主方法目标**。一次明确的encoding/readout修订后仍失败，转唯一Backup，不无限微调。

### R4：只对通过R3者扩展

其余四SNR做相同Q/C与预定seed；candidate12/16与K8/12作为独立factorial；Token/LoRA不与核心改动混做。已有test的sensing统计不用于预测超参；architecture、checkpoint规则和阈值冻结后才统一评估SinD prediction test。

最终报告overall/各SNR/ADE-FDE时间曲线/highclosing分层、训练墙钟、参数、memory、shots敏感性以及模型失败场景。窗口是相关样本，只有两段公开记录，不能宣称跨城市/完整SinD已证实泛化。

### Backup启用条件

必须有R0/R1/R2证据指出Primary的configuration读取/编译/约束是问题，或者R3公平比较失败；只启用第18节TES定义，不另增加第三备用。TES也按相同global与公平门槛接受淘汰。若两者都失败，科学结论是“当前任务未验证quantum-specific predictive advantage”，而不是削弱baseline或更换test。

## 22. 工程交接：已冻结与必须实测的清单

冻结：Primary/Backup唯一性；四历史块、role42/pair12、Ω1/Ω2；Primary4/6featurequbits、K8默认；structured24nodeangles/12edgeangles；H/V顺序；35维可测readout与8token；sharedRTCN和所有公平权限；当前Motion Token/LoRA先保留；无新正式训练。

必须原型实测：实际parameternumel；complex64梯度数值精度；K8端到端profile；encoding/readout有效信息；不同seed/时间块稳定性；硬件gate分解与shot影响（若论文声称硬件意义）。这些未测项不阻碍选择原型方向，但阻止将该设计描述成已验证最优架构。

后续新对话先读本报告与 `selection_summary.json`，再按需要读 `01_historical_audit.md`、`02_literature_and_candidates.md`、`algebra_sanity.json`、`complexity_tables.json`、`source_manifest.json`。不得从旧freeze标题误以为本设计已训练或进入正式分支模型。

---

## 附录A：文献与候选机制研究（自包含副本）

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

## 附录B：服务器证据定位

- 系统：`docs/dataset_migration/SIND_MIGRATION_HANDOFF.md`；`frontend/sind_prediction_dataset.py`；`prediction/q0/motion_token_llm.py`；`prediction/qgnn_final/common.py`与`model.py`；`configs/qgnn_final_tokens.json`。
- SinD基线诊断：`docs/qgnn/SIND_Q0_INTERACTION_DIAGNOSTIC_20260919.md`；`reports/q0/sind_q0_interaction_diagnostic_0db_seed2026.json`。
- 早期：`prediction/quantum.py`；`experiments/qgat_factorial/graph.py`；`experiments/qgat_value/graph.py`；`reports/qgat_formal/summary.json`；`reports/qgat_factorial_full/seed2026/summary.json`；`reports/qgnn_bottleneck/bdx01_20260914_01/summary.json`。
- RC：`docs/qgnn/FINAL_QGNN_ARCHITECTURE_FREEZE_20260919.md`；`prediction/qgnn_final/quantum.py`、`relational.py`、`classical.py`。
- Adaptive：`docs/qgnn/ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md`；`ROUND3_SELECTED_FEEDBACK_AUDIT_20260919.md`；`ROUND3_FEEDBACK_RETRAINING_FINDINGS_20260919.md`；`HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md`。
- Raj主要源码：`prediction/qgnn_paper_native/raj_subset.py`、`raj_paper.py`、`raj_paper_math.py`、`raj_mechanism.py`、`model.py`；trainer `scripts/train_qgnn_paper_native.py`。
- CandidateA：`docs/qgnn/QGNN_SELECTION_CANDIDATE_A_RAJ_FREEZE_20260919.md`；`reports/qgnn/ROUND4_RAJ_FULLTRAIN_RESULTS_20260919.json`；`reports/qgnn/round4_raj_fulltrain_{quantum,johnson}_0db_seed2026/summary.json`；`round4_raj_weighted_multij_{quantum,johnson}_20e_{2026,seed2027}/summary.json`（路径中花括号表示对应多个目录，不是实际文件名）。
- fidelity：`docs/qgnn/ROUND4_RAJ_PAPER_FIDELITY_AUDIT_20260919.md`；`ROUND4_RAJ_INTERMEDIATE_WEIGHTED_MATCH_FINDINGS_20260919.md`；`ROUND4_RAJ_FEATURE_BUILDER_ISOLATION_FINDINGS_20260919.md`；`ROUND4_RAJ_ROWLOCAL_REUPLOAD_FINDINGS_20260919.md`；`reports/qgnn/round4_raj_paper_math_expadj_quantum_4096_seed2026/summary.json`。
- CandidateB：`docs/qgnn/QGNN_SELECTION_CANDIDATE_B_SQM_GNN_INITIAL_AUDIT_20260919.md`；`SQM_GNN_UMSG_SOURCE_BOUNDARY_20260919.md`。

精确SHA256以source_manifest为准。未将仅存在工作区的新结果伪称已经GitHub发布；本轮研究文件未自动commit/push。
