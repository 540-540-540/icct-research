# ICCT QGNN Architecture Selection Report — Server Handoff

版本：THEORETICAL_SELECTION_V1_20260920。状态：理论选型完成；新架构未实施、未训练、未取得新ADE/FDE。
项目：/home/dell/YrM/ICCT，分支qgnn，核验HEAD 094def5b38fa21ccc09cda2328fe78ee8cc694c1。服务器优先于GitHub。本文件是自包含的服务器交接版；长版报告随本轮对话交付，二者的结构、参数和决策一致。

## 1. 最终选择与研究边界

Primary：**TO-JQGNN — Time-Ordered Joint-register Quantum Graph Neural Network，时间有序联合寄存器量子图网络。**
唯一Backup：**T-SQW-GNN — Temporal Subset Quantum Walk GNN，时间子集量子游走图网络。**

主线不是堆叠量子模块，而是：把目标与邻车的多段历史，按观察顺序加载到同一量子联合态，令目标—邻车和邻车—邻车关系通过非对易图演化共同作用，再把可测的多时间前缀表示交给GPT-2。
这是一项待验证的任务归纳偏置选择，不是已证明量子优越，也不宣称该组合在所有文献中从未出现。Backup保留Raj的subset机制，但改为可制备、可测的量子链路，不继承旧A成绩。

冻结SinD、当前3-BS controlled sensing、20帧感知[x_hat,y_hat,vx_hat,vy_hat]、未来20帧、GPT-2、Motion Token和LoRA。不改正式代码、配置、数据、ISAC、历史结果；本轮只有文档和短代数核算，无新训练。

## 2. 当前真正的interaction bottleneck

现有系统把交互模块输出的每车64维向量送入逐车GPT-2：1graph+19motion+20future query，总40token。GPT-2不在同一序列直接看全车；g既通过GPT也直接进入coordinate head。预测为p_last+k*dt*v_last+16*tanh(raw/4)，不是逐个生成离散token再积分。[P1,P2]

SinD为长春/西安公开四轮车子集，train15802、val1880、test2086；约10Hz，历史20点跨度1.90s、预测2.00s。90.73%窗口从大于8辆的候选中围绕一个focal裁剪，却评价图内所有目标，非focal可能缺关键邻车。原离线选图基于干净GT历史；未来routing必须基于感知历史。完整40帧轨迹可用性是标签条件，不等于用未来坐标排序。[P1,P3]

现有3-BS不包含真实检测/身份关联失败/时间tracker。不要把QGNN定位成修漏检或关联。位置差分含相邻帧误差，单车去噪和跨车交互收益要分账。

Q0单seed0dB、旧临时token：NoGraph ADE/FDE .524150/1.063804；pair .502264/1.030864；pair+triplet .498218/1.004633。Triplet比pair额外改善约.81%/2.54%，动态closing分层更明显，密度本身不稳定。[P4]

因此需要同时区分：context覆盖、时序/物理表示、真正多邻车联合计算、预测端利用。尚无证据经典已达上限；密集小图不支持直接把长距离传播/1-WL限制当主因。

## 3. 历史机制审计与Raj的关键解释

早期QGNN已用2qubit/node和ZX物理边，双qubit不是新贡献。QGAT的索引/query-key/量子value/postselection来自旧协议；旧seed2023 QGAT J1.377805、GNN1.221526，不能与SinD跨任务排行。独立edge PQC后经典softmax/aggregation，不自动构成量子主导的联合推理。[P5]

RC-HQGNN：C4独立Q8通道、D3、ZZ/ZZZ和相关性读出。Full15802 Q J1.073970，adaptive classical1.031109，整体落后；高closing小分层有收益。Scene-adaptive feedback重训Q-on/off J1.221929/1.229810，C-on/off1.191771/1.191865，反馈Q收益约.64%仍不超过C。删门变差只证明依赖，不证明优于经典；K3非零不是纠缠证据。[P6,P7]

Weighted Multi-j Raj A：j2/j3子集、GRU32和98维subset统计、20维复数embedding、物理weighted Johnson exp(iH)、compound SO(6)更新、add/row-normalize、JK Re/Im和incidence readout。

| 条件 | Q ADE/FDE/J | C ADE/FDE/J |
|---|---|---|
|4096 seed2026|.516458/1.102781/1.067849|.525952/1.129782/1.090843|
|4096 seed2027|.530895/1.148234/1.105012|.550423/1.189891/1.145368|
|15802 seed2026|.481259/1.051173/1.006845|.482314/1.049728/1.007179|

4096两seed均值约+2.70%ADE/+2.96%FDE/+2.83%J；full约+.219%ADE/-.138%FDE/+.033%J，接近打平。[P8]

不能直接解释为量子样本效率：4096、B32、20epoch只有2560updates，full有9880；full最佳epoch9已经4446。seed还改变所抽子集。Q用physical weighted H，当前Multi-j Johnson实际固定Johnson邻接，功能未完全匹配。Full epoch5约2470updates时C J1.035046、Q1.042177；因此也不能仅说C靠更多steps追上。数据覆盖、LR轨迹、约束/常规正则化与优化暴露共同混杂。Full后期train loss继续下降而val不改善，是泛化限制迹象，不是decoder/Bayes ceiling证明。

量子物理边界：内部exp(iH)和compound可以是真幺正，但整条add/逐行归一化/直接ReIm数值链路不是已说明成本的确定性量子线路。ReIm依赖全局相位；未知态加任意向量再归一化不可能一般由固定幺正实现。官方CFI代码也存在这些操作，不代表整个paper都无效。若用LCU、干涉测量或重新制备补齐，应明确其成本。[P9,L3]

Raj paper-fidelity同时改变loader、joint mixing、RDM读出；不能把负结果归到一处。新exp-adjacency已经COMPLETED：ADE .783533250548/FDE1.619350526672/J1.593208513884，仍远弱于A。原文fermionic 1-RDM不是两qubit完整RDM。[P10]

SQM仅来源初审，无ICCT训练证据。原文U_MSG先作用neighbor-edge，center不动，之后U_UPD汇入center。共享参数不能自动消除顺序非对易U_UPD的排列问题；未核验可直接照抄的完整官方门表。[P11,L4]

Probe：readout input R2约.9216、compressed interaction64 .8245、final local+Q64 .9187。它预测冻结经典residual，不是未来GT；不能据此断言64维就是唯一瓶颈。[P12]

## 4. 外部机制与六类候选的淘汰

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

## 5. Primary：车辆数、输入权限和selector

区分N_scene、候选C、评价目标T、register车辆M。默认T保持原有效目标，初轮原cache C<=8；后续C12为候选设计、C16覆盖消融；M<=6（root+K<=5），Q<=12。

后续额外候选取同split、当前时刻已观察且有完整20历史的四轮车；60m内最近C-1（最近5帧平均距离）。候选不要求未来20帧，只有评价目标需要标签。若cache缺候选历史，另建context-only cache，用同冻结ISAC，不改旧数据。

每5帧一个段s=1..4。风险
w_jk,s=exp(-d/30)*[.25+.75*sigmoid(c/2)]*exp(-DCPA/10)。
R_ij=.5 max_s(w_ij,s)+.5 w_ij,4，取top5。tie用距离和完整观测历史字典序，不用ID/slot；完全相同历史视为不可区分输入。不复制目标充padding。

训练集coverage记录sum_selected R/sum_available R和active-neighbor recall。若超过10%目标保留不足80%候选风险质量，M6充分性假设失败；先分清routing限制。此阈值为提出的门禁，非已测结果。

## 6. Primary：temporal、node、edge、higher-order representation

保留现有单车encoder结构：Linear6->32、SiLU、GRU32、LN32，6624参数。对每车
q_j,t=[(p_hat_j,t-p_hat_j,20)/10, v_hat_j,t/10, p_hat_j,20/50]，6维；保存GRU t=5/10/15/20的LN hidden h_j,s（32维）。没有经典跨车操作。最后观察位置作锚，未用future，但前缀h不宣称在线t5即可获得。

root条件node：x_j|i,s=[h_j,s,(mean p_j-mean p_i)/30,(mean v_j-mean v_i)/10,||mean v_j||/10,1[j=i]]，38维，shape[B,T,4,M,38]。

所有有效非自身pair包含neighbor-neighbor，不止root-star。r_jk=mean p_k-mean p_j，u_jk=mean v_k-mean v_j；a_j为5帧速度对真实timestamps的LS斜率。
d=sqrt(||r||²+1e-6)，c=-r·u/d，tau=clip(-r·u/(||u||²+1e-4),0,4)，DCPA=sqrt(||r+tau*u||²+1e-6)，alignment=v_j·v_k/(||v_j||||v_k||+.04)。
e+=[d/30,c/10,tau/4,DCPA/10,||u||/10,alignment]（交换不变）；
e-=[r_x/30,r_y/30,u_x/10,u_y/10,(a_k-a_j)_x/5,(a_k-a_j)_y/5]（交换反号）。
edge shape[B,T,4,M,M,12]。尺度是工程初值，不用test调。加速度可能放大噪声，Q/C共享信息并另做消融。

三车信息xi_ijk=(x_i,x_j,x_k,e_ij,e_ik,e_jk)分别进入量子态，不先经强经典triplet MLP压为单角。

## 7. Primary：初态和多bank encoding

每node两角色qubit a_j,b_j，有效|++>，padding|00>并使所有相关门identity。psi0为有效node product态。
局部Pauli顺序P=(Za,Ya,Xa,Zb,Yb,Xb,XaXb,YaYb,ZaZb,XaZb)。38维补2个identity位置，分4个10维bank。
真实第d维alpha_d=(pi/2)tanh(g_d*x_d+b_d)，g_d=tanh(raw_g_d)，初始化g约.25、bias0；补零位置不配bias。
V_s,r为逐node、bank内按该Pauli顺序的exp(-i alpha P/2)乘积。不同node对易，同node顺序共享。
W_r=product_j exp(-i eta_r Ya_j Xb_j/2)，eta=.2tanh(raw_eta)，初始化约.05。

不是38->一个angle，但也不是高维信息无损编码；两qubit的读出仍有限。数据跨bank上传并与图演化交替，避免未经验证的前置scalar压缩。

## 8. Primary：完整graph-dependent unitary

偶特征beta_d=(pi/2)tanh(g+_d e+_d+b+_d)；奇特征gamma_d=(pi/2)tanh(g-_d e-_d)，无bias保反号。gain同样有界，初始化约.25。每bank取两维，固定索引序列I+=(1,2,3,4,5,6,2,4)，I-=(1,2,3,4,5,6,3,4)。
设kappa_jk=w_jk/5；同role的JZ=kappa*beta_(I+[2r-1])、JX=kappa*beta_(I+[2r])；有向JD1=kappa*gamma_(I-[2r-1])、JD2=kappa*gamma_(I-[2r])，按e_jk定向（指向约定全实现一致即可，经典得到同信息）。

H_Z=sum_{j<k} JZ_jk Za_j Za_k；H_X=sum_{j<k} JX_jk Xa_j Xa_k；
H_D1=sum_{j!=k} JD1_jk Za_j Xb_k；H_D2=sum_{j!=k} JD2_jk Xa_j Zb_k。

各族内部所有项对易：D1始终a上Z、b上X；D2始终a上X、b上Z。族间一般不对易。每bank执行顺序：V、W、exp(-iH_Z/2)、exp(-iH_X/2)、exp(-iH_D1/2)、exp(-iH_D2/2)。四bank按1..4，四历史段按1..4，**同一态持续演化，不在段间reset或经典反馈**。

总量子参数98=38*2 + 6*2 + 6 + 4；跨node、target、时间共享。不同时间输入导致不同unitary，参数不需要按帧独立。

节点置换成块交换(a,b)，同时交换所有输入/边；共享参数、族内对易与偶/奇对称保证U(piX,piE)=P_pi U(X,E) P_pi†。padding全identity保证不依赖填充数。保留绝对坐标，因此不宣称世界旋转/平移等变。有限Trotter跨不同族的标签任意序不在此保证内。

## 9. Joint interaction和时间顺序的严格示例

A=Z0 Z1，B=X1 X2，则[A,B]=2i Z0 Y1 X2。两体图算子的非对易乘积可产生三体混合，root observable的Heisenberg反向传播可包括邻车间边。
在初态|Y+>0|Y+>1|X+>2、观测X0、旋转exp(-i theta P/2)下：先A后B的读出0；先B后A为-sin(a)sin(b)。a=.4,b=.7为-.250870183850，与小矩阵解析计算一致。

这证明该机制有能力区分某些顺序并产生非加性，而非证明SinD的真实未来需要该项。经典微小乘积模型就能计算这个例子；强tensor/attention不受pair-additive限制。必须要求收益超过这种强控制。

## 10. Measurement、RDM、Q输出

每个s prefix：root两qubit RDM rho_i,s，读取15个非identity Pauli moment m_i,s；rho=(I+sum m P)/4。这是两qubit完整RDM，不是fermionic 1-RDM。

每root-neighbor保留8个connected correlator：同a-role XX/YY/ZZ三项，同b-role XX/YY/ZZ三项，Za_i Xb_j和Xa_i Zb_j两项，均减相应单点均值乘积。c shape[B,T,4,K,8]。
z_i,s=[m15,mean_valid c8,sum_j w_ij c_ij/(sum_j w_ij+1e-6)]，31维，shape[B,T,4,31]；无邻居时后16置0。pool为确定性，不是强经典关系value网络。

解析模拟可沿一次trajectory记录4prefix；硬件必须每个prefix重新制备。全体a用轴A、全体b用轴B，A,B∈XYZ，共9settings覆盖全部上述moments；connected统计须处理finite-shot偏差。不能免费读取state ReIm或测早期后继续同一个无扰动态。

## 11. QGNN-to-LLM、Motion Token与LoRA

q_i,s=W_Q LN31(z_i,s)+b_Q+learned_time_s，768维，投影跨s共享；另q_own=W_o h_i,4+b_o。
GPT输入：own1 + interaction4 + 原19motion + 20future queries =44 tokens/target。保留原history adapter，但不再重复添加单一graph到每个motion。每个future query可读取不同prefix，无需20次quantum circuit。

新coordinate head默认LN770->256->GELU->2，只读future hidden768+expected motion2，不给原g直通旁路。最终p=p_last+k dt v_last+16*tanh(F/4)，F最后层zero-init，保留CV初始点。此去旁路是待检验假设，可能降低性能；必须core×reader公平对照，若双方同幅退化，恢复共同旁路而不是Q独享。

不改41×41非均匀motion码本、4corner双线性embedding、continuous delta/overflow adapter；codebook步长不是连续输出精度下界。保留GPT-2前4blocks768与QKV LoRA r8 alpha16。只有共同reader证据支持，才同预算比较其他LoRA位置。loss仍ADE+.5FDE+.035tokenCE。

## 12. Matched classical：Temporal Rooted Tensor GNN (TR-TGN)

同context、38node、12edge、共享单车encoder、四时间段、每段四round、同44token/head。
u_init=W_x x+b，32维；p_init=W_e[u_init_j,u_init_k,e12]+b，32维。新时间投影与carry state相加再用同LN；首段直接初始化。

每round共享：t_jk=weighted_mean_{l!=j,k}[(A p_jl+a) elementwise (B p_lk+b)]，16维，weight=w_jl*w_lk；
p'_jk=LN[p_jk+MLP([p32,u_j32,u_k32,e12,t16])], MLP124->64->32 SiLU；
m_j=weighted_mean_k(C p'_jk+c)，C32->32；u'_j=LN[GRUCell32(m_j,u_j)]。

输出root linear32->15、pair linear32->8，使用同均值/risk pooling和4prefix接口。经典不需被强制成合法密度矩阵。width32时interaction23127参数，单车encoder6624另计；width64作为相同调参预算内加强版。
保留GatedPairTriplet/AdaptiveClassical/project reference；Raj对照必须真正physical-weighted Multi-j Johnson。还需同输入的orthogonal/complex temporal dynamics或低秩Fourier/tensor surrogate，检验收益是否仅来自范数/频谱/时间偏置。把复数态改为两个实数组只是同一函数，不是假想独立学习baseline。

两层公平性：机制归因使用与量子相同M6的TR-TGN；项目竞争力必须额外保留同一候选池全部C8/C12上下文的强经典版本及原项目参考，不能因为量子register小而强行裁掉经典邻车。最终总体优势门禁以validation预先选定的最强合规经典为准，不只挑最易超过的M6模型。

## 13. Backup T-SQW-GNN：完整理论定义

K<=7邻车，中心作为classical条件，K occupancy qubits+2feature qubits，Q<=9；两分支r1和r2分别表示rooted pair/triplet。
每个r子集S使用f114=[h_i32,mean h_S32,std h_S32,mean e+_iS6,mean e-_iS6,mean内边e+6]。r1的std/内边明确0，std用population版本；不是strong MLP推理，但仍有统计压缩风险。

初态normalized Dicke(K,r)⊗|++>；只用真实有效K，不含padding。每时间段12bank，每bank10维，114补6identity，Pauli顺序同Primary。
V_s,b=sum_S |S><S|⊗product_{d in bank} exp(-i alpha_d(f_S) P_d/2)，alpha=(pi/2)tanh(g_d f_d+b_d)，是controlled unitary，不是amplitude add/row-normalize。
H_J,s=sum_{a<b} (w_ab,s/7)(XaXb+YaYb)/2，在r-subset间single swap且保持r。
每bank U= [exp(-i tau_r,b H_J)⊗I] [I⊗W_b] V_s,b；tau=.5tanh(raw)，初值.1；W_b=exp(-i eta_b Ya Xb/2)，eta=.2tanh(raw)，初值.05。四段连续演化，12bank/段。
参数264=114*2 + 12 onsite + 2*12 walk，全部跨时间共享；r共用encoder但walk不同。

测p_S=<PiS⊗I>，nu_S,alpha=<PiS⊗P_alpha>（未除pS）；omega_iS=mean_j∈S w_ij；z_r=[sum nu15,sum omega nu15,sum omega p]31。两分支concat62/段->LN62 linear768->4interaction tokens，仍44总token。无有效subset时对应branch输出0。
这些joint moments可在occupancy Z和feature9settings测得，无逐subset后选择。不能宣称继承原A的ReIm JK性能。

等变来自Dicke、对称subset特征和Hamiltonian总和的置换协变。精确reduced-basis指数实现此公式；XY边一般不对易，硬件Trotter按标签序列不能自动保持精确等变，应显式报告模拟误差/置换误差。

控制为同114features、同r1/r2、同physical weighted Johnson、同temporal/reader的强classical subset模型；固定r可经典多项式模拟，不宣称计算加速。Backup只在Primary失败后启动，不作为平行无限搜索。

## 14. 资源：Q、gate、calls、memory、参数

Primary M6，E15，S4，R4：旋转排程上界SR(11M+6E)=2496；其中1920 two-qubit Pauli旋转（naive3840CNOT，不含routing）。48个零角identity可去除，非平凡位置2448/1872。门数不是深度；完全连接下族内可边着色，逻辑深度量级O(SRM)，硬件SWAP另计。
解析算量O(BT SR(M+E)2^(2M))，readout另计；全候选pair O(N_scene²)，selector排序另计，不是整条pipeline O(N)。

complex64 Q12=4096幅值=32KiB单态。B32：
|目标T|trajectory/scene|/batch|裸batch state|2496份逐门保存上界|
|8|8|256|8MiB|19.5GiB|
|12|12|384|12MiB|29.25GiB|
|16|16|512|16MiB|39GiB|

16bank边界checkpoint分别128/192/256MiB；一段156门recompute若逐门存储约1.22/1.83/2.44GiB。还需GPT激活/optimizer/workspace，因此bank checkpoint或adjoint是前置要求。仅forward chunk、最后统一backward不一定省总图。B32可用16x2累积，双方优化语义一致。两张24GB不等于一张48GB；默认独立模型/seed分卡。
扩大候选C但T保持8，仍是第一行。反例全部N车2qubit：N8/12/16时单态.5MiB/128MiB/32GiB，N16已不适合单4090。

Backup K7：名义Q9；reduced d1=4*C(7,1)=28，d2=84，总112complex=896bytes/目标；B32,T8一份两分支态.21875MiB。Johnson edges21/105。固定r规模多项式，dense exp按O(C(K,r)^3)，应用O(C(K,r)^2*4)，不同bank可复用同stage H的结构。
2T模拟trajectory/scene。每branch4*12*10=480个条件旋转位置，通用uniformly-controlled分解每位置可随2^K增长，K7约480*128级分解因子；Dicke、XY模拟、shots另计。GPU reduced便宜不等于硬件浅电路。

参数推导（非新模型numel实测）：
|组|Primary|Backup|TR-TGN32|
|量子core|98|264|0|
|经典interaction|0|0|23127|
|单车encoder|6624|6624|6624|
|LLM端非LoRA训练含adapter/head|4038869|4062739|4038869|
|LoRA|98304|98304|98304|
|总可训练|4143895|4167931|4166924|
Primary interaction adapter27710、own25344；Backup interaction adapter51580。冻结GPT约67.74M。不能只数98参数给整条4M系统套少数据泛化定理。

## 15. 时间与硬件采样费用

旧Raj full20epoch Q1121.75s、C1204.08s，非新方案profile。15802/B32/20epoch=9880steps。
|假设step|纯训练min|加20%规划余量min|
|.10s|16.5|19.8|
|.25s|41.2|49.4|
|.50s|82.3|98.8|
|1.00s|164.7|197.6|
以上为假设场景，不是预告实际时间；两小时理想预算在20%余量下约step<=.607s。Primary需真实GPU profile，Backup存储更轻但不承诺吞吐。两小时不作绝对科研门槛。

解析Primary T、Backup2T轨迹；物理测所有前缀则Primary4*9*T*shots、Backup2*4*9*T*shots，T8时288/576个shot单位。单个±1观测标准误差<=1/sqrt(shots)，联合置信区间/connected估计额外考虑。parameter-shift按门出现次数计，不是共享98参数x2就全部完成；encoder数据角的反传也要计费。没有硬件训练加速主张。

## 16. 优势论证链与最强classical反驳

SinD动态closing的triplet信号 -> 目标条件的时序联合影响假设 -> 静态有限预算pair聚合的偏置可能不足 -> 同一量子态的时间有序非对易图演化 -> 局部读出可含非加性、邻车间和顺序项 -> 同信息/同reader下full-data ADE/FDE可测假设。

限制：强temporal tensor/attention天然可处理同类作用；本例-sin a sin b经典很容易；Q12可完整模拟，Backup固定r多项式。量子态复杂不等于少数低阶observable难以被经典surrogate学习。频率数量多不等于频谱系数自由或有效容量。旧A已提示full-data classical追赶。输入缺失/噪声/道路信号灯/reader可能限制所有方法，不可据此为量子倒推任务。[L12-L16]

有界旋转共享参数和范数保持可能提供不同偏置，也可能不适合带噪连续任务。没有无barren plateau定理、无lossless编码定理、无普适泛化或计算优势。幺正范数保持不是对角噪声不敏感：||psi(theta)-psi(theta')||<=.5 sum_g|delta theta_g|；observable差<=||O|| sum_g|delta theta_g|，最坏界会随门数累积。小角、局部readout、Q上限是工程措施，不是性能证明。

## 17. 已证实/推导/文献/假设分层

项目事实：Q0邻车价值、Raj两seed小样本正/full近打平、最新expadj完成、source实现ReIm/add-renorm。
本文推导：各族内部对易、共享参数下的block等变、norm/padding数学定义、可测非加性顺序例子、资源算术。
文献支持：合法图条件unitary、subset机制、理论的条件和反例；没有现成SinD同GPT2优势证明。
待验证：完整实现正确性、真实梯度和GPU峰值、时序是否主要剩余瓶颈、M6/C12充分性、四tokens收益、full-data相对强经典的ADE/FDE。
短核算只用NumPy小矩阵，不导入项目/数据、不训练。commutator误差0；顺序值0与-.25087018385；全局相位不变Born、改变ReIm；归一化加法将正交内积变1/sqrt2。不能把这些当完整原型unit test。

## 18. 下一轮最小ROADMAP

G0：新建隔离实验模块，不覆盖历史代码。shape/mask/padding/置换/norm/float64参考/float32GPU/有限梯度/前缀readout；用小三车例子确认root对neighbor-neighbor和顺序真有响应。若小gain*w/5使信号落到数值噪声，训练前记录一次初始化校准。真实B8/16/32和bank-checkpoint profile。

G1：同旧C8、同目标、固定4096子集、val1880、有效B32、2560updates，两个模型seed。Primary vs TR-TGN主对，NoGraph参考；共享encoder/reader初始化哈希、数据顺序一致。两侧同两个LR候选1e-4/3e-4，LoRA1/4。训练subset seed和模型seed分开。core×旧/新reader的2x2检查：四段31维固定拼接投影64接旧reader；Q/C同样处理。不将token数、旁路、重复添加一起改变后的效果归因于单一因素，发现interface主导再拆消融。

G2：同reader重训time-reset、同四段平均graph、只留root-star等消融；经典做对应控制。平均graph改变信息，不单独证明非对易；checkpoint顺序重排也只是敏感性，因为GRU前缀本身带时间。最终归因需强temporal tensor/orthogonal同信息控制。

G3：通过前门后才做nested4096/8192/15802、统一2560/4446updates，LR按update；先4096与15802端点，必要才加8192；模型seed2026/2027/2028独立于数据seed。strict weighted Johnson。分开同epoch/同steps问题，小数据优势若仅有限预算成立就限缩声明，不伪装full胜利。

G4：先train coverage再C8/12/16（评价目标不变），context-only cache新命名且同ISAC。full15802三seed0dB后再冻结五SNR；只在结构/阈值/模型选择冻结后打开最终test。报告ADE/FDE/J、所有seed、wall/显存/calls/失败run。

统计：连续stride1重叠窗口非独立样本。paired差值按连续时间block重采样，至少40帧支持并检查train相关性，建议起始10s且20s敏感性。仅两个recording不能支持跨城市总体结论；不能把1880窗口当1880次独立试验。

## 19. Kill criteria（本次提出，训练前预注册）

硬停：future/GT/ID违规、norm/mask/permutation错误、意图机制对root observable无响应。float64参考误差门禁1e-8、norm1e-10；float32初值1e-5并按累积数值误差预设，不事后任意放宽。
Context充分性失败：>10%目标保留<80%可用风险质量，先报告选择限制。
Full主收益门禁：三seed均值J改善至少1%、FDE至少2%、ADE退化不超过.5%；希望高closing FDE>=3%作额外机制支持，不可代替总体结果。普通组J退化>1%或收益被单seed主导、paired block CI含0，不能宣告稳定总体胜利。
强tensor/orthogonal/surrogate消除量子增益，则quantum-specific解释失败，即便超过弱MPNN。
checkpoint合理microbatch后仍超24GB或长期超预算而无科学收益，停Primary资源路线，启唯一Backup；两小时本身非硬否定。
只允许预先登记的一次实现/初始化校准、同预算LR选择。两条都失败后保留负结果与强经典，不无限加门/加controller/换名。

## 20. 交接与文件

本目录此前已保存01_historical_audit_source.md、02_quantum_validity_source.md、historical_raw_summary_snapshot.json、access_manifest.json，历史来源可继续追查。本轮最终报告就是本文件；selection_summary.json给机器可读关键决定，NEXT_SESSION.md给开始顺序。没有commit/push，没有正式模型实现或新训练成绩。
后续对话先读本报告与JSON、核验HEAD/保护hash/GPU状态，从G0工程开始，不重新审计SinD/ISAC，不回退远端旧状态。

## 21. 来源索引（自包含）

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
