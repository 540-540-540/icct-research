# 量子机制合法性、反例与成本边界

版本20260920_v1。以下命题是本轮分析推导，不是被引用论文的原有定理；适用条件逐项列出。没有运行量子模型或新增训练。
目标不是要求小规模GPU可模拟QGNN必须“不可经典模拟”，而是区分：真实量子线路、量子启发的经典计算函数、有限预算下的预测收益、以及渐近量子计算优势。

## 1. Observable读出与原始振幅不是同一对象

对任意Hermitian observable O：
\[
f_O(\psi)=\langle\psi|O|\psi\rangle
=f_O(e^{i\varphi}\psi).
\]
因此普通测量统计对全局相位不变。直接读出的向量
\[
r(\psi)=[\operatorname{Re}\psi,\operatorname{Im}\psi]
\]
却一般随全局相位变化。若模型把它交给无约束MLP，就不能在不说明访问模型的情况下等同于测量输出。
这不等于振幅在所有模型下都无法估计：已知制备电路、有参考分支及controlled-U访问时，可以设计干涉式估计；也可以研究层析。必须报告参考相位、制备/控制门、精度与重复次数，不能把模拟器读取数组的成本算成硬件readout。
ICCT对应：raj_subset.py:64-66，直接拼接z.real/z.imag。作者固定版本cfi.py也有此读出；这里只审查这些实现，不否定其他observable实现。

## 2. 加法重归一化并非一般的确定性unitary

考虑固定f=|0>及映射T_f(z)=(z+f)/||z+f||。两个正交输入给出：
\[
T_f(|0\rangle)=|0\rangle,\qquad
T_f(|1\rangle)=(|0\rangle+|1\rangle)/\sqrt2.
\]
输入内积为0，输出内积为1/sqrt(2)。unitary保持内积，因此不存在一个相同unitary对所有这些未知输入实现T_f。
结论范围：排除的是“把未知保留态作确定性加法归一化=普通量子门”这一解释。它不排除已知经典数据重新制备状态、辅助比特/后选择等不同访问模型。按每个subset行独立归一化也不能直接冒充对整个寄存器的unitary。
ICCT Candidate A的x+=encoder(feat)、row-normalize需要此说明；完整paper-math的unitary controlled re-upload应独立审查，不与该操作混称。

## 3. 共享参数不足以证明邻车置换不变性

若共享更新门依次作用在中心0与邻车j：
\[
U_j=\exp(-i\theta G_{0j}),
\]
交换两个邻车的执行顺序，需要证明U_1 U_2与U_2 U_1在所需输入/读出上等价；共享theta本身不保证该结论。
取完全相同的局部规则G_0j=X_0 X_j+Z_0 Z_j，则
\[
[G_{01},G_{02}]=2iY_0(Z_1X_2-X_1Z_2)\ne0.
\]
这提供一般反例，而不是断言SQM作者任何未公开具体门实现必然失败。
可行的证明路线包括：同一层所有更新相互交换；用等变Hamiltonian总和的指数；使用严格对称的通道；或明确采用以物理特征定义的规范排序并处理ties和不连续性。按任意槽位顺序执行非交换门再声称等变不可接受。

### 等变Hamiltonian的充分条件
令P_pi对应车辆置换，若H(pi x)=P_pi H(x) P_pi^dagger，则
\[
 e^{-itH(\pi x)}=P_\pi e^{-itH(x)}P_\pi^\dagger.
\]
初态和readout也按相同规则变换即可得到所需等变性。这个证明不能自动迁移到标签排序的非交换Trotter门序；要检查实际实现而不是只检查连续时间公式。

## 4. 三体相关非零不等于存在纠缠

取完全可分的经典混合态
\[
\rho=\tfrac34|000\rangle\langle000|+\tfrac14|111\rangle\langle111|.
\]
有<Z_i>=1/2，<Z_i Z_j>=1，<Z_1 Z_2 Z_3>=1/2。因此二阶cumulant为3/4；三阶cumulant为
\[
\kappa_{123}=\tfrac12-3\tfrac12+2(\tfrac12)^3=-\tfrac34\ne0.
\]
该态没有纠缠却有非零二/三体Z cumulant。RC中的这些统计量可作为交互特征，但不能单独作为量子资源见证。即使线路确实产生纠缠，还需要证明或检验它对目标任务有用。

## 5. 一类“高阶量子聚合”存在O(K)经典乘积形式

条件：中心初态|+>，邻居初态为product rho_1⊗...⊗rho_K，单层星形ZZ演化
\[
U=\exp[-i Z_0\sum_j\theta_j Z_j].
\]
中心的非对角元为
\[
(\rho_0')_{01}=\tfrac12\prod_j\operatorname{Tr}(\rho_j e^{-2i\theta_jZ_j}).
\]
令m_j=Tr(rho_j Z_j)，则
\[
\langle X_0\rangle
=\operatorname{Re}\prod_j[\cos(2\theta_j)-i m_j\sin(2\theta_j)].
\]
展开包含多个邻车的乘积项，确实不是简单加法；但给定m_j/theta_j，经典O(K)乘积即可精确计算。故“输出包含高阶项”不自动提供相对classical product/Fourier aggregation的优势。
此外，Z_0与U交换，仅测Z_0看不到该相位信息。若附加的邻车-邻车门也全部为同层ZZ，它们与中心星形ZZ交换，且部分迹消去其纯邻车unitary；因此本例中它们不会改变中心读出。
设计启示：需要检查实际读出的因果锥、测量基，以及层间非交换旋转如何让邻车-邻车关系影响目标。不能仅统计用了多少条ZZ或ZZZ门。
这只是上述单层/product-input结构的精确化简，不推广为所有多层QGNN都O(K)可模拟。

## 6. Raj复数动力学的精确实数反事实

对实对称H，设exp(iH)=C+iS，C=cos(H)、S=sin(H)。若z=a+ib，则
\[
\begin{bmatrix}\operatorname{Re}(Uz)\\\operatorname{Im}(Uz)\end{bmatrix}
=\begin{bmatrix}C&-S\\S&C\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}.
\]
Candidate A的复数混合、实compound演化、行归一化和实虚读出均可表为这样的实数计算图。这是同一函数的等价表达，不是本轮新的实验baseline。
固定j时S_j=C(N,j)=O(N^j)，embedding维数20固定，研究的reduced-basis函数具有多项式规模。其潜在收益应先定位为结构性归纳偏置/参数化差异，而不是由qubit标称数推断指数计算加速。
严谨对照可同时包含：同信息weighted Johnson/高阶网络、实数正交动力学、sin/cos特征网络。等价实数实现的存在不阻止该函数在有限预算下胜过某一MLP，但限制了“量子独有函数”的说法。

## 7. 信息链路：缺失上下文与压缩不能靠后端补回

若同一目标历史H_i下，两种邻车场景被映成相同g_i，则确定性后端D(H_i,g_i)给出同一预测。该观察说明量子readout和接口必须保留任务相关差异；它不证明64维必然不够。
新的interface比较必须让双方使用相同输出token数/维度和相同GPT-2可训练预算。先扩大量子接口而固定经典接口，不构成公平机制比较。
目标若要研究“邻车j对i的影响取决于k”，可构造模型内的交互差分诊断：F(H_i,j,k)-F(H_i,j)-F(H_i,k)+F(H_i)。非零表示函数非加性，但仍不证明量子专有，也不直接证明真实交通中的因果交互。

## 8. 资源核算

### 8.1 区分三个规模
N_scene=场景候选车辆总数；K=每目标保留的邻车数；Q=单次量子寄存器qubit数。增大N_scene不必增大Q，但会增加routing、目标patch或采样覆盖成本。
full global circuit若每车q qubits：Q=q*N，statevector指数增长。
SQM原始标量node+edge寄存器：Q=1+2K；K=7时Q=15。不能把80维单车历史直接按scalar=qubit映射后还沿用Q=15，必须说明temporal压缩或重上传。
固定激发数subset：普通完整寄存器Q=N+D；仅在结构严格保持子空间时用C(N,j)*C(D,k)等reduced表示。若joint mixer只保持总激发数，需计C(N+D,j+k)，不再是两个子空间维数乘积。N=8,D=6,j=k=3时分别1120与3003。

### 8.2 Statevector原始存储不是训练峰值
complex64单纯态存储8*2^Q字节。Q=8/9/12/13/15/16/20/24分别2KiB/4KiB/32KiB/64KiB/256KiB/512KiB/8MiB/128MiB。
B=32、N=8、C=2、每目标一条线路，共512条并行状态；Q9每层原始状态2MiB，Q15则128MiB。反向传播可能保存许多门前状态与角度/临时张量，还要加GPT-2、优化器、激活；这些数字不能用作总显存承诺。
若显式模拟mixed density matrix，原始存储8*4^Q，不能仍用statevector预算。deferred measurement的unitary dilation则要保留被丢弃qubits/ancilla并计入Q。
2×24GiB不是默认一个48GiB地址空间。现有脚本单GPU，可并行两个独立seed/模型；DDP/分片需要另行工程验证。

### 8.3 线路调用与shots
若P为每scene patch数、C通道、L个需单独读出的prefix，batch独立制备数约B*P*C*L，再乘测量setting、shots和梯度重复。经典单车temporal encoder先压缩20帧时，不应无故乘20。
末端r个qubit全Pauli读出有4^r-1个非平凡期望；local Pauli product测量可用3^r组settings覆盖，r=2为15项/9组。每项±1样本均值的标准误差不大于1/sqrt(shots)，同时控制多项精度还有置信度开销。
普通参数移位若每个可移位门出现次数为R，朴素梯度可能需约2R次移位计算；共享参数、多特征编码及外部角度网络必须按实际可微线路计账，不能仅报“量子自由参数少所以成本低”。GPU反向传播与量子硬件参数移位不是同一成本模型。

## 9. 进入原型前的理论门禁

G1 输入权限：只用历史sensing及公开允许的物理量，所有target/context集合定义清楚。
G2 核心职责：联合邻车作用确实进入量子域；不能主要由强GNN或高容量classical triplet-value网络先完成。
G3 物理链路：制备、演化/通道、measurement、条件概率和重上传都有具体定义。
G4 对称性：证明实际门顺序/读出对车辆置换和padding的行为，不只口头共享参数。
G5 可用交互：目标readout的因果锥包含所需邻车-邻车和时间信息；排除测量不可见/精确独立乘积的退化结论。
G6 成本：Q、门数、patch、prefix、settings、shots和backprop分列；不由裸statevector推总训练时间。
G7 归因：保持同信息强经典、高阶和功能匹配对照；量子子块依赖不等于量子优势。
通过这些门禁只表示“值得原型验证”，不保证ADE/FDE提升。最终实验失败条件必须在测试集打开前固定。

## 10. 外部依据与本轮推导边界

- Skolik等，Equivariant quantum circuits for learning on weighted graphs，npj Quantum Information 9,47(2023)，https://www.nature.com/articles/s41534-023-00710-y ：加权图条件演化与节点置换框架。
- Mernyei等，Equivariant quantum graph circuits: constructions for universal approximation over graphs，Quantum Machine Intelligence 5,6(2023)，https://link.springer.com/article/10.1007/s42484-022-00086-w ：图量子线路及普适性需按其模型家族和读出条件理解。
- Raj等，arXiv:2606.26873v1；作者cfi.py固定pin与ICCT代码定位见02。
- Giang等，arXiv:2601.18198v1，III-A/III-C：U_MSG/U_UPD与星形寄存器的原文描述。
- Sweke等，Potential and limitations of random Fourier features for dequantizing quantum machine learning，Quantum 9,1640(2025)，https://quantum-journal.org/papers/q-2025-02-20-1640/ ：RFF反事实具有条件，不是对所有QML的通用否定。
- Constrained and Vanishing Expressivity of Quantum Fourier Models，Quantum 9,1847(2025)，https://quantum-journal.org/papers/q-2025-09-03-1847/ ：频谱规模不等于独立可训练系数规模。
命题1-7及资源算术为本轮独立推导；不得误标为上述作者针对ICCT证明的优势定理。
