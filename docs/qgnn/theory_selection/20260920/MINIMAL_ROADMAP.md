# TRC-QGNN 最小原型验证路线

本文件摘自完整选型报告；没有新训练已执行。

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
