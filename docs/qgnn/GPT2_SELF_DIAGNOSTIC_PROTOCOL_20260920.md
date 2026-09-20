# GPT-2 Self 弱势：第一轮方向与适配诊断

日期：2026-09-20。状态：实现和短预检完成，正式训练待用户启动。

本轮用户明确要求优先处理 GPT-2，覆盖此前任务重设计文件的阶段顺序。当前工作仍是旧 2→2 任务上的机制诊断；不改变 IC4 阈值、正式数据或研究结论，不启动 QGNN。

## 1. 对上传分析的核对

认同的事实：
- TCN 0.376311/0.816700，GPT-2 ego 0.604277/1.313128，ADE/FDE 均以米计；已有完整验证集回放。
- 固定旧目标后差距仍在，不能仅归因于评价人群变化。
- 经典模型保留全局方向，GPT-2 ego 接口抹去了它；当前模型比较混合了表示与训练方式的差异。
- 当前 GPT-2 是四层、主要冻结主干、QKV rank-8 LoRA；不是完整十二层全参数轨迹适配。
- 预训练参数正确加载，本车连续状态通路有效。方向与适配限制值得优先检验。

需要限制的解释：
- 全局方向在预测时可用，是合法特征。固定路口方向先验可能影响泛化，不能直接称“作弊”。
- 旋转不等变说明模型依赖坐标方向，不能量化这种依赖解释了多少验证误差。
- 小批次 CE 梯度证据不支持当前局部主导，但不能排除训练早期/全程影响。
- 96 个目标中的 token 分布不代表全数据词表利用率；屏蔽 expected_motion 的约 5 mm 预测变化，只涉及坐标头的显式读出，不等于移除历史 token 输入或 CE。
- 未来完整性筛选会改变群体，改变难度的方向尚未测量，不能直接认定它必然让目标更容易。
- 单 seed 大差距值得排查，但尚未测量跨 seed 方差。

参数更新比值已重算：own_history_adapter 的初始 L2=0.555817，净位移=29.967740，比值=53.916593；LoRA A+B 初始 L2=31.975871，净位移=6.656813，比值=0.208182。adapter 小初始化与 LoRA B 零初始化使分母不可横比。四层实际 QKV 修正矩阵范数相对原权重约为 8.83%、14.44%、13.44%、12.88%；仍不等于精度贡献，不能据此说“主干几乎不动”。

第 15→20 轮，GPT 训练总 loss 从 0.739367 到 0.696446；验证 ADE 从 0.609725 到 0.604277，FDE 从 1.314570 到 1.313128，选中点仍为第 20 轮。结论是验证收益放缓，不能证明续训绝不可能改善，也没有证据保证能追平 TCN。若续训，必须重新定义调度与匹配预算，不能盲目超出原 cosine T_max=20。

## 2. 本轮五个新训练配置

| 名称 | 方向接口 | GPT 主干初始化 | 主干训练方式 |
|---|---|---|---|
| tcn_ego | 本车坐标 | 不适用 | 全部 TCN 参数 |
| gpt_heading_pretrained_lora | 本车坐标 + 历史方向 | 预训练 | QKV LoRA |
| gpt_heading_pretrained_full | 同上 | 预训练 | 全部实际使用的 GPT 主干 |
| gpt_heading_random_lora | 同上 | 随机 | QKV LoRA |
| gpt_heading_random_full | 同上 | 随机 | 全部实际使用的 GPT 主干 |

复用已完成的 TCN-global 与 GPT-ego-pretrained-LoRA 为参考，不重跑相同控制。新模型旧 checkpoint 回放与三步更新均精确一致，初始化、CUDA dropout 种子和优化顺序已核对。

LLM 始终保留原 19 个运动 token、本车连续状态、token CE、Self head、20 步输出及 16m residual cap。本轮不同时改 continuous-only 或去除 CE，以免把方向/训练自由度与表示变化混在一起。

方向接入采用由历史计算的 [cos(theta), sin(theta)]，经零初始化线性层加入历史 token；没有方向时输入为零。额外 1,536 个参数，相比约 409 万可训练参数很小但须披露。这是“接入方向信息”的接口因素实验，不是证明所有模型输入和容量完全等价。

TCN ego 使用同一历史方向估计、局部坐标输出与逆变换，无方向时抑制矢量残差。与 global TCN 的差别还包括局部残差范围和停驶约定；因此只称 ego 接口对照，不把全部差值严格归于一对 heading 数值。

random 只替换 GPT h/wpe/ln_f。所有公共外围、运动 embedding、LoRA 初始 A/B、queries 和 head 保持相同；运动 embedding 仍带有共同的语言词表初始化。这识别的是 Transformer 主干预训练作用，不是清除一切语言来源信息。未用于 inputs_embeds 前向的 wte 始终冻结。

full 模式使实际使用的 GPT blocks、position embedding 和 LayerNorm 可训练，LoRA 冻结且 B=0。LoRA 模式保留原冻结主干。可训练参数：heading+LoRA 4,088,727；heading+full 33,129,879。不得把未用词嵌入的参数量当成有效运动建模容量。

## 3. 固定训练合同

- 原 target_views_v1：319,855 train / 31,211 val，0 dB，20→20，逐目标、center-only。
- 原数据全量与样本顺序，origin-macro 权重；不削减 TCN 训练集或邻居可见性。
- seed 2026；20 epoch；batch256；25,000 updates。
- 外围初始 LR=3e-4，LoRA/全参数主干 LR=7.5e-5；AdamW weight_decay=2e-4；gradient clip=3。
- 原 CosineAnnealingLR，T_max=20，共同绝对 eta_min=2.4e-5。两组并非全程保持 4:1 LR。
- GPT CE=0.035，TCN 无 CE；此差别保留，当前首先识别方向与主干训练方式。
- 每个模型按同一 J=ADE+0.5FDE 选择 checkpoint，分别报告该点 ADE/FDE，并保留末轮结果。
- 明确使用复用验证集做机制诊断，不称新独立确认。不读 test。
- full/random 检查点保存完整模型、optimizer、scheduler、CPU/CUDA RNG、epoch/batch 和部分 epoch 累计量。禁止沿用旧“省略冻结 GPT 参数”的保存方式。
- 用源码快照字节比对与 data manifest 防止错误恢复；旧训练、模型和数据文件保持原样。

此固定 LR 和 20 轮预算支持同预算机制比较，不保证所有架构都已调到最优。若边界最佳、训练不充分或随机主干收敛更慢，后续应给对应比较双方合理且一致的追加机会，不用一次筛查声称普遍正/负迁移。

## 4. 已完成验收

证据：
- reports/qgnn/gpt2_self_diagnostic/preflight.json
- reports/qgnn/gpt2_self_diagnostic/runner_preflight.json

通过：
- 旧 TCN-global、GPT-ego 以及零方向注入的旧 checkpoint：32 个实际训练目标前向最大差为 0。
- 旧/新 GPT ego 同输入与配对 dropout 三次更新：参数与预测最大差均为 0。
- 五个配置 epoch0 均为 CV；可训练参数有更新、冻结参数不变、梯度有限。
- random 仅改变指定 GPT 参数，外围初始化一致；unused wte 冻结。
- 扰动邻居/无效槽位不改变 Self 输出；TCN ego 旋转检查在 float32 容差内。
- 五个配置同时执行各三次优化步并正常停止。两张 4090 的实测总显存峰值约 11.24 / 10.95 GiB。
- random/full 在第 1 步停止再恢复至第 3 步，跨 epoch 边界：与连续执行的模型、optimizer、scheduler 和 RNG 完全一致。
- shell 语法、只打印命令模式与模型/runner只读集成审查通过。
- 临时预检 checkpoint 已清理，保留小型验收报告；正式训练未启动。

短步测时包含初始化预热，不能当成可靠总耗时；按原 GPT 约 47 分钟的记录及新同卡并行开销，整批粗估约 1–2 小时，实际看首轮速度。启动器的每次运行保护上限为 6 小时，达到后保存并停下，不表示预计训练需要 6 小时。

## 5. 一条用户启动命令

在服务器终端执行：

~~~bash
cd /home/dell/YrM/ICCT && bash scripts/run_gpt2_self_diagnostic.sh
~~~

五个配置同时启动：GPU0 为 TCN + 两个 pretrained GPT，GPU1 为两个 random GPT。每个 GPT 峰值约 4–5 GiB，加上并行运行上下文仍有余量。所有任务写独立目录和日志；中断后同一命令从完整 checkpoint 续跑，已完成配置不会追加训练。结束时自动显示两个旧参考和五个新配置的选中 ADE/FDE。

## 6. 如何读结果

- heading+LoRA 相比旧 ego+LoRA 改善：支持方向接入能解释部分差距，不能把改善误称主干变强。
- TCN ego 退化而 GPT heading 改善：进一步支持当前坐标方向接口影响排名；仍保留 TCN-global 强参考。
- 在相同 heading 与预训练下，full 明显优于 LoRA：支持原适配限制是因素之一。
- 同 full 模式下 pretrained 优于 random：支持这一预算/接口下主干预训练有收益；反过来则当前没有正迁移证据。
- 同 LoRA 模式下 pretrained 对 random：识别冻结适配条件下的主干预训练贡献，不能用 pretrained+LoRA 对 random+full 代替。
- 四个 heading GPT 均弱：接着检查 continuous-only、层数与预算等，不从本轮直接断言 GPT 架构不适合，也不继续以 GPT 必须获胜为目标。
- token 整体有效性只能由后续固定条件的重训练消融回答，当前 5 mm 读出干预不能代替。

新配置完成之前，GPT-2 指标偏弱的各因素贡献仍未知；本轮完成的是可执行、可归因的诊断准备。
