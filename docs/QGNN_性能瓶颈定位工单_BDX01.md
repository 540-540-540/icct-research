# BDX-01｜QGNN 性能瓶颈定位工单

**交付对象：Codex**  
**版本：1.0｜编制日期：2026-09-14**  
**仓库：`540-540-540/icct-research`**  
**编制依据提交：`6652c7df09c47401e7685540b8618e6887564965`**  
**本轮性质：现有模型的只读诊断；允许新增诊断代码与报告，参数更新次数必须为 0。**  
**交付目标：形成可以上传 GitHub、供后续重新分析的证据包，而不是再试一个改进模型。**

> 执行要求：阅读本工单和仓库适用的 AGENTS.md 后，在具备项目数据与检查点的环境中完成实现、检查、诊断和报告。不要只输出实施计划就结束。正式模型训练、加训、新种子实验和结构替换不属于本轮；需要这些实验时，写入下一阶段建议，完成本轮后停止。无法完成的项目必须保留状态、原因和已取得的结果。

## 0. 本轮必须回答的五个问题

1. 原 QGNN、候选 A 和强 GNN 在同口径、推理模式下，是否都表现为训练集继续改善而开发集退化？
2. 剩余误差主要随预测时域增长，还是集中在特定交通片段、短历史或低邻居数样本？
3. QGNN 与 GNN 是否主要错在相同的目标、时刻和方向？它们的预测究竟有多大差别？
4. 当前感知估计误差、源轨迹偏离恒速的程度，以及 SNR 引起的输入变化，分别有多大？
5. 候选 A 没有提高整体指标，是新增分支几乎未影响输出，还是确实改变了预测但收益与损失相抵？

本轮能够提供瓶颈定位证据，不能单凭这些诊断宣布“贝叶斯误差下限”“量子性能上限”或“量子优势已证明”。也不能预先指定必须找到一种提高 QGNN 指标的方法。

## 1. 当前基线与执行边界

### 1.1 已有事实与应复用的工作

当前主线仍是师兄双层 `InheritedQGNNGraph`；候选 A 是新增 2,560 个经典参数的可学习有向上下文残差。候选 A 与原 Q 的同 seed 配对结果为：

| 对象 | 最佳 epoch | 最佳 ADE/m | 最佳 FDE/m | 最佳 J/m |
|---|---:|---:|---:|---:|
| 原 QGNN | 75 | 0.542767 | 1.095422 | 1.090478 |
| 候选 A：directed_context | 54 | 0.542487 | 1.096786 | 1.090880 |
| 同协议强 GNN | 82 | 0.550223 | 1.118283 | 从原始结果读取 |

这些是历史核对锚点，不是本次诊断的目标值；验收使用原 JSON 精度，而不是表中的六位小数。[S1、S3]

`reports/qgnn_directed_context/d0.json` 已有监督长度、原模型参数漂移、固定 train batch 梯度和部分门控诊断。本轮先校验来源并复用，不能把这些旧结果写成本轮新实验，也不需要再做一套完整的梯度研究。[S2]

### 1.2 本轮可执行与保留不动的内容

- 可执行：train/V_select 的现有检查点评估、源行对齐、误差统计、固定模型上的候选 A 关闭实验、诊断代码测试、图表和报告生成。
- 保留不动：主模型源码、已有配置、训练损失与选模规则、原检查点、旧实验报告、GPT-2 基座、数据缓存和现有划分。
- 本轮不运行 optimizer.step、训练 smoke、warmup、adapt、joint fit、学习率搜索、多种子训练、候选 B、QGAT 或新的本车/经典核心训练。
- 允许读取的集合为 `train` 和 `V_select`。不加载 V_confirm/test 输入、标签、预测或新指标；读取旧协议中的曝光说明不等于新访问这些集合。
- 原始源文件中的必要行仅用于 train/V_select 对齐与事后诊断；参考状态、源身份及未来标签不进入模型输入。
- 默认不自动 commit/push。生成发布白名单和提交建议，由用户同步；用户另行明确授权提交时，只处理本工单产物。

### 1.3 运行环境

预期服务器项目为 `/home/dell/YrM/ICCT`，Python 为 `/home/dell/YrM/envs/ICCT/bin/python`。首先核对真实环境；路径不存在时检查已有项目说明，不得把仅有文档的本地目录当作完整运行环境。

读取当前 HEAD、工作区状态、适用 AGENTS.md 和训练协议。HEAD 更新时，比较相关源码与检查点所存合同；不要强制 checkout 旧提交或清理用户改动。记录本轮实际代码版本与差异。

使用现有 GPU 环境，最多占用两卡、每卡一个诊断进程；复用现有精度与数学后端。不得终止其他进程。资源不足时串行或减小推理 batch，并先通过批次等价检查；不改变样本集合。

## 2. 文件范围与新入口

新增以下独立目录，名称如有冲突，使用新的 run_id，保留既有版本：

```text
experiments/qgnn_bottleneck/
  diagnose.py               # 唯一主入口：prepare / run / verify
  metrics.py                # 统一统计口径与合成测试
  analysis.py               # 源对齐、SNR、误差重合与候选A干预
  plot_results.py           # 从已落盘表格绘图
  tests/                    # 必需单元/集成测试

configs/qgnn_bottleneck_diagnostic.json

reports/qgnn_bottleneck/<run_id>/
  PROTOCOL.md
  REPORT.md
  NEXT_DECISION.md
  run_manifest.json
  summary.json
  checks.json
  run_status.json
  artifacts_manifest.json
  PUBLISH_MANIFEST.md
  run.log
  tables/
  figures/
  source_snapshot/           # 实際使用但尚未被Git跟踪的关键公共源码

results/qgnn_bottleneck/<run_id>/
  cache/                    # 全量预测、残差、掩码；仅保存在服务器
```

本工单要求实现的入口，不是仓库已经存在的命令：

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u experiments/qgnn_bottleneck/diagnose.py \
  --config configs/qgnn_bottleneck_diagnostic.json --prepare

/home/dell/YrM/envs/ICCT/bin/python -u experiments/qgnn_bottleneck/diagnose.py \
  --config configs/qgnn_bottleneck_diagnostic.json --run

/home/dell/YrM/envs/ICCT/bin/python -u experiments/qgnn_bottleneck/diagnose.py \
  --config configs/qgnn_bottleneck_diagnostic.json --verify
```

`--run` 应自动完成准备复核、检查、评估、分析、绘图和报告，不调用任何训练入口。支持显式 `--resume`：只有检查点、源码、输入、掩码、归一化、配置、统计规则完全一致的缓存才能复用。状态按子任务原子保存；恢复不能覆盖已经确认的结果。

配置至少包含：run_id、两项允许 split、四档 SNR、检查点清单、推理 batch/设备、预测时域、诊断干预、分组规则、数值容差、源数据访问范围、输出路径。所有规则在正式评估前落盘。

## 3. D0：检查点和来源核对——先通过，再分析

### 3.1 使用六个指定检查点

| 标识 | 预期 run 目录 | 检查点 | 预期 epoch |
|---|---|---|---:|
| Q_best | `results/qgnn_directed_context/seed2026/original/` | best.pt | 75 |
| Q_latest | 同上 | latest.pt | 95 |
| A_best | `results/qgnn_directed_context/seed2026/directed_context/` | best.pt | 54 |
| A_latest | 同上 | latest.pt | 74 |
| G_best | `results/qgnn_motionframe/seed2026/gnn/` | best.pt | 82 |
| G_latest | 同上 | latest.pt | 102 |

epoch 是根据已提交报告确定的预期值，实际以 `progress`、summary/history 与归档验证文件一致为准。出现差异时先记录原因，不能偷偷切换到效果更好的 run。

优先使用本轮配对的 original，而不是把 `qgnn_inherited100` 的旧双卡结果混进来。`qgnn_readout/.../original` 只有在所需可变权重及完整来源相等时，才可作为替代来源；记录为什么等价。若缺少匹配 GNN 检查点，不得静默换成 `qgat_extend100` 历史 GNN。

### 3.2 来源清单与只读保证

`run_manifest.json` 记录：

- 当前 HEAD、相关工作区差异、脚本/公共依赖 SHA256、Python 与核心库版本、设备与精度。
- 六个检查点的真实路径、SHA256、epoch、run 配置、模型可变键集合；冻结基座另记模型来源和文件校验。
- train/V_select 的 input、label、metadata、normalization 的实际路径及 SHA256；沿用项目已有输入语义哈希。
- 源行映射依赖、允许读取的字段与实际使用的 split。
- 开始/结束时间、运行命令、随机种子、缓存键及实际完成矩阵。

完整模型应按所属 run 的真实类构造，严格核对所有可变权重键。冻结 GPT-2 未写入小检查点是既有设计；只能明确允许这类键，不能用无条件 `strict=False` 掩盖缺失权重。

当前 Git 中可能没有独立的 `prediction/` 或 `frontend/`。在服务器优先用实际文件，并与验收报告 `source.code_text`/检查点合同核对。相关公共源码不在 Git 时，将实际使用版本的纯文本快照放入本轮 `source_snapshot/`，附 SHA256 与使用说明；不改原文件，不整份复制包含无关内容的大 JSON，不复制认证文件或模型权重。[S4]

### 3.3 预测复现检查

先在完整 V_select 四 SNR 上复现三个 best 的归档验证结果：

- 检查每条 `(origin, snr)` 的 ADE/FDE、有效目标数与宏平均。
- 整体和逐场景 ADE/FDE 绝对误差容差预设为 `1e-5 m`；归档为 null 的场景继续为 null。
- 核对 best/latest 的 epoch 与实际加载权重；latest 不能被共享 fit 逻辑重新恢复成 best。
- 使用 `model.eval()` 和项目已验证的 `torch.no_grad()` 路径。不要无验证改成 `torch.inference_mode()` 或把全部参数 requires_grad 强行关闭；当前 PennyLane 包装对无梯度执行有特殊处理。
- 在固定的至多 16 个 origin 上核对有/无诊断 hook 的未干预输出及不同推理分批输出，张量容差 `atol=2e-5, rtol=2e-5`；布尔掩码必须完全一致。
- 前后比较模型参数/缓冲区，检查点文件哈希必须不变。A 关闭实验在内存副本进行，单独记录其预期临时变化与恢复。

归档复现失败时，停止依赖该模型的性能分析，先报告差异。不能在查看差异后放宽容差或修改主模型来“对齐”。其他不依赖失败模型的诊断继续，并标记 partial。

## 4. D1：同口径 train/V_select 评估

### 4.1 固定评估矩阵

六个检查点 × `{train, V_select}` × `{5,10,15,20 dB}`，全部有效起点。预期 train 5549、V_select 400；实际数量变化时核查数据版本。另加一个无需训练的 `CV_hat` 参照：

\[
\widehat p^{CV}_{i,h}=\widehat p_{i,t}+h\Delta t\widehat v_{i,t},\quad \Delta t=0.1\text{s}.
\]

CV_hat 使用与各模型完全一致的预测资格与标签掩码。它是恒速参照，不是“训练过的仅本车模型”，也不是系统性能上限。

四档 SNR 全部评估，不能因为结果接近只保留 20 dB。每个权重只作这次固定评估，不重新选检查点，不给最新轮做偏向性挑选。

### 4.2 明确三个统计口径

设 `eligible_i = exists_last_i & (history_exists_count_i >= 3)`；
`valid_hi = label_valid_hi & eligible_i`；`H_i = sum_h valid_hi`。预测时域 K 取 5、10、20 步。

**主指标：沿用既有车辆宏平均，再场景宏平均，再 SNR 宏平均。**

\[
ADE_K(s)=\frac{1}{|I_{s,K}|}\sum_{i\in I_{s,K}}
\frac{\sum_{h=1}^{K}valid_{hi}\,\|\hat p_{hi}-p_{hi}\|}{\sum_{h=1}^{K}valid_{hi}},
\]

其中 `I_s,K` 为前 K 步至少一个有效标签的合格目标。`FDE_K` 只使用**第 K 步标签有效**的目标；不能拿该车最后一个有效点代替终点。ADE 和 FDE 分别排除没有相应监督的场景，分别记录分母。

**训练式指标：**在同一 eval() 预测上调用或精确重建原 `masked_trajectory_loss`，报告点加权的 `ADE_point`、`FDE`、`J_trainrule_eval`。保留原训练对无监督场景记零且仍进入 batch/全体均值的规则，并另报排除无监督场景后的辅助数值，避免混淆。[S2、S4]

**固定目标辅助指标：**在 `label_valid` 全 20 步有效的目标上计算全部时域的误差，独立报告样本覆盖。主指标不因这一辅助统计而过滤短标签目标。

这样能够区分：训练日志与 eval 的差别、点加权与车辆加权的差别、以及随时域变化的目标构成差别。

### 4.3 应输出的比较

1. 六个检查点在两集合的完整 ADE20、FDE20、J20，train 与 V_select 的同口径差值；不要直接称这个差值是纯过拟合量，因为两集合难度也可能不同。
2. 每个模型内部 latest−best 的变化：train 是否改善、V_select 是否退化；同一权重评估两集合，保持 dropout 关闭。
3. Q_best−G_best、A_best−Q_best，分别报告绝对差与相对改善；正向改善统一定义 `100*(reference-candidate)/reference`。
4. 0.5/1/2 秒的 ADE_K/FDE_K，以及第 1–20 步误差曲线。变长/固定全轨迹目标两种结果分开。
5. CV_hat 的结果与各 best 的改进幅度。所有分母与合格目标应可以复算。

历史训练曲线可以直接从 history/summary 导出；`loss` 标为“训练模式日志损失”，不与同口径 eval 曲线混成一个指标。没有中间检查点时，不伪造中间 epoch 的 eval-train 曲线。

**产物：**`tables/metrics.csv`、`tables/best_latest.csv`、`tables/horizon_metrics.csv`、`tables/scene_metrics_V_select.csv`。

## 5. D2：误差重合、来源片段与输入分组

本节主比较用 Q_best、G_best、A_best；先对同一目标-起点的四 SNR 误差求平均，再作配对统计。计数单位明确为“目标-起点对”，不能称为独立车辆数。

### 5.1 完整输出以下量

- 目标-起点 ADE20/FDE20 的 Q/G Pearson 和 Spearman 相关；有效数为零或方差为零时输出 null 和原因。
- Q/G 各自最差 10% 的目标-起点对的交集数量、交集/集合大小、Jaccard。`k=ceil(0.1*n)`，并列时用 `(origin,slot)` 排序确定；ADE/FDE 分别计算。
- 两个模型在相同目标上同时好/同时差不能只凭相关系数判断；同时给出成对误差差值分布、胜负数量、中位数与 P90。
- 0.5/1/2 秒处预测分歧 `||p_Q-p_G||` 的均值、中位数、P90、P99。
- 对终点有标签的样本，计算两个终点残差向量的夹角余弦与同向比例；任一范数不超过 `1e-6 m` 时余弦记为 unavailable，保留数量。
- 在估计末帧速度大于等于 `1 m/s` 的目标上，将残差投影到该目标前向/横向，输出有符号均值及绝对值统计；低速单独统计，不定义虚假的行驶方向。

相关高只能说明错误相关，不证明误差不可约；与共同真值比较也可能天然产生相关。事后 top10% 只作诊断，不用于训练筛样、改主指标或选择宣传案例。

### 5.2 来源分组

复用 metadata 中经核验的 `episode_id`、`source_block`、`source_groups` 和源身份连接规则。已有案例代码有完整实现，但硬编码了旧 Q/G 路径和全局 CUDA 环境，不能原样运行或不加审查地 import。[S5]

分别输出 episode、source block、保守 source component 的样本数与配对差值。增加留一 episode 和留一 component 的均值范围与符号变化。训练集与 V_select 的依赖结构分别计算；不新构造其它 split。

本轮不对重叠窗口、目标-起点或四档 SNR 做 IID t 检验或普通 bootstrap，不报伪精确 p 值/置信区间。component 少时报告证据稀疏，不把 episode 留一检查冒充独立显著性检验。

### 5.3 限定分组，避免无边界搜索

只使用下列五类输入条件；每类单独统计，不枚举交叉组合：

| 条件 | 定义与区间 |
|---|---|
| 末帧现存轨迹数 | 1–2 / 3–5 / 6–8；不解释为道路总车数 |
| 目标真实输入邻居数 | 有效非自环、末帧距离≤45m：0 / 1–2 / ≥3 |
| 目标有效历史帧数 | 3–9 / 10–19 / 20 |
| 历史速度变化代理 | 复用首/末五帧方法；只用 train 的三分位边界，固定后用于 V_select |
| 历史航向变化代理 | 首/末五帧、各至少3帧、均速≥1m/s：<5° / 5–15° / ≥15° |

缺少变化或有效样本的分组输出 unavailable。不要把航向变化代理写成变道真值、缺槽写成漏检、加减速写成已证实的交互事件。

可以另报 source-reference CV 偏离程度的事后分组，但必须单列 `posthoc_label_based`，不得与输入可得分组混称部署条件，也不用于决定本轮阈值。

**产物：**`tables/target_metrics_V_select.csv`、`tables/by_episode.csv`、`tables/by_source_component.csv`、`tables/by_input_stratum.csv`、`tables/error_overlap.json`。逐目标公开表可用四 SNR 平均后的宽表；完整逐 SNR/逐步残差留本地缓存。

## 6. D3：感知输入、SNR 和恒速残差分解

### 6.1 必做：先查 SNR 输入确实不同

在 train/V_select 内，按同一 source/episode/frame/slot 对齐四档输入，输出：

- 输入文件及数组语义哈希；state_hat 的逐维统计；exists/detected 的一致性及变化计数。
- 相对 20 dB，各档位置/速度向量差的均值、中位数、P90、P99、最大值、完全相等比例。仅在现存目标上计算；同时记录窗口条目数和按 episode/frame/slot 去重后的计数，重叠窗口不视为独立观测。
- 对 Q/G/A 的 best，报告逐目标预测 `||p_snr-p_20||`，以及 ADE/FDE 的**配对有符号差**和差的绝对值统计；不能仅比较四个总均值。
- 检查评估缓存键明确含输入内容/归一化/检查点哈希及 SNR，不能把20dB结果误复用到其它条件。

如果输入或预测异常完全一致，先检查加载器、条件路由、缓存与量化，不能直接宣布“抗噪性很好”。不同 SNR 的原噪声流可能共用随机种子，不能把它们当成独立采样实验。[S4、S6]

### 6.2 条件必做：参考状态可核验时完成物理分解

依据 train/V_select metadata 的 `track_keys/source_keys/source_groups/origin_ms` 和现有源生成规则，把估计状态与源侧同一车辆、同一时刻的参考位置速度对齐。先选择固定少量样本人工/程序交叉核验坐标轴、单位、时间步与身份，再全量计算。

源侧速度必须核实定义。优先复用生成感知仿真状态的那个速度；若实际是历史位置差分/回归估计，称为 `v_ref`，写明窗口与算法，不称“精确真实瞬时速度”。不得用未来位置中心差分构造起点速度，也不得借未来标签补齐缺失源行。

原始 CSV 混合集合时，按 metadata 白名单处理本轮必要源行；不对其余行做统计或建立新预测集合。源映射/速度来源无法核验时，将本项标为 `blocked_source_alignment`，保留6.1和D1/D2，不编造真值分解。

对有参考状态的目标，令

\[
A_i=p^{ref}_{i,t}-\hat p_{i,t},\qquad
B_{i,h}=h\Delta t(v^{ref}_{i,t}-\hat v_{i,t}),
\]
\[
C_{i,h}=p_{i,t+h}-p^{ref}_{i,t}-h\Delta t v^{ref}_{i,t}.
\]

则

\[
b_{i,h}=p_{i,t+h}-(\hat p_{i,t}+h\Delta t\hat v_{i,t})=A_i+B_{i,h}+C_{i,h}.
\]

- 输出起点位置/速度估计误差、A/B/C 各自范数的均值/中位数/P90，以及0.5/1/2秒的结果。
- 输出模型修正 `r_hat = p_model - CV_hat`，并校验最终残差 `p_future-p_model = A+B+C-r_hat`。
- 可额外输出平方范数分解：`||b||² = ||A||²+||B||²+||C||²+2A·B+2A·C+2B·C`，所有项使用相同掩码和相同权重；交叉项允许负值。
- A/B/C 的范数不能直接相加、归一化成“误差来源占比”，不能把 C 全部解释成交通交互或不可预测性。
- 在参考状态可用的公共子集上，增加 `CV_ref = p_ref + h*dt*v_ref`，同时重算该子集的 CV_hat 和 Q/G/A。CV_ref 只是源侧参考恒速参照，不是完整模型的 oracle，也不是可实现精度下限。

向量恒等式用 float64 校验，预设 `atol=1e-8, rtol=1e-10`；同一组坐标数据应支持该代数精度。主指标仍保持既有精度。参考状态缺失的目标仅从本项剔除，不能从主指标删除；输出覆盖率、被剔除原因及两套明确分母。

### 6.3 残差与估计误差的联系

在同一公共子集按源片段报告：估计误差较大时模型是否也更差，Q/G 的变化是否一致；先描述关联，不据相关性宣布感知误差是因果主导因素。无需改输入、重跑感知前端、改变 SNR 范围或给模型输入参考状态。

**产物：**`tables/snr_input_output.csv`、`tables/source_alignment.json`、`tables/residual_decomposition.csv`、`tables/reference_cv_metrics.csv`。

## 7. D4：候选 A 的固定模型关闭实验

本节只测试 A_best、A_latest 的两个状态：原样 on、两层有向上下文矩阵同时关闭 off。不开新的超参网格，不测试候选 B。

1. 用候选 A 自己的检查点加载独立内存副本；仅把两层 `receiver_context` 和 `sender_context` 置零，其余权重完全保留。
2. on 结果直接复用 D1；off 在 train/V_select 全部起点、四档 SNR 上评估。
3. 输出 `off−on` 的 ADE/FDE/J 绝对变化、相对变化、逐片段变化、逐目标改善/恶化数量及预测变化范数的中位数/P90/P99。
4. off 后恢复，再验证固定 batch 输出回到 on；磁盘检查点 SHA256 保持不变。记录实际临时变化的键集合，必须恰好为四个指定矩阵。
5. 在固定、来源均衡的 train 小子集上可记录原上下文与新增残差的尺度，并报告零/近零分母；这仅解释分支量级，不升级为新梯度研究。

解释规则：

- 新增参数变化，不代表新增分支一定有显著预测作用。
- 平均 off−on 接近零，不代表逐目标预测都没变；必须同时看预测变化和正负收益分布。
- off 后变差，说明完整 A 依赖该路径，不证明 A 优于原 Q。
- off 后变好，也不能直接宣布“删掉分支得到新最佳模型”；它是事后干预结果。
- A_off 不等价于从头训练的原 Q，因为其余共享参数已经走过不同训练轨迹。

**产物：**`tables/context_intervention.csv`、`tables/context_intervention_by_episode.csv`、恢复检查记录。

## 8. D5：报告、结论边界与下一阶段决策

### 8.1 最少四张图，全部由已保存表格生成

1. `01_best_latest`：三个模型各自 best/latest 在 train/V_select 的同口径 ADE/FDE。
2. `02_horizon_error`：Q_best、G_best、A_best、CV_hat 的0.1–2秒误差；明确每步评价分母，并附固定完整目标辅助结果。
3. `03_paired_errors`：Q/G 配对 ADE/FDE 的散点与等误差线；颜色/分组如使用，只来自预先定义的源或输入条件。
4. `04_context_intervention`：A on/off 在 best/latest、两集合上的变化，附预测改变量或片段分布，避免平均数掩盖抵消。

每图输出 PNG 与 SVG，保留绘图代码和源表路径。不存在的诊断不画示意数据，不用平滑曲线制造改善。差值图标明单位（m或mm），局部纵轴明确标注。

### 8.2 REPORT.md 固定结构

1. 执行范围、实际检查点、完成/阻断状态。
2. 归档指标复现与只读验收。
3. 同口径 best/latest、train/V_select 结果。
4. 时域、输入分组与源片段诊断。
5. Q/G 错误重合和预测分歧。
6. SNR 输入输出核查、参考状态和残差分解。
7. 候选 A on/off 的实际作用。
8. 每个瓶颈假设的支持证据、反证、未解决项。
9. 下一步只推荐一项最有信息量的动作，并说明为什么。

所有数值引用相对路径和 JSON 键/CSV 行选择条件，例如 `tables/best_latest.csv: model=Q, split=V_select`。不能只写“见日志”。报告开头不写未经多种子验证的“稳定优势”或“已经收敛到极限”。

### 8.3 必须覆盖的假设表

| 假设 | 本轮可提供的证据 | 仍不能从本轮推出的结论 |
|---|---|---|
| 后期泛化退化 | eval-train改善而eval-V_select退化、片段方向 | 唯一原因必为过拟合或容量已足够 |
| 当前SNR扰动不是主要问题 | 输入/输出逐样本变化、参考估计误差 | 感知无误差、任何噪声条件都无影响 |
| Q/G受共同难例限制 | 配对误差、方向、困难集合重合 | 误差不可约、公共时间模块一定有问题 |
| 新增上下文未形成净增益 | on/off、预测变化、收益抵消 | 所有有向关系编码都无效 |
| 图交互或量子额外收益有限 | CV比较及已有图干预的有限线索 | 从头训练本车/同架构经典核心的成绩 |

`NEXT_DECISION.md` 可以在以下后续实验中推荐一项，但不实现、不启动：

- 从头训练的仅本车时序模型：测量当前已实现的跨车增量。
- 同混合架构的合理经典核心对照：定位量子核心/接口是否限制性能；不能只用明显受限的 rank-2 核心充当充分对照。
- 原 Q 与强 GNN 的配对多种子：估计训练稳定性；明确种子不增加独立交通场景。
- 共同训练日程/泛化诊断：需给 Q/G 对称机会，不拿新规则 Q 对比旧规则 G。

不写自动触发条件去启动训练，不把“工单通过”解释成已授权下一轮。

## 9. 验收测试与失败处理

### 9.1 必须通过的测试

- **掩码：**不存在槽位、历史不足3帧、无未来标签、只部分未来有效、第20步缺失、全空场景；FDE不偷换最后有效步。
- **指标：**同场景两个目标分别20点和5点，逐点误差分别1m和3m时，点加权ADE为1.4m、车辆宏平均ADE为2m；另测全20点时两口径一致。合成数值只标为单元测试。
- **分母：**train 中无监督场景显式计数；主指标排除方式与归档一致，训练式指标保留其原规则；最后不足 batch 的尾批按实际数量处理。
- **时域：**第5/10/20步对应0.5/1/2秒，prefix ADE和单步FDE不混淆；固定完整目标子集不会反向改变主指标。
- **配对：**所有比较显式按 `(split,origin,snr,slot)` join，核对源身份和mask，不依赖“两个数组刚好顺序一致”。
- **聚合：**原始残差→逐目标→逐场景→SNR→宏平均的链路可复算；episode加权恢复总体，与episode等权结果区分。
- **复现：**三个best归档复现，六检查点身份正确，无hook/恢复后预测等价，缓存不串模型/SNR。
- **只读：**零优化器更新；原模型文件、数据、已有结果及检查点哈希不变；A临时变化仅在副本、恢复通过。
- **访问：**实际访问日志仅涉及允许的集合与源行，配置检查不允许绕过；只写 `confirmation_opened=false` 不算访问审计。
- **恢复：**用至多两个固定origin生成诊断缓存后模拟中断，验证恢复与未中断结果相同；这不是训练smoke，不做任何参数更新。

`checks.json` 每项都有 status、实测误差/计数、容差和证据路径。对已有D0写 `reused_verified`，本轮新测试写实际状态。严禁只给所有项目填 true。

### 9.2 状态与部分完成

统一状态使用 `pending/running/passed/failed/blocked/not_applicable/reused_verified`。结束时区分：

- `execution_status=complete`：本轮主诊断全部完成，来源与恒等式等条件项的适用性已经核验。
- `execution_status=partial`：缺源映射、缺某个检查点等造成部分诊断无法完成；列出影响哪些结论。
- `execution_status=blocked`：主检查点/输入无法访问或主复现失败，不能进行有效主比较。

`research_conclusion` 与 `execution_status` 分开：程序执行成功，不代表某项性能假设成立。缺少参考速度时不得把分解记为 passed；没有训练过的对照不得出现在结果表中。

## 10. 供 GitHub 回传的产物与 Codex 最终回复

### 10.1 必需回传内容

将工单保存到 `docs/QGNN_性能瓶颈定位工单_BDX01.md`，并准备以下发布白名单：

- 新增诊断代码、配置、测试与绘图代码。
- 本轮 PROTOCOL、REPORT、NEXT_DECISION、summary、checks、run_manifest、run_status。
- 本轮所有小型 CSV/JSON 汇总表、完整 V_select 逐场景指标、四SNR平均的逐目标诊断表、源片段差值表。
- PNG/SVG 图和必要的公共源码快照。
- `PUBLISH_MANIFEST.md`：每个待发布文件的相对路径、用途、字节数、SHA256；与服务器保留资产分开。

建议单个回传表不超过20MB，大型逐步数组保留在服务器并记录准确路径、形状、dtype、字段与哈希。不得通过截取“最好的一部分样本”来减小文件。汇总报告必须足以离线复算主要结论，不能把关键内容全部留在服务器日志或二进制缓存里。

不上传检查点、GPT-2权重、原始大数据、缓存全量NPZ、账号认证、访问令牌、环境私密配置或无关文件。保留既有Git忽略规则；需要回传被忽略的新小文件时采用精确白名单，不使用 `git add .`。

`artifacts_manifest.json` 与 `PUBLISH_MANIFEST.md` 的哈希白名单均排除这两个清单本身，避免自引用或相互引用哈希。执行基准SHA和最终发布SHA分开记录；最终发布SHA在真正提交后写于用户回传说明，不提前编造。

### 10.2 summary.json 必需结构

```json
{
  "schema_version": "BDX-01-v1",
  "execution_status": "pending",
  "run_id": null,
  "code_base_commit": null,
  "optimizer_steps_executed": null,
  "actual_splits_accessed": [],
  "checkpoint_manifest": {},
  "archive_reproduction": {},
  "same_metric_best_latest": {},
  "horizon_diagnosis": {},
  "error_overlap": {},
  "source_group_sensitivity": {},
  "snr_input_output_checks": {},
  "reference_state_decomposition": {},
  "directed_context_intervention": {},
  "supported_hypotheses": [],
  "unsupported_hypotheses": [],
  "unresolved_questions": [],
  "recommended_next_action": null,
  "new_training_started": null,
  "blocking_issues": [],
  "evidence_paths": []
}
```

这是输出schema模板，不是预设实测结果。运行后填入真实值，不能用默认0/false替代实际访问和调用核查。

### 10.3 Codex 完成后只需明确交代

实际完成/阻断范围；零参数更新与未读取确认/测试的证据；三条最重要发现；仍未解决的问题；`REPORT.md` 的路径；待上传白名单；若用户授权且确已推送，给出真实提交SHA，否则写明“未推送”。

工单结束于本轮报告与发布准备。由用户回传GitHub提交后再作结果评审，不自主扩展到新训练或模型改造。

## 11. 本工单的仓库来源索引

以下路径以编制依据提交 `6652c7df09c47401e7685540b8618e6887564965` 为基准；执行时还应记录实际来源版本。

- **S1：**`reports/qgnn_directed_context/STATUS.md`、`seed2026/summary.json`、`configs/qgnn_directed_context.json`：候选A范围、六个检查点中的Q/A结果及训练协议。
- **S2：**`reports/qgnn_directed_context/d0.json`、`experiments/qgnn_directed_context/diagnose.py`：已有标签长度、固定batch梯度、参数漂移等诊断。
- **S3：**`reports/qgnn_motionframe/seed2026/RESULT.md`及对应summary：与原Q同协议的从头训练强GNN，区别于历史extend100对照。
- **S4：**`reports/qgnn_readout/training_checks.json`、`reports/qgnn_motionframe/training_checks.json` 中冻结的 `prediction/temporal.py`、`prediction/training.py`、`prediction/classical.py`、`frontend/symbol_dataset.py` 等源码；服务器对应文件优先核对。
- **S5：**`experiments/qgnn_cases/stratify.py`、`diagnose.py`、`weighted_gate_control.py`及`reports/qgnn_cases/REPORT.md`：源关联、案例指标和固定模型干预；旧脚本含历史检查点硬编码。
- **S6：**`configs/symbol_frontend.json`、`configs/f01a.json`、`reports/f01d/F01D_REPORT.md`：上游状态、时域、SNR、数据划分和源状态生成边界。

本文件是待执行工单。除明确引用的历史结果外，没有声称本轮诊断已运行，也没有预先给出瓶颈裁决。

