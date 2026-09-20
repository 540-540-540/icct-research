# ICCT 轨迹预测任务重设计 — Codex 执行冻结任务书（2026-09-20）

## 0. 本任务书的地位

本文件冻结 ICCT 项目下一阶段的最高优先级任务：

> **先彻底解决“当前轨迹预测任务过于简单、单车历史即可取得很低 ADE/FDE”这个上游问题，再回到 GPT-2，再回到 QGNN。**

服务器正式项目：

`/home/dell/YrM/ICCT`

正式主分支：

`qgnn`

当前冻结优先级：

[
oxed{	ext{Task Definition} > 	ext{GPT-2 / LoRA} > 	ext{QGNN}}
]

在本任务完成前：

- 不继续优化 GPT-2；
- 不继续调 LoRA / Motion Token；
- 不启动 Raj-QGNN 正式训练；
- 不以“让 GPT-2 赢”或“让 QGNN 赢”为 benchmark 设计目标。

---

# 1. 当前已确认的事实

## 1.1 旧 2s → 2s 任务太容易

当前逐目标、self-only、0 dB SinD 实验已经完成 20 epoch：

| Self-only 模型 | ADE / m | FDE / m |
|---|---:|---:|
| TCN | 0.376311 | 0.816700 |
| LSTM | 0.377409 | 0.822467 |
| Transformer | 0.388013 | 0.829766 |
| GPT-2 + LoRA | 0.604277 | 1.313128 |
| CV 初始基线 | 0.784092 | 1.629495 |

相关正式证据：

- `results/qgnn/self_repair_v1/`
- `reports/qgnn/sind_target_self_repair/completed_audit.md`
- `docs/qgnn/SELF_COMPARABILITY_AND_LLM_DIAGNOSIS_20260920.md`

因此旧任务的主要风险不是“某个模型调得不够好”，而是：

> **2 秒历史 → 2 秒未来很大程度上属于短时惯性 / 平滑运动外推，给 multi-agent interaction reasoning 留出的增益空间太小。**

旧 2→2 任务从现在起降级为：

`historical diagnostic / reference task`

不再作为论文正式主任务。

---

## 1.2 GPT-2 当前确实弱，但暂时不是最高优先级

服务器审计已经排除：

- GPT-2 权重未加载；
- backbone 随机初始化；
- 梯度完全断开；
- own-history adapter 完全没用；
- token CE 主导训练。

同时发现：

- 经典 LSTM/TCN/Transformer 当前直接看到 world-frame 全局方向；
- GPT-2 ego_v1 被强制做成旋转等变，信息接口不完全公平；
- GPT-2 只使用前 4 层；
- 大部分 GPT-2 backbone 冻结；
- 只有 QKV rank-8 LoRA；
- Motion Token 的 explicit expected-motion 对最终坐标的直接影响很弱。

这些问题全部保留，**但当前不解决**。

只有新任务定义被冻结并通过 Graph-necessity Gate 后，才重新开启 GPT-2 专项。

---

# 2. 6Pro 研究结论：作为主候选，不当作已验证最终答案

6Pro 研究产物已经回收至主项目：

主报告：

`docs/task_redesign/GPT6PRO_OPTIMAL_TRAJECTORY_TASK_SELECTION_20260920.md`

机器摘要：

`reports/task_redesign/optimal_task_selection_20260920.json`

历史审计：

`reports/task_redesign/history_only_audit_20260920.json`

阈值预登记：

`reports/task_redesign/design_preregistration_20260920.json`

最终复核：

`reports/task_redesign/revalidation_20260920.json`

分析脚本：

`scripts/task_redesign/`

6Pro 推荐唯一主候选：

[
oxed{	ext{SinD-IC4: history≈2s → future≈4s}}
]

结构：

- target-centered；
- self-only 模型只能看 target 自身 history；
- graph 模型额外看 neighbors；
- Full4 作为完整场景辅助评测；
- IC4 作为 interaction-critical 主评测；
- k>=2 只作为 higher-order 分层，不作为主 inclusion。

但必须明确：

> **6Pro 没有实际训练新任务上的 strong Self / strong Graph，因此没有证明 IC4 一定解决了任务过于简单的问题。**

所以当前状态是：

[
oxed{	ext{Primary Candidate Pending Graph-Necessity Validation}}
]

---

# 3. P0 硬 Gate：先解决历史输入的因果可用性

这是 Codex 第一优先级。

6Pro 复核发现：

- 当前使用的 `Veh_smoothed_tracks.csv` 属于 SinD 平滑轨迹体系；
- SinD 原论文明确描述 Rauch–Tung–Striebel (RTS) smoothing；
- RTS smoother 通常会使用后续观测反向修正过去状态；
- 当前 Changchun / Xi'an 两份文件的具体生成链尚未被证明为严格 causal。

因此：

> **不能直接把当前 smoothed state 当成“预测时刻严格可获得历史”，然后宣称构建了严格 online forecasting benchmark。**

## P0 必须完成的工作

Codex 必须系统检查：

1. 当前两份 Changchun / Xi'an 数据是否存在：
   - unsmoothed raw tracks；
   - raw detector/tracker positions；
   - causal filtered states；
   - 可从官方数据重建的逐帧原始观测。

2. 当前 `Veh_smoothed_tracks.csv`：
   - position 是否也经过 RTS smoothing；
   - velocity / acceleration 是否由 RTS 估计；
   - 是否有生成脚本或官方说明可追溯。

3. 如果存在 raw / causal source：
   - 构建 causal history；
   - velocity 只能由当前及过去观测估计；
   - 禁止任何 future-frame smoothing；
   - frozen 3-BS ISAC frontend 只接 causal state source。

4. 如果只存在非因果 smoothed trajectories：
   - 可以将 smoothed future 作为离线 GT label；
   - 但 history input 不能无说明地继续使用 noncausal state；
   - 若无法构建 causal history，则必须明确把任务定位为：
     `offline latent-state simulation`
     而不是 strict online forecasting；
   - 如果该定位不能满足论文目标，则停止并报告“需要更换/补充 causal source”。

## P0 PASS 条件

必须输出一份 source-causality audit，明确：

- history source；
- future label source；
- velocity derivation；
- causal/noncausal status；
- 对旧结果的影响范围；
- 新 benchmark 是否获准称为 online forecasting。

在 P0 未通过前，不进入正式 Gate G 性能结论。

---

# 4. P1：重建新的 target-centered 数据 contract

在独立新 namespace 中工作。

建议：

`data/task_redesign/IC4_v1/`

不要覆盖：

- 旧 SinD split；
- 旧 sensing cache；
- 旧 target views；
- 旧 2→2 结果。

## 4.1 时间定义

采样率保持原始约：

[
9.99	ext{ Hz}
]

不强制重采样到精确 10 Hz。

历史：

[
20	ext{ frames}approx2.0s
]

主未来：

[
40	ext{ frames}approx4.0s
]

同时保留 20-frame future 作为旧任务对照。

因此至少构建：

- Full2 / IC2
- Full4 / IC4

其中 IC membership 必须用同一 history-only 规则定义，不能根据 future horizon 改变。

---

## 4.2 target-centered contract

每个样本只监督一个 target：

[
(scene_id,t_0,target_track)
]

输入：

`history_state [20,N,4]`

状态：

[
[x,y,v_x,v_y]
]

target 固定语义 slot 0。

Graph 模型：

- target + neighbors；
- common interface 最多 N=8；
- 强 classical graph 额外保留 all-neighbor 版本，不能只因 QGNN N<=8 就削弱经典 baseline。

Self-only 模型：

- 只能看 target history；
- 不能访问 neighbor state；
- 不能访问 interaction score 作为输入特征。

---

## 4.3 membership 必须先于 label

样本是否纳入只能由 history 决定。

禁止：

- target 必须 future 完整才能进入；
- neighbor 必须 future 完整才能进入；
- 根据 future maneuver / future error / future collision 选样本。

正确顺序：

[
	ext{history membership}

ightarrow
	ext{freeze sample key}

ightarrow
	ext{attach future label + mask}
]

future 缺失只影响：

`future_mask`

不能反过来改变 membership。

---

# 5. P2：冻结 IC4 interaction-critical 规则

以下规则来自 6Pro 的 train-only preregistration，本轮 Codex **不得根据模型表现重新挑阈值**。

候选 neighbor 首先满足：

[
d_{ij}le50m
]

## 5.1 CPA / approaching interaction

定义：

- relative position (r_{ij})
- relative velocity (u_{ij})

closing：

[
c_{ij}
=
-rac{r_{ij}^{T}u_{ij}}{|r_{ij}|}
]

TCPA：

[
	au_{CPA}
=
-rac{r_{ij}^{T}u_{ij}}
{|u_{ij}|^2}
]

DCPA：

[
DCPA
=
|r_{ij}+	au_{CPA}u_{ij}|
]

CPA critical 条件：

[
c_{ij}ge0.5m/s
]

[
0<	au_{CPA}le4.004s
]

[
DCPAle5m
]

---

## 5.2 following proxy

同时满足：

- target speed >= 0.5 m/s；
- neighbor speed >= 0.5 m/s；
- heading difference <= 30°；
- neighbor 位于 target 前方；
- abs(lateral offset) <= 2.5 m；
- longitudinal time headway <= 2 s。

---

## 5.3 主 inclusion

若 target 至少存在一个：

[
CPA critical lor FOLLOW
]

neighbor，则：

[
k_ige1
]

进入 IC。

主任务：

[
oxed{IC4:kge1}
]

高阶分层：

[
kge2
]

只用于 higher-order mechanism 分析，不作为主任务硬筛选。

---

# 6. P3：统一坐标接口，避免“global-heading shortcut”混淆任务难度

新 Task Redesign Pilot 中，Self 与 Graph 必须使用**同一坐标信息**。

主 pilot 默认：

[
oxed{	ext{target-local / ego-local coordinate}}
]

原因：

- 当前 TCN/LSTM/Transformer 使用 world-frame 时会利用固定路口 absolute heading；
- GPT-2 ego_v1 不使用这个 shortcut；
- 如果 Task 本身要研究 interaction necessity，应先把地图/道路方向 shortcut 与 interaction signal 分开。

所有模型共同：

- target 最后一帧 sensing position 为局部原点；
- heading 只由 history 估计；
- 不能用 future yaw；
- Self/Graph 使用同一个 transform；
- output inverse-transform 回 world meter 后统一计算 ADE/FDE。

额外可保留 world-frame 作为 diagnostic control，但不能让 Self 与 Graph 使用不同 coordinate information。

---

# 7. P4：Graph-Necessity Pilot — 本任务真正的决定性实验

当前先不使用 GPT-2，也不使用 QGNN。

核心问题只有一个：

[
oxed{	ext{给模型加入 neighbors 后，是否真的产生显著且稳定的增益？}}
]

## 7.1 Primary Self backbone

首选：

`TCN`

原因：

- 当前旧任务下它是最强 self-only baseline；
- 如果 Graph 连最强 Self 都打不过，就没有继续 QGNN 的意义。

确认性 backbone：

`scratch Transformer`

仅在 TCN pilot 给出正信号后进行确认。

---

## 7.2 必须至少有四个模型

### A. Strong Self

TCN target-only。

### B. Own-only capacity control

在 A 上加入与 Graph 分支近似参数量的 target-only residual MLP。

目的：

> 排除“Graph 只是参数更多”。

### C. Context pooling control

Self + neighbor DeepSets / symmetric pooling。

目的：

> 判断“任何邻车信息”是否有用。

### D. Strong Graph

Self + edge-aware MPNN / attention message passing。

要求：

- 与 A 使用同一 target temporal encoder；
- 同一 decoder；
- 同一 loss；
- 同一 coordinate system；
- 唯一新增信息是 neighbor history / pair relation。

### E. All-neighbor strong Graph

保留 50m 内所有 eligible neighbors，不受 N=8 cap。

目的：

> 检查 QGNN 的 N<=8 工程接口是否损失重要 context。

---

# 8. P5：Pilot matrix

必须至少跑以下四个 evaluation cells：

| Horizon | Full | Interaction-critical |
|---|---|---|
| 2s future | Full2 | IC2 |
| 4s future | Full4 | IC4 |

Self/Graph 分别在对应 horizon 训练。

IC2 与 IC4 使用**相同 history-only interaction membership 规则**，不要根据 horizon 改阈值。

核心观察：

[
GraphGain
=
Error(Self)
-
Error(Graph)
]

我们希望验证：

1. Graph 在 IC4 有明显增益；
2. Graph 在 IC4 的 gain 大于 Full4；
3. GraphGain(4s) 大于 GraphGain(2s)。

只有这三个方向基本成立，才能说：

> 任务从“短时惯性外推”成功转向“需要 multi-agent interaction reasoning”。

---

# 9. Gate G — 新任务是否真正解决“太简单”的核心验收

这是本轮最高优先级性能 Gate。

## 9.1 Design pilot

先在 train-internal design split：

seeds:

[
2026,2027
]

要求：

- Strong Graph 在 IC4 的 ADE/FDE 两个指标都优于 Strong Self；
- Own-only capacity control 不能解释主要增益；
- all-neighbor control 不应显示 N=8 丢失了大部分 graph gain。

若两 seed 方向都不稳定，停止扩大实验。

---

## 9.2 Frozen confirmation

若 design pilot 通过，再冻结配置，使用：

[
2026,2027,2028,2029,2030
]

五个 paired seeds。

Gate G 建议沿用 6Pro preregistration：

### IC4

相对增益：

[
ADE gainge10%
]

[
FDE gainge10%
]

绝对增益：

[
Delta ADEge0.05m
]

[
Delta FDEge0.10m
]

且：

- 5 seeds 至少 4 个同时正向；
- paired cluster bootstrap 95% CI 下界 > 0；
- Full4 任一主指标相对退化不超过 2%；
- 各城市平均增益方向不得明显反向，样本过少必须披露而不是删掉。

### Horizon claim

如果论文要写：

> “延长 prediction horizon 后 interaction 更重要”

还必须验证：

[
GraphGain_{4s}-GraphGain_{2s}>0
]

并给 paired CI。

如果没有通过，只能写：

> “IC4 中 neighbor information 有价值”

不能写 horizon causality。

---

# 10. Gate G 失败时的处理

禁止：

- 改 DCPA 到一个刚好能赢的值；
- 改 radius 找最好结果；
- 把 k>=1 改成某个模型最有利的 k；
- 用 future error 筛 hardest samples；
- 继续延长 horizon 直到 graph 赢。

若 IC4 Gate G 失败：

1. 保留所有结果；
2. 判断：
   - SinD 当前两录像 interaction dependency 是否本身不足；
   - target definition 是否仍被单车运动主导；
   - all-neighbor graph 是否也无明显增益；
3. 输出结论：
   - 如果 all-neighbor graph 也无明显 gain，则当前 SinD 不适合作为 interaction-centric QGNN 主任务；
   - 此时应考虑更换/补充数据集或研究目标，而不是继续调 benchmark。

这是允许的科研结论。

---

# 11. GPT-2 专项明确延后

在 Gate G 通过前：

[
oxed{	ext{不要再优化 GPT-2}}
]

已有 GPT-2 问题保留为下一阶段任务：

- ego-local vs world-frame 信息公平；
- full fine-tune vs LoRA；
- pretrained vs random-init；
- 4-layer vs deeper；
- Motion Token vs continuous-only；
- LoRA rank / placement；
- token auxiliary objective。

只有新任务冻结后，才在新 Full4 / IC4 上重新解决。

旧 2→2 的 GPT-2 = 0.604 不需要继续追。

---

# 12. QGNN 继续关闭

Raj-PennyLane P1.1 工程核心保留，不修改。

只有同时满足：

1. 新 task contract 冻结；
2. P0 causal-source 结论明确；
3. Gate G 通过；

才允许重新打开：

[
Classical GNN
quad vsquad
Raj	ext{-}QGNN
]

QGNN 不参与 benchmark 选择。

---

# 13. Codex Git / worktree 规则

当前还有其他 agent 使用正式：

`/home/dell/YrM/ICCT`

因此 Codex 不得在正式工作目录切分支。

推荐从**启动时 qgnn 已提交 HEAD**创建：

branch:

`task-redesign-codex-20260920`

worktree:

`/home/dell/YrM/ICCT_task_redesign_codex`

Codex：

- 不清理正式 qgnn 工作区；
- 不 stash 其他 agent 文件；
- 不把正式 working-tree 未提交内容带入分支；
- 所有 task-redesign 修改只在独立 worktree；
- 完成后只提交临时分支；
- 不自行 merge 回 qgnn；
- 不打开 test。

---

# 14. 建议 Codex 产物

## 14.1 Source causality

`docs/task_redesign/CODEX_SOURCE_CAUSALITY_AUDIT.md`

`reports/task_redesign/codex_source_causality.json`

## 14.2 New data contract

`docs/task_redesign/CODEX_IC4_DATA_CONTRACT.md`

`reports/task_redesign/ic4_dataset_manifest.json`

## 14.3 Pilot

`docs/task_redesign/CODEX_GRAPH_NECESSITY_PILOT.md`

`reports/task_redesign/graph_necessity_pilot.json`

## 14.4 Final freeze recommendation

`docs/task_redesign/CODEX_FINAL_TASK_FREEZE.md`

必须最终给出：

- PASS：IC4 正式冻结；
- 或 FAIL：SinD 当前设置仍不足以支撑 interaction-centric 主任务。

不能停在“建议再试几个阈值”。

---

# 15. 本轮完成定义

Codex 任务只有在下面全部回答后才算完成：

1. **历史输入是否严格 causal？**
2. **Full4 / IC4 数据 contract 是否无 future-based membership？**
3. **Self 与 Graph 是否使用同样坐标与时间信息？**
4. **Strong Graph 是否在 IC4 明显优于 strongest Self？**
5. **该 gain 是否不能被 own-only extra capacity 解释？**
6. **4s 是否比 2s 提供更多 interaction headroom？**
7. **N=8 是否保留了大部分 all-neighbor graph gain？**
8. **最终是否批准 IC4 成为正式论文主任务？**

---

## 一句话执行原则

> **先证明任务真的需要邻车信息，再讨论 GPT-2 怎么做得更好；先证明 classical graph 有存在价值，再讨论 QGNN。**
