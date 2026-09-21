你现在继续接手 ICCT 项目的正式主线。

服务器正式项目：

/home/dell/YrM/ICCT

正式分支：

qgnn

本轮第一件事，完整阅读：

1. docs/task_redesign/ICCT_SIND_CANONICAL_DATASET_DECISION_20260921.md
2. docs/task_redesign/ICCT_PAPER_NARRATIVE_REBUILD_TASK_20260921.md
3. docs/task_redesign/CODEX_GATE_A_INDEPENDENT_AUDIT_20260921.md
4. docs/task_redesign/CODEX_GATE_A_FINAL_REPORT_20260921.md
5. docs/qgnn/RAJ_QGNN_MAIN_SOLUTION_FREEZE_20260921.md

其中第 1 个文件具有最高优先级。

==================================================
一、当前正式决策
==================================================

正式主数据集保持为：

SinD public Changchun + Xi'an

当前官方 Veh_smoothed_tracks.csv 直接视为 SinD canonical trajectory data。

不要再把 SinD 官方 smoothing 当作阻塞项目主线的 data-leakage issue。

不要再搜索 pre-smoothing raw position 作为本轮前置条件。

不要再切回 Lankershim 或其他数据集完成正式主实验。

Lankershim 0a18050 的 Gate A 结果仅保留为：

auxiliary causal sanity experiment

它不是正式 SinD Gate A。

当前正式状态：

SinD Gate A = NOT YET EVALUATED ON THE REDESIGNED TASK

==================================================
二、本轮必须产生大进展
==================================================

不要再“审计一份报告然后停”。

本轮目标是完整推进：

SinD task contract repair
→ SinD-IC4 benchmark implementation
→ 3-BS sensing integration
→ five classical controls
→ preflight
→ seed2026 pilot
→ diagnosis / sufficient-training extension
→ seed2027 confirmation if warranted
→ SinD Gate A decision
→ Gate B migration readiness if PASS

最终目标：

把项目从“旧 2→2 benchmark 太简单、旧 sample contract 有 future-dependent inclusion”

推进到：

“新的 SinD-IC4 上，Graph necessity 已经有可复现实验判决”。

==================================================
三、保留什么，废弃什么
==================================================

保留：

- 官方 Changchun/Xi'an Veh_smoothed_tracks.csv；
- 现有 3-BS controlled ISAC 物理定义；
- 已冻结 interaction rule 的核心思想；
- Raj Weighted Multi-J QGNN 冻结架构；
- 已有 train/val/test physical-vehicle isolation 思路；
- 旧结果作为 historical diagnostics。

废弃 / 修复：

- target 必须 future 完整才能成为 sample 的规则；
- neighbor 必须 future 完整才能进入 context 的规则；
- 根据 future availability 改变 membership；
- 旧 2→2 作为正式论文主任务；
- FDE 使用“最后可见点”代替固定 horizon endpoint；
- 只因 QGNN N<=8 而限制所有 classical baselines。

==================================================
四、正式 SinD-IC4 数据合同
==================================================

主任务：

20 history frames (~2 s)
→ 40 future frames (~4 s)

同时支持：

Full2 / IC2 / Full4 / IC4

每个 sample key：

(scene_id, t0, target_vehicle_id)

sample inclusion 必须先于 future label：

history qualification
→ freeze sample key
→ attach future_xy / future_mask

要求：

1. target 在 history 20 帧完整可见；
2. sample 是否纳入只能看 history；
3. target future 是否完整不能反向决定 membership；
4. neighbor 只要求满足 history-side availability / context rule；
5. neighbor future 完整性不得影响 context；
6. target slot 0；
7. common N<=8 interface 用 history-only ranking；
8. strong all-neighbor baseline 保留 <=50m 的所有合法 history neighbors；
9. future label 只监督 target；
10. future 不完整只由 future_mask 表达；
11. IC membership 只根据 history-side CPA / following rule；
12. IC2 与 IC4 使用同一 membership。

不得根据模型表现重挑 interaction threshold。

==================================================
五、3-BS sensing
==================================================

history 的官方 SinD [x,y,vx,vy] 继续经过现有 controlled 3-BS sensing。

主 pilot 使用 0 dB。

必须保证所有模型使用相同 sensed history population。

不要重做 raw-video tracking，不要为 smoothing 另建 causal reconstruction。

输出新的 SinD-IC4 sensing/data manifest，记录：

- source SHA；
- official record；
- split；
- sample count；
- history/future contract；
- SNR；
- sensing config hash；
- max neighbors / all-neighbor statistics。

==================================================
六、split 与 evaluation
==================================================

继续保持 train / val / sealed test。

禁止 test tuning。

保持 physical vehicle 跨 split 隔离，检查现有 split 是否可直接安全扩展 20→40；如 40-step horizon 需要扩大 guard，重新构建 guard，但不能根据模型结果选 guard。

需要 train 内部 fit / design 机制时，用时间块划分并 purge 跨界轨迹。

Full / IC 是 evaluation strata，不训练两套不同 population 的模型。

==================================================
七、五个 classical controls
==================================================

统一强 temporal backbone，优先复用/重构现有 TCN。

A. Strong Self
- target history only。

B. Own-only Capacity Control
- target only；
- 增加与 graph branch 尽可能接近的额外容量。

C. Context Pooling
- target + neighbors；
- symmetric pooling；
- 不做 edge-aware message passing。

D. Strong Graph N<=8
- target + 最多 7 个 history-ranked neighbors；
- edge-aware message passing / attention。

E. All-neighbor Strong Graph
- 使用 <=50m 的全部 history-qualified neighbors；
- 不受未来 Raj N<=8 工程接口约束。

所有模型尽量共享：

- target temporal encoder；
- coordinate transform；
- decoder；
- residual anchor；
- loss；
- optimizer family；
- checkpoint rule；
- effective train exposure。

禁止弱化 Self / Own / Pool。

==================================================
八、评价指标必须修正
==================================================

ADE：

- 使用 horizon 内真实可见 future_mask；
- 同时报告 common-complete cohort sensitivity。

FDE：

- FDE2 只能计算 t0+2s 对应的固定 endpoint；
- FDE4 只能计算 t0+4s 对应的固定 endpoint；
- endpoint 不存在的 sample 不进入对应 FDE；
- 禁止用最后一个可见 future point替代固定 endpoint。

checkpoint selection 继续只使用一个预定义 J，不得分别挑 ADE/FDE 最优 checkpoint 拼表。

报告：

- ADE
- FDE
- J
- absolute gain
- relative gain
- Graph vs Self
- Graph vs Own
- Graph vs Pool
- All-neighbor vs N<=8
- Full4 vs IC4 gain concentration
- 2s vs 4s
- k / closing / CPA strata

==================================================
九、训练推进规则
==================================================

先 unit / smoke / preflight。

然后直接 seed2026。

不要因为完成 preflight 就停。

如果 bounded pilot 最佳 checkpoint 卡在训练上限或学习曲线仍明显下降：

自动扩大训练预算，直到：

- 基本收敛；
- 或明确达到预注册资源上限；
- 或出现足以判断机制失败的稳定反向结果。

必须记录：

- total train samples；
- actual samples used；
- epochs；
- updates；
- best epoch；
- learning curves；
- CV baseline；
- GPU / runtime。

不要把明显未收敛的 pilot ADE 当最终性能。

seed2026 出现稳定正信号后直接 seed2027。

若 seed2026 明显失败，先做针对性的 contract / optimization diagnosis，不允许调 IC 阈值追求胜利。

==================================================
十、SinD Gate A 判决
==================================================

正式问题：

Does history-side neighbor information provide reproducible predictive value beyond a strong target-only model and matched capacity controls on the redesigned SinD-IC4 benchmark?

PASS 至少应看到：

- Graph / AllGraph 在 IC4 的 ADE 和 FDE 均优于 Self；
- Own 不能解释主要 gain；
- Pool 用于区分“neighbor information”与“message passing”；
- 两 seed 主要方向一致；
- IC4 neighbor gain 高于或至少不弱于 Full4；
- paired / origin-cluster uncertainty 支持主要结论。

如果 Pool ≈ Graph：

结论应是 neighbors useful，但不能声称 message passing indispensable。

如果 AllGraph 明显优于 N<=8：

必须明确未来 Raj 受到 8-node information cap。

输出只能：

PASS
CONDITIONAL PASS
FAIL

==================================================
十一、若 Gate A PASS：直接做到 Gate B readiness
==================================================

不要重新搜索 Raj 架构。

使用已冻结：

Raj Weighted Multi-J QGNN (j=2+j=3)

准备它迁移到同一个 SinD-IC4：

- target + first 7 history-ranked neighbors；
- matched Multi-J Johnson comparator；
- all-neighbor classical graph 作为 information-cap reference；
- same sensed history；
- same target future mask；
- same decoder / loss；
- paired seeds；
- sealed test。

完成：

- interface adapter；
- unit test；
- smoke；
- parameter/fairness audit；
- 正式 Gate B config。

如果 Raj 正式训练属于长实验，准备到用户只需执行一条命令后停止，不要偷偷启动大 sweep。

==================================================
十二、GPT-2 暂不混入 Gate A
==================================================

GPT-2 / LoRA Utility 仍然是 Gate C。

Gate A 不要使用当前明显弱的 GPT-2 作为 temporal backbone 来证明 Graph necessity。

Gate A 先用强 classical temporal model把 interaction necessity 立住。

==================================================
十三、工作区和 Git
==================================================

当前正式工作区存在历史 q0 untracked results。

禁止：

- git clean；
- hard reset；
- stash 别人的工作；
- 删除无关结果；
- 把历史 q0 untracked 文件纳入 commit。

你可以连续做多个内部阶段，不需要每完成一个小步骤等用户。

每个重要 milestone 可以 commit，最终 push qgnn。

最终报告：

- commit SHA；
- configs；
- data manifest；
- preflight；
- seed results；
- Gate decision；
- next command if long experiment remains。

==================================================
十四、本轮最终交付
==================================================

理想情况下，本轮结束时至少有：

1. corrected SinD-IC4 data contract；
2. rebuilt train/val benchmark；
3. new sensing/data manifest；
4. five classical controls；
5. unit/smoke/preflight；
6. converged-enough seed2026；
7. seed2027 confirmation if triggered；
8. SinD Gate A final report；
9. fixed-endpoint FDE implementation；
10. Gate B migration readiness if PASS。

正式主数据集始终是 SinD。

现在开始，直接沿主线大步推进。
