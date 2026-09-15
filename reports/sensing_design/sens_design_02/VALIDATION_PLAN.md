# VALIDATION_PLAN.md — REBUILD-03 验收测试计划（T1–T8）

所有测试在 REBUILD-03 实现时落到 `frontend/check_shared_dataset.py`（结构与隔离）
或独立诊断脚本（压力/消融）。只读 train/V_select 或合成场景，禁止 V_confirm/test。

## T1 Shared-echo structural test（共享回波结构）

**目的**：证明"一个 BS 一份共享观测，先求和、后加噪"。

**方法**：
1. 固定 3 个目标状态（几何覆盖 3 BS），同一帧种子。
2. 生成单目标 clean 回波 `Y_i`（noise=False，逐目标单独调用）与共享 clean 回波 `Y_shared`。
3. 断言 `Y_shared == Σ_i Y_i`（float64，相对误差 <1e-12），逐 BS、逐 entry。
4. 生成共享噪声回波（noise=True）：`N = Y_noisy − Y_shared`。
   断言 `E|N|²` 与配置噪声方差一致（样本估计，±5%），且噪声在各目标位置处不可区分
   （不存在"逐目标噪声"：同一 (b,a,k,n) 只有一份噪声实现）。
5. 置换不变性（P0-2）：
   - 目标顺序置换 → clean 共享回波不变（仅浮点求和顺序容差）；
   - 输出 slot / 目标列表顺序置换 → `X_b`、`W_b` 逐比特不变；
   - 同一 source_key 两次调用相位一致；不同 source_key 相位独立。

**通过条件**：线性求和成立；噪声只加一次；目标顺序与 slot 置换不影响共享观测；
waveform/noise 与 target 解耦。
**失败即架构错误**（退回 oracle 分离假设）。

## T2 No-GT detection test（检测器无 GT 输入）

**目的**：从 API 与运行两个层面证明检测/关联/跟踪看不到 GT。

**方法**：
1. 静态 AST 检查：`frontend/sensing/detector.py`、`frontend/sensing/coords.py`、
   `frontend/fusion/association.py`、`frontend/tracking/cv_kf.py` 的 import 图不含
   `frontend.echo_source`、`frontend.scene_manifest`、`source_states`。
2. 签名检查：上述模块公开函数参数中不含 target/truth/gt/vehicle/id/count 类参数。
3. import 哨兵：把 `frontend.echo_source` 替换为抛异常 stub 后运行整条 B 域链，必须无异常。
4. 运行期置换测试：同一组 (状态集合) 以打乱顺序生成共享回波，detector 输出检测集合
   （未排序集合意义下）不变；输出中不含任何 GT 字段。
5. 用"数量相同但内容不同"的目标集合（如同数量随机状态）生成观测：检测数量随观测质量变化，
   而不是恒定等于目标数。
6. 协方差 LUT（P0-3）：`C_xy` 只能由检测的 observed `peak_to_noise_db` 查表得到；
   更换 GT 内容/顺序不影响同一观测的 `C_xy`；不同 q 分箱给出单调不增的 σ（保守性检查）。

**通过条件**：四类检查全部通过；检测数为观测的函数，不是 GT 的函数。

## T3 Sensing-quality response（感知质量随 SNR 单调响应）

**目的**：证明 SNR 真实作用于检测/估计（消除审计中的"5–20 dB 平台"）。

**方法**：
1. `scripts/calibrate_f01e_snr.py` 在固定 train 子集（例如 2 episodes × 若干帧）上
   sweep `snr_ref_db ∈ {-10,-5,0,5,10,15,20,25,30,35,40}`（配对种子）。
2. 统计：
   - 检测召回（与 GT 的 C 域匹配）与虚警数/帧；
   - 单站位置 RMSE（匹配成功样本）；
   - 融合后位置 RMSE；
   - 航迹连续性（track_exists 连续长度、ID switch 次数）；
   - `detected=0 & track_exists=1` 的 coast 比例。
3. 断言趋势：随 SNR 下降，召回下降、RMSE 上升、miss 上升；虚警允许变化（CFAR 固定 Pfa 时近似不变，
   上调 Pfa 时应上升）。
4. 输出曲线与"clean / nominal / challenging"候选区间（只报告，不冻结）。

**通过条件**：存在明显斜坡与拐点；不存在"整个 sweep 内几乎不变"的平台。

## T4 Density stress（密度压力）

**方法**：用固定脚本构造 3/5/8 目标场景（从 train 真实帧抽取，或合成同分布状态），
记录：每 BS 检测数（含合并）、关联正确率、融合位置 RMSE、航迹确认率、ID switch、假航迹数。

**通过条件**：性能随密度平滑退化（无爆炸/断路）；8 目标时关联正确率与确认率仍可用
（阈值在 REBUILD-03 实测后由用户评审确定，设计不预写数字）。

## T5 Close-target stress（近邻目标分辨）

**方法**：两目标从分离逐步靠近，分别扫三类间距：
`Δr = 0.5…4 m`、`Δu` 对应角距 `1°…15°`、`Δv = 0.5…5 m/s`。

**检查**：何时从 2 检测合并为 1 检测；合并后是否只更新一个航迹/是否产生假第二航迹；
NMS 半径与 CFAR 保护单元是否与理论分辨率（1.61 m / 6.35° / 1.97 m/s）一致。

**通过条件**：合并发生在理论分辨极限附近（±1 个栅格）；不产生由旁瓣引起的稳定假航迹；
结果写入报告作为已知局限。

**V1 限制声明（P0-1）**：每个 RD 峰只取最强 AoA 峰（`aoa_max_peaks=1`），因此同一
距离-速度单元内的多目标必然合并为 1 个检测；T5 必须把该合并曲线写入报告。

## T6 3-BS ablation（多站消融，P0-4 修订为两层）

### T6-A sensing/fusion ablation（tracker 之前）
同一共享回波数据，分别只用 1 / 2 / 3 BS 做检测+关联+融合（不进 tracker），比较：
per-frame detection recall、false alarms、单站/融合位置 RMSE。
**通过条件**：3 BS ≥ 2 BS ≥ 1 BS（RMSE 与召回），差异可解释（几何/角度覆盖）。

### T6-B tracking ablation（公平比较传感器数量的信息增益）
1/2/3 BS 三档使用**同一套消融确认规则**：

```text
最近 3 帧内 >=2 次 measurement hit（忽略 n_bs>=2 要求）
```

实现方式：仅 T6-B 允许 `confirm_requires_nbs2=false`；
正式 mainline tracker 始终保留 D12（`>=2 hits/3 frames + 至少 1 次 n_bs>=2`）。
报告必须标注：

> ablation mode 只用于公平比较传感器数量，不改变正式 mainline tracker。

**通过条件**：三档均能形成完整航迹并给出位置/速度 RMSE 与连续性；
3 BS ≥ 2 BS ≥ 1 BS（个别几何退化档位可例外并解释）。

## T7 Tracking dropout（漏检/丢帧）

**方法**：
1. 合成一位形已知目标，在第 10–13 帧人为丢弃其检测（或把功率降到 CFAR 以下）。
2. 断言：coast 期间 `track_exists=1, detected=0`；KF 预测误差有界；
   连续 miss > `max_missed=5` 后航迹删除且 slot 冷却；恢复检测后新航迹重新确认。
3. 断言全程 `detected & ~track_exists` 为 False。

**通过条件**：语义与状态机完全符合 `SYSTEM_MODEL.md` §7-8；无状态跳变/NaN。

## T8 Oracle-vs-sensing downstream gate（下游闸门，REBUILD-04 执行）

**方法**：冻结同一预测训练协议，分别在
`data/f01d`（oracle/known-target）与 `data/f01e`（sensing）上训练下游模型，比较：

```text
Δ_sensing = (ADE_sensing − ADE_oracle) / ADE_oracle
```

**工程目标（非领域标准）**：正常 operating point 下 Δ_sensing ≲ 5–10%。
仅在 sensing 校准完成、用户批准后执行；DESIGN-02/REBUILD-03 不训练下游模型。

## 测试覆盖矩阵

| 测试 | 阶段 | 落点 | 依赖 |
|---|---|---|---|
| T1 | P1 | check_shared_dataset / unit | simulator |
| T2 | P1-P4 | check_shared_dataset | detector/fusion/tracker 接口 |
| T3 | P6 | calibrate_f01e_snr | 全链 + 少量 train |
| T4 | P5 | diagnostics densify | 全链 |
| T5 | P5 | diagnostics close-target | 全链 |
| T6-A | P3/P5 | diagnostics ablation | detector+fusion |
| T6-B | P4/P5 | diagnostics ablation（`confirm_requires_nbs2=false`） | tracker |
| T7 | P4 | unit/integration | tracker |
| T8 | REBUILD-04 | prediction 训练协议 | 新 cache + 用户批准 |