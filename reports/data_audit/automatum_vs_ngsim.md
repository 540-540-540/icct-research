# Automatum T-Crossing vs NGSIM Lankershim 项目层面对比

- 性质：**对比审计（AUDIT ONLY）**，不改动任何原始数据、不预设结论。
- Automatum 证据：`reports/data_audit/automatum_t_crossing_audit.md` + `reports/data_audit/automatum_t_crossing_stats.json`（本轮实测）。
- NGSIM 证据：`reports/data_cleaning/source_session_audit.md` + `reports/data_cleaning/source_session_stats.json`（仓库已完成审计）。
- 对比口径：统一按“ICCT 目标 = 3 fixed BS + multi-vehicle ISAC + N_max=8 + 2s history/2s future @10 Hz + GNN/QGNN 公平比较”评估，而不是单纯比数据量。

---

## 0. 数据身份

| | NGSIM Lankershim | Automatum T-Crossing |
|---|---|---|
| 来源 | FHWA Next Generation Simulation | Automatum Data |
| 采集地 | 洛杉矶 Lankershim Blvd（约 500 m 干道，3 信号交叉口） | 德国 Ingolstadt 附近两个 T 型路口（相距 ~5.0 km） |
| 文件 | `Lankershim_Vehicle_Trajectories.csv`（1,607,319 行） | `automatum_data_crossing.zip`（SHA256 `ea2c528f…f19f`） |
| 审计 | `source_session_stats.json` | `automatum_t_crossing_stats.json` |

---

## 1. 逐项对比（工单 §17 表格）

| 维度 | NGSIM Lankershim | Automatum T-Crossing | 依据 |
|---|---|---|---|
| **数据时长** | 2 时段共 2,167.4 s（1,120.4 + 1,047.0，含 100.6 s 时钟重叠） | 1,776.3 s（650.7 + 1,125.6，无重叠） | NGSIM 长 22% |
| **车辆数量** | 2,442 条 (session, ID) 轨迹 / 1,506 个 ID | 683 条 UUID 轨迹 / 683 个 UUID | NGSIM 轨迹数多 3.6× |
| **数据点** | 1,607,319 行 | 244,377 点 | NGSIM 多 6.6× |
| **原始频率** | 10.00 Hz（原生） | 29.97 Hz（精确均匀网格） | NGSIM 原生即目标频率 |
| **坐标单位** | 英尺（Local_X/Y、Global_X/Y）；跨时段 Local 有平移 | 米；局部系与 xodr 同系，metadata 给 UTM/WGS84 | Automatum |
| **vx/vy** | 只有标量 `v_Vel`、`v_Acc`，无矢量速度 | 车体系 `vx/vy` + `ax/ay` + `jerk` + `curvature` | Automatum |
| **ID 问题** | 936 个 ID 跨时段复用，必须用 (session, ID)；跨段端点跳跃 P50 314 m | UUID 段内唯一、跨 recording 0 重叠 | Automatum |
| **重复轨迹** | 4 条边界车被两段同时导出（速度相关 0.99+） | 0（无重复点、无重复帧） | Automatum |
| **时间排序** | 文件序 228 个负步 + 220 个 200 ms + 220 个 ≥1 s 假跳变，需按时间重排 | 0；683/683 严格递增，Δt 恒 0.0333667 s | Automatum |
| **地图质量** | 无地图文件，仅轨迹坐标 | `staticWorld.xodr` lane 级几何，轨迹 99.8%/100% 落在车道面内 | Automatum |
| **清洗复杂度** | 中：行序修复 + 会话键 + 单位换算 + 4 条重复取舍 + 跨段坐标平移核对 | **无**（0 缺失/0 断轨/0 时间错误/0 跳点/0 重复/0 ID 冲突） | Automatum |
| **预处理复杂度** | 低：原生 10 Hz，切窗即可 | 中：29.97→10 Hz 一次（推荐网格最近邻法），其余同 | NGSIM 略优 |
| **多车交互** | 极密：62.9 / 86.1 辆每帧（含信号排队） | 6.45 / 3.51 辆每帧；交叉配对帧 52.0% / 26.9%；主+支路同现 15.3% / 29.6% | 看目标：ICCT 要“≤8 车交互”，Automatum 更匹配 |
| **3-BS 几何** | 500 m 直线干道，3 BS 近似共线，方位/角度多样性差 | T 型三臂：候选三角形面积 ~1,800 m²，最小内角 44°–56°，100% 车点 <100 m | Automatum |
| **2s→2s 适配** | 好（原生 10 Hz，排序后连续） | 好（时间戳严格均匀，10 Hz 重采样无插值、半帧误差） | 平 |
| **N≤8 场景适配** | 需从 63–86 辆/帧重度裁剪，局部窗口构建复杂 | **天然满足**：B 99.1% 帧 ≤8；A 需 25 m 级局部裁剪 | Automatum |
| **训练规模** | 1.6 M 状态点；按 41 帧窗估算约 1.5 M 单车窗口 | 244 k 状态点；54,113 单车窗口，其中 2–8 车场景 46,841 | NGSIM 大，但 ICCT 目标不需要 |
| **绝对精度佐证** | 无独立参考（同为视觉轨迹） | 无独立参考（同为视觉轨迹）；厂商 KPI 与实测完全吻合 | 平 |

### 补充证据细节

- NGSIM 的来源结构经专项审计确认：两个 source session（帧时钟 offset 差恰 1,020,000 ms），Vehicle_ID 分时段重新分配；4 条重复轨迹集中在时段边界。
- Automatum 的两个 recording 各自独立（不同 UTM 参考点、不同场景），UUID 命名空间互不重叠，合并时零 ID 冲突。
- Automatum 的 vx/vy 经位置差分独立验证：MAE 0.038/0.041 m/s（旋转 psi 后世界系 0.051/0.060 m/s）；若按世界系误用则误差 14–23 m/s——**使用时必须按车体系语义处理**。
- Automatum 官方 aux 字段（lane/road/TTC/lane-change）在 Release 3.0 中为空，车道级标签需项目自行由 xodr 投影生成（NGSIM 同样不提供车道几何，其 Lane_ID 是标量编号且无几何定义）。

---

## 2. 对 ICCT 目标的影响分析

| ICCT 目标 | NGSIM | Automatum | 结论 |
|---|---|---|---|
| 3 fixed BS | 直线走廊上难形成非共线几何 | 每个路口 3 臂天然非共线 | Automatum |
| multi-vehicle ISAC | 邻域车辆过多（60+），sensing 场景需重度裁剪 | 天然 2–8 车邻域，距离中位 ~14 / ~22 m | Automatum |
| N_max = 8 | 必须裁剪，且密度分布与 8 车约束不匹配 | B 天然 ≤8；A 轻裁剪 | Automatum |
| 2 s history + 2 s future @10 Hz | 原生 10 Hz，直接切窗 | 网格最近邻降采样（半帧误差），窗口供给 4.7 万（2–8 车） | 平（Automatum 需一次重采样） |
| GNN/QGNN 公平比较 | 单走廊、单一时段风格，域多样性和划分策略受限 | 2 个路口 × 2 种密度/车型/速度分布，可做跨域验证 | Automatum |
| 训练时间可控 | 数据大 6.6×，需下采样控制成本 | 规模适中，直接符合预算 | Automatum |

## 3. 客观反面证据（不能只看 Automatum 优点）

1. **绝对规模**：NGSIM 的轨迹数与数据点是 Automatum 的 3.6× / 6.6×，若未来需要预训练或大规模基准，Automatum 单一来源规模偏小。
2. **时长**：NGSIM 36.1 min（含重叠）对 Automatum 29.6 min；单录制可用时长 B 为 18.8 min，稍短。
3. **原生频率**：NGSIM 10 Hz 零转换成本；Automatum 每次引入 29.97→10 Hz 的重采样约定（尽管误差 ≤16.7 ms）。
4. **元数据缺口**：Automatum 缺 lane/road/TTC 标签；NGSIM 至少有 Lane_ID/Direction 标量字段。
5. **精度声明**：两者都是视觉轨迹，均无外部真值校验；Automatum 声称的绝对精度无法从文件内证明。

## 4. 结论

在 ICCT 当前任务（3-BS、≤8 车、2s→2s、多车交互、GNN/QGNN 对比）的约束下，Automatum T-Crossing 在“清洗成本、地图对齐、ID 稳定、N≤8 适配、3-BS 几何、交互真实性”上全面优于 NGSIM，仅在“数据规模、原生频率、元数据字段”上各有落后项，且这些落后项对当前目标不构成阻塞。

**建议：正式迁移 Automatum 为 ICCT 主 Ground Truth；NGSIM 保留归档作为历史/辅助数据，不做删除，也不在本轮做任何清理。**