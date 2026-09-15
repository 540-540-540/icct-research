# EXTERNAL_REUSE_MATRIX.md — 外部参考与本仓库复用矩阵

本轮允许临时 clone/阅读外部代码，禁止整体嵌入或大段复制无明确许可的源码。
所有"移植"均指在理解算法后用本项目风格重写为 Python/CUDA 实现，并在注释/文档中标注算法来源。

## 1. NIST 5GNRad

- 仓库：`https://github.com/usnistgov/5GNRad`
- 审计版本：commit `e10114ee78a813b19992640f9b7729cd451d7178`（2026-02-13），MATLAB R2025a，278 文件。
- 许可：`license/license.md` = NIST 开发软件许可（可修改/分发，保留声明、注明改动并致谢 NIST）。
  不整工程嵌入；若后续移植算法，在 `REBUILD_03` 文档中写"algorithm adapted from NIST 5GNRad <file>"。

| 5GNRad 文件 | 做什么 | V1 采用方式 | 不采用的部分 |
|---|---|---|---|
| `src/+nrRadar/+sens/getRangeDoppler.m` | PRS 提取 → 加窗 range IFFT → 慢时间去均值 → 加窗 Doppler FFT | **结构参考**：同序处理，但我们的 X 是全网格 QPSK，无 PRS destaggering；clutter 去均值关闭（无静态杂波模型） | PRS 资源映射、destaggering、range bin 截断、MATLAB 实现 |
| `src/+nrRadar/+sens/cfar2D.m` | 2D CA-CFAR，边缘自适应训练窗 | **移植**到 3D（range/doppler/angle），保留"训练环 - 保护环"与自适应计数 | 2D 投影做法（我们保留 3D） |
| `src/+nrRadar/+sens/pick_peaks_nms.m` | CFAR 掩码上的局部极大 + 贪心 NMS | **移植**：本地极大 + 物理抑制半径贪心 | — |
| `src/+nrRadar/+sens/rdmDetection.m` | 检测流水线组织：CFAR→NMS→每峰 AoA（beamspace FFT / Bartlett）→DBSCAN→旁瓣抑制→几何定位→径向速度 | **组织参考**：同阶段划分；AoA 采用 FFT/Bartlett 扫描；DBSCAN 推迟（V1 单散点） | DBSCAN、4D 聚类、sidelobe 细节、elevation、beamforming 架构 |
| `src/+nrRadar/+sens/estimateRxAngle.m` | 波束/角度选择：ideal/nearest/scan | 支持我们"FFT/Bartlett、不用 MUSIC"的选择（该文件对 MUSIC 直接未实现） | ideal/nearest 模式（那是 oracle 辅助，禁止） |
| `src/+nrRadar/+util/scoreAssociationsPos.m` | 3D Mahalanobis + χ²(3) 门控 + Munkres(Hungarian) + TP/FN/FP 指标 | **两点采用**：(a) 检测-检测门控/匹配数学（换成 χ²(2) 位置门）；(b) C 域 track↔GT 匹配与指标模板 | 用 GT 的 3D 位置关联本身（只允许存在于 C 域） |
| `src/+nrRadar/+channel/getSigmaRCS.m` | 3GPP TR 38.901 车辆 RCS（方向性 + 对数正态） | **V1.1 备选**：V1 固定 RCS=10 m²；该文件作为后续起伏/方向性模型入口 | 角度表全量复刻（超 V1 范围） |
| `src/+nrRadar/+channel/getTargetPower.m` | 目标信道功率求和 | 概念参考（Σ 目标贡献） | 结构耦合 |
| `src/+nrRadar/+sens/suppressSidelobes.m`、`cluster_peaks_4d.m` | 旁瓣抑制、4D 聚类 | V1 不做（NMS 足够；DBSCAN 触发条件见 D07） | — |
| `examples/*/Output/detStats.csv`、`error.csv` | 检测统计与误差表模板 | **指标表模板参考**（TP/FN/FP/Pfa、位置/速度误差） | 场景参数 | 
| `docs/5G_NR_Radar_doc.pdf`、`prsConfig.txt` 等 | 5G NR PRS 完整标准链 | 不采用（我们是自定义冻结波形的最小实现） | 全部 |

结论：NIST 的价值集中在 **RD 处理顺序、CA-CFAR/NMS、AoA 扫描、关联/指标模板**；
其 MATLAB/5G Toolbox/波束成形/信道模型不进入本项目。

## 2. ISAC-Simulation-Framework-5G（jose）

- 仓库：`https://github.com/joseetenreiro/ISAC-Simulation-Framework-5G`
- 审计版本：commit `d6437858152a37a5f2b99e683fc74ca2f5c3ccae`（2026-07-29），MATLAB。
- 许可：README 明确 **"A dedicated repository licence has not yet been added"**，且部分文件来自 MathWorks 示例
  （保留其版权声明）。因此 **只做组织与概念参考，禁止复制任何源码**。

| 参考点 | 内容 | 采用方式 |
|---|---|---|
| 流水线组织 | Channel estimation → 静态分量抑制 → Range → AoA → Range-AoA map → 2D CFAR → DBSCAN → JIPDA tracking | 组织参考：前六步与我们一致；后两步换成 NMS + Hungarian/CV-KF |
| Range–AoA（无 Doppler）检测 | 说明只靠 Range–AoA + tracker 也能出速度 | 支持 V1"速度来自 KF"，也支持 AoA 采用 |
| 评测分层 | detection / clustering / tracking 三层指标；truth hit rate、previous-frame hit rate | 作为 C 域指标分层模板 |
| 部署/可视化/GUI/OSM 场景/视频导出 | 与本项目无关 | 不采用 |
| JIPDA、Sensor Fusion Tracker 库 | V1 范围外 | 不采用 |

## 3. Favarelli et al., ICC 2023

- E. Favarelli, E. Matricardi, L. Pucci, E. Paolini, W. Xu, A. Giorgetti,
  "Sensor Fusion and Extended Multi-Target Tracking in Joint Sensing and Communication Networks",
  IEEE ICC 2023, pp. 5737–5742（IEEE Xplore document 10279310；博洛尼亚大学 postprint 可公开获取）。
- 系统依据（摘要级事实）：多 BS 协同、经 **fusion center (FC)** 对 supervised area 内目标做
  **detect + track**；目标为 extended（车辆）模型。
- V1 采用：
  - 架构：多固定 BS → 融合中心 → 检测/跟踪（与我们 3-BS + 中央 fusion/tracker 一致）；
  - 状态定义：目标状态以位置+速度的笛卡尔运动状态为核心（与我们 `[x,y,vx,vy]` 一致）。
- V1 不采用：PF/MHT/随机矩阵等复杂多目标滤波（工单明确免除）；
  论文的蒙特卡洛 operating point 参数不复刻（其几何/波形与本项目不同）。
- 与 NIST 的区别：Favarelli 提供系统级依据（为什么要多 BS + FC），NIST 提供单 BS 处理链细节。

## 4. 本仓库内部复用

| 来源 | 复用内容 | 目标文件 | 等级 |
|---|---|---|---|
| `frontend/ofdm_echo.py` | 目标求和回波式、(100/r)² 幅度、FoV/range 可见性、噪声方差定义 | `frontend/sensing/simulator.py` | 公式移植 |
| `frontend/detector.py` | 3D CA-CFAR（torch 盒滤波）、NMS、旁瓣抑制、抛物线插值、越界几何剔除、逐检测质量字段 | `frontend/sensing/detector.py` | 算法移植+适配 |
| `frontend/tracker.py` | F/Q、门控更新、birth/confirm/coast/delete、`detected` vs `exists`、非 GT top-8 选择 | `frontend/tracking/cv_kf.py` | 状态机移植（改为位置-only 量测） |
| `frontend/station_geometry.py` | 3-BS 坐标/视轴/可见性 | 新模块直接 import | 直接 KEEP |
| `frontend/scene_manifest.py` | episodes/splits/origins/source_states | 不变 | 直接 KEEP |
| `frontend/pack_symbol_dataset.py` + `symbol_dataset.py` | 20 帧因果窗口、labels 分离、metadata 路由、白名单 loader、train-only normalization | `pack_shared_dataset.py` + `symbol_dataset.py` | 框架 KEEP_AND_MODIFY |
| `frontend/symbol_level_gpu.py` | 单目标 oracle 估计（作为 oracle-vs-sensing 对照） | 只读引用 | REFERENCE_ONLY |
| `data/f01d` 全部缓存 | ideal/oracle sensing baseline（T8 Δ_sensing 的分母来源） | 只读、隔离保存 | 冻结资产 |

## 5. 许可与合规结论

- NIST：可移植算法，需注明来源与改动。**计划移植前在 REBUILD_03 中登记具体文件与改动说明。**
- jose：无许可，**零代码复制**，只引用其 README 层面的组织与评测分层思想。
- Favarelli：公开论文，方法层面引用，无代码可用。
- 不引入任何第三方运行时依赖；新实现只使用项目已安装的 numpy/scipy/torch（与 ICCT 环境一致）。