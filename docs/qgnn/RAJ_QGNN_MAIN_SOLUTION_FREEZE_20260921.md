# Raj Weighted Multi-J QGNN 主方案冻结（2026-09-21）

## 冻结结论
- QGNN 主架构冻结为 **Raj Weighted Multi-J QGNN（j=2+j=3）**。
- 核心公平基线为 **Matched Multi-J Johnson classical core**。
- 当前历史性能冠军使用可微复数态矢量/哈密顿量模拟，不冒充真实量子硬件或 PennyLane-native 性能结果。
- PennyLane 分支保留用于 quantum realizability / circuit validation，不强制替换当前性能主模型。

## 已核验核心证据
- 4096-sample exact matched setting，frozen test 共 2086 scenes。
- 两 seed 平均：ADE 0.502408 vs 0.517977，改善 **3.006%**。
- FDE 1.069034 vs 1.105594，改善 **3.307%**。
- J 1.036924 vs 1.070774，改善 **3.161%**。
- history-only 预定义高风险 top10：J 改善 **4.276%**。
- seed2026 / seed2027 test overall J 改善分别为 **3.154% / 3.169%**。
- 图核心参数量：126,535 vs 125,697，差约 **0.67%**。

## 边界
- 上述主结果来自 4096 个训练 scene windows，不是完整 15,802 full-train。
- 15,802 full-train 历史实验使用了更大的 classical Johnson（179,201 graph params），不能与 4096 exact-match 直接归因为“数据越多优势消失”。
- 当前 task 仍为旧 2s→2s legacy benchmark；任务重设计继续推进。
- 新主任务必须先证明 Graph > Self，再验证 Raj QGNN > matched classical Graph。

## 证据保存
- GitHub：关键 JSON、完整非-checkpoint 证据压缩包、checkpoint manifest、冻结说明。
- 本地/服务器正式仓库：额外保留 8 个 best.pt checkpoint。
- checkpoint 不上传 GitHub；SHA256 固化在 manifest 中。
