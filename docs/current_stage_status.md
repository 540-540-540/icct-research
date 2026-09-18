# ICCT 当前阶段状态说明与交接清单

## 1. 当前阶段成果状态

* **Automatum 数据集预处理**: **COMPLETE**
  * Canonical 数据规范化完成（`29.97 Hz / stride 3 -> 9.99 Hz / dt = 0.1001001s`）
  * 8:1:1 严格时域划分完成（40 帧保护隔离，跨场景车辆无重叠）
  * 20+20 样本定义打包完成（严格过滤 2 ≤ N ≤ 8）
  * 样本统计：`train: 10,366`，`val: 893`，`test: 1,135`

* **Frozen ISAC Route-B**: **FROZEN**
  * Revision: `AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN` (`status = FROZEN`)
  * Config: `configs/automatum_controlled_isac.json`
  * 架构：3-BS 确定性几何（Scene 0 BS2 boresight 64.55° 修正，Scene 1 原样保留），SNR-controlled 无检测/无关联/无平滑纯物理链路
  * 五档 SNR 误差严格单调且通过全部目标带：
    * +10 dB: ~0.131 m / 0.159 m/s
    *  +5 dB: ~0.233 m / 0.283 m/s
    *   0 dB: ~0.414 m / 0.503 m/s
    *  -5 dB: ~0.736 m / 0.894 m/s
    * -10 dB: ~1.309 m / 1.591 m/s

* **Production ISAC Sensing Cache**: **COMPLETE**
  * 路径：`data/automatum_t_crossing/isac/{train,val,test}/sensing_cache.npz`
  * 状态规模：train 49,765，val 4,779，test 5,727 唯一感知状态
  * 校验：缺失 0，重复 0，非有限值 0，Frontend 数值逐 bit 匹配，确定性压缩哈希复现通过

* **Model-Agnostic Prediction Dataset**: **COMPLETE**
  * 模块：`frontend/automatum_prediction_dataset.py` (`AutomatumPredictionDataset`)
  * 契约：`configs/automatum_prediction.json` (`AUTOMATUM-PREDICTION-DATA-FROZEN`)
  * 输入输出：`history_state [20, 8, 4]` (来自 ISAC cache)，`future_state [20, 8, 4]` (来自 Automatum GT)，`vehicle_mask [8]`，`history_timestamp [20]`
  * 真实时间轴：`dt = 3 / 29.97 s`，杜绝硬编码 0.1
  * 历史泄漏测试：PASS（样本 history 扰动不影响 Dataset，感知缓存为唯一历史来源）
  * 通用 DataLoader 烟测：PASS（支持 batching, shuffling, multi-worker）

* **当前阶段仓库精简**: **COMPLETE**
  * 历史资产保护：历史 Stage0 / Stage1 由不可变 Git tags（`stage0-senior-original` 与 `stage1-meeting-freeze-20260915`）永久保存；当前 `main` 工作树仅维护当前最新有效工作版本。
  * 纠偏清理：上一轮误恢复到工作树的 897 个历史归档文件已全部自当前工作树移除（0 遗留）。
  * 中间文件精简：当前阶段 146 个过时/探索性中间文件（early smoke, calib v1, temporary audits）彻底移除。
  * 仅保留权威脚本、配置与终审报告。

---

## 2. 下一阶段模型设计边界

* **Formal Classical GNN Baseline**: **PENDING (NOT SELECTED)**
* **Formal QGNN 主模型**: **PENDING (NOT SELECTED)**

> **注意**: 现有仓库中保留的早期/历史模型代码（`prediction/`、`code/` 等）属于上一历史阶段资产，**不作为下一阶段正式模型实现**。当前阶段不包含任何特定于模型的特征工程（如 7-D 边特征、图邻接阈值、角度编码、量子尺度、mean/std 归一化等），所有模型架构选型、边特征设计与公平对比协议将在下一阶段统一进行。

---

## 3. 下一步工作内容

1. Classical GNN baseline 架构选型与图表示设计
2. QGNN 量子-经典混合架构选型与参数化量子电路设计
3. 动态时空交互图结构与边特征设计
4. 模型输入 Normalization 与特征编码规范
5. 统一训练门禁、损失函数与公平对比协议
6. 正式全量多 SNR 对比训练与实验评估
