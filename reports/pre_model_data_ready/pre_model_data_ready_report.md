# ICCT Pre-Model Data Ready 门禁报告

## 1. 门禁总览

* **PRE-MODEL DATA READY**: **YES**
* **CURRENT-STAGE REPOSITORY CLEAN**: **YES**
* **Formal GNN Selected**: **NO (保留至下一阶段设计选型)**
* **Formal QGNN Selected**: **NO (保留至下一阶段设计选型)**

---

## 2. 门禁检查项逐项核验结果

| 序号 | 门禁检查项 | 标准与要求 | 状态 | 详细记录 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Automatum Dataset** | 规范化、8:1:1 时域隔离划分、20+20 样本有效性 | **PASS** | `split_manifest.json` 与各 split `sample_manifest.json` 全部校验通过 |
| 2 | **Frozen ISAC** | V2-FROZEN 配置、3-BS 确定性几何、无平滑/无漏检纯物理链 | **PASS** | 5 档单调性全部通过，指标与终审完全吻合 |
| 3 | **Production Sensing Cache** | train/val/test 唯一感知状态覆盖、逐状态数值等价、确定性打包 | **PASS** | 缺失 0，重复 0，非有限 0，max diff = 0.00e+00 |
| 4 | **Model-Agnostic Dataset** | `AutomatumPredictionDataset` 接口，输出 `[20, 8, 4]` 原始物理状态 | **PASS** | 契约对齐，零 sample-level 重复文件 |
| 5 | **GT-History Leakage** | 严格防范历史泄漏，Dataset 历史状态与 `samples.npz.history` 完全解耦 | **PASS** | 样本历史严重扰动后，Dataset 输出最大偏差 0.00e+00 |
| 6 | **Five-SNR Alignment** | 同一时间窗口在 5 档 SNR 下除 `history_state` 外所有元数据与未来完全一致 | **PASS** | 跨 SNR 样本完全对齐 |
| 7 | **Clock 3/29.97** | 真实时间步 `dt = 3 / 29.97 ≈ 0.1001001s`，杜绝硬编码 0.1 | **PASS** | 时间戳严格对齐 |
| 8 | **Generic DataLoader** | PyTorch DataLoader 支持 batching, shuffling, multi-worker | **PASS** | 批量迭代与张量转换通过 |
| 9 | **Current-Stage Cleanup** | 移除探索阶段过时代码与冗余报告，保持 active tree 权威精简 | **PASS** | 146 个中间过时文件已清理 |
| 10 | **Historical Tags Intact** | 历史阶段资产由不可变 Git tags 永久存档，当前 main 仅保留有效基线 | **PASS** | `stage0` (`ca1682a4...`) 与 `stage1` (`8adc24c5...`) tags 完好无损，工作树误恢复的 897 个历史文件已全数清除（遗留 0） |

---

## 3. 正式数据与缓存清单

* **数据划分与样本**:
  * `train`: 10,366 样本，769,400 历史行，49,765 unique states (15.46x 去重)
  * `val`: 893 样本，69,520 历史行，4,779 unique states (14.55x 去重)
  * `test`: 1,135 样本，95,440 历史行，5,727 unique states (16.66x 去重)
* **感知缓存路径**:
  * `data/automatum_t_crossing/isac/train/sensing_cache.npz` (`d38d2698...`)
  * `data/automatum_t_crossing/isac/val/sensing_cache.npz` (`b4edbb5c...`)
  * `data/automatum_t_crossing/isac/test/sensing_cache.npz` (`438e681b...`)
* **本地与服务器数据一致性**:
  * `LOCAL ↔ SERVER DATA CONSISTENT = YES` (100% SHA256 逐 bit 吻合)

---

## 4. 下一阶段交接备忘

当前仓库已具备进入正式模型研发所需的全部纯净物理数据与缓存。
下一阶段工作重心为：
1. 经典 GNN baseline 架构选型与图表示设计
2. QGNN 量子-经典混合架构设计
3. 动态交互图结构与边特征构建
4. 特征标准化与数值编码协议
5. 统一基准评测与对比实验
