# ICCT 当前阶段仓库精简与历史快照保护报告

## 1. 历史保护与分支边界原则

* **核心定位原则**:
  * **Git Tag = 历史阶段永久不可变快照**：历史阶段全部代码、文档、日志与数据配置由不可变 Git tags 永久存档。
  * **Git Main = 当前最新有效工作版本**：当前工作树仅维护当前有效的工作流水线，有意不再包含历史快照中的废弃与过渡文件。
* **历史基线 Tags**:
  * `stage0-senior-original` (`ca1682a47e9129fbaf50dab4e2a6abc574b60098`)
  * `stage1-meeting-freeze-20260915` (`8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7`)
* **历史保护核验结果**:
  * `stage0` commit SHA 严格一致：`PASS`
  * `stage1` commit SHA 严格一致：`PASS`
  * 历史 Tag 不移动、不重建、不修改。

---

## 2. 误恢复历史文件纠偏清除审计

在上一轮操作中，曾将 Stage 1 历史分支中非当前阶段的 897 个历史归档文件误恢复至工作树中。本轮根据精确集合定义完成彻底纠偏：

* **误恢复文件定义**:
  $$\text{Unintended} = (\text{Stage1 Tree} \setminus \text{Pre-Restore Tree}) \cap \text{HEAD Tree}$$
  * `Stage1 Tree` (`stage1-meeting-freeze-20260915`): 2,457 文件
  * `Pre-Restore Tree` (`e3c385be...`): 1,912 文件
  * `HEAD Tree` (`0a3d3215...`): 2,673 文件
* **检测到的误恢复历史文件**: **897 个**
* **本轮自工作树删除数量**: **897 个**（通过 `git rm -f` 批量删除）
* **当前工作树残留误恢复历史文件**: **0 个**
* **重点清除的根目录过期历史文档**:
  * `QGNN预测优势优化修订稿.md`
  * `ROADMAP.md`
  * `amendment_log.md`
  * `研究方案书.md`
  * `跨车连接隔离实验方案_审阅稿.md`
  *(以上文档完整留存在 `stage1-meeting-freeze-20260915` tag 快照中)*

---

## 3. 当前阶段中间文件精简统计

当前阶段（`stage1..HEAD`）在探索与验证过程中产生的早期中间文件已全数清理：

* **已移除过时/中间/探索文件 (SUPERSEDED)**: **146 个**
  * `experiments/automatum_isac_smoke_01/` (4 个)
  * `experiments/automatum_controlled_isac_calib_v1/` (5 个)
  * `experiments/automatum_controlled_isac_final_audit/` (7 个)
  * `reports/isac_smoke/` (19 个)
  * `reports/isac_calibration/` (13 个)
  * `reports/isac_final_audit/` (24 个)
  * `reports/controlled_isac/` (12 个)
  * `reports/isac_adaptation/` (2 个)
  * `reports/data_audit/` (16 个)
  * `reports/data_cleaning/` (7 个)
  * `reports/sensing_audit/` (15 个)
  * `reports/sensing_design/` (5 个)
  * `tools/data_audit/` (1 个)
  * `tools/data_cleaning/` (4 个)
  * `tools/isac_adaptation/` (1 个)
  * `configs/automatum_isac_smoke.json` (1 个)
  * `frontend/automatum_scene_source.py` (1 个)
  * `frontend/fusion/`, `frontend/sensing/`, `frontend/tracking/` (9 个)

* **保留的当前有效权威资产 (KEEP)**:
  * **数据预处理**:
    * `tools/data_preprocessing/build_automatum_canonical.py`
    * `tools/data_preprocessing/validate_automatum_canonical.py`
    * `tools/data_preprocessing/build_automatum_splits.py`
    * `tools/data_preprocessing/validate_automatum_splits.py`
    * `tools/data_preprocessing/build_automatum_samples.py`
    * `tools/data_preprocessing/validate_automatum_samples.py`
  * **Frozen ISAC 纯物理感知与生产 Cache**:
    * `frontend/controlled_isac/` (`__init__.py`, `automatum_frontend.py`, `automatum_measurement.py`, `cross_bs_association.py`, `measurement.py`, `state_quality.py`, `temporal_association.py`, `velocity_fusion.py`)
    * `configs/automatum_controlled_isac.json` (`AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN`)
    * `experiments/automatum_controlled_isac_final/` (8 个权威复现脚本)
    * `tools/data_preprocessing/build_isac_production_cache.py`
    * `tools/data_preprocessing/validate_isac_production_cache.py`
  * **下游模型无关预测 Dataset 适配器**:
    * `frontend/automatum_prediction_dataset.py` (`AutomatumPredictionDataset`)
    * `configs/automatum_prediction.json` (`AUTOMATUM-PREDICTION-DATA-FROZEN`)
    * `tools/data_preprocessing/validate_prediction_dataset.py`
  * **权威终审与门禁报告**:
    * `reports/isac_final/` (ISAC 冻结指标与闭环审查)
    * `reports/isac_production_cache/` (缓存构建报告与适配器验证)
    * `reports/data_preprocessing/` (split / sample 权威 manifest)
    * `reports/repository_cleanup/` (精简清单与审计报告)
    * `reports/pre_model_data_ready/` (Pre-model 门禁总审)
  * **状态交接文档**:
    * `docs/current_stage_status.md`

---

## 4. 结论

1. 历史资产（Stage 0, Stage 1）由不可变 Git tags 永久完整保全；
2. 工作树中误恢复的 897 个历史文件已全数清除（残留 0）；
3. 当前阶段 146 个探索性中间文件已全数清理；
4. 当前 active 工作树恢复纯净、权威、精简，全面准备就绪进入下游模型设计阶段。
