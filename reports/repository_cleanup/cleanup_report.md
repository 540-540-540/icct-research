# ICCT 当前阶段仓库精简与历史资产保护报告

## 1. 阶段边界与保护原则

* **冻结边界 Tag**: `stage1-meeting-freeze-20260915` (`8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7`)
* **历史基线 Tag**: `stage0-senior-original` (`ca1682a47e9129fbaf50dab4e2a6abc574b60098`)
* **保护原则**:
  1. `stage1-meeting-freeze-20260915` 之前的所有历史代码、报告、日志与文档资产全部原样保留。
  2. 此前分支整理中被删除的 897 个历史资产文件已通过 `git checkout stage1-meeting-freeze-20260915` **全部恢复**。
  3. 精简仅针对当前阶段（`stage1..HEAD`）在探索过程中产生的早期 smoke、初版标定、探索性探针及重复中间报告。

---

## 2. 历史资产保护审计

* **Stage 1 资产总数**: 2,457 个文件
* **已恢复历史文件 (Restored)**: **897 个文件**（覆盖 `results/`, `archive/`, `docs/` 以及历史方案文档）
* **允许保留更新的文件 (Still Changed)**: 3 个文件
  * `.gitignore`（维护当前大文件与生产数据排除策略）
  * `ENVIRONMENT.md`（维护服务器专用 ICCT 虚拟环境说明）
  * `README.md`（维护当前项目架构与分支说明）
* **历史资产保护状态**: **100% 完整保留 (YES)**

---

## 3. 当前阶段精简统计

* **精简前当前阶段文件数**: 206
* **已移除过时/中间/重复文件 (SUPERSEDED)**: **146 个**
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
* **保留权威文件 (KEEP)**:
  * **数据预处理**:
    * `tools/data_preprocessing/build_automatum_canonical.py`
    * `tools/data_preprocessing/validate_automatum_canonical.py`
    * `tools/data_preprocessing/build_automatum_splits.py`
    * `tools/data_preprocessing/validate_automatum_splits.py`
    * `tools/data_preprocessing/build_automatum_samples.py`
    * `tools/data_preprocessing/validate_automatum_samples.py`
  * **Frozen ISAC 核心与生产 Cache**:
    * `frontend/controlled_isac/` (`__init__.py`, `automatum_frontend.py`, `automatum_measurement.py`, `cross_bs_association.py`, `measurement.py`, `state_quality.py`, `temporal_association.py`, `velocity_fusion.py`)
    * `configs/automatum_controlled_isac.json` (`AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN`)
    * `experiments/automatum_controlled_isac_final/` (8 个最终验收与复现脚本)
    * `tools/data_preprocessing/build_isac_production_cache.py`
    * `tools/data_preprocessing/validate_isac_production_cache.py`
  * **模型无关预测 Dataset 适配器**:
    * `frontend/automatum_prediction_dataset.py` (`AutomatumPredictionDataset`)
    * `configs/automatum_prediction.json` (`AUTOMATUM-PREDICTION-DATA-FROZEN`)
    * `tools/data_preprocessing/validate_prediction_dataset.py`
  * **权威报告**:
    * `reports/isac_final/` (终审指标、候选分布与冻结清单)
    * `reports/isac_production_cache/` (缓存构建报告、各 split manifest 与验证汇总)
    * `reports/data_preprocessing/` (split / sample 权威 manifest)
    * `reports/repository_cleanup/` (本精简清单与方案)
    * `reports/pre_model_data_ready/` (进入模型前门禁报告)
  * **状态说明**:
    * `docs/current_stage_status.md`

---

## 4. 结论

当前阶段所有探索性冗余文件已完全从 active tree 中移除，Git 历史完整保存其提交记录。
历史阶段（stage0、stage1）资产 100% 保持原样，未发生破坏。
项目代码库恢复清晰、精简、权威状态。
