# ICCT QGNN 理论选型交接包

先读 `ICCT_QGNN_Architecture_Selection_Report.md`，再读 `selection_summary.json`。

已选Primary：TRC-QGNN；唯一Backup：TES-QGNN。选型已完成，尚未实现或训练新模型；ADE/FDE优势、实测训练时间与硬件编译仍待验证。

完整报告自包含：历史机制、文献与六族淘汰、数学结构、接口、基线、资源、反论证及kill criteria。`02_literature_and_candidates.md` 是独立文献/候选篇；`MINIMAL_ROADMAP.md` 是执行门禁；`complexity_tables.json` 是算术而不是profile结果。

`algebra_sanity.py/json` 是辅助CPU代数检查，不读取ICCT，不训练。该脚本用2个feature qubits验证配置结构恒等式；不是完整4/6feature-qubit Primary。

服务器目录：`/home/dell/YrM/ICCT/docs/qgnn/theory_selection/20260920/`。
服务器另有本轮初期保存的 `01_historical_audit.md` 与 `source_manifest.json`；它们记录直接读取路径和源文件hash。最终报告已经包含可独立接手所需的历史结论。

正式模型/数据/ISAC/配置不修改；本轮研究文件不自动commit/push。下一轮从ROADMAP R0开始，不把本报告当作已验收模型freeze。
