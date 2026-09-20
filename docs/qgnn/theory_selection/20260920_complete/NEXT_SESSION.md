# ICCT QGNN 接手顺序

先完整读取ICCT_QGNN_Architecture_Selection_Server_Report.md，再读selection_summary.json。核验qgnn/HEAD、Git状态、GPU进程；服务器优先，勿回退远端。

不再重新审计SinD/ISAC。从报告G0开始准备隔离原型，Primary TO-JQGNN，唯一Backup T-SQW-GNN。冻结单车encoder6624参数、QKV LoRA和motion codebook；Q与强TR-TGN使用同context和reader。须bank-checkpoint，不能用裸statevector宣称显存够。

本轮没有新模型实现或新ADE/FDE；大训练仍需用户后续授权及启动。报告里的阈值与运行时间情景是提出的实验门禁，不是测得优势。旧Raj full近打平，expadj已完成J1.5932085；不要重复等待。

本目录来源快照和access_manifest支持追查；delivery_manifest记录本轮结束保护hash核验。未commit/push。
