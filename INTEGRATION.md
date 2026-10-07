# 仅整合交付记录

本次按用户确认的“保留原代码，用于复现组员结果或提交仅整合作业”要求重新制作。

## 原件完整性

原Python文件共11个：数据处理4个、Ridge1个、Random Forest1个、回归XGBoost1个、Ranker脚本及原__init__共2个、回测/策略参考2个。全部从 `archive/original_source_before_integration.zip` 的用户原路径恢复；没有使用备份中当时刚创建的 `data_processing/` 中间副本。

文件仅迁入按功能命名的目录，源文件名、编码、注释和字节内容保留。整合入口以原件SHA256和压缩包原件逐字节比对；运行前后还核对原CSV及参考资料。映射在 `integration/source_manifest.json`。

原数据、单因子方法文档、英文筛选表、历史调优结果、Ranker已有结果及评估报告均保留。已有中文筛选工作簿直接从 `archive/history_xgbranker/data/feature_search/` 复制，解决原RF对中文sheet名的输入格式要求；其因子顺序和数值经核对与英文版本一致，未转换或重新筛选。

## 仅新增的整合功能

- 根 `run.py`：统一入口，转交外部调用层。
- `integration/runner.py`：原件校验、依赖检查、隔离工作目录、原命令执行、控制台日志与运行清单。
- `integration/check_original.py`：直接调用原Ridge/Ranker既有加载校验函数；RF/XGBoost使用原 `--check-only`。
- `tests/`：验证原件一致、原脚本的隔离复制和原命令参数。
- 文档和依赖清单：解释真实可运行范围、原输出位置和缺失资料。

不再抽取原公共函数，不重算指标，不增加单因子实现，不修改模型参数或筛选边界，不修改原回测收益来源和基准。不把原脚本没有导出的逐股预测作为新导出功能补入。RF/XGBoost原本打印表格，整合层保留原控制台日志。

## 修改版的处理

之前的逻辑修复、公共训练工具、新单因子代码、统一评价/回测实现与相关测试、结果，全部移至 `archive/modified_integration_20261008/`。当前正式入口不导入它们。旧修改版留存便于追溯，但不属于作业正式代码。

## 缺失项的处理

原单因子Python实现未提供，保留原文档和结果。原回测完整引擎、恢复/重放脚本、native预测和原目录上下文未提供，保留原参考代码及报告，并让入口明确blocked。没有自行重建这些研究功能。补齐原项目后可继续按原件校验原则接入。

## 验证

执行 `python run.py verify` 查看完整校验；执行 `python run.py check` 调用原四模型输入校验；执行 `python -m tests` 测试外部衔接层。真实执行/结果对账证据保存在 `outputs/integration_only_verification/`。完整回归超参数网格的运行状态以日志为准，未执行的实验不标记为已复现。

已完成的实测：11个原Python文件逐字节一致，原数据与参考资料哈希一致；4项整合层验证通过；原四模型输入检查通过；原Ridge和回归XGBoost的fixed模式（全部3因子组、原完整候选网格）执行成功；原Ranker完整执行成功，其22,430条预测的全部列及CSV字节与原已提供文件完全一致。未执行RF全部九实验和回归滚动全网格，原完整回测引擎仍缺失。

证据：integration/verification_report.json，outputs/integration_only_verification/，outputs/original_ranker_reproduction/reproduction_check.json，outputs/original_regression_replay/。requirements-tested.txt记录本次已验证环境的精确版本；它不是组员原环境锁定文件。
