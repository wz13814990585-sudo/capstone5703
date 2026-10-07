# 整合记录

## 完成范围

将原独立脚本整合为按功能命名的Python包，并提供根入口 `run.py`。默认流程为已有数据审计、单因子分析、四算法训练、统一模型评估、15种策略回测。下载/刷新上游数据需要显式运行数据处理子命令。

公共代码抽到 `model_training/`，包括固定因子定义、训练期预处理、输出契约、JSON和部分指标基础函数。原算法选参逻辑有实质差异，因此其内部选择指标与调度保留，最终评价通过统一预测格式接入。

## 迁移与删除

| 原位置 | 现位置 / 处理 |
|---|---|
| `basic data/*.py` | `data_processing/`；去除旧副本 |
| `basic data/data/` | `data/`；仅移动，不改写CSV |
| `Single_Factor_Methodology_CN_EN.docx` / 筛选xlsx | `single_factor_analysis/reference/` |
| `train_ridge_comparison.py` | `ridge/train.py` + 公共训练工具；移除旧入口 |
| `random_forest.py` | `random_forest/train.py` + 公共训练工具；移除旧入口 |
| `train_xgboost_comparison(2).py` | `xgboost_model/train.py` + 公共训练工具；移除带括号旧入口 |
| `XGboost_ranker/xgboost_ranker.py` | `xgboost_ranker/train.py`；统一小写包名并移除旧副本 |
| `XGboost_ranker/data/` 与 `models/` | `outputs/reference/xgboost_ranker/` |
| `XGboost_ranker/presentation/` | `reports/presentations/` |
| `报告章节材料/` | `reports/reference_evaluation/`；报告与图表保留 |
| `报告章节材料/核心代码参考/` | 可用规则进入 `backtesting/`，缺失引擎功能重建；旧副本存入源码备份 |
| `history_xgbranker/` | `archive/history_xgbranker/`；独立历史调优实验保留，不纳入主流程 |
| `README.md`、`README(1).md` | 原文备份；根README替换为整合说明，移除多余旧副本 |
| `.DS_Store`、生成的Python缓存 | 清理 |

共移除12个被整合实现替代的文件（10个Python源文件和2个旧说明文件）。每个文件删除前都与源码压缩包中的原件逐字节核对，详见 `archive/cleanup_manifest.json`。历史实验不是重复代码的简单副本，因此归档保留而非直接销毁。

## 关键修复和保留边界

1. 修复默认数据路径、包内import、大小写目录和依赖缺失提示；输出集中到独立运行目录。
2. 原单因子材料无实现，依据方法文档补上RankIC、年度一致性与相关性筛选。默认仅使用标签结束日早于2024-09-01的样本；全样本探索可显式启用。默认分析不修改冻结模型因子。
3. 数据标签校验不会跨缺周连接；27,651条输入中27,569条可在同表直接核验，66条末周和16条缺下一周记录单列审计，可核验误差最大约4.54e-16。
4. 原回测参考依赖未提供的 `run.py`、历史预测压缩包和机器路径，不能直接运行。本次重建当前四模型所需评价与回测接口，保留15策略、持仓漂移、独立窗口与费用规则。
5. 原周因子表过滤过部分价格。新增 `build_weekly_prices` 从原日频清洗表按信号日期精确取价：52,197行、522周、100股票，仅ELF上市前3个日期缺价，不填补或使用未来价格筛股。修复WVVI/REED等因因子缺失造成的回测估值缺口。
6. 默认基准为传入完整价格表的可用股票等权收益，通常100只，与66只预测股票池不同；完整价格计算的风险/趋势指标也可能不同于旧报告。该定义记录在输出 `strategy_rules.json` 和回测说明中。
7. Ranking分数不按预测收益计算RMSE/MAE。统一评估排除显式缺少标签结束日期的终端行，原Ranker组件诊断保留原提供target口径。
8. 保留原实验窗口、标签可得性、候选网格、选择门槛和RF fallback；RF按原筛选工作簿列顺序训练，避免随机特征抽样因重排而改变。
9. 全部候选未通过为正常 `no_selected_models` 状态，不伪造成绩；所有独立模型及根入口拒绝覆盖非空结果目录。quick标记在标准预测及下游摘要中保留。
10. 旧报告中的更多历史调优模型、图表、PDF、共识模型重放不自动并入当前四算法主流程；历史资料保持可查。

## 验证证据

运行环境使用本机已有Python 3.12研究环境，未修改其安装内容。已核验依赖版本：numpy 2.5.3、pandas 2.2.3、scipy 1.18.1、scikit-learn 1.9.1、xgboost 3.4.1。新环境依照requirements安装。

```bash
python -m tests
python run.py all --quick --feature-set 7 --output outputs/verified_pipeline
python -m xgboost_ranker --output outputs/verification_ranker_full --n-jobs 2
```

- **29项测试通过**：数据缺周/坏标签/价格不填补、筛选区间隔离、训练预处理/选择边界/输出契约、IC/RankIC与排名处理、持仓漂移与成本、未来价格扰动、独立窗口、终端审计、失败清单/输出保护/quick继承。
- 活跃Python源文件语法解析通过。
- 回归模型真实数据check-only及quick多模式通过；共享预处理对7/45因子与原脚本逐元素一致，RF和XGBoost抽样预测与原代码一致。
- 完整统一快速流程状态为completed：84,277条预测、7个有效配置、7行统一模型摘要、19行按原stage摘要、315行回测摘要（7配置 × 15策略 × 3费用）。快速滚动只预测部分批次，空档按持现金处理，收益结果不能当作正式研究结果。
- Ranker按原完整参数重跑，22,430条集成分数和三个组件原始分数，与用户提供原CSV的最大绝对差异均为0；见 `outputs/verification_ranker_full/reproduction_check.json`。
- 原Ranker完整365周已实现回测通过，Top5/10bps年化9.7032%，与原报告9.70%显示精度一致。
- 原全样本探索筛选仍得到相同7因子；限制筛选截止后，overnight_gap替代ret_120d，输出独立保存，未修改冻结7因子配置。

全量三个回归模型的全部超参数网格没有重新跑完；数据下载、SEC/FRED刷新和完整日频因子重建未执行。代码已连通及验证，不把快速验收结果宣称为全量模型成绩。

## 后续复查

修复统一入口对不含 `model` 列的旧单模型预测CSV的兼容性。旧文件不含 `quick_run` 时，清单和摘要将该状态记为未知，而非默认认定为正式运行。新增回归验证后，当前30项测试全部通过。

发现仍需优先修订的实验边界：XGBoost发展期候选评分没有剔除 `label_end_date >= validation_start` 的记录。当前已有预测中，发展期最后信号为2024-08-30，其中62条标签到2024-09-06（首个验证信号日）才结束；这与该模型训练采用的 `label_end_date < fit_date` 信息可得性规则不一致。应先按标签结束日期剔除跨边界记录，再重新选参及评估。该调整可能改变选择结果，本次只记录问题，未改写原实验协议或已有结果。

横向模型比较还应增加统一测试时间与股票样本的结果表；默认全历史回测覆盖不同时间，且默认基准股票池为完整价格表的100只股票，预测池为66只。原全样本选择的7因子、Ranker的PositiveScore策略以及依赖版本锁定，也应在正式研究版本中明确处理。
