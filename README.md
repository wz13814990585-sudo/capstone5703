# 周频多因子项目：仅整合版

本版用于保存和调用组员原代码。11个原Python文件从整合前备份逐字节恢复，算法、因子公式、样本过滤、参数、选参、评估与组合规则均没有修改。新增代码仅位于 `integration/` 和根目录 `run.py`，负责校验、目录衔接、运行原脚本和保存日志。

## 目录与对应原文件

| 目录 | 内容 |
|---|---|
| `data_processing/` | 原四个行情、技术、基本面/宏观数据脚本，文件名与内容保留 |
| `single_factor_analysis/reference/` | 原方法DOCX、英文筛选表，以及历史资料中已有的中文筛选表 |
| `ridge/` | 原 `train_ridge_comparison.py` 与原说明 |
| `random_forest/` | 原 `random_forest.py` 与原说明 |
| `xgboost_model/` | 原 `train_xgboost_comparison(2).py`，保留括号文件名 |
| `xgboost_ranker/` | 原Ranker脚本与原说明 |
| `model_evaluation/` | 原指标所在脚本和已有结果的索引；不重新实现指标 |
| `backtesting/` | 原完整历史回测和策略两个参考脚本；原引擎缺失，暂不能完整运行 |
| `data/` | 原提供的五个CSV，未改写 |
| `reports/`、`outputs/reference/` | 原报告、图表、Ranker已有预测/模型与配置 |
| `integration/` | 外部调用层、原文件校验清单、检查工具 |
| `archive/` | 原备份、历史调优资料、之前修改版的归档，不参与正式入口 |
| `tests/` | 仅测试整合层，不改原脚本 |

## 安装与验证

建议使用Python 3.12。先在项目根目录创建环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 与原备份逐字节核对源码；并检查原资料/数据哈希
python run.py verify
# 检查原脚本所需依赖与缺失项
python run.py status --modules ridge random_forest xgboost xgboost_ranker backtest
# 调用原脚本自己的输入校验，不训练、不改变参数
python run.py check
```

也可用 `--python /path/to/python` 指定已有研究环境。原RF要求中文sheet和分类名称，因此默认直接使用历史资料中已有的中文工作簿，未改动原读取函数或英文原件。中文表与英文表的45个因子顺序及数值一致。

## 运行原算法

```bash
# 原四算法，完整原网格；RF仍执行原9个实验
python run.py run

# 分模块执行：仅使用原脚本支持的参数
python run.py run --modules ridge --mode both
python run.py run --modules random_forest
python run.py run --modules xgboost --mode both --xgboost-feature-set all
python run.py run --modules xgboost_ranker
```

没有缩减网格的 `--quick` 模式。`--mode` 只传给原Ridge/XGBoost，RF保持原有3种训练设计；`--xgboost-feature-set` 只影响原XGBoost。输入可用 `--input` 指定，RF工作簿可用 `--screening-file` 指定。

结果默认放在 `outputs/original_时间/`，也可指定新的空 `--output` 目录。外部入口将原脚本原样复制到本次输出的 `模块/workspace/` 后运行。需要按当前工作目录读写的原数据脚本在临时工作目录内使用输入副本，因子重建不会覆盖项目原CSV。调用过程不替换函数、不注入参数网格、不改写目标标签。

`run_manifest.json` 保存原脚本路径、命令、依赖可用性、原文件与输入SHA256、执行状态；`source_verification.json` 和 `source_verification_after.json` 核验运行前后原件一致。

原RF与回归XGBoost主要输出控制台表格，因此按原行为保存到 `random_forest.log`、`xgboost.log`。原Ridge的CSV保存在 `ridge/results/`；原Ranker的指标和预测在 `xgboost_ranker/results/data/`，模型在 `xgboost_ranker/results/models/`。本版不新增统一预测CSV或重新计算跨模型指标。

## 数据脚本

默认模型重放使用已有数据，不重新下载。需要主动刷新时，另安装 `requirements-download.txt`，再显式执行：

```bash
python run.py run --modules download
python run.py run --modules technical_factors
python run.py run --modules external_factors
```

外部因子脚本仍需原来的 `SEC_USER_AGENT`，并会访问SEC/FRED；重新下载数据可能与原始快照不同，不属于原结果复现。每个命令使用各自现有输入快照，不自动把新下载数据串入下一次运行；新输出必须由使用者明确选择。原35技术因子代码与6基本面因子代码均原样保留。

## 当前材料的实际缺口

- 单因子分析仅提供了方法文档和筛选结果，原Python实现没有提供。本版保留原资料，不自行实现新的筛选规则。
- 原回测说明明确它是核心代码参考，不是完整项目。缺少其引擎 `run.py`、`run_full_history.py`、`native_replay.py`、`restore_ridge.py`、对应native预测和原项目目录上下文。`python run.py run --modules backtest` 会记录blocked并返回非零状态，不以重写回测代替原代码。
- 原模型评估函数位于各算法原脚本内部；已提供原评估CSV和报告。独立评估/回测全部历史实验的原项目尚不完整。

因此，本版可以作为**原源码未改动的模块化整合作业**；要声称原评估/回测全项目完整复现，还需要补齐组员的原回测项目。说明书不会将缺失模块或未执行网格标成成功。

具体保留、归档与验证证据见 `INTEGRATION.md`。之前修改版保存在 `archive/modified_integration_20261008/`，其中的修复代码和新结果不纳入当前正式入口。

本版验证结果：11个原Python文件逐字节一致、4项整合层测试通过、四模型原生输入校验通过。原Ridge与XGBoost fixed模式的全部3因子组/原完整候选网格运行成功；原Ranker 22,430条预测与组员提供文件逐字节一致。RF完整九实验及回归完整滚动网格尚未重跑。详情见integration/verification_report.json。

requirements-tested.txt保存本次已有验证环境的版本，可用于重建相近环境；它不能代替组员未提供的原环境信息。submission_only.zip为可提交的仅整合包，包含原代码、原数据和原报告，排除之前修改版及历史调优大目录。
# capstone5703
