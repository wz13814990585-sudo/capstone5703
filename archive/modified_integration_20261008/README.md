# 周频多因子研究：整合版

本项目将数据处理、单因子分析、Ridge、Random Forest、XGBoost 回归、XGBoostRanker、模型评估和投资组合回测整合为一个可运行的 Python 项目。各功能有独立目录，根目录 `run.py` 是统一入口。默认读取已有数据，执行过程无需联网。

## 目录

| 目录 / 文件 | 功能 |
|---|---|
| `data_processing/` | 原始行情清洗、35个技术因子、6个基本面因子、4个宏观因子、周频标签和数据审计 |
| `single_factor_analysis/` | 单因子 RankIC / ICIR、年度一致性、相关性筛选；`reference/` 保存原方法文档和筛选表 |
| `ridge/` | Ridge 训练、原选参规则、滚动/固定窗口预测 |
| `random_forest/` | Random Forest 三种窗口、候选筛选与预测 |
| `xgboost_model/` | XGBoost 回归；目录名避免遮蔽第三方 `xgboost` 包 |
| `xgboost_ranker/` | 冻结的三个 XGBRanker 及等权排名集成 |
| `model_training/` | 三个回归模型复用的因子定义、预处理、指标基础函数和预测导出 |
| `model_evaluation/` | 统一预测校验、IC、RankIC、ICIR、回归误差 |
| `backtesting/` | 15种投资组合策略、真实价格收益、换手成本、持仓与异常审计 |
| `data/` | 原始/清洗行情、35/45因子周频表、因子字典；原始数据未改写 |
| `outputs/` | 每次运行的训练日志、预测、指标、回测；`reference/` 为原已提供结果 |
| `reports/` | 原评估报告、图表及Ranker演示文稿 |
| `archive/` | 原始源码压缩备份、校验值、旧Ranker调优实验；不参与主流程 |
| `tests/` | 整合后的边界与回归验证 |
| `run.py` | 统一命令行入口 |

## 安装与运行

使用 Python 3.10 或以上。在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 一条命令：数据审计 → 单因子分析 → 四个算法 → 统一评估 → 回测
python run.py all
```

全量运行保留原来的候选网格、特征组合和训练窗口，会训练较多模型。建议先用快速模式检查环境与整条链路：

```bash
python run.py all --quick --feature-set 7
```

`--quick` 会缩减候选、树数或预测批次，输出有明确标记，仅用于软件验收，不能用来汇报模型表现。输出默认放在 `outputs/run_日期_时间/`，也可使用 `--output outputs/my_run` 指定新的空目录，避免混入上一次的文件。

## 单独运行模块

```bash
python run.py audit
python run.py factors --selection-end 2024-09-01
python run.py train --models ridge random_forest --feature-set 7 --mode fixed
python run.py evaluate --predictions outputs/my_run/predictions.csv
python run.py backtest --predictions outputs/my_run/predictions.csv --costs-bps 0 10 25

python -m ridge --help
python -m random_forest --help
python -m xgboost_model --help
python -m xgboost_ranker --help
python -m data_processing --help
python -m single_factor_analysis --help
```

默认训练数据为 `data/test_45_factors_weekly.csv`（66只股票、522周）。回测从 `data/clean_basic_data.csv` 精确提取这些周日期的真实价格，保存到本次运行的 `data_audit/weekly_prices.csv`。这样不会丢掉因技术因子缺失而被周频因子表删除的价格。默认价格与等权基准覆盖原行情表的100只股票，基准股票池与66只股票预测池不同，输出规则会注明；可以用 `--prices` 提供自备完整周价格和所需基准股票池。`--input`、`--daily-prices` 可替换上游输入。Ranker 保留原冻结实验对522周数据的要求。

数据下载仅由显式命令触发；已有数据的整合验收不会重复抓取。若需重新构建上游数据，另安装 `requirements-download.txt`，再参照 `data_processing/README.md`。SEC 下载需要有效的 `SEC_USER_AGENT`。重新下载结果会受数据源修订与股票样本变化影响。

## 输出与接口

`predictions.csv` 是各模型与评估/回测之间的统一接口：

| 列 | 含义 |
|---|---|
| `date`, `ticker` | 信号日期、股票代码 |
| `model` | 能区分算法、因子组合与训练模式的唯一模型名称 |
| `score` | 预测收益或排序分数 |
| `score_kind` | `return` 或 `rank`；Ranker不参与收益误差比较 |
| `target` | 下一周复权收盘价的对数收益 |
| `label_end_date` | 标签结束日；终端不可实现信号单独审计 |
| `stage` | 原模型阶段或历史滚动预测阶段 |
| `quick_run` | 是否为缩减训练的验收运行 |

每次完整运行输出包含：

- `run_manifest.json`：参数、输入审计、运行状态；异常会记录为 failed。
- `data_audit/`、`single_factor_analysis/`：数据审计与新单因子分析结果。
- 四个模型目录及对应 `.log`：原协议的选参、预测和诊断。
- `predictions.csv`：本次所有通过原选参条件的模型预测。
- `model_evaluation/summary.csv`、`summary_by_stage.csv` 与 `weekly_metrics.csv`：统一模型指标和阶段明细。
- `backtesting/summary.csv`、`weekly.csv`、`positions.csv`、`audit.csv`：策略指标、净值、持仓、费用和排除原因。

某个配置未通过原选参门槛时会记录状态，不会强行选出模型。如果所有候选均未通过，清单记录 `no_selected_models`，保存带表头的空预测表并跳过评估/回测。三种回归模型的原选择指标并不相同，训练目录中的候选指标保持原定义；横向比较应使用统一评估和回测输出。

## 保留的实验差异

| 模型 | 重训与选择规则 |
|---|---|
| Ridge | 原4周滚动或阶段固定训练；发展期筛选，门槛不通过可跳过 |
| Random Forest | 原3年/6年滚动或阶段固定；发展期候选短名单、验证期选1，保留原fallback |
| XGBoost | 原自然月滚动或阶段固定；发展期筛选 |
| XGBoostRanker | 原52/156周组件、26周更新、0–4 relevance、排名集成 |

整合没有把这四种协议强行改成一种。完整历史指标覆盖各自可用的预测历史，不能直接解释为完全相同时间段的公平竞赛；应结合日期范围和阶段分析。回测末周没有本地下一周价格时不计已实现收益，买入候选的选择也不能利用下一周是否有价格的信息。

原单因子材料是全样本探索性分析，原7因子与Ranker参数的历史选择也使用过同一份数据。新增单因子模块默认只用 `2024-09-01` 之前已结束的标签，并提供显式的全样本探索选项；新结果不会自动替换原冻结模型的7/11/45因子配置。整合后的历史结果不因此变成完全未参与筛选的测试结果。

## 整合和验证记录

```bash
python -m tests
```

已完成全部四个算法的快速端到端验证，结果位于 `outputs/verified_pipeline/`；这不是全量模型成绩。Ranker另按原完整参数重跑，22,430条集成与组件预测分数和提供的原始结果完全一致，见 `outputs/verification_ranker_full/reproduction_check.json`。所有回归模型的完整超参数网格尚未重新跑完，联网下载也未执行。

见 `INTEGRATION.md`：迁移/删除清单、原代码缺失项、修复范围、实测命令与结果。原源码可从 `archive/original_source_before_integration.zip` 恢复，压缩包内路径对应整合前的位置。
