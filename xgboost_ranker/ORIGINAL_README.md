# XGBoost Ranker模块

本模块使用三个XGBRanker预测股票下一周的横截面排名。模型只输出排名及IC/RankIC指标，不进行投资组合回测。

## 调优过程

历史调优代码和结果保存在`../history_fine/`。

1. 比较XGBoost树深、学习率、树数量、正则化和ranking objective。
2. 比较不同基本面Feature组合。
3. 比较52、78、104、130、156、208周及更长训练窗口。
4. 比较2、4、8、13、26周更新周期。
5. 比较单模型和多个模型的周内百分位等权集成。
6. 在相同Week 156–521区间比较连续target与0–4 relevance label，最终选择平均RankIC更高的0–4 relevance label。

## 最终模型

三个模型均使用`rank:pairwise`，每26周重新训练。Feature先转换为当周横截面percentile rank。

| 模型 | Features | 窗口 | 主要参数 |
|---|---|---:|---|
| Model 1 | revenue growth、operating profitability、cash flow/assets、debt/assets、asset growth | 156周 | depth=2, lr=0.025, trees=360, lambda=3 |
| Model 2 | operating profitability、debt/assets | 52周 | depth=3, lr=0.04, trees=240, lambda=2 |
| Model 3 | 与Model 1相同 | 52周 | depth=3, lr=0.03, trees=280, alpha=0.1, lambda=5, gamma=0.05 |

每个训练周内，根据连续`target`的横截面排名生成0–4 relevance label。数据中的`target2`不参与训练。

三个模型的预测分数先转换为周内百分位，再等权平均：

```text
ensemble_ranking_score
= (model_1_percentile + model_2_percentile + model_3_percentile) / 3
```

`ensemble_ranking_score`是排名分数，不是预测收益率。最终评估仍使用连续下一周对数收益：

```text
target = ln(AdjClose(t+1) / AdjClose(t))
```

## 最终结果

评估范围为Week 156–521，共366周。

| 指标 | 结果 |
|---|---:|
| IC | 0.034260 |
| ICIR | 0.203984 |
| RankIC | 0.033109 |
| RankICIR | 0.199146 |
| 正RankIC周比例 | 61.20% |
| 逐股预测记录 | 22,430 |
| 训练模型实例 | 53 |
| 集成更新批次 | 15 |

历史`RankIC=0.048960`来自Week 260–521模型选择区间，不是本模块的最终全时期结果。

## 文件

- `xgboost_ranker.py`：最终训练、滚动预测和评估入口。
- `models/`：最后一次更新保存的三个XGBoost模型。
- `data/overall_metrics.csv`：最终集成整体指标。
- `data/component_metrics.csv`：三个组件及集成指标。
- `data/weekly_metrics.csv`：366周的IC和RankIC。
- `data/refit_model_metrics.csv`：53个训练模型实例的后续预测指标。
- `data/ensemble_block_metrics.csv`：15个集成更新区间指标。
- `data/fold_metrics.csv`：四段时间稳定性指标。
- `data/stock_predictions.csv`：每周每只股票的预测分数、预测排名、连续真实target和真实排名。
- `data/model_configuration.json`：最终Feature和参数。
- `data/data_audit.csv`、`data/run_manifest.json`：数据校验和运行环境。

## 运行

在项目根目录执行：

```bash
.venv/bin/python XGboost_ranker/xgboost_ranker.py
```

使用其他数据时：

```bash
python XGboost_ranker/xgboost_ranker.py --input /path/to/weekly_data.csv
```

输入数据必须包含`date`、`ticker`、`adj_close`、连续`target`以及最终模型使用的五个基本面Feature。

## 结果边界

参数选择曾参考同一份历史数据，因此该结果不是全新的untouched test。模块不包含交易成本、持仓构建或收益回测。
