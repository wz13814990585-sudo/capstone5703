# XGBRanker 实验验收报告

## 数据

- 输入：`test_45_factors_weekly.csv`
- SHA-256：`26e234363e83f2eaefe0ba787b6909be0999f6d4405fbf6ef2e2c7cb8b918d97`
- 股票数：66
- 总周数：522
- 总样本数：27651
- 日期：2016-09-02 至 2026-08-28
- 每周股票数：30 至 66
- Target：`target`（下一周周末调整收盘价对数收益）
- Factor set：`priority_7`，7 个特征：`revenue_growth_yoy, ret_120d, cashflow_to_assets, ret_1d, ret_5d, close_location_value, body_range_ratio`
- 相邻且该股连续出现的可核验 target 行：27569；不匹配：0

## Warm-up 与冻结

- Warm-up：156 周；指定 Train 104 周 + Validation 52 周。
- Warm-up 实际训练使用 104 个完整周；训练信息 cutoff 为 week 105，所有训练行均满足 `label_end_week < cutoff`，且没有 validation row 进入 `fit()`。
- 候选配置数：4；所有候选见 `warmup_hyperparameter_results.csv`。
- 选择依据：Validation Mean Rank IC，Rank ICIR 作 tie-break。
- 最终候选：`pairwise_d4_bins5`；objective=`rank:pairwise`；relevance bins=5。
- Validation Rank IC=0.032030，Rank ICIR=0.178144。

## Rolling OOS 与泄漏防护

- Rolling window：156 周；每 4 周 refit；每周预测下一周。
- Warm-up 后使用 1 周 label-observability embargo，确保最后一个验证标签在首个 OOS 信号之前完全可知。
- Refit models：92；OOS prediction weeks：365。
- 每次训练从 trailing window 中再强制 `feature_week < refit_week` 且 `label_end_week < refit_week`。
- X 由集中 feature list 生成；`target/target2`、OHLCV、future/fwd/label 和 prediction 字段均由 hard-error guard 排除。
- Relevance 只在各周内按 `target` 分层；每周是独立 ranking group。

## 模型结果

- IC：0.014767
- ICIR：0.092769
- Rank IC：0.012385
- Rank ICIR：0.086316
- Positive IC Ratio：0.534247
- Positive Rank IC Ratio：0.531507

预测样例：

```text
      date ticker  ranking_score  predicted_rank  actual_return  actual_rank
2019-09-06   BF-B      -0.029794              43       0.030197           21
2019-09-06    BGS      -0.044992              45       0.077663            5
2019-09-06    CAG       0.013087              28       0.041352           18
2019-09-06   CALM       0.033770              16       0.061974           12
2019-09-06   CENT      -0.010386              35       0.056042           15
2019-09-06  CENTA       0.018021              25       0.083365            4
2019-09-06    CHD      -0.141020              52      -0.053029           48
2019-09-06   CHEF       0.164191               1       0.070383            8
```

## Strategy 与 Backtest

- Strategy：Top 5，`equal_weight`，`always_invest`，每 1 周换仓。
- Transaction cost：0.0 bps；slippage：0.0 bps。
- Benchmark：`equal_weight_universe`。
- Annualized Return：17.835931%
- Information Ratio：0.608863
- Average weekly turnover：0.842081
- Ending portfolio value：3.164571

## 数据限制

`target` 定义文件明确给出持有期间为当周周末到下周周末。因子 CSV 无法内部核验某股下周缺席的行以及最后一个样本周；管道未对这些 target 再 shift 或另行发明成交价假设。
