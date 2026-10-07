# SafeCapital XGBRanker Rank IC Optimization

本项目只运行 XGBRanker 机器学习与 IC/Rank IC 评估，不运行投资策略或回测。

默认流程使用三阶段时间隔离：

```text
208周 Hyperparameter Train + 52周 Validation
106周 Feature / Fixed / Rolling 时间参数选择
104周最终不可触碰 Holdout
```

搜索内容：

- 20 个预注册 Feature 集，包括 Priority 子组合、经济类别、Technical 31、Stock 37 和 Stock 37 + Macro。
- Fixed windows：`all_history, 156, 208, 260, 312`。
- Rolling windows：`104, 156, 208, 260, 312`。
- Rolling refit intervals：`2, 4, 8, 13`。

主选择指标是研究选择期 Mean Rank IC，Rank ICIR 作 tie-break。最后 104 周只评估在查看 holdout 之前已冻结的 best-fixed 和 best-rolling，不用 holdout 结果重新选参。

完整实施规格见 [task.md](task.md)。

## 运行

```bash
.venv/bin/python run_xgbranker.py
```

查看可配置参数：

```bash
.venv/bin/python run_xgbranker.py --help
```

默认输出目录：`outputs/xgbranker_rankic_optimization/`。

## 分批 Feature 组合研究

在保持已冻结的 XGBRanker 参数、rolling 104 周窗口和 13 周 refit 不变的前提下，执行三批 Feature 搜索：

```bash
.venv/bin/python run_batched_feature_search.py
```

三批分别是 63 个基本面子集、70 个单因子增量组合和 38 个多因子增量组合。该流程仅使用 Week 312–417，不再评价已查看的最终 holdout。

默认输出目录：`outputs/xgbranker_batched_feature_search/`，其中：

- `combination_catalog.csv`：每个组合的公式与完整 Feature list。
- `batch_1_metrics.csv` 至 `batch_3_metrics.csv`：各批指标及相对基准改善。
- `all_combination_metrics.csv`：171 个组合的汇总结果。
- `robust_improvement_candidates.csv`：发现期和确认期都超过基准的组合。
- `recommended_candidate.json`：最佳新研究候选与保守决选结果。
- `batched_feature_search_report.md`：中文分批方法、组合与改善报告。

## 主要输出

- `feature_set_catalog.csv`：所有预注册 Feature 候选。
- `feature_screening_metrics.csv`：20 个 Feature 集在统一 baseline fixed/rolling 下的指标。
- `configuration_search_metrics.csv`：进入精细网格的 Feature 集及所有时间配置指标。
- `selected_configuration.json`：查看最终 holdout 之前已冻结的配置。
- `final_holdout_metrics.csv`：best-fixed 和 best-rolling 的最后 104 周指标。
- `final_holdout_predictions.csv`：最终 holdout 完整股票排名。
- `optimization_report.md`：中文实验和验收报告。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
