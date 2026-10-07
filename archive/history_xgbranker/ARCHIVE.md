# Historical tuning archive

该目录保存最终模型确定前的研究代码和结果，不是最终模型运行入口。

## code

- `run_xgbranker.py`：Feature、fixed/rolling窗口和更新周期搜索。
- `run_batched_feature_search.py`：分批Feature组合搜索。
- `run_algorithm_optimization.py`：XGBoost参数、时间配置和集成搜索。
- `run_label_comparison.py`：连续target与周内0–4 relevance label比较。
- `safecapital_xgbranker/`：历史搜索使用的公共模块。
- `tests/`：历史流程与最终模型的测试。

## data

- `fixed_rolling_comparison/`：fixed与rolling训练方式的早期比较。
- `rankic_optimization/`：Feature、窗口和更新周期搜索结果。
- `feature_search/`：分批Feature组合及指标。
- `algorithm_optimization/`：超参数、时间配置、联合候选和集成结果。
- `label_comparison/`：实验C/D及完整历史结果工作簿。
- `continuous_target_final_run/`：连续target版本的冻结运行结果。
- `legacy_priority7/`：早期Priority 7实验，仅作历史保留。
- `source_data/`：早期版本的周度数据文件。

最终可交付模型位于`../XGboost_ranker/`。
