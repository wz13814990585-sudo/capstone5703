# 原模型评估资料

原模型的IC、RankIC、收益与误差计算位于各自原算法脚本中，本版未把函数拆出或改写。

- Ridge：`../ridge/train_ridge_comparison.py` 的 weekly/metrics 等函数。
- Random Forest：`../random_forest/random_forest.py` 的 weekly/summary 等函数。
- XGBoost：`../xgboost_model/train_xgboost_comparison(2).py` 的 stage_metrics/display_result 等函数。
- Ranker：`../xgboost_ranker/xgboost_ranker.py` 的 weekly_metrics/summarize 等函数。
- 已提供原评估表与报告：`../reports/reference_evaluation/`。

原完整历史评估引擎没有提供，不能把当前材料称为可独立运行的完整评估项目。正式复现按原模型输出与原报告对账，不使用之前修改版的统一指标实现。
