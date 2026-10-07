# XGBoostRanker

最终三组件模型的实现位于 `train.py`，使用原来的五个/两个基本面因子、52/156周窗口、26周更新周期、0–4 relevance 标签与周内百分位等权集成。模型输出排名分数；统一评估不把它当作预测收益计算 RMSE。

```bash
python -m xgboost_ranker --check-only
python -m xgboost_ranker --output outputs/ranker_run
```

默认输入为项目 `data/test_45_factors_weekly.csv`。此冻结实验仍要求522周；自定义输入必须满足原实验的周数、特征和标签校验。`--quick` 仅将每个组件缩为8棵树，用于验证代码衔接。

输出保留原来的模型、组件指标、分块指标、分期指标和 `stock_predictions.csv`，新增统一接口 `predictions.csv`。`label_end_date` 缺失的最后一周无法用本地下一周价格回测，由回测模块单独审计。原组件诊断仍按提供的 target 计算。

历史参数选取使用过同一份历史数据，本模块的历史滚动结果不能当成全新、未参与选择的测试集。原始结果位于 `outputs/reference/xgboost_ranker/`；旧调优实验位于 `archive/history_xgbranker/`。
