# XGBoost 回归

目录命名为 `xgboost_model`，避免遮蔽第三方 `xgboost` 包。

```bash
python -m xgboost_model --check-only
python -m xgboost_model --output outputs/xgboost
python -m xgboost_model --feature-set 7 --mode both --quick --output outputs/xgb_smoke
```

Python 入口为 `xgboost_model.train.main(argv=None)`。默认输入 `data/test_45_factors_weekly.csv`。支持 `--mode rolling|fixed|both`、`--feature-set all|7|7_macro|45`、`--rolling-years`，其默认值为 3 年。

保留原树参数和 12 组网格：depth 2/3、min_child_weight 5/20、trees 50/150/300；学习率 0.03、subsample/colsample_bytree 0.8、reg_lambda 10、随机种子 42。预处理只从训练窗口估计填补和缩尾，不标准化。

滚动模式按自然月且阶段分开重训，fixed 按阶段起点重训。起点是 2019-09-01、2024-09-01、2025-09-01 之后的首个有效信号日；标签实现日取同股下一条相距 5 至 14 天记录，训练标签必须严格早于拟合日。原发展期末周选参口径保留，尚未与 Ridge 的标签到期截断规则统一。

发展期门槛为 Rank IC > 0、Rank ICIR >= 0.05、正 Rank IC >= 50%，按周超额收益复利年化、Rank IC、Rank ICIR 选优；不通过则记录跳过。候选只训练发展期，选定参数后训练其他阶段，移除原实现对每个候选提前重复训练验证/测试的无用计算，不改变选参值。

根输出包含公共 `predictions.csv`、选择状态、阶段指标、配置与清单；模式/因子子目录记录候选评分、选定参数、预测和训练范围。`--quick` 使用单候选、8 棵树和各阶段前两个自然月折，仅用于测试；fixed 仍覆盖全阶段。输出目录必须不存在或为空。
