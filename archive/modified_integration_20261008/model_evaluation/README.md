# 模型评估

统一四种模型的逐周截面评估，替代分散脚本中的重复指标计算。

```bash
python -m model_evaluation --predictions outputs/run/predictions.csv --output outputs/run/evaluation
```

输入为 `date,ticker,score,target,label_end_date,model,score_kind`。`target` 是下一全局周复权收盘到收盘的**对数收益**。兼容旧分数列 `pred/prediction/ranking_score/ensemble_ranking_score` 和标签列 `actual_return`。同一 `model/date/ticker` 只允许一条记录；不同超参/因子组应使用不同 `model` 名称。

可选 `target_type` 为 `log`（默认）或 `simple`。显式标为 `simple` 的标签和回归分数通过 `log1p` 统一成对数收益；排名分数保持不变。低于或等于 -100% 的简单收益会报错，不能转换成有限对数收益。未知标签类型同样报错。

`score_kind=return` 表示收益预测，`rank` 表示只用于排序的分数。后者不计算 MAE/RMSE，避免把排名百分位当收益率。旧预测缺少该列时，从原分数字段或含 `ranker/consensus` 的模型名推断；有歧义时用 `--score-kind rank` 明确指定。

IC/Rank IC 分别为每周 Pearson/Spearman 相关；Spearman 并列使用平均名次。先逐周计算，再对有效周等权平均。单只股票、常数分数或常数标签的周相关为 NaN，不能用 0 代替。ICIR 为周相关均值除以样本标准差，不年化。回归模型 MAE/RMSE 为可用股票行的总体误差。

显式提供 `label_end_date` 时，其缺失行视为未实现标签并排除；`target` 缺失同样排除。旧文件没有结束日期时，只能依赖其提供的标签，无法独立断定末周收益是否真实可用。要校验标签与真实价格的一致性，应通过 `backtesting.run_backtest`，或先由数据处理模块生成已验证的标签结束日期。

API：

```python
from model_evaluation import evaluate_predictions
summary, weekly = evaluate_predictions(predictions)
```

输出 `summary.csv` 与 `weekly_metrics.csv`，包含各模型覆盖、可用行/周数与缺失计数。验证：`python -m unittest discover -s model_evaluation/tests -v`。
