# 组合回测

独立可运行的 15 策略回测引擎。保留报告参考实现中的等权 Top5/Top10、正分数过滤、过去 4/13 周非亏损过滤、排名加权，以及低波动、排名风险、前 10 保留缓冲、2/4 周调仓和 26 周趋势过滤。

```bash
python -m backtesting --predictions outputs/run/predictions.csv --output outputs/run/backtest
```

可选 `--strategies Top5 Top10 Top5_Rebalance4w`、`--costs-bps 0 10 25`、`--start YYYY-MM-DD --end YYYY-MM-DD`。起止日期均为信号日并包含边界。`--initial-delay-weeks N` 可验证调仓相位；等待周仍计入完整年化分母。

```python
from backtesting import run_backtest
result = run_backtest(predictions, prices, costs_bps=(0, 10, 25),
                      windows={"FullHistory": (None, None),
                               "Test": ("2025-09-01", "2026-08-28")})
result.save("outputs/run/backtest")
```

预测遵循 `model_evaluation` 的统一格式；支持多模型，每个模型和窗口从现金独立开始。价格输入可以是含 `date,ticker,adj_close` 的周表，或日期 × 股票的 DataFrame。应传入完整价格历史，以保留信号前的 4/13/26 周风险信息。不要以已删去缺因子的样本表代替完整价格表；`test_data_weekly.csv` 和 `test_45_factors_weekly.csv` 都包含这种筛选，不能作为完整价格来源。

CLI 默认调用 `data_processing.build_weekly_prices`，从 `data/clean_basic_data.csv` 按 `data/test_45_factors_weekly.csv` 的全局周日期精确提取价格，保存到本次输出目录的 `weekly_prices.csv`。不填充缺价、不跳到其他交易日。可通过 `--daily-prices` 和 `--calendar` 指定其他原始文件，或直接 `--prices path/to/complete_weekly_prices.csv` 使用自备完整周价格。

交易和审计规则：

- 只用信号日及之前数据选股。股票排序按分数降序、代码升序；不足 K 只的缺额留现金。
- 收益按相邻**全局价格日期**的复权收盘价计算。提供的 `target` 与 `label_end_date` 必须匹配；缺 target 可由价格补出。
- 末个价格日无下一日期，其预测逐行记录在 `audit.csv`，不计算已实现收益。其他股票缺下一周价格时先保留预测；若实际持仓涉及该股票，则明确报错，要求补全价格。不会先按未来价格可用性筛股，也不会跳到该股票更晚日期。
- 各模型历史内部无信号周持现金，清仓及恢复交易均收成本；2/4 周规则恢复信号时重设调仓计时。间隔周股票和现金权重自然漂移。
- 成交权重为目标权重与上期收益后漂移权重的绝对差之和，含买卖两侧；完全换仓可接近 200%。净收益采用原报告约定 `(1 - 成交权重 × bps / 10000) × (1 + 毛收益) - 1`，即在配置组合前按比例扣费。默认 0/10/25 bps；现金利息和无风险利率为 0，期末不另加清仓费用。
- 低波动和排名风险权重采用截至当周的完整 26 周周收益样本标准差，单股上限 35%；趋势为截至当周股票等权价格收益指数与 26 周均线比较。未来价格不进入这些输入。
- 基准采用价格表中有效相邻价格对的股票等权毛收益；缺失价格对不填充。默认完整价格池含 100 只股票，与 45 因子表/模型预测的 66 只股票可能不同，输出不应解释为严格匹配股票池的超额收益。用户可通过显式提供价格表确定基准股票池，但仍需完整覆盖持仓价格和风险历史。

输出 `summary.csv`、`weekly.csv`、`positions.csv`、`audit.csv` 和 `strategy_rules.json`。持仓不随费用情景改变，因此 positions 只按模型/窗口/策略输出一次。年化按 52 周，最大回撤为负值并计入初始净值 1。

来源为整合前 `报告章节材料/核心代码参考/` 的策略规则（原件在 `archive/original_source_before_integration.zip`）；原回测入口依赖缺失的 `run.py`、训练恢复脚本和特定历史压缩包，不能原样复现。本模块重建这些必要功能，保留原报告用于参考。完整价格输入与统一基准可能使风险/趋势/基准结果与原报告不同；原报告全部历史实验、图表和模型共识不由本模块自动重放。

验证：`python -m unittest discover -s backtesting/tests -v`。测试包含手算费用和漂移、窗口启动、缺信号重启、未来价格不影响当期全部策略权重、终端记录、缺价报错及标签核验。
