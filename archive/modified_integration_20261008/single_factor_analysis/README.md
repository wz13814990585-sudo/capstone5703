# 单因子分析

原材料只有 `reference/` 中的中英文方法说明和结果表，没有对应 Python 实现。本模块为依该说明新增的可重复计算代码，输出独立 CSV 与 JSON，保留参考文档原件。

```bash
python -m single_factor_analysis --output outputs/single_factor_analysis
python -m single_factor_analysis --selection-end 2024-09-01 --output outputs/screening_before_validation
python -m single_factor_analysis --exploratory-full-sample --output outputs/screening_exploratory
```

默认同时限制信号日期、标签到期日期早于 `2024-09-01`，避免因子筛选读取验证期和最终测试期收益。全样本模式仅用于探索。结果不会自动修改模型的既有 7 因子组合：原组合来自截至 2026 年的全样本探索，保留它复现原模型时仍存在因子选择偏差，不能将其宣称为完全独立的最终样本外验证。

方法与阈值沿用说明：周度横截面 Spearman Rank IC，ICIR 使用样本标准差且不年化；每周至少 20 只股票、5 个不同因子值、2 个不同收益值；每个可评价年度至少 40 周；覆盖率至少 80%、至少 104 个有效周和 3 个可评价年份、绝对平均 IC 至少 0.01、绝对 ICIR 至少 0.08、年度方向一致性至少 60%。按绝对平均 IC 降序筛选，以平均周度 Spearman 相关系数的绝对值达到 0.85 为去冗余条件；负向因子保留原值。

标签统一经 `data_processing.load_weekly_panel` 核验。原报告只排除最后一周 66 行；本实现还排除缺少下一日历周同股票价格的 16 行，因此全样本探索指标可能与原报告略有不同。重复列判断和筛选均只使用允许的筛选区间。

输出：`screening_results.csv`、`weekly_rank_ic.csv`、`annual_rank_ic.csv`、`candidate_correlations.csv`、`unverifiable_labels.csv` 和 `screening_manifest.json`。Manifest 包括输入哈希、实际筛选边界、阈值、选择结果以及原报告差异说明。少于门槛或没有优先因子是有效结果，不补造因子。
