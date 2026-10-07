# Random Forest 回归

在项目根目录运行：

```bash
python -m random_forest --check-only
python -m random_forest --output outputs/random_forest
python -m random_forest --feature-set 7 --mode rolling_3y --quick --output outputs/rf_smoke
```

Python 入口为 `random_forest.train.main(argv=None)`。默认 CSV 为 `data/test_45_factors_weekly.csv`，支持 `--input`（旧名 `--data-file` 亦可）。默认读取 `single_factor_analysis/reference/Single_Factor_Screening_66_Stocks_EN.xlsx`，保持原工作簿的因子顺序；可用 `--screening-file` 指定其他工作簿，兼容英文版和原中文版。`--universe-file`、`--expected-stocks`、`--target-column`、`--target-type log|simple` 保留。

默认运行 rolling_3y、rolling_6y、fixed 三设计与三因子组；`--mode all|rolling_3y|rolling_6y|fixed`、`--feature-set all|7|7_macro|45` 可筛选。滚动每 4 个观察周重训，在各阶段边界重新计数。3 年版发展期始于 2019-09-01，6 年版和 fixed 始于 2022-09-01；验证与最终测试分别始于 2024-09-01、2025-09-01。训练标签允许在拟合日到期，评价标签不能晚于该阶段终点。

正式网格保留 18 组，100 棵树，depth 4/6/8、leaf 10/20/30、max_features sqrt/0.5，随机种子 5703。发展期先按筛选是否通过和“策略年化减基准年化”排名选 5 组；验证期选最终一组，保留原 fallback 规则。最终测试不再选参。

新增根 `predictions.csv`、`selection_status.csv`、`all_stage_metrics.csv`、`config.json`、`run_manifest.json`，子目录记录发展/验证评分、折表、最终重要度与选择依据。初始训练期指标仍为样本内诊断，仅计算最终选中候选，删去不影响选参的重复训练和大段终端排版。

`--quick` 改为单候选、8 棵树、每阶段最多两个滚动折，供连通性测试；fixed 仍覆盖整阶段。所有模式要求新的或空的输出目录。
