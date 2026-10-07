# Ridge 回归

在项目根目录运行：

```bash
python -m ridge --check-only
python -m ridge --input data/test_45_factors_weekly.csv --output outputs/ridge
python -m ridge --feature-set 7 --mode both --quick --output outputs/ridge_smoke
```

Python 入口为 `ridge.train.main(argv=None)`。默认运行 7、7+4、45 因子与 rolling/fixed 两种模式；可用 `--feature-set 7|7_macro|45` 和 `--mode rolling|fixed|both` 缩小范围。输出目录必须不存在或为空，输入不变。

保留原协议：Ridge 不惩罚截距，alpha 为 0.01、0.1、1、10、100、1000；训练期拟合填补、缩尾和标准化。滚动方式连续每 4 周重训，历史 3 年；fixed 在 2019-09-01、2024-09-01、2025-09-01 用已有历史重训并在阶段内冻结。训练标签必须严格早于拟合日，发展期选参标签必须早于 2024-09-01。

发展期要求 Rank IC > 0、未年化 Rank ICIR >= 0.05、正 Rank IC 比例 >= 50%；通过后按周超额收益复利年化、Rank IC、Rank ICIR 排序。没有候选通过时明确跳过，不使用候补参数。

根 `predictions.csv` 使用公共预测格式，`selection_status.csv`、`all_stage_metrics.csv`、`final_test_comparison.csv` 汇总实验，各模式子目录保留系数、预处理参数、折表、逐周指标和独立重算核验。该模型仍要求原 66 股、2016-09-02 至 2026-08-28 数据，核验目标与下一全市场周同股价格对应关系，不能直接套用新日期数据冒充复现。

`--quick` 只试两个 alpha，并限制滚动折数；fixed 仍覆盖完整阶段。仅供流程测试，不可当正式选参结果。
