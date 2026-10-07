# 三模型 XGBRanker 实验 C/D 说明

该目录保留三份最终交付文件：

1. `xgbranker_algorithm.py`：冻结参数的三模型walk-forward算法和C/D受控实验。
2. `xgbranker_results.xlsx`：历史选择过程、C/D比较、每次重训指标和股票级预测。
3. `README.md`：数据、方法、运行方式和结果边界。

## 最终模型

最终模型是三个 `rank:pairwise` XGBRanker 的等权排名集成。Feature、窗口、更新周期、树参数、随机种子和集成权重均在最终运行前冻结。

| 模型 | Features | Rolling window | Refit |
|---|---|---:|---:|
| Model 1 | `revenue_growth_yoy, operating_profitability, cashflow_to_assets, debt_to_assets, asset_growth_yoy` | 156周 | 26周 |
| Model 2 | `operating_profitability, debt_to_assets` | 52周 | 26周 |
| Model 3 | 与 Model 1 相同，正则化更强 | 52周 | 26周 |

所有Feature先在每周股票横截面转换为percentile rank。两项实验仅改变训练标签：

- 实验C直接使用连续`target`。
- 实验D在每个训练周内按连续`target`排名，生成0–4五档relevance label。

原始连续target定义为：

```text
target(t) = ln[AdjClose(t+1) / AdjClose(t)]
```

两项实验都不使用数据中的`target2`。实验D的五档标签由训练窗口内每个星期的连续`target`现场生成，不跨周分箱，也不会进入Feature。

## 预测值

XGBRanker 输出的是无单位的 `raw_ranking_score`，不是预测收益率。三个模型的原始分数分别转换为当周百分位，然后计算：

```text
ensemble_ranking_score
= (model_1_percentile + model_2_percentile + model_3_percentile) / 3
```

最终根据 `ensemble_ranking_score` 从高到低得到 `ensemble_predicted_rank`。真实值是连续下一周对数收益 `target`，真实排名是 `actual_rank`。

## 时间安排

- Model 2 和 Model 3 从 Week 52 开始独立预测。
- Model 1 需要完整156周训练窗口。
- 正式三模型集成从 Week 156 开始。
- 正式评估期为 Week 156–521，共366周。
- 更新周为 `156, 182, 208, ..., 520`，共15个集成批次。
- 正式集成期训练45个基础模型；包括短窗口早期阶段后，总共训练53个基础模型实例。

模型在每周收盘数据形成后训练，因此训练样本满足：

```text
label_end_week <= refit_week
```

在 Week `t` 训练时可以使用到 Week `t-1` 的 `target`，但绝不使用 Week `t` 或未来的 `target`。

## 实验C/D结果

两项实验均在Week 156–521的366周上评估，并统一对原始连续target计算RankIC：

| 实验 | 训练标签 | IC | RankIC | RankICIR | 正RankIC周比例 |
|---|---:|---:|---:|---:|---:|
| C | 连续target | 0.034119 | 0.031638 | 0.193076 | 59.56% |
| **D** | **周内0–4五档标签** | **0.034260** | **0.033109** | **0.199146** | **61.20%** |

按平均RankIC，实验D高于实验C：绝对增加`0.001471`，相对增加`4.65%`。但D的逐周RankIC胜率只有`50.27%`，配对t统计量为`0.588`，因此这次提升较小，不能解释为强统计证据。D在四段稳定性检验中有三段高于C。

原来的`RankIC=0.048960`仍来自Week 260–521，当前D扩展到Week 156–521后为`0.033109`。时间范围不同，所以旧值不能替代本次受控比较结果。

## 输出文件

算法默认把明细写到`outputs/xgbranker_cd_comparison/`。目录顶部包括：

- `experiment_summary.csv`：C/D集成整体指标和获胜实验。
- `paired_weekly_comparison.csv`：366周C/D RankIC及逐周差异。
- `paired_comparison_statistics.csv`：平均差异、相对变化、逐周胜率和配对t统计量。
- `component_experiment_comparison.csv`：两个实验的三个组件和集成结果。
- `fold_experiment_comparison.csv`：四个连续时期的稳定性比较。
- `comparison_manifest.json`：数据哈希、受控条件和获胜实验。

两个子目录`experiment_C_continuous/`和`experiment_D_relevance5/`分别包括：

- `frozen_configuration.json`：全部冻结参数和时间规则。
- `component_overall_evaluation.csv`：单模型与集成的整体指标。
- `refit_model_evaluation.csv`：53个基础模型实例的后续预测期指标。
- `ensemble_block_evaluation.csv`：15个集成更新批次的指标。
- `weekly_metrics.csv`：366周的单模型和集成 IC/RankIC。
- `stock_predictions.csv`：每周、每只股票的预测值、排名、真实 target 和真实排名。
- `fold_metrics.csv`：四个连续时间段的稳定性指标。
- `target_audit.csv`：target 对齐和使用情况。
- `latest_model_*.json`：Week 520 最新一次重训的三个 XGBoost 模型。
- `run_manifest.json`：数据哈希、软件版本和输出数量。

## 运行

在项目目录执行：

```bash
.venv/bin/python final_submission/xgbranker_algorithm.py
```

也可以指定输入、输出目录和并行线程数：

```bash
.venv/bin/python final_submission/xgbranker_algorithm.py \
  --input test_45_factors_weekly.csv \
  --output-dir outputs/xgbranker_cd_comparison \
  --n-jobs 4
```

## 结果边界

- 27,569行 target 可以在当前 CSV 内使用下一周 `adj_close` 复算；82行由上游提供并按用户决定保留。
- 这是冻结模型参数后的标签受控实验，但参数此前参考过同一份数据，因此不是全新的untouched test。
- ICIR 和 RankICIR 是周度均值除以周度样本标准差，没有年化。
- 项目只进行预测排名和 IC/RankIC 评估，没有进行投资组合或收益回测。
