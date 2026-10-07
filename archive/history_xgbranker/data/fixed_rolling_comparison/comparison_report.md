# XGBRanker 固定窗口 vs 滚动窗口、Feature 组合比较

## 实验定义

- 数据：`test_45_factors_weekly.csv`，66 只股票，522 周，27651 条记录。
- Target：连续下一周周末调整收盘价对数收益 `target`。
- Warm-up：156 周（104 train + 52 validation）。
- OOS 前保留 1 周 label-observability embargo。
- 滚动窗口：每次从最近 156 个候选周取严格可观测样本，每 4 周重训。
- 固定窗口：在首个 OOS 时点使用同一个首期候选窗口训练一次，之后不再 refit，但每周仍输入新特征做排序。
- 两种方法对每个 feature set 共用该组的 Warm-up 冻结超参和完全相同的 OOS 周。
- 本次不生成策略、仓位或回测结果。

## Feature 组合方法

将已确认的 Priority 7 预先分成三个有经济含义的 block：

```text
F = ['revenue_growth_yoy', 'cashflow_to_assets']
R = ['ret_120d', 'ret_1d', 'ret_5d']
P = ['close_location_value', 'body_range_ratio']
```

实验使用三个 block 的全部 7 个非空组合，而不是在 OOS 结果上搜索 127 个任意子集：

```text
           factor_set_name blocks                                                                                                             feature_list
             fundamental_2      F                                                                             ["revenue_growth_yoy", "cashflow_to_assets"]
                 returns_3      R                                                                                         ["ret_120d", "ret_1d", "ret_5d"]
            price_action_2      P                                                                             ["close_location_value", "body_range_ratio"]
     fundamental_returns_5    F+R                                             ["revenue_growth_yoy", "cashflow_to_assets", "ret_120d", "ret_1d", "ret_5d"]
fundamental_price_action_4    F+P                                 ["revenue_growth_yoy", "cashflow_to_assets", "close_location_value", "body_range_ratio"]
               technical_5    R+P                                             ["ret_120d", "ret_1d", "ret_5d", "close_location_value", "body_range_ratio"]
                priority_7  F+R+P ["revenue_growth_yoy", "ret_120d", "cashflow_to_assets", "ret_1d", "ret_5d", "close_location_value", "body_range_ratio"]
```

## 每个 Feature 组合的 OOS Metrics

```text
           factor_set_name training_mode  number_of_features        IC      ICIR   Rank_IC  Rank_ICIR  Positive_Rank_IC_Ratio  number_of_prediction_weeks
             fundamental_2         fixed                   2  0.007898  0.052698  0.006788   0.047538                0.523288                         365
             fundamental_2       rolling                   2  0.013719  0.083797  0.001018   0.006710                0.506849                         365
fundamental_price_action_4         fixed                   4  0.008778  0.053782  0.010603   0.079457                0.545205                         365
fundamental_price_action_4       rolling                   4  0.015835  0.089044  0.008673   0.058792                0.509589                         365
     fundamental_returns_5         fixed                   5  0.002014  0.012512 -0.001945  -0.014405                0.493151                         365
     fundamental_returns_5       rolling                   5  0.003761  0.021300  0.001031   0.006611                0.504110                         365
            price_action_2         fixed                   2 -0.003181 -0.021953  0.002682   0.020511                0.512329                         365
            price_action_2       rolling                   2  0.004127  0.028252  0.009652   0.067992                0.509589                         365
                priority_7         fixed                   7  0.007216  0.041705  0.007020   0.049581                0.531507                         365
                priority_7       rolling                   7  0.014767  0.092769  0.012385   0.086316                0.531507                         365
                 returns_3         fixed                   3 -0.004578 -0.027771 -0.007828  -0.054484                0.476712                         365
                 returns_3       rolling                   3  0.013099  0.084038  0.005992   0.041470                0.501370                         365
               technical_5         fixed                   5  0.000419  0.002547 -0.000637  -0.004781                0.515068                         365
               technical_5       rolling                   5  0.006412  0.042091  0.003756   0.026837                0.526027                         365
```

## Rolling 减 Fixed

正值表示滚动窗口更高，负值表示固定窗口更高。

```text
           factor_set_name  rolling_minus_fixed_IC  rolling_minus_fixed_ICIR  rolling_minus_fixed_Rank_IC  rolling_minus_fixed_Rank_ICIR
             fundamental_2                0.005821                  0.031099                    -0.005770                      -0.040829
fundamental_price_action_4                0.007057                  0.035262                    -0.001929                      -0.020666
     fundamental_returns_5                0.001747                  0.008788                     0.002976                       0.021016
            price_action_2                0.007309                  0.050205                     0.006970                       0.047481
                priority_7                0.007551                  0.051064                     0.005365                       0.036735
                 returns_3                0.017676                  0.111809                     0.013820                       0.095953
               technical_5                0.005993                  0.039543                     0.004393                       0.031618
```

## Warm-up 选中配置

```text
           factor_set_name      candidate_id     objective  relevance_bins   Rank_IC  Rank_ICIR
             fundamental_2 pairwise_d4_bins5 rank:pairwise               5 -0.029989  -0.188018
                 returns_3 pairwise_d4_bins5 rank:pairwise               5  0.028869   0.186567
            price_action_2     ndcg_d4_bins5     rank:ndcg               5  0.052773   0.356419
     fundamental_returns_5 pairwise_d4_bins5 rank:pairwise               5  0.009869   0.055717
fundamental_price_action_4 pairwise_d4_bins5 rank:pairwise               5 -0.014497  -0.082565
               technical_5 pairwise_d4_bins5 rank:pairwise               5  0.047689   0.306087
                priority_7 pairwise_d4_bins5 rank:pairwise               5  0.032030   0.178144
```

按 OOS Rank IC 描述性排序，最高的已观察组合为 `priority_7` + `rolling`，Rank IC=0.012385，Rank ICIR=0.086316。这只是比较结果，没有用来反向修改因子集或超参。

## 泄漏防护

- 所有训练行必须同时满足 `feature_week < refit_week` 和 `label_end_week < refit_week`。
- Feature 组合和 block 在 OOS 比较前预先固定。
- 每周独立生成 relevance label 和 ranking group。
- 超参只在 Warm-up 中选择；固定和滚动方法都不使用 OOS 结果调参。
