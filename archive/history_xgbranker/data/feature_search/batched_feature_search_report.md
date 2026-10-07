# XGBRanker 分批 Feature 组合优化报告

## 结论

本轮在原选择期 Week 312–417 内新增了 171 个组合，保持 XGBRanker 参数、rolling 104 周窗口和每 13 周更新不变，因此差异只来自 Feature 组合。

当前基准是 Fundamental 6：完整选择期 RankIC = 0.029575。

在“发现期和确认期都必须高于基准”的稳健性约束下，再按完整选择期 RankIC 排序，本轮最佳新研究候选为 `b2_f4_f5_add_m04`：

- 组合公式：`F4+F5+M04`
- 具体 Features：`['cashflow_to_assets', 'debt_to_assets', 'macro_dgs10_change_5obs']`
- 发现期 RankIC：0.029453
- 确认期 RankIC：0.070869
- 完整选择期 RankIC：0.050161
- 相对 Fundamental 6 的完整选择期改善：+0.020586
- 相对基准的发现期改善：+0.027622
- 相对基准的确认期改善：+0.013549
- 相对基准的完整期改善比例：+69.60%
- 逐周 RankIC 超过基准的比例：59.43%

更严格地，如果只允许“发现期前 10 名 + 原基准”进入确认期决选，胜者是 `b1_f1_f2_f3_f4_f5_f6`，即原 Fundamental 6。因此上述最佳组合应作为下一轮候选，不应宣称已在全新 holdout 上证明胜出。

## 分批方法

1. Batch 1：穷举 6 个基本面因子的 63 个非空子集。
2. Batch 2：仅对发现期最优的 2 个基本面子集，逐个加入 31 个技术因子和 4 个宏观因子。
3. Batch 3：每个基础组合只保留 Batch 2 发现期最好的 6 个新增因子，再测试两两组合和前 4 个的三因子组合。
4. 评价分为 Week 312–364 发现期和 Week 365–417 确认期。只有发现期前 10 名与原基准进入确认期决选。

Batch 摘要：

```text
                      batch  candidates  best_full_Rank_IC  best_discovery_Rank_IC  best_confirmation_Rank_IC
batch_1_fundamental_subsets          63           0.044628                0.023341                   0.081020
   batch_2_single_additions          70           0.050161                0.043920                   0.086492
    batch_3_multi_additions          38           0.048786                0.061112                   0.059226
```

Batch 1 晋级的基本面组合：

```text
candidate_id combination_formula  discovery_Rank_IC  confirmation_Rank_IC
    b1_f5_f6               F5+F6           0.023341              0.029149
    b1_f4_f5               F4+F5           0.023079              0.059872
```

## 改善最大的组合

```text
            candidate_id                       batch combination_formula  full_Rank_IC  delta_full_Rank_IC_vs_baseline  discovery_Rank_IC  confirmation_Rank_IC  weekly_win_ratio_vs_baseline
        b2_f4_f5_add_m04    batch_2_single_additions           F4+F5+M04      0.050161                        0.020586           0.029453              0.070869                      0.594340
b3_f4_f5_add_t21_m01_t25     batch_3_multi_additions   F4+F5+T21+M01+T25      0.048786                        0.019210           0.061112              0.036459                      0.518868
    b3_f4_f5_add_t21_t25     batch_3_multi_additions       F4+F5+T21+T25      0.048039                        0.018463           0.040254              0.055823                      0.575472
        b2_f4_f5_add_t21    batch_2_single_additions           F4+F5+T21      0.047115                        0.017539           0.043920              0.050310                      0.603774
    b3_f4_f5_add_t21_t13     batch_3_multi_additions       F4+F5+T21+T13      0.045317                        0.015742           0.058765              0.031870                      0.537736
             b1_f3_f4_f5 batch_1_fundamental_subsets            F3+F4+F5      0.044628                        0.015053           0.008236              0.081020                      0.528302
    b3_f4_f5_add_m01_t25     batch_3_multi_additions       F4+F5+M01+T25      0.043214                        0.013638           0.050836              0.035592                      0.594340
        b2_f4_f5_add_t22    batch_2_single_additions           F4+F5+T22      0.041888                        0.012313          -0.002715              0.086492                      0.518868
    b3_f5_f6_add_t17_t21     batch_3_multi_additions       F5+F6+T17+T21      0.041886                        0.012310           0.046082              0.037690                      0.500000
          b1_f3_f4_f5_f6 batch_1_fundamental_subsets         F3+F4+F5+F6      0.041516                        0.011941           0.015230              0.067803                      0.584906
                b1_f4_f5 batch_1_fundamental_subsets               F4+F5      0.041476                        0.011900           0.023079              0.059872                      0.547170
    b3_f4_f5_add_t21_m01     batch_3_multi_additions       F4+F5+T21+M01      0.041311                        0.011735           0.047326              0.035296                      0.547170
    b3_f4_f5_add_t25_m04     batch_3_multi_additions       F4+F5+T25+M04      0.039847                        0.010271           0.020468              0.059226                      0.584906
b3_f4_f5_add_m01_t25_m04     batch_3_multi_additions   F4+F5+M01+T25+M04      0.039767                        0.010191           0.040481              0.039052                      0.500000
b3_f4_f5_add_t21_m01_m04     batch_3_multi_additions   F4+F5+T21+M01+M04      0.039600                        0.010025           0.040395              0.038806                      0.556604
```

## 发现期 Shortlist 的确认结果

```text
            candidate_id combination_formula  discovery_Rank_IC  confirmation_Rank_IC  full_Rank_IC  delta_full_Rank_IC_vs_baseline
    b1_f1_f2_f3_f4_f5_f6   F1+F2+F3+F4+F5+F6           0.001831              0.057320      0.029575                        0.000000
        b2_f4_f5_add_t21           F4+F5+T21           0.043920              0.050310      0.047115                        0.017539
b3_f4_f5_add_m01_t25_m04   F4+F5+M01+T25+M04           0.040481              0.039052      0.039767                        0.010191
    b3_f5_f6_add_t17_t21       F5+F6+T17+T21           0.046082              0.037690      0.041886                        0.012310
b3_f4_f5_add_t21_m01_t25   F4+F5+T21+M01+T25           0.061112              0.036459      0.048786                        0.019210
    b3_f4_f5_add_m01_t25       F4+F5+M01+T25           0.050836              0.035592      0.043214                        0.013638
    b3_f4_f5_add_t21_m01       F4+F5+T21+M01           0.047326              0.035296      0.041311                        0.011735
        b2_f4_f5_add_m01           F4+F5+M01           0.042568              0.034341      0.038454                        0.008879
    b3_f4_f5_add_t21_t13       F4+F5+T21+T13           0.058765              0.031870      0.045317                        0.015742
    b3_f5_f6_add_m01_m04       F5+F6+M01+M04           0.046789              0.026802      0.036795                        0.007220
    b3_f5_f6_add_t21_m04       F5+F6+T21+M04           0.042229              0.011413      0.026821                       -0.002755
```

## 解读边界

这是在既有研究选择期内做的新一轮 Feature 研究。本轮没有运行 Week 418 之后的已有最终 holdout，也没有回测。由于之前已经查看过原 holdout，本报告的推荐应视为“新候选”，而不是拥有新的未见样本证明的最终结论。
