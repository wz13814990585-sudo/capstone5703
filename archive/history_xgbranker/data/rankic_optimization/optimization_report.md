# XGBRanker Rank IC 优化与最终检验报告

## 方法

数据共 522 周、66 只股票、27651 条记录。为避免在最终样本上反复搜索，使用严格三阶段时间隔离：

1. Week 0–259：超参期，208 周 train + 52 周 validation。
2. Week 312–417：研究选择期，用于 Feature、Fixed 长度、Rolling 窗口和 refit 周期选择。
3. Week 418–521：最后 104 周 holdout，不参与任何选择。

所有阶段仍强制 `label_end_week < refit_week`。最终 holdout 只评估在研究选择期预先确定的 fixed 和 rolling 候选，不用 holdout 结果返回修改参数。

## Fixed 训练集大小

已比较：`['all_history', 156, 208, 260, 312]`。`all_history` 表示在 OOS 开始时使用当时所有严格可观测历史，然后整个评价期不再 refit。在最终 104 周 holdout 下，`all_history` 接近 80% 开发 / 20% 最终测试，比原先 156 周训练、365 周 OOS 的分配更适合 fixed model。

## Feature 筛选

预注册 Feature 集数：20。包含 Priority 子组合、基本面/收益/趋势/波动/流动性等经济类别、所有 31 个非基本面因子、全部 37 个非重复股票因子，以及显式的 +Macro 实验。进入精细时间网格的 Feature 集：`['fundamental_6', 'returns_3', 'liquidity_volume_6']`。

Feature screening 结果：

```text
           factor_set_name training_mode  number_of_features   Rank_IC  Rank_ICIR
             fundamental_6       rolling                   6  0.017116   0.100776
             fundamental_6         fixed                   6  0.007710   0.049670
        liquidity_volume_6       rolling                   6 -0.002097  -0.010521
                 returns_3         fixed                   3 -0.002927  -0.017398
               technical_5         fixed                   5 -0.004370  -0.024937
            price_action_2       rolling                   2 -0.005779  -0.039757
               technical_5       rolling                   5 -0.006039  -0.037570
   priority_plus_liquidity       rolling                  13 -0.006300  -0.031380
                 returns_3       rolling                   3 -0.007586  -0.044342
            price_action_2         fixed                   2 -0.010415  -0.081641
     priority_7_plus_macro       rolling                  11 -0.010923  -0.058582
          microstructure_6         fixed                   6 -0.013406  -0.090946
          microstructure_6       rolling                   6 -0.014018  -0.101739
                priority_7       rolling                   7 -0.015140  -0.082408
        returns_momentum_6       rolling                   6 -0.015876  -0.094001
        returns_momentum_6         fixed                   6 -0.016071  -0.090128
       priority_plus_trend       rolling                  14 -0.016846  -0.096133
fundamental_price_action_4       rolling                   4 -0.018823  -0.109508
     priority_7_plus_macro         fixed                  11 -0.019088  -0.104886
     fundamental_returns_5       rolling                   5 -0.019129  -0.101917
     fundamental_returns_5         fixed                   5 -0.020068  -0.108291
                priority_7         fixed                   7 -0.022713  -0.123977
        liquidity_volume_6         fixed                   6 -0.023092  -0.108529
   priority_plus_liquidity         fixed                  13 -0.024357  -0.108717
                  stock_37       rolling                  37 -0.025470  -0.117603
       priority_plus_trend         fixed                  14 -0.027029  -0.148480
fundamental_price_action_4         fixed                   4 -0.028881  -0.174254
                   trend_7         fixed                   7 -0.029200  -0.183566
              technical_31       rolling                  31 -0.029999  -0.139280
                   trend_7       rolling                   7 -0.030925  -0.198388
  priority_plus_volatility         fixed                  13 -0.032151  -0.138695
              technical_31         fixed                  31 -0.033609  -0.143528
                  stock_37         fixed                  37 -0.034389  -0.146473
             fundamental_2       rolling                   2 -0.034566  -0.182620
       stock_37_plus_macro         fixed                  41 -0.038471  -0.160709
              volatility_6         fixed                   6 -0.039272  -0.169141
  priority_plus_volatility       rolling                  13 -0.039445  -0.169850
              volatility_6       rolling                   6 -0.040539  -0.173620
       stock_37_plus_macro       rolling                  41 -0.046270  -0.193770
             fundamental_2         fixed                   2 -0.046922  -0.268745
```

## 时间参数搜索

- Rolling windows: `[104, 156, 208, 260, 312]`
- Refit intervals: `[2, 4, 8, 13]`
- Fixed windows: `['all_history', 156, 208, 260, 312]`

选择期 Rank IC 最高的 25 组：

```text
   factor_set_name         config_id training_mode training_window_weeks  refit_interval_weeks   Rank_IC  Rank_ICIR
     fundamental_6  rolling_w104_r13       rolling                   104             13.000000  0.029575   0.197336
     fundamental_6   rolling_w104_r8       rolling                   104              8.000000  0.028424   0.174164
     fundamental_6   rolling_w156_r8       rolling                   156              8.000000  0.024270   0.147615
     fundamental_6   rolling_w156_r4       rolling                   156              4.000000  0.017116   0.100776
     fundamental_6   rolling_w312_r8       rolling                   312              8.000000  0.015537   0.097002
     fundamental_6  rolling_w260_r13       rolling                   260             13.000000  0.014006   0.096146
     fundamental_6   rolling_w208_r8       rolling                   208              8.000000  0.013785   0.086375
     fundamental_6   rolling_w260_r8       rolling                   260              8.000000  0.013760   0.081464
     fundamental_6  rolling_w156_r13       rolling                   156             13.000000  0.013217   0.093262
     fundamental_6   rolling_w156_r2       rolling                   156              2.000000  0.011053   0.060189
     fundamental_6   rolling_w104_r4       rolling                   104              4.000000  0.010718   0.064525
     fundamental_6   rolling_w104_r2       rolling                   104              2.000000  0.008979   0.049275
     fundamental_6  rolling_w312_r13       rolling                   312             13.000000  0.008452   0.055695
     fundamental_6 fixed_all_history         fixed           all_history                   NaN  0.007710   0.049670
     fundamental_6         fixed_312         fixed                   312                   NaN  0.007710   0.049670
     fundamental_6   rolling_w312_r2       rolling                   312              2.000000  0.006990   0.041240
     fundamental_6  rolling_w208_r13       rolling                   208             13.000000  0.002966   0.019286
     fundamental_6   rolling_w312_r4       rolling                   312              4.000000  0.001647   0.010210
     fundamental_6   rolling_w208_r4       rolling                   208              4.000000  0.000755   0.004808
         returns_3   rolling_w312_r2       rolling                   312              2.000000 -0.001043  -0.006260
     fundamental_6   rolling_w260_r2       rolling                   260              2.000000 -0.001113  -0.006492
         returns_3   rolling_w260_r8       rolling                   260              8.000000 -0.001170  -0.006944
         returns_3   rolling_w312_r4       rolling                   312              4.000000 -0.001362  -0.008119
liquidity_volume_6   rolling_w156_r4       rolling                   156              4.000000 -0.002097  -0.010521
         returns_3   rolling_w312_r8       rolling                   312              8.000000 -0.002184  -0.013080
```

## 最终 Holdout

在查看最终 holdout 之前，主配置已冻结为 `fundamental_6` + `rolling_w104_r13`。同时保留选择期最优 fixed 和最优 rolling 各一个作诊断比较，但不根据最终 holdout 重选配置。

```text
              finalist_role  is_primary_pre_holdout_selection factor_set_name         config_id       IC     ICIR  Rank_IC  Rank_ICIR  Positive_Rank_IC_Ratio
  best_fixed_from_selection                             False   fundamental_6 fixed_all_history 0.040041 0.225388 0.029534   0.213822                0.567308
best_rolling_from_selection                              True   fundamental_6  rolling_w104_r13 0.036757 0.178686 0.039558   0.197808                0.634615
```

本项目只运行 XGBRanker 机器学习与 IC/Rank IC 评估，未运行策略或回测。
