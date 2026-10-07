# XGBRanker 算法优化报告

## 结论

在不新增数据的前提下，本轮把 Week 260–521 共 262 周用于多时段 walk-forward 交叉验证和算法搜索。因为没有全新的未见数据，这里的“最优”仅指本次搜索空间与现有历史数据上的最高交叉验证 RankIC，不能证明未来真正的数学全局最优。

- 当前三因子算法 CV RankIC：0.021432
- 本轮胜者类型：`ensemble`
- 优化后 CV RankIC：0.048960
- RankIC 绝对改善：+0.027528
- RankIC 相对改善：+128.44%
- 逐周 RankIC 超过当前算法的比例：54.96%
- 配对周度改善 t-stat：2.597
- 胜者四个时间段 RankIC：[0.03794764704715081, 0.04177145080645313, 0.07512540743679051, 0.04072156409620215]

## 为什么这样优化

1. **横截面预处理**：比较 raw、每周横截面 percentile rank 和横截面 z-score。这用来减少财务因子跨时间尺度变化对树分裂的影响。宏观因子在同一周对所有股票相同，所以保留原值，不做横截面排名。
2. **LambdaRank 和树结构**：搜索 pairwise/ndcg、树深、叶子数、learning rate、树数、min child weight、正则化和 LambdaRank pair sampling。原因是 RankIC 取决于顺序，不是收益数值误差。
3. **窗口与更新频率**：比较 52/78/104/130/156/208 周和 expanding history，以及 2/4/8/13/26 周 refit。较短窗口更灵敏，较长窗口更稳定，必须由时间 CV 决定。
4. **Feature 复核**：不仅使用之前的三因子，还重新比较了之前在两个子时期均有改善的组合。
5. **排名集成**：对最强单模型的每周预测分数先转成 percentile rank，再等权平均。这避免不同 XGBRanker 分数尺度不同，并检验模型多样性是否能提高排名稳定性。

## 各阶段最优结果

```text
                stage                                                                                                                                                                  configuration_id  Rank_IC  Rank_ICIR  minimum_fold_Rank_IC
    Current algorithm                                                                                                                             current_pairwise_d4__raw__b2_f4_f5_add_m04__w104__r13 0.021432   0.144766             -0.017639
   Best preprocessing                                                                                                                         current_pairwise_d4__cs_rank__b2_f4_f5_add_m04__w104__r13 0.023328   0.154923              0.004822
 Best hyperparameters                                                                                                                           pairwise_d3_mcw20__cs_rank__b2_f4_f5_add_m04__w104__r13 0.026351   0.173684             -0.009193
        Best schedule                                                                                                                            pairwise_d3_mcw20__cs_rank__b2_f4_f5_add_m04__w52__r26 0.036618   0.252445              0.011024
Best feature finalist                                                                                                                           pairwise_d3_mcw20__cs_rank__b1_f1_f3_f4_f5_f6__w52__r26 0.039754   0.242570              0.036407
    Best joint single                                                                                                                           pairwise_d2_slow__cs_rank__b1_f1_f3_f4_f5_f6__w156__r26 0.043487   0.276625              0.033053
        Best ensemble ensemble_3_pairwise_d2_slow__cs_rank__b1_f1_f3_f4_f5_f6__w156__r26__pairwise_d3_mcw20__cs_rank__b1_f3_f5__w52__r26__pairwise_d3_regularized__cs_rank__b1_f1_f3_f4_f5_f6__w52__r26 0.048960   0.300250              0.037948
```

## 胜者配置

```json
{
  "IC": 0.047343342220929244,
  "ICIR": 0.28186479479195237,
  "Rank_IC": 0.048960463670464784,
  "Rank_ICIR": 0.3002504965014801,
  "Positive_IC_Ratio": 0.6259541984732825,
  "Positive_Rank_IC_Ratio": 0.6412213740458015,
  "number_of_prediction_weeks": 262,
  "number_of_valid_IC_weeks": 262,
  "number_of_valid_Rank_IC_weeks": 262,
  "stage": "ensemble_search",
  "configuration_id": "ensemble_3_pairwise_d2_slow__cs_rank__b1_f1_f3_f4_f5_f6__w156__r26__pairwise_d3_mcw20__cs_rank__b1_f3_f5__w52__r26__pairwise_d3_regularized__cs_rank__b1_f1_f3_f4_f5_f6__w52__r26",
  "ensemble_size": 3,
  "component_configuration_ids": "[\"pairwise_d2_slow__cs_rank__b1_f1_f3_f4_f5_f6__w156__r26\", \"pairwise_d3_mcw20__cs_rank__b1_f3_f5__w52__r26\", \"pairwise_d3_regularized__cs_rank__b1_f1_f3_f4_f5_f6__w52__r26\"]",
  "combination_method": "weekly percentile-rank then equal-weight mean",
  "fold_1_start_week": 260,
  "fold_1_end_week": 324,
  "fold_1_Rank_IC": 0.03794764704715081,
  "fold_1_Rank_ICIR": 0.2428941659929422,
  "fold_2_start_week": 325,
  "fold_2_end_week": 389,
  "fold_2_Rank_IC": 0.04177145080645313,
  "fold_2_Rank_ICIR": 0.31114123212874256,
  "fold_3_start_week": 390,
  "fold_3_end_week": 455,
  "fold_3_Rank_IC": 0.07512540743679051,
  "fold_3_Rank_ICIR": 0.46413652798091365,
  "fold_4_start_week": 456,
  "fold_4_end_week": 521,
  "fold_4_Rank_IC": 0.04072156409620215,
  "fold_4_Rank_ICIR": 0.2092958028487922,
  "minimum_fold_Rank_IC": 0.03794764704715081,
  "fold_Rank_IC_std": 0.01756349215508472,
  "all_folds_positive": true,
  "robust_score": 0.031328025191564426
}
```

本轮只运行 XGBRanker 机器学习和 IC/RankIC 评估，没有运行交易策略或回测。
