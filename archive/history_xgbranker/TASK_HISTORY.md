# SafeCapital XGBRanker 实施任务书

## 受控实验 C/D（2026-09-29，覆盖最终标签限制用于本轮比较）

保持三模型结构、Feature、156/52/52周滚动窗口、26周更新周期、XGBoost参数、随机种子、三模型等权集成和Week 156–521评估区间完全一致，只比较训练标签：

- 实验C：直接使用连续下一周对数收益 `target`。
- 实验D：在每个训练周内根据连续 `target` 横截面排序，生成0–4五档 relevance label；不使用数据列 `target2`。

两项实验都必须使用原始连续 `target` 计算周度IC、RankIC和最终平均RankIC。输出两项实验的整体、组件、四段稳定性、每周配对差异、每次重训和逐股预测结果。选择标准为Week 156–521的三模型集成平均RankIC，但同时报告RankIC差值、相对变化、逐周胜率和配对差异统计。项目仍不运行投资组合或收益回测。

## 最终冻结模型运行修订（2026-09-29）

保留此前全部 Feature、参数、窗口、更新周期和集成搜索记录，但最终运行不再继续选择参数。最终模型固定为三个 XGBRanker：156 周五因子模型、52 周两因子模型、52 周强正则五因子模型；三者每 26 周重新训练，周内预测 percentile rank 等权平均。

模型直接使用连续 `target = ln(AdjClose(t+1) / AdjClose(t))` 训练。`target2`、0–4 分类标签和 relevance bins 均不得进入训练。模型在每周收盘后运行，训练标签满足 `label_end_week <= refit_week`；52 周组件从 Week 52 开始独立预测，正式三模型集成从 Week 156 开始并覆盖 Week 156–521。

最终必须输出53个基础模型实例的后续预测期指标、15个集成更新批次指标、366周周度 IC/RankIC、每周每只股票的三个组件分数与百分位、集成分数与预测排名、连续 `target` 与实际排名。参数在最终运行前冻结，不能根据新结果再次修改。项目不进行投资组合或收益回测。

## 新增：XGBRanker 算法联合优化（2026-09-29）

在不新增数据的前提下，将 Week 260–521 共 262 周用于研究用 walk-forward 时间交叉验证。分阶段搜索：横截面特征变换、20 组 XGBRanker/LambdaRank 参数、35 组窗口与 refit 周期、14 个 Feature 复核候选、27 个联合候选以及前 5 个单模型的等权排名集成。

主选择标准是 262 周 Mean RankIC，同时输出四个连续时间段 RankIC、最差时段 RankIC、RankICIR、正 RankIC 比例和相对当前算法的配对周度改善。最终配置必须保留严格 `label_end_week < refit_week` 防泄漏规则。

本轮没有新的 untouched holdout，所以“最优”只指预先定义搜索空间和现有历史数据上的最高时间 CV RankIC，不宣称为未来数学全局最优。仍不运行策略或回测。

## 新增：分批 Feature 组合优化（2026-09-29）

在已冻结的 XGBRanker `pairwise_d4_bins5` 、rolling 104 周窗口和每 13 周 refit 基础上，继续研究更多 Feature 组合。本轮只在原 Week 312–417 研究选择期运行，不重复查看 Week 418–521 的已有最终 holdout，不运行回测。

按以下三批完成：

1. Batch 1：穷举 Fundamental 6 的 63 个非空子集。
2. Batch 2：发现期最优的 2 个基本面子集，分别逐个加入 31 个技术因子与 4 个宏观因子。
3. Batch 3：对每个基础组合的前 6 个新增因子做两两组合，并对前 4 个做三因子组合。

时间内部再分为 Week 312–364 发现期与 Week 365–417 确认期。每个组合必须输出具体 Feature list、完整期/发现期/确认期 RankIC，以及相对 Fundamental 6 的 RankIC 绝对改善、相对改善比例和逐周胜率。

最佳新研究候选要求在发现期和确认期都超过 Fundamental 6，然后以完整选择期 Mean RankIC 排序。同时另行报告“发现期 shortlist + 确认期决选”的保守结果。由于没有新的未见 holdout，新组合只能标记为下一轮候选，不得宣称为已证明的最终样本外最优。

## 最新 Rank IC 优化修订（覆盖之前的简单 7×2 比较）

本轮目标是在不泄漏最终测试集的前提下，系统搜索 XGBRanker 的 Feature 集、Fixed 训练长度、Rolling 窗口和更新周期，并最大化选择期 Mean Rank IC。

### 数据时间隔离

```text
Week 0–207     Hyperparameter Train（208周）
Week 208–259   Hyperparameter Validation（52周）
Week 260–311   额外历史积累，不用于选择指标
Week 312–417   Feature/Window/Refit 选择期（106周）
Week 418–521   最终不可触碰 Holdout（104周）
```

最后 104 周不得用于 Feature、超参、窗口、refit 周期或训练方式选择。最终查看 holdout 后不得再返回修改配置。

### Fixed 搜索

在相同选择期上比较：

```text
all_history
156 weeks
208 weeks
260 weeks
312 weeks
```

Fixed 模式在评价期开始时训练一次，之后不再 refit。`all_history` 在最终 holdout 中对应约 80% 开发、20% 最终测试。

### Rolling 搜索

```text
Rolling windows = 104, 156, 208, 260, 312 weeks
Refit intervals = 2, 4, 8, 13 weeks
```

所有组合都按每周预测下一周，并严格执行 `label_end_week < refit_week`。

### Feature 搜索

预先固定候选集，包括：

- Priority 7 的 F/R/P block 组合。
- 全部 6 个基本面、6 个收益/动量、7 个趋势、6 个波动、6 个流动性/成交量等经济类别。
- 所有 31 个非基本面股票因子。
- 全部 37 个非重复股票因子。
- Priority 7 + 4 Macro，以及 Stock 37 + 4 Macro。
- Priority 7 与趋势/波动/流动性 block 的增量组合。

先使用统一 baseline 时间配置筛选 Feature，再对 fixed 和 rolling 各自表现最好的前若干 Feature 集进行完整时间网格搜索。

主选择标准是选择期 Mean Rank IC，Rank ICIR 只作 tie-break。最终 holdout 只评估预先冻结的 best-fixed 和 best-rolling，不参与选择。

本轮仍只实现 XGBRanker ML 和 IC/Rank IC 评估，不运行策略或回测。

---

## 0. 最新范围修订（优先级最高）

本轮要求覆盖本文后续的 Strategy/Backtest 交付要求：

1. 保留原有 156 周 trailing rolling window，每 4 周 refit。
2. 新增 fixed-window 方法：在第一个 OOS 时点用与滚动模型相同的首期历史候选窗口训练一次，之后不再 refit；仍每周使用当周新 X 预测下一周排名。
3. Fixed 与 rolling 必须使用完全一致的 OOS 周、target、feature set、Warm-up 调参结果和 leakage guard。
4. Priority 7 按经济含义预先分成三个 block，比较三个 block 的 7 个非空组合：

```text
F (Fundamental) = revenue_growth_yoy, cashflow_to_assets
R (Return horizons) = ret_120d, ret_1d, ret_5d
P (Price action) = close_location_value, body_range_ratio

F, R, P, F+R, F+P, R+P, F+R+P
```

5. 每个 feature 组合都分别输出 fixed/rolling 的 IC、ICIR、Rank IC、Rank ICIR、正 IC 比例、正 Rank IC 比例，并输出 rolling minus fixed 差值。
6. 不使用 OOS 结果改写 feature 组合，不进行 127 个任意 Priority-7 子集的事后穷举搜索。
7. **本项目当前不运行、不输出、不比较 Strategy 或 Backtest。** Annualized Return、Information Ratio、turnover、transaction cost 均不属于本轮验收项。

---

## 1. 任务目标与边界

基于现有周频因子数据，实现一条完整、可复现、无明显时间泄漏的流程：

```text
Weekly Cross-Sectional XGBRanker
→ Warm-up Tuning
→ Frozen Configuration
→ Rolling OOS Ranking
→ Top-X Strategy
→ Weekly Backtest
→ Evaluation and Reproducibility Report
```

本任务只实现 XGBRanker，不重做 Ridge、Random Forest 或普通 XGBoost，也不重做或修改既有单因子分析。

已固定的因子结论不变：45 个原始因子中剔除 4 个重复因子，4 个宏观因子单独保留，剩余 37 个股票层面因子，其中 7 个优先因子为：

```text
revenue_growth_yoy
ret_120d
cashflow_to_assets
ret_1d
ret_5d
close_location_value
body_range_ratio
```

4 个宏观因子为：

```text
macro_vix_level
macro_vix_change_5obs
macro_yield_spread_10y_2y
macro_dgs10_change_5obs
```

Baseline 只使用 Priority 7；架构同时支持 Stock Factor 37，以及未来显式配置的 `7/37 Factors + Macro` 实验，但不自动把宏观因子混入 baseline。

## 2. 数据与 Target

- 主要输入：当前目录中的周频因子 CSV，优先使用 `test_45_factors_weekly.csv`。
- 经济目标始终是连续的下一期真实收益 `target`。
- `target2` 可保留于原始数据，但不得进入 X，也不得作为最终经济评价指标。
- 必须检查上游文件或数据特征，确认 `target` 是否已按当前 signal row 对齐的 forward target。如已对齐，不得再次 shift。如无法确认，必须报告，不得自行改变 alignment 或交易时点假设。
- 内部用排序后的唯一周频 `date` 生成 `week_index`，所有时间窗口都按实际存在的周计数，不用日历天换算。
- 至少验证：`date+ticker` 唯一、日期可排序、周频合理、`target` 有效、数值中无无穷值。

## 3. Feature 管理

Feature list 必须集中配置，每次实验记录 `factor_set_name`、`feature_list` 和 `number_of_features`。

X 绝对禁止包含：

```text
date, ticker
open, high, low, close, adj_close, volume
target, target2
future_*, fwd_*, label_*
y_true, actual_return, actual_rank
ranking_score, predicted_rank
任何其他未来收益、forward label、metadata 或 prediction result
```

发现违禁字段进入 X 必须直接报错。XGBRanker baseline 不强制做 StandardScaler/MinMaxScaler。如需填补、winsorization 或 clipping，参数只能在当期训练窗口 fit，再 apply 到预测数据；不得利用未来数据。

## 4. 时间配置

所有参数都必须集中可配置，不得为更换实验周期修改核心算法：

```text
WARMUP_WEEKS = 156
WARMUP_TRAIN_WEEKS = 104
WARMUP_VALIDATION_WEEKS = 52
ROLLING_WINDOW_WEEKS = 156
REFIT_INTERVAL_WEEKS = 4
FORECAST_HORIZON_WEEKS = 1
```

Warm-up 按完整周做 chronological split：前 104 周训练，后 52 周验证。禁止 random split、shuffle 或拆分同一周的 ranking group。

## 5. XGBRanker Label 与 Group

- 每个 `date` 是一个独立 ranking group。
- 在每周内，依据连续 `target` 横截面排序，转为整数 `relevance_label`。
- `RELEVANCE_BINS` 必须可配置，支持如 5/10/20 等 levels。绝对不得对全时期样本统一分箱。
- 如 relevance bins 参与模型选择，只能在 Warm-up 决定，之后冻结。
- 训练时必须按 `date` 排序数据，并保证 group sizes 之和等于训练行数。

## 6. Warm-up 调参与冻结

Warm-up 是唯一允许调参的阶段。用受控、固定、可复现的 candidate configurations，可考察：

```text
objective, max_depth, learning_rate, n_estimators
min_child_weight, subsample, colsample_bytree
reg_alpha, reg_lambda, gamma, relevance_bins
```

每组候选都记录验证 IC、ICIR、Rank IC 和 Rank ICIR。主选择标准是 Validation Mean Rank IC，接近时用 Rank ICIR 辅助判断，不使用后续 Rolling OOS 结果反向调参。

Warm-up 后生成 frozen config，至少包含：

```text
warmup_weeks, warmup_train_weeks, warmup_validation_weeks
factor_set_name, feature_list, objective, relevance_bins
hyperparameters, selection_metric, random_seed
```

同时保存所有 Warm-up 候选结果。

## 7. Rolling OOS 与防泄漏

Warm-up 后进入正式 Rolling Evaluation：

- 每次 refit 只使用最近 156 个历史周。
- 每 4 周重新训练一个模型；模型在该 block 的每周都预测，而不是每 4 周才预测。
- 预测频率为每周，默认 forecast horizon 为 1 周。
- 每个 prediction week 只能对应一个 `model_id`。

核心泄漏规则：若 signal row 为周 t，`target` 表示周 t+1 的收益，则样本只能在其标签所覆盖期间已经完全结束后用于训练。实现必须显式建立 `label_end_week` 或等价字段，并且训练样本必须同时满足：

```text
feature_week < refit_week
label_end_week < refit_week
```

不得仅因 X 的日期早于 refit date 就将样本纳入训练。

## 8. 预测与模型评估

每周必须保存所有可预测股票的完整排序，至少包含：

```text
week_index, date, ticker
model_id, refit_week, factor_set_name
ranking_score, predicted_rank
actual_return, actual_rank, relevance_label
```

`ranking_score` 是原始排序分数，不得命名或解释为 predicted return。`predicted_rank` 和 `actual_rank` 都在每周内由高到低产生，Rank 1 为最高。

每个 refit model 需保存训练和配置元数据，并计算它负责的 OOS block 指标。单个约 4 周 block 的 ICIR 只用于稳定性诊断；最终比较使用全部 OOS 周指标。

指标必须先按周横截面计算，不得计算 pooled correlation：

```text
IC_t = Pearson(ranking_score, actual_return) within week t
RankIC_t = Spearman(ranking_score, actual_return) within week t
IC = mean(IC_t)
ICIR = mean(IC_t) / sample_std(IC_t)
Rank IC = mean(RankIC_t)
Rank ICIR = mean(RankIC_t) / sample_std(RankIC_t)
```

Overall 输出还包含 Positive IC Ratio、Positive Rank IC Ratio 和 prediction week 数。

## 9. Strategy

Strategy 作为独立模块读取 weekly predictions，不得与 XGBRanker 训练耦合。

第一版 baseline：

```text
TOP_X = 可配置
WEIGHT_METHOD = equal_weight
CASH_RULE = always_invest
REBALANCE_INTERVAL_WEEKS = 1
```

每周选择 `predicted_rank` 的 Rank 1 至 Rank X，等权分配。架构预留 score-based/risk-based weight 和其他 cash rule 接口，但不引入未确认的复杂规则。

## 10. Backtest

回测必须与上游 `target` 的持有周期和价格定义一致。如交易时点无法从上游证据确认，报告该限制，不自行发明。

配置项至少包含：

```text
TRANSACTION_COST (0, 5bps, 10bps, ...)
SLIPPAGE
BENCHMARK_METHOD
```

- 每周根据实际目标权重变化计算 turnover，成本按换仓量计算，不是固定扣减。
- 默认 benchmark 可使用可配置的 weekly equal-weight universe return。
- 每周记录 `date, selected_stocks, predicted_ranks, weights, cash_weight, gross_return, turnover, transaction_cost, slippage_cost, net_return, portfolio_value`。
- Annualized Return 从周度净收益复利换算，每年 52 周。如 `target` 是 log return，组合前先转为 simple return，不得将 log return 直接加权复利。
- `IR = mean(weekly excess return) / sample_std(weekly excess return) * sqrt(52)`，并明确区分 IR 与 ICIR/Rank ICIR。

## 11. 结果文件

至少生成：

```text
warmup_hyperparameter_results
frozen_xgbranker_config
xgbranker_weekly_predictions
xgbranker_refit_model_metrics
xgbranker_weekly_metrics
xgbranker_overall_metrics
strategy_weekly_positions
backtest_weekly_returns
backtest_summary
```

可根据内容选择 CSV/JSON，并保持字段稳定。

## 12. 可复现性

正式实验必须记录：

```text
input filename and hash
factor set and exact feature list
all time settings
random seed
Python and XGBoost versions
hyperparameters, objective and relevance bins
target column
strategy, transaction cost, slippage and benchmark settings
```

## 13. 自动验收

至少包含以下测试：

1. 数据：主键唯一、日期排序、周频合理、target 有效、无 Inf。
2. Feature：所有禁止列和未来/标签/预测列不在 X。
3. Warm-up：恰好前 156 个有效周，104 周 train，52 周 validation，且时间严格先后。
4. Rolling：训练窗口最多 156 周，每 4 周 refit，每周 prediction，prediction 唯一对应 model id。
5. Leakage：每个训练行的 `label_end_week < refit_week`。
6. Ranker：每周一个 group，group sizes 之和等于行数，relevance 只在周内生成。
7. Prediction：`date+ticker` 唯一，score/ranks/actual return 字段齐全，周内 rank 正确。

## 14. 显式禁止项

不得重做单因子分析、修改 Priority 7 结论、用 Rolling OOS 重选因子或超参、随机切分/打乱时间、将 target/target2 加入 X、在全样本 fit preprocessing、Warm-up 后继续调参、将 ranking score 称为 predicted return、把 Top-X 写死在模型中、将策略与训练强耦合，或依据最终回测结果自动反复调模型。

## 15. 完成报告

实现后提供可核验报告，包括：

- 实际 CSV、股票数、周数、样本数、target 和 features。
- Warm-up/train/validation 周数、候选配置、最终配置与选择依据。
- Rolling window/refit interval、模型数和预测周数。
- label observability 和违禁列防泄漏方法。
- relevance/group/objective 设计。
- Overall IC、ICIR、Rank IC、Rank ICIR，及小部分预测样例。
- Top X、权重、cash rule、rebalance frequency。
- Annualized Return、Information Ratio、turnover、cost 和 benchmark。
- 新增/修改/生成文件和测试结果。
- 任何无法确认的 target alignment、交易时点或上游数据定义限制。
