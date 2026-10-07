# Random Forest Factor Research — Nine Experiments

## Overview

This project compares three factor sets under three training designs, producing nine independent Random Forest experiments. Each experiment tunes its own hyperparameters during Development, selects one candidate during Validation, and evaluates the frozen hyperparameters on a common Final Test period.

The results below were transcribed from the supplied final comparison screenshots. They are reported at the precision displayed by the script; they were not recomputed for this README.

## Experiment design

| Factor set | Description |
|---|---|
| 7 factors | Seven priority factors from the screening workbook |
| 7 + 4 macro | The seven priority factors plus four macro factors (11 predictors) |
| 45 factors | All 45 workbook factors, including the four macro factors |

The target and identifiers are excluded from the predictors. The script expects 66 distinct stocks by default; the number of evaluable stocks can vary across weeks because target availability varies.

All date ranges below include the start and exclude the end.

| Training design | Initial Training | Development | Validation | Final Test | Development : Validation : Test |
|---|---|---|---|---|---|
| Rolling 3y | 2016-09-01–2019-09-01 | 2019-09-01–2024-09-01 | 2024-09-01–2025-09-01 | 2025-09-01–2026-09-01 | 5:1:1 |
| Rolling 6y | 2016-09-01–2022-09-01 | 2022-09-01–2024-09-01 | 2024-09-01–2025-09-01 | 2025-09-01–2026-09-01 | 2:1:1 |
| Fixed stages | 2016-09-01–2022-09-01 | 2022-09-01–2024-09-01 | 2024-09-01–2025-09-01 | 2025-09-01–2026-09-01 | 2:1:1 |

- **Rolling 3y:** refit every four observed prediction weeks using the trailing three calendar years of eligible observations.
- **Rolling 6y:** refit every four observed prediction weeks using the trailing six calendar years of eligible observations.
- **Fixed stages:** fit once at the start of each stage using all eligible history since 2016-09-01. The training history expands between stages; the fitted model remains fixed within a stage.
- The four-week rolling schedule restarts at each stage boundary.
- During Final Test, hyperparameters remain frozen. Rolling models still refit on newly available historical labels according to the predefined schedule; fixed-stage models do not refit within Final Test.

Validation and Final Test dates are identical across all nine experiments. Development periods differ, so differences between rolling designs reflect both the training window and the tuning history.

## Four-stage workflow

1. **Initial Training:** fit all 18 candidates and print in-sample diagnostics. These results are not out-of-sample evidence and are not used to rank Development candidates.
2. **Development:** evaluate all 18 candidates using the experiment's training design. Shortlist five candidates.
3. **Validation:** evaluate the five shortlisted candidates, select one, and freeze its hyperparameters.
4. **Final Test:** evaluate the selected configuration and print the nine-experiment comparison. Final Test is not another tuning stage.

Candidate screening requires all three conditions:

- Mean Rank IC > 0.
- Rank ICIR ≥ 0.05.
- Positive Rank IC Ratio ≥ 50%.

Candidates with finite excess annualized returns are eligible for selection. Passing candidates are ranked first by Top-5 excess annualized return, with candidate ID as the tie-breaker. If fewer than five pass in Development, the remaining places are filled by non-passing candidates ranked by excess annualized return. Validation uses the same ordering to select one candidate, including a fallback if none pass. A Final Test PASS therefore describes the three IC conditions only; it does not require positive returns or benchmark outperformance.

## Hyperparameter search

The grid contains 18 combinations:

| Parameter | Values |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 4, 6, 8 |
| `min_samples_leaf` | 10, 20, 30 |
| `max_features` | `sqrt`, 0.5 |
| `random_state` | 5703 |
| `n_jobs` | -1 |

Candidate numbering follows depth first, then minimum leaf size, then `max_features` (`sqrt` before 0.5).

## Data and preprocessing

The default target is `target`, interpreted as a **one-week forward log return**. Use `--target-type simple` only when the supplied target contains simple returns. Multiweek targets are not supported by the current label-timing logic.

The script:

- Checks required columns, distinct stock counts, duplicate date/ticker pairs, and workbook factor counts.
- Converts nonfinite feature and target values to missing values.
- Fits median imputation and 1st/99th percentile clipping on each training sample, then applies those fitted transformations to predictions.
- Uses only nonmissing training targets whose label end date is at or before the fitting cutoff.
- Approximates label availability using the next observed panel date. Labels ending exactly at the fitting cutoff are assumed available after that day's market close.
- Restricts evaluation to labels ending within the relevant stage.

The screening workbook must contain 7 priority factors, 4 duplicates, 4 separately retained macro factors, and 30 backup factors under the labels expected by the script. Actual factor names are printed at runtime.

## Portfolio and benchmark

Each week, the strategy selects the five stocks with the highest predicted returns and assigns equal weights. Ties are resolved by ticker. The benchmark equally weights the stocks with evaluable predictions and actual returns that week.

For log-return targets, each stock's return is converted separately using `exp(log_return) - 1` before portfolio averaging. The strategy rebalances weekly, even when a rolling model is refitted only every four weeks. Returns exclude transaction costs and slippage.

## Final Test results

All nine experiments contain **51 evaluated weeks**, **51 valid Rank IC weeks**, and **3,306 stock-week observations**. The equal-weight benchmark annualized return is **6.43%** for every experiment.

### Selected hyperparameters

| Experiment | Factors | Selected candidate | Trees | Max depth | Min leaf | Max features |
|---|---:|---|---:|---:|---:|---|
| 7 factors / rolling_3y | 7 | candidate11 | 100 | 6 | 30 | sqrt |
| 7 factors / rolling_6y | 7 | candidate17 | 100 | 8 | 30 | sqrt |
| 7 factors / fixed | 7 | candidate11 | 100 | 6 | 30 | sqrt |
| 7 + 4 macro / rolling_3y | 11 | candidate9 | 100 | 6 | 20 | sqrt |
| 7 + 4 macro / rolling_6y | 11 | candidate10 | 100 | 6 | 20 | 0.5 |
| 7 + 4 macro / fixed | 11 | candidate11 | 100 | 6 | 30 | sqrt |
| 45 factors / rolling_3y | 45 | candidate4 | 100 | 4 | 20 | 0.5 |
| 45 factors / rolling_6y | 45 | candidate3 | 100 | 4 | 20 | sqrt |
| 45 factors / fixed | 45 | candidate7 | 100 | 6 | 10 | sqrt |

### Prediction and correlation metrics

| Experiment | Screen | RMSE | IC | ICIR | Rank IC | Rank ICIR | Positive Rank IC |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 factors / rolling_3y | PASS | 0.059206 | 0.075828 | 0.371861 | 0.057368 | 0.365926 | 72.55% |
| 7 factors / rolling_6y | PASS | 0.059225 | 0.074931 | 0.335884 | 0.066413 | 0.446510 | 64.71% |
| 7 factors / fixed | PASS | 0.059273 | 0.051195 | 0.226353 | 0.038516 | 0.260159 | 60.78% |
| 7 + 4 macro / rolling_3y | PASS | 0.059576 | 0.044988 | 0.196070 | 0.009170 | 0.052307 | 54.90% |
| 7 + 4 macro / rolling_6y | PASS | 0.059474 | 0.069173 | 0.275941 | 0.061956 | 0.385461 | 62.75% |
| 7 + 4 macro / fixed | PASS | 0.059268 | 0.066680 | 0.280742 | 0.047469 | 0.325353 | 62.75% |
| 45 factors / rolling_3y | FAIL | 0.059456 | 0.009774 | 0.043508 | -0.011817 | -0.065290 | 52.94% |
| 45 factors / rolling_6y | PASS | 0.059387 | 0.026115 | 0.101713 | 0.018900 | 0.108219 | 56.86% |
| 45 factors / fixed | PASS | 0.059300 | 0.034033 | 0.144038 | 0.026908 | 0.179515 | 58.82% |

### Portfolio performance and sample counts

Excess annualized return is expressed in percentage points (pp).

| Experiment | Annualized return | Benchmark | Excess annualized return | Information ratio | Weeks | Valid Rank IC weeks | Observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7 factors / rolling_3y | 10.93% | 6.43% | 4.50 pp | 0.344540 | 51 | 51 | 3306 |
| 7 factors / rolling_6y | 5.06% | 6.43% | -1.37 pp | 0.062483 | 51 | 51 | 3306 |
| 7 factors / fixed | 8.55% | 6.43% | 2.12 pp | 0.219297 | 51 | 51 | 3306 |
| 7 + 4 macro / rolling_3y | -18.02% | 6.43% | -24.45 pp | -0.919423 | 51 | 51 | 3306 |
| 7 + 4 macro / rolling_6y | 22.51% | 6.43% | 16.08 pp | 0.964870 | 51 | 51 | 3306 |
| 7 + 4 macro / fixed | 13.02% | 6.43% | 6.59 pp | 0.471355 | 51 | 51 | 3306 |
| 45 factors / rolling_3y | -36.56% | 6.43% | -42.99 pp | -2.452582 | 51 | 51 | 3306 |
| 45 factors / rolling_6y | -25.74% | 6.43% | -32.17 pp | -1.957740 | 51 | 51 | 3306 |
| 45 factors / fixed | 1.81% | 6.43% | -4.62 pp | -0.146052 | 51 | 51 | 3306 |

### Interpretation

The **7 + 4 macro / rolling_6y** experiment achieved the highest observed Final Test annualized return (22.51%), excess annualized return (16.08 pp), and information ratio (0.964870). Its selected configuration was candidate10: 100 trees, maximum depth 6, minimum leaf size 20, and `max_features=0.5`.

The **7 factors / rolling_6y** experiment had the highest mean Rank IC (0.066413) and Rank ICIR (0.446510), but its Top-5 portfolio underperformed the benchmark by 1.37 pp. Full-universe ranking quality and returns from the five highest-ranked stocks measure different outcomes.

The **7 factors / rolling_3y** experiment had the highest positive Rank IC ratio (72.55%) and lowest RMSE (0.059206). All three 45-factor experiments underperformed the benchmark; the three-year version also failed the IC screening conditions.

Adding macro factors improved observed portfolio returns under the six-year and fixed-stage designs but reduced returns under the three-year design. This comparison does not isolate the causal effect of macro factors because each experiment selects its own hyperparameters, and the rolling designs use different Development histories.

These are descriptive Final Test comparisons. Selecting a new preferred strategy from these results would use the test set for model selection and should be followed by a new untouched evaluation period.

## Metric definitions

| Metric | Definition used in the script |
|---|---|
| RMSE | Square root of the mean squared prediction error over stock-week observations, in target-return units |
| IC | Mean weekly cross-sectional Pearson correlation between predictions and actual returns |
| ICIR | Mean weekly IC divided by its sample standard deviation; not annualized |
| Rank IC | Mean weekly cross-sectional Spearman correlation |
| Rank ICIR | Mean weekly Rank IC divided by its sample standard deviation; not annualized |
| Positive Rank IC Ratio | Fraction of valid Rank IC weeks with Rank IC strictly greater than zero |
| Annualized Return | Geometric Top-5 return: `(product(1 + weekly_return)) ** (52 / weeks) - 1` |
| Benchmark Annualized Return | The same geometric formula applied to weekly equal-weight benchmark returns |
| Excess Annualized Return | Strategy annualized return minus benchmark annualized return; displayed in pp |
| Information Ratio | Mean weekly strategy-minus-benchmark return divided by its sample standard deviation, multiplied by `sqrt(52)` |
| Weeks | Number of evaluated portfolio weeks |
| Valid Rank IC Weeks | Number of weeks with a defined Rank IC |
| Observations | Number of stock-week prediction rows supplied to the summary |

Information ratio uses arithmetic weekly excess returns, whereas excess annualized return subtracts two geometric annualized returns. Their signs can differ, as in the seven-factor six-year rolling result.

## Run the code

Rename the Python script to **`random_forest.py`**. Place these files in the same directory:

- `random_forest.py`
- `test_45_factors_weekly.csv`
- `Single_Factor_Screening_66_Stocks.xlsx`

Install the required packages:

```bash
python -m pip install numpy pandas scipy scikit-learn openpyxl
```

Run all nine experiments:

```bash
python random_forest.py
```

Check inputs and stage coverage without fitting models:

```bash
python random_forest.py --check-only
```

Specify input paths if your filenames differ:

```bash
python random_forest.py --data-file "your_weekly_data.csv" --screening-file "your_screening_workbook.xlsx"
```

For a one-week simple-return target:

```bash
python random_forest.py --target-column fwd_1w_return --target-type simple
```

An optional `--universe-file` accepts a CSV with a `ticker` column. `--expected-stocks` defaults to 66. The script prints results to the terminal and does not automatically export CSV files.

## Terminal output

For each experiment, the script prints the feature list, all candidate hyperparameters and Initial Training metrics, Development rankings and Top 5, Validation rankings, frozen parameters, four-stage metrics, fitting schedules, and mean Final Test feature importances across fits. The final table compares all nine experiments and includes every metric listed above.

## Limitations

- Full-sample factor screening can introduce selection look-ahead even when model fitting respects label availability.
- Final Test covers 51 evaluated weeks, so results provide limited evidence across market regimes.
- Transaction costs and slippage are excluded; weekly Top-5 rebalancing may materially reduce net returns.
- Point-in-time universe membership, corporate-action handling, and factor publication timing depend on the supplied data and are not verified by the script.
- Macro variables must reflect information available on each prediction date. Their inclusion in a model does not by itself establish that they improve cross-sectional selection.
- Initial Training metrics are in-sample. Development metrics reflect tuning, and Validation metrics reflect selection among five candidates.
- The label-timing rule assumes the next panel observation represents the end of the one-week target horizon; this must match the dataset's construction.
