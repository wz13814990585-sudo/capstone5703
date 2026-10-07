from __future__ import annotations

import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost

from .config import (
    FEATURE_BLOCKS,
    FEATURE_COMBINATION_BLOCKS,
    FEATURE_COMBINATIONS,
    ExperimentConfig,
)
from .data import eligible_training_rows, load_and_prepare_data
from .metrics import calculate_weekly_metrics, summarize_weekly_metrics
from .modeling import fit_ranker, predict_ranking_block
from .pipeline import build_refit_weeks, rolling_oos, tune_warmup


METRIC_COLUMNS = [
    "IC",
    "ICIR",
    "Rank_IC",
    "Rank_ICIR",
    "Positive_IC_Ratio",
    "Positive_Rank_IC_Ratio",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _first_oos_week(frame: pd.DataFrame, config: ExperimentConfig) -> int:
    refits = build_refit_weeks(
        int(frame["week_index"].nunique()),
        config.time.warmup_weeks,
        config.time.forecast_horizon_weeks,
        config.time.refit_interval_weeks,
    )
    if not refits:
        raise ValueError("No OOS week remains after Warm-up and observability embargo")
    return refits[0]


def _prefix_rolling_outputs(
    feature_set_name: str,
    predictions: pd.DataFrame,
    weekly: pd.DataFrame,
    refit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_ids = sorted(predictions["model_id"].unique())
    mapping = {
        old: f"{feature_set_name}_rolling_{ordinal:03d}"
        for ordinal, old in enumerate(old_ids, start=1)
    }
    for table in (predictions, weekly, refit):
        table["model_id"] = table["model_id"].map(mapping)
        table["training_mode"] = "rolling"
        table["factor_set_name"] = feature_set_name
    return predictions, weekly, refit


def fixed_oos(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train once on the first OOS historical window and never refit."""
    total_weeks = int(frame["week_index"].nunique())
    first_oos = _first_oos_week(frame, config)
    candidate_start = max(0, first_oos - config.time.rolling_window_weeks)
    candidate_weeks = list(range(candidate_start, first_oos))
    training_rows = eligible_training_rows(frame, candidate_weeks, first_oos)
    if training_rows.empty:
        raise ValueError("Fixed-window training set is empty")
    if training_rows["week_index"].nunique() > config.time.rolling_window_weeks:
        raise AssertionError("Fixed training window exceeds configured window")

    model = fit_ranker(
        training_rows=training_rows,
        features=config.features,
        objective=str(frozen["objective"]),
        relevance_bins=int(frozen["relevance_bins"]),
        params=dict(frozen["hyperparameters"]),
        random_seed=config.random_seed,
        n_jobs=config.n_jobs,
    )
    prediction_rows = frame.loc[frame["week_index"].ge(first_oos)].copy()
    refit_date = pd.Timestamp(
        frame.loc[frame["week_index"].eq(first_oos), "date"].iloc[0]
    )
    model_id = f"{config.factor_set_name}_fixed_001"
    predictions = predict_ranking_block(
        model=model,
        prediction_rows=prediction_rows,
        features=config.features,
        relevance_bins=int(frozen["relevance_bins"]),
        model_id=model_id,
        refit_week=first_oos,
        refit_date=refit_date,
        factor_set_name=config.factor_set_name,
    )
    predictions["training_mode"] = "fixed"
    weekly = calculate_weekly_metrics(predictions)
    weekly["training_mode"] = "fixed"
    weekly["factor_set_name"] = config.factor_set_name
    overall = summarize_weekly_metrics(weekly)
    overall.update(
        {
            "first_prediction_week": first_oos,
            "last_prediction_week": total_weeks - 1,
            "number_of_refit_models": 1,
            "training_mode": "fixed",
            "fixed_candidate_window_weeks": config.time.rolling_window_weeks,
            "forecast_horizon_weeks": config.time.forecast_horizon_weeks,
            "warmup_to_oos_embargo_weeks": config.time.forecast_horizon_weeks,
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "training_mode": "fixed",
                "refit_week": first_oos,
                "refit_date": refit_date,
                "requested_window_start_week": candidate_start,
                "requested_window_end_week": first_oos - 1,
                "training_start_week": int(training_rows["week_index"].min()),
                "training_end_week": int(training_rows["week_index"].max()),
                "number_of_training_weeks": int(
                    training_rows["week_index"].nunique()
                ),
                "number_of_training_rows": int(len(training_rows)),
                "factor_set_name": config.factor_set_name,
                "number_of_features": len(config.features),
                "feature_list": json.dumps(config.features),
                "objective": frozen["objective"],
                "relevance_bins": frozen["relevance_bins"],
                "hyperparameters": json.dumps(
                    frozen["hyperparameters"], sort_keys=True
                ),
                **summarize_weekly_metrics(weekly),
            }
        ]
    )
    expected_weeks = list(range(first_oos, total_weeks))
    if sorted(predictions["week_index"].unique().tolist()) != expected_weeks:
        raise AssertionError("Fixed predictions do not cover the full OOS period")
    if predictions["model_id"].nunique() != 1:
        raise AssertionError("Fixed-window mode must use exactly one model")
    return predictions, weekly, metadata, overall


def rolling_for_comparison(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions, weekly, refit, overall = rolling_oos(frame, config, frozen)
    predictions, weekly, refit = _prefix_rolling_outputs(
        config.factor_set_name, predictions, weekly, refit
    )
    overall["training_mode"] = "rolling"
    return predictions, weekly, refit, overall


def _comparison_row(
    feature_set_name: str,
    config: ExperimentConfig,
    frozen: dict[str, Any],
    training_mode: str,
    overall: dict[str, Any],
) -> dict[str, Any]:
    return {
        "factor_set_name": feature_set_name,
        "blocks": FEATURE_COMBINATION_BLOCKS[feature_set_name],
        "number_of_features": len(config.features),
        "feature_list": json.dumps(config.features),
        "training_mode": training_mode,
        "objective": frozen["objective"],
        "relevance_bins": frozen["relevance_bins"],
        "selected_candidate_id": frozen["selected_candidate_id"],
        **overall,
    }


def _make_delta_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set_name, group in comparison.groupby("factor_set_name", sort=False):
        by_mode = group.set_index("training_mode")
        if not {"fixed", "rolling"}.issubset(by_mode.index):
            raise AssertionError(f"Missing training mode for {feature_set_name}")
        row: dict[str, Any] = {
            "factor_set_name": feature_set_name,
            "blocks": group["blocks"].iloc[0],
            "number_of_features": int(group["number_of_features"].iloc[0]),
        }
        for metric in METRIC_COLUMNS:
            fixed_value = float(by_mode.loc["fixed", metric])
            rolling_value = float(by_mode.loc["rolling", metric])
            row[f"fixed_{metric}"] = fixed_value
            row[f"rolling_{metric}"] = rolling_value
            row[f"rolling_minus_fixed_{metric}"] = rolling_value - fixed_value
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_definition_table(feature_sets: list[str]) -> pd.DataFrame:
    descriptions = {
        "F": "Fundamental: revenue growth and cash-flow-to-assets",
        "R": "Return horizons: 120-day momentum plus 1-day and 5-day returns",
        "P": "Price action: close location and body/range ratio",
    }
    rows = []
    for name in feature_sets:
        blocks = FEATURE_COMBINATION_BLOCKS[name]
        rows.append(
            {
                "factor_set_name": name,
                "blocks": blocks,
                "block_descriptions": " | ".join(
                    descriptions[block] for block in blocks.split("+")
                ),
                "number_of_features": len(FEATURE_COMBINATIONS[name]),
                "feature_list": json.dumps(FEATURE_COMBINATIONS[name]),
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    diagnostics: dict[str, Any],
    base_config: ExperimentConfig,
    feature_definitions: pd.DataFrame,
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    warmup: pd.DataFrame,
) -> str:
    metric_view = comparison[
        [
            "factor_set_name",
            "training_mode",
            "number_of_features",
            "IC",
            "ICIR",
            "Rank_IC",
            "Rank_ICIR",
            "Positive_Rank_IC_Ratio",
            "number_of_prediction_weeks",
        ]
    ].sort_values(["factor_set_name", "training_mode"])
    delta_view = deltas[
        [
            "factor_set_name",
            "rolling_minus_fixed_IC",
            "rolling_minus_fixed_ICIR",
            "rolling_minus_fixed_Rank_IC",
            "rolling_minus_fixed_Rank_ICIR",
        ]
    ]
    warmup_view = warmup.loc[
        warmup["is_selected"],
        [
            "factor_set_name",
            "candidate_id",
            "objective",
            "relevance_bins",
            "Rank_IC",
            "Rank_ICIR",
        ],
    ]
    best = comparison.sort_values(
        ["Rank_IC", "Rank_ICIR"], ascending=False, kind="stable"
    ).iloc[0]
    return f"""# XGBRanker 固定窗口 vs 滚动窗口、Feature 组合比较

## 实验定义

- 数据：`{diagnostics['input_filename']}`，{diagnostics['tickers']} 只股票，{diagnostics['weeks']} 周，{diagnostics['rows']} 条记录。
- Target：连续下一周周末调整收盘价对数收益 `target`。
- Warm-up：{base_config.time.warmup_weeks} 周（{base_config.time.warmup_train_weeks} train + {base_config.time.warmup_validation_weeks} validation）。
- OOS 前保留 {base_config.time.forecast_horizon_weeks} 周 label-observability embargo。
- 滚动窗口：每次从最近 {base_config.time.rolling_window_weeks} 个候选周取严格可观测样本，每 {base_config.time.refit_interval_weeks} 周重训。
- 固定窗口：在首个 OOS 时点使用同一个首期候选窗口训练一次，之后不再 refit，但每周仍输入新特征做排序。
- 两种方法对每个 feature set 共用该组的 Warm-up 冻结超参和完全相同的 OOS 周。
- 本次不生成策略、仓位或回测结果。

## Feature 组合方法

将已确认的 Priority 7 预先分成三个有经济含义的 block：

```text
F = {FEATURE_BLOCKS['F']}
R = {FEATURE_BLOCKS['R']}
P = {FEATURE_BLOCKS['P']}
```

实验使用三个 block 的全部 7 个非空组合，而不是在 OOS 结果上搜索 127 个任意子集：

```text
{feature_definitions[['factor_set_name', 'blocks', 'feature_list']].to_string(index=False)}
```

## 每个 Feature 组合的 OOS Metrics

```text
{metric_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## Rolling 减 Fixed

正值表示滚动窗口更高，负值表示固定窗口更高。

```text
{delta_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## Warm-up 选中配置

```text
{warmup_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

按 OOS Rank IC 描述性排序，最高的已观察组合为 `{best['factor_set_name']}` + `{best['training_mode']}`，Rank IC={best['Rank_IC']:.6f}，Rank ICIR={best['Rank_ICIR']:.6f}。这只是比较结果，没有用来反向修改因子集或超参。

## 泄漏防护

- 所有训练行必须同时满足 `feature_week < refit_week` 和 `label_end_week < refit_week`。
- Feature 组合和 block 在 OOS 比较前预先固定。
- 每周独立生成 relevance label 和 ranking group。
- 超参只在 Warm-up 中选择；固定和滚动方法都不使用 OOS 结果调参。
"""


def run_feature_comparison(
    base_config: ExperimentConfig,
    feature_sets: list[str] | None = None,
) -> dict[str, Any]:
    base_config.validate()
    selected = feature_sets or list(FEATURE_COMBINATIONS)
    unknown = sorted(set(selected).difference(FEATURE_COMBINATIONS))
    if unknown:
        raise ValueError(f"Unknown comparison feature sets: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("Feature-set list contains duplicates")

    union_features = list(
        dict.fromkeys(
            feature
            for feature_set_name in selected
            for feature in FEATURE_COMBINATIONS[feature_set_name]
        )
    )
    frame, diagnostics = load_and_prepare_data(
        input_path=base_config.input_path,
        factor_definitions_path=base_config.factor_definitions_path,
        features=union_features,
        forecast_horizon_weeks=base_config.time.forecast_horizon_weeks,
    )
    if frame["week_index"].nunique() <= (
        base_config.time.warmup_weeks
        + base_config.time.forecast_horizon_weeks
    ):
        raise ValueError("Not enough weeks for Warm-up, embargo, and OOS evaluation")

    warmup_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    weekly_tables: list[pd.DataFrame] = []
    refit_tables: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    frozen_configs: dict[str, Any] = {}

    for ordinal, feature_set_name in enumerate(selected, start=1):
        print(
            f"[{ordinal}/{len(selected)}] tuning and evaluating {feature_set_name}",
            flush=True,
        )
        config = replace(base_config, factor_set_name=feature_set_name)
        config.validate()
        warmup, frozen = tune_warmup(frame, config)
        warmup.insert(0, "factor_set_name", feature_set_name)
        warmup["is_selected"] = warmup["candidate_id"].eq(
            frozen["selected_candidate_id"]
        )
        warmup_tables.append(warmup)
        frozen_configs[feature_set_name] = frozen

        rolling_predictions, rolling_weekly, rolling_refit, rolling_overall = (
            rolling_for_comparison(frame, config, frozen)
        )
        fixed_predictions, fixed_weekly, fixed_refit, fixed_overall = fixed_oos(
            frame, config, frozen
        )
        if sorted(rolling_predictions["week_index"].unique()) != sorted(
            fixed_predictions["week_index"].unique()
        ):
            raise AssertionError("Fixed and rolling OOS weeks must match exactly")

        prediction_tables.extend([rolling_predictions, fixed_predictions])
        weekly_tables.extend([rolling_weekly, fixed_weekly])
        refit_tables.extend([rolling_refit, fixed_refit])
        comparison_rows.extend(
            [
                _comparison_row(
                    feature_set_name,
                    config,
                    frozen,
                    "rolling",
                    rolling_overall,
                ),
                _comparison_row(
                    feature_set_name,
                    config,
                    frozen,
                    "fixed",
                    fixed_overall,
                ),
            ]
        )

    warmup_all = pd.concat(warmup_tables, ignore_index=True)
    predictions_all = pd.concat(prediction_tables, ignore_index=True)
    weekly_all = pd.concat(weekly_tables, ignore_index=True)
    refit_all = pd.concat(refit_tables, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["factor_set_name", "training_mode"], kind="stable"
    )
    deltas = _make_delta_table(comparison)
    definitions = _feature_definition_table(selected)

    key = ["factor_set_name", "training_mode", "date", "ticker"]
    if predictions_all.duplicated(key).any():
        raise AssertionError("Comparison predictions contain duplicate keys")
    expected_rows = (
        predictions_all.groupby(["factor_set_name", "training_mode"])
        .size()
        .unique()
    )
    if len(expected_rows) != 1:
        raise AssertionError("Every feature/mode comparison must cover identical OOS rows")

    out = base_config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    definitions.to_csv(out / "feature_combination_definitions.csv", index=False)
    warmup_all.to_csv(out / "warmup_hyperparameter_results.csv", index=False)
    _write_json(out / "frozen_configs_by_feature_set.json", frozen_configs)
    comparison.to_csv(out / "model_comparison_metrics.csv", index=False)
    deltas.to_csv(out / "rolling_vs_fixed_deltas.csv", index=False)
    predictions_all.to_csv(out / "xgbranker_all_weekly_predictions.csv", index=False)
    weekly_all.to_csv(out / "xgbranker_all_weekly_metrics.csv", index=False)
    refit_all.to_csv(out / "xgbranker_all_refit_model_metrics.csv", index=False)
    _write_json(out / "data_diagnostics.json", diagnostics)

    manifest = {
        "input_filename": diagnostics["input_filename"],
        "input_sha256": diagnostics["input_sha256"],
        "feature_combination_method": "pre-registered full non-empty combinations of F/R/P economic blocks",
        "feature_blocks": FEATURE_BLOCKS,
        "feature_sets": {
            name: FEATURE_COMBINATIONS[name] for name in selected
        },
        "training_modes": {
            "rolling": "trailing configured window, refit at configured interval",
            "fixed": "train once on the first OOS historical window; never refit",
        },
        "time_settings": asdict(base_config.time),
        "random_seed": base_config.random_seed,
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "target_column": base_config.target_column,
        "warmup_candidate_space": base_config.candidates,
        "backtest_performed": False,
    }
    _write_json(out / "experiment_manifest.json", manifest)
    report = _build_report(
        diagnostics,
        base_config,
        definitions,
        comparison,
        deltas,
        warmup_all,
    )
    (out / "comparison_report.md").write_text(report, encoding="utf-8")
    return {
        "diagnostics": diagnostics,
        "feature_sets": selected,
        "number_of_comparisons": int(len(comparison)),
        "comparison_metrics": comparison.to_dict(orient="records"),
        "output_dir": str(out.resolve()),
        "backtest_performed": False,
    }
