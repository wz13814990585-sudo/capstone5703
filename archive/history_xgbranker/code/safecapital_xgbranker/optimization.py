from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost

from .config import (
    MACRO_4,
    OPTIMIZATION_FEATURE_SETS,
    ExperimentConfig,
    TimeConfig,
)
from .data import eligible_training_rows, load_and_prepare_data
from .metrics import calculate_weekly_metrics, summarize_weekly_metrics
from .modeling import fit_ranker, predict_ranking_block
from .pipeline import tune_warmup


@dataclass(frozen=True)
class SearchConfig:
    hyperparameter_train_weeks: int = 208
    hyperparameter_validation_weeks: int = 52
    selection_start_week: int = 312
    final_holdout_weeks: int = 104
    screening_rolling_window_weeks: int = 156
    screening_refit_interval_weeks: int = 4
    rolling_window_candidates: tuple[int, ...] = (104, 156, 208, 260, 312)
    refit_interval_candidates: tuple[int, ...] = (2, 4, 8, 13)
    fixed_window_candidates: tuple[str | int, ...] = (
        "all_history",
        156,
        208,
        260,
        312,
    )
    top_feature_sets_per_mode: int = 2

    @property
    def hyperparameter_warmup_weeks(self) -> int:
        return self.hyperparameter_train_weeks + self.hyperparameter_validation_weeks

    def final_start_week(self, total_weeks: int) -> int:
        return total_weeks - self.final_holdout_weeks

    def validate(self, total_weeks: int) -> None:
        if self.hyperparameter_warmup_weeks >= self.selection_start_week:
            raise ValueError("Hyperparameter period must end before selection OOS")
        final_start = self.final_start_week(total_weeks)
        if self.selection_start_week >= final_start:
            raise ValueError("Selection OOS must start before the final holdout")
        if self.final_holdout_weeks <= 0:
            raise ValueError("final_holdout_weeks must be positive")
        if any(value <= 0 for value in self.rolling_window_candidates):
            raise ValueError("Rolling windows must be positive")
        if any(value <= 0 for value in self.refit_interval_candidates):
            raise ValueError("Refit intervals must be positive")
        if self.top_feature_sets_per_mode <= 0:
            raise ValueError("top_feature_sets_per_mode must be positive")


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


def _window_label(window: str | int | None) -> str:
    return "all_history" if window in {None, "all_history"} else str(int(window))


def evaluate_schedule(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    frozen: dict[str, Any],
    *,
    phase: str,
    evaluation_start_week: int,
    evaluation_end_week: int,
    training_mode: str,
    training_window_weeks: str | int,
    refit_interval_weeks: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if training_mode not in {"fixed", "rolling"}:
        raise ValueError("training_mode must be fixed or rolling")
    if evaluation_start_week >= evaluation_end_week:
        raise ValueError("Evaluation period is empty")
    window = None if training_window_weeks == "all_history" else int(training_window_weeks)
    if window is not None and window <= 0:
        raise ValueError("training_window_weeks must be positive")
    if training_mode == "fixed":
        refit_weeks = [evaluation_start_week]
        interval = evaluation_end_week - evaluation_start_week
    else:
        if refit_interval_weeks is None or refit_interval_weeks <= 0:
            raise ValueError("Rolling mode requires a positive refit interval")
        interval = int(refit_interval_weeks)
        refit_weeks = list(range(evaluation_start_week, evaluation_end_week, interval))

    window_name = _window_label(window)
    config_id = (
        f"fixed_{window_name}"
        if training_mode == "fixed"
        else f"rolling_w{window_name}_r{interval}"
    )
    prediction_blocks: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for ordinal, refit_week in enumerate(refit_weeks, start=1):
        candidate_start = 0 if window is None else max(0, refit_week - window)
        candidate_weeks = list(range(candidate_start, refit_week))
        training_rows = eligible_training_rows(frame, candidate_weeks, refit_week)
        if training_rows.empty:
            raise ValueError(f"No training data for {config_id} at week {refit_week}")
        model = fit_ranker(
            training_rows=training_rows,
            features=config.features,
            objective=str(frozen["objective"]),
            relevance_bins=int(frozen["relevance_bins"]),
            params=dict(frozen["hyperparameters"]),
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
        )
        block_end = (
            evaluation_end_week
            if training_mode == "fixed"
            else min(refit_week + interval, evaluation_end_week)
        )
        prediction_rows = frame.loc[
            frame["week_index"].between(refit_week, block_end - 1)
        ].copy()
        refit_date = pd.Timestamp(
            frame.loc[frame["week_index"].eq(refit_week), "date"].iloc[0]
        )
        model_id = (
            f"{config.factor_set_name}_{phase}_{config_id}_{ordinal:03d}"
        )
        block = predict_ranking_block(
            model=model,
            prediction_rows=prediction_rows,
            features=config.features,
            relevance_bins=int(frozen["relevance_bins"]),
            model_id=model_id,
            refit_week=refit_week,
            refit_date=refit_date,
            factor_set_name=config.factor_set_name,
        )
        block["phase"] = phase
        block["config_id"] = config_id
        block["training_mode"] = training_mode
        block["training_window_weeks"] = window_name
        block["refit_interval_weeks"] = (
            np.nan if training_mode == "fixed" else interval
        )
        prediction_blocks.append(block)
        metadata_rows.append(
            {
                "phase": phase,
                "config_id": config_id,
                "factor_set_name": config.factor_set_name,
                "training_mode": training_mode,
                "training_window_weeks": window_name,
                "refit_interval_weeks": (
                    np.nan if training_mode == "fixed" else interval
                ),
                "model_id": model_id,
                "refit_week": refit_week,
                "refit_date": refit_date,
                "candidate_start_week": candidate_start,
                "candidate_end_week": refit_week - 1,
                "training_start_week": int(training_rows["week_index"].min()),
                "training_end_week": int(training_rows["week_index"].max()),
                "number_of_training_weeks": int(
                    training_rows["week_index"].nunique()
                ),
                "number_of_training_rows": int(len(training_rows)),
                "number_of_features": len(config.features),
                "objective": frozen["objective"],
                "relevance_bins": frozen["relevance_bins"],
            }
        )

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    expected_weeks = list(range(evaluation_start_week, evaluation_end_week))
    if sorted(predictions["week_index"].unique().tolist()) != expected_weeks:
        raise AssertionError(f"{config_id} does not cover every evaluation week")
    if predictions.duplicated(["date", "ticker"]).any():
        raise AssertionError(f"{config_id} has duplicate date+ticker predictions")
    if not predictions.groupby("week_index")["model_id"].nunique().eq(1).all():
        raise AssertionError(f"{config_id} maps a week to multiple models")
    metadata = pd.DataFrame(metadata_rows)
    if not (metadata["training_end_week"] + config.time.forecast_horizon_weeks).lt(
        metadata["refit_week"]
    ).all():
        raise AssertionError(f"{config_id} violates strict label observability")

    weekly = calculate_weekly_metrics(predictions)
    weekly["phase"] = phase
    weekly["config_id"] = config_id
    weekly["factor_set_name"] = config.factor_set_name
    weekly["training_mode"] = training_mode
    weekly["training_window_weeks"] = window_name
    weekly["refit_interval_weeks"] = (
        np.nan if training_mode == "fixed" else interval
    )
    overall = summarize_weekly_metrics(weekly)
    overall.update(
        {
            "phase": phase,
            "config_id": config_id,
            "factor_set_name": config.factor_set_name,
            "feature_list": json.dumps(config.features),
            "number_of_features": len(config.features),
            "training_mode": training_mode,
            "training_window_weeks": window_name,
            "refit_interval_weeks": (
                np.nan if training_mode == "fixed" else interval
            ),
            "number_of_refit_models": len(refit_weeks),
            "evaluation_start_week": evaluation_start_week,
            "evaluation_end_week": evaluation_end_week - 1,
            "objective": frozen["objective"],
            "relevance_bins": frozen["relevance_bins"],
            "selected_candidate_id": frozen["selected_candidate_id"],
        }
    )
    return predictions, weekly, metadata, overall


def _best_row(table: pd.DataFrame) -> pd.Series:
    # If fixed all-history and the largest finite window are identical during
    # selection (because all available history fits inside that window), prefer
    # all-history.  This is a pre-holdout, data-volume tie-break rather than an
    # arbitrary lexical config-id tie-break.
    fixed_history_preference = np.where(
        table["training_mode"].eq("fixed")
        & table["training_window_weeks"].astype(str).eq("all_history"),
        1,
        0,
    )
    ranked = table.assign(
        _rank_ic=table["Rank_IC"].fillna(-np.inf),
        _rank_icir=table["Rank_ICIR"].fillna(-np.inf),
        _fixed_history_preference=fixed_history_preference,
    ).sort_values(
        [
            "_rank_ic",
            "_rank_icir",
            "_fixed_history_preference",
            "factor_set_name",
            "config_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    return ranked.iloc[0].drop(
        labels=["_rank_ic", "_rank_icir", "_fixed_history_preference"]
    )


def _feature_catalog(feature_sets: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_sets:
        features = OPTIMIZATION_FEATURE_SETS[name]
        rows.append(
            {
                "factor_set_name": name,
                "number_of_features": len(features),
                "includes_macro": bool(set(features).intersection(MACRO_4)),
                "feature_list": json.dumps(features),
            }
        )
    return pd.DataFrame(rows)


def _report(
    diagnostics: dict[str, Any],
    search: SearchConfig,
    screening: pd.DataFrame,
    grid: pd.DataFrame,
    final_metrics: pd.DataFrame,
    selected_feature_sets: list[str],
    primary_key: tuple[str, str],
) -> str:
    screen_view = screening[
        [
            "factor_set_name",
            "training_mode",
            "number_of_features",
            "Rank_IC",
            "Rank_ICIR",
        ]
    ].sort_values("Rank_IC", ascending=False)
    grid_view = grid[
        [
            "factor_set_name",
            "config_id",
            "training_mode",
            "training_window_weeks",
            "refit_interval_weeks",
            "Rank_IC",
            "Rank_ICIR",
        ]
    ].sort_values("Rank_IC", ascending=False).head(25)
    final_view = final_metrics[
        [
            "finalist_role",
            "is_primary_pre_holdout_selection",
            "factor_set_name",
            "config_id",
            "IC",
            "ICIR",
            "Rank_IC",
            "Rank_ICIR",
            "Positive_Rank_IC_Ratio",
        ]
    ]
    primary_feature, primary_config = primary_key
    return f"""# XGBRanker Rank IC 优化与最终检验报告

## 方法

数据共 {diagnostics['weeks']} 周、{diagnostics['tickers']} 只股票、{diagnostics['rows']} 条记录。为避免在最终样本上反复搜索，使用严格三阶段时间隔离：

1. Week 0–{search.hyperparameter_warmup_weeks - 1}：超参期，{search.hyperparameter_train_weeks} 周 train + {search.hyperparameter_validation_weeks} 周 validation。
2. Week {search.selection_start_week}–{search.final_start_week(diagnostics['weeks']) - 1}：研究选择期，用于 Feature、Fixed 长度、Rolling 窗口和 refit 周期选择。
3. Week {search.final_start_week(diagnostics['weeks'])}–{diagnostics['weeks'] - 1}：最后 {search.final_holdout_weeks} 周 holdout，不参与任何选择。

所有阶段仍强制 `label_end_week < refit_week`。最终 holdout 只评估在研究选择期预先确定的 fixed 和 rolling 候选，不用 holdout 结果返回修改参数。

## Fixed 训练集大小

已比较：`{list(search.fixed_window_candidates)}`。`all_history` 表示在 OOS 开始时使用当时所有严格可观测历史，然后整个评价期不再 refit。在最终 104 周 holdout 下，`all_history` 接近 80% 开发 / 20% 最终测试，比原先 156 周训练、365 周 OOS 的分配更适合 fixed model。

## Feature 筛选

预注册 Feature 集数：{screening['factor_set_name'].nunique()}。包含 Priority 子组合、基本面/收益/趋势/波动/流动性等经济类别、所有 31 个非基本面因子、全部 37 个非重复股票因子，以及显式的 +Macro 实验。进入精细时间网格的 Feature 集：`{selected_feature_sets}`。

Feature screening 结果：

```text
{screen_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 时间参数搜索

- Rolling windows: `{list(search.rolling_window_candidates)}`
- Refit intervals: `{list(search.refit_interval_candidates)}`
- Fixed windows: `{list(search.fixed_window_candidates)}`

选择期 Rank IC 最高的 25 组：

```text
{grid_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 最终 Holdout

在查看最终 holdout 之前，主配置已冻结为 `{primary_feature}` + `{primary_config}`。同时保留选择期最优 fixed 和最优 rolling 各一个作诊断比较，但不根据最终 holdout 重选配置。

```text
{final_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

本项目只运行 XGBRanker 机器学习与 IC/Rank IC 评估，未运行策略或回测。
"""


def run_rank_ic_optimization(
    base_config: ExperimentConfig,
    search: SearchConfig | None = None,
    feature_sets: list[str] | None = None,
) -> dict[str, Any]:
    search = search or SearchConfig()
    selected_features = feature_sets or list(OPTIMIZATION_FEATURE_SETS)
    unknown = sorted(set(selected_features).difference(OPTIMIZATION_FEATURE_SETS))
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")
    if len(selected_features) != len(set(selected_features)):
        raise ValueError("Feature sets must be unique")

    union_features = list(
        dict.fromkeys(
            feature
            for name in selected_features
            for feature in OPTIMIZATION_FEATURE_SETS[name]
        )
    )
    tuning_time = TimeConfig(
        warmup_weeks=search.hyperparameter_warmup_weeks,
        warmup_train_weeks=search.hyperparameter_train_weeks,
        warmup_validation_weeks=search.hyperparameter_validation_weeks,
        rolling_window_weeks=search.screening_rolling_window_weeks,
        refit_interval_weeks=search.screening_refit_interval_weeks,
        forecast_horizon_weeks=base_config.time.forecast_horizon_weeks,
    )
    base_config = replace(base_config, time=tuning_time)
    base_config.validate()
    frame, diagnostics = load_and_prepare_data(
        input_path=base_config.input_path,
        factor_definitions_path=base_config.factor_definitions_path,
        features=union_features,
        forecast_horizon_weeks=base_config.time.forecast_horizon_weeks,
    )
    total_weeks = int(frame["week_index"].nunique())
    search.validate(total_weeks)
    final_start = search.final_start_week(total_weeks)

    warmup_tables: list[pd.DataFrame] = []
    frozen_configs: dict[str, dict[str, Any]] = {}
    screening_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str, int | None], tuple] = {}

    print(
        f"Stage 1/3: tuning and screening {len(selected_features)} feature sets",
        flush=True,
    )
    for ordinal, feature_set_name in enumerate(selected_features, start=1):
        print(
            f"  feature {ordinal}/{len(selected_features)}: {feature_set_name}",
            flush=True,
        )
        config = replace(base_config, factor_set_name=feature_set_name)
        warmup, frozen = tune_warmup(frame, config)
        warmup.insert(0, "factor_set_name", feature_set_name)
        warmup["is_selected"] = warmup["candidate_id"].eq(
            frozen["selected_candidate_id"]
        )
        warmup_tables.append(warmup)
        frozen_configs[feature_set_name] = frozen

        screening_specs = [
            ("fixed", "all_history", None),
            (
                "rolling",
                str(search.screening_rolling_window_weeks),
                search.screening_refit_interval_weeks,
            ),
        ]
        for mode, window, interval in screening_specs:
            result = evaluate_schedule(
                frame,
                config,
                frozen,
                phase="selection",
                evaluation_start_week=search.selection_start_week,
                evaluation_end_week=final_start,
                training_mode=mode,
                training_window_weeks=window,
                refit_interval_weeks=interval,
            )
            key = (feature_set_name, mode, str(window), interval)
            cache[key] = result
            screening_rows.append(result[3])

    warmup_all = pd.concat(warmup_tables, ignore_index=True)
    screening = pd.DataFrame(screening_rows)
    top_feature_names: list[str] = []
    for mode in ("fixed", "rolling"):
        mode_table = screening.loc[screening["training_mode"].eq(mode)].copy()
        mode_table = mode_table.sort_values(
            ["Rank_IC", "Rank_ICIR"], ascending=False, kind="stable"
        ).head(search.top_feature_sets_per_mode)
        top_feature_names.extend(mode_table["factor_set_name"].tolist())
    top_feature_names = list(dict.fromkeys(top_feature_names))

    print(
        f"Stage 2/3: time-grid search for {len(top_feature_names)} feature sets: "
        f"{top_feature_names}",
        flush=True,
    )
    grid_rows: list[dict[str, Any]] = []
    grid_weekly: list[pd.DataFrame] = []
    grid_refits: list[pd.DataFrame] = []
    total_grid = len(top_feature_names) * (
        len(search.fixed_window_candidates)
        + len(search.rolling_window_candidates)
        * len(search.refit_interval_candidates)
    )
    grid_ordinal = 0
    for feature_set_name in top_feature_names:
        config = replace(base_config, factor_set_name=feature_set_name)
        frozen = frozen_configs[feature_set_name]
        specs: list[tuple[str, str | int, int | None]] = [
            ("fixed", window, None) for window in search.fixed_window_candidates
        ] + [
            ("rolling", window, interval)
            for window in search.rolling_window_candidates
            for interval in search.refit_interval_candidates
        ]
        for mode, window, interval in specs:
            grid_ordinal += 1
            if grid_ordinal == 1 or grid_ordinal % 10 == 0:
                print(f"  grid {grid_ordinal}/{total_grid}", flush=True)
            key = (feature_set_name, mode, str(window), interval)
            result = cache.get(key)
            if result is None:
                result = evaluate_schedule(
                    frame,
                    config,
                    frozen,
                    phase="selection",
                    evaluation_start_week=search.selection_start_week,
                    evaluation_end_week=final_start,
                    training_mode=mode,
                    training_window_weeks=window,
                    refit_interval_weeks=interval,
                )
                cache[key] = result
            _, weekly, refits, overall = result
            grid_rows.append(overall)
            grid_weekly.append(weekly)
            grid_refits.append(refits)

    grid = pd.DataFrame(grid_rows)
    best_fixed = _best_row(grid.loc[grid["training_mode"].eq("fixed")])
    best_rolling = _best_row(grid.loc[grid["training_mode"].eq("rolling")])
    finalists = pd.DataFrame([best_fixed, best_rolling])
    primary = _best_row(finalists)
    primary_key = (str(primary["factor_set_name"]), str(primary["config_id"]))

    print(
        f"Stage 3/3: final untouched {search.final_holdout_weeks}-week holdout",
        flush=True,
    )
    final_predictions: list[pd.DataFrame] = []
    final_weekly: list[pd.DataFrame] = []
    final_refits: list[pd.DataFrame] = []
    final_rows: list[dict[str, Any]] = []
    for _, finalist in finalists.iterrows():
        feature_set_name = str(finalist["factor_set_name"])
        mode = str(finalist["training_mode"])
        window: str | int = str(finalist["training_window_weeks"])
        if window != "all_history":
            window = int(window)
        interval = (
            None
            if mode == "fixed"
            else int(float(finalist["refit_interval_weeks"]))
        )
        config = replace(base_config, factor_set_name=feature_set_name)
        result = evaluate_schedule(
            frame,
            config,
            frozen_configs[feature_set_name],
            phase="final_holdout",
            evaluation_start_week=final_start,
            evaluation_end_week=total_weeks,
            training_mode=mode,
            training_window_weeks=window,
            refit_interval_weeks=interval,
        )
        predictions, weekly, refits, overall = result
        finalist_role = f"best_{mode}_from_selection"
        is_primary = (feature_set_name, str(finalist["config_id"])) == primary_key
        predictions["finalist_role"] = finalist_role
        predictions["is_primary_pre_holdout_selection"] = is_primary
        weekly["finalist_role"] = finalist_role
        weekly["is_primary_pre_holdout_selection"] = is_primary
        refits["finalist_role"] = finalist_role
        refits["is_primary_pre_holdout_selection"] = is_primary
        overall.update(
            {
                "finalist_role": finalist_role,
                "is_primary_pre_holdout_selection": is_primary,
                "selection_period_Rank_IC": float(finalist["Rank_IC"]),
                "selection_period_Rank_ICIR": float(finalist["Rank_ICIR"]),
            }
        )
        final_predictions.append(predictions)
        final_weekly.append(weekly)
        final_refits.append(refits)
        final_rows.append(overall)

    final_metrics = pd.DataFrame(final_rows)
    out = base_config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _feature_catalog(selected_features).to_csv(
        out / "feature_set_catalog.csv", index=False
    )
    warmup_all.to_csv(out / "warmup_hyperparameter_results.csv", index=False)
    _write_json(out / "frozen_configs_by_feature_set.json", frozen_configs)
    screening.to_csv(out / "feature_screening_metrics.csv", index=False)
    grid.to_csv(out / "configuration_search_metrics.csv", index=False)
    pd.concat(grid_weekly, ignore_index=True).to_csv(
        out / "configuration_search_weekly_metrics.csv", index=False
    )
    pd.concat(grid_refits, ignore_index=True).to_csv(
        out / "configuration_search_refit_models.csv", index=False
    )
    final_metrics.to_csv(out / "final_holdout_metrics.csv", index=False)
    pd.concat(final_predictions, ignore_index=True).to_csv(
        out / "final_holdout_predictions.csv", index=False
    )
    pd.concat(final_weekly, ignore_index=True).to_csv(
        out / "final_holdout_weekly_metrics.csv", index=False
    )
    pd.concat(final_refits, ignore_index=True).to_csv(
        out / "final_holdout_refit_models.csv", index=False
    )
    _write_json(
        out / "selected_configuration.json",
        {
            "primary_pre_holdout_selection": primary.to_dict(),
            "best_fixed_pre_holdout": best_fixed.to_dict(),
            "best_rolling_pre_holdout": best_rolling.to_dict(),
            "selection_rule": "maximum selection-period Mean Rank IC; Rank ICIR tie-break",
            "final_holdout_used_for_selection": False,
        },
    )
    _write_json(out / "data_diagnostics.json", diagnostics)
    manifest = {
        "input_filename": diagnostics["input_filename"],
        "input_sha256": diagnostics["input_sha256"],
        "target_column": base_config.target_column,
        "search_config": asdict(search),
        "feature_sets": {
            name: OPTIMIZATION_FEATURE_SETS[name] for name in selected_features
        },
        "random_seed": base_config.random_seed,
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "backtest_performed": False,
        "final_holdout_used_for_selection": False,
    }
    _write_json(out / "experiment_manifest.json", manifest)
    report = _report(
        diagnostics,
        search,
        screening,
        grid,
        final_metrics,
        top_feature_names,
        primary_key,
    )
    (out / "optimization_report.md").write_text(report, encoding="utf-8")
    return {
        "diagnostics": diagnostics,
        "searched_feature_sets": selected_features,
        "time_grid_feature_sets": top_feature_names,
        "primary_selected_configuration": primary.to_dict(),
        "final_holdout_metrics": final_metrics.to_dict(orient="records"),
        "output_dir": str(out.resolve()),
        "backtest_performed": False,
        "final_holdout_used_for_selection": False,
    }
