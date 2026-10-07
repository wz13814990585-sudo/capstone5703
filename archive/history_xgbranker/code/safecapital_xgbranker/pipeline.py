from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import eligible_training_rows, load_and_prepare_data
from .metrics import calculate_weekly_metrics, summarize_weekly_metrics
from .modeling import fit_ranker, predict_ranking_block


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
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validation_predictions(
    model: Any,
    rows: pd.DataFrame,
    features: list[str],
    candidate: dict[str, Any],
    factor_set_name: str,
) -> pd.DataFrame:
    return predict_ranking_block(
        model=model,
        prediction_rows=rows,
        features=features,
        relevance_bins=int(candidate["relevance_bins"]),
        model_id=f"warmup_{candidate['candidate_id']}",
        refit_week=int(rows["week_index"].min()),
        refit_date=pd.Timestamp(rows["date"].min()),
        factor_set_name=factor_set_name,
    )


def tune_warmup(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    time = config.time
    warmup_indices = list(range(time.warmup_weeks))
    train_indices = warmup_indices[: time.warmup_train_weeks]
    validation_indices = warmup_indices[time.warmup_train_weeks :]
    if len(train_indices) != time.warmup_train_weeks:
        raise AssertionError("Warm-up train week count is wrong")
    if len(validation_indices) != time.warmup_validation_weeks:
        raise AssertionError("Warm-up validation week count is wrong")
    if max(train_indices) >= min(validation_indices):
        raise AssertionError("Warm-up training must be strictly before validation")

    # The Warm-up split must truly use all 104 designated training signal weeks.
    # Its information cutoff is one forecast horizon after the first validation
    # signal week, so the final training label satisfies the same strict
    # label_end_week < cutoff rule without admitting any validation row to fit().
    warmup_training_cutoff = (
        min(validation_indices) + time.forecast_horizon_weeks
    )
    training_rows = eligible_training_rows(
        frame, train_indices, refit_week=warmup_training_cutoff
    )
    validation_rows = frame.loc[frame["week_index"].isin(validation_indices)].copy()
    if training_rows.empty or validation_rows.empty:
        raise ValueError("Warm-up produced empty train or validation data")
    if training_rows["week_index"].nunique() != time.warmup_train_weeks:
        raise AssertionError("Warm-up fit must contain exactly the configured train weeks")
    if validation_rows["week_index"].nunique() != time.warmup_validation_weeks:
        raise AssertionError("Warm-up evaluation must contain exactly the validation weeks")

    results: list[dict[str, Any]] = []
    for candidate in config.candidates:
        model = fit_ranker(
            training_rows=training_rows,
            features=config.features,
            objective=str(candidate["objective"]),
            relevance_bins=int(candidate["relevance_bins"]),
            params=dict(candidate["params"]),
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
        )
        predictions = _validation_predictions(
            model, validation_rows, config.features, candidate, config.factor_set_name
        )
        weekly = calculate_weekly_metrics(predictions)
        summary = summarize_weekly_metrics(weekly)
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "objective": candidate["objective"],
                "relevance_bins": int(candidate["relevance_bins"]),
                "hyperparameters": json.dumps(candidate["params"], sort_keys=True),
                "designated_training_weeks": len(train_indices),
                "observable_training_weeks": int(training_rows["week_index"].nunique()),
                "validation_weeks": int(validation_rows["week_index"].nunique()),
                "training_rows": int(len(training_rows)),
                "validation_rows": int(len(validation_rows)),
                **summary,
            }
        )
    table = pd.DataFrame(results)
    table["_rank_ic_sort"] = table["Rank_IC"].fillna(-np.inf)
    table["_rank_icir_sort"] = table["Rank_ICIR"].fillna(-np.inf)
    table = table.sort_values(
        ["_rank_ic_sort", "_rank_icir_sort", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    ).drop(columns=["_rank_ic_sort", "_rank_icir_sort"])
    winner_id = str(table.iloc[0]["candidate_id"])
    winner = next(
        candidate for candidate in config.candidates if candidate["candidate_id"] == winner_id
    )
    frozen = {
        "warmup_weeks": time.warmup_weeks,
        "warmup_train_weeks": time.warmup_train_weeks,
        "warmup_validation_weeks": time.warmup_validation_weeks,
        "observable_warmup_training_weeks": int(
            training_rows["week_index"].nunique()
        ),
        "warmup_training_information_cutoff_week": int(warmup_training_cutoff),
        "factor_set_name": config.factor_set_name,
        "feature_list": config.features,
        "number_of_features": len(config.features),
        "objective": winner["objective"],
        "relevance_bins": int(winner["relevance_bins"]),
        "hyperparameters": dict(winner["params"]),
        "selection_metric": "Validation Mean Rank IC (Rank ICIR tie-break)",
        "selected_candidate_id": winner_id,
        "random_seed": config.random_seed,
    }
    return table.reset_index(drop=True), frozen


def build_refit_weeks(
    total_weeks: int,
    warmup_weeks: int,
    forecast_horizon_weeks: int,
    refit_interval_weeks: int,
) -> list[int]:
    # The embargo ensures the last Warm-up validation label is observable before
    # hyperparameters are frozen and the first truly OOS signal is scored.
    first_refit = warmup_weeks + forecast_horizon_weeks
    return list(range(first_refit, total_weeks, refit_interval_weeks))


def rolling_oos(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    total_weeks = int(frame["week_index"].nunique())
    refit_weeks = build_refit_weeks(
        total_weeks,
        config.time.warmup_weeks,
        config.time.forecast_horizon_weeks,
        config.time.refit_interval_weeks,
    )
    if not refit_weeks:
        raise ValueError("No Rolling OOS weeks remain after Warm-up and embargo")

    prediction_blocks: list[pd.DataFrame] = []
    model_metadata: list[dict[str, Any]] = []
    for ordinal, refit_week in enumerate(refit_weeks, start=1):
        candidate_start = max(0, refit_week - config.time.rolling_window_weeks)
        candidate_weeks = list(range(candidate_start, refit_week))
        training_rows = eligible_training_rows(frame, candidate_weeks, refit_week)
        if training_rows.empty:
            raise ValueError(f"No observable training rows for refit_week={refit_week}")
        if training_rows["week_index"].nunique() > config.time.rolling_window_weeks:
            raise AssertionError("Rolling training window exceeds configured maximum")

        model = fit_ranker(
            training_rows=training_rows,
            features=config.features,
            objective=str(frozen["objective"]),
            relevance_bins=int(frozen["relevance_bins"]),
            params=dict(frozen["hyperparameters"]),
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
        )
        block_end = min(
            refit_week + config.time.refit_interval_weeks,
            total_weeks,
        )
        block_rows = frame.loc[
            frame["week_index"].between(refit_week, block_end - 1)
        ].copy()
        model_id = f"xgbranker_{ordinal:03d}"
        refit_date = pd.Timestamp(
            frame.loc[frame["week_index"].eq(refit_week), "date"].iloc[0]
        )
        predictions = predict_ranking_block(
            model=model,
            prediction_rows=block_rows,
            features=config.features,
            relevance_bins=int(frozen["relevance_bins"]),
            model_id=model_id,
            refit_week=refit_week,
            refit_date=refit_date,
            factor_set_name=config.factor_set_name,
        )
        prediction_blocks.append(predictions)
        block_weekly = calculate_weekly_metrics(predictions)
        block_summary = summarize_weekly_metrics(block_weekly)
        model_metadata.append(
            {
                "model_id": model_id,
                "refit_week": int(refit_week),
                "refit_date": refit_date,
                "requested_window_start_week": int(candidate_start),
                "requested_window_end_week": int(refit_week - 1),
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
                **block_summary,
            }
        )

    all_predictions = pd.concat(prediction_blocks, ignore_index=True)
    if all_predictions.duplicated(["date", "ticker"]).any():
        raise AssertionError("Rolling predictions contain duplicate date+ticker rows")
    expected_weeks = list(range(refit_weeks[0], total_weeks))
    actual_weeks = sorted(all_predictions["week_index"].unique().tolist())
    if actual_weeks != expected_weeks:
        raise AssertionError("Rolling prediction weeks are not continuous")
    if not all_predictions.groupby("week_index")["model_id"].nunique().eq(1).all():
        raise AssertionError("A prediction week was assigned to multiple models")

    weekly = calculate_weekly_metrics(all_predictions)
    overall = summarize_weekly_metrics(weekly)
    overall.update(
        {
            "first_prediction_week": int(all_predictions["week_index"].min()),
            "last_prediction_week": int(all_predictions["week_index"].max()),
            "number_of_refit_models": len(refit_weeks),
            "rolling_window_weeks": config.time.rolling_window_weeks,
            "refit_interval_weeks": config.time.refit_interval_weeks,
            "forecast_horizon_weeks": config.time.forecast_horizon_weeks,
            "warmup_to_oos_embargo_weeks": config.time.forecast_horizon_weeks,
        }
    )
    return all_predictions, weekly, pd.DataFrame(model_metadata), overall


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Compatibility wrapper for a one-feature-set, no-backtest comparison run."""
    from .comparison import run_feature_comparison

    return run_feature_comparison(config, feature_sets=[config.factor_set_name])
