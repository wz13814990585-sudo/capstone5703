"""Controlled C/D label experiment for the frozen XGBRanker ensemble.

Experiment C trains on the continuous next-week log return. Experiment D
converts that same return into within-week 0--4 relevance labels. All model,
feature and time settings are otherwise identical. Both experiments are
evaluated against the original continuous target and no backtest is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRanker


TOTAL_EXPECTED_WEEKS = 522
ENSEMBLE_START_WEEK = 156
REFIT_INTERVAL_WEEKS = 26
RANDOM_SEED = 5703
FOLD_BOUNDARIES = (156, 247, 338, 430, 522)
RELEVANCE_BINS = 5

CONTINUOUS_LABEL = "continuous_target"
RELEVANCE_LABEL = "weekly_relevance_5"

FIVE_FEATURES = (
    "revenue_growth_yoy",
    "operating_profitability",
    "cashflow_to_assets",
    "debt_to_assets",
    "asset_growth_yoy",
)
TWO_FEATURES = ("operating_profitability", "debt_to_assets")


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    features: tuple[str, ...]
    training_window_weeks: int
    params: dict[str, Any]

    @property
    def resolved_features(self) -> tuple[str, ...]:
        return tuple(f"csrank__{feature}" for feature in self.features)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    label_mode: str
    description: str

    @property
    def uses_relevance_labels(self) -> bool:
        return self.label_mode == RELEVANCE_LABEL


EXPERIMENTS = (
    ExperimentSpec(
        "experiment_C_continuous",
        CONTINUOUS_LABEL,
        "Continuous next-week log return used directly as the training label",
    ),
    ExperimentSpec(
        "experiment_D_relevance5",
        RELEVANCE_LABEL,
        "Within-week target ranks converted to integer relevance labels 0-4",
    ),
)


COMMON_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "tree_method": "hist",
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "eval_metric": "ndcg@10",
}

COMPONENTS = (
    ComponentSpec(
        "model_1_w156_five_factors",
        FIVE_FEATURES,
        156,
        COMMON_PARAMS
        | {
            "max_depth": 2,
            "learning_rate": 0.025,
            "n_estimators": 360,
            "min_child_weight": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
            "gamma": 0.0,
        },
    ),
    ComponentSpec(
        "model_2_w52_two_factors",
        TWO_FEATURES,
        52,
        COMMON_PARAMS
        | {
            "max_depth": 3,
            "learning_rate": 0.04,
            "n_estimators": 240,
            "min_child_weight": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
            "gamma": 0.0,
        },
    ),
    ComponentSpec(
        "model_3_w52_five_factors_regularized",
        FIVE_FEATURES,
        52,
        COMMON_PARAMS
        | {
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 280,
            "min_child_weight": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "gamma": 0.05,
        },
    ),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_configuration(label_mode: str = CONTINUOUS_LABEL) -> dict[str, Any]:
    if label_mode not in {CONTINUOUS_LABEL, RELEVANCE_LABEL}:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    uses_relevance = label_mode == RELEVANCE_LABEL
    return {
        "target": {
            "column": "target",
            "definition": "ln(AdjClose(t+1) / AdjClose(t))",
            "label_type": (
                "within_week_relevance_0_to_4"
                if uses_relevance
                else "continuous_next_week_log_return"
            ),
            "target2_used": False,
            "classification_or_relevance_bins_used": uses_relevance,
            "relevance_bins": RELEVANCE_BINS if uses_relevance else None,
            "evaluation_target": "continuous_next_week_log_return",
        },
        "prediction_timing": {
            "decision_time": "after each weekly close",
            "label_observability_rule": "label_end_week <= refit_week",
            "formal_ensemble_start_week": ENSEMBLE_START_WEEK,
            "evaluation_end_week": TOTAL_EXPECTED_WEEKS - 1,
        },
        "feature_transform": "weekly cross-sectional percentile rank",
        "refit_interval_weeks": REFIT_INTERVAL_WEEKS,
        "random_seed": RANDOM_SEED,
        "components": [
            {
                "component_id": spec.component_id,
                "features": spec.features,
                "resolved_features": spec.resolved_features,
                "training_window_weeks": spec.training_window_weeks,
                "params": spec.params,
            }
            for spec in COMPONENTS
        ],
        "ensemble": {
            "method": "weekly percentile-rank then equal-weight mean",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "prediction_value": "ensemble_ranking_score",
        },
        "stability_fold_boundaries": FOLD_BOUNDARIES,
        "controlled_experiment": "C versus D; training label is the only changed setting",
        "parameters_frozen_before_final_run": True,
        "backtest_performed": False,
    }


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    raw_features = tuple(
        dict.fromkeys(feature for spec in COMPONENTS for feature in spec.features)
    )
    required = {"date", "ticker", "adj_close", "target", *raw_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("date+ticker must be unique")
    if frame["target"].isna().any():
        raise ValueError("target contains missing values")
    if frame[list(raw_features)].isna().any().any():
        raise ValueError("A frozen model feature contains missing values")
    numeric = frame[["adj_close", "target", *raw_features]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Selected numeric columns contain NaN or infinity")

    dates = pd.Index(sorted(frame["date"].unique()))
    if len(dates) != TOTAL_EXPECTED_WEEKS:
        raise ValueError(
            f"Expected {TOTAL_EXPECTED_WEEKS} unique weeks, found {len(dates)}"
        )
    gaps = pd.Series(dates[1:] - dates[:-1]).dt.days
    if not gaps.between(3, 11).all():
        raise ValueError("Unique dates do not have the expected weekly spacing")
    week_map = {date: index for index, date in enumerate(dates)}
    frame["week_index"] = frame["date"].map(week_map).astype(int)
    frame["label_end_week"] = frame["week_index"] + 1
    frame = frame.sort_values(["week_index", "ticker"], kind="stable").reset_index(drop=True)

    by_ticker = frame.sort_values(["ticker", "week_index"], kind="stable")
    next_week = by_ticker.groupby("ticker", sort=False)["week_index"].shift(-1)
    next_adj = by_ticker.groupby("ticker", sort=False)["adj_close"].shift(-1)
    verifiable = next_week.eq(by_ticker["week_index"] + 1) & next_adj.notna()
    expected_target = np.log(next_adj[verifiable] / by_ticker.loc[verifiable, "adj_close"])
    errors = (by_ticker.loc[verifiable, "target"] - expected_target).abs()
    if (errors > 1e-8).any():
        raise ValueError("target does not match the documented next-week log return")

    frame["target_verification_status"] = "supplied_upstream"
    verified_keys = pd.MultiIndex.from_frame(by_ticker.loc[verifiable, ["date", "ticker"]])
    frame_keys = pd.MultiIndex.from_frame(frame[["date", "ticker"]])
    frame.loc[frame_keys.isin(verified_keys), "target_verification_status"] = (
        "verified_from_next_row"
    )

    grouped = frame.groupby("week_index", sort=False)
    for feature in raw_features:
        frame[f"csrank__{feature}"] = grouped[feature].rank(method="average", pct=True)

    audit = pd.DataFrame(
        [
            ["input_rows", int(len(frame)), "verified"],
            ["unique_weeks", int(frame["week_index"].nunique()), "verified"],
            ["unique_tickers", int(frame["ticker"].nunique()), "verified"],
            ["target_rows_verified_from_next_row", int(verifiable.sum()), "verified"],
            ["target_rows_supplied_upstream", int((~verifiable).sum()), "accepted_by_user"],
            ["maximum_verified_target_error", float(errors.max()), "verified"],
            ["target2_available_in_input", "target2" in frame.columns, "informational"],
            ["target2_used_by_either_experiment", False, "verified"],
            ["evaluation_target_for_both_experiments", "continuous target", "verified"],
        ],
        columns=["check", "value", "status"],
    )
    return frame, audit


def build_refit_weeks(window: int, total_weeks: int) -> list[int]:
    if window <= 0 or total_weeks <= window:
        raise ValueError("Window must leave at least one prediction week")
    return list(range(window, total_weeks, REFIT_INTERVAL_WEEKS))


def add_weekly_relevance_labels(
    frame: pd.DataFrame,
    bins: int = RELEVANCE_BINS,
) -> pd.Series:
    """Return equal-frequency relevance labels generated independently by week."""
    if bins < 2:
        raise ValueError("relevance bins must be at least 2")
    grouped = frame.groupby("week_index", sort=False)["target"]
    within_week_rank = grouped.rank(method="average", ascending=True)
    group_size = grouped.transform("count")
    labels = np.floor((within_week_rank - 1.0) * bins / group_size)
    labels = labels.clip(0, bins - 1).astype(np.int32)
    if not labels.between(0, bins - 1).all():
        raise AssertionError("Invalid relevance label")
    return labels


def fit_ranker(
    training: pd.DataFrame,
    features: tuple[str, ...],
    params: dict[str, Any],
    label_mode: str,
    n_jobs: int,
) -> XGBRanker:
    ordered = training.sort_values(["week_index", "ticker"], kind="stable")
    groups = ordered.groupby("week_index", sort=False).size().to_numpy(dtype=np.uint32)
    model = XGBRanker(
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )
    if label_mode == CONTINUOUS_LABEL:
        labels = ordered["target"].to_numpy(dtype=np.float32)
    elif label_mode == RELEVANCE_LABEL:
        labels = add_weekly_relevance_labels(ordered).to_numpy(dtype=np.int32)
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    model.fit(
        ordered[list(features)].to_numpy(dtype=np.float32),
        labels,
        group=groups,
        verbose=False,
    )
    return model


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 2 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return np.nan
    return float(left[valid].corr(right[valid], method=method))


def calculate_weekly_metrics(
    predictions: pd.DataFrame,
    score_column: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week_index, week in predictions.groupby("week_index", sort=True):
        rows.append(
            {
                "model": model_name,
                "week_index": int(week_index),
                "date": pd.Timestamp(week["date"].iloc[0]),
                "number_of_stocks": int(len(week)),
                "IC": _safe_corr(week[score_column], week["target"], "pearson"),
                "Rank_IC": _safe_corr(week[score_column], week["target"], "spearman"),
            }
        )
    return pd.DataFrame(rows)


def summarize_weekly_metrics(
    weekly: pd.DataFrame,
    *,
    model_name: str,
    evaluation_scope: str,
) -> dict[str, Any]:
    ic = weekly["IC"].dropna().astype(float)
    rank_ic = weekly["Rank_IC"].dropna().astype(float)
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else np.nan
    return {
        "model": model_name,
        "evaluation_scope": evaluation_scope,
        "evaluation_start_week": int(weekly["week_index"].min()),
        "evaluation_end_week": int(weekly["week_index"].max()),
        "prediction_weeks": int(weekly["week_index"].nunique()),
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        "Rank_IC": float(rank_ic.mean()),
        "Rank_ICIR": float(rank_ic.mean() / rank_std) if np.isfinite(rank_std) and rank_std > 0 else np.nan,
        "Positive_IC_Ratio": float((ic > 0).mean()),
        "Positive_Rank_IC_Ratio": float((rank_ic > 0).mean()),
        "Median_Rank_IC": float(rank_ic.median()),
        "Weekly_Rank_IC_Std": rank_std,
        "Best_Weekly_Rank_IC": float(rank_ic.max()),
        "Worst_Weekly_Rank_IC": float(rank_ic.min()),
    }


def evaluate_component(
    frame: pd.DataFrame,
    spec: ComponentSpec,
    experiment: ExperimentSpec,
    output_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_weeks = int(frame["week_index"].nunique())
    refit_weeks = build_refit_weeks(spec.training_window_weeks, total_weeks)
    prediction_blocks: list[pd.DataFrame] = []
    refit_rows: list[dict[str, Any]] = []

    for ordinal, refit_week in enumerate(refit_weeks, start=1):
        training_start = refit_week - spec.training_window_weeks
        training_end = refit_week - 1
        training = frame.loc[
            frame["week_index"].between(training_start, training_end)
            & frame["label_end_week"].le(refit_week)
        ].copy()
        if training["week_index"].nunique() != spec.training_window_weeks:
            raise AssertionError(f"{spec.component_id} does not contain its exact frozen window")
        if not training["label_end_week"].le(refit_week).all():
            raise AssertionError("Training data violates label observability")
        if training["week_index"].max() != refit_week - 1:
            raise AssertionError("The latest observable training week was not included")

        model = fit_ranker(
            training,
            spec.resolved_features,
            spec.params,
            experiment.label_mode,
            n_jobs,
        )
        if refit_week == refit_weeks[-1]:
            model.save_model(output_dir / f"latest_{spec.component_id}.json")

        prediction_end = min(refit_week + REFIT_INTERVAL_WEEKS, total_weeks) - 1
        block = frame.loc[frame["week_index"].between(refit_week, prediction_end)].copy()
        model_id = (
            f"{experiment.experiment_id}_{spec.component_id}_refit_{refit_week:04d}"
        )
        block["raw_ranking_score"] = model.predict(
            block[list(spec.resolved_features)].to_numpy(dtype=np.float32)
        )
        block["percentile_score"] = block.groupby("week_index", sort=False)[
            "raw_ranking_score"
        ].rank(method="average", pct=True)
        block["predicted_rank"] = block.groupby("week_index", sort=False)[
            "percentile_score"
        ].rank(method="first", ascending=False).astype(int)
        block["actual_rank"] = block.groupby("week_index", sort=False)["target"].rank(
            method="first", ascending=False
        ).astype(int)
        block["model_id"] = model_id
        block["component_id"] = spec.component_id
        block["experiment_id"] = experiment.experiment_id
        block["training_label_mode"] = experiment.label_mode
        block["refit_week"] = refit_week
        block["refit_date"] = pd.Timestamp(
            frame.loc[frame["week_index"].eq(refit_week), "date"].iloc[0]
        )
        block["prediction_block"] = ordinal
        block_output = block[
            [
                "week_index",
                "date",
                "ticker",
                "target",
                "target_verification_status",
                "actual_rank",
                "component_id",
                "experiment_id",
                "training_label_mode",
                "model_id",
                "refit_week",
                "refit_date",
                "prediction_block",
                "raw_ranking_score",
                "percentile_score",
                "predicted_rank",
            ]
        ].copy()
        prediction_blocks.append(block_output)

        block_weekly = calculate_weekly_metrics(block_output, "percentile_score", spec.component_id)
        summary = summarize_weekly_metrics(
            block_weekly,
            model_name=spec.component_id,
            evaluation_scope=f"refit_block_{refit_week}",
        )
        refit_rows.append(
            {
                "model_id": model_id,
                "experiment_id": experiment.experiment_id,
                "component_id": spec.component_id,
                "refit_week": refit_week,
                "refit_date": block["refit_date"].iloc[0],
                "training_start_week": training_start,
                "training_end_week": training_end,
                "training_weeks": int(training["week_index"].nunique()),
                "training_rows": int(len(training)),
                "prediction_start_week": refit_week,
                "prediction_end_week": prediction_end,
                "prediction_weeks": int(block_output["week_index"].nunique()),
                "prediction_rows": int(len(block_output)),
                "feature_list": json.dumps(spec.features),
                "resolved_feature_list": json.dumps(spec.resolved_features),
                "training_window_weeks": spec.training_window_weeks,
                "refit_interval_weeks": REFIT_INTERVAL_WEEKS,
                "objective": spec.params["objective"],
                "training_label_mode": experiment.label_mode,
                "continuous_target_used": not experiment.uses_relevance_labels,
                "classification_label_used": experiment.uses_relevance_labels,
                "relevance_bins": (
                    RELEVANCE_BINS if experiment.uses_relevance_labels else None
                ),
                "evaluation_target": "continuous_target",
                "hyperparameters": json.dumps(spec.params, sort_keys=True),
                **{
                    key: value
                    for key, value in summary.items()
                    if key
                    not in {
                        "model",
                        "evaluation_scope",
                        "evaluation_start_week",
                        "evaluation_end_week",
                        "prediction_weeks",
                    }
                },
            }
        )

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    expected_weeks = list(range(spec.training_window_weeks, total_weeks))
    if sorted(predictions["week_index"].unique().tolist()) != expected_weeks:
        raise AssertionError(f"{spec.component_id} prediction weeks are incomplete")
    if predictions.duplicated(["date", "ticker"]).any():
        raise AssertionError(f"{spec.component_id} has duplicate date+ticker predictions")
    if not predictions.groupby("week_index")["model_id"].nunique().eq(1).all():
        raise AssertionError(f"{spec.component_id} maps a week to multiple models")
    weekly = calculate_weekly_metrics(predictions, "percentile_score", spec.component_id)
    return predictions, weekly, pd.DataFrame(refit_rows)


def build_ensemble(
    component_predictions: dict[str, pd.DataFrame],
    experiment: ExperimentSpec,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component_number, spec in enumerate(COMPONENTS, start=1):
        source = component_predictions[spec.component_id]
        source = source.loc[source["week_index"].ge(ENSEMBLE_START_WEEK)].copy()
        source = source[
            [
                "week_index",
                "date",
                "ticker",
                "target",
                "target_verification_status",
                "actual_rank",
                "experiment_id",
                "training_label_mode",
                "model_id",
                "refit_week",
                "raw_ranking_score",
                "percentile_score",
                "predicted_rank",
            ]
        ].rename(
            columns={
                "model_id": f"model_{component_number}_id",
                "refit_week": f"model_{component_number}_refit_week",
                "raw_ranking_score": f"model_{component_number}_raw_score",
                "percentile_score": f"model_{component_number}_percentile",
                "predicted_rank": f"model_{component_number}_predicted_rank",
            }
        )
        if merged is None:
            merged = source
        else:
            source = source.drop(
                columns=[
                    "target",
                    "target_verification_status",
                    "actual_rank",
                    "experiment_id",
                    "training_label_mode",
                ]
            )
            merged = merged.merge(
                source,
                on=["week_index", "date", "ticker"],
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise AssertionError("No component predictions were available")

    refit_columns = [f"model_{index}_refit_week" for index in range(1, 4)]
    if not merged[refit_columns].nunique(axis=1).eq(1).all():
        raise AssertionError("Component refit schedules are not aligned")
    merged["refit_week"] = merged["model_1_refit_week"]
    merged["prediction_block"] = (
        (merged["refit_week"] - ENSEMBLE_START_WEEK) // REFIT_INTERVAL_WEEKS + 1
    ).astype(int)
    merged["ensemble_model_id"] = merged["refit_week"].map(
        lambda value: f"{experiment.experiment_id}_ensemble_refit_{int(value):04d}"
    )
    score_columns = [f"model_{index}_percentile" for index in range(1, 4)]
    merged["ensemble_ranking_score"] = merged[score_columns].mean(axis=1)
    merged["ensemble_predicted_rank"] = merged.groupby("week_index", sort=False)[
        "ensemble_ranking_score"
    ].rank(method="first", ascending=False).astype(int)
    merged["rank_difference"] = merged["ensemble_predicted_rank"] - merged["actual_rank"]

    if sorted(merged["week_index"].unique()) != list(
        range(ENSEMBLE_START_WEEK, TOTAL_EXPECTED_WEEKS)
    ):
        raise AssertionError("Ensemble prediction weeks are incomplete")
    if merged.duplicated(["date", "ticker"]).any():
        raise AssertionError("Ensemble has duplicate date+ticker predictions")
    return merged.sort_values(["week_index", "ticker"], kind="stable").reset_index(drop=True)


def build_weekly_comparison(
    component_weekly: dict[str, pd.DataFrame],
    ensemble_weekly: pd.DataFrame,
) -> pd.DataFrame:
    result = ensemble_weekly[
        ["week_index", "date", "number_of_stocks", "IC", "Rank_IC"]
    ].rename(columns={"IC": "ensemble_IC", "Rank_IC": "ensemble_Rank_IC"})
    for index, spec in enumerate(COMPONENTS, start=1):
        weekly = component_weekly[spec.component_id]
        weekly = weekly.loc[weekly["week_index"].ge(ENSEMBLE_START_WEEK)]
        weekly = weekly[["week_index", "IC", "Rank_IC"]].rename(
            columns={"IC": f"model_{index}_IC", "Rank_IC": f"model_{index}_Rank_IC"}
        )
        result = result.merge(weekly, on="week_index", validate="one_to_one")
        result[f"ensemble_beats_model_{index}"] = (
            result["ensemble_Rank_IC"] > result[f"model_{index}_Rank_IC"]
        )
    return result


def evaluate_ensemble_blocks(
    ensemble: pd.DataFrame,
    ensemble_weekly: pd.DataFrame,
    experiment: ExperimentSpec,
) -> pd.DataFrame:
    rows = []
    for refit_week, block in ensemble.groupby("refit_week", sort=True):
        weeks = sorted(block["week_index"].unique())
        weekly = ensemble_weekly.loc[ensemble_weekly["week_index"].isin(weeks)]
        summary = summarize_weekly_metrics(
            weekly,
            model_name=f"{experiment.experiment_id}_ensemble",
            evaluation_scope=f"refit_block_{int(refit_week)}",
        )
        rows.append(
            {
                "ensemble_model_id": (
                    f"{experiment.experiment_id}_ensemble_refit_{int(refit_week):04d}"
                ),
                "experiment_id": experiment.experiment_id,
                "training_label_mode": experiment.label_mode,
                "refit_week": int(refit_week),
                "prediction_start_week": min(weeks),
                "prediction_end_week": max(weeks),
                "prediction_weeks": len(weeks),
                "prediction_rows": int(len(block)),
                **{
                    key: value
                    for key, value in summary.items()
                    if key
                    not in {
                        "model",
                        "evaluation_scope",
                        "evaluation_start_week",
                        "evaluation_end_week",
                        "prediction_weeks",
                    }
                },
            }
        )
    return pd.DataFrame(rows)


def build_fold_metrics(weekly_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, weekly in weekly_by_model.items():
        for fold_number, (start, end) in enumerate(
            zip(FOLD_BOUNDARIES[:-1], FOLD_BOUNDARIES[1:]), start=1
        ):
            fold = weekly.loc[weekly["week_index"].between(start, end - 1)]
            rows.append(
                {
                    "fold": fold_number,
                    **summarize_weekly_metrics(
                        fold,
                        model_name=model_name,
                        evaluation_scope=f"fold_{fold_number}",
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_experiment(
    frame: pd.DataFrame,
    target_audit: pd.DataFrame,
    input_path: Path,
    output_dir: Path,
    n_jobs: int,
    experiment: ExperimentSpec,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_predictions: dict[str, pd.DataFrame] = {}
    component_weekly: dict[str, pd.DataFrame] = {}
    refit_tables: list[pd.DataFrame] = []
    for spec in COMPONENTS:
        print(f"Training {experiment.experiment_id}: {spec.component_id}", flush=True)
        predictions, weekly, refits = evaluate_component(
            frame, spec, experiment, output_dir, n_jobs
        )
        component_predictions[spec.component_id] = predictions
        component_weekly[spec.component_id] = weekly
        refit_tables.append(refits)

    refit_evaluation = pd.concat(refit_tables, ignore_index=True)
    ensemble = build_ensemble(component_predictions, experiment)
    ensemble_name = f"{experiment.experiment_id}_ensemble"
    ensemble_weekly = calculate_weekly_metrics(
        ensemble, "ensemble_ranking_score", ensemble_name
    )
    weekly_comparison = build_weekly_comparison(component_weekly, ensemble_weekly)
    weekly_comparison.insert(0, "training_label_mode", experiment.label_mode)
    weekly_comparison.insert(0, "experiment_id", experiment.experiment_id)
    ensemble_blocks = evaluate_ensemble_blocks(
        ensemble, ensemble_weekly, experiment
    )

    overall_rows: list[dict[str, Any]] = []
    formal_weekly: dict[str, pd.DataFrame] = {}
    for spec in COMPONENTS:
        weekly = component_weekly[spec.component_id]
        formal = weekly.loc[weekly["week_index"].ge(ENSEMBLE_START_WEEK)].copy()
        formal_weekly[spec.component_id] = formal
        summary = summarize_weekly_metrics(
            formal,
            model_name=spec.component_id,
            evaluation_scope="formal_ensemble_period",
        )
        summary["prediction_rows"] = int(
            component_predictions[spec.component_id]
            .loc[lambda table: table["week_index"].ge(ENSEMBLE_START_WEEK)]
            .shape[0]
        )
        summary["experiment_id"] = experiment.experiment_id
        summary["training_label_mode"] = experiment.label_mode
        overall_rows.append(summary)

        if spec.training_window_weeks == 52:
            early = weekly.loc[weekly["week_index"].lt(ENSEMBLE_START_WEEK)].copy()
            early_summary = summarize_weekly_metrics(
                early,
                model_name=spec.component_id,
                evaluation_scope="early_component_only_period",
            )
            early_summary["prediction_rows"] = int(
                component_predictions[spec.component_id]
                .loc[lambda table: table["week_index"].lt(ENSEMBLE_START_WEEK)]
                .shape[0]
            )
            early_summary["experiment_id"] = experiment.experiment_id
            early_summary["training_label_mode"] = experiment.label_mode
            overall_rows.append(early_summary)

    ensemble_summary = summarize_weekly_metrics(
        ensemble_weekly,
        model_name=ensemble_name,
        evaluation_scope="formal_ensemble_period",
    )
    ensemble_summary["prediction_rows"] = int(len(ensemble))
    ensemble_summary["experiment_id"] = experiment.experiment_id
    ensemble_summary["training_label_mode"] = experiment.label_mode
    overall_rows.append(ensemble_summary)
    overall = pd.DataFrame(overall_rows)

    formal_weekly[ensemble_name] = ensemble_weekly
    folds = build_fold_metrics(formal_weekly)
    folds.insert(0, "training_label_mode", experiment.label_mode)
    folds.insert(0, "experiment_id", experiment.experiment_id)

    experiment_audit = pd.concat(
        [
            target_audit,
            pd.DataFrame(
                [
                    ["experiment_id", experiment.experiment_id, "configured"],
                    ["training_label_mode", experiment.label_mode, "configured"],
                    [
                        "weekly_relevance_bins",
                        RELEVANCE_BINS if experiment.uses_relevance_labels else "not used",
                        "configured",
                    ],
                ],
                columns=target_audit.columns,
            ),
        ],
        ignore_index=True,
    )
    experiment_audit.to_csv(output_dir / "target_audit.csv", index=False)
    refit_evaluation.to_csv(output_dir / "refit_model_evaluation.csv", index=False)
    ensemble_blocks.to_csv(output_dir / "ensemble_block_evaluation.csv", index=False)
    overall.to_csv(output_dir / "component_overall_evaluation.csv", index=False)
    weekly_comparison.to_csv(output_dir / "weekly_metrics.csv", index=False)
    ensemble.to_csv(output_dir / "stock_predictions.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    _write_json(
        output_dir / "frozen_configuration.json",
        frozen_configuration(experiment.label_mode),
    )

    manifest = {
        "input_filename": input_path.name,
        "input_sha256": _sha256(input_path),
        "output_dir": output_dir.resolve(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgboost.__version__,
        "formal_prediction_weeks": int(ensemble["week_index"].nunique()),
        "formal_prediction_rows": int(len(ensemble)),
        "refit_model_instances": int(len(refit_evaluation)),
        "ensemble_refit_blocks": int(len(ensemble_blocks)),
        "parameters_frozen_before_final_run": True,
        "experiment_id": experiment.experiment_id,
        "training_label_mode": experiment.label_mode,
        "continuous_training_target_used": not experiment.uses_relevance_labels,
        "target2_used": False,
        "classification_labels_used": experiment.uses_relevance_labels,
        "relevance_bins": RELEVANCE_BINS if experiment.uses_relevance_labels else None,
        "continuous_target_used_for_evaluation": True,
        "backtest_performed": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {
        "overall": overall,
        "ensemble_summary": pd.DataFrame([ensemble_summary]),
        "weekly": weekly_comparison,
        "folds": folds,
        "refits": refit_evaluation,
        "ensemble_blocks": ensemble_blocks,
        "ensemble_predictions": ensemble,
    }


def build_paired_weekly_comparison(
    c_weekly: pd.DataFrame,
    d_weekly: pd.DataFrame,
) -> pd.DataFrame:
    c = c_weekly[
        ["week_index", "date", "number_of_stocks", "ensemble_IC", "ensemble_Rank_IC"]
    ].rename(
        columns={"ensemble_IC": "C_IC", "ensemble_Rank_IC": "C_Rank_IC"}
    )
    d = d_weekly[
        ["week_index", "ensemble_IC", "ensemble_Rank_IC"]
    ].rename(
        columns={"ensemble_IC": "D_IC", "ensemble_Rank_IC": "D_Rank_IC"}
    )
    paired = c.merge(d, on="week_index", validate="one_to_one")
    paired["IC_delta_D_minus_C"] = paired["D_IC"] - paired["C_IC"]
    paired["Rank_IC_delta_D_minus_C"] = (
        paired["D_Rank_IC"] - paired["C_Rank_IC"]
    )
    paired["D_beats_C_Rank_IC"] = paired["Rank_IC_delta_D_minus_C"].gt(0)
    return paired


def build_experiment_summary(
    summaries: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    result = summaries.copy()
    c_rank_ic = float(
        result.loc[
            result["experiment_id"].eq("experiment_C_continuous"), "Rank_IC"
        ].iloc[0]
    )
    result["Rank_IC_delta_vs_C"] = result["Rank_IC"] - c_rank_ic
    result["Rank_IC_relative_change_vs_C"] = result["Rank_IC_delta_vs_C"] / abs(c_rank_ic)
    result["paired_week_win_ratio_vs_C"] = np.nan
    d_mask = result["experiment_id"].eq("experiment_D_relevance5")
    result.loc[d_mask, "paired_week_win_ratio_vs_C"] = float(
        paired["D_beats_C_Rank_IC"].mean()
    )
    result.loc[
        result["experiment_id"].eq("experiment_C_continuous"),
        "paired_week_win_ratio_vs_C",
    ] = 0.5
    winner = result.loc[result["Rank_IC"].idxmax(), "experiment_id"]
    result["winner_by_mean_Rank_IC"] = result["experiment_id"].eq(winner)
    return result


def build_paired_statistics(
    experiment_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    delta = paired["Rank_IC_delta_D_minus_C"].astype(float)
    standard_error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    c_rank_ic = float(
        experiment_summary.loc[
            experiment_summary["experiment_id"].eq("experiment_C_continuous"),
            "Rank_IC",
        ].iloc[0]
    )
    d_rank_ic = float(
        experiment_summary.loc[
            experiment_summary["experiment_id"].eq("experiment_D_relevance5"),
            "Rank_IC",
        ].iloc[0]
    )
    return pd.DataFrame(
        [
            {
                "evaluation_start_week": int(paired["week_index"].min()),
                "evaluation_end_week": int(paired["week_index"].max()),
                "paired_weeks": int(len(paired)),
                "C_mean_Rank_IC": c_rank_ic,
                "D_mean_Rank_IC": d_rank_ic,
                "Rank_IC_delta_D_minus_C": d_rank_ic - c_rank_ic,
                "relative_change_D_vs_C": (d_rank_ic - c_rank_ic) / abs(c_rank_ic),
                "median_weekly_delta_D_minus_C": float(delta.median()),
                "weekly_delta_standard_deviation": float(delta.std(ddof=1)),
                "weekly_delta_standard_error": standard_error,
                "paired_t_statistic": (
                    float(delta.mean() / standard_error) if standard_error > 0 else np.nan
                ),
                "D_weekly_win_ratio": float(delta.gt(0).mean()),
                "C_D_weekly_Rank_IC_correlation": float(
                    paired["C_Rank_IC"].corr(paired["D_Rank_IC"])
                ),
            }
        ]
    )


def run(input_path: Path, output_dir: Path, n_jobs: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, target_audit = load_and_validate_data(input_path)
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for experiment in EXPERIMENTS:
        experiment_dir = output_dir / experiment.experiment_id
        results[experiment.experiment_id] = run_experiment(
            frame,
            target_audit,
            input_path,
            experiment_dir,
            n_jobs,
            experiment,
        )

    c_result = results["experiment_C_continuous"]
    d_result = results["experiment_D_relevance5"]
    paired = build_paired_weekly_comparison(c_result["weekly"], d_result["weekly"])
    summaries = pd.concat(
        [c_result["ensemble_summary"], d_result["ensemble_summary"]],
        ignore_index=True,
    )
    experiment_summary = build_experiment_summary(summaries, paired)
    paired_statistics = build_paired_statistics(experiment_summary, paired)
    all_component_results = pd.concat(
        [c_result["overall"], d_result["overall"]], ignore_index=True
    )
    all_folds = pd.concat([c_result["folds"], d_result["folds"]], ignore_index=True)

    experiment_summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_weekly_comparison.csv", index=False)
    paired_statistics.to_csv(
        output_dir / "paired_comparison_statistics.csv", index=False
    )
    all_component_results.to_csv(
        output_dir / "component_experiment_comparison.csv", index=False
    )
    all_folds.to_csv(output_dir / "fold_experiment_comparison.csv", index=False)

    winner = str(
        experiment_summary.loc[
            experiment_summary["winner_by_mean_Rank_IC"], "experiment_id"
        ].iloc[0]
    )
    comparison_manifest = {
        "input_filename": input_path.name,
        "input_sha256": _sha256(input_path),
        "comparison": "Experiment C continuous target vs Experiment D weekly relevance 0-4",
        "controlled_settings": (
            "features, windows, refit interval, model parameters, seed, ensemble weights, "
            "prediction weeks and continuous-target evaluation"
        ),
        "only_changed_setting": "training_label_mode",
        "evaluation_start_week": ENSEMBLE_START_WEEK,
        "evaluation_end_week": TOTAL_EXPECTED_WEEKS - 1,
        "prediction_weeks_per_experiment": int(paired["week_index"].nunique()),
        "refit_model_instances_per_experiment": int(len(c_result["refits"])),
        "winner_by_mean_Rank_IC": winner,
        "Rank_IC_delta_D_minus_C": float(
            paired_statistics["Rank_IC_delta_D_minus_C"].iloc[0]
        ),
        "D_weekly_win_ratio": float(paired_statistics["D_weekly_win_ratio"].iloc[0]),
        "paired_t_statistic": float(paired_statistics["paired_t_statistic"].iloc[0]),
        "target2_used": False,
        "backtest_performed": False,
    }
    _write_json(output_dir / "comparison_manifest.json", comparison_manifest)
    return experiment_summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    archive_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled Experiments C and D for the frozen three-component "
            "XGBRanker ensemble."
        )
    )
    parser.add_argument("--input", type=Path, default=root / "test_45_factors_weekly.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=archive_root / "data" / "label_comparison",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.input, args.output_dir, args.n_jobs)
    print("\nControlled Experiment C/D evaluation")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
