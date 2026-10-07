"""Final three-model XGBRanker walk-forward evaluation.

The final model uses within-week 0--4 relevance labels for training and the
continuous next-week log return for evaluation. It produces rankings and
IC/RankIC metrics only; it does not run a trading backtest.
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
FORMAL_START_WEEK = 156
REFIT_INTERVAL_WEEKS = 26
RELEVANCE_BINS = 5
RANDOM_SEED = 5703
FOLD_BOUNDARIES = (156, 247, 338, 430, 522)

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
    def ranked_features(self) -> tuple[str, ...]:
        return tuple(f"csrank__{feature}" for feature in self.features)


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


def model_configuration() -> dict[str, Any]:
    return {
        "model_type": "three-component XGBRanker equal-weight ensemble",
        "training_label": {
            "source": "continuous target ranked independently within each training week",
            "type": "integer relevance label",
            "levels": [0, 1, 2, 3, 4],
            "target2_used": False,
        },
        "evaluation_target": {
            "column": "target",
            "definition": "ln(AdjClose(t+1) / AdjClose(t))",
            "type": "continuous next-week log return",
        },
        "feature_transform": "weekly cross-sectional percentile rank",
        "formal_prediction_weeks": [FORMAL_START_WEEK, TOTAL_EXPECTED_WEEKS - 1],
        "refit_interval_weeks": REFIT_INTERVAL_WEEKS,
        "label_observability_rule": "label_end_week <= refit_week",
        "random_seed": RANDOM_SEED,
        "components": [
            {
                "component_id": spec.component_id,
                "features": spec.features,
                "ranked_features": spec.ranked_features,
                "training_window_weeks": spec.training_window_weeks,
                "params": spec.params,
            }
            for spec in COMPONENTS
        ],
        "ensemble": {
            "method": "weekly percentile scores followed by equal-weight mean",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "prediction_value": "ensemble_ranking_score",
        },
        "backtest_performed": False,
    }


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    frame = frame.sort_values(["week_index", "ticker"], kind="stable").reset_index(
        drop=True
    )

    by_ticker = frame.sort_values(["ticker", "week_index"], kind="stable")
    next_week = by_ticker.groupby("ticker", sort=False)["week_index"].shift(-1)
    next_adj_close = by_ticker.groupby("ticker", sort=False)["adj_close"].shift(-1)
    verifiable = next_week.eq(by_ticker["week_index"] + 1) & next_adj_close.notna()
    expected = np.log(
        next_adj_close[verifiable] / by_ticker.loc[verifiable, "adj_close"]
    )
    errors = (by_ticker.loc[verifiable, "target"] - expected).abs()
    if (errors > 1e-8).any():
        raise ValueError("target does not match the documented next-week log return")

    frame["target_verification_status"] = "supplied_upstream"
    verified_keys = pd.MultiIndex.from_frame(
        by_ticker.loc[verifiable, ["date", "ticker"]]
    )
    frame_keys = pd.MultiIndex.from_frame(frame[["date", "ticker"]])
    frame.loc[frame_keys.isin(verified_keys), "target_verification_status"] = (
        "verified_from_next_row"
    )

    grouped = frame.groupby("week_index", sort=False)
    for feature in raw_features:
        frame[f"csrank__{feature}"] = grouped[feature].rank(
            method="average", pct=True
        )

    audit = pd.DataFrame(
        [
            ["input_rows", int(len(frame)), "verified"],
            ["unique_weeks", int(frame["week_index"].nunique()), "verified"],
            ["unique_tickers", int(frame["ticker"].nunique()), "verified"],
            ["target_rows_verified", int(verifiable.sum()), "verified"],
            ["target_rows_supplied_upstream", int((~verifiable).sum()), "accepted"],
            ["maximum_verified_target_error", float(errors.max()), "verified"],
            ["target2_used", False, "verified"],
            ["training_relevance_bins", RELEVANCE_BINS, "configured"],
        ],
        columns=["check", "value", "status"],
    )
    return frame, audit


def build_refit_weeks(window: int, total_weeks: int) -> list[int]:
    if window <= 0 or total_weeks <= window:
        raise ValueError("Window must leave at least one prediction week")
    return list(range(window, total_weeks, REFIT_INTERVAL_WEEKS))


def relevance_labels(frame: pd.DataFrame) -> np.ndarray:
    grouped = frame.groupby("week_index", sort=False)["target"]
    ranks = grouped.rank(method="average", ascending=True)
    group_size = grouped.transform("count")
    labels = np.floor((ranks - 1.0) * RELEVANCE_BINS / group_size)
    labels = labels.clip(0, RELEVANCE_BINS - 1).astype(np.int32)
    if not labels.between(0, RELEVANCE_BINS - 1).all():
        raise AssertionError("Invalid relevance label")
    return labels.to_numpy(dtype=np.int32)


def fit_ranker(
    training: pd.DataFrame,
    spec: ComponentSpec,
    n_jobs: int,
) -> XGBRanker:
    ordered = training.sort_values(["week_index", "ticker"], kind="stable")
    groups = ordered.groupby("week_index", sort=False).size().to_numpy(dtype=np.uint32)
    model = XGBRanker(
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        verbosity=0,
        **spec.params,
    )
    model.fit(
        ordered[list(spec.ranked_features)].to_numpy(dtype=np.float32),
        relevance_labels(ordered),
        group=groups,
        verbose=False,
    )
    return model


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 2 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return np.nan
    return float(left[valid].corr(right[valid], method=method))


def weekly_metrics(
    predictions: pd.DataFrame,
    score_column: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
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


def summarize(
    weekly: pd.DataFrame,
    model_name: str,
    evaluation_scope: str,
) -> dict[str, Any]:
    ic = weekly["IC"].dropna().astype(float)
    rank_ic = weekly["Rank_IC"].dropna().astype(float)
    ic_std = float(ic.std(ddof=1))
    rank_std = float(rank_ic.std(ddof=1))
    return {
        "model": model_name,
        "evaluation_scope": evaluation_scope,
        "evaluation_start_week": int(weekly["week_index"].min()),
        "evaluation_end_week": int(weekly["week_index"].max()),
        "prediction_weeks": int(weekly["week_index"].nunique()),
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic_std),
        "Rank_IC": float(rank_ic.mean()),
        "Rank_ICIR": float(rank_ic.mean() / rank_std),
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
    model_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_weeks = int(frame["week_index"].nunique())
    refit_weeks = build_refit_weeks(spec.training_window_weeks, total_weeks)
    blocks: list[pd.DataFrame] = []
    refit_rows: list[dict[str, Any]] = []

    for ordinal, refit_week in enumerate(refit_weeks, start=1):
        training_start = refit_week - spec.training_window_weeks
        training = frame.loc[
            frame["week_index"].between(training_start, refit_week - 1)
            & frame["label_end_week"].le(refit_week)
        ].copy()
        if training["week_index"].nunique() != spec.training_window_weeks:
            raise AssertionError(f"{spec.component_id} has an incomplete training window")
        if not training["label_end_week"].le(refit_week).all():
            raise AssertionError("Training data violates label observability")

        model = fit_ranker(training, spec, n_jobs)
        if refit_week == refit_weeks[-1]:
            model.save_model(model_dir / f"{spec.component_id}.json")

        prediction_end = min(refit_week + REFIT_INTERVAL_WEEKS, total_weeks) - 1
        block = frame.loc[
            frame["week_index"].between(refit_week, prediction_end)
        ].copy()
        model_id = f"{spec.component_id}_refit_{refit_week:04d}"
        block["raw_ranking_score"] = model.predict(
            block[list(spec.ranked_features)].to_numpy(dtype=np.float32)
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
        block["refit_week"] = refit_week
        block["prediction_block"] = ordinal
        output = block[
            [
                "week_index",
                "date",
                "ticker",
                "target",
                "target_verification_status",
                "actual_rank",
                "component_id",
                "model_id",
                "refit_week",
                "prediction_block",
                "raw_ranking_score",
                "percentile_score",
                "predicted_rank",
            ]
        ].copy()
        blocks.append(output)

        block_weekly = weekly_metrics(output, "percentile_score", spec.component_id)
        block_summary = summarize(
            block_weekly, spec.component_id, f"refit_block_{refit_week}"
        )
        refit_rows.append(
            {
                "model_id": model_id,
                "component_id": spec.component_id,
                "refit_week": refit_week,
                "training_start_week": training_start,
                "training_end_week": refit_week - 1,
                "training_weeks": int(training["week_index"].nunique()),
                "training_rows": int(len(training)),
                "prediction_start_week": refit_week,
                "prediction_end_week": prediction_end,
                "prediction_weeks": int(output["week_index"].nunique()),
                "prediction_rows": int(len(output)),
                "features": json.dumps(spec.features),
                "training_window_weeks": spec.training_window_weeks,
                "refit_interval_weeks": REFIT_INTERVAL_WEEKS,
                "training_label": "weekly_relevance_0_to_4",
                "evaluation_target": "continuous_target",
                "hyperparameters": json.dumps(spec.params, sort_keys=True),
                **{
                    key: value
                    for key, value in block_summary.items()
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

    predictions = pd.concat(blocks, ignore_index=True)
    if predictions.duplicated(["date", "ticker"]).any():
        raise AssertionError(f"Duplicate predictions for {spec.component_id}")
    expected = list(range(spec.training_window_weeks, total_weeks))
    if sorted(predictions["week_index"].unique()) != expected:
        raise AssertionError(f"Incomplete prediction weeks for {spec.component_id}")
    return (
        predictions,
        weekly_metrics(predictions, "percentile_score", spec.component_id),
        pd.DataFrame(refit_rows),
    )


def build_ensemble(component_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for number, spec in enumerate(COMPONENTS, start=1):
        source = component_predictions[spec.component_id]
        source = source.loc[source["week_index"].ge(FORMAL_START_WEEK)].copy()
        source = source[
            [
                "week_index",
                "date",
                "ticker",
                "target",
                "target_verification_status",
                "actual_rank",
                "model_id",
                "refit_week",
                "raw_ranking_score",
                "percentile_score",
                "predicted_rank",
            ]
        ].rename(
            columns={
                "model_id": f"model_{number}_id",
                "refit_week": f"model_{number}_refit_week",
                "raw_ranking_score": f"model_{number}_raw_score",
                "percentile_score": f"model_{number}_percentile",
                "predicted_rank": f"model_{number}_predicted_rank",
            }
        )
        if merged is None:
            merged = source
        else:
            source = source.drop(
                columns=["target", "target_verification_status", "actual_rank"]
            )
            merged = merged.merge(
                source,
                on=["week_index", "date", "ticker"],
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise AssertionError("No component predictions")

    refit_columns = [f"model_{number}_refit_week" for number in range(1, 4)]
    if not merged[refit_columns].nunique(axis=1).eq(1).all():
        raise AssertionError("Component refit schedules are not aligned")
    merged["refit_week"] = merged["model_1_refit_week"]
    merged["prediction_block"] = (
        (merged["refit_week"] - FORMAL_START_WEEK) // REFIT_INTERVAL_WEEKS + 1
    ).astype(int)
    merged["ensemble_model_id"] = merged["refit_week"].map(
        lambda value: f"ensemble_refit_{int(value):04d}"
    )
    percentile_columns = [f"model_{number}_percentile" for number in range(1, 4)]
    merged["ensemble_ranking_score"] = merged[percentile_columns].mean(axis=1)
    merged["ensemble_predicted_rank"] = merged.groupby("week_index", sort=False)[
        "ensemble_ranking_score"
    ].rank(method="first", ascending=False).astype(int)
    merged["rank_difference"] = merged["ensemble_predicted_rank"] - merged["actual_rank"]
    merged.insert(6, "training_label", "weekly_relevance_0_to_4")

    if sorted(merged["week_index"].unique()) != list(
        range(FORMAL_START_WEEK, TOTAL_EXPECTED_WEEKS)
    ):
        raise AssertionError("Formal ensemble prediction weeks are incomplete")
    if merged.duplicated(["date", "ticker"]).any():
        raise AssertionError("Formal ensemble contains duplicate predictions")
    return merged.sort_values(["week_index", "ticker"], kind="stable").reset_index(
        drop=True
    )


def build_weekly_table(
    component_weekly: dict[str, pd.DataFrame],
    ensemble_weekly: pd.DataFrame,
) -> pd.DataFrame:
    table = ensemble_weekly[
        ["week_index", "date", "number_of_stocks", "IC", "Rank_IC"]
    ].rename(columns={"IC": "ensemble_IC", "Rank_IC": "ensemble_Rank_IC"})
    for number, spec in enumerate(COMPONENTS, start=1):
        weekly = component_weekly[spec.component_id]
        weekly = weekly.loc[weekly["week_index"].ge(FORMAL_START_WEEK)]
        weekly = weekly[["week_index", "IC", "Rank_IC"]].rename(
            columns={"IC": f"model_{number}_IC", "Rank_IC": f"model_{number}_Rank_IC"}
        )
        table = table.merge(weekly, on="week_index", validate="one_to_one")
    return table


def build_block_metrics(
    ensemble: pd.DataFrame,
    ensemble_weekly: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for refit_week, block in ensemble.groupby("refit_week", sort=True):
        weeks = sorted(block["week_index"].unique())
        block_weekly = ensemble_weekly.loc[ensemble_weekly["week_index"].isin(weeks)]
        block_summary = summarize(
            block_weekly, "final_three_model_ensemble", f"refit_block_{int(refit_week)}"
        )
        rows.append(
            {
                "ensemble_model_id": f"ensemble_refit_{int(refit_week):04d}",
                "refit_week": int(refit_week),
                "prediction_start_week": min(weeks),
                "prediction_end_week": max(weeks),
                "prediction_weeks": len(weeks),
                "prediction_rows": int(len(block)),
                **{
                    key: value
                    for key, value in block_summary.items()
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
        for fold, (start, end) in enumerate(
            zip(FOLD_BOUNDARIES[:-1], FOLD_BOUNDARIES[1:]), start=1
        ):
            fold_weekly = weekly.loc[weekly["week_index"].between(start, end - 1)]
            rows.append(
                {
                    "fold": fold,
                    **summarize(fold_weekly, model_name, f"fold_{fold}"),
                }
            )
    return pd.DataFrame(rows)


def run(
    input_path: Path,
    output_dir: Path,
    model_dir: Path,
    n_jobs: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_data(input_path)

    component_predictions: dict[str, pd.DataFrame] = {}
    component_weekly: dict[str, pd.DataFrame] = {}
    refit_tables = []
    for spec in COMPONENTS:
        print(f"Training {spec.component_id}", flush=True)
        predictions, weekly, refits = evaluate_component(
            frame, spec, model_dir, n_jobs
        )
        component_predictions[spec.component_id] = predictions
        component_weekly[spec.component_id] = weekly
        refit_tables.append(refits)

    refit_metrics = pd.concat(refit_tables, ignore_index=True)
    ensemble = build_ensemble(component_predictions)
    ensemble_weekly = weekly_metrics(
        ensemble, "ensemble_ranking_score", "final_three_model_ensemble"
    )
    weekly_table = build_weekly_table(component_weekly, ensemble_weekly)
    block_metrics = build_block_metrics(ensemble, ensemble_weekly)

    component_rows = []
    formal_weekly: dict[str, pd.DataFrame] = {}
    for spec in COMPONENTS:
        formal = component_weekly[spec.component_id].loc[
            lambda table: table["week_index"].ge(FORMAL_START_WEEK)
        ].copy()
        formal_weekly[spec.component_id] = formal
        row = summarize(formal, spec.component_id, "formal_period")
        row["prediction_rows"] = int(
            component_predictions[spec.component_id]
            .loc[lambda table: table["week_index"].ge(FORMAL_START_WEEK)]
            .shape[0]
        )
        component_rows.append(row)

    ensemble_summary = summarize(
        ensemble_weekly, "final_three_model_ensemble", "formal_period"
    )
    ensemble_summary["prediction_rows"] = int(len(ensemble))
    component_rows.append(ensemble_summary)
    component_metrics = pd.DataFrame(component_rows)
    overall_metrics = pd.DataFrame([ensemble_summary])

    formal_weekly["final_three_model_ensemble"] = ensemble_weekly
    folds = build_fold_metrics(formal_weekly)

    overall_metrics.to_csv(output_dir / "overall_metrics.csv", index=False)
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    weekly_table.to_csv(output_dir / "weekly_metrics.csv", index=False)
    refit_metrics.to_csv(output_dir / "refit_model_metrics.csv", index=False)
    block_metrics.to_csv(output_dir / "ensemble_block_metrics.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    ensemble.to_csv(output_dir / "stock_predictions.csv", index=False)
    audit.to_csv(output_dir / "data_audit.csv", index=False)
    _write_json(output_dir / "model_configuration.json", model_configuration())
    _write_json(
        output_dir / "run_manifest.json",
        {
            "input_filename": input_path.name,
            "input_sha256": _sha256(input_path),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "xgboost_version": xgboost.__version__,
            "formal_prediction_weeks": int(ensemble["week_index"].nunique()),
            "formal_prediction_rows": int(len(ensemble)),
            "refit_model_instances": int(len(refit_metrics)),
            "ensemble_refit_blocks": int(len(block_metrics)),
            "target2_used": False,
            "backtest_performed": False,
        },
    )
    return overall_metrics


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    module_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the final three-model XGBRanker ranking module."
    )
    parser.add_argument(
        "--input", type=Path, default=project_root / "test_45_factors_weekly.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=module_root / "data")
    parser.add_argument("--model-dir", type=Path, default=module_root / "models")
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.input, args.output_dir, args.model_dir, args.n_jobs)
    print("\nFinal XGBRanker evaluation")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nMetrics and predictions: {args.output_dir.resolve()}")
    print(f"Saved models: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
