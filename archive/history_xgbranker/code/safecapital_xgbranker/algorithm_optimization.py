from __future__ import annotations

import itertools
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost

from .config import FUNDAMENTAL_6, MACRO_4, ExperimentConfig, TimeConfig
from .data import load_and_prepare_data
from .metrics import calculate_weekly_metrics, summarize_weekly_metrics
from .optimization import evaluate_schedule


CURRENT_BEST_3 = [
    "cashflow_to_assets",
    "debt_to_assets",
    "macro_dgs10_change_5obs",
]


@dataclass(frozen=True)
class AlgorithmSearchConfig:
    evaluation_start_week: int = 260
    evaluation_end_week: int = 522
    fold_boundaries: tuple[int, ...] = (260, 325, 390, 456, 522)
    baseline_window_weeks: int = 104
    baseline_refit_interval_weeks: int = 13
    window_candidates: tuple[str | int, ...] = (
        52,
        78,
        104,
        130,
        156,
        208,
        "all_history",
    )
    refit_interval_candidates: tuple[int, ...] = (2, 4, 8, 13, 26)
    top_algorithms_for_joint_search: int = 3
    top_schedules_for_joint_search: int = 3
    top_features_for_joint_search: int = 3
    top_single_models_for_ensemble: int = 5

    def validate(self, total_weeks: int) -> None:
        if self.evaluation_start_week < 1:
            raise ValueError("evaluation_start_week must leave training history")
        if not self.evaluation_start_week < self.evaluation_end_week <= total_weeks:
            raise ValueError("Evaluation interval is outside the data")
        if self.fold_boundaries[0] != self.evaluation_start_week:
            raise ValueError("First fold boundary must equal evaluation_start_week")
        if self.fold_boundaries[-1] != self.evaluation_end_week:
            raise ValueError("Last fold boundary must equal evaluation_end_week")
        if tuple(sorted(set(self.fold_boundaries))) != self.fold_boundaries:
            raise ValueError("Fold boundaries must be strictly increasing")
        if any(value <= 0 for value in self.refit_interval_candidates):
            raise ValueError("Refit intervals must be positive")


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
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def algorithm_candidates() -> list[dict[str, Any]]:
    common = {
        "tree_method": "hist",
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "eval_metric": "ndcg@10",
    }

    def candidate(
        candidate_id: str,
        objective: str,
        bins: int,
        **params: Any,
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "objective": objective,
            "relevance_bins": bins,
            "params": common | params,
        }

    return [
        candidate(
            "current_pairwise_d4",
            "rank:pairwise",
            5,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=160,
            min_child_weight=10,
            subsample=1.0,
            colsample_bytree=0.9,
        ),
        candidate("pairwise_d2", "rank:pairwise", 5, max_depth=2, learning_rate=0.05, n_estimators=240, min_child_weight=10),
        candidate("pairwise_d3", "rank:pairwise", 5, max_depth=3, learning_rate=0.05, n_estimators=200, min_child_weight=10),
        candidate("pairwise_d5", "rank:pairwise", 5, max_depth=5, learning_rate=0.03, n_estimators=220, min_child_weight=10),
        candidate("pairwise_d3_mcw20", "rank:pairwise", 5, max_depth=3, learning_rate=0.04, n_estimators=240, min_child_weight=20, reg_lambda=2.0),
        candidate("pairwise_d3_mcw40", "rank:pairwise", 5, max_depth=3, learning_rate=0.04, n_estimators=240, min_child_weight=40, reg_lambda=3.0),
        candidate("pairwise_d3_regularized", "rank:pairwise", 5, max_depth=3, learning_rate=0.03, n_estimators=280, min_child_weight=20, reg_alpha=0.1, reg_lambda=5.0, gamma=0.05),
        candidate("pairwise_d2_slow", "rank:pairwise", 5, max_depth=2, learning_rate=0.025, n_estimators=360, min_child_weight=20, reg_lambda=3.0),
        candidate("pairwise_d4_fast", "rank:pairwise", 5, max_depth=4, learning_rate=0.08, n_estimators=100, min_child_weight=15, reg_lambda=2.0),
        candidate("pairwise_leaves15", "rank:pairwise", 5, max_depth=0, max_leaves=15, grow_policy="lossguide", learning_rate=0.04, n_estimators=220, min_child_weight=15, reg_lambda=2.0),
        candidate("pairwise_leaves31", "rank:pairwise", 5, max_depth=0, max_leaves=31, grow_policy="lossguide", learning_rate=0.03, n_estimators=240, min_child_weight=20, reg_lambda=3.0),
        candidate("ndcg_d2_b5", "rank:ndcg", 5, max_depth=2, learning_rate=0.04, n_estimators=260, min_child_weight=10),
        candidate("ndcg_d3_b5", "rank:ndcg", 5, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=10),
        candidate("ndcg_d4_b5", "rank:ndcg", 5, max_depth=4, learning_rate=0.03, n_estimators=240, min_child_weight=15, reg_lambda=2.0),
        candidate("ndcg_d3_b10", "rank:ndcg", 10, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=15),
        candidate("ndcg_d4_b10", "rank:ndcg", 10, max_depth=4, learning_rate=0.03, n_estimators=240, min_child_weight=20, reg_lambda=2.0),
        candidate("ndcg_mean_pairs4", "rank:ndcg", 5, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=15, lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=4),
        candidate("ndcg_mean_pairs8", "rank:ndcg", 5, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=15, lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8),
        candidate("ndcg_topk_pairs8", "rank:ndcg", 5, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=15, lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8),
        candidate("ndcg_topk_pairs16", "rank:ndcg", 5, max_depth=3, learning_rate=0.04, n_estimators=220, min_child_weight=15, lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=16),
    ]


def add_cross_sectional_transforms(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    stock_features = [feature for feature in features if feature not in MACRO_4]
    grouped = result.groupby("week_index", sort=False)
    for feature in stock_features:
        ranks = grouped[feature].rank(method="average", pct=True)
        result[f"csrank__{feature}"] = ranks.astype(float)
        means = grouped[feature].transform("mean")
        stds = grouped[feature].transform("std").replace(0, np.nan)
        result[f"cszscore__{feature}"] = (
            (result[feature] - means) / stds
        ).clip(-5.0, 5.0)
    return result


def resolve_features(features: list[str], transform: str) -> list[str]:
    if transform not in {"raw", "cs_rank", "cs_zscore"}:
        raise ValueError(f"Unknown transform: {transform}")
    resolved = []
    for feature in features:
        if transform == "raw" or feature in MACRO_4:
            resolved.append(feature)
        elif transform == "cs_rank":
            resolved.append(f"csrank__{feature}")
        else:
            resolved.append(f"cszscore__{feature}")
    return resolved


def _fold_metrics(
    weekly: pd.DataFrame,
    search: AlgorithmSearchConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fold_values: list[float] = []
    for index, (start, end) in enumerate(
        zip(search.fold_boundaries[:-1], search.fold_boundaries[1:]), start=1
    ):
        fold = weekly.loc[weekly["week_index"].between(start, end - 1)]
        summary = summarize_weekly_metrics(fold)
        result[f"fold_{index}_start_week"] = start
        result[f"fold_{index}_end_week"] = end - 1
        result[f"fold_{index}_Rank_IC"] = summary["Rank_IC"]
        result[f"fold_{index}_Rank_ICIR"] = summary["Rank_ICIR"]
        fold_values.append(float(summary["Rank_IC"]))
    result["minimum_fold_Rank_IC"] = float(min(fold_values))
    result["fold_Rank_IC_std"] = float(np.std(fold_values, ddof=1))
    result["all_folds_positive"] = bool(all(value > 0 for value in fold_values))
    result["robust_score"] = float(np.mean(fold_values) - np.std(fold_values, ddof=1))
    return result


def _rank(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(
        ["Rank_IC", "Rank_ICIR", "minimum_fold_Rank_IC", "configuration_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )


def _window_label(window: str | int) -> str:
    return "all_history" if window == "all_history" else str(int(window))


def _load_feature_candidates(path: Path) -> dict[str, list[str]]:
    robust = pd.read_csv(path)
    result: dict[str, list[str]] = {
        str(row["candidate_id"]): json.loads(row["feature_list"])
        for _, row in robust.iterrows()
    }
    result["fundamental_6"] = list(FUNDAMENTAL_6)
    result["b3_f4_f5_add_t21_m01_t25"] = [
        "cashflow_to_assets",
        "debt_to_assets",
        "volume_zscore_20",
        "macro_vix_level",
        "close_location_value",
    ]
    result["b2_f4_f5_add_t21"] = [
        "cashflow_to_assets",
        "debt_to_assets",
        "volume_zscore_20",
    ]
    return result


def _make_frozen(algorithm: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": algorithm["objective"],
        "relevance_bins": algorithm["relevance_bins"],
        "hyperparameters": algorithm["params"],
        "selected_candidate_id": algorithm["candidate_id"],
    }


def run_algorithm_optimization(
    base_config: ExperimentConfig,
    *,
    search: AlgorithmSearchConfig | None = None,
    feature_candidate_path: Path = Path(
        "history_fine/data/feature_search/robust_improvement_candidates.csv"
    ),
) -> dict[str, Any]:
    search = search or AlgorithmSearchConfig()
    algorithms = {item["candidate_id"]: item for item in algorithm_candidates()}
    feature_sets = _load_feature_candidates(feature_candidate_path)
    union_features = list(
        dict.fromkeys(
            feature for features in feature_sets.values() for feature in features
        )
    )
    time = TimeConfig(
        warmup_weeks=260,
        warmup_train_weeks=208,
        warmup_validation_weeks=52,
        rolling_window_weeks=search.baseline_window_weeks,
        refit_interval_weeks=search.baseline_refit_interval_weeks,
        forecast_horizon_weeks=1,
    )
    base_config = replace(base_config, time=time, feature_override=None)
    frame, diagnostics = load_and_prepare_data(
        input_path=base_config.input_path,
        factor_definitions_path=base_config.factor_definitions_path,
        features=union_features,
        forecast_horizon_weeks=1,
    )
    total_weeks = int(frame["week_index"].nunique())
    search.validate(total_weeks)
    frame = add_cross_sectional_transforms(frame, union_features)

    cache: dict[tuple[str, str, str, str, int], tuple] = {}

    def evaluate(
        algorithm_id: str,
        transform: str,
        feature_set_id: str,
        window: str | int,
        refit_interval: int,
        stage: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        key = (
            algorithm_id,
            transform,
            feature_set_id,
            _window_label(window),
            int(refit_interval),
        )
        result = cache.get(key)
        features = feature_sets[feature_set_id]
        resolved = resolve_features(features, transform)
        if result is None:
            config = replace(
                base_config,
                factor_set_name=f"{feature_set_id}__{transform}__{algorithm_id}",
                feature_override=tuple(resolved),
            )
            result = evaluate_schedule(
                frame,
                config,
                _make_frozen(algorithms[algorithm_id]),
                phase="all_data_time_cv_research",
                evaluation_start_week=search.evaluation_start_week,
                evaluation_end_week=search.evaluation_end_week,
                training_mode="rolling",
                training_window_weeks=window,
                refit_interval_weeks=refit_interval,
            )
            cache[key] = result
        predictions, weekly, _, overall = result
        configuration_id = (
            f"{algorithm_id}__{transform}__{feature_set_id}__"
            f"w{_window_label(window)}__r{refit_interval}"
        )
        row = dict(overall)
        row.update(
            {
                "stage": stage,
                "configuration_id": configuration_id,
                "algorithm_id": algorithm_id,
                "feature_transform": transform,
                "feature_set_id": feature_set_id,
                "original_feature_list": json.dumps(features),
                "resolved_feature_list": json.dumps(resolved),
                "training_window_weeks": _window_label(window),
                "refit_interval_weeks": int(refit_interval),
                "algorithm_parameters": json.dumps(
                    algorithms[algorithm_id], sort_keys=True
                ),
            }
        )
        row.update(_fold_metrics(weekly, search))
        return predictions, weekly, row

    current_algorithm = "current_pairwise_d4"
    current_feature_set = "b2_f4_f5_add_m04"
    current_window = search.baseline_window_weeks
    current_refit = search.baseline_refit_interval_weeks

    print("Stage 1/6: cross-sectional preprocessing search", flush=True)
    preprocessing_rows = []
    for transform in ("raw", "cs_rank", "cs_zscore"):
        _, _, row = evaluate(
            current_algorithm,
            transform,
            current_feature_set,
            current_window,
            current_refit,
            "preprocessing_search",
        )
        preprocessing_rows.append(row)
    preprocessing = _rank(pd.DataFrame(preprocessing_rows))
    best_transform = str(preprocessing.iloc[0]["feature_transform"])

    print(
        f"Stage 2/6: {len(algorithms)} XGBRanker parameter candidates "
        f"with transform={best_transform}",
        flush=True,
    )
    hyper_rows = []
    for index, algorithm_id in enumerate(algorithms, start=1):
        print(f"  hyper {index}/{len(algorithms)}: {algorithm_id}", flush=True)
        _, _, row = evaluate(
            algorithm_id,
            best_transform,
            current_feature_set,
            current_window,
            current_refit,
            "hyperparameter_search",
        )
        hyper_rows.append(row)
    hyper = _rank(pd.DataFrame(hyper_rows))
    best_algorithm = str(hyper.iloc[0]["algorithm_id"])

    time_specs = list(
        itertools.product(search.window_candidates, search.refit_interval_candidates)
    )
    print(
        f"Stage 3/6: {len(time_specs)} rolling-window/refit schedules",
        flush=True,
    )
    time_rows = []
    for index, (window, refit_interval) in enumerate(time_specs, start=1):
        if index == 1 or index % 5 == 0 or index == len(time_specs):
            print(f"  schedule {index}/{len(time_specs)}", flush=True)
        _, _, row = evaluate(
            best_algorithm,
            best_transform,
            current_feature_set,
            window,
            refit_interval,
            "time_schedule_search",
        )
        time_rows.append(row)
    schedules = _rank(pd.DataFrame(time_rows))
    best_schedule = schedules.iloc[0]
    best_window: str | int = str(best_schedule["training_window_weeks"])
    if best_window != "all_history":
        best_window = int(best_window)
    best_refit = int(best_schedule["refit_interval_weeks"])

    print(f"Stage 4/6: {len(feature_sets)} feature finalists", flush=True)
    feature_rows = []
    for index, feature_set_id in enumerate(feature_sets, start=1):
        print(f"  feature {index}/{len(feature_sets)}: {feature_set_id}", flush=True)
        _, _, row = evaluate(
            best_algorithm,
            best_transform,
            feature_set_id,
            best_window,
            best_refit,
            "feature_finalist_search",
        )
        feature_rows.append(row)
    features_table = _rank(pd.DataFrame(feature_rows))

    top_algorithms = hyper.head(search.top_algorithms_for_joint_search)[
        "algorithm_id"
    ].tolist()
    top_schedules = schedules.head(search.top_schedules_for_joint_search)[
        ["training_window_weeks", "refit_interval_weeks"]
    ].to_dict(orient="records")
    top_feature_sets = features_table.head(search.top_features_for_joint_search)[
        "feature_set_id"
    ].tolist()
    joint_specs = list(itertools.product(top_algorithms, top_schedules, top_feature_sets))
    print(f"Stage 5/6: {len(joint_specs)} joint finalists", flush=True)
    joint_rows = []
    for index, (algorithm_id, schedule_row, feature_set_id) in enumerate(
        joint_specs, start=1
    ):
        window = str(schedule_row["training_window_weeks"])
        if window != "all_history":
            window = int(window)
        refit_interval = int(schedule_row["refit_interval_weeks"])
        if index == 1 or index % 5 == 0 or index == len(joint_specs):
            print(f"  joint {index}/{len(joint_specs)}", flush=True)
        _, _, row = evaluate(
            algorithm_id,
            best_transform,
            feature_set_id,
            window,
            refit_interval,
            "joint_finalist_search",
        )
        joint_rows.append(row)
    joint = _rank(pd.DataFrame(joint_rows)).drop_duplicates("configuration_id")

    print("Stage 6/6: equal-weight rank ensembles", flush=True)
    top_single = joint.head(search.top_single_models_for_ensemble).copy()
    prediction_tables: dict[str, pd.DataFrame] = {}
    for _, single in top_single.iterrows():
        window = str(single["training_window_weeks"])
        if window != "all_history":
            window = int(window)
        predictions, _, _ = evaluate(
            str(single["algorithm_id"]),
            str(single["feature_transform"]),
            str(single["feature_set_id"]),
            window,
            int(single["refit_interval_weeks"]),
            "ensemble_component",
        )
        component_id = str(single["configuration_id"])
        normalized = predictions[
            ["week_index", "date", "ticker", "actual_return", "ranking_score"]
        ].copy()
        normalized["normalized_score"] = normalized.groupby(
            "week_index", sort=False
        )["ranking_score"].rank(method="average", pct=True)
        prediction_tables[component_id] = normalized.drop(columns="ranking_score")

    ensemble_rows = []
    ensemble_weekly: dict[str, pd.DataFrame] = {}
    component_ids = list(prediction_tables)
    for size in range(2, len(component_ids) + 1):
        for members in itertools.combinations(component_ids, size):
            merged = prediction_tables[members[0]].rename(
                columns={"normalized_score": "score_0"}
            )
            for member_index, member in enumerate(members[1:], start=1):
                addition = prediction_tables[member][
                    ["week_index", "ticker", "normalized_score"]
                ].rename(columns={"normalized_score": f"score_{member_index}"})
                merged = merged.merge(
                    addition,
                    on=["week_index", "ticker"],
                    how="inner",
                    validate="one_to_one",
                )
            score_columns = [f"score_{index}" for index in range(size)]
            merged["ranking_score"] = merged[score_columns].mean(axis=1)
            merged["model_id"] = "equal_weight_rank_ensemble"
            weekly = calculate_weekly_metrics(merged)
            overall = summarize_weekly_metrics(weekly)
            ensemble_id = f"ensemble_{size}_" + "__".join(members)
            row = dict(overall)
            row.update(
                {
                    "stage": "ensemble_search",
                    "configuration_id": ensemble_id,
                    "ensemble_size": size,
                    "component_configuration_ids": json.dumps(members),
                    "combination_method": "weekly percentile-rank then equal-weight mean",
                }
            )
            row.update(_fold_metrics(weekly, search))
            ensemble_rows.append(row)
            ensemble_weekly[ensemble_id] = weekly
    ensembles = _rank(pd.DataFrame(ensemble_rows))

    best_single = joint.iloc[0]
    best_ensemble = ensembles.iloc[0]
    ensemble_wins = float(best_ensemble["Rank_IC"]) > float(best_single["Rank_IC"])
    winner_type = "ensemble" if ensemble_wins else "single_model"
    winner = best_ensemble if ensemble_wins else best_single

    _, current_weekly, current_row = evaluate(
        current_algorithm,
        "raw",
        current_feature_set,
        current_window,
        current_refit,
        "current_baseline",
    )
    if ensemble_wins:
        winning_weekly = ensemble_weekly[str(winner["configuration_id"])].copy()
    else:
        winner_window = str(winner["training_window_weeks"])
        if winner_window != "all_history":
            winner_window = int(winner_window)
        _, winning_weekly, _ = evaluate(
            str(winner["algorithm_id"]),
            str(winner["feature_transform"]),
            str(winner["feature_set_id"]),
            winner_window,
            int(winner["refit_interval_weeks"]),
            "winner",
        )
    paired = winning_weekly[["week_index", "Rank_IC"]].merge(
        current_weekly[["week_index", "Rank_IC"]],
        on="week_index",
        suffixes=("_winner", "_current"),
        validate="one_to_one",
    )
    paired["Rank_IC_delta"] = paired["Rank_IC_winner"] - paired["Rank_IC_current"]
    delta_mean = float(paired["Rank_IC_delta"].mean())
    delta_std = float(paired["Rank_IC_delta"].std(ddof=1))
    paired_t = delta_mean / (delta_std / np.sqrt(len(paired))) if delta_std > 0 else np.nan
    improvement = {
        "current_CV_Rank_IC": float(current_row["Rank_IC"]),
        "winner_CV_Rank_IC": float(winner["Rank_IC"]),
        "absolute_Rank_IC_improvement": float(winner["Rank_IC"] - current_row["Rank_IC"]),
        "relative_Rank_IC_improvement": float(
            (winner["Rank_IC"] - current_row["Rank_IC"])
            / abs(float(current_row["Rank_IC"]))
        ),
        "weekly_win_ratio": float((paired["Rank_IC_delta"] > 0).mean()),
        "paired_delta_t_stat": float(paired_t),
    }

    out = base_config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    preprocessing.to_csv(out / "preprocessing_search_metrics.csv", index=False)
    hyper.to_csv(out / "hyperparameter_search_metrics.csv", index=False)
    schedules.to_csv(out / "time_schedule_search_metrics.csv", index=False)
    features_table.to_csv(out / "feature_candidate_search_metrics.csv", index=False)
    joint.to_csv(out / "joint_finalist_search_metrics.csv", index=False)
    ensembles.to_csv(out / "ensemble_search_metrics.csv", index=False)
    winning_weekly.to_csv(out / "winning_weekly_metrics.csv", index=False)
    paired.to_csv(out / "winner_vs_current_weekly_comparison.csv", index=False)

    selected = {
        "winner_type": winner_type,
        "winner": winner.to_dict(),
        "best_single_model": best_single.to_dict(),
        "best_ensemble": best_ensemble.to_dict(),
        "improvement_vs_current_best3_algorithm": improvement,
        "selection_metric": "maximum 262-week walk-forward Mean Rank IC",
        "research_only_no_untouched_holdout": True,
        "backtest_performed": False,
    }
    _write_json(out / "selected_algorithm.json", selected)
    manifest = {
        "search_config": asdict(search),
        "algorithm_candidates": algorithms,
        "feature_candidates": feature_sets,
        "candidate_counts": {
            "preprocessing": len(preprocessing),
            "hyperparameters": len(hyper),
            "time_schedules": len(schedules),
            "features": len(features_table),
            "joint_finalists": len(joint),
            "ensembles": len(ensembles),
        },
        "input_filename": diagnostics["input_filename"],
        "input_sha256": diagnostics["input_sha256"],
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "research_only_no_untouched_holdout": True,
        "backtest_performed": False,
    }
    _write_json(out / "experiment_manifest.json", manifest)

    fold_columns = [
        f"fold_{index}_Rank_IC"
        for index in range(1, len(search.fold_boundaries))
    ]
    stage_summary = pd.DataFrame(
        [
            ["Current algorithm", current_row["configuration_id"], current_row["Rank_IC"], current_row["Rank_ICIR"], current_row["minimum_fold_Rank_IC"]],
            ["Best preprocessing", preprocessing.iloc[0]["configuration_id"], preprocessing.iloc[0]["Rank_IC"], preprocessing.iloc[0]["Rank_ICIR"], preprocessing.iloc[0]["minimum_fold_Rank_IC"]],
            ["Best hyperparameters", hyper.iloc[0]["configuration_id"], hyper.iloc[0]["Rank_IC"], hyper.iloc[0]["Rank_ICIR"], hyper.iloc[0]["minimum_fold_Rank_IC"]],
            ["Best schedule", schedules.iloc[0]["configuration_id"], schedules.iloc[0]["Rank_IC"], schedules.iloc[0]["Rank_ICIR"], schedules.iloc[0]["minimum_fold_Rank_IC"]],
            ["Best feature finalist", features_table.iloc[0]["configuration_id"], features_table.iloc[0]["Rank_IC"], features_table.iloc[0]["Rank_ICIR"], features_table.iloc[0]["minimum_fold_Rank_IC"]],
            ["Best joint single", best_single["configuration_id"], best_single["Rank_IC"], best_single["Rank_ICIR"], best_single["minimum_fold_Rank_IC"]],
            ["Best ensemble", best_ensemble["configuration_id"], best_ensemble["Rank_IC"], best_ensemble["Rank_ICIR"], best_ensemble["minimum_fold_Rank_IC"]],
        ],
        columns=["stage", "configuration_id", "Rank_IC", "Rank_ICIR", "minimum_fold_Rank_IC"],
    )
    stage_summary.to_csv(out / "stage_summary.csv", index=False)

    winner_fold_values = [float(winner[column]) for column in fold_columns]
    report = f"""# XGBRanker 算法优化报告

## 结论

在不新增数据的前提下，本轮把 Week {search.evaluation_start_week}–{search.evaluation_end_week - 1} 共 {search.evaluation_end_week - search.evaluation_start_week} 周用于多时段 walk-forward 交叉验证和算法搜索。因为没有全新的未见数据，这里的“最优”仅指本次搜索空间与现有历史数据上的最高交叉验证 RankIC，不能证明未来真正的数学全局最优。

- 当前三因子算法 CV RankIC：{improvement['current_CV_Rank_IC']:.6f}
- 本轮胜者类型：`{winner_type}`
- 优化后 CV RankIC：{improvement['winner_CV_Rank_IC']:.6f}
- RankIC 绝对改善：{improvement['absolute_Rank_IC_improvement']:+.6f}
- RankIC 相对改善：{improvement['relative_Rank_IC_improvement']:+.2%}
- 逐周 RankIC 超过当前算法的比例：{improvement['weekly_win_ratio']:.2%}
- 配对周度改善 t-stat：{improvement['paired_delta_t_stat']:.3f}
- 胜者四个时间段 RankIC：{winner_fold_values}

## 为什么这样优化

1. **横截面预处理**：比较 raw、每周横截面 percentile rank 和横截面 z-score。这用来减少财务因子跨时间尺度变化对树分裂的影响。宏观因子在同一周对所有股票相同，所以保留原值，不做横截面排名。
2. **LambdaRank 和树结构**：搜索 pairwise/ndcg、树深、叶子数、learning rate、树数、min child weight、正则化和 LambdaRank pair sampling。原因是 RankIC 取决于顺序，不是收益数值误差。
3. **窗口与更新频率**：比较 52/78/104/130/156/208 周和 expanding history，以及 2/4/8/13/26 周 refit。较短窗口更灵敏，较长窗口更稳定，必须由时间 CV 决定。
4. **Feature 复核**：不仅使用之前的三因子，还重新比较了之前在两个子时期均有改善的组合。
5. **排名集成**：对最强单模型的每周预测分数先转成 percentile rank，再等权平均。这避免不同 XGBRanker 分数尺度不同，并检验模型多样性是否能提高排名稳定性。

## 各阶段最优结果

```text
{stage_summary.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 胜者配置

```json
{json.dumps(_json_safe(winner.to_dict()), ensure_ascii=False, indent=2)}
```

本轮只运行 XGBRanker 机器学习和 IC/RankIC 评估，没有运行交易策略或回测。
"""
    (out / "algorithm_optimization_report.md").write_text(report, encoding="utf-8")

    return {
        "output_dir": str(out.resolve()),
        "winner_type": winner_type,
        "winner": winner.to_dict(),
        "improvement": improvement,
        "research_only_no_untouched_holdout": True,
        "backtest_performed": False,
    }
