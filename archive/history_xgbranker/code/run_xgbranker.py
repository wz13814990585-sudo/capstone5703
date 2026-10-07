from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecapital_xgbranker.config import (
    OPTIMIZATION_FEATURE_SETS,
    ExperimentConfig,
)
from safecapital_xgbranker.optimization import SearchConfig, run_rank_ic_optimization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-aware XGBRanker feature/time search with an untouched final holdout. "
            "No strategy or backtest is run."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("test_45_factors_weekly.csv"))
    parser.add_argument(
        "--factor-definitions", type=Path, default=Path("factor_definitions(1).csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("history_fine/data/rankic_optimization"),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=list(OPTIMIZATION_FEATURE_SETS),
        default=list(OPTIMIZATION_FEATURE_SETS),
    )
    parser.add_argument("--hyperparameter-train-weeks", type=int, default=208)
    parser.add_argument("--hyperparameter-validation-weeks", type=int, default=52)
    parser.add_argument("--selection-start-week", type=int, default=312)
    parser.add_argument("--final-holdout-weeks", type=int, default=104)
    parser.add_argument(
        "--rolling-window-candidates",
        nargs="+",
        type=int,
        default=[104, 156, 208, 260, 312],
    )
    parser.add_argument(
        "--refit-interval-candidates",
        nargs="+",
        type=int,
        default=[2, 4, 8, 13],
    )
    parser.add_argument(
        "--fixed-window-candidates",
        nargs="+",
        default=["all_history", "156", "208", "260", "312"],
    )
    parser.add_argument("--top-feature-sets-per-mode", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=5703)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def _parse_fixed_windows(values: list[str]) -> tuple[str | int, ...]:
    parsed: list[str | int] = []
    for value in values:
        parsed.append("all_history" if value == "all_history" else int(value))
    return tuple(parsed)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        input_path=args.input,
        factor_definitions_path=args.factor_definitions,
        output_dir=args.output_dir,
        factor_set_name=args.feature_sets[0],
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    search = SearchConfig(
        hyperparameter_train_weeks=args.hyperparameter_train_weeks,
        hyperparameter_validation_weeks=args.hyperparameter_validation_weeks,
        selection_start_week=args.selection_start_week,
        final_holdout_weeks=args.final_holdout_weeks,
        rolling_window_candidates=tuple(args.rolling_window_candidates),
        refit_interval_candidates=tuple(args.refit_interval_candidates),
        fixed_window_candidates=_parse_fixed_windows(args.fixed_window_candidates),
        top_feature_sets_per_mode=args.top_feature_sets_per_mode,
    )
    result = run_rank_ic_optimization(
        config,
        search=search,
        feature_sets=args.feature_sets,
    )
    concise = {
        "output_dir": result["output_dir"],
        "time_grid_feature_sets": result["time_grid_feature_sets"],
        "primary_selected_configuration": result[
            "primary_selected_configuration"
        ],
        "final_holdout_metrics": result["final_holdout_metrics"],
        "backtest_performed": result["backtest_performed"],
        "final_holdout_used_for_selection": result[
            "final_holdout_used_for_selection"
        ],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
