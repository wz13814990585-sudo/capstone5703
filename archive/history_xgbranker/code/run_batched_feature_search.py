from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecapital_xgbranker.batched_feature_search import (
    BatchedFeatureSearchConfig,
    run_batched_feature_search,
)
from safecapital_xgbranker.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-batch XGBRanker feature combination search on the research "
            "selection period only. The existing final holdout is not evaluated."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("test_45_factors_weekly.csv"))
    parser.add_argument(
        "--factor-definitions", type=Path, default=Path("factor_definitions(1).csv")
    )
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=Path("history_fine/data/rankic_optimization/frozen_configs_by_feature_set.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("history_fine/data/feature_search"),
    )
    parser.add_argument("--selection-start-week", type=int, default=312)
    parser.add_argument("--selection-end-week", type=int, default=418)
    parser.add_argument("--discovery-weeks", type=int, default=53)
    parser.add_argument("--rolling-window-weeks", type=int, default=104)
    parser.add_argument("--refit-interval-weeks", type=int, default=13)
    parser.add_argument("--top-fundamental-bases", type=int, default=2)
    parser.add_argument("--top-single-additions-per-base", type=int, default=6)
    parser.add_argument("--top-discovery-shortlist", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=5703)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = ExperimentConfig(
        input_path=args.input,
        factor_definitions_path=args.factor_definitions,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    search = BatchedFeatureSearchConfig(
        selection_start_week=args.selection_start_week,
        selection_end_week=args.selection_end_week,
        discovery_weeks=args.discovery_weeks,
        rolling_window_weeks=args.rolling_window_weeks,
        refit_interval_weeks=args.refit_interval_weeks,
        top_fundamental_bases=args.top_fundamental_bases,
        top_single_additions_per_base=args.top_single_additions_per_base,
        top_discovery_shortlist=args.top_discovery_shortlist,
    )
    result = run_batched_feature_search(
        base,
        search=search,
        frozen_config_path=args.frozen_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
