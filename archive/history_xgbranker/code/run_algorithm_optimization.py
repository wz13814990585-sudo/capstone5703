from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecapital_xgbranker.algorithm_optimization import (
    AlgorithmSearchConfig,
    run_algorithm_optimization,
)
from safecapital_xgbranker.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-stage XGBRanker algorithm optimization using all available data "
            "as research-only walk-forward time cross-validation. No backtest."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("test_45_factors_weekly.csv"))
    parser.add_argument(
        "--factor-definitions", type=Path, default=Path("factor_definitions(1).csv")
    )
    parser.add_argument(
        "--feature-candidates",
        type=Path,
        default=Path("history_fine/data/feature_search/robust_improvement_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("history_fine/data/algorithm_optimization"),
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=5703)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        input_path=args.input,
        factor_definitions_path=args.factor_definitions,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed,
    )
    result = run_algorithm_optimization(
        config,
        search=AlgorithmSearchConfig(),
        feature_candidate_path=args.feature_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
