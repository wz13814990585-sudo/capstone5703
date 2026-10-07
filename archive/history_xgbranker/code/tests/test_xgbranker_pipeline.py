from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from safecapital_xgbranker.comparison import fixed_oos
from safecapital_xgbranker.batched_feature_search import (
    BatchedFeatureSearchConfig,
    fundamental_subset_candidates,
)
from safecapital_xgbranker.algorithm_optimization import (
    AlgorithmSearchConfig,
    add_cross_sectional_transforms,
    algorithm_candidates,
    resolve_features,
)
from safecapital_xgbranker.config import (
    FEATURE_BLOCKS,
    FEATURE_COMBINATIONS,
    MACRO_4,
    OPTIMIZATION_FEATURE_SETS,
    STOCK_37,
    ExperimentConfig,
    TimeConfig,
)
from safecapital_xgbranker.data import assert_feature_safety, eligible_training_rows
from safecapital_xgbranker.labels import add_relevance_labels, group_sizes
from safecapital_xgbranker.metrics import calculate_weekly_metrics, summarize_weekly_metrics
from safecapital_xgbranker.pipeline import build_refit_weeks
from safecapital_xgbranker.optimization import SearchConfig, evaluate_schedule
from XGboost_ranker.xgboost_ranker import (
    COMPONENTS as FROZEN_COMPONENTS,
    build_refit_weeks as build_frozen_refit_weeks,
    model_configuration,
    relevance_labels,
)


class DataAndLeakageTests(unittest.TestCase):
    def test_forbidden_features_raise(self) -> None:
        for feature in ["target", "target2", "future_return", "fwd_1w", "label_x"]:
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                assert_feature_safety(["ret_1d", feature])

    def test_observability_is_strict(self) -> None:
        frame = pd.DataFrame(
            {
                "week_index": [0, 1, 2, 3],
                "label_end_week": [1, 2, 3, 4],
            }
        )
        eligible = eligible_training_rows(frame, range(4), refit_week=3)
        self.assertEqual(eligible["week_index"].tolist(), [0, 1])
        self.assertTrue((eligible["label_end_week"] < 3).all())

    def test_refit_schedule_has_observability_embargo(self) -> None:
        self.assertEqual(build_refit_weeks(15, 6, 1, 4), [7, 11])


class FrozenEnsembleTests(unittest.TestCase):
    def test_frozen_schedules_start_after_full_windows(self) -> None:
        self.assertEqual(build_frozen_refit_weeks(156, 522)[0], 156)
        self.assertEqual(build_frozen_refit_weeks(156, 522)[-1], 520)
        self.assertEqual(build_frozen_refit_weeks(52, 522)[0], 52)
        self.assertEqual(build_frozen_refit_weeks(52, 522)[-1], 520)

    def test_final_model_uses_weekly_relevance_labels(self) -> None:
        config = model_configuration()
        self.assertEqual(config["evaluation_target"]["column"], "target")
        self.assertEqual(config["training_label"]["levels"], [0, 1, 2, 3, 4])
        self.assertFalse(config["training_label"]["target2_used"])
        self.assertEqual([spec.training_window_weeks for spec in FROZEN_COMPONENTS], [156, 52, 52])

    def test_final_labels_are_generated_within_week(self) -> None:
        frame = pd.DataFrame(
            {
                "week_index": [0] * 5 + [1] * 5,
                "target": [10, 20, 30, 40, 50, -5, -4, -3, -2, -1],
            }
        )
        labels = relevance_labels(frame)
        self.assertEqual(labels[:5].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(labels[5:].tolist(), [0, 1, 2, 3, 4])


class RankingTests(unittest.TestCase):
    def test_relevance_is_generated_within_each_week(self) -> None:
        frame = pd.DataFrame(
            {
                "week_index": [0] * 5 + [1] * 5,
                "target": [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
            }
        )
        labelled = add_relevance_labels(frame, bins=5)
        self.assertEqual(labelled.loc[:4, "relevance_label"].tolist(), list(range(5)))
        self.assertEqual(labelled.loc[5:, "relevance_label"].tolist(), list(range(5)))
        self.assertEqual(group_sizes(labelled).tolist(), [5, 5])

    def test_weekly_metrics_are_cross_sectional(self) -> None:
        predictions = pd.DataFrame(
            {
                "week_index": [0, 0, 0, 1, 1, 1],
                "date": pd.to_datetime(["2020-01-03"] * 3 + ["2020-01-10"] * 3),
                "model_id": ["m1"] * 6,
                "ranking_score": [1, 2, 3, 3, 2, 1],
                "actual_return": [1, 2, 3, 1, 2, 3],
            }
        )
        weekly = calculate_weekly_metrics(predictions)
        summary = summarize_weekly_metrics(weekly)
        self.assertAlmostEqual(weekly.iloc[0]["Rank_IC"], 1.0)
        self.assertAlmostEqual(weekly.iloc[1]["Rank_IC"], -1.0)
        self.assertAlmostEqual(summary["Rank_IC"], 0.0)


class ComparisonTests(unittest.TestCase):
    def test_feature_suite_is_complete_block_factorial(self) -> None:
        self.assertEqual(len(FEATURE_COMBINATIONS), 7)
        expected = set(FEATURE_BLOCKS["F"] + FEATURE_BLOCKS["R"] + FEATURE_BLOCKS["P"])
        self.assertEqual(set(FEATURE_COMBINATIONS["priority_7"]), expected)
        self.assertEqual(
            set(FEATURE_COMBINATIONS["technical_5"]),
            set(FEATURE_BLOCKS["R"] + FEATURE_BLOCKS["P"]),
        )

    def test_fixed_mode_trains_once_and_covers_all_oos_weeks(self) -> None:
        rows = []
        for week_index in range(8):
            for ticker_index, ticker in enumerate(["A", "B", "C"]):
                rows.append(
                    {
                        "week_index": week_index,
                        "label_end_week": week_index + 1,
                        "date": pd.Timestamp("2020-01-03")
                        + pd.DateOffset(weeks=week_index),
                        "ticker": ticker,
                        "target": float(ticker_index - 1) / 100,
                        "revenue_growth_yoy": float(ticker_index),
                        "cashflow_to_assets": float(2 - ticker_index),
                    }
                )
        frame = pd.DataFrame(rows)
        config = ExperimentConfig(
            factor_set_name="fundamental_2",
            time=TimeConfig(
                warmup_weeks=2,
                warmup_train_weeks=1,
                warmup_validation_weeks=1,
                rolling_window_weeks=2,
                refit_interval_weeks=2,
                forecast_horizon_weeks=1,
            ),
        )
        frozen = {
            "objective": "rank:pairwise",
            "relevance_bins": 3,
            "hyperparameters": {},
        }

        class FakeRanker:
            def predict(self, values: np.ndarray) -> np.ndarray:
                return values[:, 0]

        with patch(
            "safecapital_xgbranker.comparison.fit_ranker",
            return_value=FakeRanker(),
        ) as fit:
            predictions, weekly, metadata, overall = fixed_oos(
                frame, config, frozen
            )
        fit.assert_called_once()
        self.assertEqual(predictions["model_id"].nunique(), 1)
        self.assertEqual(predictions["week_index"].nunique(), 5)
        self.assertEqual(len(weekly), 5)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(overall["number_of_refit_models"], 1)
        self.assertTrue(
            metadata["training_end_week"].iloc[0] + 1
            < metadata["refit_week"].iloc[0]
        )


class OptimizationTests(unittest.TestCase):
    def test_algorithm_search_folds_cover_evaluation_period(self) -> None:
        search = AlgorithmSearchConfig()
        search.validate(total_weeks=522)
        covered = []
        for start, end in zip(
            search.fold_boundaries[:-1], search.fold_boundaries[1:]
        ):
            covered.extend(range(start, end))
        self.assertEqual(covered, list(range(260, 522)))

    def test_cross_sectional_transform_preserves_macro_level(self) -> None:
        frame = pd.DataFrame(
            {
                "week_index": [0, 0, 1, 1],
                "cashflow_to_assets": [1.0, 3.0, 10.0, 20.0],
                "macro_dgs10_change_5obs": [0.1, 0.1, -0.2, -0.2],
            }
        )
        transformed = add_cross_sectional_transforms(
            frame, ["cashflow_to_assets", "macro_dgs10_change_5obs"]
        )
        self.assertEqual(
            transformed["csrank__cashflow_to_assets"].tolist(),
            [0.5, 1.0, 0.5, 1.0],
        )
        self.assertEqual(
            resolve_features(
                ["cashflow_to_assets", "macro_dgs10_change_5obs"], "cs_rank"
            ),
            ["csrank__cashflow_to_assets", "macro_dgs10_change_5obs"],
        )

    def test_algorithm_candidate_ids_are_unique(self) -> None:
        candidates = algorithm_candidates()
        self.assertEqual(len(candidates), 20)
        self.assertEqual(
            len({item["candidate_id"] for item in candidates}), len(candidates)
        )

    def test_feature_override_supports_research_combinations(self) -> None:
        config = ExperimentConfig(
            factor_set_name="research_candidate",
            feature_override=("revenue_growth_yoy", "asset_growth_yoy"),
        )
        self.assertEqual(
            config.features, ["revenue_growth_yoy", "asset_growth_yoy"]
        )

    def test_batched_search_contains_all_fundamental_subsets(self) -> None:
        candidates = fundamental_subset_candidates()
        self.assertEqual(len(candidates), 63)
        self.assertEqual(len({item["candidate_id"] for item in candidates}), 63)
        self.assertEqual(
            {len(item["features"]) for item in candidates}, {1, 2, 3, 4, 5, 6}
        )

    def test_batched_search_split_is_disjoint(self) -> None:
        search = BatchedFeatureSearchConfig()
        search.validate(total_weeks=522)
        self.assertEqual(search.confirmation_start_week, 365)
        self.assertEqual(search.selection_end_week, 418)

    def test_default_time_split_leaves_untouched_two_year_holdout(self) -> None:
        search = SearchConfig()
        search.validate(total_weeks=522)
        self.assertEqual(search.hyperparameter_warmup_weeks, 260)
        self.assertEqual(search.selection_start_week, 312)
        self.assertEqual(search.final_start_week(522), 418)
        self.assertEqual(522 - search.final_start_week(522), 104)

    def test_broad_feature_candidates_include_all_factors_and_macro(self) -> None:
        self.assertEqual(OPTIMIZATION_FEATURE_SETS["stock_37"], STOCK_37)
        self.assertEqual(
            set(OPTIMIZATION_FEATURE_SETS["stock_37_plus_macro"]),
            set(STOCK_37 + MACRO_4),
        )
        for name, features in OPTIMIZATION_FEATURE_SETS.items():
            with self.subTest(name=name):
                self.assertEqual(len(features), len(set(features)))

    def test_schedule_respects_requested_evaluation_period(self) -> None:
        rows = []
        for week_index in range(12):
            for ticker_index, ticker in enumerate(["A", "B", "C"]):
                rows.append(
                    {
                        "week_index": week_index,
                        "label_end_week": week_index + 1,
                        "date": pd.Timestamp("2020-01-03")
                        + pd.DateOffset(weeks=week_index),
                        "ticker": ticker,
                        "target": float(ticker_index - 1) / 100,
                        "revenue_growth_yoy": float(ticker_index),
                        "cashflow_to_assets": float(2 - ticker_index),
                    }
                )
        frame = pd.DataFrame(rows)
        config = ExperimentConfig(factor_set_name="fundamental_2")
        frozen = {
            "objective": "rank:pairwise",
            "relevance_bins": 3,
            "hyperparameters": {},
            "selected_candidate_id": "test",
        }

        class FakeRanker:
            def predict(self, values: np.ndarray) -> np.ndarray:
                return values[:, 0]

        with patch(
            "safecapital_xgbranker.optimization.fit_ranker",
            return_value=FakeRanker(),
        ) as fit:
            predictions, weekly, metadata, overall = evaluate_schedule(
                frame,
                config,
                frozen,
                phase="selection",
                evaluation_start_week=6,
                evaluation_end_week=10,
                training_mode="rolling",
                training_window_weeks=4,
                refit_interval_weeks=2,
            )
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(sorted(predictions["week_index"].unique()), [6, 7, 8, 9])
        self.assertEqual(len(weekly), 4)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(overall["number_of_refit_models"], 2)
        self.assertTrue((metadata["training_end_week"] + 1 < metadata["refit_week"]).all())


if __name__ == "__main__":
    unittest.main()
