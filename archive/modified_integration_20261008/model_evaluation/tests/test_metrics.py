import unittest

import numpy as np
import pandas as pd

from model_evaluation import evaluate_predictions, normalize_predictions


class MetricsTest(unittest.TestCase):
    def test_equal_weight_weeks_and_constant_cross_section(self):
        frame = pd.DataFrame({
            "date": ["2025-01-03"] * 3 + ["2025-01-10"] * 2 + ["2025-01-17"] * 2,
            "ticker": ["A", "B", "C", "A", "B", "A", "B"],
            "pred": [1., 2., 3., 2., 1., 1., 1.],
            "target": [1., 2., 3., 1., 2., 2., 3.],
        })
        summary, weekly = evaluate_predictions(frame)
        self.assertAlmostEqual(summary.iloc[0]["ic"], 0)
        self.assertAlmostEqual(summary.iloc[0]["rank_ic"], 0)
        self.assertEqual(summary.iloc[0]["valid_ic_weeks"], 2)
        self.assertTrue(np.isnan(weekly.iloc[2]["ic"]))

    def test_ranker_never_compares_dimensionless_score_to_return(self):
        frame = pd.DataFrame({"date": ["2025-01-03"] * 3, "ticker": ["A", "B", "C"],
                              "ranking_score": [0.5, 0.5, 1.], "actual_return": [0.1, 0.1, 0.2]})
        summary, _ = evaluate_predictions(frame)
        self.assertAlmostEqual(summary.iloc[0]["rank_ic"], 1)
        self.assertTrue(np.isnan(summary.iloc[0]["mae"]))
        self.assertTrue(np.isnan(summary.iloc[0]["rmse"]))

    def test_duplicate_keys_fail_but_models_can_share_keys(self):
        frame = pd.DataFrame({"date": ["2025-01-03"] * 2, "ticker": ["A"] * 2,
                              "score": [0.1, 0.2], "target": [0.1, 0.1]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_predictions(frame)
        frame["model"] = ["ridge", "xgboost"]
        self.assertEqual(len(normalize_predictions(frame)), 2)

    def test_explicit_missing_end_date_excludes_unrealized_label(self):
        frame = pd.DataFrame({"date": ["2025-01-03"] * 2 + ["2025-01-10"] * 2,
                              "ticker": ["A", "B", "A", "B"], "score": [1, 2, 2, 1],
                              "target": [1, 2, 1, 2],
                              "label_end_date": ["2025-01-10", "2025-01-10", None, None]})
        summary, weekly = evaluate_predictions(frame)
        self.assertEqual(summary.iloc[0].evaluated_rows, 2)
        self.assertAlmostEqual(summary.iloc[0].ic, 1)
        self.assertEqual(weekly.iloc[1].observations, 0)

    def test_simple_returns_are_converted_without_changing_ranking_scores(self):
        frame = pd.DataFrame({"date": ["2025-01-03"] * 2, "ticker": ["A", "B"],
                              "score": [0.1, 0.2], "target": [0.1, 0.2], "target_type": ["simple"] * 2,
                              "model": ["rf", "ranker"], "score_kind": ["return", "rank"]})
        normalized = normalize_predictions(frame).set_index("model")
        self.assertAlmostEqual(normalized.loc["rf", "score"], np.log1p(0.1))
        self.assertAlmostEqual(normalized.loc["ranker", "score"], 0.2)
        self.assertAlmostEqual(normalized.loc["ranker", "target"], np.log1p(0.2))
        self.assertTrue(normalized.target_type.eq("log").all())


if __name__ == "__main__":
    unittest.main()
