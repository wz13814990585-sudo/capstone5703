"""Small tests for leakage boundaries and contracts introduced by integration."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from model_training.artifacts import save_predictions, standard_predictions
from model_training.preprocessing import apply_preprocessor, fit_preprocessor
from ridge import train as ridge
from random_forest import train as forest
from xgboost_model import train as xgboost


class TrainingContractTests(unittest.TestCase):
    def test_preprocessing_uses_training_statistics_only(self):
        train = pd.DataFrame({"factor": [1., 2., 3., np.nan]})
        held = pd.DataFrame({"factor": [np.nan, 1e9]})
        _, params = fit_preprocessor(train, ["factor"])
        result = apply_preprocessor(held, ["factor"], params)
        np.testing.assert_allclose(result[:, 0], [2., 2.97])
        self.assertEqual(params.median[0], 2.)

    def test_all_missing_handling_is_model_specific(self):
        frame = pd.DataFrame({"factor": [np.nan, np.nan]})
        with self.assertRaises(ValueError):
            fit_preprocessor(frame, ["factor"])
        x, _ = fit_preprocessor(frame, ["factor"], missing_fill=0.)
        np.testing.assert_array_equal(x, np.zeros((2, 1)))

    def test_ridge_unpenalized_intercept_and_independent_solver(self):
        x = np.array([[1., 8.], [2., 3.], [4., 2.], [7., 6.]])
        y = np.array([3., 4., 9., 12.])
        coef, intercept = ridge.core.fit_ridge(x, y, 10.)
        np.testing.assert_allclose(coef, ridge.core.fit_ridge_lstsq_check(x, y, 10.))
        self.assertAlmostEqual(float((x @ coef + intercept).mean()), float(y.mean()))

    def test_original_label_boundary_difference_is_preserved(self):
        cutoff = pd.Timestamp("2024-01-15")
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "label_end_date": pd.to_datetime(["2024-01-08", "2024-01-15"]),
            "target": [0.1, 0.2]})
        strict = ridge.training_rows(frame, {"fit_date": cutoff}, "rolling")
        same_day = forest.training_rows(frame, pd.Timestamp("2023-01-01"), cutoff, "target")
        self.assertEqual(len(strict), 1)
        self.assertEqual(len(same_day), 2)

    def test_selection_gate_does_not_invent_ridge_fallback(self):
        scores = pd.DataFrame({"alpha": [1., 10.], "Rank IC": [-.1, .1],
                               "Rank ICIR": [.1, .01], "Positive Rank IC": [.6, .6],
                               "Top5 Excess": [.5, .6]})
        _, chosen = ridge.choose(scores)
        self.assertIsNone(chosen)

    def test_xgboost_candidate_only_fits_requested_stage(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="7D")
        panel = pd.DataFrame({"date": dates, "ticker": ["A"] * 5,
                              "label_end_date": pd.date_range("2024-01-08", periods=5, freq="7D"),
                              "_model_target": [0.1] * 5})
        boundaries = dict(zip(["development", "validation", "final_test"], dates[2:]))
        calls = []
        def fake_predict(train, predict, features, candidate):
            calls.append((train.copy(), predict.copy()))
            return predict[["date", "ticker", "label_end_date"]].assign(y_true=.1, y_pred=.1)
        with patch.object(xgboost, "fit_predict", side_effect=fake_predict):
            result = xgboost.run_fixed(panel, [], {}, boundaries, stages={"development"})
        self.assertEqual(len(calls), 1)
        self.assertTrue((calls[0][0].label_end_date < dates[2]).all())
        self.assertEqual(result.stage.unique().tolist(), ["development"])

    def test_standard_prediction_identity_distinguishes_experiments(self):
        source = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["A"],
                               "label_end_date": [pd.Timestamp("2024-01-08")],
                               "actual": [.03], "prediction": [.02], "stage": ["final_test"]})
        fixed = standard_predictions(source, model="random_forest", mode="fixed", feature_set="7")
        rolling = standard_predictions(source, model="random_forest", mode="rolling_3y", feature_set="7")
        self.assertNotEqual(fixed.model.iloc[0], rolling.model.iloc[0])
        self.assertEqual(fixed.score.iloc[0], .02)
        self.assertEqual(fixed.target.iloc[0], .03)
        self.assertEqual(fixed.score_kind.iloc[0], "return")

    def test_prediction_csv_retains_quick_flag_and_empty_schema(self):
        source = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["A"],
                               "label_end_date": [pd.Timestamp("2024-01-08")],
                               "target": [.03], "score": [.02], "stage": ["final_test"]})
        prediction = standard_predictions(source, model="ridge", mode="fixed", feature_set="7")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            for quick_run in (True, False):
                with self.subTest(quick_run=quick_run):
                    save_predictions(path, [prediction], quick_run=quick_run)
                    restored = pd.read_csv(path)
                    self.assertEqual(restored.quick_run.tolist(), [quick_run])
                    self.assertEqual(restored.score.tolist(), [.02])
            save_predictions(path, [], quick_run=True)
            restored = pd.read_csv(path)
            self.assertTrue(restored.empty)
            self.assertIn("quick_run", restored.columns)


if __name__ == "__main__":
    unittest.main()
