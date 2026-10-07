import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from run import main


class EntryPointTests(unittest.TestCase):
    def test_failure_records_failed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "failed"
            with self.assertRaises(FileNotFoundError):
                main(["audit", "--input", str(Path(temporary) / "missing.csv"), "--output", str(out)])
            self.assertEqual(json.loads((out / "run_manifest.json").read_text())["status"], "failed")

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            marker = out / "predictions.csv"
            marker.write_text("existing result")
            with self.assertRaises(SystemExit):
                main(["audit", "--output", str(out)])
            self.assertEqual(marker.read_text(), "existing result")

    def test_no_selected_models_is_a_successful_empty_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "empty_run"
            with patch("data_processing.audit_weekly_data", return_value={}), patch("run.train_models", return_value=pd.DataFrame()):
                main(["train", "--output", str(out)])
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["model_selection"], "no_selected_models")

    def test_evaluation_inherits_quick_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "predictions.csv"
            pd.DataFrame({"date": ["2024-01-05"] * 2, "ticker": ["A", "B"], "model": ["ridge"] * 2,
                          "score": [.01, .02], "target": [.02, .01], "label_end_date": ["2024-01-12"] * 2,
                          "score_kind": ["return"] * 2, "quick_run": [True] * 2}).to_csv(source, index=False)
            out = Path(temporary) / "evaluation"
            main(["evaluate", "--predictions", str(source), "--output", str(out)])
            self.assertTrue(json.loads((out / "run_manifest.json").read_text())["quick"])
            self.assertTrue(pd.read_csv(out / "model_evaluation/summary.csv").quick_run.all())

    def test_ranker_rejects_existing_output_before_training(self):
        from xgboost_ranker.train import run
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            marker = out / "predictions.csv"
            marker.write_text("existing result")
            with self.assertRaises(ValueError):
                run(out / "missing.csv", out, out / "models", 1)
            self.assertEqual(marker.read_text(), "existing result")

    def test_legacy_single_model_file_and_unknown_quick_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "legacy.csv"
            pd.DataFrame({"date": ["2024-01-05"] * 2, "ticker": ["A", "B"],
                          "prediction": [.01, .02], "target": [.02, .01]}).to_csv(source, index=False)
            out = Path(temporary) / "evaluation"
            main(["evaluate", "--predictions", str(source), "--output", str(out)])
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["selected_models"], ["model"])
            self.assertIsNone(manifest["quick"])
            summary = pd.read_csv(out / "model_evaluation/summary.csv")
            self.assertTrue(summary.quick_run.isna().all())


if __name__ == "__main__":
    unittest.main()
