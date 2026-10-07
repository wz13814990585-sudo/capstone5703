import argparse
from pathlib import Path
import tempfile
import unittest

from integration.runner import ROOT, child_command, sha, stage_original, verify_originals


class IntegrationOnlyTests(unittest.TestCase):
    def test_original_sources_and_data_match_frozen_provenance(self):
        verification = verify_originals()
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["original_python_files"], 11)

    def test_staging_does_not_modify_original_script(self):
        source = ROOT / "ridge/train_ridge_comparison.py"
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            staged = stage_original("ridge", Path(temporary))
            self.assertEqual(staged.read_bytes(), original)
            self.assertEqual(sha(staged), sha(source))
        self.assertEqual(source.read_bytes(), original)

    def test_original_rf_receives_original_options_without_grid_changes(self):
        args = argparse.Namespace(python="python", command="run", input=Path("original.csv"),
                                  screening_file=Path("original.xlsx"), mode="both", xgboost_feature_set="all")
        command = child_command("random_forest", Path("random_forest.py"), args, Path("result"))
        self.assertEqual(command, ["python", "-u", "random_forest.py", "--data-file", "original.csv", "--screening-file", "original.xlsx"])

    def test_original_xgboost_check_does_not_train_or_override_candidates(self):
        args = argparse.Namespace(python="python", command="check", input=Path("original.csv"),
                                  screening_file=Path("original.xlsx"), mode="fixed", xgboost_feature_set="all")
        command = child_command("xgboost", Path("train_xgboost_comparison(2).py"), args, Path("result"))
        self.assertIn("--check-only", command)
        self.assertNotIn("--quick", command)
        self.assertNotIn("--output", command)


if __name__ == "__main__":
    unittest.main()
