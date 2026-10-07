"""Run maintained tests, excluding intentionally archived historical experiments."""
import unittest

MODULES = [
    "data_processing.tests.test_panel",
    "single_factor_analysis.tests.test_screening",
    "model_training.test_contracts",
    "model_evaluation.tests.test_metrics",
    "backtesting.tests.test_engine",
    "tests.test_entrypoint",
]

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
