"""Regression checks for temporal labels and strict validation, using only unittest."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import load_weekly_panel
from data_processing.build_technical_factors import add_next_week_close_to_close_target


class WeeklyPanelTest(unittest.TestCase):
    def test_missing_calendar_week_is_not_bridged(self):
        frame = pd.DataFrame({"date": ["2024-01-05", "2024-01-12", "2024-01-26", "2024-02-02"],
                              "ticker": ["A"] * 4, "adj_close": [100., 110., 121., 133.1],
                              "target": [np.log(1.1)] * 4, "factor": [1., 2., 3., 4.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            frame.to_csv(path, index=False)
            panel, audit, excluded = load_weekly_panel(path)
            self.assertEqual(panel["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-05", "2024-01-26"])
            self.assertEqual(audit["excluded_by_reason"], {"missing_next_calendar_week": 1, "no_observed_next_price": 1})
            self.assertEqual(len(excluded), 2)
        daily = frame.drop(columns="target").assign(date=pd.to_datetime(frame["date"]))
        result = add_next_week_close_to_close_target(daily)
        self.assertAlmostEqual(result.loc[0, "target"], np.log(1.1))
        self.assertTrue(pd.isna(result.loc[1, "target"]))
        self.assertTrue(pd.isna(result.loc[3, "target"]))

    def test_bad_existing_label_is_rejected(self):
        frame = pd.DataFrame({"date": ["2024-01-05", "2024-01-12"], "ticker": ["A", "A"],
                              "adj_close": [100., 110.], "target": [0., 0.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_weekly_panel(path)


if __name__ == "__main__":
    unittest.main()
