"""Verify held-out labels cannot change selected-factor statistics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from single_factor_analysis.screening import ScreeningRules, run_screening


class ScreeningTest(unittest.TestCase):
    def test_holdout_prices_cannot_change_selection(self):
        dates = pd.date_range("2023-01-06", periods=10, freq="W-FRI")
        records = []
        for ticker in range(6):
            for index, date in enumerate(dates):
                records.append({"date": date, "ticker": f"T{ticker}",
                                "adj_close": 100. * np.exp(index * (ticker + 1) / 1000),
                                "target": (ticker + 1) / 1000,
                                "factor": float(ticker), "duplicate": float(ticker)})
        data = pd.DataFrame(records)
        cutoff = dates[6]
        rules = ScreeningRules(minimum_stocks=5, minimum_distinct_factor_values=5,
                               minimum_weeks=3, minimum_years=1, minimum_weeks_per_year=3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.csv"
            data.to_csv(source, index=False)
            run_screening(source, directory / "first", str(cutoff.date()), rules=rules)
            # Perturb only prices after the allowed selection window and rederive
            # labels so the data audit remains valid; the pre-cutoff sample is fixed.
            later = data["date"].ge(cutoff)
            data.loc[later, "adj_close"] *= 1.4
            data["target"] = data.groupby("ticker")["adj_close"].transform(lambda x: np.log(x.shift(-1) / x))
            data.loc[later, "factor"] *= -1
            data.to_csv(source, index=False)
            run_screening(source, directory / "second", str(cutoff.date()), rules=rules)
            first = pd.read_csv(directory / "first" / "screening_results.csv")
            second = pd.read_csv(directory / "second" / "screening_results.csv")
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.loc[first["factor"].eq("duplicate"), "decision"].iloc[0], "duplicate")


if __name__ == "__main__":
    unittest.main()
