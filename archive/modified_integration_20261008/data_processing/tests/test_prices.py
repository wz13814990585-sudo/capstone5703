"""Regression tests for the unfiltered, exact-date price input used by backtests."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_processing import build_weekly_prices


class WeeklyPricesTest(unittest.TestCase):
    def test_all_stocks_and_no_cross_date_fill(self):
        # B is absent from the factor calendar, and its Thursday quote must not
        # be substituted for a missing quote on the second signal Friday.
        daily = pd.DataFrame({"date": ["2024-01-05", "2024-01-05", "2024-01-11", "2024-01-12"],
                              "ticker": ["A", "B", "B", "A"], "adj_close": [100., 5., 7., 110.]})
        calendar = pd.DataFrame({"date": ["2024-01-05", "2024-01-12"], "ticker": ["A", "A"]})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            daily_file, calendar_file, output = directory / "daily.csv", directory / "calendar.csv", directory / "prices.csv"
            daily.to_csv(daily_file, index=False)
            calendar.to_csv(calendar_file, index=False)
            original = daily_file.read_bytes()
            result = build_weekly_prices(daily_file, output, calendar_file)
            prices = pd.read_csv(output)
            self.assertEqual(result["tickers"], 2)
            self.assertEqual(result["missing_ticker_date_pairs"], 1)
            self.assertEqual(prices.loc[prices.ticker.eq("B"), "adj_close"].tolist(), [5.])
            self.assertEqual(prices.loc[prices.ticker.eq("B"), "date"].tolist(), ["2024-01-05"])
            self.assertEqual(daily_file.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                build_weekly_prices(daily_file, daily_file, calendar_file)


if __name__ == "__main__":
    unittest.main()
