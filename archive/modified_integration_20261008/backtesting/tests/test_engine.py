import unittest

import numpy as np
import pandas as pd

from backtesting import STRATEGIES, capped_weights, run_backtest
from backtesting.portfolio_strategies import inputs


def predictions_for(prices, dates=None):
    dates = prices.index[:-1] if dates is None else dates
    return pd.DataFrame([{"date": date, "ticker": ticker, "score": float(len(prices.columns) - i),
                          "score_kind": "rank"}
                         for date in dates for i, ticker in enumerate(prices.columns)])


class BacktestTest(unittest.TestCase):
    def test_cash_shortfall_and_cost_are_hand_calculated(self):
        dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 110., 110.]}, index=dates)
        result = run_backtest(predictions_for(prices), prices, strategies=["Top5"], costs_bps=[10])
        first, second = result.weekly.iloc[0], result.weekly.iloc[1]
        self.assertAlmostEqual(first.gross_return, 0.02)
        self.assertAlmostEqual(first.cash_weight, 0.8)
        self.assertAlmostEqual(first.traded_weight, 0.2)
        self.assertAlmostEqual(first.net_return, (1 - 0.2 * 0.001) * 1.02 - 1)
        self.assertAlmostEqual(second.traded_weight, 0.22 / 1.02 - 0.2)

    def test_rebalance_drift_and_gap_restart(self):
        dates = pd.date_range("2025-01-03", periods=6, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 110., 121., 121., 121., 121.]}, index=dates)
        predictions = predictions_for(prices, dates[[0, 1, 3, 4]])
        result = run_backtest(predictions, prices, strategies=["Top5_Rebalance4w"], costs_bps=[0])
        weeks = result.weekly
        self.assertEqual(len(weeks), 5)
        self.assertAlmostEqual(weeks.iloc[1].traded_weight, 0)
        self.assertAlmostEqual(weeks.iloc[1].cash_weight, 0.8 / 1.02)
        self.assertTrue(weeks.iloc[2].no_signal_cash_week)
        self.assertAlmostEqual(weeks.iloc[2].cash_weight, 1)
        self.assertGreater(weeks.iloc[2].traded_weight, 0)
        self.assertTrue(weeks.iloc[3].rebalance_event)
        self.assertAlmostEqual(weeks.iloc[3].traded_weight, 0.2)

    def test_window_starts_with_cash_and_new_rebalance_clock(self):
        dates = pd.date_range("2025-01-03", periods=5, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 110., 121., 121., 121.]}, index=dates)
        result = run_backtest(predictions_for(prices), prices, strategies=["Top5_Rebalance4w"], costs_bps=[0],
                              windows={"full": (None, None), "last": (dates[1], None)})
        restart = result.weekly.loc[result.weekly.window.eq("last")].iloc[0]
        self.assertTrue(restart.rebalance_event)
        self.assertAlmostEqual(restart.traded_weight, 0.2)
        self.assertAlmostEqual(restart.cash_weight, 0.8)

    def test_selected_missing_future_price_raises_without_survivor_filter(self):
        dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., np.nan, 100.], "B": [100., 100., 100.]}, index=dates)
        prediction = predictions_for(prices, dates[:1])
        with self.assertRaisesRegex(ValueError, "Cannot value selected holdings"):
            run_backtest(prediction, prices, strategies=["Top5"], costs_bps=[0])

    def test_future_changes_do_not_change_current_weights_or_risk_inputs(self):
        dates = pd.date_range("2024-01-05", periods=33, freq="W-FRI")
        changes = np.sin(np.arange(33)[:, None] + np.arange(7)[None, :]) * 0.02 + 0.002
        prices = pd.DataFrame(100 * np.cumprod(1 + changes, axis=0), index=dates, columns=list("ABCDEFG"))
        changed = prices.copy()
        changed.loc[dates[31]:] *= np.arange(1, 8)
        for name in ("vol", "trend", "lag4", "lag13"):
            before_input, after_input = inputs(prices)[name].loc[:dates[30]], inputs(changed)[name].loc[:dates[30]]
            if isinstance(before_input, pd.Series):
                pd.testing.assert_series_equal(before_input, after_input)
            else:
                pd.testing.assert_frame_equal(before_input, after_input)
        prediction = predictions_for(prices, dates[30:31])
        before = run_backtest(prediction, prices, costs_bps=[0])
        after = run_backtest(prediction, changed, costs_bps=[0])
        pd.testing.assert_frame_equal(before.positions.drop(columns="realized_return"), after.positions.drop(columns="realized_return"))

    def test_terminal_prediction_is_audited_not_return(self):
        dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 110., 121.]}, index=dates)
        result = run_backtest(predictions_for(prices, dates), prices, strategies=["Top5"], costs_bps=[0])
        self.assertEqual(len(result.weekly), 2)
        self.assertEqual(len(result.audit), 1)
        self.assertEqual(result.audit.iloc[0].reason, "terminal_next_week_unavailable")
        terminal_only = run_backtest(predictions_for(prices, dates[-1:]), prices, strategies=["Top5"], costs_bps=[0])
        self.assertTrue(terminal_only.weekly.empty)
        self.assertIn("net_return", terminal_only.weekly)
        self.assertEqual(len(terminal_only.audit), 1)

    def test_wrong_labels_are_rejected_and_all_strategies_exist(self):
        dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 110., 121.]}, index=dates)
        prediction = predictions_for(prices)
        prediction["target"] = 0
        with self.assertRaisesRegex(ValueError, "Targets disagree"):
            run_backtest(prediction, prices)
        self.assertEqual(len(STRATEGIES), 15)
        self.assertAlmostEqual(sum(capped_weights({"A": 100., "B": 1.}, budget=1).values()), 0.7)

    def test_drawdown_includes_initial_nav_and_delay_keeps_calendar(self):
        dates = pd.date_range("2025-01-03", periods=4, freq="W-FRI")
        prices = pd.DataFrame({"A": [100., 50., 50., 50.]}, index=dates)
        predictions = predictions_for(prices)
        result = run_backtest(predictions, prices, strategies=["Top5"], costs_bps=[0])
        self.assertAlmostEqual(result.summary.iloc[0].max_drawdown, -0.1)
        delayed = run_backtest(predictions, prices, strategies=["Top5"], costs_bps=[0], initial_delay_weeks=1)
        self.assertEqual(len(delayed.weekly), 3)
        self.assertAlmostEqual(delayed.summary.iloc[0].total_return, 0)


if __name__ == "__main__":
    unittest.main()
