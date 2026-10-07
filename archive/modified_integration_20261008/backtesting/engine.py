"""Price-verified weekly portfolio simulation with explicit missing-data handling."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from model_evaluation import normalize_predictions
from .portfolio_strategies import RULES, STRATEGIES, choose_weights, inputs


@dataclass
class BacktestResult:
    summary: pd.DataFrame
    weekly: pd.DataFrame
    positions: pd.DataFrame
    audit: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("summary", "weekly", "positions", "audit"):
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)
        (directory / "strategy_rules.json").write_text(
            json.dumps({"strategies": RULES, "frequency": "weekly", "periods_per_year": 52,
                        "target": "next global week adjusted-close log return",
                        "cost": "(1 - traded_weight * bps / 10000) * (1 + gross_return) - 1",
                        "benchmark": "equal-weight gross return over valid adjacent price pairs",
                        "missing_selected_next_price": "error, never filter using future availability",
                        "cash_interest": 0, "final_liquidation_cost": False},
                       ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_prices(prices: pd.DataFrame, price_column: str = "adj_close") -> pd.DataFrame:
    """Accept date/ticker/adj_close rows or a DatetimeIndex × ticker matrix."""
    if {"date", "ticker", price_column}.issubset(prices.columns):
        rows = prices[["date", "ticker", price_column]].copy()
        rows["date"] = pd.to_datetime(rows["date"], errors="raise")
        if rows[["date", "ticker"]].isna().any().any():
            raise ValueError("Price dates and tickers must not be missing")
        rows["ticker"] = rows["ticker"].astype(str)
        if rows.duplicated(["date", "ticker"]).any():
            raise ValueError("Duplicate date/ticker prices")
        matrix = rows.pivot(index="date", columns="ticker", values=price_column)
    elif isinstance(prices.index, pd.DatetimeIndex):
        matrix = prices.copy()
        if matrix.index.has_duplicates or matrix.columns.has_duplicates:
            raise ValueError("Duplicate dates/tickers in price matrix")
        matrix.columns = matrix.columns.astype(str)
    else:
        raise ValueError(f"Prices require date, ticker, {price_column} columns or a DatetimeIndex")
    matrix = matrix.apply(pd.to_numeric, errors="raise").sort_index()
    values = matrix.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError("Observed adjusted-close prices must be finite and positive")
    if len(matrix) < 2:
        raise ValueError("At least two price dates are required")
    return matrix


def performance(weekly: pd.DataFrame, periods_per_year: int = 52) -> dict[str, float | int]:
    returns = weekly["net_return"].to_numpy(dtype=float)
    if not len(returns) or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Performance requires finite, nonempty weekly returns above -100%")
    nav = np.r_[1.0, np.cumprod(1 + returns)]
    benchmark = weekly["benchmark_return"].to_numpy(dtype=float)
    benchmark_growth = float(np.prod(1 + benchmark))
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan")
    annualized = float(nav[-1] ** (periods_per_year / len(returns)) - 1)
    benchmark_annualized = float(benchmark_growth ** (periods_per_year / len(returns)) - 1)
    return {
        "weeks": len(returns), "total_return": float(nav[-1] - 1),
        "annualized_return": annualized,
        "annualized_volatility": volatility * np.sqrt(periods_per_year),
        "sharpe": float(returns.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 0 else float("nan"),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "positive_week_ratio": float((returns > 0).mean()),
        "benchmark_total_return": benchmark_growth - 1,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_annualized_return": annualized - benchmark_annualized,
        "mean_traded_weight": float(weekly["traded_weight"].mean()),
        "mean_cash_weight": float(weekly["cash_weight"].mean()),
        "no_signal_cash_weeks": int(weekly["no_signal_cash_week"].sum()),
    }


def _validate_predictions(frame: pd.DataFrame, prices: pd.DataFrame, label_tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = list(prices.index)
    next_dates = dict(zip(dates[:-1], dates[1:]))
    lookup = prices.stack()
    keys = pd.MultiIndex.from_frame(frame[["date", "ticker"]])
    current = lookup.reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(current).all():
        bad = frame.loc[~np.isfinite(current), ["model", "date", "ticker"]].head(3)
        raise ValueError(f"Predictions lack a signal-date price: {bad.to_dict('records')}")
    expected_end = frame["date"].map(next_dates)
    specified = frame["label_end_date"].notna() & expected_end.notna()
    if frame.loc[specified, "label_end_date"].ne(expected_end.loc[specified]).any():
        raise ValueError("label_end_date must equal the next global price-calendar date")
    future_keys = pd.MultiIndex.from_arrays([expected_end, frame["ticker"]])
    future = lookup.reindex(future_keys).to_numpy(dtype=float)
    verified = np.isfinite(future) & frame["target"].notna().to_numpy()
    errors = np.abs(np.log(future[verified] / current[verified]) - frame.loc[verified, "target"].to_numpy())
    if len(errors) and np.max(errors) > label_tolerance:
        raise ValueError(f"Targets disagree with next global week price log returns (max error={np.max(errors):.6g})")
    audit_rows = []
    for mask, reason in ((expected_end.isna().to_numpy(), "terminal_next_week_unavailable"),
                         (expected_end.notna().to_numpy() & ~np.isfinite(future), "next_stock_price_missing_not_filtered")):
        for row in frame.loc[mask, ["model", "date", "ticker"]].to_dict("records"):
            audit_rows.append({**row, "reason": reason})
    frame = frame.copy()
    frame["label_end_date"] = expected_end
    # Missing targets can be recovered from prices; supplied targets remain checked above.
    frame["target"] = frame["target"].fillna(pd.Series(np.log(future / current), index=frame.index))
    audit = pd.DataFrame(audit_rows, columns=["model", "date", "ticker", "reason"])
    return frame, audit


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    strategies: Sequence[str] | None = None,
    costs_bps: Sequence[float] = (0, 10, 25),
    windows: Mapping[str, tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]] | None = None,
    score_kind: str | None = None,
    price_column: str = "adj_close",
    label_tolerance: float = 1e-8,
    initial_delay_weeks: int = 0,
) -> BacktestResult:
    """Backtest all models, with each inclusive date window starting from cash.

    Targets never determine portfolio membership or portfolio returns. Selection
    uses current/past data; P&L uses observed adjacent adjusted-close prices.
    A selected holding with no next price raises instead of using future price
    availability to improve selection. Terminal signals are audited and omitted.
    Signal gaps inside each model's history count as calendar weeks in cash.

    Costs are one-way basis points on the sum of absolute buy/sell weight changes.
    Cash earns zero; no forced terminal liquidation is charged. ``initial_delay_weeks``
    keeps the first N calendar weeks in cash, for rebalance phase sensitivity.
    """
    chosen = list(STRATEGIES if strategies is None else strategies)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(STRATEGIES):
        raise ValueError(f"Choose unique strategy names from {STRATEGIES}")
    costs = [float(cost) for cost in costs_bps]
    if not costs or len(set(costs)) != len(costs) or any(not np.isfinite(c) or c < 0 or c >= 5000 for c in costs):
        raise ValueError("costs_bps must be unique finite numbers in [0,5000)")
    if not isinstance(initial_delay_weeks, int) or initial_delay_weeks < 0:
        raise ValueError("initial_delay_weeks must be a nonnegative integer")
    matrix = prepare_prices(prices, price_column)
    frame = normalize_predictions(predictions, score_kind=score_kind, require_target=False)
    frame, audit = _validate_predictions(frame, matrix, label_tolerance)
    context = inputs(matrix)
    stock_returns = matrix.shift(-1) / matrix - 1
    benchmark = stock_returns.mean(axis=1)
    next_dates = dict(zip(matrix.index[:-1], matrix.index[1:]))
    windows = {"FullHistory": (None, None)} if windows is None else windows
    if not windows:
        raise ValueError("At least one window is required")
    all_weekly, all_positions, summaries = [], [], []
    gap_rows = []
    for model, model_frame in frame.groupby("model", sort=True):
        available = model_frame.loc[model_frame["label_end_date"].notna()]
        if available.empty:
            continue
        first, last = model_frame["date"].min(), model_frame["date"].max()
        model_calendar = matrix.index[(matrix.index >= first) & (matrix.index <= last) & matrix.index.isin(next_dates)]
        grouped = {date: group for date, group in available.groupby("date", sort=True)}
        for date in model_calendar:
            if date not in grouped:
                gap_rows.append({"model": model, "date": date, "ticker": "", "reason": "no_signal_cash_week"})
        for window, (start, end) in windows.items():
            calendar = model_calendar
            if start is not None:
                calendar = calendar[calendar >= pd.Timestamp(start)]
            if end is not None:
                calendar = calendar[calendar <= pd.Timestamp(end)]
            if len(calendar) == 0:
                continue
            for strategy in chosen:
                previous, prior_returns, prior_cash = {}, {}, 1.0
                last_rebalance = None
                rows = []
                for week_index, date in enumerate(calendar):
                    growth = prior_cash + sum(weight * (1 + prior_returns[ticker]) for ticker, weight in previous.items())
                    drift = {ticker: weight * (1 + prior_returns[ticker]) / growth for ticker, weight in previous.items()}
                    group = grouped.get(date, available.iloc[:0])
                    delayed = week_index < initial_delay_weeks
                    selection_group = available.iloc[:0] if delayed else group
                    weights, rebalanced, last_rebalance = choose_weights(
                        strategy, selection_group, date, context, previous, drift, week_index, last_rebalance)
                    realized = {ticker: float(stock_returns.loc[date, ticker]) for ticker in weights}
                    missing = [ticker for ticker, value in realized.items() if not np.isfinite(value)]
                    if missing:
                        raise ValueError(f"Cannot value selected holdings at next week: model={model}, strategy={strategy}, date={date.date()}, tickers={missing}. Supply complete weekly prices; no future-price filtering is applied.")
                    cash_weight = max(0.0, 1 - sum(weights.values()))
                    if sum(weights.values()) > 1 + 1e-12 or any(weight < 0 for weight in weights.values()):
                        raise ValueError("Invalid portfolio allocation")
                    gross = sum(weights[ticker] * realized[ticker] for ticker in weights)
                    traded = sum(abs(weights.get(ticker, 0) - drift.get(ticker, 0)) for ticker in weights.keys() | drift.keys())
                    market_return = float(benchmark.loc[date])
                    if not np.isfinite(market_return):
                        raise ValueError(f"No price pair available for benchmark at {date}")
                    rows.append({"model": model, "window": window, "strategy": strategy, "date": date,
                                 "label_end_date": next_dates[date], "gross_return": gross,
                                 "benchmark_return": market_return, "traded_weight": traded,
                                 "holdings": len(weights), "cash_weight": cash_weight,
                                 "no_signal_cash_week": group.empty, "initial_delay_cash_week": delayed,
                                 "rebalance_event": rebalanced})
                    all_positions.extend({"model": model, "window": window, "strategy": strategy,
                                          "date": date, "label_end_date": next_dates[date], "ticker": ticker,
                                          "weight": weight, "realized_return": realized[ticker]}
                                         for ticker, weight in weights.items())
                    previous, prior_returns, prior_cash = weights, realized, cash_weight
                base = pd.DataFrame(rows)
                for cost in costs:
                    weekly = base.copy()
                    weekly["cost_bps"] = cost
                    weekly["transaction_cost_fraction"] = weekly["traded_weight"] * cost / 10000
                    weekly["net_return"] = (1 - weekly["transaction_cost_fraction"]) * (1 + weekly["gross_return"]) - 1
                    weekly["nav"] = (1 + weekly["net_return"]).cumprod()
                    weekly["benchmark_nav"] = (1 + weekly["benchmark_return"]).cumprod()
                    weekly["drawdown"] = weekly["nav"] / weekly["nav"].cummax().clip(lower=1) - 1
                    summaries.append({"model": model, "window": window, "strategy": strategy,
                                      "cost_bps": cost, "start": calendar[0], "end": next_dates[calendar[-1]],
                                      **performance(weekly)})
                    all_weekly.append(weekly)
    if gap_rows:
        audit = pd.concat([audit, pd.DataFrame(gap_rows)], ignore_index=True) if not audit.empty else pd.DataFrame(gap_rows)
    position_columns = ["model", "window", "strategy", "date", "label_end_date", "ticker", "weight", "realized_return"]
    summary_columns = ["model", "window", "strategy", "cost_bps", "start", "end", "weeks", "total_return",
                       "annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "positive_week_ratio",
                       "benchmark_total_return", "benchmark_annualized_return", "excess_annualized_return",
                       "mean_traded_weight", "mean_cash_weight", "no_signal_cash_weeks"]
    weekly_columns = ["model", "window", "strategy", "date", "label_end_date", "gross_return", "benchmark_return",
                      "traded_weight", "holdings", "cash_weight", "no_signal_cash_week", "initial_delay_cash_week",
                      "rebalance_event", "cost_bps", "transaction_cost_fraction", "net_return", "nav", "benchmark_nav", "drawdown"]
    return BacktestResult(pd.DataFrame(summaries, columns=summary_columns),
                          pd.concat(all_weekly, ignore_index=True) if all_weekly else pd.DataFrame(columns=weekly_columns),
                          pd.DataFrame(all_positions, columns=position_columns), audit)
