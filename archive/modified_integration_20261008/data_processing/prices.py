"""Build backtest prices from observed daily closes, without factor-validity filters."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR, DEFAULT_WEEKLY_FILE


def build_weekly_prices(daily_path: str | Path = DATA_DIR / "clean_basic_data.csv",
                        output_path: str | Path = DATA_DIR / "weekly_prices.csv",
                        calendar_path: str | Path = DEFAULT_WEEKLY_FILE) -> dict:
    """Save all observed ticker prices on the exact global dates in a weekly panel.

    Only date/ticker/adj_close are needed from the cleaned daily data. No technical
    factor, target, volume, or future-availability condition is used to retain a
    price. Missing exact-date observations remain absent: no forward/backward fill
    and no substitution of a different trading day is permitted.
    """
    daily_path = Path(daily_path).expanduser().resolve()
    calendar_path = Path(calendar_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path in (daily_path, calendar_path):
        raise ValueError("Weekly price output must not overwrite a source dataset")
    calendar = pd.read_csv(calendar_path, usecols=["date"])
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    if calendar.empty or calendar["date"].isna().any():
        raise ValueError("Weekly calendar must contain nonmissing signal dates")
    dates = pd.DatetimeIndex(calendar["date"].unique()).sort_values()
    if dates.to_period("W-FRI").duplicated().any():
        raise ValueError("Calendar must specify exactly one global signal date per week")
    daily = pd.read_csv(daily_path, usecols=["date", "ticker", "adj_close"])
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    observed = daily.loc[daily["date"].isin(dates), ["date", "ticker", "adj_close"]].copy()
    if observed.empty:
        raise ValueError("Daily data contains no observations on the weekly signal dates")
    if observed["ticker"].isna().any():
        raise ValueError("Observed daily prices contain a missing ticker")
    observed["ticker"] = observed["ticker"].astype(str)
    if observed.duplicated(["date", "ticker"]).any():
        raise ValueError("Daily data contains duplicate date/ticker prices on signal dates")
    observed["adj_close"] = pd.to_numeric(observed["adj_close"], errors="coerce")
    if not (np.isfinite(observed["adj_close"]) & observed["adj_close"].gt(0)).all():
        raise ValueError("Observed adjusted-close prices must be finite and positive")
    missing_dates = dates.difference(pd.DatetimeIndex(observed["date"].unique()))
    if len(missing_dates):
        raise ValueError(f"Daily prices are missing entire signal dates: {missing_dates.strftime('%Y-%m-%d').tolist()}")
    observed = observed.sort_values(["date", "ticker"]).reset_index(drop=True)
    counts = observed.groupby("date")["ticker"].nunique()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed.to_csv(output_path, index=False)
    return {
        "daily_input": str(daily_path), "calendar_input": str(calendar_path), "output": str(output_path),
        "daily_sha256": hashlib.sha256(daily_path.read_bytes()).hexdigest(),
        "calendar_sha256": hashlib.sha256(calendar_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": len(observed), "weeks": len(dates), "tickers": int(observed["ticker"].nunique()),
        "start": dates.min().strftime("%Y-%m-%d"), "end": dates.max().strftime("%Y-%m-%d"),
        "min_stocks_per_week": int(counts.min()), "max_stocks_per_week": int(counts.max()),
        "missing_ticker_date_pairs": int(len(dates) * observed["ticker"].nunique() - len(observed)),
        "selection": "Exact global signal-date observations from cleaned daily prices; all tickers; no factor/target filtering or price filling",
    }
