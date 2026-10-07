"""Canonical offline validation of supplied next-calendar-week log-return labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DEFAULT_WEEKLY_FILE


IDENTIFIER_AND_LABEL_COLUMNS = {
    "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
    "target", "target2", "label_end_date", "daily_return",
}


def load_weekly_panel(input_path: str | Path = DEFAULT_WEEKLY_FILE, *, verify_target: bool = True
                      ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Validate rather than overwrite supplied targets; reject unverifiable next-week rows.

    Labels require the same ticker in the following calendar week. Missing observations
    are never bridged with a later week's price. The final input week is not evaluable
    without another observed price, even if its supplied label happens to be finite.
    """
    source = Path(input_path).expanduser().resolve()
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("Weekly data contains no rows")
    required = {"date", "ticker", "adj_close", "target"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Weekly data is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["ticker"].isna().any():
        raise ValueError("date and ticker must be nonmissing")
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Weekly data contains duplicate date/ticker keys")
    if frame.assign(_week=frame["date"].dt.to_period("W-FRI")).duplicated(["ticker", "_week"]).any():
        raise ValueError("Weekly data contains multiple observations per ticker/calendar week")
    for name in ("adj_close", "target"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not (np.isfinite(frame["adj_close"]) & frame["adj_close"].gt(0)).all():
        raise ValueError("adj_close must contain finite positive values")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", sort=False)
    next_date = grouped["date"].shift(-1)
    next_price = grouped["adj_close"].shift(-1)
    consecutive = next_date.dt.to_period("W-FRI").eq(frame["date"].dt.to_period("W-FRI") + 1)
    reconstructed = np.log(next_price / frame["adj_close"])
    finite_target = pd.Series(np.isfinite(frame["target"]), index=frame.index)
    evaluable = consecutive & finite_target
    errors = (frame["target"] - reconstructed).abs()
    mismatch = evaluable & ~np.isclose(frame["target"], reconstructed, rtol=1e-7, atol=1e-8)
    if verify_target and mismatch.any():
        examples = frame.loc[mismatch, ["date", "ticker", "target"]].head(3).to_dict("records")
        raise ValueError(f"{int(mismatch.sum())} labels do not match next-week adjusted-close log returns: {examples}")
    frame["label_end_date"] = next_date.where(consecutive)
    reasons = pd.Series("", index=frame.index)
    reasons.loc[~finite_target] = "nonfinite_target"
    reasons.loc[next_date.notna() & ~consecutive] = "missing_next_calendar_week"
    reasons.loc[next_date.isna()] = "no_observed_next_price"
    excluded = frame.loc[~evaluable, ["date", "ticker", "target"]].copy()
    excluded["reason"] = reasons.loc[~evaluable]
    clean = frame.loc[evaluable].sort_values(["date", "ticker"]).reset_index(drop=True)
    factors = [name for name in frame.columns if name not in IDENTIFIER_AND_LABEL_COLUMNS]
    numeric = frame[factors].apply(pd.to_numeric, errors="coerce")
    report = {
        "input": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_rows": len(frame), "valid_label_rows": len(clean), "excluded_rows": len(excluded),
        "tickers": int(frame["ticker"].nunique()), "weeks": int(frame["date"].nunique()),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
        "factor_count": len(factors), "factor_names": factors,
        "nonfinite_factor_cells": int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()),
        "target_mismatches": int(mismatch.sum()),
        "max_absolute_target_error": float(errors.loc[evaluable].max()) if evaluable.any() else None,
        "excluded_by_reason": {str(k): int(v) for k, v in excluded["reason"].value_counts().items()},
        "target_definition": "next calendar week's final adjusted close / current week's final adjusted close, log return",
    }
    return clean, report, excluded


def audit_weekly_data(input_path: str | Path = DEFAULT_WEEKLY_FILE,
                      output_dir: str | Path | None = None) -> dict:
    """Audit the existing CSV offline, optionally saving JSON and excluded-row evidence."""
    _, report, excluded = load_weekly_panel(input_path)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        excluded.to_csv(destination / "excluded_rows.csv", index=False)
    return report
