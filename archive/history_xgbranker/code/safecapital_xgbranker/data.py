from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXACT_FORBIDDEN_FEATURES = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "target",
    "target2",
    "y_true",
    "actual_return",
    "actual_rank",
    "ranking_score",
    "predicted_rank",
}
FORBIDDEN_PREFIX = re.compile(r"^(future|fwd|forward|label)(_|$)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_feature_safety(features: list[str]) -> None:
    duplicates = sorted({name for name in features if features.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate features: {duplicates}")
    unsafe = []
    for feature in features:
        lower = feature.lower()
        if lower in EXACT_FORBIDDEN_FEATURES or FORBIDDEN_PREFIX.search(lower):
            unsafe.append(feature)
        elif "future" in lower or lower.startswith("target"):
            unsafe.append(feature)
    if unsafe:
        raise ValueError(f"Leakage/metadata columns are forbidden in X: {unsafe}")


def _target_definition_text(path: Path) -> str:
    if not path.exists():
        return ""
    definitions = pd.read_csv(path, encoding="utf-8-sig")
    if not {"factor", "calculation"}.issubset(definitions.columns):
        return ""
    target_rows = definitions.loc[definitions["factor"].eq("target"), "calculation"]
    return "" if target_rows.empty else str(target_rows.iloc[0])


def _verify_target_alignment(df: pd.DataFrame, tolerance: float = 1e-8) -> dict[str, Any]:
    """Verify rows for which the same ticker is present in the immediately next week."""
    by_ticker = df.sort_values(["ticker", "week_index"])
    next_week = by_ticker.groupby("ticker", sort=False)["week_index"].shift(-1)
    next_adj = by_ticker.groupby("ticker", sort=False)["adj_close"].shift(-1)
    consecutive = next_week.eq(by_ticker["week_index"] + 1) & next_adj.notna()
    expected = np.log(next_adj[consecutive] / by_ticker.loc[consecutive, "adj_close"])
    observed = by_ticker.loc[consecutive, "target"]
    differences = (observed - expected).abs()
    max_abs = float(differences.max()) if len(differences) else float("nan")
    mismatches = int((differences > tolerance).sum())
    if mismatches:
        examples = by_ticker.loc[
            consecutive & (by_ticker["target"].sub(np.log(next_adj / by_ticker["adj_close"])).abs() > tolerance),
            ["date", "ticker", "target"],
        ].head(5)
        raise ValueError(
            "target does not match the documented next-week adjusted-close log return "
            f"for {mismatches} consecutive observations. Examples:\n{examples}"
        )
    return {
        "formula": "log(adj_close[t+1] / adj_close[t])",
        "comparable_consecutive_rows": int(consecutive.sum()),
        "mismatch_rows": mismatches,
        "max_absolute_difference": max_abs,
        "unverifiable_rows": int((~consecutive).sum()),
        "note": (
            "Rows are unverifiable inside this factor file when a ticker is absent in the "
            "next weekly cross-section or when the row is the ticker's final observation."
        ),
    }


def load_and_prepare_data(
    input_path: Path,
    factor_definitions_path: Path,
    features: list[str],
    forecast_horizon_weeks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_feature_safety(features)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    frame = pd.read_csv(input_path)
    required = {
        "date",
        "ticker",
        "adj_close",
        "target",
        "target2",
        *features,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame[["date", "ticker"]].isna().any().any():
        raise ValueError("date and ticker cannot be missing")
    duplicate_count = int(frame.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        raise ValueError(f"date+ticker must be unique; found {duplicate_count} duplicates")
    if frame["target"].isna().any():
        raise ValueError("target contains missing values")

    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(frame[numeric_columns].to_numpy(dtype=float, copy=False)).sum()
    if inf_counts:
        raise ValueError(f"Numeric input contains {int(inf_counts)} infinite values")
    non_numeric_features = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric_features:
        raise ValueError(f"All features must be numeric: {non_numeric_features}")

    dates = pd.Index(sorted(frame["date"].unique()))
    date_to_index = {date: i for i, date in enumerate(dates)}
    frame["week_index"] = frame["date"].map(date_to_index).astype(int)
    frame["label_end_week"] = frame["week_index"] + forecast_horizon_weeks
    frame = frame.sort_values(["week_index", "ticker"], kind="stable").reset_index(drop=True)

    if not frame[["week_index", "ticker"]].equals(
        frame[["week_index", "ticker"]].sort_values(
            ["week_index", "ticker"], kind="stable"
        ).reset_index(drop=True)
    ):
        raise AssertionError("Prepared data must be sorted by week_index and ticker")

    date_gaps = pd.Series(dates).diff().dropna().dt.days
    if not date_gaps.between(3, 11).all():
        raise ValueError("Unique decision dates do not look weekly (expected 3-11 day gaps)")

    definition = _target_definition_text(factor_definitions_path)
    normalized = definition.lower().replace(" ", "")
    definition_confirms_forward_log_return = (
        "ln(adjclose_week_end(t+1)/adjclose_week_end(t))" in normalized
        and "nextweek" in normalized
    )
    if not definition_confirms_forward_log_return:
        raise ValueError(
            "Could not confirm from factor definitions that target is the aligned next-week "
            "adjusted-close log return. No automatic shift was performed."
        )
    alignment = _verify_target_alignment(frame)

    diagnostics = {
        "input_filename": input_path.name,
        "input_sha256": sha256_file(input_path),
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()),
        "weeks": int(len(dates)),
        "start_date": dates.min().strftime("%Y-%m-%d"),
        "end_date": dates.max().strftime("%Y-%m-%d"),
        "min_stocks_per_week": int(frame.groupby("week_index").size().min()),
        "max_stocks_per_week": int(frame.groupby("week_index").size().max()),
        "target_column": "target",
        "target_definition": definition,
        "target_alignment": alignment,
        "missing_feature_values": int(frame[features].isna().sum().sum()),
    }
    return frame, diagnostics


def eligible_training_rows(
    frame: pd.DataFrame,
    candidate_week_indices: list[int] | range,
    refit_week: int,
) -> pd.DataFrame:
    candidate_set = set(candidate_week_indices)
    result = frame.loc[
        frame["week_index"].isin(candidate_set)
        & frame["week_index"].lt(refit_week)
        & frame["label_end_week"].lt(refit_week)
    ].copy()
    if not result.empty and not result["label_end_week"].lt(refit_week).all():
        raise AssertionError("Training data violates label_end_week < refit_week")
    return result

