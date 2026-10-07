from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(x_rank, y_rank)


def calculate_weekly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week_index, week in predictions.groupby("week_index", sort=True):
        score = week["ranking_score"].to_numpy(dtype=float)
        actual = week["actual_return"].to_numpy(dtype=float)
        rows.append(
            {
                "week_index": int(week_index),
                "date": pd.Timestamp(week["date"].iloc[0]),
                "model_id": str(week["model_id"].iloc[0]),
                "IC": _pearson(score, actual),
                "Rank_IC": _spearman(score, actual),
                "number_of_stocks": int(len(week)),
            }
        )
    return pd.DataFrame(rows)


def summarize_weekly_metrics(weekly: pd.DataFrame) -> dict[str, Any]:
    def summarize(column: str) -> tuple[float, float, float, int]:
        values = weekly[column].dropna().astype(float)
        if values.empty:
            return float("nan"), float("nan"), float("nan"), 0
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        ratio = mean / std if np.isfinite(std) and std > 0 else float("nan")
        positive = float((values > 0).mean())
        return mean, ratio, positive, int(len(values))

    ic, icir, positive_ic, valid_ic = summarize("IC")
    rank_ic, rank_icir, positive_rank_ic, valid_rank_ic = summarize("Rank_IC")
    return {
        "IC": ic,
        "ICIR": icir,
        "Rank_IC": rank_ic,
        "Rank_ICIR": rank_icir,
        "Positive_IC_Ratio": positive_ic,
        "Positive_Rank_IC_Ratio": positive_rank_ic,
        "number_of_prediction_weeks": int(weekly["week_index"].nunique()),
        "number_of_valid_IC_weeks": valid_ic,
        "number_of_valid_Rank_IC_weeks": valid_rank_ic,
    }

