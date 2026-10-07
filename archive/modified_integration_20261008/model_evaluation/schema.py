"""One prediction contract shared by training, evaluation, and backtesting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_predictions(
    predictions: pd.DataFrame,
    *,
    score_kind: str | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Return validated predictions with date/ticker/score/target/model metadata.

    ``target`` is the next global week's close-to-close log return. Missing
    targets are permitted for live/terminal predictions, but invalid scores and
    duplicate model/date/ticker keys are errors. Ranking scores are dimensionless.
    """
    frame = predictions.copy()
    aliases = {
        "score": ("prediction", "pred", "ranking_score", "ensemble_ranking_score"),
        "target": ("actual_return", "y_true"),
    }
    for canonical, alternatives in aliases.items():
        if canonical not in frame:
            found = next((name for name in alternatives if name in frame), None)
            if found is not None:
                frame = frame.rename(columns={found: canonical})
    required = {"date", "ticker", "score"} | ({"target"} if require_target else set())
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Predictions are empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Prediction dates must not be missing")
    if frame["ticker"].isna().any():
        raise ValueError("Prediction tickers must not be missing")
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    if frame["ticker"].eq("").any():
        raise ValueError("Prediction tickers must not be empty")
    if "model" not in frame:
        frame["model"] = "model"
    if frame["model"].isna().any():
        raise ValueError("Prediction model names must not be missing")
    frame["model"] = frame["model"].astype(str)
    if frame.duplicated(["model", "date", "ticker"]).any():
        raise ValueError("Duplicate model/date/ticker predictions; choose one configuration first")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores must be finite")
    if "target" not in frame:
        frame["target"] = np.nan
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(float)
    if np.isinf(frame["target"].to_numpy(dtype=float)).any():
        raise ValueError("Prediction targets may be missing, but must not be infinite")
    if "label_end_date" not in frame:
        frame["label_end_date"] = pd.NaT
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="raise")
    invalid_end = frame["label_end_date"].notna() & frame["label_end_date"].le(frame["date"])
    if invalid_end.any():
        raise ValueError("label_end_date must be later than the signal date")
    if score_kind is not None:
        frame["score_kind"] = score_kind
    elif "score_kind" not in frame:
        is_rank = frame["model"].str.contains("ranker|consensus", case=False)
        if any(name in predictions for name in ("ranking_score", "ensemble_ranking_score")):
            is_rank[:] = True
        frame["score_kind"] = np.where(is_rank, "rank", "return")
    frame["score_kind"] = frame["score_kind"].replace({"ranking": "rank", "regression": "return"})
    if not frame["score_kind"].isin(["rank", "return"]).all():
        raise ValueError("score_kind must be 'return' or 'rank'")
    if frame.groupby("model")["score_kind"].nunique().gt(1).any():
        raise ValueError("Each model must have exactly one score_kind")
    if "target_type" not in frame:
        frame["target_type"] = "log"
    if not frame["target_type"].isin(["log", "simple"]).all():
        raise ValueError("target_type must be 'log' or 'simple'")
    simple = frame["target_type"].eq("simple")
    simple_return_score = simple & frame["score_kind"].eq("return")
    if frame.loc[simple, "target"].le(-1).any() or frame.loc[simple_return_score, "score"].le(-1).any():
        raise ValueError("Simple returns must exceed -100% to convert to log returns")
    frame.loc[simple, "target"] = np.log1p(frame.loc[simple, "target"])
    frame.loc[simple_return_score, "score"] = np.log1p(frame.loc[simple_return_score, "score"])
    frame["target_type"] = "log"
    return frame.sort_values(["model", "date", "ticker"], kind="stable").reset_index(drop=True)
