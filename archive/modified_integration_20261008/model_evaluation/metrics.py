"""Weekly cross-sectional predictive metrics, with equal weighting of weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import normalize_predictions


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; constant or undersized cross sections are undefined."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_predictions(
    predictions: pd.DataFrame, *, score_kind: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-model summary, weekly metrics) under the shared contract.

    ICIR uses sample standard deviation and is not annualized. MAE/RMSE are
    pooled row errors in log-return units and are omitted for ranking scores.
    Weeks containing constant scores/targets have NaN correlation, never zero.
    When label_end_date is supplied, missing end dates are treated as unrealized
    and excluded. Legacy files without this column rely on their supplied labels.
    """
    frame = normalize_predictions(predictions, score_kind=score_kind)
    frame["_label_available"] = frame["target"].notna()
    if "label_end_date" in predictions:
        frame["_label_available"] &= frame["label_end_date"].notna()
    weekly_rows = []
    summaries = []
    for model, model_frame in frame.groupby("model", sort=True):
        kind = model_frame["score_kind"].iloc[0]
        for date, group in model_frame.groupby("date", sort=True):
            group = group.loc[group["_label_available"]]
            score = group["score"].to_numpy(dtype=float)
            target = group["target"].to_numpy(dtype=float)
            ic = correlation(score, target)
            rank_ic = correlation(
                pd.Series(score).rank(method="average").to_numpy(),
                pd.Series(target).rank(method="average").to_numpy(),
            )
            weekly_rows.append({"model": model, "date": date, "score_kind": kind,
                                "observations": len(group), "ic": ic, "rank_ic": rank_ic})
        own = pd.DataFrame([row for row in weekly_rows if row["model"] == model])
        valid = model_frame.loc[model_frame["_label_available"]]
        summary = {"model": model, "score_kind": kind, "rows": len(model_frame),
                   "evaluated_rows": len(valid), "missing_target_rows": int(model_frame["target"].isna().sum()),
                   "unevaluable_rows": len(model_frame) - len(valid),
                   "weeks": len(own), "first_date": model_frame["date"].min(),
                   "last_date": model_frame["date"].max()}
        for name in ("ic", "rank_ic"):
            values = own[name].dropna()
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary[name] = mean
            summary[f"{name}ir"] = mean / std if np.isfinite(std) and std > 0 else float("nan")
            summary[f"positive_{name}_ratio"] = float(values.gt(0).mean()) if len(values) else float("nan")
            summary[f"valid_{name}_weeks"] = len(values)
        errors = (valid["score"] - valid["target"]).to_numpy()
        summary["mae"] = float(np.abs(errors).mean()) if kind == "return" and len(errors) else float("nan")
        summary["rmse"] = float(np.sqrt(np.square(errors).mean())) if kind == "return" and len(errors) else float("nan")
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(weekly_rows)
