from __future__ import annotations

import numpy as np
import pandas as pd


def add_relevance_labels(
    frame: pd.DataFrame,
    bins: int,
    target_column: str = "target",
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("relevance bins must be at least 2")
    result = frame.copy()
    grouped = result.groupby("week_index", sort=False)[target_column]
    within_week_rank = grouped.rank(method="average", ascending=True)
    group_size = grouped.transform("count")
    labels = np.floor((within_week_rank - 1.0) * bins / group_size)
    result["relevance_label"] = labels.clip(0, bins - 1).astype(int)
    if not result["relevance_label"].between(0, bins - 1).all():
        raise AssertionError("Invalid relevance label")
    return result


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    if not frame["week_index"].is_monotonic_increasing:
        raise ValueError("Ranking rows must be sorted by week_index")
    sizes = frame.groupby("week_index", sort=False).size().to_numpy(dtype=np.uint32)
    if int(sizes.sum()) != len(frame):
        raise AssertionError("Ranking group sizes do not sum to training rows")
    return sizes

