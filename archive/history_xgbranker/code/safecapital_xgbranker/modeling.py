from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from .labels import add_relevance_labels, group_sizes


def fit_ranker(
    training_rows: pd.DataFrame,
    features: list[str],
    objective: str,
    relevance_bins: int,
    params: dict[str, Any],
    random_seed: int,
    n_jobs: int,
) -> XGBRanker:
    ordered = training_rows.sort_values(["week_index", "ticker"], kind="stable")
    labelled = add_relevance_labels(ordered, relevance_bins)
    groups = group_sizes(labelled)
    model = XGBRanker(
        objective=objective,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )
    model.fit(
        labelled[features].to_numpy(dtype=np.float32),
        labelled["relevance_label"].to_numpy(dtype=np.int32),
        group=groups,
        verbose=False,
    )
    return model


def predict_ranking_block(
    model: XGBRanker,
    prediction_rows: pd.DataFrame,
    features: list[str],
    relevance_bins: int,
    model_id: str,
    refit_week: int,
    refit_date: pd.Timestamp,
    factor_set_name: str,
) -> pd.DataFrame:
    ordered = prediction_rows.sort_values(["week_index", "ticker"], kind="stable").copy()
    ordered["ranking_score"] = model.predict(
        ordered[features].to_numpy(dtype=np.float32)
    )
    ordered["model_id"] = model_id
    ordered["refit_week"] = int(refit_week)
    ordered["refit_date"] = pd.Timestamp(refit_date)
    ordered["factor_set_name"] = factor_set_name
    ordered["actual_return"] = ordered["target"].astype(float)
    ordered["predicted_rank"] = ordered.groupby("week_index", sort=False)[
        "ranking_score"
    ].rank(method="first", ascending=False).astype(int)
    ordered["actual_rank"] = ordered.groupby("week_index", sort=False)[
        "actual_return"
    ].rank(method="first", ascending=False).astype(int)
    ordered = add_relevance_labels(ordered, relevance_bins)
    output_columns = [
        "week_index",
        "date",
        "ticker",
        "model_id",
        "refit_week",
        "refit_date",
        "factor_set_name",
        "ranking_score",
        "predicted_rank",
        "actual_return",
        "actual_rank",
        "relevance_label",
    ]
    result = ordered[output_columns].copy()
    if result.duplicated(["date", "ticker"]).any():
        raise AssertionError("Prediction date+ticker must be unique")
    per_week_model_count = result.groupby("week_index")["model_id"].nunique()
    if not per_week_model_count.eq(1).all():
        raise AssertionError("Each prediction week must map to one model_id")
    return result

