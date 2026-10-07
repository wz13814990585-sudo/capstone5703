"""Fit imputation and winsorization on a training window only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessParams:
    median: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def fit_preprocessor(frame, features, *, standardize=False, missing_fill=None):
    raw = frame[features].replace([np.inf, -np.inf], np.nan)
    median = raw.median()
    if missing_fill is not None:
        median = median.fillna(missing_fill)
    if median.isna().any():
        raise ValueError("An entire training feature is missing")
    filled = raw.fillna(median)
    low, high = filled.quantile(.01), filled.quantile(.99)
    clipped = filled.clip(low, high, axis=1)
    mean = clipped.mean() if standardize else pd.Series(0., index=features)
    scale = clipped.std(ddof=0).mask(lambda x: x == 0, 1.) if standardize else pd.Series(1., index=features)
    params = PreprocessParams(*(s.to_numpy(float) for s in (median, low, high, mean, scale)))
    x = ((clipped - mean) / scale).to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite training values after preprocessing")
    return x, params


def apply_preprocessor(frame, features, params):
    raw = frame[features].to_numpy(float)
    filled = np.where(np.isfinite(raw), raw, params.median)
    x = (np.clip(filled, params.clip_low, params.clip_high) - params.mean) / params.scale
    if not np.isfinite(x).all():
        raise ValueError("Non-finite prediction inputs after preprocessing")
    return x
