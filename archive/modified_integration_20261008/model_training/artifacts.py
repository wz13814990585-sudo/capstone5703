"""Common artifact format; the individual model selection protocols stay explicit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "test_45_factors_weekly.csv"
PREDICTION_COLUMNS = ["date", "ticker", "score", "target", "label_end_date",
                      "stage", "model", "mode", "feature_set", "score_kind", "target_type"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                   indent=2, allow_nan=False), encoding="utf-8")


def output_directory(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def standard_predictions(predictions, *, model, mode, feature_set, target_type="log"):
    frame = predictions.copy()
    if "score" not in frame:
        frame["score"] = frame["prediction" if "prediction" in frame else "y_pred"]
    if "target" not in frame:
        frame["target"] = frame["actual" if "actual" in frame else "y_true"]
    frame["model"] = f"{model}_{feature_set}_{mode}"
    frame["mode"] = mode
    frame["feature_set"] = feature_set
    frame["score_kind"] = "return"
    frame["target_type"] = target_type
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction metadata: {sorted(missing)}")
    return frame[PREDICTION_COLUMNS + [c for c in frame if c not in PREDICTION_COLUMNS]]


def save_predictions(path, parts, *, quick_run=False):
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PREDICTION_COLUMNS)
    frame["quick_run"] = bool(quick_run)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def validate_keys(frame):
    if frame[["date", "ticker"]].isna().any().any():
        raise ValueError("Missing date/ticker values")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker pairs")


def annualize(values):
    values = pd.Series(values).dropna().astype(float)
    return float(np.expm1(np.log1p(values).mean() * 52)) if len(values) and (values > -1).all() else np.nan


def mean_std_ratio(values):
    values = pd.Series(values).dropna()
    return float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan
