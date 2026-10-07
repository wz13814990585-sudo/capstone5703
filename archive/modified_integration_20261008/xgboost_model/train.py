"""XGBoost regression with the original monthly/fixed comparison protocol."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from model_training.features import SEVEN, MACRO, ALL45
from model_training.preprocessing import fit_preprocessor, apply_preprocessor
from model_training.artifacts import (ROOT, DEFAULT_INPUT, annualize, save_json,
    standard_predictions, save_predictions, output_directory, sha, validate_keys)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


DATE, TICKER, TARGET = "date", "ticker", "target"
MODEL_TARGET = "_model_target"
# These are calendar anchors only.  The actual stage boundaries are resolved
# to the first observed weekly signal date on or after each anchor.
DEV_START = pd.Timestamp("2019-09-01")
VALID_START = pd.Timestamp("2024-09-01")
TEST_START = pd.Timestamp("2025-09-01")
TOP_K = 5
MIN_RANK_ICIR = 0.05
MIN_POSITIVE_RANK_IC = 0.50


def factor_sets(source: Path):
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = set(ALL45) - set(header)
    if missing:
        raise ValueError(f"Missing factor columns: {sorted(missing)}")
    # Retain source column order for the full baseline, ignoring metadata.
    return {"7": SEVEN, "7_macro": SEVEN + MACRO,
            "45": [c for c in header if c in ALL45]}


def load_panel(source: Path, features: list[str]) -> pd.DataFrame:
    data = pd.read_csv(source, parse_dates=[DATE], encoding="utf-8-sig")
    required = {DATE, TICKER, TARGET, "adj_close", *features}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    validate_keys(data)
    data = data.sort_values([TICKER, DATE]).reset_index(drop=True)
    next_date = data.groupby(TICKER, sort=False)[DATE].shift(-1)
    data["label_end_date"] = next_date
    data[MODEL_TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    gap = (next_date - data[DATE]).dt.days
    data["label_end_date"] = data["label_end_date"].where(gap.between(5, 14))
    for col in features + [TARGET, "adj_close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    usable = np.isfinite(data[MODEL_TARGET]) & data["label_end_date"].notna()
    usable &= data[features].notna().any(axis=1)
    panel = data.loc[usable].copy().reset_index(drop=True)
    if panel.empty:
        raise ValueError("No usable rows remain after Target/date checks")
    return panel


def resolve_stage_dates(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Map calendar anchors to dates that actually exist in the weekly panel."""
    observed = pd.Series(panel[DATE].dropna().sort_values().unique())
    resolved = {}
    for name, anchor in [("development", DEV_START),
                         ("validation", VALID_START),
                         ("final_test", TEST_START)]:
        candidates = observed[observed >= anchor]
        if candidates.empty:
            raise ValueError(f"No observed weekly signal date on/after {anchor.date()} for {name}")
        resolved[name] = pd.Timestamp(candidates.iloc[0])
    if not (resolved["development"] < resolved["validation"] < resolved["final_test"]):
        raise ValueError(f"Resolved stage dates are not strictly increasing: {resolved}")
    return resolved


def candidates():
    for depth, child, trees in itertools.product([2, 3], [5, 20], [50, 150, 300]):
        yield {"max_depth": depth, "min_child_weight": child, "n_estimators": trees}


def params(candidate):
    return {
        "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
        "n_jobs": 4, "random_state": 42, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0,
        "reg_lambda": 10, **candidate,
    }


def prepared(train, predict, features):
    x_train, prep = fit_preprocessor(train, features, missing_fill=0.0)
    return x_train, apply_preprocessor(predict, features, prep)


def fit_predict(train, predict, features, candidate):
    if xgb is None:
        raise SystemExit(
            "Missing XGBoost. Install with: python -m pip install "
            "scikit-learn xgboost"
        )
    x_train, x_pred = prepared(train, predict, features)
    model = xgb.XGBRegressor(**params(candidate))
    model.fit(x_train, train[MODEL_TARGET].to_numpy(float))
    out = predict[[DATE, TICKER, "label_end_date"]].copy()
    out["y_true"] = predict[MODEL_TARGET].to_numpy(float)
    out["y_pred"] = model.predict(x_pred)
    return out


def rank_series(pred, method):
    values = {}
    for date, group in pred.groupby(DATE, sort=True):
        group = group[["y_pred", "y_true"]].dropna()
        if len(group) >= 2 and group.y_pred.nunique() > 1 and group.y_true.nunique() > 1:
            values[date] = group.y_pred.corr(group.y_true, method=method)
    return pd.Series(values, dtype=float)



def stage_metrics(pred):
    ic, ric = rank_series(pred, "pearson"), rank_series(pred, "spearman")
    top, market = {}, {}
    for date, group in pred.groupby(DATE, sort=True):
        group = group.dropna(subset=["y_pred", "y_true"])
        if len(group) < TOP_K:
            continue
        actual = np.expm1(group.y_true)
        chosen = group.nlargest(TOP_K, "y_pred")
        top[date] = float(np.expm1(chosen.y_true).mean())
        market[date] = float(actual.mean())
    top, market = pd.Series(top), pd.Series(market)
    excess = top - market.reindex(top.index)
    def ir(series):
        series = series.dropna()
        return float(series.mean() / series.std(ddof=1)) if len(series) > 1 and series.std(ddof=1) > 0 else np.nan
    return {
        "IC": ic.mean(), "ICIR": ir(ic), "Rank IC": ric.mean(),
        "Rank ICIR": ir(ric), "Annualized Return": annualize(top),
        "Information Ratio": ir(excess) * np.sqrt(52),
        "Top5 Excess": annualize(excess),
        "Positive Rank IC": (ric > 0).mean() if len(ric) else np.nan,
    }


def rolling_folds(panel, rolling_years):
    # A rolling experiment must have a complete history window before its
    # first prediction.  With --rolling-years 6, predictions therefore start
    # six years after the first observation; the calendar validation/test
    # boundaries remain unchanged for comparability.
    first_fit = panel[DATE].min() + pd.DateOffset(years=rolling_years)
    eligible = panel[panel[DATE] >= first_fit]
    rows = []
    for (_, stage), part in eligible.groupby([eligible[DATE].dt.to_period("M"), "stage"], sort=True):
        fit_date = part[DATE].min()
        train = panel[(panel[DATE] >= fit_date - pd.DateOffset(years=rolling_years)) &
                      (panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date)]
        if not train.empty:
            rows.append({"stage": stage, "fit_date": fit_date, "end": part[DATE].max()})
    return rows


def add_stages(panel, stage_dates):
    first_fit = stage_dates["development"]
    panel = panel.copy()
    panel["stage"] = np.select(
        [panel[DATE] < first_fit,
         panel[DATE] < stage_dates["validation"],
         panel[DATE] < stage_dates["final_test"]],
        ["initial_history", "development", "validation"], default="final_test")
    return panel


def run_rolling(panel, features, candidate, folds, rolling_years, stages=None):
    parts = []
    for fold in folds:
        if stages is not None and fold["stage"] not in stages:
            continue
        fit_date = fold["fit_date"]
        train = panel[(panel[DATE] >= fit_date - pd.DateOffset(years=rolling_years)) &
                      (panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date)]
        pred = panel[(panel[DATE] >= fit_date) & (panel[DATE] <= fold["end"])]
        out = fit_predict(train, pred, features, candidate)
        out["stage"] = fold["stage"]
        out["fit_date"] = fit_date
        out["training_first_date"] = train[DATE].min()
        out["training_last_date"] = train[DATE].max()
        out["latest_training_label"] = train["label_end_date"].max()
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[DATE,TICKER,"label_end_date","y_true","y_pred","stage"])


def run_fixed(panel, features, candidate, stage_dates, stages=None):
    parts = []
    splits = [("development", stage_dates["development"], stage_dates["validation"],
               stage_dates["development"]),
              ("validation", stage_dates["validation"], stage_dates["final_test"],
               stage_dates["validation"]),
              ("final_test", stage_dates["final_test"], None, stage_dates["final_test"])]
    for stage, start, end, cutoff in splits:
        if stages is not None and stage not in stages:
            continue
        train = panel[(panel[DATE] < cutoff) & (panel["label_end_date"] < cutoff)]
        pred = panel[panel[DATE] >= start]
        if end is not None:
            pred = pred[pred[DATE] < end]
        out = fit_predict(train, pred, features, candidate)
        out["stage"] = stage
        out["fit_date"] = cutoff
        out["training_first_date"] = train[DATE].min()
        out["training_last_date"] = train[DATE].max()
        out["latest_training_label"] = train["label_end_date"].max()
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[DATE,TICKER,"label_end_date","y_true","y_pred","stage"])


def run_one(mode, feature_set, features, panel, folds, rolling_years,
            stage_dates, output, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    candidates_list = list(candidates())
    if quick:
        candidates_list = [{**candidates_list[0], "n_estimators": 8}]
    def predict(candidate, stages=None):
        return (run_rolling(panel, features, candidate, folds, rolling_years, stages)
                if mode == "rolling" else run_fixed(panel, features, candidate, stage_dates, stages))
    development_scores = []
    for number, candidate in enumerate(candidates_list, 1):
        # Candidate scores depend only on Development. Later-stage fits for
        # rejected candidates never influenced selection and were redundant.
        dev = predict(candidate, {"development"})
        development_scores.append({"candidate": number, **candidate, **stage_metrics(dev)})
        print(f"XGBoost {mode}/{feature_set}: candidate {number}/{len(candidates_list)}", flush=True)
    scores = pd.DataFrame(development_scores)
    scores["passes_gate"] = ((scores["Rank IC"] > 0) &
                              (scores["Rank ICIR"] >= MIN_RANK_ICIR) &
                              (scores["Positive Rank IC"] >= MIN_POSITIVE_RANK_IC))
    scores.to_csv(output / "development_scores.csv", index=False, encoding="utf-8-sig")
    eligible = scores[scores.passes_gate & scores["Top5 Excess"].notna()]
    status = {"model": f"xgboost_{feature_set}_{mode}", "mode": mode, "feature_set": feature_set,
              "status": "no_candidate_passed", "passing_candidates": len(eligible)}
    if eligible.empty:
        print(f"XGBoost {mode}/{feature_set}: no candidate passed; skipped", flush=True)
        save_json(output / "locked_selection.json", status)
        return status, None, []
    chosen = eligible.sort_values(["Top5 Excess", "Rank IC", "Rank ICIR"],
                                  ascending=[False, False, False]).iloc[0]
    cid = int(chosen.candidate)
    candidate = candidates_list[cid - 1]
    selected = standard_predictions(predict(candidate), model="xgboost", mode=mode, feature_set=feature_set)
    selected["candidate"] = cid
    selected.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    status.update(status="selected", candidate=cid, parameters=candidate)
    save_json(output / "locked_selection.json", status)
    metrics = [{"model": status["model"], "mode": mode, "feature_set": feature_set,
                "stage": stage, "candidate": cid, **stage_metrics(part)}
               for stage, part in selected.groupby("stage", sort=False)]
    pd.DataFrame(metrics).to_csv(output / "performance_metrics.csv", index=False, encoding="utf-8-sig")
    return status, selected, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mode", choices=["rolling", "fixed", "both"], default="both")
    parser.add_argument("--feature-set", choices=["all", "7", "7_macro", "45"], default="all")
    parser.add_argument("--rolling-years", type=int, default=3,
                        help="Training history used by rolling mode; stage split remains unchanged (default: 3).")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "xgboost")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: one candidate, eight trees, two monthly blocks per stage")
    args = parser.parse_args(argv)
    if args.rolling_years < 1:
        parser.error("--rolling-years must be at least 1")
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"CSV not found: {source}")
    groups = factor_sets(source)
    selected = list(groups) if args.feature_set == "all" else [args.feature_set]
    panel_cache = {}
    stage_dates_cache = {}
    for name in selected:
        panel = load_panel(source, groups[name])
        stage_dates = resolve_stage_dates(panel)
        panel = add_stages(panel, stage_dates)
        panel_cache[name] = panel
        stage_dates_cache[name] = stage_dates
    print(f"Input: {source}")
    print("Target: target (already forward-aligned log return)")
    print("Original 12-candidate protocol; quick mode uses one eight-tree candidate." if args.quick else
          "Same data, split dates and 12-candidate grid for every factor set.")
    print(f"Rolling training history: {args.rolling_years} years")
    first_dates = stage_dates_cache[selected[0]]
    print("Signal-aligned stages: "
          f"Development starts {first_dates['development'].date()}, "
          f"Validation starts {first_dates['validation'].date()}, "
          f"Final Test starts {first_dates['final_test'].date()}")
    if args.check_only:
        for name in selected:
            panel = panel_cache[name]
            print(f"{name}: {len(groups[name])} factors, {len(panel):,} rows, "
                  f"{panel[TICKER].nunique()} stocks")
        return {"status": "checked", "feature_sets": selected}
    output = output_directory(args.output)
    save_json(output / "config.json", {
        "source": source, "source_sha256": sha(source), "quick": args.quick,
        "feature_sets": {name: groups[name] for name in selected},
        "rolling_years": args.rolling_years, "rolling_cadence": "calendar month, split by stage",
        "stage_dates": stage_dates_cache,
        "candidate_grid": [{**list(candidates())[0], "n_estimators": 8}] if args.quick else list(candidates()),
        "fixed_parameters": params({}),
        "label_rule": "next same-ticker observation 5–14 days later; label_end_date < fit_date",
        "selection": "Development gate, then Top5 Excess/Rank IC/Rank ICIR descending; no fallback",
        "selection_boundary": "Original Development signal dates retained; no new label-boundary purge",
    })
    statuses, predictions, metrics = [], [], []
    for name in selected:
        panel = panel_cache[name]
        folds = rolling_folds(panel, args.rolling_years)
        if args.quick:
            folds = [f for stage in ["development", "validation", "final_test"]
                     for f in [x for x in folds if x["stage"] == stage][:2]]
        modes = ["rolling", "fixed"] if args.mode == "both" else [args.mode]
        for mode in modes:
            status, pred, metric = run_one(mode, name, groups[name], panel, folds,
                args.rolling_years, stage_dates_cache[name], output / mode / name, args.quick)
            statuses.append(status)
            if pred is not None:
                predictions.append(pred)
            metrics.extend(metric)
    save_predictions(output / "predictions.csv", predictions, quick_run=args.quick)
    pd.DataFrame(statuses).to_csv(output / "selection_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics, columns=["model", "mode", "feature_set", "stage", "candidate",
        "IC", "ICIR", "Rank IC", "Rank ICIR", "Annualized Return", "Information Ratio",
        "Top5 Excess", "Positive Rank IC"]).to_csv(output / "all_stage_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "run_manifest.json", {"status": "completed", "quick": args.quick,
        "source_sha256": sha(source), "results": statuses})
    return {"status": "completed", "output": str(output), "predictions": str(output / "predictions.csv")}



if __name__ == "__main__":
    main()
