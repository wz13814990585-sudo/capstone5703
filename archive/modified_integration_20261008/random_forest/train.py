"""Nine Random Forest experiments: 7, 11 and 45 factors x three training designs.
Rolling 3y: initial 3 years; Development/Validation/Final Test: 5/1/1 years.
Rolling 6y and fixed: initial 6 years; subsequent stages: 2/1/1 years.
Rolling: trailing three or six calendar years, refitted every four observation weeks.
Fixed: all eligible history, one fit at each stage start per candidate.
Development shortlists five of 18 candidates; Validation chooses one.
Final Test freezes parameters. Predictions and audit tables are exported.
Full-sample factor screening can introduce selection look-ahead.
"""
from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from model_training.features import FEATURE_SETS
from model_training.preprocessing import fit_preprocessor, apply_preprocessor
from model_training.artifacts import (ROOT, DEFAULT_INPUT, output_directory, save_json,
    standard_predictions, save_predictions, sha, validate_keys)
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

# FOUR STAGES: start inclusive, end exclusive. Validation/Test are shared.
# The 3y rolling design starts Development three years earlier.
START_HISTORY = pd.Timestamp("2016-09-01")
START_DEV = pd.Timestamp("2022-09-01")
START_VALID = pd.Timestamp("2024-09-01")
START_TEST = pd.Timestamp("2025-09-01")
END_TEST = pd.Timestamp("2026-09-01")
TRAINING_DESIGNS = {
    "rolling_3y": {"mode": "rolling", "years": 3,
                   "dev_start": pd.Timestamp("2019-09-01")},
    "rolling_6y": {"mode": "rolling", "years": 6, "dev_start": START_DEV},
    "fixed": {"mode": "fixed", "years": 6, "dev_start": START_DEV},
}
REFIT_WEEKS = 4
TOP_K = 5
SHORTLIST_SIZE = 5

# The workbook contains Chinese labels. Unicode escapes preserve exact matching
# while keeping this Python file and all user-facing output in English.
SCREENING_SHEET = "\u7b5b\u9009\u7ed3\u679c"
PRIORITY_LABEL = "\u4f18\u5148\u5c1d\u8bd5"
DUPLICATE_LABEL = "\u5254\u9664\u91cd\u590d"
MACRO_LABEL = "\u5355\u72ec\u4fdd\u7559"
BACKUP_TEXT = "\u5907\u9009"


def screening_table(path):
    workbook = pd.ExcelFile(path)
    sheet = SCREENING_SHEET if SCREENING_SHEET in workbook.sheet_names else "Screening Results"
    table = pd.read_excel(workbook, sheet_name=sheet, header=4)
    table.columns = [str(c).strip() for c in table.columns]
    if not {"Factor", "Decision"}.issubset(table.columns):
        raise ValueError("Screening workbook needs Factor and Decision columns")
    table = table.loc[table.Factor.notna()].copy()
    table["Factor"] = table.Factor.astype(str).str.strip()
    english = {"Priority candidate": PRIORITY_LABEL, "Remove duplicate": DUPLICATE_LABEL,
               "Retain separately": MACRO_LABEL, "General alternative": BACKUP_TEXT,
               "Weak-signal alternative": BACKUP_TEXT, "Correlated alternative": BACKUP_TEXT}
    table["Decision"] = table.Decision.astype(str).str.strip().replace(english)
    decisions = table.Decision
    counts = {"priority": int((decisions == PRIORITY_LABEL).sum()),
              "duplicate": int((decisions == DUPLICATE_LABEL).sum()),
              "macro": int((decisions == MACRO_LABEL).sum()),
              "backup": int(decisions.str.contains(BACKUP_TEXT).sum())}
    if counts != {"priority": 7, "duplicate": 4, "macro": 4, "backup": 30}:
        raise ValueError(f"Unexpected screening counts: {counts}")
    return table


def read_panel(path: Path, features: list[str], target: str, universe: Path | None,
               expected_stocks: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(["date", "ticker", target, *features]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in weekly data: {sorted(missing)}")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    validate_keys(df)
    df["ticker"] = df.ticker.astype(str).str.strip()
    if universe:
        names = pd.read_csv(universe)
        if "ticker" not in names:
            raise ValueError("Universe CSV must have a ticker column")
        tickers = names.ticker.astype(str).str.strip().tolist()
        if len(set(tickers)) != expected_stocks or len(tickers) != expected_stocks:
            raise ValueError(f"Universe file must list {expected_stocks} distinct tickers")
        absent = set(tickers) - set(df.ticker)
        if absent:
            raise ValueError(f"Universe tickers missing in panel: {sorted(absent)}")
        df = df[df.ticker.isin(tickers)].copy()
    if df.ticker.nunique() != expected_stocks:
        raise ValueError(f"Expected {expected_stocks} stocks, found {df.ticker.nunique()}. "
                         "Supply the matching weekly dataset or --universe-file.")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker pairs")
    df = df.loc[(df.date >= START_HISTORY) & (df.date < END_TEST)].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df.loc[~np.isfinite(df[target]), target] = np.nan
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    # A one-week forward label is known at the next observation date. Exclude
    # observations whose label could not have been observed at fit time.
    dates = sorted(df.date.unique())
    df["label_end_date"] = df.date.map(dict(zip(dates[:-1], dates[1:])))
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def candidate_grid():
    for depth, leaf, max_features in product([4, 6, 8], [10, 20, 30], ["sqrt", 0.5]):
        yield {"max_depth": depth, "min_samples_leaf": leaf,
               "max_features": max_features, "n_estimators": 100}


def fit_predict(train, evaluate, features, target, params, return_model=False):
    x, prep = fit_preprocessor(train, features)
    model = RandomForestRegressor(**params, random_state=5703, n_jobs=-1)
    model.fit(x, train[target].to_numpy(float))
    pred = evaluate[["date", "ticker", "label_end_date", target]].rename(columns={target: "actual"}).copy()
    pred["prediction"] = model.predict(apply_preprocessor(evaluate, features, prep))
    return (pred, model) if return_model else pred


def weekly_metrics(pred: pd.DataFrame, target_type: str = "log") -> pd.DataFrame:
    """Weekly full-universe cross-sectional Pearson IC and Spearman Rank IC."""
    rows = []
    for date, group in pred.groupby("date", sort=True):
        group = group.dropna(subset=["prediction", "actual"])
        if len(group) < 2:
            continue
        variable = group.prediction.nunique() > 1 and group.actual.nunique() > 1
        ic = group.prediction.corr(group.actual, method="pearson") if variable else np.nan
        rank_ic = group.prediction.corr(group.actual, method="spearman") if variable else np.nan
        top = group.sort_values(["prediction", "ticker"], ascending=[False, True]).head(TOP_K)
        if len(top) != TOP_K:
            continue
        """
        Equal-weight Top-5, cash-free, weekly rebalancing. Benchmark is that
        week's equal-weight universe; zero costs. Overlapping weeks are not
        permitted by the one-week label convention. The supplied 'target'
         is a LOG return; convert each stock separately before averaging. 
         """
        
        actual_simple = (np.expm1(group.actual) if target_type == "log"
                         else group.actual)
        top_simple = (np.expm1(top.actual) if target_type == "log"
                      else top.actual)
        gross = float(top_simple.mean())
        excess = gross - float(actual_simple.mean())
        rows.append({"date": date, "IC": ic, "Rank IC": rank_ic,
                     "top5_return": gross, "excess_return": excess,
                     "stocks": len(group)})
    return pd.DataFrame(rows)


def summary(pred: pd.DataFrame, stage: str, target_type: str = "log") -> dict:
    w = weekly_metrics(pred, target_type)
    if w.empty:
        raise ValueError(f"No evaluable weeks for {stage}")

    def mean_ir(s):
        x = s.dropna()
        std = x.std(ddof=1)
        return float(x.mean() / std) if len(x) >= 2 and std > 0 else np.nan

    # Geometric annualization is undefined if a return is <= -100%; do not
    # silently clip an implausible target series or mislabel a log return.
    gross = w.top5_return.to_numpy()
    annual = (float(np.prod(1 + gross) ** (52 / len(gross)) - 1)
              if np.all(gross > -1) else np.nan)
    benchmark = gross - w.excess_return.to_numpy()
    benchmark_annual = (float(np.prod(1 + benchmark) ** (52 / len(benchmark)) - 1)
                        if np.all(benchmark > -1) else np.nan)
    valid_rank = w["Rank IC"].dropna()
    positive_ratio = float((valid_rank > 0).mean()) if len(valid_rank) else np.nan
    return {"Model Name": "Random Forest",
            "Dataset": stage, "IC": w.IC.mean(), "ICIR": mean_ir(w.IC),
            "Rank IC": w["Rank IC"].mean(), "Rank ICIR": mean_ir(w["Rank IC"]),
            "Positive Rank IC Ratio": positive_ratio,
            "Valid Rank IC Weeks": len(valid_rank),
            "Annualized Return": annual,
            "Benchmark Annualized Return": benchmark_annual,
            "Excess Annualized Return": annual - benchmark_annual,
            "Information Ratio": mean_ir(w.excess_return) * np.sqrt(52),
            "Weeks": len(w), "Observations": len(pred),
            "RMSE": float(np.sqrt(np.mean((pred.prediction - pred.actual) ** 2)))}


METRIC_COLUMNS = ["RMSE", "IC", "ICIR", "Rank IC", "Rank ICIR", "Positive Rank IC Ratio",
                  "Annualized Return", "Benchmark Annualized Return",
                  "Excess Annualized Return", "Information Ratio", "Weeks",
                  "Valid Rank IC Weeks", "Observations"]


def rank_candidates(results):
    """Passing candidates first, then excess CAGR; candidate ID breaks ties."""
    scores = pd.DataFrame(results).copy()
    scores["Pass Screening"] = ((scores["Rank IC"] > 0)
        & (scores["Rank ICIR"] >= 0.05)
        & (scores["Positive Rank IC Ratio"] >= 0.50))
    scores["Finite Return"] = np.isfinite(scores["Excess Annualized Return"])
    return scores.sort_values(
        ["Finite Return", "Pass Screening", "Excess Annualized Return", "candidate"],
        ascending=[False, False, False, True], na_position="last")


def select_candidates(scores, count):
    eligible = scores.loc[scores["Finite Return"]]
    if len(eligible) < count:
        raise ValueError(f"Need {count} candidates with finite excess annualized returns")
    chosen = eligible.head(count).copy()
    passing = int(chosen["Pass Screening"].sum())
    print(f"Selected {count}: {passing} passed screening; "
          f"{count - passing} fallback candidate(s) ranked by excess annualized return.", flush=True)
    return chosen


def training_rows(df, start, cutoff, target):
    subset = df[(df.date >= start) & (df.date < cutoff) &
                (df.label_end_date <= cutoff) & df[target].notna()].copy()
    if subset.empty:
        raise ValueError(f"Empty eligible training window: {start} to {cutoff}")
    return subset


def evaluation_rows(df, start, stop, target, stage_stop):
    subset = df[(df.date >= start) & (df.date < stop) &
                (df.label_end_date <= stage_stop) & df[target].notna()].copy()
    return subset


def stage_predictions(df, start, stop, features, target, params, mode,
                      keep_models=False, rolling_years=6, max_blocks=None):
    """Refit preprocessing together with each forest; never use unknown labels.

    Four-week cadence restarts at each stage boundary. Fixed mode fits at the
    calendar boundary, before the first prediction date. Labels ending exactly
    at fit time are assumed observable after that day's market close.
    """
    held = evaluation_rows(df, start, stop, target, stop)
    dates = sorted(held.date.unique())
    if not dates:
        raise ValueError(f"No evaluation dates: {start} to {stop}")
    if mode not in {"rolling", "fixed"}:
        raise ValueError(mode)
    blocks = [dates] if mode == "fixed" else [dates[i:i + REFIT_WEEKS]
                                              for i in range(0, len(dates), REFIT_WEEKS)]
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    parts, folds, importances = [], [], []
    for block in blocks:
        cutoff = start if mode == "fixed" else pd.Timestamp(block[0])
        train_start = START_HISTORY if mode == "fixed" else max(
            START_HISTORY, cutoff - pd.DateOffset(years=rolling_years))
        train = training_rows(df, train_start, cutoff, target)
        evaluate = held[held.date.isin(block)]
        pred, model = fit_predict(train, evaluate, features, target, params, True)
        pred["fit_date"] = cutoff
        parts.append(pred)
        folds.append({"fit_date": cutoff, "training_window_start": train_start,
                      "training_first_date": train.date.min(),
                      "training_last_date": train.date.max(),
                      "latest_training_label": train.label_end_date.max(),
                      "training_rows": len(train),
                      "prediction_first_date": evaluate.date.min(),
                      "prediction_last_date": evaluate.date.max(),
                      "prediction_weeks": len(block)})
        if keep_models:
            importances.append(model.feature_importances_)
    return (pd.concat(parts, ignore_index=True), pd.DataFrame(folds),
            np.mean(importances, axis=0) if importances else None)


def read_feature_sets(path=None):
    if path is None:
        return {key: list(features) for key, features in FEATURE_SETS.items()}
    table = screening_table(path)
    priority = table.loc[table.Decision == PRIORITY_LABEL, "Factor"].tolist()
    all_features = table.Factor.tolist()
    macro = table.loc[table.Decision == MACRO_LABEL, "Factor"].tolist()
    if len(all_features) != 45 or len(set(all_features)) != 45 or len(macro) != 4:
        raise ValueError("Expected 45 distinct workbook factors, including four macro factors")
    if set(priority) & set(macro) or len(set(priority)) != 7:
        raise ValueError("Expected seven distinct non-macro priority factors")
    if set(all_features) & {"target", "target2", "date", "ticker"}:
        raise ValueError("Targets/identifiers cannot be predictors")
    return {"7": priority, "7_macro": priority + macro, "45": all_features}


def run_experiment(df, features, feature_set, design_name, args, grid, output):
    design = TRAINING_DESIGNS[design_name]
    mode, years, dev_start = design["mode"], design["years"], design["dev_start"]
    output.mkdir(parents=True, exist_ok=True)
    def predict(start, stop, params, keep_models=False):
        return stage_predictions(df, start, stop, features, args.target_column,
            params, mode, keep_models=keep_models, rolling_years=years,
            max_blocks=2 if args.quick else None)
    initial = training_rows(df, START_HISTORY, dev_start, args.target_column)
    dev_predictions, dev_folds, dev_results = {}, {}, []
    for cid, parameters in enumerate(grid, 1):
        pred, folds, _ = predict(dev_start, START_VALID, parameters)
        dev_predictions[cid], dev_folds[cid] = pred, folds
        dev_results.append({"candidate": cid, **parameters, **summary(pred, "Development", args.target_type)})
        print(f"Random Forest {design_name}/{feature_set}: candidate {cid}/{len(grid)}", flush=True)
    dev_scores = rank_candidates(dev_results)
    dev_scores.to_csv(output / "development_scores.csv", index=False, encoding="utf-8-sig")
    status = {"model": f"random_forest_{feature_set}_{design_name}", "mode": design_name,
              "feature_set": feature_set, "status": "no_finite_candidate"}
    shortlist_count = min(SHORTLIST_SIZE, len(grid)) if args.quick else SHORTLIST_SIZE
    if dev_scores["Finite Return"].sum() < shortlist_count:
        save_json(output / "locked_selection.json", status)
        return status, None, []
    shortlist = select_candidates(dev_scores, shortlist_count)
    valid_results, valid_predictions, valid_folds = [], {}, {}
    for cid in shortlist.candidate.astype(int):
        parameters = grid[cid - 1]
        pred, folds, _ = predict(START_VALID, START_TEST, parameters)
        valid_predictions[cid], valid_folds[cid] = pred, folds
        valid_results.append({"candidate": cid, **parameters, **summary(pred, "Validation", args.target_type)})
    valid_scores = rank_candidates(valid_results)
    valid_scores.to_csv(output / "validation_scores.csv", index=False, encoding="utf-8-sig")
    if not valid_scores["Finite Return"].any():
        save_json(output / "locked_selection.json", status)
        return status, None, []
    best = select_candidates(valid_scores, 1).iloc[0]
    cid = int(best.candidate)
    parameters = grid[cid - 1]
    final_pred, folds, importance = predict(START_TEST, END_TEST, parameters, True)
    # Only the selected candidate's in-sample diagnostics are required in the
    # final comparison; fitting all candidates twice had no selection effect.
    initial_pred = fit_predict(initial, initial, features, args.target_column, parameters)
    initial_metric = summary(initial_pred, "Initial Training (IN-SAMPLE)", args.target_type)
    stage_predictions_list, metrics, fold_parts = [], [initial_metric], []
    for stage, pred, fit_schedule in [
            ("development", dev_predictions[cid], dev_folds[cid]),
            ("validation", valid_predictions[cid], valid_folds[cid]),
            ("final_test", final_pred, folds)]:
        pred = pred.assign(stage=stage, candidate=cid)
        stage_predictions_list.append(standard_predictions(pred, model="random_forest",
            mode=design_name, feature_set=feature_set, target_type=args.target_type))
        metrics.append(summary(pred, stage, args.target_type))
        fold_parts.append(fit_schedule.assign(stage=stage))
    predictions = pd.concat(stage_predictions_list, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_parts, ignore_index=True).to_csv(output / "folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    metrics = [{**m, "model": status["model"], "mode": design_name,
                "feature_set": feature_set, "stage": m["Dataset"], "candidate": cid} for m in metrics]
    pd.DataFrame(metrics).to_csv(output / "performance_metrics.csv", index=False, encoding="utf-8-sig")
    status.update(status="selected", candidate=cid, parameters=parameters,
                  validation_passed_gate=bool(best["Pass Screening"]),
                  development_shortlist_fallbacks=int((~shortlist["Pass Screening"]).sum()))
    save_json(output / "locked_selection.json", status)
    return status, predictions, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--data-file", dest="data_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "random_forest")
    parser.add_argument("--screening-file", type=Path,
                        default=ROOT / "single_factor_analysis" / "reference" / "Single_Factor_Screening_66_Stocks_EN.xlsx",
                        help="Chinese or English original screening workbook; feature order is preserved")
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--target-column", default="target", help="A one-week forward return only")
    parser.add_argument("--target-type", choices=["log", "simple"], default="log")
    parser.add_argument("--expected-stocks", default=66, type=int)
    parser.add_argument("--mode", choices=["all", "both", "rolling_3y", "rolling_6y", "fixed"], default="all")
    parser.add_argument("--feature-set", choices=["all", "7", "7_macro", "45"], default="all")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: one candidate, eight trees, two blocks per stage; not research results")
    args = parser.parse_args(argv)
    sets = read_feature_sets(args.screening_file)
    df = read_panel(args.data_file, sets["45"], args.target_column, args.universe_file, args.expected_stocks)
    selected = list(sets) if args.feature_set == "all" else [args.feature_set]
    designs = list(TRAINING_DESIGNS) if args.mode in {"all", "both"} else [args.mode]
    for design_name in designs:
        design = TRAINING_DESIGNS[design_name]
        training_rows(df, START_HISTORY, design["dev_start"], args.target_column)
        for stage, start, stop in [("development", design["dev_start"], START_VALID),
                                   ("validation", START_VALID, START_TEST), ("final_test", START_TEST, END_TEST)]:
            rows = evaluation_rows(df, start, stop, args.target_column, stop)
            if rows.empty:
                raise ValueError(f"Empty {design_name}/{stage}")
            print(f"Random Forest {design_name}/{stage}: {len(rows)} rows, {rows.date.nunique()} weeks")
    if args.check_only:
        return {"status": "checked", "rows": len(df)}
    output = output_directory(args.output)
    grid = list(candidate_grid())
    if args.quick:
        grid = [{**grid[0], "n_estimators": 8}]
    save_json(output / "config.json", {
        "source": args.data_file.resolve(), "source_sha256": sha(args.data_file), "quick": args.quick,
        "feature_sets": {key: sets[key] for key in selected}, "training_designs": TRAINING_DESIGNS,
        "selected_designs": designs, "candidate_grid": grid, "random_state": 5703,
        "screening_source": args.screening_file or "original shared factor lists",
        "target_type": args.target_type, "refit_weeks": REFIT_WEEKS,
        "selection": "Development Top5 -> Validation winner, screening-first ranking with fallback",
        "selection_return_metric": "strategy annualized return minus benchmark annualized return",
        "label_rule": "next global observation date; label_end_date <= fit_date and <= stage_stop",
        "rolling_cadence": "four observations; restarts at every stage boundary",
    })
    statuses, predictions, metrics = [], [], []
    for feature_set in selected:
        for design in designs:
            status, pred, metric = run_experiment(df, sets[feature_set], feature_set,
                design, args, grid, output / design / feature_set)
            statuses.append(status)
            if pred is not None:
                predictions.append(pred)
            metrics.extend(metric)
    save_predictions(output / "predictions.csv", predictions, quick_run=args.quick)
    pd.DataFrame(statuses).to_csv(output / "selection_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics, columns=["model", "mode", "feature_set", "stage", "candidate", *METRIC_COLUMNS]).to_csv(
        output / "all_stage_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "run_manifest.json", {"status": "completed", "quick": args.quick,
        "source_sha256": sha(args.data_file), "results": statuses})
    return {"status": "completed", "output": str(output), "predictions": str(output / "predictions.csv")}


if __name__ == "__main__":
    main()
