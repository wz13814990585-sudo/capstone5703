"""Nine Random Forest experiments: 7, 11 and 45 factors x three training designs.
Rolling 3y: initial 3 years; Development/Validation/Final Test: 5/1/1 years.
Rolling 6y and fixed: initial 6 years; subsequent stages: 2/1/1 years.
Rolling: trailing three or six calendar years, refitted every four observation weeks.
Fixed: all eligible history, one fit at each stage start per candidate.
Development shortlists five of 18 candidates; Validation chooses one.
Final Test freezes parameters. Results are printed.
Full-sample factor screening can introduce selection look-ahead.
"""
from __future__ import annotations

import argparse
import warnings
import shutil
import textwrap
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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


def read_priority_features(path: Path) -> list[str]:
    table = pd.read_excel(path, sheet_name=SCREENING_SHEET, header=4)
    table.columns = [str(c).strip() for c in table.columns]
    if not {"Factor", "Decision"}.issubset(table.columns):
        raise ValueError("Screening workbook needs Factor and Decision columns")
    decisions = table.Decision.astype(str).str.strip()
    counts = {
        "priority": int((decisions == PRIORITY_LABEL).sum()),
        "duplicate": int((decisions == DUPLICATE_LABEL).sum()),
        "macro": int((decisions == MACRO_LABEL).sum()),
        "backup": int(decisions.str.contains(BACKUP_TEXT).sum()),
    }
    if counts != {"priority": 7, "duplicate": 4, "macro": 4, "backup": 30}:
        raise ValueError(f"Unexpected screening counts: {counts}")
    features = table.loc[decisions == PRIORITY_LABEL, "Factor"].astype(str).str.strip().tolist()
    if len(features) != 7 or len(set(features)) != 7:
        raise ValueError(f"Expected seven distinct priority factors, found {features}")
    return features


def read_panel(path: Path, features: list[str], target: str, universe: Path | None,
               expected_stocks: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(["date", "ticker", target, *features]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in weekly data: {sorted(missing)}")
    df["date"] = pd.to_datetime(df.date, errors="raise")
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


def prep_fit(frame: pd.DataFrame, features: list[str]):
    x = frame[features].to_numpy(float)
    med = np.nanmedian(x, axis=0)
    if not np.isfinite(med).all():
        raise ValueError("Some factors have no finite values in the training window")
    x = np.where(np.isfinite(x), x, med)
    lo, hi = np.quantile(x, [0.01, 0.99], axis=0)
    return np.clip(x, lo, hi), (med, lo, hi)


def prep_apply(frame: pd.DataFrame, features: list[str], prep):
    med, lo, hi = prep
    x = frame[features].to_numpy(float)
    return np.clip(np.where(np.isfinite(x), x, med), lo, hi)


def candidate_grid():
    for depth, leaf, max_features in product([4, 6, 8], [10, 20, 30], ["sqrt", 0.5]):
        yield {"max_depth": depth, "min_samples_leaf": leaf,
               "max_features": max_features, "n_estimators": 100}


def fit_predict(train, evaluate, features, target, params, return_model=False):
    x, prep = prep_fit(train, features)
    model = RandomForestRegressor(**params, random_state=5703, n_jobs=-1)
    model.fit(x, train[target].to_numpy(float))
    pred = evaluate[["date", "ticker", target]].rename(columns={target: "actual"}).copy()
    pred["prediction"] = model.predict(prep_apply(evaluate, features, prep))
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


PARAMETER_COLUMNS = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
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


def output_width():
    return max(40, min(100, shutil.get_terminal_size((88, 24)).columns - 2))


def display_value(name, value):
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        if name == "Excess Annualized Return":
            return f"{value * 100:.2f} pp"
        if name in {"Positive Rank IC Ratio", "Annualized Return",
                    "Benchmark Annualized Return"}:
            return f"{value:.2%}"
        if name in {"Weeks", "Valid Rank IC Weeks", "Observations"}:
            return str(int(value))
        return f"{value:.6f}"
    return str(value)


def show_heading(title):
    width = output_width()
    print("\n" + "=" * width, flush=True)
    print(textwrap.fill(title, width=width), flush=True)
    print("=" * width, flush=True)


def show_fields(items):
    """Aligned vertical fields, with long values wrapped inside the terminal."""
    items = list(items)
    width = output_width()
    label_width = min(max((len(str(k)) for k, _ in items), default=0), width // 2)
    for name, value in items:
        labels = textwrap.wrap(str(name), width=label_width) or [""]
        for label in labels[:-1]:
            print(label, flush=True)
        prefix = f"{labels[-1]:<{label_width}} : "
        lines = textwrap.wrap(display_value(name, value),
                              width=max(1, width - len(prefix))) or [""]
        print(prefix + lines[0], flush=True)
        for line in lines[1:]:
            print(" " * len(prefix) + line, flush=True)


def show_parameters(candidate, params):
    print(f"\ncandidate{candidate}:", flush=True)
    show_fields((name, params[name]) for name in PARAMETER_COLUMNS)


def show_metrics(result):
    show_fields((name, result[name]) for name in METRIC_COLUMNS)


def show_table(title, frame):
    """Split wide tables into column blocks, repeating the identifying column."""
    show_heading(title)
    if frame.empty:
        print("No rows.", flush=True)
        return
    columns = list(frame.columns)
    values = [[display_value(c, v) for c, v in zip(columns, row)]
              for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(row[i]) for row in values))
              for i, c in enumerate(columns)]
    blocks, block = [], [0]
    for i in range(1, len(columns)):
        if sum(widths[j] for j in block + [i]) + 2 * len(block) > output_width():
            if len(block) > 1:
                blocks.append(block)
            block = [0]
        block.append(i)
    blocks.append(block)
    for block in blocks:
        if sum(widths[j] for j in block) + 2 * (len(block) - 1) > output_width():
            for row in values:
                show_fields((columns[j], row[j]) for j in block)
                print()
            continue
        print("  ".join(str(columns[j]).ljust(widths[j]) for j in block))
        print("  ".join("-" * widths[j] for j in block))
        for row in values:
            print("  ".join(row[j].ljust(widths[j]) for j in block))
        print(flush=True)


def show_candidates(title, scores):
    # Use short column labels so comparison tables fit ordinary terminals.
    columns = ["candidate", *PARAMETER_COLUMNS, "Pass Screening", *METRIC_COLUMNS]
    labels = {"candidate": "Candidate", "n_estimators": "Trees",
              "max_depth": "Depth", "min_samples_leaf": "Min leaf",
              "max_features": "Max features", "Pass Screening": "Screen",
              "Positive Rank IC Ratio": "Positive IC %",
              "Annualized Return": "Annual return",
              "Benchmark Annualized Return": "Benchmark",
              "Excess Annualized Return": "Excess (pp)",
              "Information Ratio": "Info ratio", "Valid Rank IC Weeks": "Valid IC weeks"}
    display = scores[columns].copy()
    for col in columns:
        display[col] = display[col].map(lambda value: display_value(col, value))
    display["candidate"] = display["candidate"].map(lambda value: f"candidate{value}")
    show_table(title, display.rename(columns=labels))


def show_stage_comparison(compare):
    # Metrics are rows; the four stages are columns, split if space is limited.
    stage_labels = ["Train (IS)", "Development", "Validation", "Final Test"]
    display = pd.DataFrame({"Metric": METRIC_COLUMNS})
    for label, (_, result) in zip(stage_labels, compare.iterrows()):
        display[label] = [display_value(name, result[name]) for name in METRIC_COLUMNS]
    show_table("FOUR-STAGE COMPARISON - INITIAL TRAINING IS IN-SAMPLE", display)
    print("Returns and positive IC ratios are percentages; excess return is in pp.")


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
                      keep_models=False, rolling_years=6):
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


def read_feature_sets(path):
    priority = read_priority_features(path)
    table = pd.read_excel(path, sheet_name=SCREENING_SHEET, header=4)
    table.columns = table.columns.astype(str).str.strip()
    table = table.loc[table.Factor.notna()].copy()
    all_features = table.Factor.astype(str).str.strip().tolist()
    macro = table.loc[table.Decision.astype(str).str.strip() == MACRO_LABEL,
                      "Factor"].astype(str).str.strip().tolist()
    if len(all_features) != 45 or len(set(all_features)) != 45 or len(macro) != 4:
        raise ValueError("Expected 45 distinct workbook factors, including four macro factors")
    if set(priority) & set(macro):
        raise ValueError("Priority and macro factors overlap")
    if set(all_features) & {"target", "target2", "date", "ticker"}:
        raise ValueError("Targets/identifiers cannot be predictors")
    return {"7 factors": priority, "7 + 4 macro": priority + macro,
            "45 factors": all_features}


def run_experiment(df, features, name, design_name, args, grid):
    design = TRAINING_DESIGNS[design_name]
    mode, years, dev_start = design["mode"], design["years"], design["dev_start"]
    show_heading(f"EXPERIMENT: {name} / {design_name.upper()}")
    print(f"Initial history: {START_HISTORY.date()} to {dev_start.date()}")
    print(f"Development: {dev_start.date()} to {START_VALID.date()}")
    print("Features:", ", ".join(features), flush=True)
    initial = training_rows(df, START_HISTORY, dev_start, args.target_column)
    initial_results, dev_predictions, dev_folds, dev_results = {}, {}, {}, []
    # STAGE 1: design-specific three- or six-year initial history; diagnostics are in-sample only.
    show_heading("STAGE 1 - INITIAL TRAINING (IN-SAMPLE)")
    for cid, params in enumerate(grid, 1):
        pred = fit_predict(initial, initial, features, args.target_column, params)
        initial_results[cid] = summary(pred, "Initial Training (IN-SAMPLE)", args.target_type)
        show_parameters(cid, params)
        show_metrics(initial_results[cid])
    # STAGE 2: Development independently tunes each of the nine experiments.
    show_heading("STAGE 2 - DEVELOPMENT: ALL 18 CANDIDATES")
    for cid, params in enumerate(grid, 1):
        show_parameters(cid, params)
        pred, folds, _ = stage_predictions(df, dev_start, START_VALID, features,
                                          args.target_column, params, mode, rolling_years=years)
        dev_predictions[cid], dev_folds[cid] = pred, folds
        result = summary(pred, "Development", args.target_type)
        dev_results.append({"candidate": cid, **params, **result})
        show_metrics(result)
    dev_scores = rank_candidates(dev_results)
    show_candidates("DEVELOPMENT CANDIDATES", dev_scores)
    shortlist = select_candidates(dev_scores, SHORTLIST_SIZE)
    show_candidates("DEVELOPMENT TOP 5", shortlist)
    # STAGE 3: compare the shortlisted parameters and freeze the winner.
    show_heading("STAGE 3 - VALIDATION: TOP 5 CANDIDATES")
    valid_results, valid_predictions, valid_folds = [], {}, {}
    for cid in shortlist.candidate.astype(int):
        params = grid[cid - 1]
        show_parameters(cid, params)
        pred, folds, _ = stage_predictions(df, START_VALID, START_TEST, features,
                                          args.target_column, params, mode, rolling_years=years)
        valid_predictions[cid], valid_folds[cid] = pred, folds
        result = summary(pred, "Validation", args.target_type)
        valid_results.append({"candidate": cid, **params, **result})
        show_metrics(result)
    valid_scores = rank_candidates(valid_results)
    show_candidates("VALIDATION CANDIDATES", valid_scores)
    best = select_candidates(valid_scores, 1).iloc[0]
    cid = int(best.candidate)
    params = grid[cid - 1]
    show_heading("FROZEN PARAMETERS")
    show_parameters(cid, params)
    # STAGE 4: no parameter or feature selection on Final Test outcomes.
    show_heading("STAGE 4 - FINAL TEST")
    pred, folds, importance = stage_predictions(df, START_TEST, END_TEST, features,
                                    args.target_column, params, mode, True, rolling_years=years)
    final = summary(pred, "Final Test", args.target_type)
    results = [initial_results[cid], summary(dev_predictions[cid], "Development", args.target_type),
               summary(valid_predictions[cid], "Validation", args.target_type), final]
    show_stage_comparison(pd.DataFrame(results))
    show_table("SELECTED CANDIDATE: DEVELOPMENT FIT SCHEDULE", dev_folds[cid])
    show_table("SELECTED CANDIDATE: VALIDATION FIT SCHEDULE", valid_folds[cid])
    show_table("FINAL TEST FIT SCHEDULE", folds)
    show_table("FINAL TEST FEATURE IMPORTANCE (MEAN ACROSS FITS)",
               pd.DataFrame({"feature": features, "importance": importance})
               .sort_values("importance", ascending=False))
    passing = bool(final["Rank IC"] > 0 and final["Rank ICIR"] >= .05
                   and final["Positive Rank IC Ratio"] >= .5)
    return {"Experiment": f"{name} / {design_name}", "Factors": len(features),
            "Selected candidate": f"candidate{cid}", **params,
            "Pass Screening": passing, **final}


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path,
                        default=script_dir / "test_45_factors_weekly.csv")
    parser.add_argument("--screening-file", type=Path,
                        default=script_dir / "Single_Factor_Screening_66_Stocks.xlsx")
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--target-column", default="target",
                        help="A one-week forward return only")
    parser.add_argument("--target-type", choices=["log", "simple"], default="log")
    parser.add_argument("--expected-stocks", default=66, type=int)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate inputs, factor sets and stage schedules without fitting")
    args = parser.parse_args()
    sets = read_feature_sets(args.screening_file)
    df = read_panel(args.data_file, sets["45 factors"], args.target_column,
                    args.universe_file, args.expected_stocks)
    show_heading("NINE EXPERIMENTS - THREE TRAINING DESIGNS")
    print("Rolling 3y: Initial 2016-09-01 to 2019-09-01 (3 years).")
    print("Rolling 3y: Development 2019-09-01 to 2024-09-01 (5 years); ratio 5:1:1.")
    print("Rolling 6y / Fixed: Initial 2016-09-01 to 2022-09-01 (6 years).")
    print("Rolling 6y / Fixed: Development 2022-09-01 to 2024-09-01 (2 years); ratio 2:1:1.")
    print("All designs: Validation 2024-09-01 to 2025-09-01 (1 year).")
    print("All designs: Final Test 2025-09-01 to 2026-09-01 (1 year).")
    print("Rolling: trailing 3 or 6 years; refit every 4 observed weeks; restart at stage boundaries.")
    print("Fixed: all known history; one fit at each stage start per candidate.")
    print("Each experiment: Development Top 5 -> Validation winner -> frozen Final Test parameters.")
    print("Development periods differ by design; Final Test uses common dates.")
    print("Screen: mean Rank IC > 0; Rank ICIR >= 0.05; positive Rank IC ratio >= 50%.")
    print("Rank passing candidates by Top-5 excess annualized return; fallback if fewer than five pass.")
    print("Target:", args.target_column, args.target_type, "one-week return; stocks:", df.ticker.nunique())
    print("45 factors includes the four macro factors; target/target2 are excluded.")
    print("NOTE: Full-sample factor screening can leak future selection information.")
    print("Top-5 and equal-weight benchmark returns exclude transaction costs.")
    for name, features in sets.items():
        print(f"{name} ({len(features)}): {', '.join(features)}")
    for design_name, design in TRAINING_DESIGNS.items():
        initial = training_rows(df, START_HISTORY, design["dev_start"], args.target_column)
        print(f"{design_name} initial: {initial.date.nunique()} weeks, {len(initial)} rows")
        for stage, start, stop in [("Development", design["dev_start"], START_VALID),
                                   ("Validation", START_VALID, START_TEST),
                                   ("Final Test", START_TEST, END_TEST)]:
            rows = evaluation_rows(df, start, stop, args.target_column, stop)
            if rows.empty:
                raise ValueError(f"Empty {design_name} / {stage}")
            print(f"{design_name} / {stage}: {rows.date.nunique()} evaluable weeks, {len(rows)} rows")
    if args.check_only:
        print("Input validation passed. No models fitted.")
        return
    grid = list(candidate_grid())
    for cid, params in enumerate(grid, 1):
        show_parameters(cid, params)
    results = []
    for name, features in sets.items():
        for design_name in TRAINING_DESIGNS:
            results.append(run_experiment(df, features, name, design_name, args, grid))
    columns = ["Experiment", "Factors", "Selected candidate", *PARAMETER_COLUMNS,
               "Pass Screening", *METRIC_COLUMNS]
    show_table("NINE-EXPERIMENT FINAL TEST COMPARISON (NOT A NEW TUNING STAGE)",
               pd.DataFrame(results)[columns])


if __name__ == "__main__":
    main()
