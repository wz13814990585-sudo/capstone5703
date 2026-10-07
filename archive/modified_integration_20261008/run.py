"""Unified offline entry point for factor research, model training and backtests."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ("ridge", "random_forest", "xgboost", "xgboost_ranker")
MODULES = {"ridge": "ridge.train", "random_forest": "random_forest.train",
           "xgboost": "xgboost_model.train", "xgboost_ranker": "xgboost_ranker.train"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "audit", "factors", "train", "evaluate", "backtest"])
    parser.add_argument("--input", type=Path, default=ROOT / "data/test_45_factors_weekly.csv")
    parser.add_argument("--prices", type=Path,
                        help="Full weekly price panel; otherwise build exact-date prices from --daily-prices")
    parser.add_argument("--daily-prices", type=Path, default=ROOT / "data/clean_basic_data.csv")
    parser.add_argument("--output", type=Path, help="New output directory; defaults to a timestamped run")
    parser.add_argument("--predictions", type=Path, help="Required for standalone evaluate/backtest")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--feature-set", choices=["all", "7", "7_macro", "45"], default="all")
    parser.add_argument("--mode", choices=["all", "rolling", "fixed"], default="all")
    parser.add_argument("--quick", action="store_true", help="Integration check only: reduced candidates/trees/folds")
    parser.add_argument("--selection-end", default="2024-09-01")
    parser.add_argument("--costs-bps", nargs="+", type=float, default=[0, 10, 25])
    return parser.parse_args(argv)


def train_models(args, output):
    import pandas as pd
    pieces = []
    for name in dict.fromkeys(args.models):
        print(f"Training {name} ...", flush=True)
        model_output = output / name
        argv = ["--input", str(args.input), "--output", str(model_output)]
        if args.quick:
            argv.append("--quick")
        if name != "xgboost_ranker":
            argv += ["--feature-set", args.feature_set]
            mode = ("all" if name == "random_forest" else "both") if args.mode == "all" else args.mode
            if name == "random_forest" and mode == "rolling":
                mode = "rolling_3y"
            argv += ["--mode", mode]
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                importlib.import_module(MODULES[name]).main(argv)
        predictions = pd.read_csv(model_output / "predictions.csv", low_memory=False)
        if not predictions.empty:
            pieces.append(predictions)
        print(f"  {len(predictions):,} prediction rows; details: {name}.log", flush=True)
    if pieces:
        combined = pd.concat(pieces, ignore_index=True)
    else:
        from model_training.artifacts import PREDICTION_COLUMNS
        combined = pd.DataFrame(columns=PREDICTION_COLUMNS)
    combined["quick_run"] = args.quick
    combined.to_csv(output / "predictions.csv", index=False)
    return combined


def quick_by_model(predictions):
    if "quick_run" not in predictions:
        return {}
    flags = predictions["quick_run"].astype(str).str.lower().isin(["true", "1"])
    return flags.groupby(predictions["model"]).any().to_dict()


def evaluate(predictions, output):
    import pandas as pd
    from model_evaluation import evaluate_predictions
    output.mkdir(parents=True, exist_ok=True)
    summary, weekly = evaluate_predictions(predictions)
    summary["quick_run"] = summary["model"].map(quick_by_model(predictions)).astype("boolean")
    summary.to_csv(output / "summary.csv", index=False)
    weekly.to_csv(output / "weekly_metrics.csv", index=False)
    if "stage" in predictions:
        stages = []
        for stage, group in predictions.groupby("stage", sort=True):
            stage_summary, _ = evaluate_predictions(group)
            stage_summary["quick_run"] = stage_summary["model"].map(quick_by_model(group)).astype("boolean")
            stage_summary.insert(1, "stage", stage)
            stages.append(stage_summary)
        if stages:
            pd.concat(stages, ignore_index=True).to_csv(output / "summary_by_stage.csv", index=False)
    return summary


def backtest(predictions, args, output):
    import pandas as pd
    from backtesting import run_backtest
    output.mkdir(parents=True, exist_ok=True)
    result = run_backtest(predictions, pd.read_csv(args.prices), costs_bps=tuple(args.costs_bps))
    result.summary["quick_run"] = result.summary["model"].map(quick_by_model(predictions)).astype("boolean")
    result.save(output)
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command in {"evaluate", "backtest"} and args.predictions is None:
        raise SystemExit("--predictions is required for evaluate/backtest")
    args.input = args.input.resolve()
    args.prices = args.prices.resolve() if args.prices is not None else None
    args.daily_prices = args.daily_prices.resolve()
    output = (args.output or ROOT / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**vars(args), "output": str(output), "status": "running", "python": sys.executable}
    def save_manifest():
        (output / "run_manifest.json").write_text(json.dumps(manifest, default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    print(f"Output: {output}", flush=True)
    if args.quick:
        print("QUICK RUN: reduced training; results are for integration checks only.", flush=True)
    try:
        if args.command in {"all", "audit", "train", "factors"}:
            from data_processing import audit_weekly_data
            manifest["data_audit"] = audit_weekly_data(args.input, output / "data_audit")
            print("Data audit completed.", flush=True)
        if args.command in {"all", "factors"}:
            from single_factor_analysis import run_screening
            run_screening(args.input, output / "single_factor_analysis", selection_end=args.selection_end)
            print("Single-factor analysis completed.", flush=True)
        if args.command in {"all", "train"}:
            predictions = train_models(args, output)
        elif args.command in {"evaluate", "backtest"}:
            import pandas as pd
            predictions = pd.read_csv(args.predictions, low_memory=False)
            # Legacy single-model files are also supported by the downstream schema.
            if "model" not in predictions:
                predictions["model"] = "model"
            manifest["source_quick_metadata"] = "present" if "quick_run" in predictions else "not_provided"
            if args.quick:
                predictions["quick_run"] = True
            manifest["quick"] = any(quick_by_model(predictions).values()) if "quick_run" in predictions else None
        has_predictions = args.command in {"all", "train", "evaluate", "backtest"}
        if has_predictions:
            manifest["selected_models"] = sorted(predictions["model"].unique().tolist()) if not predictions.empty else []
            manifest["model_selection"] = "selected" if not predictions.empty else "no_selected_models"
            if predictions.empty:
                print("No candidates passed selection. Empty predictions saved; evaluation/backtest skipped.", flush=True)
        if args.command in {"all", "evaluate"} and not predictions.empty:
            evaluate(predictions, output / "model_evaluation")
            print("Model evaluation completed.", flush=True)
        if args.command in {"all", "backtest"} and not predictions.empty:
            if args.prices is None:
                from data_processing import build_weekly_prices
                args.prices = output / "data_audit" / "weekly_prices.csv"
                manifest["price_audit"] = build_weekly_prices(args.daily_prices, args.prices, calendar_path=args.input)
                manifest["prices"] = str(args.prices)
            backtest(predictions, args, output / "backtesting")
            print("Portfolio backtests completed.", flush=True)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    manifest["status"] = "completed"
    save_manifest()
    print(f"Completed: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
