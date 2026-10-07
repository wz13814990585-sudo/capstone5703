"""python -m model_evaluation --predictions FILE --output DIRECTORY"""
import argparse
from pathlib import Path

import pandas as pd

from .metrics import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weekly model predictions")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--score-kind", choices=["return", "rank"])
    args = parser.parse_args()
    summary, weekly = evaluate_predictions(pd.read_csv(args.predictions), score_kind=args.score_kind)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    weekly.to_csv(args.output / "weekly_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
