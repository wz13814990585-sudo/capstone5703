"""python -m backtesting --predictions FILE --prices FILE --output DIRECTORY"""
import argparse
from pathlib import Path

import pandas as pd

from .engine import run_backtest
from .portfolio_strategies import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run price-verified weekly portfolio backtests")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prices", type=Path, help="Explicit complete weekly prices; otherwise build from daily data")
    parser.add_argument("--daily-prices", type=Path, default=ROOT / "data/clean_basic_data.csv")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/test_45_factors_weekly.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/backtest"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    parser.add_argument("--costs-bps", nargs="+", type=float, default=[0, 10, 25])
    parser.add_argument("--score-kind", choices=["return", "rank"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-delay-weeks", type=int, default=0)
    args = parser.parse_args()
    price_path = args.prices
    if price_path is None:
        from data_processing import build_weekly_prices
        price_path = args.output / "weekly_prices.csv"
        build_weekly_prices(args.daily_prices, price_path, calendar_path=args.calendar)
    result = run_backtest(pd.read_csv(args.predictions), pd.read_csv(price_path),
                          strategies=args.strategies, costs_bps=args.costs_bps,
                          windows={"Selected": (args.start, args.end)} if args.start or args.end else None,
                          score_kind=args.score_kind, initial_delay_weeks=args.initial_delay_weeks)
    result.save(args.output)
    print(result.summary.to_string(index=False))
    print(f"Saved results and {len(result.audit)} audit rows to {args.output}")


if __name__ == "__main__":
    main()
