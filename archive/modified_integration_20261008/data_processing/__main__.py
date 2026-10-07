"""Run with python -m data_processing; audit is the network-free default."""
import argparse
import json
from pathlib import Path

from .panel import audit_weekly_data
from .paths import DATA_DIR, DEFAULT_WEEKLY_FILE


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weekly data preparation; downloads are explicit commands.")
    subs = parser.add_subparsers(dest="command")
    audit = subs.add_parser("audit", help="Validate the existing weekly CSV without network access")
    audit.add_argument("--input", type=Path, default=DEFAULT_WEEKLY_FILE)
    audit.add_argument("--output", type=Path)
    prices = subs.add_parser("prices", help="Build exact-date weekly prices from local daily data without factor filtering")
    prices.add_argument("--input", type=Path, default=DATA_DIR / "clean_basic_data.csv")
    prices.add_argument("--calendar", type=Path, default=DEFAULT_WEEKLY_FILE)
    prices.add_argument("--output", type=Path, default=DATA_DIR / "weekly_prices.csv")
    download = subs.add_parser("download", help="Explicitly download daily Yahoo Finance prices")
    download.add_argument("--data-dir", type=Path, default=DATA_DIR)
    download.add_argument("--start", default="2015-09-01")
    download.add_argument("--end", default="2026-09-01", help="Exclusive download end date")
    build = subs.add_parser("build", help="Rebuild the 35 internal factors from existing daily prices")
    build.add_argument("--input", type=Path, default=DATA_DIR / "clean_basic_data.csv")
    build.add_argument("--output", type=Path, default=DATA_DIR / "test_data_weekly.csv")
    build.add_argument("--start", default="2016-09-01")
    build.add_argument("--end", default="2026-09-30")
    external = subs.add_parser("external", help="Explicitly fetch SEC/FRED external factors")
    external.add_argument("--input", type=Path, default=DATA_DIR / "clean_basic_data.csv")
    external.add_argument("--technical", type=Path, default=DATA_DIR / "test_data_weekly.csv")
    external.add_argument("--output", type=Path, default=DEFAULT_WEEKLY_FILE)
    args = parser.parse_args(argv)
    if args.command in (None, "audit"):
        result = audit_weekly_data(getattr(args, "input", DEFAULT_WEEKLY_FILE), getattr(args, "output", None))
    elif args.command == "prices":
        from .prices import build_weekly_prices
        result = build_weekly_prices(args.input, args.output, args.calendar)
    elif args.command == "download":
        from .fetch_and_clean import download_prices
        result = download_prices(args.data_dir, args.start, args.end)
    elif args.command == "build":
        from .build_technical_factors import build_technical_factors
        result = build_technical_factors(args.input, args.output, args.start, args.end)
    else:
        from .build_external_factors import build_external_factors
        result = build_external_factors(args.input, args.technical, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
