import argparse
import json
from pathlib import Path

from data_processing.paths import DEFAULT_WEEKLY_FILE, PROJECT_ROOT
from .screening import run_screening


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recompute single-factor screening before a holdout cutoff.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WEEKLY_FILE)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "single_factor_analysis")
    parser.add_argument("--selection-end", default="2024-09-01", help="Exclusive signal and label-end cutoff")
    parser.add_argument("--exploratory-full-sample", action="store_true", help="Use all verifiable labels for exploration; not holdout-safe selection")
    args = parser.parse_args(argv)
    result = run_screening(args.input, args.output, args.selection_end, exploratory_full_sample=args.exploratory_full_sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
