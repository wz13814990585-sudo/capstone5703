"""Invoke existing original validation functions without rewriting their source."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=["ridge", "xgboost_ranker"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_research_module", args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.module == "ridge":
        raw, panel, _, excluded, error = module.load_panel(args.input)
        print(f"Original Ridge validation: input={len(raw)}, usable={len(panel)}, excluded={len(excluded)}, label_error={error}")
    else:
        _, audit = module.load_data(args.input)
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
