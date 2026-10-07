"""Stage and launch byte-identical original scripts; never alter research logic."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_MODULES = ["ridge", "random_forest", "xgboost", "xgboost_ranker"]
SCRIPTS = {
    "ridge": "ridge/train_ridge_comparison.py",
    "random_forest": "random_forest/random_forest.py",
    "xgboost": "xgboost_model/train_xgboost_comparison(2).py",
    "xgboost_ranker": "xgboost_ranker/xgboost_ranker.py",
    "technical_factors": "data_processing/build_technical_factors.py",
    "external_factors": "data_processing/build_external_factors.py",
    "download": "data_processing/fetch_and_clean.py",
    "backtest": "backtesting/full_history_backtest.py",
}
DEPENDENCIES = {
    "ridge": ["numpy", "pandas"],
    "random_forest": ["numpy", "pandas", "scipy", "sklearn", "openpyxl"],
    "xgboost": ["numpy", "pandas", "scipy", "sklearn", "xgboost"],
    "xgboost_ranker": ["numpy", "pandas", "scipy", "sklearn", "xgboost"],
    "technical_factors": ["numpy", "pandas", "exchange_calendars"],
    "external_factors": ["numpy", "pandas", "exchange_calendars"],
    "download": ["pandas", "yfinance"],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def verify_originals():
    manifest = json.loads((ROOT / "integration/source_manifest.json").read_text(encoding="utf-8"))
    rows = []
    with zipfile.ZipFile(ROOT / "archive/original_source_before_integration.zip") as backup:
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            actual = sha(path) if path.is_file() else None
            matches = actual == entry["sha256"]
            if "backup_member" in entry:
                matches = matches and path.is_file() and path.read_bytes() == backup.read(entry["backup_member"])
            rows.append({**entry, "current_sha256": actual, "matches": matches})
    return {"protocol": "integration_only", "passed": all(row["matches"] for row in rows),
            "original_python_files": sum(row["kind"] == "original_source" for row in rows), "files": rows}


def dependency_status(python, modules):
    names = sorted({name for module in modules for name in DEPENDENCIES.get(module, [])})
    code = "import importlib.util,json; names=" + repr(names) + "; print(json.dumps({n:importlib.util.find_spec(n) is not None for n in names}))"
    result = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect Python environment: {result.stderr.strip()}")
    return json.loads(result.stdout)


def stage_original(module, output):
    script = ROOT / SCRIPTS[module]
    staged = output / module / "workspace"
    staged.mkdir(parents=True)
    siblings = list(script.parent.glob("*.py")) if module in ["technical_factors", "external_factors", "download", "backtest"] else [script]
    if module == "xgboost_ranker":
        siblings.append(script.parent / "__init__.py")
    for source in siblings:
        shutil.copy2(source, staged / source.name)
        if sha(source) != sha(staged / source.name):
            raise RuntimeError(f"Staging changed original source: {source}")
    if module in ["technical_factors", "external_factors"]:
        (staged / "data").mkdir()
        files = ["clean_basic_data.csv"] + (["test_data_weekly.csv"] if module == "external_factors" else [])
        for name in files:
            shutil.copy2(ROOT / "data" / name, staged / "data" / name)
    return staged / script.name


def child_command(module, script, args, result_dir):
    base = [args.python, "-u", str(script)]
    if args.command == "check" and module in ["ridge", "xgboost_ranker"]:
        return [args.python, "-u", str(ROOT / "integration/check_original.py"), module,
                "--source", str(script), "--input", str(args.input)]
    if module == "ridge":
        return base + ["--input", str(args.input), "--output", str(result_dir), "--mode", args.mode]
    if module == "random_forest":
        return base + ["--data-file", str(args.input), "--screening-file", str(args.screening_file)] + (["--check-only"] if args.command == "check" else [])
    if module == "xgboost":
        return base + ["--input", str(args.input), "--mode", args.mode,
                       "--feature-set", args.xgboost_feature_set] + (["--check-only"] if args.command == "check" else [])
    if module == "xgboost_ranker":
        return base + ["--input", str(args.input), "--output-dir", str(result_dir / "data"),
                       "--model-dir", str(result_dir / "models")]
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="verify", choices=["verify", "status", "check", "run"])
    parser.add_argument("--modules", nargs="+", choices=list(SCRIPTS), default=MODEL_MODULES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--input", type=Path, default=ROOT / "data/test_45_factors_weekly.csv")
    parser.add_argument("--screening-file", type=Path, default=ROOT / "single_factor_analysis/reference/Single_Factor_Screening_66_Stocks.xlsx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["rolling", "fixed", "both"], default="both", help="Original Ridge/XGBoost option; RF keeps its original nine experiments")
    parser.add_argument("--xgboost-feature-set", choices=["all", "7", "7_macro", "45"], default="all")
    args = parser.parse_args(argv)
    executable = shutil.which(args.python)
    args.python = executable or str(Path(args.python).expanduser().resolve())
    verification = verify_originals()
    if args.command == "verify":
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 0 if verification["passed"] else 1
    if not verification["passed"]:
        raise SystemExit("Original source/data integrity check failed. Inspect python run.py verify.")
    modules = list(dict.fromkeys(args.modules))
    deps = dependency_status(args.python, modules)
    if args.command == "status":
        for module in modules:
            missing = [name for name in DEPENDENCIES.get(module, []) if not deps.get(name)]
            status = "blocked: original backtest engine/replay project is missing" if module == "backtest" else ("missing packages: " + ", ".join(missing) if missing else "original script available")
            print(f"{module}: {status}")
        print("single_factor_analysis: original workbook/document available; original Python implementation was not supplied")
        print("model_evaluation: original algorithms' built-in metrics and supplied report tables retained")
        return 0
    args.input = args.input.expanduser().resolve()
    args.screening_file = args.screening_file.expanduser().resolve()
    output = (args.output or ROOT / "outputs" / datetime.now().strftime("original_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Choose a new or empty output directory: {output}")
    if output == ROOT or output in ROOT.parents or output == args.input or output == args.screening_file:
        raise SystemExit("Output must not replace the project or an input file")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "source_verification.json", verification)
    manifest = {"protocol": "integration_only", "original_source_modified": False, "command": args.command,
                "python": args.python, "input": str(args.input), "screening_file": str(args.screening_file),
                "status": "running", "modules": [], "environment": deps}
    for name, path in [("input_sha256", args.input), ("screening_sha256", args.screening_file)]:
        if path.is_file():
            manifest[name] = sha(path)
    save_json(output / "run_manifest.json", manifest)
    print(f"Output: {output}", flush=True)
    try:
        for module in modules:
            record = {"module": module, "original_script": SCRIPTS[module]}
            manifest["modules"].append(record)
            missing = [name for name in DEPENDENCIES.get(module, []) if not deps.get(name)]
            if module == "backtest":
                record.update(status="blocked", reason="Missing original engine run.py, native replay/recovery scripts and matching native predictions. No replacement implementation is used.")
            elif missing:
                record.update(status="blocked", reason="Missing packages: " + ", ".join(missing))
            elif module in MODEL_MODULES and not args.input.is_file():
                record.update(status="blocked", reason=f"Missing original input: {args.input}")
            elif module == "random_forest" and not args.screening_file.is_file():
                record.update(status="blocked", reason=f"Missing original screening workbook: {args.screening_file}")
            elif args.command == "check" and module not in MODEL_MODULES:
                compile((ROOT / SCRIPTS[module]).read_bytes(), SCRIPTS[module], "exec")
                record.update(status="syntax_checked", reason="No download or data rebuilding performed")
            else:
                script = stage_original(module, output)
                result_dir = output / module / "results"
                command = child_command(module, script, args, result_dir)
                record.update(status="running", command=command, cwd=str(script.parent), source_sha256=sha(script))
                save_json(output / "run_manifest.json", manifest)
                print(f"{module}: executing original script; log={output / (module + '.log')}", flush=True)
                with (output / (module + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=script.parent, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                record.update(status=("checked" if args.command == "check" else "completed") if result.returncode == 0 else "failed", returncode=result.returncode)
            print(f"{module}: {record['status']}", flush=True)
            save_json(output / "run_manifest.json", manifest)
        unchanged = verify_originals()
        for key, path in [("input_sha256", args.input), ("screening_sha256", args.screening_file)]:
            if key in manifest and sha(path) != manifest[key]:
                raise RuntimeError(f"Input changed during original execution: {path}")
        if not unchanged["passed"]:
            raise RuntimeError("Original sources/data changed during execution")
        save_json(output / "source_verification_after.json", unchanged)
        manifest["status"] = "completed" if all(r["status"] in ["completed", "checked", "syntax_checked"] for r in manifest["modules"]) else "incomplete"
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "run_manifest.json", manifest)
        raise
    save_json(output / "run_manifest.json", manifest)
    print(f"{manifest['status']}: {output}", flush=True)
    return 0 if manifest["status"] == "completed" else 1
