"""Weekly cross-sectional IC screening; default selection never uses holdout outcomes.

This module is newly implemented from reference/Single_Factor_Methodology_CN_EN.docx.
The reference contains full-sample exploratory results, not executable analysis code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.panel import load_weekly_panel
from data_processing.paths import DEFAULT_WEEKLY_FILE


@dataclass(frozen=True)
class ScreeningRules:
    minimum_stocks: int = 20
    minimum_distinct_factor_values: int = 5
    minimum_weeks: int = 104
    minimum_weeks_per_year: int = 40
    minimum_years: int = 3
    minimum_coverage: float = 0.80
    minimum_abs_mean_ic: float = 0.01
    minimum_abs_icir: float = 0.08
    minimum_annual_sign_agreement: float = 0.60
    maximum_abs_mean_correlation: float = 0.85


def _weekly_ic(panel: pd.DataFrame, factor: str, rules: ScreeningRules) -> pd.DataFrame:
    rows = []
    for date, group in panel.groupby("date", sort=True):
        values = group[[factor, "target"]].replace([np.inf, -np.inf], np.nan).dropna()
        valid = (len(values) >= rules.minimum_stocks
                 and values[factor].nunique() >= rules.minimum_distinct_factor_values
                 and values["target"].nunique() >= 2)
        ic = values[factor].rank(method="average").corr(values["target"].rank(method="average")) if valid else np.nan
        rows.append({"date": date, "factor": factor, "observations": len(values), "rank_ic": ic})
    return pd.DataFrame(rows)


def _mean_weekly_correlation(panel: pd.DataFrame, left: str, right: str, rules: ScreeningRules) -> float:
    values = []
    for _, group in panel.groupby("date", sort=False):
        pairs = group[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        if (len(pairs) >= rules.minimum_stocks
                and pairs[left].nunique() >= rules.minimum_distinct_factor_values
                and pairs[right].nunique() >= rules.minimum_distinct_factor_values):
            values.append(pairs[left].rank(method="average").corr(pairs[right].rank(method="average")))
    return float(np.mean(values)) if values else np.nan


def run_screening(input_path: str | Path = DEFAULT_WEEKLY_FILE,
                  output_dir: str | Path = "outputs/single_factor_analysis",
                  selection_end: str = "2024-09-01", *, exploratory_full_sample: bool = False,
                  rules: ScreeningRules | None = None) -> dict:
    """Save screening evidence and return a manifest; never alter frozen model factors.

    selection_end is exclusive for both signal dates and label end dates. Running full
    sample is an explicitly exploratory operation and must not be reported as an
    independent factor selection step preceding a holdout backtest.
    """
    rules = rules or ScreeningRules()
    panel, audit, excluded = load_weekly_panel(input_path)
    cutoff = pd.Timestamp(selection_end)
    if not exploratory_full_sample:
        panel = panel.loc[panel["date"].lt(cutoff) & panel["label_end_date"].lt(cutoff)].copy()
    if panel.empty:
        raise ValueError("No verifiable labels remain before the factor-selection cutoff")
    factors = audit["factor_names"]
    if not factors:
        raise ValueError("No candidate factor columns found in the weekly panel")
    panel[factors] = panel[factors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    records, weekly_tables, annual_tables = [], [], []
    unique_factors = []
    for name in factors:
        coverage = float(panel[name].notna().mean())
        duplicate_of = next((kept for kept in unique_factors if panel[name].equals(panel[kept])), "")
        is_macro = name.startswith("macro_")
        record = {"factor": name, "coverage": coverage, "duplicate_of": duplicate_of,
                  "mean_rank_ic": np.nan, "rank_icir": np.nan, "valid_weeks": 0,
                  "eligible_years": 0, "annual_sign_agreement": np.nan,
                  "direction": "undefined", "passes_thresholds": False, "correlated_with": ""}
        if duplicate_of:
            record["decision"] = "duplicate"
        elif is_macro:
            unique_factors.append(name)
            record["decision"] = "macro_retained_separately"
        else:
            unique_factors.append(name)
            weekly = _weekly_ic(panel, name, rules)
            weekly_tables.append(weekly)
            valid = weekly.dropna(subset=["rank_ic"])
            mean_ic = float(valid["rank_ic"].mean())
            std = float(valid["rank_ic"].std(ddof=1))
            icir = mean_ic / std if std > 0 else np.nan
            annual = valid.assign(year=valid["date"].dt.year).groupby("year", as_index=False).agg(
                mean_rank_ic=("rank_ic", "mean"), valid_weeks=("rank_ic", "count")
            )
            annual["factor"] = name
            annual["eligible"] = annual["valid_weeks"].ge(rules.minimum_weeks_per_year)
            annual["same_direction"] = np.sign(annual["mean_rank_ic"]).eq(np.sign(mean_ic)) & (mean_ic != 0)
            annual_tables.append(annual)
            eligible = annual.loc[annual["eligible"]]
            agreement = float(eligible["same_direction"].mean()) if len(eligible) else np.nan
            passes = bool(coverage >= rules.minimum_coverage and len(valid) >= rules.minimum_weeks
                          and len(eligible) >= rules.minimum_years
                          and abs(mean_ic) >= rules.minimum_abs_mean_ic
                          and abs(icir) >= rules.minimum_abs_icir
                          and agreement >= rules.minimum_annual_sign_agreement)
            record.update(mean_rank_ic=mean_ic, rank_icir=icir, valid_weeks=len(valid),
                          eligible_years=len(eligible), annual_sign_agreement=agreement,
                          direction="positive" if mean_ic > 0 else "negative" if mean_ic < 0 else "undefined",
                          passes_thresholds=passes,
                          decision="candidate" if passes else "general_alternative" if abs(mean_ic) >= 0.005
                          else "weak_signal_alternative")
        records.append(record)
    summary = pd.DataFrame(records).set_index("factor", drop=False)
    candidates = summary.loc[summary["passes_thresholds"]].reset_index(drop=True).copy()
    candidates["_abs_ic"] = candidates["mean_rank_ic"].abs()
    candidates = candidates.sort_values(["_abs_ic", "factor"], ascending=[False, True])
    retained, correlation_records = [], []
    for name in candidates["factor"]:
        conflicts = []
        for kept in retained:
            correlation = _mean_weekly_correlation(panel, name, kept, rules)
            correlation_records.append({"factor": name, "retained_factor": kept, "mean_weekly_spearman": correlation})
            if np.isfinite(correlation) and abs(correlation) >= rules.maximum_abs_mean_correlation:
                conflicts.append(kept)
        summary.loc[name, "correlated_with"] = ",".join(conflicts)
        summary.loc[name, "decision"] = "correlated_alternative" if conflicts else "priority_candidate"
        if not conflicts:
            retained.append(name)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary.reset_index(drop=True).to_csv(output / "screening_results.csv", index=False)
    pd.concat(weekly_tables, ignore_index=True).to_csv(output / "weekly_rank_ic.csv", index=False) if weekly_tables else None
    pd.concat(annual_tables, ignore_index=True).to_csv(output / "annual_rank_ic.csv", index=False) if annual_tables else None
    pd.DataFrame(correlation_records, columns=["factor", "retained_factor", "mean_weekly_spearman"]).to_csv(output / "candidate_correlations.csv", index=False)
    excluded.to_csv(output / "unverifiable_labels.csv", index=False)
    manifest = {
        "implementation": "newly reconstructed from supplied methodology, with strict calendar-week label validation",
        "source_audit": audit, "mode": "full_sample_exploration" if exploratory_full_sample else "before_selection_cutoff",
        "selection_end_exclusive": None if exploratory_full_sample else cutoff.strftime("%Y-%m-%d"),
        "selection_start": panel["date"].min().strftime("%Y-%m-%d"),
        "selection_last_signal_date": panel["date"].max().strftime("%Y-%m-%d"),
        "selection_last_label_end_date": panel["label_end_date"].max().strftime("%Y-%m-%d"),
        "rows": len(panel), "weeks": int(panel["date"].nunique()), "rules": asdict(rules),
        "priority_factors": retained,
        "macro_factors": summary.loc[summary["decision"].eq("macro_retained_separately"), "factor"].tolist(),
        "decision_counts": {str(k): int(v) for k, v in summary["decision"].value_counts().items()},
        "output": str(output),
        "historical_reference_note": "Original 7-factor shortlist used full-sample exploration through 2026; frozen legacy model comparisons retain this selection bias. This run does not overwrite that shortlist. The original document removed only 66 terminal rows; this implementation also excludes 16 unverifiable historical missing-week labels in the supplied CSV.",
    }
    (output / "screening_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
