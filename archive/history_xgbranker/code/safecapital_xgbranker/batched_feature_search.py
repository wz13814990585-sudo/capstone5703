from __future__ import annotations

import hashlib
import itertools
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost

from .config import (
    FUNDAMENTAL_6,
    MACRO_4,
    TECHNICAL_31,
    ExperimentConfig,
    TimeConfig,
)
from .data import load_and_prepare_data
from .metrics import summarize_weekly_metrics
from .optimization import evaluate_schedule


@dataclass(frozen=True)
class BatchedFeatureSearchConfig:
    """Staged feature search that never evaluates the existing final holdout."""

    selection_start_week: int = 312
    selection_end_week: int = 418
    discovery_weeks: int = 53
    rolling_window_weeks: int = 104
    refit_interval_weeks: int = 13
    top_fundamental_bases: int = 2
    top_single_additions_per_base: int = 6
    top_discovery_shortlist: int = 10

    @property
    def confirmation_start_week(self) -> int:
        return self.selection_start_week + self.discovery_weeks

    def validate(self, total_weeks: int) -> None:
        if not 0 <= self.selection_start_week < self.selection_end_week <= total_weeks:
            raise ValueError("Selection interval is outside the available data")
        if not self.selection_start_week < self.confirmation_start_week < self.selection_end_week:
            raise ValueError("Discovery and confirmation periods must both be non-empty")
        positive_fields = (
            self.rolling_window_weeks,
            self.refit_interval_weeks,
            self.top_fundamental_bases,
            self.top_single_additions_per_base,
            self.top_discovery_shortlist,
        )
        if any(value <= 0 for value in positive_fields):
            raise ValueError("All window, promotion, and shortlist values must be positive")


FUNDAMENTAL_CODES = {
    feature: f"F{index}"
    for index, feature in enumerate(FUNDAMENTAL_6, start=1)
}
ADDITION_FEATURES = list(dict.fromkeys(TECHNICAL_31 + MACRO_4))
ADDITION_CODES = {
    **{
        feature: f"T{index:02d}"
        for index, feature in enumerate(TECHNICAL_31, start=1)
    },
    **{
        feature: f"M{index:02d}"
        for index, feature in enumerate(MACRO_4, start=1)
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def fundamental_subset_candidates() -> list[dict[str, Any]]:
    """Return all 2^6 - 1 non-empty Fundamental-6 subsets."""
    rows: list[dict[str, Any]] = []
    for size in range(1, len(FUNDAMENTAL_6) + 1):
        for subset in itertools.combinations(FUNDAMENTAL_6, size):
            codes = [FUNDAMENTAL_CODES[feature] for feature in subset]
            rows.append(
                {
                    "candidate_id": "b1_" + "_".join(code.lower() for code in codes),
                    "batch": "batch_1_fundamental_subsets",
                    "combination_formula": "+".join(codes),
                    "base_candidate_id": None,
                    "base_features": list(subset),
                    "added_features": [],
                    "features": list(subset),
                }
            )
    return rows


def _metric_fields(weekly: pd.DataFrame, prefix: str) -> dict[str, Any]:
    summary = summarize_weekly_metrics(weekly)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _rank_candidates(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return table.sort_values(
        [f"{prefix}_Rank_IC", f"{prefix}_Rank_ICIR", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    )


def _evaluate_candidate(
    *,
    frame: pd.DataFrame,
    base_config: ExperimentConfig,
    frozen: dict[str, Any],
    search: BatchedFeatureSearchConfig,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = replace(
        base_config,
        factor_set_name=str(candidate["candidate_id"]),
        feature_override=tuple(candidate["features"]),
    )
    _, weekly, _, overall = evaluate_schedule(
        frame,
        config,
        frozen,
        phase="feature_research_selection_only",
        evaluation_start_week=search.selection_start_week,
        evaluation_end_week=search.selection_end_week,
        training_mode="rolling",
        training_window_weeks=search.rolling_window_weeks,
        refit_interval_weeks=search.refit_interval_weeks,
    )
    discovery = weekly.loc[
        weekly["week_index"].lt(search.confirmation_start_week)
    ]
    confirmation = weekly.loc[
        weekly["week_index"].ge(search.confirmation_start_week)
    ]
    row = {
        "candidate_id": candidate["candidate_id"],
        "batch": candidate["batch"],
        "combination_formula": candidate["combination_formula"],
        "base_candidate_id": candidate.get("base_candidate_id"),
        "component_single_ids": json.dumps(
            candidate.get("component_single_ids", []), ensure_ascii=False
        ),
        "number_of_features": len(candidate["features"]),
        "base_features": json.dumps(candidate["base_features"], ensure_ascii=False),
        "added_features": json.dumps(candidate["added_features"], ensure_ascii=False),
        "feature_list": json.dumps(candidate["features"], ensure_ascii=False),
        "full_IC": overall["IC"],
        "full_ICIR": overall["ICIR"],
        "full_Rank_IC": overall["Rank_IC"],
        "full_Rank_ICIR": overall["Rank_ICIR"],
        "full_Positive_Rank_IC_Ratio": overall["Positive_Rank_IC_Ratio"],
    }
    row.update(_metric_fields(discovery, "discovery"))
    row.update(_metric_fields(confirmation, "confirmation"))
    weekly = weekly.copy()
    weekly.insert(0, "candidate_id", candidate["candidate_id"])
    weekly.insert(1, "batch", candidate["batch"])
    return row, weekly


def _candidate_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "features": json.loads(row["feature_list"]),
    }


def _add_baseline_comparisons(
    results: pd.DataFrame,
    weekly_all: pd.DataFrame,
    baseline_id: str,
) -> pd.DataFrame:
    result = results.copy()
    baseline = result.set_index("candidate_id").loc[baseline_id]
    for period in ("full", "discovery", "confirmation"):
        result[f"delta_{period}_Rank_IC_vs_baseline"] = (
            result[f"{period}_Rank_IC"] - baseline[f"{period}_Rank_IC"]
        )
    denominator = abs(float(baseline["full_Rank_IC"]))
    result["relative_full_Rank_IC_improvement_pct"] = np.where(
        denominator > 0,
        result["delta_full_Rank_IC_vs_baseline"] / denominator * 100.0,
        np.nan,
    )

    baseline_weekly = (
        weekly_all.loc[weekly_all["candidate_id"].eq(baseline_id), ["week_index", "Rank_IC"]]
        .rename(columns={"Rank_IC": "baseline_weekly_Rank_IC"})
    )
    paired = weekly_all[["candidate_id", "week_index", "Rank_IC"]].merge(
        baseline_weekly, on="week_index", how="left", validate="many_to_one"
    )
    paired["weekly_delta"] = paired["Rank_IC"] - paired["baseline_weekly_Rank_IC"]
    paired_stats = paired.groupby("candidate_id", sort=False)["weekly_delta"].agg(
        paired_weekly_delta_mean="mean",
        paired_weekly_delta_std="std",
        paired_weeks="count",
        weekly_win_ratio_vs_baseline=lambda values: float((values > 0).mean()),
    )
    paired_stats["paired_delta_t_stat"] = (
        paired_stats["paired_weekly_delta_mean"]
        / (paired_stats["paired_weekly_delta_std"] / np.sqrt(paired_stats["paired_weeks"]))
    )
    return result.merge(paired_stats.reset_index(), on="candidate_id", how="left")


def _add_parent_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    lookup = result.set_index("candidate_id")
    for period in ("full", "discovery", "confirmation"):
        parent_metric = result["base_candidate_id"].map(lookup[f"{period}_Rank_IC"])
        result[f"delta_{period}_Rank_IC_vs_parent"] = (
            result[f"{period}_Rank_IC"] - parent_metric
        )
    return result


def _make_batch_2_candidates(
    promoted_bases: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for _, base_row in promoted_bases.iterrows():
        base = _candidate_from_row(base_row)
        for addition in ADDITION_FEATURES:
            code = ADDITION_CODES[addition]
            candidates.append(
                {
                    "candidate_id": f"b2_{base['candidate_id'][3:]}_add_{code.lower()}",
                    "batch": "batch_2_single_additions",
                    "combination_formula": f"{base_row['combination_formula']}+{code}",
                    "base_candidate_id": base["candidate_id"],
                    "base_features": base["features"],
                    "added_features": [addition],
                    "features": base["features"] + [addition],
                }
            )
    return candidates


def _make_batch_3_candidates(
    results: pd.DataFrame,
    promoted_bases: pd.DataFrame,
    top_per_base: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for _, base_row in promoted_bases.iterrows():
        base = _candidate_from_row(base_row)
        singles = results.loc[
            results["batch"].eq("batch_2_single_additions")
            & results["base_candidate_id"].eq(base["candidate_id"])
        ]
        singles = _rank_candidates(singles, "discovery").head(top_per_base)
        single_records = singles.to_dict(orient="records")
        combinations: list[tuple[dict[str, Any], ...]] = list(
            itertools.combinations(single_records, 2)
        )
        combinations.extend(itertools.combinations(single_records[:4], 3))
        for components in combinations:
            additions = [json.loads(item["added_features"])[0] for item in components]
            codes = [ADDITION_CODES[feature] for feature in additions]
            candidates.append(
                {
                    "candidate_id": (
                        f"b3_{base['candidate_id'][3:]}_add_"
                        + "_".join(code.lower() for code in codes)
                    ),
                    "batch": "batch_3_multi_additions",
                    "combination_formula": (
                        f"{base_row['combination_formula']}+" + "+".join(codes)
                    ),
                    "base_candidate_id": base["candidate_id"],
                    "component_single_ids": [item["candidate_id"] for item in components],
                    "base_features": base["features"],
                    "added_features": additions,
                    "features": base["features"] + additions,
                }
            )
    return candidates


def _make_catalog(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "batch": candidate["batch"],
                "combination_formula": candidate["combination_formula"],
                "base_candidate_id": candidate.get("base_candidate_id"),
                "base_features": json.dumps(candidate["base_features"], ensure_ascii=False),
                "added_features": json.dumps(candidate["added_features"], ensure_ascii=False),
                "feature_list": json.dumps(candidate["features"], ensure_ascii=False),
                "number_of_features": len(candidate["features"]),
            }
        )
    return pd.DataFrame(rows)


def _report(
    search: BatchedFeatureSearchConfig,
    results: pd.DataFrame,
    promoted_bases: pd.DataFrame,
    shortlist: pd.DataFrame,
    recommended: pd.Series,
    conservative_winner: pd.Series,
    baseline_id: str,
) -> str:
    baseline = results.set_index("candidate_id").loc[baseline_id]
    batch_summary = (
        results.groupby("batch", sort=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_full_Rank_IC=("full_Rank_IC", "max"),
            best_discovery_Rank_IC=("discovery_Rank_IC", "max"),
            best_confirmation_Rank_IC=("confirmation_Rank_IC", "max"),
        )
        .reset_index()
    )
    top_full = _rank_candidates(results, "full").head(15)[
        [
            "candidate_id",
            "batch",
            "combination_formula",
            "full_Rank_IC",
            "delta_full_Rank_IC_vs_baseline",
            "discovery_Rank_IC",
            "confirmation_Rank_IC",
            "weekly_win_ratio_vs_baseline",
        ]
    ]
    shortlist_view = shortlist[
        [
            "candidate_id",
            "combination_formula",
            "discovery_Rank_IC",
            "confirmation_Rank_IC",
            "full_Rank_IC",
            "delta_full_Rank_IC_vs_baseline",
        ]
    ]
    promoted_view = promoted_bases[
        ["candidate_id", "combination_formula", "discovery_Rank_IC", "confirmation_Rank_IC"]
    ]
    feature_names = json.loads(recommended["feature_list"])
    return f"""# XGBRanker 分批 Feature 组合优化报告

## 结论

本轮在原选择期 Week {search.selection_start_week}–{search.selection_end_week - 1} 内新增了 {len(results)} 个组合，保持 XGBRanker 参数、rolling {search.rolling_window_weeks} 周窗口和每 {search.refit_interval_weeks} 周更新不变，因此差异只来自 Feature 组合。

当前基准是 Fundamental 6：完整选择期 RankIC = {baseline['full_Rank_IC']:.6f}。

在“发现期和确认期都必须高于基准”的稳健性约束下，再按完整选择期 RankIC 排序，本轮最佳新研究候选为 `{recommended['candidate_id']}`：

- 组合公式：`{recommended['combination_formula']}`
- 具体 Features：`{feature_names}`
- 发现期 RankIC：{recommended['discovery_Rank_IC']:.6f}
- 确认期 RankIC：{recommended['confirmation_Rank_IC']:.6f}
- 完整选择期 RankIC：{recommended['full_Rank_IC']:.6f}
- 相对 Fundamental 6 的完整选择期改善：{recommended['delta_full_Rank_IC_vs_baseline']:+.6f}
- 相对基准的发现期改善：{recommended['delta_discovery_Rank_IC_vs_baseline']:+.6f}
- 相对基准的确认期改善：{recommended['delta_confirmation_Rank_IC_vs_baseline']:+.6f}
- 相对基准的完整期改善比例：{recommended['relative_full_Rank_IC_improvement_pct']:+.2f}%
- 逐周 RankIC 超过基准的比例：{recommended['weekly_win_ratio_vs_baseline']:.2%}

更严格地，如果只允许“发现期前 {search.top_discovery_shortlist} 名 + 原基准”进入确认期决选，胜者是 `{conservative_winner['candidate_id']}`，即原 Fundamental 6。因此上述最佳组合应作为下一轮候选，不应宣称已在全新 holdout 上证明胜出。

## 分批方法

1. Batch 1：穷举 6 个基本面因子的 63 个非空子集。
2. Batch 2：仅对发现期最优的 {search.top_fundamental_bases} 个基本面子集，逐个加入 31 个技术因子和 4 个宏观因子。
3. Batch 3：每个基础组合只保留 Batch 2 发现期最好的 {search.top_single_additions_per_base} 个新增因子，再测试两两组合和前 4 个的三因子组合。
4. 评价分为 Week {search.selection_start_week}–{search.confirmation_start_week - 1} 发现期和 Week {search.confirmation_start_week}–{search.selection_end_week - 1} 确认期。只有发现期前 {search.top_discovery_shortlist} 名与原基准进入确认期决选。

Batch 摘要：

```text
{batch_summary.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

Batch 1 晋级的基本面组合：

```text
{promoted_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 改善最大的组合

```text
{top_full.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 发现期 Shortlist 的确认结果

```text
{shortlist_view.to_string(index=False, float_format=lambda value: f'{value:.6f}')}
```

## 解读边界

这是在既有研究选择期内做的新一轮 Feature 研究。本轮没有运行 Week {search.selection_end_week} 之后的已有最终 holdout，也没有回测。由于之前已经查看过原 holdout，本报告的推荐应视为“新候选”，而不是拥有新的未见样本证明的最终结论。
"""


def run_batched_feature_search(
    base_config: ExperimentConfig,
    *,
    search: BatchedFeatureSearchConfig | None = None,
    frozen_config_path: Path = Path(
        "history_fine/data/rankic_optimization/frozen_configs_by_feature_set.json"
    ),
) -> dict[str, Any]:
    search = search or BatchedFeatureSearchConfig()
    all_features = list(dict.fromkeys(FUNDAMENTAL_6 + ADDITION_FEATURES))
    time = TimeConfig(
        warmup_weeks=260,
        warmup_train_weeks=208,
        warmup_validation_weeks=52,
        rolling_window_weeks=search.rolling_window_weeks,
        refit_interval_weeks=search.refit_interval_weeks,
        forecast_horizon_weeks=1,
    )
    base_config = replace(base_config, time=time, feature_override=None)
    frame, diagnostics = load_and_prepare_data(
        input_path=base_config.input_path,
        factor_definitions_path=base_config.factor_definitions_path,
        features=all_features,
        forecast_horizon_weeks=1,
    )
    search.validate(int(frame["week_index"].nunique()))
    frozen_all = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    frozen = frozen_all["fundamental_6"]
    if frozen["selected_candidate_id"] != "pairwise_d4_bins5":
        raise ValueError("Expected the previously frozen Fundamental-6 configuration")

    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    weekly_tables: list[pd.DataFrame] = []

    def evaluate_batch(batch_candidates: list[dict[str, Any]], label: str) -> None:
        candidates.extend(batch_candidates)
        print(f"{label}: {len(batch_candidates)} candidates", flush=True)
        for index, candidate in enumerate(batch_candidates, start=1):
            if index == 1 or index % 10 == 0 or index == len(batch_candidates):
                print(f"  {label} {index}/{len(batch_candidates)}", flush=True)
            row, weekly = _evaluate_candidate(
                frame=frame,
                base_config=base_config,
                frozen=frozen,
                search=search,
                candidate=candidate,
            )
            rows.append(row)
            weekly_tables.append(weekly)

    batch_1 = fundamental_subset_candidates()
    evaluate_batch(batch_1, "Batch 1")
    interim = pd.DataFrame(rows)
    baseline_id = next(
        item["candidate_id"]
        for item in batch_1
        if item["features"] == FUNDAMENTAL_6
    )
    promoted_bases = _rank_candidates(interim, "discovery").head(
        search.top_fundamental_bases
    )

    batch_2 = _make_batch_2_candidates(promoted_bases)
    evaluate_batch(batch_2, "Batch 2")
    interim = pd.DataFrame(rows)
    batch_3 = _make_batch_3_candidates(
        interim, promoted_bases, search.top_single_additions_per_base
    )
    evaluate_batch(batch_3, "Batch 3")

    weekly_all = pd.concat(weekly_tables, ignore_index=True)
    results = pd.DataFrame(rows)
    results = _add_baseline_comparisons(results, weekly_all, baseline_id)
    results = _add_parent_comparisons(results)
    shortlist = _rank_candidates(results, "discovery").head(
        search.top_discovery_shortlist
    )
    if baseline_id not in set(shortlist["candidate_id"]):
        shortlist = pd.concat(
            [shortlist, results.loc[results["candidate_id"].eq(baseline_id)]],
            ignore_index=True,
        )
    shortlist = _rank_candidates(shortlist, "confirmation")
    conservative_winner = shortlist.iloc[0]
    robust_candidates = results.loc[
        results["delta_discovery_Rank_IC_vs_baseline"].gt(0)
        & results["delta_confirmation_Rank_IC_vs_baseline"].gt(0)
    ].copy()
    if robust_candidates.empty:
        recommended = results.loc[results["candidate_id"].eq(baseline_id)].iloc[0]
    else:
        recommended = _rank_candidates(robust_candidates, "full").iloc[0]

    out = base_config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    catalog = _make_catalog(candidates)
    catalog.to_csv(out / "combination_catalog.csv", index=False)
    results.to_csv(out / "all_combination_metrics.csv", index=False)
    for batch_number, batch_name in (
        (1, "batch_1_fundamental_subsets"),
        (2, "batch_2_single_additions"),
        (3, "batch_3_multi_additions"),
    ):
        results.loc[results["batch"].eq(batch_name)].to_csv(
            out / f"batch_{batch_number}_metrics.csv", index=False
        )
    weekly_all.to_csv(out / "all_combination_weekly_metrics.csv", index=False)
    shortlist.to_csv(out / "discovery_shortlist_confirmation.csv", index=False)
    _rank_candidates(robust_candidates, "full").to_csv(
        out / "robust_improvement_candidates.csv", index=False
    )
    promoted_bases.to_csv(out / "promoted_fundamental_bases.csv", index=False)
    _write_json(
        out / "recommended_candidate.json",
        {
            "candidate": recommended.to_dict(),
            "selection_rule": (
                "must improve on Fundamental-6 in both discovery and confirmation "
                "periods, then maximum full-selection-period Mean Rank IC"
            ),
            "conservative_discovery_shortlist_winner": conservative_winner.to_dict(),
            "conservative_selection_rule": (
                "top discovery-period candidates plus Fundamental-6 baseline, "
                "then maximum confirmation-period Mean Rank IC"
            ),
            "existing_final_holdout_evaluated": False,
            "backtest_performed": False,
        },
    )
    manifest = {
        "input_filename": diagnostics["input_filename"],
        "input_sha256": diagnostics["input_sha256"],
        "search_config": asdict(search),
        "frozen_config_path": str(frozen_config_path),
        "frozen_config_sha256": hashlib.sha256(
            frozen_config_path.read_bytes()
        ).hexdigest(),
        "frozen_model_configuration": frozen,
        "feature_codebook": FUNDAMENTAL_CODES | ADDITION_CODES,
        "candidate_count": len(results),
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "existing_final_holdout_evaluated": False,
        "backtest_performed": False,
    }
    _write_json(out / "experiment_manifest.json", manifest)
    (out / "batched_feature_search_report.md").write_text(
        _report(
            search,
            results,
            promoted_bases,
            shortlist,
            recommended,
            conservative_winner,
            baseline_id,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(out.resolve()),
        "candidate_count": len(results),
        "baseline_id": baseline_id,
        "recommended_candidate": recommended.to_dict(),
        "conservative_discovery_shortlist_winner": conservative_winner.to_dict(),
        "existing_final_holdout_evaluated": False,
        "backtest_performed": False,
    }
