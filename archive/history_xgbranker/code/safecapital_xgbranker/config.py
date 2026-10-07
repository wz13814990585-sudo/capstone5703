from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PRIORITY_7 = [
    "revenue_growth_yoy",
    "ret_120d",
    "cashflow_to_assets",
    "ret_1d",
    "ret_5d",
    "close_location_value",
    "body_range_ratio",
]

# Pre-registered economic blocks for an interpretable 2^3 - 1 feature study.
# F: accounting/fundamental quality and growth; R: return/momentum horizons;
# P: within-bar price action.  The suite is fixed before OOS evaluation.
FEATURE_BLOCKS = {
    "F": ["revenue_growth_yoy", "cashflow_to_assets"],
    "R": ["ret_120d", "ret_1d", "ret_5d"],
    "P": ["close_location_value", "body_range_ratio"],
}

FEATURE_COMBINATION_BLOCKS = {
    "fundamental_2": "F",
    "returns_3": "R",
    "price_action_2": "P",
    "fundamental_returns_5": "F+R",
    "fundamental_price_action_4": "F+P",
    "technical_5": "R+P",
    "priority_7": "F+R+P",
}

FEATURE_COMBINATIONS = {
    name: [feature for block in blocks.split("+") for feature in FEATURE_BLOCKS[block]]
    for name, blocks in FEATURE_COMBINATION_BLOCKS.items()
}
# Preserve the originally confirmed Priority-7 column order.
FEATURE_COMBINATIONS["priority_7"] = PRIORITY_7

MACRO_4 = [
    "macro_vix_level",
    "macro_vix_change_5obs",
    "macro_yield_spread_10y_2y",
    "macro_dgs10_change_5obs",
]

DUPLICATE_4 = [
    "cs_rank_amihud20",
    "cs_rank_ret60",
    "cs_rank_vol20",
    "cs_rank_ret20",
]

STOCK_37 = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "momentum_20_5",
    "overnight_gap",
    "intraday_return",
    "ema20_distance",
    "ema60_distance",
    "ema20_60_spread",
    "rsi14_centered",
    "macd_hist_atr",
    "breakout_position_60",
    "realized_vol_5",
    "realized_vol_20",
    "vol_ratio_5_20",
    "parkinson_vol_20",
    "overnight_vol_20",
    "downside_vol_20",
    "volume_zscore_20",
    "volume_trend_5_20",
    "amihud_illiq_20",
    "signed_volume_pressure_20",
    "close_location_value",
    "body_range_ratio",
    "trend_r2_20",
    "return_autocorr_20",
    "volume_cv_20",
    "return_volume_corr_20",
    "gap_followthrough_20",
    "revenue_growth_yoy",
    "gross_margin_change_yoy",
    "operating_profitability",
    "cashflow_to_assets",
    "debt_to_assets",
    "asset_growth_yoy",
]

FUNDAMENTAL_6 = [
    "revenue_growth_yoy",
    "gross_margin_change_yoy",
    "operating_profitability",
    "cashflow_to_assets",
    "debt_to_assets",
    "asset_growth_yoy",
]

RETURNS_MOMENTUM_6 = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "momentum_20_5",
]

GAP_INTRADAY_2 = ["overnight_gap", "intraday_return"]

TREND_7 = [
    "ema20_distance",
    "ema60_distance",
    "ema20_60_spread",
    "rsi14_centered",
    "macd_hist_atr",
    "breakout_position_60",
    "trend_r2_20",
]

VOLATILITY_6 = [
    "realized_vol_5",
    "realized_vol_20",
    "vol_ratio_5_20",
    "parkinson_vol_20",
    "overnight_vol_20",
    "downside_vol_20",
]

LIQUIDITY_VOLUME_6 = [
    "volume_zscore_20",
    "volume_trend_5_20",
    "amihud_illiq_20",
    "signed_volume_pressure_20",
    "volume_cv_20",
    "return_volume_corr_20",
]

PRICE_ACTION_2 = ["close_location_value", "body_range_ratio"]
DEPENDENCY_2 = ["return_autocorr_20", "gap_followthrough_20"]
TECHNICAL_31 = [feature for feature in STOCK_37 if feature not in FUNDAMENTAL_6]


def ordered_union(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(feature for group in groups for feature in group))


# Research candidates are defined before the selection and final-holdout periods.
# They include interpretable blocks, Priority-7 ablations, broad feature families,
# all non-duplicate stock factors, and explicit macro-augmented variants.
OPTIMIZATION_FEATURE_SETS = {
    **FEATURE_COMBINATIONS,
    "fundamental_6": FUNDAMENTAL_6,
    "returns_momentum_6": RETURNS_MOMENTUM_6,
    "trend_7": TREND_7,
    "volatility_6": VOLATILITY_6,
    "liquidity_volume_6": LIQUIDITY_VOLUME_6,
    "microstructure_6": ordered_union(
        GAP_INTRADAY_2, PRICE_ACTION_2, DEPENDENCY_2
    ),
    "technical_31": TECHNICAL_31,
    "stock_37": STOCK_37,
    "priority_7_plus_macro": ordered_union(PRIORITY_7, MACRO_4),
    "stock_37_plus_macro": ordered_union(STOCK_37, MACRO_4),
    "priority_plus_trend": ordered_union(PRIORITY_7, TREND_7),
    "priority_plus_volatility": ordered_union(PRIORITY_7, VOLATILITY_6),
    "priority_plus_liquidity": ordered_union(PRIORITY_7, LIQUIDITY_VOLUME_6),
}

FACTOR_SETS = {
    **OPTIMIZATION_FEATURE_SETS,
}


def default_candidates() -> list[dict[str, Any]]:
    """Small, fixed candidate space used only during Warm-up."""
    common = {
        "tree_method": "hist",
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "eval_metric": "ndcg@10",
    }
    return [
        {
            "candidate_id": "ndcg_d3_bins5",
            "objective": "rank:ndcg",
            "relevance_bins": 5,
            "params": common
            | {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 140},
        },
        {
            "candidate_id": "ndcg_d4_bins5",
            "objective": "rank:ndcg",
            "relevance_bins": 5,
            "params": common
            | {
                "max_depth": 4,
                "learning_rate": 0.04,
                "n_estimators": 180,
                "reg_lambda": 2.0,
            },
        },
        {
            "candidate_id": "ndcg_d3_bins10",
            "objective": "rank:ndcg",
            "relevance_bins": 10,
            "params": common
            | {
                "max_depth": 3,
                "learning_rate": 0.04,
                "n_estimators": 180,
                "min_child_weight": 15,
            },
        },
        {
            "candidate_id": "pairwise_d4_bins5",
            "objective": "rank:pairwise",
            "relevance_bins": 5,
            "params": common
            | {
                "max_depth": 4,
                "learning_rate": 0.05,
                "n_estimators": 160,
                "subsample": 1.0,
            },
        },
    ]


@dataclass(frozen=True)
class TimeConfig:
    warmup_weeks: int = 156
    warmup_train_weeks: int = 104
    warmup_validation_weeks: int = 52
    rolling_window_weeks: int = 156
    refit_interval_weeks: int = 4
    forecast_horizon_weeks: int = 1

    def validate(self) -> None:
        if self.warmup_train_weeks + self.warmup_validation_weeks != self.warmup_weeks:
            raise ValueError(
                "warmup_train_weeks + warmup_validation_weeks must equal warmup_weeks"
            )
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ExperimentConfig:
    input_path: Path = Path("test_45_factors_weekly.csv")
    factor_definitions_path: Path = Path("factor_definitions(1).csv")
    output_dir: Path = Path("history_fine/data/fixed_rolling_comparison")
    factor_set_name: str = "priority_7"
    feature_override: tuple[str, ...] | None = None
    target_column: str = "target"
    random_seed: int = 5703
    n_jobs: int = 1
    time: TimeConfig = field(default_factory=TimeConfig)
    candidates: list[dict[str, Any]] = field(default_factory=default_candidates)

    @property
    def features(self) -> list[str]:
        if self.feature_override is not None:
            return list(self.feature_override)
        try:
            return list(FACTOR_SETS[self.factor_set_name])
        except KeyError as exc:
            raise ValueError(
                f"Unknown factor_set_name={self.factor_set_name!r}; "
                f"choose from {sorted(FACTOR_SETS)}"
            ) from exc

    def validate(self) -> None:
        self.time.validate()
        _ = self.features
        if self.target_column != "target":
            raise ValueError("This study fixes the economic target to 'target'")
        if not self.candidates:
            raise ValueError("At least one fixed Warm-up candidate is required")
