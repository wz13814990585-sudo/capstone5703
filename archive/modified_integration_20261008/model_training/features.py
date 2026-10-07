"""The original seven selected signals and the unchanged 45-factor baseline."""

SEVEN = [
    "revenue_growth_yoy", "ret_120d", "cashflow_to_assets", "ret_1d",
    "ret_5d", "close_location_value", "body_range_ratio",
]
MACRO = [
    "macro_vix_level", "macro_vix_change_5obs", "macro_yield_spread_10y_2y",
    "macro_dgs10_change_5obs",
]
INTERNAL35 = """ret_1d ret_5d ret_20d ret_60d ret_120d momentum_20_5
overnight_gap intraday_return ema20_distance ema60_distance ema20_60_spread
rsi14_centered macd_hist_atr breakout_position_60 realized_vol_5 realized_vol_20
vol_ratio_5_20 parkinson_vol_20 overnight_vol_20 downside_vol_20 volume_zscore_20
volume_trend_5_20 amihud_illiq_20 signed_volume_pressure_20 close_location_value
body_range_ratio cs_rank_ret20 cs_rank_ret60 cs_rank_vol20 cs_rank_amihud20
trend_r2_20 return_autocorr_20 volume_cv_20 return_volume_corr_20 gap_followthrough_20""".split()
FINANCIAL6 = [
    "revenue_growth_yoy", "gross_margin_change_yoy", "operating_profitability",
    "cashflow_to_assets", "debt_to_assets", "asset_growth_yoy",
]
ALL45 = INTERNAL35 + FINANCIAL6 + MACRO
FEATURE_SETS = {"7": SEVEN, "7_macro": SEVEN + MACRO, "45": ALL45}
