"""计算 35 个内部周频候选因子及当周开盘至收盘收益率目标。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from collect_sec_fundamentals import FUNDAMENTAL_DEFINITIONS


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "clean_basic_data.csv"
WEEKLY_OUTPUT_FILE = DATA_DIR / "test_data_weekly.csv"
DEFINITIONS_FILE = DATA_DIR / "factor_definitions.csv"
MODEL_START = pd.Timestamp("2016-09-01")
MODEL_END = pd.Timestamp("2026-09-30")
EPSILON = 1e-12
BASE_PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

FACTOR_DEFINITIONS = [
    ("ret_1d", "ln(C(t) / C(t-1))"),
    ("ret_5d", "ln(C(t) / C(t-5))"),
    ("ret_20d", "ln(C(t) / C(t-20))"),
    ("ret_60d", "ln(C(t) / C(t-60))"),
    ("ret_120d", "ln(C(t) / C(t-120))"),
    ("momentum_20_5", "ln(C(t-5) / C(t-20))"),
    ("overnight_gap", "ln(O(t) / C(t-1))"),
    ("intraday_return", "ln(C(t) / O(t))"),
    ("ema20_distance", "C / EMA20 - 1"),
    ("ema60_distance", "C / EMA60 - 1"),
    ("ema20_60_spread", "EMA20 / EMA60 - 1"),
    ("rsi14_centered", "(RSI14 - 50) / 50"),
    ("macd_hist_atr", "MACD histogram / ATR14"),
    ("breakout_position_60", "(C - Low60) / (High60 - Low60)"),
    ("realized_vol_5", "std_5(ret_1d)"),
    ("realized_vol_20", "std_20(ret_1d)"),
    ("vol_ratio_5_20", "realized_vol_5 / realized_vol_20"),
    ("parkinson_vol_20", "sqrt(mean_20(ln(H/L)^2) / (4 ln(2)))"),
    ("overnight_vol_20", "std_20(overnight_gap)"),
    ("downside_vol_20", "sqrt(mean_20(min(ret_1d, 0)^2))"),
    ("volume_zscore_20", "(ln(V) - mean_20(ln(V))) / std_20(ln(V))"),
    ("volume_trend_5_20", "mean_5(V) / mean_20(V) - 1"),
    ("amihud_illiq_20", "mean_20(abs(ret_1d) / (raw Close * Volume))"),
    ("signed_volume_pressure_20", "sum_20(sign(ret_1d) * V) / sum_20(V)"),
    ("close_location_value", "(2C - H - L) / (H - L)"),
    ("body_range_ratio", "(C - O) / (H - L)"),
    ("cs_rank_ret20", "same-date percentile rank of ret_20d"),
    ("cs_rank_ret60", "same-date percentile rank of ret_60d"),
    ("cs_rank_vol20", "same-date percentile rank of realized_vol_20"),
    ("cs_rank_amihud20", "same-date percentile rank of amihud_illiq_20"),
]

INTERNAL_FACTORS = [
    "trend_r2_20", "return_autocorr_20", "volume_cv_20", "return_volume_corr_20", "gap_followthrough_20",
]
INTERNAL_FACTOR_NAMES = [name for name, _ in FACTOR_DEFINITIONS] + INTERNAL_FACTORS
ADDITIONAL_FACTOR_DEFINITIONS = [
    ("trend_r2_20", "OLS R² of adjusted close over the last 20 trading sessions"),
    ("return_autocorr_20", "corr(ret[t-19:t], ret[t-20:t-1])"),
    ("volume_cv_20", "std_20(volume, ddof=1) / mean_20(volume)"),
    ("return_volume_corr_20", "corr(simple return, log volume change) over 20 sessions"),
    ("gap_followthrough_20", "corr(overnight gap, intraday return) over 20 sessions"),
]
EXTERNAL_FACTOR_DEFINITIONS = [
    *FUNDAMENTAL_DEFINITIONS,
    ("macro_vix_level", "Latest available FRED VIXCLS level"),
    ("macro_vix_change_5obs", "VIXCLS less its value five valid observations earlier"),
    ("macro_yield_spread_10y_2y", "(DGS10 - DGS2) / 100 on the same observation date"),
    ("macro_dgs10_change_5obs", "(DGS10 - DGS10 five valid observations earlier) / 100"),
]
TARGET_DEFINITION = (
    "target",
    "ln(AdjClose_week_end(t+1) / AdjClose_week_end(t)); next week's final trading-day adjusted Close divided by this week's final trading-day adjusted Close. Label only, not a model input.",
)
TARGET2_DEFINITION = (
    "target2",
    "Classification of target: 0 if target < -3%; 1 if -3% <= target < -1%; 2 if -1% <= target <= 1%; 3 if 1% < target <= 3%; 4 if target > 3%. Label only, not a model input.",
)


def adjusted_ohlc(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """将 Yahoo 的未复权 OHLC 转成与 Adj Close 一致的价格尺度。"""
    adjustment = frame["adj_close"] / frame["close"]
    return (
        frame["open"] * adjustment,
        frame["high"] * adjustment,
        frame["low"] * adjustment,
        frame["adj_close"],
    )


def nyse_session_ids(dates: pd.Series) -> pd.Series:
    """Map dates to the official XNYS calendar; reject non-session observations."""
    values = pd.to_datetime(dates).dt.normalize()
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(values.min(), values.max())
    session_dates = pd.DatetimeIndex(sessions.tz_localize(None)).normalize()
    identifiers = session_dates.get_indexer(pd.DatetimeIndex(values))
    if (identifiers < 0).any():
        invalid = sorted(values[identifiers < 0].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"The daily line includes non-XNYS trading days: {invalid[:5]}")
    return pd.Series(identifiers, index=dates.index, dtype="int64")


def calculate_one_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """在单只股票的有序日频记录上计算前 26 个时序因子。"""
    result = pd.DataFrame(index=frame.index)
    open_price, high_price, low_price, close_price = adjusted_ohlc(frame)
    raw_close = frame["close"]
    volume = frame["volume"].astype(float)
    log_close = np.log(close_price)

    result["ret_1d"] = log_close.diff()
    for window in (5, 20, 60, 120):
        result[f"ret_{window}d"] = log_close - log_close.shift(window)
    result["momentum_20_5"] = log_close.shift(5) - log_close.shift(20)
    result["overnight_gap"] = np.log(open_price / close_price.shift(1))
    result["intraday_return"] = np.log(close_price / open_price)

    ema20 = close_price.ewm(span=20, adjust=False, min_periods=20).mean()
    ema60 = close_price.ewm(span=60, adjust=False, min_periods=60).mean()
    result["ema20_distance"] = close_price / ema20 - 1
    result["ema60_distance"] = close_price / ema60 - 1
    result["ema20_60_spread"] = ema20 / ema60 - 1

    price_change = close_price.diff()
    average_gain = price_change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-price_change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / (average_loss + EPSILON)
    result["rsi14_centered"] = ((100 - 100 / (1 + relative_strength)) - 50) / 50

    macd_line = close_price.ewm(span=12, adjust=False, min_periods=26).mean() - close_price.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()
    macd_histogram = macd_line - macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    true_range = pd.concat(
        [high_price - low_price, (high_price - close_price.shift(1)).abs(), (low_price - close_price.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=14).mean()
    result["macd_hist_atr"] = macd_histogram / (atr14 + EPSILON)

    high60 = high_price.rolling(60, min_periods=60).max()
    low60 = low_price.rolling(60, min_periods=60).min()
    result["breakout_position_60"] = (close_price - low60) / (high60 - low60 + EPSILON)

    result["realized_vol_5"] = result["ret_1d"].rolling(5, min_periods=5).std()
    result["realized_vol_20"] = result["ret_1d"].rolling(20, min_periods=20).std()
    result["vol_ratio_5_20"] = result["realized_vol_5"] / (result["realized_vol_20"] + EPSILON)
    result["parkinson_vol_20"] = np.sqrt(
        np.log(high_price / low_price).pow(2).rolling(20, min_periods=20).mean() / (4 * np.log(2))
    )
    result["overnight_vol_20"] = result["overnight_gap"].rolling(20, min_periods=20).std()
    result["downside_vol_20"] = np.sqrt(result["ret_1d"].clip(upper=0).pow(2).rolling(20, min_periods=20).mean())

    log_volume = np.log(volume.where(volume > 0))
    log_volume_mean = log_volume.rolling(20, min_periods=20).mean()
    log_volume_std = log_volume.rolling(20, min_periods=20).std()
    result["volume_zscore_20"] = (log_volume - log_volume_mean) / (log_volume_std + EPSILON)
    result["volume_trend_5_20"] = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean() - 1
    result["amihud_illiq_20"] = (result["ret_1d"].abs() / (raw_close * volume)).rolling(20, min_periods=20).mean()
    result["signed_volume_pressure_20"] = (
        np.sign(result["ret_1d"]) * volume
    ).rolling(20, min_periods=20).sum() / (volume.rolling(20, min_periods=20).sum() + EPSILON)
    result["close_location_value"] = (2 * close_price - high_price - low_price) / (high_price - low_price + EPSILON)
    result["body_range_ratio"] = (close_price - open_price) / (high_price - low_price + EPSILON)
    return result


def calculate_one_ticker_on_complete_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset every rolling indicator after a missing XNYS trading session."""
    segment = frame["session_id"].diff().ne(1).cumsum()
    return frame.groupby(segment, group_keys=False, sort=False).apply(calculate_one_ticker, include_groups=False)


def keep_last_trading_day_of_week(data: pd.DataFrame) -> pd.DataFrame:
    """每只股票每周仅保留最后一个实际交易日，用于周频模型训练。
    因子应先在完整日频记录上计算，再进行本筛选；这样保留的周末记录仍然
    包含该周每日价格信息形成的 rolling window 和技术指标。
    """
    weekly = data.sort_values(["ticker", "date"]).copy()
    weekly["_trading_week"] = weekly["date"].dt.to_period("W-FRI")
    weekly = (
        weekly.groupby(["ticker", "_trading_week"], group_keys=False, sort=False)
        .tail(1)
        .drop(columns="_trading_week")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return weekly


def add_weekly_open_to_close_target(data: pd.DataFrame) -> pd.DataFrame:
    """Assign each week its next-week-end log return using adjusted close prices."""
    output = data.sort_values(["ticker", "date"]).copy()
    output["_trading_week"] = output["date"].dt.to_period("W-FRI")
    week_end = (
        output.groupby(["ticker", "_trading_week"], group_keys=False, sort=False)
        .tail(1)[["ticker", "_trading_week", "adj_close"]]
        .rename(columns={"adj_close": "week_end_adj_close"})
    )
    week_end["target"] = week_end.groupby("ticker", sort=False)["week_end_adj_close"].transform(
        lambda prices: np.log(prices.shift(-1) / prices)
    )
    output = output.merge(week_end[["ticker", "_trading_week", "target"]], on=["ticker", "_trading_week"], how="left")
    return output.drop(columns="_trading_week")


def normalize_features_cross_sectionally(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Convert each week's valid factor values to robust 0--1 cross-sectional percentiles.

    A factor's lowest valid value is 0 and highest is 1 on a given weekly date.
    Missing values remain missing; target is deliberately not passed to this function.
    """
    output = data.copy()
    ranks = output.groupby("date")[features].rank(method="average")
    counts = output.groupby("date")[features].transform("count")
    normalized = (ranks - 1) / (counts - 1)
    # A one-stock cross-section has neutral rank 0.5; source NaNs must remain NaN.
    output[features] = normalized.where(counts > 1, 0.5).where(output[features].notna())
    return output


def classify_target(target: pd.Series) -> pd.Series:
    """Map the continuous forward return label to five ordered return classes."""
    output = pd.Series(pd.NA, index=target.index, dtype="Int8")
    valid = target.notna()
    output.loc[valid] = np.select(
        [target.loc[valid] < -0.03, target.loc[valid] < -0.01, target.loc[valid] <= 0.01, target.loc[valid] <= 0.03],
        [0, 1, 2, 3],
        default=4,
    )
    return output


def calculate_internal_one_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate five additional internal factors on consecutive daily sessions."""
    open_price = frame["open"] * (frame["adj_close"] / frame["close"])
    close_price = frame["adj_close"]
    volume = frame["volume"].astype(float)
    result = pd.DataFrame(index=frame.index)
    returns = close_price.pct_change()
    session_complete = lambda window: frame["session_id"].sub(frame["session_id"].shift(window - 1)).eq(window - 1)

    x = np.arange(20, dtype=float)
    sum_y = close_price.rolling(20, min_periods=20).sum()
    sst = close_price.pow(2).rolling(20, min_periods=20).sum() - sum_y.pow(2) / 20
    sum_xy = close_price.rolling(20, min_periods=20).apply(lambda values: float(np.dot(x, values)), raw=True)
    trend_r2 = (sum_xy - x.mean() * sum_y).pow(2) / (float(((x - x.mean()) ** 2).sum()) * sst)
    result["trend_r2_20"] = trend_r2.where(session_complete(20) & (sst > EPSILON)).clip(0, 1)
    result["return_autocorr_20"] = returns.rolling(20, min_periods=20).corr(returns.shift(1)).where(session_complete(22))

    volume_mean = volume.rolling(20, min_periods=20).mean()
    result["volume_cv_20"] = (volume.rolling(20, min_periods=20).std(ddof=1) / volume_mean).where(
        session_complete(20) & (volume_mean > 0)
    )
    volume_change = np.log(volume.where(volume > 0)).diff()
    result["return_volume_corr_20"] = returns.rolling(20, min_periods=20).corr(volume_change).where(session_complete(21))
    gap = open_price / close_price.shift(1) - 1
    intraday = close_price / open_price - 1
    result["gap_followthrough_20"] = gap.rolling(20, min_periods=20).corr(intraday).where(session_complete(21))
    for name in ("return_autocorr_20", "return_volume_corr_20", "gap_followthrough_20"):
        result[name] = result[name].clip(-1, 1)
    return result


def internal_factors(weekly: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily internal signals and select exactly the weekly output keys."""
    daily = prices.sort_values(["ticker", "date"]).copy()
    daily["session_id"] = nyse_session_ids(daily["date"])
    values = daily.groupby("ticker", group_keys=False, sort=False).apply(calculate_internal_one_ticker, include_groups=False)
    calculated = pd.concat([daily[["date", "ticker"]], values.reindex(daily.index)], axis=1)
    return weekly[["date", "ticker"]].merge(calculated, on=["date", "ticker"], how="left", validate="one_to_one")


def write_factor_definitions() -> None:
    """Write definitions for 45 candidate factors plus two non-feature labels."""
    DATA_DIR.mkdir(exist_ok=True)
    definitions = pd.DataFrame(
        FACTOR_DEFINITIONS + ADDITIONAL_FACTOR_DEFINITIONS + EXTERNAL_FACTOR_DEFINITIONS + [TARGET_DEFINITION, TARGET2_DEFINITION],
        columns=["factor", "calculation"],
    )
    if definitions["factor"].duplicated().any() or len(definitions) != 47:
        raise RuntimeError("定义表必须包含 45 个不重复因子和 2 个标签。")
    definitions.to_csv(DEFINITIONS_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}。请先运行 fetch_and_clean.py。")
    prices = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    if missing := required - set(prices.columns):
        raise ValueError(f"输入数据缺少字段: {sorted(missing)}")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["session_id"] = nyse_session_ids(prices["date"])
    time_series_names = [name for name, _ in FACTOR_DEFINITIONS[:26]]
    factors = (
        prices.groupby("ticker", group_keys=False, sort=False)
        .apply(calculate_one_ticker_on_complete_sessions, include_groups=False)
        [time_series_names]
        .reindex(prices.index)
    )
    # session_id is an implementation check, not a model feature or output column.
    dataset = pd.concat([prices.drop(columns="session_id"), factors], axis=1)

    # 仅在同一日期、具备有效数值的股票中计算截面百分位排名。
    dataset["cs_rank_ret20"] = dataset.groupby("date")["ret_20d"].rank(pct=True, method="average")
    dataset["cs_rank_ret60"] = dataset.groupby("date")["ret_60d"].rank(pct=True, method="average")
    dataset["cs_rank_vol20"] = dataset.groupby("date")["realized_vol_20"].rank(pct=True, method="average")
    dataset["cs_rank_amihud20"] = dataset.groupby("date")["amihud_illiq_20"].rank(pct=True, method="average")

    # 预热期只服务于计算，不进入十年建模样本。
    dataset = dataset.loc[dataset["date"].between(MODEL_START, MODEL_END)].reset_index(drop=True)
    dataset = add_weekly_open_to_close_target(dataset)
    weekly_dataset = keep_last_trading_day_of_week(dataset)
    print("计算 5 个新增内部因子 ...")
    internal = internal_factors(weekly_dataset, prices)
    final = weekly_dataset[["date", "ticker", *BASE_PRICE_COLUMNS, "target", *[name for name, _ in FACTOR_DEFINITIONS]]].merge(
        internal, on=["date", "ticker"], how="left", validate="one_to_one"
    )
    if len(final) != len(weekly_dataset) or final.duplicated(["date", "ticker"]).any():
        raise RuntimeError("新增因子合并改变了原始 date+ticker 键；未写入输出文件。")
    values = final[INTERNAL_FACTOR_NAMES].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise RuntimeError("新增因子中出现无穷值；未写入输出文件。")

    final = final[["date", "ticker", *BASE_PRICE_COLUMNS, *INTERNAL_FACTOR_NAMES, "target"]]
    final = normalize_features_cross_sectionally(final, INTERNAL_FACTOR_NAMES)
    final["target2"] = classify_target(final["target"])
    output_columns = ["date", "ticker", *BASE_PRICE_COLUMNS, *INTERNAL_FACTOR_NAMES, "target", "target2"]
    rows_before_missing_drop = len(final)
    final = final.dropna(subset=output_columns).reset_index(drop=True)
    final["target2"] = final["target2"].astype("int8")
    if final.empty:
        raise RuntimeError("日期范围内没有完整的 OHLCV、35 个因子、target 和 target2 记录；未写入输出文件。")
    DATA_DIR.mkdir(exist_ok=True)
    final.to_csv(WEEKLY_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    write_factor_definitions()

    factor_names = INTERNAL_FACTOR_NAMES
    complete = final.dropna(subset=factor_names)
    print(f"已保存周频训练数据：{WEEKLY_OUTPUT_FILE}")
    print(f"行数：{len(final)}；因缺失值删除：{rows_before_missing_drop - len(final)}")
    print("内部因子有效记录数：")
    print(final[INTERNAL_FACTOR_NAMES].notna().sum().to_string())
    print("35 个因子已按每个周末做横截面百分位归一化至 0--1；target 和 target2 均为标签，不能作为模型输入。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the weekly 35-internal-factor candidate panel.")
    parser.add_argument("--definitions-only", action="store_true", help="Only refresh technical_factor_definitions.csv.")
    arguments = parser.parse_args()
    if arguments.definitions_only:
        write_factor_definitions()
        print(f"已保存 45 个因子定义、target 和 target2 标签：{DEFINITIONS_FILE}")
    else:
        main()
