"""The project's fifteen fixed portfolio rules, independent of realized returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = (
    "Top5", "Top10", "Top5_PositiveScore", "Top10_PositiveScore",
    "Top5_NoLoss4w", "Top10_NoLoss4w", "Top5_NoLoss13w", "Top10_NoLoss13w",
    "Top5_RankWeighted", "Top5_LowVol26w", "Top5_RankRisk26w", "Top5_Buffer10",
    "Top5_Rebalance2w", "Top5_Rebalance4w", "Top5_Trend26w",
)
STATEFUL = ("Top5_Buffer10", "Top5_Rebalance2w", "Top5_Rebalance4w")
RULES = {
    "Top5": "按分数降序、代码升序选前5，每只20%，缺额留现金。",
    "Top10": "按分数降序、代码升序选前10，每只10%，缺额留现金。",
    "Top5_PositiveScore": "先保留分数>0的股票，再选前5，每只20%。",
    "Top10_PositiveScore": "先保留分数>0的股票，再选前10，每只10%。",
    "Top5_NoLoss4w": "先保留截至信号日过去4周价格收益非负的股票，再选前5，每只20%。",
    "Top10_NoLoss4w": "先保留截至信号日过去4周价格收益非负的股票，再选前10，每只10%。",
    "Top5_NoLoss13w": "先保留截至信号日过去13周价格收益非负的股票，再选前5，每只20%。",
    "Top10_NoLoss13w": "先保留截至信号日过去13周价格收益非负的股票，再选前10，每只10%。",
    "Top5_RankWeighted": "前5按5:4:3:2:1/15加权，缺额名次的权重留现金。",
    "Top5_LowVol26w": "在有26周完整收益历史的股票中选前5，波动率倒数加权，单股上限35%，不足5只时预算为只数/5。",
    "Top5_RankRisk26w": "在有26周完整收益历史的股票中选前5，按(6-名次)/波动率加权，单股上限35%，不足5只时预算为只数/5。",
    "Top5_Buffer10": "保留仍在当前前10的原持仓，按排名补足5只，每只20%。",
    "Top5_Rebalance2w": "从首次信号起每2周选前5、每只20%；间隔周权重漂移，缺信号清仓，恢复信号立即重启。",
    "Top5_Rebalance4w": "从首次信号起每4周选前5、每只20%；间隔周权重漂移，缺信号清仓，恢复信号立即重启。",
    "Top5_Trend26w": "截至信号日的股票等权指数高于其26周均线时选前5，每只20%，否则现金。",
}


def inputs(prices: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Only current and past prices enter selection/risk inputs."""
    past = prices / prices.shift(1) - 1
    volatility = past.rolling(26, min_periods=26).std(ddof=1)
    market_index = (1 + past.mean(axis=1).fillna(0)).cumprod()
    trend = market_index.gt(market_index.rolling(26, min_periods=26).mean())
    return {"vol": volatility, "trend": trend, "market_index": market_index,
            "lag4": prices / prices.shift(4) - 1, "lag13": prices / prices.shift(13) - 1}


def capped_weights(scores: dict[str, float], budget: float = 1.0, cap: float = 0.35) -> dict[str, float]:
    """Redistribute excess above cap; an infeasible remainder stays in cash."""
    if not 0 <= budget <= 1 or not 0 < cap <= 1:
        raise ValueError("budget must be in [0,1] and cap in (0,1]")
    active = {ticker: float(value) for ticker, value in scores.items() if np.isfinite(value) and value > 0}
    allocation = dict.fromkeys(active, 0.0)
    remaining = min(budget, len(active) * cap)
    while active and remaining > 1e-14:
        total = sum(active.values())
        proposed = {ticker: remaining * value / total for ticker, value in active.items()}
        clipped = [ticker for ticker, weight in proposed.items() if weight > cap + 1e-14]
        if not clipped:
            allocation.update(proposed)
            break
        for ticker in clipped:
            allocation[ticker] = cap
            remaining -= cap
            del active[ticker]
    return allocation


def choose_weights(
    strategy: str, group: pd.DataFrame, date: pd.Timestamp, context: dict,
    previous: dict[str, float], drift: dict[str, float], week_index: int,
    last_rebalance: int | None,
) -> tuple[dict[str, float], bool, int | None]:
    """Select using information available on ``date``; no future returns accepted."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if group.empty:
        return {}, True, None
    ranked = group.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
    k = 5 if strategy.startswith("Top5") else 10
    if strategy in ("Top5_Rebalance2w", "Top5_Rebalance4w"):
        period = 2 if strategy.endswith("2w") else 4
        if previous and last_rebalance is not None and week_index - last_rebalance < period:
            return dict(drift), False, last_rebalance
        return dict.fromkeys(ranked.head(k)["ticker"], 1 / k), True, week_index
    if strategy == "Top5_Buffer10":
        keep = [ticker for ticker in ranked.head(10)["ticker"] if ticker in previous]
        additions = [ticker for ticker in ranked["ticker"] if ticker not in keep][:k - len(keep)]
        return dict.fromkeys(keep + additions, 1 / k), True, week_index
    if strategy == "Top5_Trend26w" and not bool(context["trend"].loc[date]):
        return {}, True, week_index
    selected = ranked
    if "PositiveScore" in strategy:
        selected = selected.loc[selected["score"].gt(0)]
    if "NoLoss" in strategy:
        lag = "lag4" if strategy.endswith("4w") else "lag13"
        selected = selected.loc[selected["ticker"].map(context[lag].loc[date]).ge(0)]
    if strategy in ("Top5_LowVol26w", "Top5_RankRisk26w"):
        selected = selected.loc[selected["ticker"].map(context["vol"].loc[date]).notna()]
    selected = selected.head(k)
    if strategy in ("Top5_LowVol26w", "Top5_RankRisk26w"):
        risk_scores = {
            ticker: (5 - rank if strategy == "Top5_RankRisk26w" else 1)
            / max(float(context["vol"].loc[date, ticker]), 1e-8)
            for rank, ticker in enumerate(selected["ticker"])
        }
        weights = capped_weights(risk_scores, budget=len(selected) / k)
    elif strategy == "Top5_RankWeighted":
        weights = {ticker: (5 - rank) / 15 for rank, ticker in enumerate(selected["ticker"])}
    else:
        weights = dict.fromkeys(selected["ticker"], 1 / k)
    return weights, True, week_index
