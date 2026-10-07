"""Backward-looking portfolio inputs and fixed, auditable strategy rules."""
import numpy as np
import pandas as pd

NEW_STRATEGIES=['Top5_LowVol26w','Top5_RankRisk26w','Top5_Buffer10','Top5_Rebalance2w','Top5_Rebalance4w','Top5_Trend26w']
STATEFUL=['Top5_Buffer10','Top5_Rebalance2w','Top5_Rebalance4w']
CONSENSUS='CONSENSUS-R3-F11-4MODELS'
COMPONENTS=['RF-R3-F11','RIDGE-R3-F11','XGB-R3-F11','RANKER-V2-ENSEMBLE']
LABELS={'Top5':'等权Top5','Top10':'等权Top10','Top5_RankWeighted':'排名加权Top5',
        'Top5_PositiveScore':'Top5正分数','Top10_PositiveScore':'Top10正分数',
        'Top5_NoLoss4w':'Top5过去4周非亏损','Top10_NoLoss4w':'Top10过去4周非亏损',
        'Top5_NoLoss13w':'Top5过去13周非亏损','Top10_NoLoss13w':'Top10过去13周非亏损',
        'Top5_LowVol26w':'低波动加权','Top5_RankRisk26w':'排名风险加权','Top5_Buffer10':'前10保留缓冲',
        'Top5_Rebalance2w':'每2周调仓','Top5_Rebalance4w':'每4周调仓','Top5_Trend26w':'市场趋势过滤'}

def inputs(prices):
    past=prices/prices.shift(1)-1
    vol=past.rolling(26,min_periods=26).std(ddof=1)
    market_index=(1+past.mean(axis=1).fillna(0)).cumprod()
    trend=(market_index>market_index.rolling(26,min_periods=26).mean()).fillna(False)
    return dict(vol=vol,trend=trend,stock_returns=prices.shift(-1)/prices-1,
                market_index=market_index,prices=prices)

def capped_weights(scores,budget=1.,cap=.35):
    """Redistribute weight exceeding the cap; any infeasible budget stays cash."""
    scores={t:float(v) for t,v in scores.items() if np.isfinite(v) and v>0}
    allocation={t:0. for t in scores};remaining=min(float(budget),len(scores)*cap)
    active=dict(scores)
    while active and remaining>1e-14:
        total=sum(active.values())
        proposed={t:remaining*s/total for t,s in active.items()}
        clipped=[t for t,w in proposed.items() if w>cap+1e-14]
        if not clipped:
            allocation.update(proposed);break
        for t in clipped:allocation[t]=cap;remaining-=cap;active.pop(t)
    return allocation

def consensus_predictions(ready):
    joint=None
    for component in COMPONENTS:
        p=ready[component][['date','ticker','target','prediction']].copy()
        p['percentile']=p.groupby('date').prediction.rank(method='average',pct=True)
        p=p[['date','ticker','target','percentile']].rename(columns={'percentile':component})
        if joint is None:joint=p
        else:
            joint=joint.merge(p,on=['date','ticker'],how='inner',suffixes=('','_check'),validate='one_to_one')
            assert np.allclose(joint.target,joint.target_check,atol=1e-12,rtol=0)
            joint=joint.drop(columns='target_check')
    assert len(joint)>0
    # Arithmetic means of rational ranks can differ at ~1e-16 for true ties.
    # Freeze precision before ranking or CSV serialization, then break ties by ticker.
    joint['prediction']=joint[COMPONENTS].mean(axis=1).round(12)
    return joint[['date','ticker','target','prediction']],joint

RULES={
 'Top5_LowVol26w':'按预测选前5只拥有26周完整历史波动率的股票，按波动率倒数分配权重，单只上限35%；不足5只时股票总预算为只数/5，余款现金。',
 'Top5_RankRisk26w':'同一可用股票池取前5名，按(6-排名)/过去26周周收益标准差分配权重，单只上限35%；不足5只时股票总预算为只数/5。',
 'Top5_Buffer10':'初次选前5；每周保留仍在当前前10名的持仓，空位按当前排名补齐，然后等权配平，每只20%。',
 'Top5_Rebalance2w':'首次可用信号从现金买前5，每隔2个日历周重新等权选股；期间持仓及现金权重自然漂移，不按新排名交易。缺信号清仓，恢复信号立即建仓并重设调仓计时。',
 'Top5_Rebalance4w':'同上，每隔4个日历周调仓。',
 'Top5_Trend26w':'以原始股票池截至当周的可核验复权价格构造等权收益指数；指数高于含当周的过去26周均线时等权买Top5，否则持现金。',
 'Consensus':'四个固定配置的当周自身股票池排名百分位等权平均，分数保留12位小数统一数值并列，再按股票代码打破并列；仅使用四方都有预测的股票和日期，未按本次成绩选择组件。派生序列也评估全部15策略。',
}
