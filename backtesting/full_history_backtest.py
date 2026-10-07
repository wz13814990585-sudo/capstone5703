"""Replay every supplied score over its own complete available signal history."""
from pathlib import Path
import sys,json,zipfile,hashlib,html,platform
import pandas as pd
import numpy as np
import run as engine
from portfolio_strategies import NEW_STRATEGIES,STATEFUL,CONSENSUS,COMPONENTS,RULES,inputs,capped_weights,consensus_predictions

ROOT=engine.ROOT
OUT=Path(__file__).resolve().parent/'full_history'
PREDS=OUT/'predictions'
NS=OUT/'native_predictions'
STRATEGIES=['Top5','Top10','Top5_PositiveScore','Top10_PositiveScore','Top5_NoLoss4w','Top10_NoLoss4w','Top5_NoLoss13w','Top10_NoLoss13w','Top5_RankWeighted']+NEW_STRATEGIES
checks=0

def csv(path,frame):frame.to_csv(path,index=False)

def load_sources():
    records=[];sources=[]
    for file in sorted(NS.glob('*.csv')):
        p=pd.read_csv(file,parse_dates=['date'])
        records.append((file.stem,p,'native_code_replay',str(file)))
    with zipfile.ZipFile(ROOT/'xgboost-ranker第二版.zip') as z:
        for n in sorted(z.namelist()):
            if not (n.endswith('.csv') and 'prediction' in n.lower() and not n.startswith('__MACOSX') and '.venv/' not in n):continue
            d=pd.read_csv(z.open(n),parse_dates=['date'])
            source=n.split('/data/')[-1]
            source_hash=hashlib.sha256(z.read(n)).hexdigest()
            sources.append(dict(source=source,rows=len(d),first=d.date.min(),last=d.date.max(),sha256=source_hash))
            if 'ensemble_ranking_score' in d:
                prefix=('RANKER-V2' if '/XGboost_ranker/' in n else 'RANKER-CONTINUOUS-FINAL' if '/continuous_target_final_run/' in n
                    else 'RANKER-C' if '/experiment_C_continuous/' in n else 'RANKER-D')
                for component,col in [('ENSEMBLE','ensemble_ranking_score'),('M1','model_1_raw_score'),('M2','model_2_raw_score'),('M3','model_3_raw_score')]:
                    p=d[['date','ticker','target',col]].rename(columns={col:'prediction'})
                    records.append((f'{prefix}-{component}',p,'supplied_prediction',source+'::'+col))
            else:
                keys=[c for c in ['factor_set_name','training_mode','config_id'] if c in d]
                for group,p in d.groupby(keys,dropna=False,sort=True):
                    vals=group if isinstance(group,tuple) else (group,)
                    prefix='RANKER-HOLDOUT' if '/rankic_optimization/' in n else 'RANKER-LEGACY' if '/legacy_priority7/' in n else 'RANKER-HISTORY'
                    name=prefix+'-'+'-'.join(str(v) for v in vals)
                    p=p[['date','ticker','actual_return','ranking_score']].rename(columns={'actual_return':'target','ranking_score':'prediction'})
                    records.append((name,p,'supplied_prediction',source))
    csv(OUT/'source_files.csv',pd.DataFrame(sources))
    return records

def prepare(records,raw,write_audits=True):
    global checks
    dates=sorted(raw.date.unique());next_dates=dict(zip(dates[:-1],dates[1:]))
    indexed=raw.set_index(['date','ticker']).adj_close
    label=raw[['date','ticker','target','adj_close']].rename(columns={'target':'data_target'})
    label['label_end_date']=label.date.map(next_dates)
    label['next_price']=indexed.reindex(pd.MultiIndex.from_arrays([label.label_end_date,label.ticker])).to_numpy()
    ready={};coverage=[];excluded=[];gaps=[]
    for name,p,origin,source in records:
        assert not p.duplicated(['date','ticker']).any(),name
        p=p.drop(columns=['label_end_date'],errors='ignore').merge(label,on=['date','ticker'],how='left',validate='one_to_one',indicator=True)
        assert p._merge.eq('both').all(),name
        assert np.isfinite(p[['prediction','target']].to_numpy()).all(),name
        assert np.max(abs(p.target-p.data_target))<1e-10,name
        known=p.label_end_date.notna()&np.isfinite(p.next_price)
        error=np.max(abs(np.log(p.loc[known,'next_price']/p.loc[known,'adj_close'])-p.loc[known,'target']))
        assert error<1e-10,(name,error)
        bad=p.loc[~known,['date','ticker']].copy();bad['experiment']=name
        bad['reason']=np.where(p.loc[~known,'label_end_date'].isna(),'terminal_next_week_unavailable','next_global_week_stock_price_missing')
        excluded.append(bad)
        first,last,observed=p.date.min(),p.date.max(),set(p.date)
        missing_dates=[d for d in dates if first<=d<=last and d not in observed]
        gaps.extend(dict(experiment=name,date=d,reason='source_has_no_prediction_on_this_week') for d in missing_dates)
        coverage.append(dict(experiment=name,origin=origin,source=source,input_rows=len(p),input_weeks=p.date.nunique(),input_first=p.date.min(),input_last=p.date.max(),
            usable_rows=int(known.sum()),excluded_rows=int((~known).sum()),signal_first=p.loc[known,'date'].min(),signal_last=p.loc[known,'date'].max(),
            realization_last=p.loc[known,'label_end_date'].max(),weeks=p.loc[known,'date'].nunique(),missing_source_weeks=len(missing_dates),label_max_error=error))
        q=p.loc[known].drop(columns=['_merge','data_target','adj_close','next_price'])
        ready[name]=q;csv(PREDS/f'{name}.csv',q);checks+=5
    if write_audits:
        csv(OUT/'coverage.csv',pd.DataFrame(coverage));csv(OUT/'excluded_rows.csv',pd.concat(excluded,ignore_index=True))
        csv(OUT/'missing_source_weeks.csv',pd.DataFrame(gaps,columns=['experiment','date','reason']))
    return ready,pd.DataFrame(coverage)

def precompute(p,lagreturns,calendar=None,market=None,next_dates=None,context=None,strategies=None):
    rows=[];positions=[]
    grouped={d:g.sort_values(['prediction','ticker'],ascending=[False,True],kind='stable') for d,g in p.groupby('date',sort=True)}
    weeks=[(d,grouped.get(d,p.iloc[:0])) for d in (calendar if calendar is not None else sorted(grouped))]
    for strategy in (STRATEGIES if strategies is None else strategies):
        k=5 if strategy.startswith('Top5') else 10
        previous={};prior_returns={};prior_cash=1.;last_rebalance=None
        for week_index,(date,g) in enumerate(weeks):
            growth=prior_cash+sum(w*(1+prior_returns[t]) for t,w in previous.items())
            drift={t:w*(1+prior_returns[t])/growth for t,w in previous.items()}
            selected=g
            if 'PositiveScore' in strategy:selected=selected[selected.prediction.gt(0)]
            if 'NoLoss' in strategy:
                lag=4 if '4w' in strategy else 13
                selected=selected[selected.ticker.map(lagreturns[lag].loc[date]).ge(0)]
            if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                selected=selected[selected.ticker.map(context['vol'].loc[date]).notna()]
            rebalance_event=True;forced_exit_count=0
            if strategy=='Top5_Buffer10':
                keep=[t for t in g.head(10).ticker if t in previous]
                additions=[t for t in g.ticker if t not in keep][:k-len(keep)]
                weights=dict.fromkeys(keep+additions,1/k)
            elif strategy in ['Top5_Rebalance2w','Top5_Rebalance4w']:
                period=2 if '2w' in strategy else 4
                if not len(g):
                    weights={};last_rebalance=None
                elif not previous or last_rebalance is None or week_index-last_rebalance>=period:
                    weights=dict.fromkeys(g.head(k).ticker,1/k);last_rebalance=week_index
                else:
                    rebalance_event=False
                    usable=context['stock_returns'].loc[date]
                    weights={t:w for t,w in drift.items() if pd.notna(usable.get(t,np.nan))}
                    forced_exit_count=len(drift)-len(weights)
            elif strategy=='Top5_Trend26w' and not context['trend'].loc[date]:weights={}
            else:
                selected=selected.head(k)
                if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                    scores={t:(5-rank if strategy=='Top5_RankRisk26w' else 1)/max(float(context['vol'].loc[date,t]),1e-8)
                            for rank,t in enumerate(selected.ticker)}
                    weights=capped_weights(scores,budget=len(selected)/k)
                else:
                    weights=(dict(zip(selected.ticker,[(5-rank)/15 for rank in range(len(selected))]))
                             if strategy=='Top5_RankWeighted' else dict.fromkeys(selected.ticker,1/k))
            cash_weight=1-sum(weights.values())
            source_returns=dict(zip(g.ticker,np.expm1(g.target)))
            returns={t:source_returns[t] if t in source_returns else float(context['stock_returns'].loc[date,t]) for t in weights}
            assert all(np.isfinite(list(returns.values())))
            assert -1e-12<=cash_weight<=1+1e-12
            traded=sum(abs(weights.get(t,0)-drift.get(t,0)) for t in set(weights)|set(drift))
            rows.append(dict(date=date,label_end_date=g.label_end_date.iloc[0] if len(g) else next_dates[date],strategy=strategy,
                gross_return=sum(weights[t]*returns[t] for t in weights),benchmark_return=np.expm1(g.target).mean() if len(g) else market[date],
                traded_weight=traded,holdings=len(weights),cash_weight=cash_weight,no_signal_cash_week=not len(g),
                rebalance_event=rebalance_event,forced_exit_count=forced_exit_count))
            positions.extend(dict(date=date,strategy=strategy,ticker=t,weight=w,realized_return=returns[t]) for t,w in weights.items())
            previous,prior_returns,prior_cash=weights,returns,cash_weight
    return pd.DataFrame(rows),pd.DataFrame(positions)

def windows(p,calendar=None):
    dates=sorted(p.date.unique()) if calendar is None else calendar
    result={'FullHistory':dates}
    for n in [13,26,52,104,156]:
        if len(dates)>=n:result[f'Last{n}w']=dates[-n:]
    for year in sorted(set(pd.Timestamp(d).year for d in dates)):result[f'Year{year}']=[d for d in dates if pd.Timestamp(d).year==year]
    for stage,start,end in [('Development3y','2019-09-01','2024-09-01'),('Development6y','2022-09-01','2024-09-01'),('Validation','2024-09-01','2025-09-01'),('FinalTest','2025-09-01','2026-09-01')]:
        chosen=[d for d in dates if pd.Timestamp(start)<=d<pd.Timestamp(end)]
        if chosen:result[stage]=chosen
    return result

def render(p,folder,name,title):
    series=[]
    for strategy,g in p.groupby('strategy',sort=False):
        ds=[g.date.iloc[0]]+list(g.label_end_date)
        series.append((strategy,ds,np.r_[1.,np.cumprod(1+g.net_return)]))
    g=p[p.strategy.eq('Top5')]
    series.append(('EqualWeight',[g.date.iloc[0]]+list(g.label_end_date),np.r_[1.,np.cumprod(1+g.benchmark_return)]))
    engine.svg_chart(folder/f'{name}.svg',title,series)

def main():
    global checks
    OUT.mkdir(exist_ok=True);PREDS.mkdir(exist_ok=True);chart=OUT/'charts';chart.mkdir(exist_ok=True)
    engine.COLORS=engine.COLORS[:8]+['#a16207','#7c3aed','#0f766e','#c026d3','#0284c7','#be123c','#65a30d','#111827']
    raw=pd.read_csv(ROOT/'test_45_factors_weekly.csv',parse_dates=['date'])
    ready,coverage=prepare(load_sources(),raw)
    consensus,joint=consensus_predictions(ready)
    csv(OUT/'consensus_components.csv',joint)
    consensus_ready,consensus_coverage=prepare([(CONSENSUS,consensus,'derived_consensus','equal percentile average: '+','.join(COMPONENTS))],raw,write_audits=False)
    assert consensus_coverage.excluded_rows.sum()==0
    ready.update(consensus_ready);coverage=pd.concat([coverage,consensus_coverage],ignore_index=True)
    dates_all=sorted(raw.date.unique())
    consensus_gaps=[d for d in dates_all if consensus.date.min()<=d<=consensus.date.max() and d not in set(consensus.date)]
    gaps=pd.read_csv(OUT/'missing_source_weeks.csv',parse_dates=['date'])
    gaps=pd.concat([gaps,pd.DataFrame([dict(experiment=CONSENSUS,date=d,reason='consensus_component_missing') for d in consensus_gaps])],ignore_index=True)
    csv(OUT/'missing_source_weeks.csv',gaps)
    engine.dump(OUT/'strategy_rules.json',dict(strategies=STRATEGIES,rules=RULES,consensus_components=COMPONENTS,
        consensus_component_weights=[.25]*4,selection='fixed R3-F11 designs plus original Ranker, no outcome-based selection',risk_window=26,stock_cap=.35))
    status=[]
    for family in ['RF','XGB','RIDGE']:
        for design in ['R3','R6','FIXED']:
            for factors in [7,11,45]:
                name=f'{family}-{design}-F{factors}'
                status.append(dict(experiment=name,status='native_replay_available' if name in ready else 'no_candidate_passed',
                    source='original code + recorded parameters',available=name in ready))
    status.append(dict(experiment='RANKER-V2-ENSEMBLE',status='supplied_prediction_available',source='XGboost_ranker/data/stock_predictions.csv',available=True))
    csv(OUT/'official_model_status.csv',pd.DataFrame(status))
    prices=raw.pivot(index='date',columns='ticker',values='adj_close')
    context=inputs(prices)
    csv(OUT/'market_trend.csv',pd.DataFrame(dict(date=prices.index,market_index=context['market_index'],
        mean26=context['market_index'].rolling(26,min_periods=26).mean(),risk_on=context['trend'])))
    lagreturns={n:prices/prices.shift(n)-1 for n in [4,13]}
    rawdates=list(prices.index);next_dates=dict(zip(rawdates[:-1],rawdates[1:]))
    future=prices.shift(-1);valid_prices=(prices.notna()&future.notna())
    market=(future/prices-1).where(valid_prices).mean(axis=1).to_dict()
    common_keys=set.intersection(*(set(zip(p.date,p.ticker)) for p in ready.values()))
    metrics=[];summaries=[];weekly=[];holdings=[];common_weekly=[];stateful_holdings=[]
    for name,p in ready.items():
        calendar=[d for d in rawdates if p.date.min()<=d<=p.date.max()]
        coverage.loc[coverage.experiment.eq(name),'backtest_calendar_weeks']=len(calendar)
        coverage.loc[coverage.experiment.eq(name),'no_signal_cash_weeks']=len(calendar)-p.date.nunique()
        for window,dates in windows(p,calendar).items():
            q=p[p.date.isin(dates)]
            m=engine.evaluate(q)
            if name.startswith('RANKER') or name==CONSENSUS:m['mae']=m['rmse']=np.nan
            metrics.append(dict(experiment=name,window=window,first=q.date.min(),last=q.date.max(),**m))
        full,pos=precompute(p,lagreturns,calendar,market,next_dates,context)
        holdings.append(pos.assign(experiment=name))
        for window,dates in windows(p,calendar).items():
            base=full[full.date.isin(dates)].copy()
            if window!='FullHistory':
                replay,window_pos=precompute(p[p.date.isin(dates)],lagreturns,dates,market,next_dates,context,STATEFUL)
                base=pd.concat([base[~base.strategy.isin(STATEFUL)],replay],ignore_index=True)
                stateful_holdings.append(window_pos.assign(experiment=name,window=window))
            # All windows start from cash. Subsequent drift is unchanged inside a contiguous interval.
            mask=base.date.eq(base.date.min());base.loc[mask,'traded_weight']=1-base.loc[mask,'cash_weight']
            for cost in [0,10,25]:
                w=base.copy();w['transaction_cost_fraction']=w.traded_weight*cost/10000
                w['net_return']=(1-w.transaction_cost_fraction)*(1+w.gross_return)-1
                for strategy,g in w.groupby('strategy'):
                    values=engine.performance(g)
                    nav=np.r_[1.,np.cumprod(1+g.net_return)]
                    assert abs(values['total_return']-(nav[-1]-1))<1e-10
                    assert abs(values['max_drawdown']-(nav/np.maximum.accumulate(nav)-1).min())<1e-10
                    checks+=2
                    summaries.append(dict(experiment=name,window=window,cost_bps=cost,strategy=strategy,**values))
                if cost==10:
                    weekly.append(w.assign(experiment=name,window=window,cost_bps=cost))
                    render(w,chart,f'{name}_{window}',f'{name} | {window} | 10bps')
        q=p[[k in common_keys for k in zip(p.date,p.ticker)]]
        if len(q):
            # Auxiliary comparison keeps the previous sparse shared-date convention.
            w,_=precompute(q,lagreturns,context=context);w['transaction_cost_fraction']=w.traded_weight*.001
            w['net_return']=(1-w.transaction_cost_fraction)*(1+w.gross_return)-1
            common_weekly.append(w.assign(experiment=name,window='CommonAuxiliary',cost_bps=10))
            for strategy,g in w.groupby('strategy'):summaries.append(dict(experiment=name,window='CommonAuxiliary',cost_bps=10,strategy=strategy,**engine.performance(g)))
        print('Full history backtested',name,p.date.min().date(),p.date.max().date(),p.date.nunique(),flush=True)
    summary=pd.DataFrame(summaries);week=pd.concat(weekly,ignore_index=True)
    csv(OUT/'coverage.csv',coverage)
    csv(OUT/'strategy_comparison.csv',summary);csv(OUT/'model_evaluation.csv',pd.DataFrame(metrics))
    csv(OUT/'weekly_backtest.csv',week);csv(OUT/'positions.csv',pd.concat(holdings,ignore_index=True))
    csv(OUT/'stateful_window_positions.csv',pd.concat(stateful_holdings,ignore_index=True))
    if common_weekly:csv(OUT/'common_auxiliary_weekly.csv',pd.concat(common_weekly,ignore_index=True))
    # Full-history native-family plots use original start dates, without imposing a common starting date.
    for family in ['RF','RIDGE','XGB']:
        for factors in [7,11,45]:
            lines=[]
            for design in ['R3','R6','FIXED']:
                exp=f'{family}-{design}-F{factors}'
                g=week[week.experiment.eq(exp)&week.window.eq('FullHistory')&week.strategy.eq('Top5')]
                if len(g):lines.append((design,[g.date.iloc[0]]+list(g.label_end_date),np.r_[1.,np.cumprod(1+g.net_return)]))
            if lines:engine.svg_chart(chart/f'{family}_F{factors}_native_history_windows.svg',f'{family} F{factors} | original full histories | Top5 | unequal starts',lines)
    fullsummary=summary[summary.window.eq('FullHistory')&summary.cost_bps.eq(10)]
    csv(OUT/'full_history_comparison.csv',fullsummary)
    table=coverage[['experiment','origin','input_first','input_last','signal_first','signal_last','realization_last','weeks','usable_rows','excluded_rows']].to_html(index=False)
    links=''.join(f'<li><a href="charts/{f.name}">{html.escape(f.stem)}</a></li>' for f in sorted(chart.glob('*.svg')))
    OUT.joinpath('index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}img{max-width:1120px;width:100%}</style><h1>全部模型原始完整历史回测</h1><p>按每个模型自身预测起止时间回测，无2022年统一截断，无主回测共同样本裁剪。末周缺少下一期实现价格的预测保留审计记录，无法算作已实现收益。</p><p><a href="报告.md">中文报告</a> · <a href="coverage.csv">模型时间覆盖表</a> · <a href="full_history_comparison.csv">全历史指标</a></p><h2>各模型数据时间与覆盖</h2>'+table+'<h2>正式Ranker完整历史</h2><img src="charts/RANKER-V2-ENSEMBLE_FullHistory.svg"><h2>全部图表</h2><ul>'+links+'</ul>',encoding='utf-8')
    report=f'''# 全部模型完整历史回测（修正版）

本版纠正上一版将所有模型裁剪到2022年之后的范围错误，并恢复项目原代码的配置。主回测以每个模型自身全部已有信号为准，共{len(ready)}条模型预测序列。每个序列的输入时间、可实现收益时间、排除记录和来源逐项见coverage.csv。不同起始日的全历史收益不是相同市场区间的公平排名，共同区间仅另存CommonAuxiliary辅助结果。

## 输入与时间规则

压缩包全部7份逐股票预测文件均读取；固定/滚动按因子与模式分组，重训后的model_id是同一配置的更新，不拆成一条短历史。4份集成文件均回测三个子模型和集成（相同的来源也保留，以便逐文件对账）。原RF采用README1已冻结参数及random_forest.py恢复完整Development、Validation、Final；原Ridge采用原6个alpha网格与原3年/6年脚本恢复，未通过门槛的45因子滚动组不伪造Final；原回归XGB采用原12候选、原月度滚动和阶段固定规则恢复。没有采用上一版的两候选新实验替代原模型。

原预测中2019-08-30或2019-09-06开始的历史完整保留；6年历史设计实际开始2022年；holdout来源本身从2024年开始。2016-09-02起的原始价格用于训练和过去收益过滤，不是已有预测的起点。2026-08-28的信号尚无下一周价格，不能编造收益，明确记入excluded_rows.csv；最后可核验信号为2026-08-21，收益到2026-08-28。部分同股票下一全局周价格缺失的记录单独排除，不因其他模型缺少记录而裁剪主回测。

RF原代码按阶段末标签到期规则不输出少数跨阶段周，原始缺周列在missing_source_weeks.csv中，没有补造信号。主回测在这些周卖出上次持仓并持现金，计卖出费用；下一信号周再买入。年度、滚动区间及年化保留这些真实日历周，coverage.csv同时区分预测周数与回测日历周数。Ridge滚动45因子的3年/6年两个实验未通过原筛选门槛，原代码没有Validation/Final预测，保留未入测状态，不编造策略成绩。原正式28配置中26个可用配置均恢复完整历史；额外保留压缩包历史、子模型和标签对照，来源相同或信号相同也逐文件保留。

## 策略与窗口

每个模型执行15策略：原9策略（等权Top5/Top10、分数过滤、4/13周亏损过滤、排名加权Top5）保留；新增低波动加权、排名风险综合加权、前10名保留缓冲、每2周/4周调仓、市场26周均线过滤。风险加权只用截至信号日的26周简单收益标准差，单股上限35%；不足5只时股票预算只数/5，余款现金。具体规则见strategy_rules.json。

原58条预测之外增加四模型等权排名百分位共识序列（CONSENSUS-R3-F11-4MODELS）：固定RF/Ridge/XGB三年11因子加原Ranker，四方共同股票和日期才有共识预测。它不替换任何模型的完整历史。组件缺周共识持现金，不以前值填补预测。共识也执行15策略，原28正式配置中26个可用配置的口径不变。

缓冲和2/4周调仓具有持仓路径依赖，每个窗口单独从现金重放；非调仓周持仓自然漂移，不按新预测交易。原模型缺信号时清仓，恢复信号立即建仓，重设调仓计时。被持有股票无法计算下一周收益时排除该持仓并记forced_exit_count；这属于可评估性处理，不代表可执行的退市清算。所有策略仍使用可核验收益的股票池，存在事后可评估性选择偏差。

每个模型覆盖FullHistory、近13/26/52/104/156周（历史足够时）、每个自然年、Development3y/Development6y、Validation及FinalTest。窗口按自身全部可用日期切片，不设置人为统一起点；单边费用0/10/25bps。每窗口初始现金净值1，买卖双方成交权重计费，后续换手按持仓收益后的漂移权重计算；期末按市值结算，无强制清仓费用。净收益=(1-费用比例)*(1+组合毛收益)-1。基准是该模型当周可评估股票等权毛收益。

## 结果文件

- coverage.csv：每个模型实际时间和行数，这是本次时间范围验收入口。
- source_files.csv：7个压缩包预测源及哈希；predictions/：完整有效预测；excluded_rows.csv：无法核验收益的全部记录。
- full_history_comparison.csv：每模型15策略完整历史10bps指标。
- strategy_comparison.csv：全部窗口、策略、成本；model_evaluation.csv：全部区间的模型指标。
- weekly_backtest.csv、positions.csv：逐周收益与全历史逐股持仓；charts/：各模型各窗口策略曲线和原训练窗口的全历史曲线。
- stateful_window_positions.csv：缓冲/2周/4周策略在各非完整历史窗口重新从现金重放的持仓，用于独立费用验收。
- consensus_components.csv：四方排名百分位和合成分数；market_trend.csv：只使用过去数据的市场趋势条件。
- common_auxiliary_weekly.csv：共有日期和股票的辅助比较，主表不依赖此文件。

## 解释边界与复现

原预测直接重放；RF/Ridge/回归XGB为原代码和原配置在本机环境的重建，并非原训练权重恢复。原README的模型历史成绩可能受依赖版本影响，应以本批记录为准。全历史包含原本用于参数选择的阶段，不能称为整个区间均为未见数据；RF原配置由Validation选择，XGB原Development末周标签跨阶段惯例保持，明确区分阶段成绩。已有因子和股票池使用全时期筛选，结果仍属探索性研究。同收盘因子与同收盘建仓的可执行性未验证，费用仅情景估计。

复现：先执行restore_ridge.py、native_replay.py RF、native_replay.py XGB，再执行full_history_backtest.py。所有脚本位于evaluation_backtest目录，原数据与训练代码保持不变。
'''
    (OUT/'报告.md').write_text(report,encoding='utf-8')
    engine.dump(OUT/'run_manifest.json',dict(status='completed',protocol='native_full_history_v3_multistrategy',model_series=len(ready),original_series=58,derived_consensus_series=1,
        total_native_rows=int(coverage.input_rows.sum()),usable_rows=int(coverage.usable_rows.sum()),excluded_rows=int(coverage.excluded_rows.sum()),
        signal_first=coverage.signal_first.min(),signal_last=coverage.signal_last.max(),realization_last=coverage.realization_last.max(),
        summary_rows=len(summary),chart_count=len(list(chart.glob('*.svg'))),in_process_checks=checks,strategies=STRATEGIES,
        full_history_common_crop=False,common_auxiliary_keys=len(common_keys),source_sha256=engine.ridge.sha(ROOT/'test_45_factors_weekly.csv'),
        archive_sha256=engine.ridge.sha(ROOT/'xgboost-ranker第二版.zip'),python=platform.python_version()))

if __name__=='__main__':main()
