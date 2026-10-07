"""Ridge comparison preserving the original development-only selection protocol."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import platform, argparse
from model_training.features import SEVEN as BASE, MACRO, ALL45
from model_training.preprocessing import fit_preprocessor, apply_preprocessor
from model_training.artifacts import (ROOT, DEFAULT_INPUT, sha, save_json, annualize,
    mean_std_ratio as ratio, standard_predictions, save_predictions)
import numpy as np
import pandas as pd

HERE = ROOT

class core:
    """Exact Ridge solver with an unpenalized intercept."""
    @staticmethod
    def fit_ridge(x,y,alpha):
        xm,ym=x.mean(axis=0),float(y.mean())
        xc,yc=x-xm,y-ym
        coef=np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@yc)
        return coef,ym-float(xm@coef)

    @staticmethod
    def fit_ridge_lstsq_check(x,y,alpha):
        xc,yc=x-x.mean(axis=0),y-y.mean()
        ax=np.vstack([xc,np.sqrt(alpha)*np.eye(x.shape[1])])
        ay=np.concatenate([yc,np.zeros(x.shape[1])])
        return np.linalg.lstsq(ax,ay,rcond=None)[0]
GROUPS={'Ridge_7':BASE,'Ridge_7plus4':BASE+MACRO,'Ridge_45':ALL45}
ALPHAS=[0.01,0.1,1.,10.,100.,1000.]

STAGES = [('development',pd.Timestamp('2019-09-01'),pd.Timestamp('2024-09-01')),
          ('validation',pd.Timestamp('2024-09-01'),pd.Timestamp('2025-09-01')),
          ('final_test',pd.Timestamp('2025-09-01'),pd.Timestamp('2026-09-01'))]
SELECTION_CUTOFF=STAGES[1][1]
PRED_COLUMNS=['date','ticker','label_end_date','target','stage','fold_id',
              'model','mode','alpha','fit_date','prediction']
METRIC_NAMES=['IC','ICIR','Rank IC','Rank ICIR','Annualized Return',
              'Information Ratio','Top5 Excess','Positive Rank IC']

def weekly(pred):
    rows=[]
    for date,g in pred.groupby('date',sort=True):
        g=g.sort_values('ticker').dropna(subset=['prediction','target'])
        if len(g)<5:continue
        varied=g.prediction.nunique()>1 and g.target.nunique()>1
        ic=float(np.corrcoef(g.prediction,g.target)[0,1]) if varied else np.nan
        ric=float(np.corrcoef(g.prediction.rank(),g.target.rank())[0,1]) if varied else np.nan
        top=g.nlargest(5,'prediction')
        ret=float(np.expm1(top.target).mean())
        bench=float(np.expm1(g.target).mean())
        rows.append(dict(date=date,ic=ic,rank_ic=ric,top5_return=ret,
                         benchmark_return=bench,active_return=ret-bench,
                         stocks=len(g),top5_tickers='|'.join(top.ticker)))
    return pd.DataFrame(rows)

def metrics(pred):
    w=weekly(pred)
    if w.empty:raise ValueError('No evaluable weeks')
    ric=w.rank_ic.dropna()
    return {'IC':float(w.ic.mean()),'ICIR':ratio(w.ic),
            'Rank IC':float(ric.mean()),'Rank ICIR':ratio(ric),
            'Annualized Return':annualize(w.top5_return),
            'Information Ratio':ratio(w.active_return)*np.sqrt(52),
            'Top5 Excess':annualize(w.active_return),
            'Positive Rank IC':float(ric.gt(0).mean()) if len(ric) else np.nan,
            'weeks':len(w),'rows':len(pred),
            'benchmark_annualized_return':annualize(w.benchmark_return),
            'legacy_excess_annualized_return_pp':
                100*(annualize(w.top5_return)-annualize(w.benchmark_return))}

def choose(scores):
    scores=scores.copy()
    scores['passes_gate']=(scores['Rank IC'].gt(0)&scores['Rank ICIR'].ge(.05)
                           &scores['Positive Rank IC'].ge(.50))
    eligible=scores[scores.passes_gate & scores['Top5 Excess'].notna()]
    # Match XGBoost primary and tie-break objectives. Exact ties keep grid order.
    ranked=eligible.sort_values(['Top5 Excess','Rank IC','Rank ICIR'],
                                 ascending=[False,False,False],kind='stable')
    return scores, None if ranked.empty else float(ranked.iloc[0].alpha)

def load_panel(source):
    raw=pd.read_csv(source,parse_dates=['date']).sort_values(['date','ticker']).reset_index(drop=True)
    missing=set(ALL45+['date','ticker','target','adj_close'])-set(raw)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    assert not raw[['date','ticker']].isna().any().any()
    assert raw.ticker.nunique()==66 and not raw.duplicated(['date','ticker']).any()
    assert np.isfinite(raw[ALL45+['target','adj_close']].to_numpy()).all()
    assert raw.adj_close.gt(0).all()
    assert raw.groupby('date')[MACRO].nunique().le(1).all().all()
    assert raw.date.min()==pd.Timestamp('2016-09-02')
    assert raw.date.max()==pd.Timestamp('2026-08-28'), 'This protocol expects the agreed ten-year source'
    dates=pd.DatetimeIndex(sorted(raw.date.unique()))
    raw['label_end_date']=raw.date.map(dict(zip(dates[:-1],dates[1:])))
    price=raw.set_index(['date','ticker']).adj_close
    next_price=price.reindex(pd.MultiIndex.from_arrays([raw.label_end_date,raw.ticker])).to_numpy()
    ok=raw.label_end_date.notna()&np.isfinite(next_price)
    label_error=float(np.max(abs(np.log(next_price[ok]/raw.loc[ok,'adj_close'])-raw.loc[ok,'target'])))
    assert label_error<1e-10
    excluded=raw.loc[~ok,['date','ticker','target','label_end_date']].copy()
    excluded['reason']=np.where(excluded.label_end_date.isna(),'last_week_no_next_price','next_week_stock_price_missing')
    panel=raw.loc[ok].copy()
    panel['stage']='warmup'
    for stage,start,end in STAGES:
        panel.loc[panel.date.ge(start)&panel.date.lt(end),'stage']=stage
    return raw,panel,dates,excluded,label_error

def make_folds(panel,dates,mode):
    if mode=='fixed':
        return [dict(fold_id=i,fit_date=start,stage=stage,start=start,end=end)
                for i,(stage,start,end) in enumerate(STAGES)]
    forecast=dates[(dates>=dates.min()+pd.DateOffset(years=3))&(dates<STAGES[-1][2])]
    folds=[]
    for i in range(0,len(forecast),4):
        end=forecast[i+4] if i+4<len(forecast) else STAGES[-1][2]
        if not panel.date.between(forecast[i],end,inclusive='left').any():continue
        folds.append(dict(fold_id=i//4,fit_date=forecast[i],start=forecast[i],end=end))
    return folds

def training_rows(panel,fold,mode):
    fit=fold['fit_date']
    start=fit-pd.DateOffset(years=3) if mode=='rolling' else panel.date.min()
    # Same strict availability convention as the supplied XGBoost code.
    train=panel[panel.date.ge(start)&panel.date.lt(fit)&panel.label_end_date.lt(fit)]
    assert not train.empty and train.date.max()<fit and train.label_end_date.max()<fit
    return train

def predict_frame(pred,fold,mode,name,alpha,xp,coef,intercept):
    out=pred[['date','ticker','label_end_date','target','stage']].copy()
    out['fold_id']=fold['fold_id'];out['model']=name;out['mode']=mode
    out['alpha']=alpha;out['fit_date']=fold['fit_date'];out['prediction']=xp@coef+intercept
    assert np.isfinite(out.prediction).all()
    return out[PRED_COLUMNS]

def run_mode(source,root,mode,raw,panel,dates,groups=None,alpha_grid=None,quick=False):
    groups = GROUPS if groups is None else groups
    alpha_grid = ALPHAS if alpha_grid is None else alpha_grid
    out=root/mode;out.mkdir(parents=True)
    folds=make_folds(panel,dates,mode)
    if quick and mode == 'rolling':
        selected_ids = set()
        for stage, _, _ in STAGES:
            matches = [f['fold_id'] for f in folds if not panel[
                panel.stage.eq(stage) & panel.date.ge(f['start']) & panel.date.lt(f['end'])].empty]
            selected_ids.update(matches[:2])
        folds = [f for f in folds if f['fold_id'] in selected_ids]
    scores_all=[];statuses=[];predictions=[];params_all=[];folds_all=[];summary=[];weekly_all=[]
    solver_errors=[]
    for name,features in groups.items():
        pieces=[]
        for fold in folds:
            pred=panel[panel.date.ge(fold['start'])&panel.date.lt(fold['end'])
                       &panel.stage.eq('development')]
            if pred.empty:continue
            train=training_rows(panel,fold,mode)
            x,prep=fit_preprocessor(train,features,standardize=True);xp=apply_preprocessor(pred,features,prep)
            for alpha in alpha_grid:
                coef,intercept=core.fit_ridge(x,train.target.to_numpy(),alpha)
                pieces.append(predict_frame(pred,fold,mode,name,alpha,xp,coef,intercept))
                if fold['fold_id']==0:
                    error=float(np.max(abs(coef-core.fit_ridge_lstsq_check(x,train.target.to_numpy(),alpha))))
                    assert error<1e-10
                    solver_errors.append(error)
            if fold['fold_id']%20==0:print(f'{mode} {name}: development block {fold["fold_id"]}',flush=True)
        dev=pd.concat(pieces,ignore_index=True)
        candidates=[]
        for alpha,p in dev.groupby('alpha',sort=True):
            # Purge labels crossing the selection date. Do NOT look into Validation.
            scored=p[p.label_end_date<SELECTION_CUTOFF]
            candidates.append(dict(mode=mode,model=name,alpha=alpha,**metrics(scored),
                selection_cutoff=SELECTION_CUTOFF,last_selection_label=scored.label_end_date.max(),
                selection_excluded_rows=len(p)-len(scored)))
        scores,alpha=choose(pd.DataFrame(candidates))
        scores_all.append(scores)
        statuses.append(dict(mode=mode,model=name,status='selected' if alpha is not None else 'no_candidate_passed',
                             alpha=alpha,passing_candidates=int(scores.passes_gate.sum())))
        print(f'{mode} {name}: alpha={alpha}; passed {int(scores.passes_gate.sum())}/{len(alpha_grid)}',flush=True)
        if alpha is None:continue
        model_predictions=[]
        for fold in folds:
            pred=panel[panel.date.ge(fold['start'])&panel.date.lt(fold['end'])]
            if pred.empty:continue
            train=training_rows(panel,fold,mode)
            x,prep=fit_preprocessor(train,features,standardize=True);xp=apply_preprocessor(pred,features,prep)
            coef,intercept=core.fit_ridge(x,train.target.to_numpy(),alpha)
            p=predict_frame(pred,fold,mode,name,alpha,xp,coef,intercept)
            model_predictions.append(p)
            folds_all.append(dict(mode=mode,model=name,fold_id=fold['fold_id'],fit_date=fold['fit_date'],
                train_start=train.date.min(),train_end=train.date.max(),last_train_label=train.label_end_date.max(),
                train_rows=len(train),prediction_start=pred.date.min(),prediction_end=pred.date.max(),
                prediction_weeks=pred.date.nunique(),alpha=alpha))
            for j,f in enumerate(features):
                params_all.append(dict(mode=mode,model=name,fold_id=fold['fold_id'],feature=f,
                    coefficient=coef[j],intercept=intercept,
                    **{n:float(getattr(prep,n)[j]) for n in ['median','clip_low','clip_high','mean','scale']}))
        selected=pd.concat(model_predictions,ignore_index=True)
        before=dev[dev.alpha.eq(alpha)].sort_values(['date','ticker'])
        after=selected[selected.stage.eq('development')].sort_values(['date','ticker'])
        assert np.allclose(before.prediction,after.prediction,rtol=0,atol=1e-14)
        assert selected.alpha.nunique()==1
        if mode=='fixed':assert selected.fit_date.nunique()==3
        predictions.append(selected)
        for stage,p in selected.groupby('stage',sort=False):
            summary.append(dict(mode=mode,**{'Model Name':name,'Dataset':stage},alpha=alpha,**metrics(p)))
            w=weekly(p);w['model']=name;w['mode']=mode;w['stage']=stage;weekly_all.append(w)
    preds=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=PRED_COLUMNS)
    scores=pd.concat(scores_all,ignore_index=True)
    metric_table=pd.DataFrame(summary,columns=['mode','Model Name','Dataset','alpha',*METRIC_NAMES,
        'weeks','rows','benchmark_annualized_return','legacy_excess_annualized_return_pp'])
    status=pd.DataFrame(statuses)
    scores.to_csv(out/'development_alpha_scores.csv',index=False,encoding='utf-8-sig')
    status.to_csv(out/'selection_status.csv',index=False,encoding='utf-8-sig')
    preds.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    metric_table.to_csv(out/'performance_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(folds_all).to_csv(out/'folds.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(params_all).to_csv(out/'model_parameters.csv',index=False,encoding='utf-8-sig')
    if weekly_all:pd.concat(weekly_all,ignore_index=True).to_csv(out/'weekly_metrics.csv',index=False,encoding='utf-8-sig')
    save_json(out/'locked_selection.json',dict(status=statuses,selection_cutoff=SELECTION_CUTOFF))
    # Rebuild saved predictions using independently read back model parameters.
    max_error=0.
    if len(preds):
        stored=pd.read_csv(out/'model_parameters.csv')
        saved=pd.read_csv(out/'predictions.csv',parse_dates=['date'])
        lookup=raw.set_index(['date','ticker'])
        expected=panel[panel.stage.ne('warmup')].sort_values(['date','ticker'])
        for name,g in saved.groupby('model'):
            ordered=g.sort_values(['date','ticker'])
            if not quick:
                assert list(zip(ordered.date,ordered.ticker))==list(zip(expected.date,expected.ticker))
                assert np.allclose(ordered.target,expected.target,atol=1e-14,rtol=0)
        for (name,fid),g in saved.groupby(['model','fold_id']):
            features=groups[name]
            par=stored[stored.model.eq(name)&stored.fold_id.eq(fid)].set_index('feature').loc[features]
            original=lookup.loc[pd.MultiIndex.from_frame(g[['date','ticker']])]
            xx=original[features].to_numpy()
            xx=(np.clip(xx,par.clip_low.to_numpy(),par.clip_high.to_numpy())-par['mean'].to_numpy())/par.scale.to_numpy()
            max_error=max(max_error,float(np.max(abs(xx@par.coefficient.to_numpy()+par.intercept.iloc[0]-g.prediction))))
        assert max_error<1e-10
    save_json(out/'verification.json',dict(prediction_readback_max_error=max_error,
        independent_solver_max_error=max(solver_errors),selected_groups=int(status.status.eq('selected').sum()),
        evaluated_groups=len(groups),same_evaluation_rows_for_selected_groups=True,
        training_labels_strictly_before_fit=True,selection_labels_strictly_before_validation=True))
    return metric_table,status,preds

def main(argv=None):
    parser=argparse.ArgumentParser(description='Ridge: XGBoost-aligned gate, stage-fixed training and metrics')
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output',type=Path,default=HERE/'outputs'/'ridge')
    parser.add_argument('--mode',choices=['both','rolling','fixed'],default='both')
    parser.add_argument('--feature-set',choices=['all','7','7_macro','45'],default='all')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--quick',action='store_true',help='Smoke test: two alphas and first two rolling blocks per stage; not research results')
    args=parser.parse_args(argv)
    names={'7':'Ridge_7','7_macro':'Ridge_7plus4','45':'Ridge_45'}
    groups = GROUPS if args.feature_set == 'all' else {names[args.feature_set]:GROUPS[names[args.feature_set]]}
    alpha_grid = [ALPHAS[0],ALPHAS[-1]] if args.quick else ALPHAS
    if not __debug__:parser.error('Do not use Python -O')
    source=args.input.resolve();root=args.output.resolve()
    if not source.is_file():parser.error(f'Missing input: {source}')
    if not args.check_only and root.exists() and (not root.is_dir() or any(root.iterdir())):parser.error('Use a new or empty output directory')
    digest=sha(source)
    raw,panel,dates,excluded,label_error=load_panel(source)
    if args.check_only:
        print(f'Ridge input validation passed: {len(panel)} rows, {len(groups)} feature sets')
        return {'status':'checked','rows':len(panel)}
    root.mkdir(parents=True,exist_ok=True)
    excluded.to_csv(root/'excluded_rows.csv',index=False,encoding='utf-8-sig')
    modes=['rolling','fixed'] if args.mode=='both' else [args.mode]
    config=dict(protocol='ridge_xgb_aligned_v2',source=source,source_sha256=digest,features=groups,quick=args.quick,
        alpha_grid=alpha_grid,stages=STAGES,selection_cutoff=SELECTION_CUTOFF,
        selection='Rank IC>0, unannualized Rank ICIR>=0.05, Positive Rank IC>=0.50; then descending Top5 Excess, Rank IC, Rank ICIR; no fallback',
        fixed='All available history, refit at 2019-09-01, 2024-09-01, 2025-09-01; freeze coefficients within stage',
        rolling='Latest three calendar years; continuous four-observation refit grid',
        training_label_rule='label_end_date < fit_date',
        selection_label_rule='label_end_date < 2024-09-01; crossing-boundary dev labels purged',
        metrics='ICIR and Rank ICIR unannualized; Information Ratio annualized sqrt(52); Top5 Excess is CAGR of weekly active returns',
        retained_differences_from_supplied_xgboost=[
            'Rolling grid is four observations, not natural months, per prior team agreement.',
            'Source target is verified against next global-week same-stock adjusted price; same 82 excluded rows retained.',
            'Development labels crossing selection cutoff are excluded to prevent boundary lookahead.',
            'Ridge applies training-only standardization; tree model does not need it.',
            'Ridge evaluates six alpha values, not twelve tree parameter sets.'],
        caveats=['Full-period feature screening and repeated final-test inspection make this exploratory.',
                 'Prior universe/cleaning/financial publication timestamps not re-audited.',
                 'Same-close execution and trading costs not verified; signal returns are not executable performance.'])
    save_json(root/'config.json',config)
    tables=[];statuses=[];prediction_parts=[]
    for mode in modes:
        table,status,preds=run_mode(source,root,mode,raw,panel,dates,groups,alpha_grid,args.quick)
        tables.append(table);statuses.append(status)
        for group_name, pred in preds.groupby('model'):
            feature_set = next(k for k,v in names.items() if v == group_name)
            prediction_parts.append(standard_predictions(pred,model='ridge',mode=mode,feature_set=feature_set))
    save_predictions(root/'predictions.csv',prediction_parts,quick_run=args.quick)
    combined=pd.concat(tables,ignore_index=True);status=pd.concat(statuses,ignore_index=True)
    combined.to_csv(root/'all_stage_metrics.csv',index=False,encoding='utf-8-sig')
    final=combined[combined.Dataset.eq('final_test')].rename(columns={'Model Name':'model'})
    comparison=status.merge(final,on=['mode','model','alpha'],how='left',validate='one_to_one')
    comparison.to_csv(root/'final_test_comparison.csv',index=False,encoding='utf-8-sig')
    status.to_csv(root/'selection_status.csv',index=False,encoding='utf-8-sig')
    assert sha(source)==digest
    save_json(root/'run_manifest.json',dict(status='completed',quick=args.quick,protocol='ridge_xgb_aligned_v2',
        source_sha256=digest,program_sha256=sha(Path(__file__)),raw_rows=len(raw),usable_rows=len(panel),
        excluded_rows=len(excluded),label_max_error=label_error,evaluated_groups=len(status),
        selected_groups=int(status.status.eq('selected').sum()),python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,results=comparison.to_dict('records')))
    print(comparison.to_string(index=False),flush=True)
    print(f'Completed: {root}',flush=True)
    return {'status':'completed','output':str(root),'predictions':str(root/'predictions.csv')}

if __name__=='__main__':main()
