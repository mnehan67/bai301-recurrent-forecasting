"""Generate honest, source-backed experiment results, figures, ANOVA and IEEE tables.
Never fills missing results with fake measurements. Partial runs are labelled partial.
"""
from __future__ import annotations
import json, math, os, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_engine import run_directory, experiment_grid, windows_for, persistence


def latex_escape(s):
    return str(s).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')


def load_results(cfg):
    root=run_directory(cfg); rows=[]
    for path in sorted((root/'runs').glob('*.json')):
        data=json.loads(path.read_text()); rows.append(data)
    return pd.DataFrame(rows)


def aggregate_paired(df):
    # Dataframe from completed runs only, primary comparisons from validation-selected
    # group; seeds remain statistical replicates instead of treating forecast steps as IID.
    keys=['dataset','architecture','tbptt','clip','forcing']
    neural=df[(df.architecture!='ESN')&(df.status=='ok')].copy()
    if neural.empty:return pd.DataFrame(),pd.DataFrame()
    val=neural.groupby(keys,dropna=False).agg(
        val_mean=('val_rmse_rollout_norm','mean'), val_std=('val_rmse_rollout_norm','std'),
        seeds=('seed','nunique')).reset_index()
    # Only preselected groups with all three seeds are eligible for selected-model claims.
    qualified=val[val.seeds==3]
    if qualified.empty:return val,pd.DataFrame()
    winners=qualified.sort_values(['dataset','architecture','val_mean']).groupby(['dataset','architecture']).head(1)
    chosen=neural.merge(winners[keys],on=keys,how='inner')
    rows=[]
    for (d,a),g in chosen.groupby(['dataset','architecture']):
        row={'dataset':d,'architecture':a,'tbptt':str(g.tbptt.iloc[0]),
             'clip':bool(g['clip'].iloc[0]),'forcing':g.forcing.iloc[0], 'seeds':int(g.seed.nunique())}
        for metric in ('test_rmse_1','test_rmse_20','test_rmse_50','test_rmse_100',
                       'test_rmse_rollout','test_rmse_rollout_norm','training_seconds',
                       'peak_gpu_memory_mb','gradient_norm_mean'):
            if metric in g:
                row[metric+'_mean']=float(g[metric].mean())
                row[metric+'_sd']=float(g[metric].std())
        rows.append(row)
    return val,pd.DataFrame(rows)


def anova_tables(df,root,cfg):
    out=[]
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except Exception as e:
        return pd.DataFrame([{'status':'statsmodels unavailable','reason':str(e)}])
    for name,g in df[(df.status=='ok') & (df.architecture!='ESN')].groupby('dataset'):
        # Reject incomplete factorial grid for full ANOVA; show incompleteness clearly.
        expected={ (i['architecture'],str(i['tbptt']),str(i['clip']),i['forcing'],int(i['seed']))
                   for i in experiment_grid(cfg) if i['dataset']==name and i['architecture']!='ESN'}
        got={ (r.architecture,str(r.tbptt),str(r['clip']),r.forcing,int(r.seed)) for _,r in g.iterrows()}
        if expected != got:
            out.append({'dataset':name,'status':'NOT RUN: factorial incomplete',
                        'complete_configs':len(expected & got),'required_configs':len(expected)})
            continue
        sub=g[['architecture','tbptt','clip','forcing','seed','test_rmse_rollout_norm']].copy()
        sub['tbptt']=sub.tbptt.astype(str)
        sub['clip']=sub['clip'].astype(str)
        sub['seed']=sub.seed.astype(str)
        sub['response']=np.log1p(sub.test_rmse_rollout_norm.clip(lower=0))
        try:
            model=ols('response ~ C(architecture)*C(tbptt)*C(clip)*C(forcing) + C(seed)',data=sub).fit()
            table=anova_lm(model,typ=2)
            for term,row in table.iterrows():
                out.append({'dataset':name,'status':'ok','effect':term,'sum_sq':row.get('sum_sq'),
                            'F':row.get('F'),'pvalue':row.get('PR(>F)'),
                            'df':row.get('df')})
        except Exception as e:out.append({'dataset':name,'status':f'ANOVA failed: {e}'})
    return pd.DataFrame(out)


def draw_figures(df,chosen,root):
    figdir=root/'figures';figdir.mkdir(exist_ok=True,parents=True)
    if chosen.empty:return
    datasets=sorted(chosen.dataset.unique())
    plt.rcParams.update({'font.size':10,'figure.figsize':(7,4.2),'axes.spines.top':False,'axes.spines.right':False})
    for d in datasets:
        part=chosen[chosen.dataset==d]
        fig,ax=plt.subplots()
        for _,r in part.iterrows():
            xx=[1,20,50,100];yy=[r.get(f'test_rmse_{h}_mean',float('nan')) for h in xx]
            yerr=[r.get(f'test_rmse_{h}_sd',float('nan')) for h in xx]
            ax.errorbar(xx,yy,yerr=yerr,marker='o',capsize=2,label=r.architecture)
        ax.set(title=f'{d}: free-running error by horizon',xlabel='Forecast horizon (steps)',ylabel='RMSE (original units)')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(figdir/f'{d}_horizon.png',dpi=170);plt.close(fig)
        fig,ax=plt.subplots()
        for _,r in part.iterrows():
            ax.scatter(r['training_seconds_mean'],r['test_rmse_rollout_mean'],s=70)
            ax.annotate(r.architecture,(r['training_seconds_mean'],r['test_rmse_rollout_mean']),xytext=(4,4),textcoords='offset points')
        ax.set(title=f'{d}: accuracy–training cost',xlabel='Training time (s, per run)',ylabel='100-step rollout RMSE')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/f'{d}_accuracy_cost.png',dpi=170);plt.close(fig)
    # Additional course-comparison figures from actual logs, with no assumed outcomes.
    neural=df[(df.architecture!='ESN')&(df.status=='ok')].copy()
    if not neural.empty:
        for d,g in neural.groupby('dataset'):
            # Teacher forcing interaction, averaged over the specified other factors.
            grp=g.groupby(['architecture','forcing'])['test_rmse_rollout_norm'].agg(['mean','std']).reset_index()
            fig,ax=plt.subplots(figsize=(8,4))
            order=['always','scheduled','never']
            for a,ga in grp.groupby('architecture'):
                sub=ga.set_index('forcing').reindex(order)
                ax.errorbar(range(len(order)),sub['mean'],yerr=sub['std'],marker='o',capsize=2,label=a)
            ax.set_xticks(range(3),order)
            ax.set(title=f'{d}: teacher-forcing strategies',xlabel='Training feedback policy',
                   ylabel='Normalized 100-step rollout RMSE')
            ax.grid(alpha=.2);ax.legend();fig.tight_layout()
            fig.savefig(figdir/f'{d}_teacher_forcing.png',dpi=170);plt.close(fig)
            # Clipping: show mean gradient norm BEFORE clipping and measured heldout error.
            grp=g.groupby(['architecture','clip'])['gradient_norm_mean'].mean().unstack('clip')
            if not grp.empty:
                fig,ax=plt.subplots(figsize=(7,3.8));grp.plot(kind='bar',ax=ax)
                ax.set(title=f'{d}: gradients by clipping policy',xlabel='Architecture',
                       ylabel='Mean pre-clip gradient norm')
                ax.legend(title='Clipping enabled');ax.grid(alpha=.18,axis='y')
                fig.tight_layout();fig.savefig(figdir/f'{d}_gradient_clipping.png',dpi=170);plt.close(fig)
    esn=df[(df.architecture=='ESN')&(df.status=='ok')]
    if not esn.empty:
        for d,g in esn.groupby('dataset'):
            summary=g.groupby('esn_size')['test_rmse_rollout_norm'].agg(['mean','std']).sort_index()
            fig,ax=plt.subplots(figsize=(7,4))
            ax.errorbar(summary.index.astype(str),summary['mean'],yerr=summary['std'],
                        marker='o',capsize=3)
            ax.set(title=f'{d}: ESN reservoir size comparison',xlabel='Reservoir units',
                   ylabel='Normalized 100-step rollout RMSE')
            ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(figdir/f'{d}_esn_sweep.png',dpi=170);plt.close(fig)
    for d,g in df[(df.architecture!='ESN')&(df.status=='ok')].groupby('dataset'):
        pv=g.groupby(['architecture','tbptt'])['test_rmse_rollout_norm'].mean().unstack('tbptt')
        pv=pv.reindex(['RNN','LSTM','GRU']).dropna(how='all')
        if pv.empty:continue
        fig,ax=plt.subplots(figsize=(7,3.5));arr=pv.to_numpy(dtype=float)
        im=ax.imshow(arr,aspect='auto');fig.colorbar(im,ax=ax,label='Normalized 100-step rollout RMSE')
        ax.set_xticks(np.arange(len(pv.columns)),[str(x) for x in pv.columns])
        ax.set_yticks(np.arange(len(pv.index)),pv.index)
        ax.set(xlabel='TBPTT length',title=f'{d}: mean error by architecture and truncation')
        fig.tight_layout();fig.savefig(figdir/f'{d}_tbptt_heatmap.png',dpi=170);plt.close(fig)


def ieee_snippet(chosen,esn,root):
    lines=['% GENERATED FROM REAL RUN LOGS. Never hand-enter invented results.',
           '% This table selects RNN/LSTM/GRU settings using validation-only RMSE.',
           r'\begin{table*}[t]',r'\centering',
           r'\caption{Held-out free-running forecasting performance (mean $\pm$ SD over three seeds).}',
           r'\begin{tabular}{llrrrr}',r'\hline',
           r'Dataset & Model & RMSE@1 & RMSE@100 & Rollout RMSE & Train time (s) \\',r'\hline']
    for d in sorted(chosen.dataset.unique()) if not chosen.empty else []:
        for _,r in chosen[chosen.dataset==d].sort_values('architecture').iterrows():
            def fmt(field):
                a=r.get(field+'_mean',np.nan);b=r.get(field+'_sd',np.nan)
                return '--' if pd.isna(a) else f'{a:.3g} $\\pm$ {b:.2g}'
            lines.append(f"{latex_escape(d)} & {latex_escape(r.architecture)} & {fmt('test_rmse_1')} & {fmt('test_rmse_100')} & {fmt('test_rmse_rollout')} & {fmt('training_seconds')} \\\\")
        g=esn[esn.dataset==d]
        if not g.empty:
            size=g.esn_size.iloc[0]
            def fm(col):
                s=g[col].astype(float)
                return f'{s.mean():.3g} $\\pm$ {s.std():.2g}'
            lines.append(f"{latex_escape(d)} & ESN ({size}) & {fm('test_rmse_1')} & {fm('test_rmse_100')} & {fm('test_rmse_rollout')} & {fm('training_seconds')} \\\\")
        lines.append(r'\hline')
    lines.extend([r'\end{tabular}',r'\end{table*}'])
    path=root/'ieee_results_generated.tex';path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    return path


def report_run_status(root, df):
    """Save the full log and expose unsuccessful runs instead of hiding them."""
    df.to_csv(root/'all_runs.csv',index=False)
    passed=df[df.status=='ok'].copy()
    failed=df[df.status!='ok'].copy()
    print('Successful runs:',len(passed),'| Failed runs:',len(failed))
    if not failed.empty:
        failed.to_csv(root/'failed_runs.csv',index=False)
        print('Failures recorded in failed_runs.csv (do not silently exclude them from conclusions).')
    return passed, failed


def select_esn_on_validation(passed, root):
    """Pick reservoir size using validation, never using test results."""
    esn=passed[passed.architecture=='ESN'].copy()
    esn_chosen=pd.DataFrame()
    if not esn.empty:
        esn.to_csv(root/'esn_all_runs.csv',index=False)
        agg=esn.groupby(['dataset','esn_size']).agg(val=('val_rmse_rollout_norm','mean'),seeds=('seed','nunique')).reset_index()
        agg=agg[agg.seeds==3]
        if not agg.empty:
            wins=agg.sort_values(['dataset','val']).groupby('dataset').head(1)
            esn_chosen=esn.merge(wins[['dataset','esn_size']],on=['dataset','esn_size'])
            esn_chosen.to_csv(root/'esn_selected_runs.csv',index=False)
    return esn_chosen


def save_ranking_agreement(chosen, esn_chosen, root):
    """Compare one-step and 100-step model orderings with Kendall rank correlation."""
    from scipy.stats import kendalltau
    rank_rows=[]
    if not chosen.empty:
        for d,g in chosen.groupby('dataset'):
            frame=g[['architecture','test_rmse_1_mean','test_rmse_rollout_mean']].copy()
            sel=esn_chosen[esn_chosen.dataset==d] if not esn_chosen.empty else pd.DataFrame()
            if len(sel)==3:
                frame=pd.concat([frame,pd.DataFrame([dict(architecture='ESN',
                    test_rmse_1_mean=sel.test_rmse_1.mean(),
                    test_rmse_rollout_mean=sel.test_rmse_rollout.mean())])],ignore_index=True)
            if len(frame)>=3:
                tau=kendalltau(frame.test_rmse_1_mean,frame.test_rmse_rollout_mean)
                rank_rows.append(dict(dataset=d,models=len(frame),kendall_tau=float(tau.statistic),
                                      pvalue=float(tau.pvalue)))
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(root/'ranking_agreement.csv',index=False)


def save_tuning_audit(root):
    """Keep the training cost of hyperparameter selection visible."""
    tune=[]
    for path in sorted((root/'tuning').glob('*.json')):
        obj=json.loads(path.read_text())
        for t in obj.get('trials',[]):
            tune.append(dict(dataset=obj['dataset'],architecture=obj['arch'],
                             score=t.get('score'),seconds=t.get('seconds'),
                             trial_settings=str(t.get('lr',t.get('params')))))
    if tune:
        pd.DataFrame(tune).to_csv(root/'tuning_trials.csv',index=False)


def save_implementation_note(cfg, root):
    """Describe settings from the run config so the paper matches the implementation."""
    (root/'methods_implementation_note.md').write_text(
      '# Exact notebook methodology\n'
      f"- Mode: {cfg['PRESET']}\n- C/H: {cfg['CONTEXT']}/{cfg['HORIZON']}\n"
      f"- Epochs: {cfg['EPOCHS']}, minibatch size: {cfg['BATCH_SIZE']}\n"
      f"- Training origins: stride {cfg['TRAIN_STRIDE']}, capped at {cfg['MAX_TRAIN_ORIGINS']} per dataset, evenly selected\n"
      f"- Evaluation origins: stride {cfg['EVAL_STRIDE']}, capped at {cfg['MAX_EVAL_ORIGINS']} per split\n"
      '- Full BPTT covers C+H-1 recurrent input steps (199 for C=H=100).\n'
      f"- Mackey–Glass: Euler step {cfg['MG_DT']}, delay {cfg['MG_DELAY']}, after transient {cfg['MG_TRANSIENT']}.\n"
      '- ESN: sparse reservoir, fixed weights, readout fitted by ridge regression.\n'
      '- Selected epochs/configurations use validation; testing is always free-running.\n'
      '- Two Kaggle T4 workers train independent configurations on cuda:0 and cuda:1; no distributed synchronization.\n'
      '- ESN CPU RSS is process-level and can include allocations from earlier experiments.\n'
      '- **IMPORTANT:** Align the accompanying manuscript to these *implemented* settings, not prospective settings.\n')


def save_persistence_baselines(cfg, root):
    """Check whether the models actually beat a no-training forecast."""
    bases=[]
    for d in cfg['DATASETS']:
        try:
            _,_,te=windows_for(cfg,d)
            bases.append({'dataset':d,**persistence(te)})
        except Exception as e:
            bases.append({'dataset':d,'error':str(e)})
    pd.DataFrame(bases).to_csv(root/'persistence_baselines.csv',index=False)


def save_summary(cfg, root, specs, df, passed, failed, chosen, esn_chosen):
    """Write a transparent report including missing runs and method limitations."""
    readable=[f'# BAI301 experiment report — {cfg["PRESET"]}',
      f'- Requested configurations: **{len(specs)}**',
      f'- Completed successfully: **{len(passed)}**',
      f'- Failed: **{len(failed)}**',
      f'- Pending: **{len(specs)-len(df)}**',
      '- No absent run is replaced with a guessed metric.',
      '- Validation-only selection; test is reserved for reporting.',
      '- CPU ESN memory and GPU neural-network memory are not directly comparable.',
      '- Training-run seconds are not full hardware-energy consumption.',
      '- Electricity is an aggregated customer series, not the standard Monash benchmark evaluation.',
      '- Mackey–Glass was generated by deterministic forward-Euler integration, not ReservoirPy.',
      '- Training windows are capped and uniformly subsampled as declared in config.json.',
      '', '## Selected trainable-model configurations',
      chosen.to_string(index=False) if not chosen.empty else 'No complete 3-seed configuration yet.',
      '', '## ESN validation-selected configurations',
      esn_chosen[['dataset','esn_size','seed','val_rmse_rollout_norm','test_rmse_rollout_norm']].to_string(index=False)
      if not esn_chosen.empty else 'No complete 3-seed ESN size yet.',
      '', '## Statistical model',
      'Four-factor ANOVA plus seed block on log(1 + normalized rollout RMSE), only when the full design is complete.',
      'The ANOVA is exploratory; repeated seeds and correlated forecasts limit inference.',
      '', '## Reproducibility',
      'See config.json, data/metadata.json, runs/, epochs/, tuning/ and environment.json.']
    (root/'SUMMARY.md').write_text('\n'.join(readable)+'\n')


def generate(cfg):
    """Produce all evidence files from measured runs. Never create fabricated metrics."""
    root=run_directory(cfg)
    root.mkdir(parents=True,exist_ok=True)
    df=load_results(cfg)
    specs=experiment_grid(cfg)
    print('Results folder:',root)
    print('Expected experiments:',len(specs),'| Result files found:',len(df))
    if df.empty:
        print('No results yet. Execute the experiment cell before analyzing.')
        return root

    passed,failed=report_run_status(root,df)
    val,chosen=aggregate_paired(df)
    if not val.empty:
        val.to_csv(root/'validation_grid.csv',index=False)
    if not chosen.empty:
        chosen.to_csv(root/'validation_selected_models.csv',index=False)
    esn_chosen=select_esn_on_validation(passed,root)
    anova=anova_tables(df,root,cfg)
    anova.to_csv(root/'anova_interactions.csv',index=False)
    draw_figures(df,chosen,root)
    ieee_snippet(chosen,esn_chosen,root)
    save_ranking_agreement(chosen,esn_chosen,root)
    save_tuning_audit(root)
    save_implementation_note(cfg,root)
    save_persistence_baselines(cfg,root)
    save_summary(cfg,root,specs,df,passed,failed,chosen,esn_chosen)
    return root

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text())
    generate(cfg)
