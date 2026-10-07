#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_summarize_asca_interactions.py

Summarize ASCA outputs from script 15 with emphasis on pH and pH x phone.
No model is refitted.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['L','a','b']

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--variance', default='outputs/multivariate_lab/asca_variance.csv')
    p.add_argument('--loadings', default='outputs/multivariate_lab/asca_loadings.csv')
    p.add_argument('--scores', default='outputs/multivariate_lab/asca_scores.csv')
    p.add_argument('--output-dir', default='outputs/asca_interaction_summary')
    p.add_argument('--scaling', default='standardized')
    p.add_argument('--dpi', type=int, default=220)
    p.add_argument('--save-pdf', action='store_true')
    return p.parse_args()

def resolve_existing(text):
    path=Path(text); candidates=[path]
    if not path.is_absolute(): candidates += [PROJECT_ROOT/path, Path.cwd()/path]
    for c in candidates:
        c=c.resolve()
        if c.exists(): return c
    raise FileNotFoundError(text)

def resolve_output(text):
    path=Path(text)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()

def validate_inputs(v,l,s):
    reqs=[('variance',v,{'region','scaling','term','component','explained_ratio_within_term'}),
          ('loadings',l,{'region','scaling','term','component','channel','loading'}),
          ('scores',s,{'region','scaling','term','component','phone','ph_nominal','score'})]
    for name,df,req in reqs:
        missing=sorted(req-set(df.columns))
        if missing: raise ValueError(f'{name} missing columns: {missing}')

def make_loading_summary(loadings, variance, scaling):
    primary=loadings.loc[(loadings['scaling']==scaling)&(loadings['term'].isin(['pH','pH_x_phone']))&(loadings['component']==1)].copy()
    primary['abs_loading']=primary['loading'].abs()
    v=variance.loc[(variance['scaling']==scaling)&(variance['term'].isin(['pH','pH_x_phone']))&(variance['component']==1),['region','term','explained_ratio_within_term']]
    primary=primary.merge(v,on=['region','term'],how='left',validate='many_to_one')
    dominant=(primary.sort_values(['region','term','abs_loading'],ascending=[True,True,False])
              .groupby(['region','term'],as_index=False).first()
              .rename(columns={'channel':'dominant_channel','loading':'dominant_loading','abs_loading':'dominant_abs_loading'}))
    wide=(primary.pivot_table(index=['region','term','explained_ratio_within_term'],columns='channel',values='loading').reset_index())
    for ch in CHANNELS:
        if ch not in wide.columns: wide[ch]=np.nan
    wide=wide.rename(columns={'L':'loading_L','a':'loading_a','b':'loading_b'})
    out=wide.merge(dominant[['region','term','dominant_channel','dominant_loading','dominant_abs_loading']],on=['region','term'],how='left',validate='one_to_one')
    return out.sort_values(['region','term']).reset_index(drop=True)

def score_summaries(scores, scaling):
    sub=scores.loc[(scores['scaling']==scaling)&(scores['term']=='pH_x_phone')&(scores['component']==1)].copy()
    phone_ph=(sub.groupby(['region','phone','ph_nominal'],as_index=False)['score'].mean().sort_values(['region','phone','ph_nominal']))
    phone_ph['abs_score']=phone_ph['score'].abs()
    peaks=(phone_ph.sort_values(['region','phone','abs_score'],ascending=[True,True,False])
           .groupby(['region','phone'],as_index=False).first()
           .rename(columns={'ph_nominal':'peak_ph','score':'peak_score','abs_score':'peak_abs_score'}))
    ph_mag=(phone_ph.groupby(['region','ph_nominal'],as_index=False)
            .agg(mean_abs_interaction_score=('abs_score','mean'),max_abs_interaction_score=('abs_score','max'),sd_interaction_score=('score','std'))
            .sort_values(['region','ph_nominal']))
    return phone_ph,peaks,ph_mag

def save_fig(fig,stem,dpi,save_pdf):
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix('.png'),dpi=dpi,bbox_inches='tight')
    if save_pdf: fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def plot_loadings(summary,region,stem,dpi,save_pdf):
    sub=summary.loc[summary['region']==region]
    fig,ax=plt.subplots(figsize=(7.2,4.8)); x=np.arange(3); width=.36
    for i,term in enumerate(['pH','pH_x_phone']):
        row=sub.loc[sub['term']==term]
        if row.empty: continue
        vals=[float(row.iloc[0][f'loading_{c}']) for c in CHANNELS]
        ax.bar(x+(i-.5)*width,vals,width=width,label=term)
    ax.axhline(0,linewidth=1,alpha=.6); ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel('ASCA PC1 loading'); ax.set_title(f'{region}: pH vs pH x phone Lab directions')
    ax.grid(True,axis='y',alpha=.25); ax.legend(frameon=False); fig.tight_layout(); save_fig(fig,stem,dpi,save_pdf)

def plot_magnitude(ph_mag,region,stem,dpi,save_pdf):
    sub=ph_mag.loc[ph_mag['region']==region].sort_values('ph_nominal')
    fig,ax=plt.subplots(figsize=(8.2,4.9))
    ax.plot(sub['ph_nominal'],sub['mean_abs_interaction_score'],marker='o',label='Mean absolute score')
    ax.plot(sub['ph_nominal'],sub['max_abs_interaction_score'],marker='s',label='Maximum absolute score')
    ax.set_xlabel('Nominal pH'); ax.set_ylabel('Absolute pH x phone ASCA PC1 score'); ax.set_xticks(sub['ph_nominal'])
    ax.set_title(f'{region}: pH-specific interaction magnitude'); ax.grid(True,alpha=.25); ax.legend(frameon=False); fig.tight_layout(); save_fig(fig,stem,dpi,save_pdf)

def main():
    a=parse_args(); vp=resolve_existing(a.variance); lp=resolve_existing(a.loadings); sp=resolve_existing(a.scores); out=resolve_output(a.output_dir); figs=out/'figures'; out.mkdir(parents=True,exist_ok=True)
    variance=pd.read_csv(vp); loadings=pd.read_csv(lp); scores=pd.read_csv(sp); validate_inputs(variance,loadings,scores)
    comp=variance.loc[(variance['scaling']==a.scaling)&(variance['term'].isin(['pH','pH_x_phone']))].copy()
    loading_summary=make_loading_summary(loadings,variance,a.scaling)
    phone_ph,peaks,ph_mag=score_summaries(scores,a.scaling)
    comp.to_csv(out/'component_summary.csv',index=False,encoding='utf-8-sig')
    loading_summary.to_csv(out/'loading_summary.csv',index=False,encoding='utf-8-sig')
    phone_ph.to_csv(out/'ph_phone_score_summary.csv',index=False,encoding='utf-8-sig')
    peaks.to_csv(out/'phone_peak_interactions.csv',index=False,encoding='utf-8-sig')
    ph_mag.to_csv(out/'ph_interaction_magnitude.csv',index=False,encoding='utf-8-sig')
    for region in sorted(loading_summary['region'].unique()):
        plot_loadings(loading_summary,region,figs/f'{region}_pH_vs_interaction_loadings',a.dpi,a.save_pdf)
        plot_magnitude(ph_mag,region,figs/f'{region}_interaction_magnitude_by_ph',a.dpi,a.save_pdf)
    audit={'variance_input':str(vp),'loadings_input':str(lp),'scores_input':str(sp),'output':str(out),'scaling':a.scaling,'note':'ASCA loading signs are arbitrary; interpret relative magnitudes.'}
    (out/'analysis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'analysis_notes.txt').write_text('ASCA pH x phone interpretation\n'+loading_summary.to_string(index=False)+'\n\nPeak interactions\n'+peaks.to_string(index=False),encoding='utf-8')
    print('='*92); print('20. ASCA pH x phone interpretation'); print('='*92)
    print('\nLoading summary'); print(loading_summary.to_string(index=False)); print('\nPeak interactions'); print(peaks.to_string(index=False)); print('\nMain result:'); print(out/'loading_summary.csv')

if __name__=='__main__': main()
