#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a decision-ready synthesis from analyses 14-20."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--region-effects',default='outputs/factorial_effects/region_effect_summary.csv')
    p.add_argument('--multivariate',default='outputs/multivariate_lab/effect_composition.csv')
    p.add_argument('--cell-model-comparison',default='outputs/model_structure_comparison/paired_model_comparison.csv')
    p.add_argument('--strip-diagnostics',default='outputs/strip_holdout_diagnostics/subset_sensitivity.csv')
    p.add_argument('--boundary-stabilized',default='outputs/boundary_stabilized_models/subset_comparisons.csv')
    p.add_argument('--asca',default='outputs/asca_interaction_summary/loading_summary.csv')
    p.add_argument('--augmentation',default='')
    p.add_argument('--output-dir',default='outputs/final_evidence_synthesis')
    p.add_argument('--primary-knot-count',type=int,default=4)
    return p.parse_args()

def resolve_existing(text):
    path=Path(text); candidates=[path]
    if not path.is_absolute(): candidates += [PROJECT_ROOT/path,Path.cwd()/path]
    for c in candidates:
        c=c.resolve()
        if c.exists(): return c
    raise FileNotFoundError(text)

def resolve_output(text):
    path=Path(text); return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()

def first_value(df,mask,column):
    sub=df.loc[mask,column]
    return float(sub.iloc[0]) if not sub.empty else np.nan

def build_matrix(region_effects,multivariate,cell,strip,stabilized,asca,primary_knot):
    rows=[]; standardized=multivariate.loc[multivariate['scaling']=='standardized']
    regions=sorted(set(region_effects['region'])|set(standardized['region']))
    for region in regions:
        row={'region':region}
        rr=region_effects.loc[region_effects['region']==region]
        if not rr.empty:
            r=rr.iloc[0]; row.update({
                'univariate_mean_pH_signal_ratio':float(r['mean_pH_signal_ratio']),
                'univariate_mean_nuisance_main_ratio':float(r['mean_nuisance_main_ratio']),
                'univariate_mean_pH_dependent_interaction_ratio':float(r['mean_pH_dependent_interaction_ratio'])})
        mm=standardized.loc[standardized['region']==region]
        if not mm.empty:
            r=mm.iloc[0]; row.update({
                'multivariate_pH_signal_share':float(r['pH_signal_share']),
                'multivariate_nuisance_main_share':float(r['nuisance_main_share']),
                'multivariate_pH_dependent_interaction_share':float(r['pH_dependent_interaction_share'])})
        mask=(cell['region']==region)&(cell['metric']=='deltaE76')&(cell['comparison']=='ph_dependent_interactions minus additive_nuisance')&(cell['scheme']=='leave_one_cell_out')
        row['cell_level_interaction_deltaE_difference']=first_value(cell,mask,'mean_difference')
        row['cell_level_interaction_ci_upper']=first_value(cell,mask,'bootstrap_ci_upper')
        mask=(strip['region']==region)&(strip['basis_count']==5)&(strip['subset']=='interior_ph_2_12')
        row['bspline_phone_interior_deltaE_difference']=first_value(strip,mask,'mean_difference')
        row['bspline_phone_interior_ci_upper']=first_value(strip,mask,'bootstrap_ci_upper')
        mask=(strip['region']==region)&(strip['basis_count']==5)&(strip['subset']=='boundary_ph_1_13')
        row['bspline_phone_boundary_deltaE_difference']=first_value(strip,mask,'mean_difference')
        mask=(stabilized['region']==region)&(stabilized['knot_count']==primary_knot)&(stabilized['subset']=='interior_ph_2_12')&(stabilized['numerator_model']=='rcs_phone_ridge')&(stabilized['denominator_model']=='rcs_additive')
        row['ridge_rcs_interior_deltaE_difference']=first_value(stabilized,mask,'mean_difference')
        row['ridge_rcs_interior_ci_upper']=first_value(stabilized,mask,'bootstrap_ci_upper')
        mask=(stabilized['region']==region)&(stabilized['knot_count']==primary_knot)&(stabilized['subset']=='boundary_ph_1_13')&(stabilized['numerator_model']=='rcs_phone_ridge')&(stabilized['denominator_model']=='rcs_additive')
        row['ridge_rcs_boundary_deltaE_difference']=first_value(stabilized,mask,'mean_difference')
        row['ridge_rcs_boundary_ci_upper']=first_value(stabilized,mask,'bootstrap_ci_upper')
        aa=asca.loc[(asca['region']==region)&(asca['term']=='pH_x_phone')]
        if not aa.empty:
            a=aa.iloc[0]; row.update({'asca_interaction_pc1_explained_ratio':float(a['explained_ratio_within_term']),'asca_interaction_dominant_channel':str(a['dominant_channel']),'asca_interaction_dominant_abs_loading':float(a['dominant_abs_loading'])})
        rows.append(row)
    return pd.DataFrame(rows)

def evidence_summary(matrix):
    def rv(region,col):
        sub=matrix.loc[matrix['region']==region]
        return sub.iloc[0][col] if (not sub.empty and col in sub.columns) else np.nan
    rows=[]
    bg=rv('bg','multivariate_nuisance_main_share'); padn=rv('pad','multivariate_nuisance_main_share'); conn=rv('contrast','multivariate_nuisance_main_share'); pads=rv('pad','multivariate_pH_signal_share'); cons=rv('contrast','multivariate_pH_signal_share')
    rows += [
        {'evidence_id':'E1','question':'Is background primarily nuisance?','value':bg,'direction':'Higher supports nuisance dominance.','interpretation_rule':'>0.8 is strong descriptive dominance.'},
        {'evidence_id':'E2','question':'Does contrast reduce global nuisance vs pad?','value':padn-conn,'direction':'Positive means nuisance reduction.','interpretation_rule':'Use magnitude descriptively.'},
        {'evidence_id':'E3','question':'Does contrast increase relative pH signal?','value':cons-pads,'direction':'Positive means higher pH signal share.','interpretation_rule':'Use magnitude descriptively.'},
    ]
    for region in ['pad','contrast']:
        rows += [
            {'evidence_id':f'E4_{region}','question':f'Does categorical pH-dependent interaction improve held-out cells in {region}?','value':rv(region,'cell_level_interaction_deltaE_difference'),'direction':'Negative favors interaction.','interpretation_rule':'CI upper < 0 = strong within-cell evidence.'},
            {'evidence_id':f'E5_{region}','question':f'Does ridge RCS phone trajectory improve unseen-strip interior pH in {region}?','value':rv(region,'ridge_rcs_interior_deltaE_difference'),'direction':'Negative favors phone-specific RCS.','interpretation_rule':'CI upper < 0 = strong transferable-structure evidence; mean < 0 with CI crossing 0 = suggestive.'},
            {'evidence_id':f'E6_{region}','question':f'Is boundary prediction stabilized in {region}?','value':rv(region,'ridge_rcs_boundary_deltaE_difference'),'direction':'Near/below 0 is favorable.','interpretation_rule':'Large positive values mean boundary-conditioned augmentation remains risky.'},
        ]
    return pd.DataFrame(rows)

def decision_questions():
    return pd.DataFrame([
        {'decision_step':1,'question':'Is background/global nuisance reproducible enough for a new additive augmentation pilot?','look_at':'Background nuisance share plus prior augmentation results.','if_yes':'A small redesigned global-nuisance pilot is justified.','if_no':'Do not repeat empirical nuisance augmentation yet.'},
        {'decision_step':2,'question':'Does ridge-stabilized phone-specific pH structure improve unseen-strip interior interpolation?','look_at':'Pad/Contrast ridge_rcs_interior_deltaE_difference and CI.','if_yes':'A pH-conditioned trajectory augmentation pilot is scientifically motivated.','if_no':'Do not encode phone x pH trajectories from the current dataset.'},
        {'decision_step':3,'question':'Are pH 1 and 13 stable after RCS + ridge?','look_at':'Pad/Contrast ridge_rcs_boundary_deltaE_difference.','if_yes':'A pilot may cover the full pH range.','if_no':'Restrict to interpolation pH or collect boundary replicates.'},
        {'decision_step':4,'question':'Would the new experiment separate strip variation from phone variation?','look_at':'Use repeated independent strips and the same strip across phones/lights; estimate augmentation only from training folds.','if_yes':'Proceed with a small pilot first.','if_no':'Use the current negative augmentation result in the paper instead of repeating the experiment.'},
    ])

def main():
    a=parse_args(); paths={
        'region_effects':resolve_existing(a.region_effects),
        'multivariate':resolve_existing(a.multivariate),
        'cell_model_comparison':resolve_existing(a.cell_model_comparison),
        'strip_diagnostics':resolve_existing(a.strip_diagnostics),
        'boundary_stabilized':resolve_existing(a.boundary_stabilized),
        'asca':resolve_existing(a.asca)}
    out=resolve_output(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    dfs={k:pd.read_csv(v) for k,v in paths.items()}
    matrix=build_matrix(dfs['region_effects'],dfs['multivariate'],dfs['cell_model_comparison'],dfs['strip_diagnostics'],dfs['boundary_stabilized'],dfs['asca'],a.primary_knot_count)
    evidence=evidence_summary(matrix); decisions=decision_questions()
    matrix.to_csv(out/'region_evidence_matrix.csv',index=False,encoding='utf-8-sig')
    evidence.to_csv(out/'evidence_summary.csv',index=False,encoding='utf-8-sig')
    decisions.to_csv(out/'decision_questions.csv',index=False,encoding='utf-8-sig')
    stab_cols=[c for c in matrix.columns if ('bspline_phone' in c or 'ridge_rcs' in c)]
    matrix[['region']+stab_cols].to_csv(out/'boundary_stabilization_summary.csv',index=False,encoding='utf-8-sig')
    aug_path=None
    if a.augmentation.strip():
        aug_path=resolve_existing(a.augmentation); pd.read_csv(aug_path).to_csv(out/'optional_augmentation_results_copy.csv',index=False,encoding='utf-8-sig')
    audit={'inputs':{k:str(v) for k,v in paths.items()},'augmentation_input':str(aug_path) if aug_path else None,'primary_knot_count':a.primary_knot_count,'purpose':'decision-ready synthesis; no automatic scientific decision'}
    (out/'analysis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    notes='Final evidence synthesis\n'+'='*80+'\n\nRegion evidence matrix\n'+matrix.to_string(index=False)+'\n\nEvidence summary\n'+evidence.to_string(index=False)+'\n\nDecision questions\n'+decisions.to_string(index=False)
    (out/'final_evidence_notes.txt').write_text(notes,encoding='utf-8')
    print('='*96); print('21. Final evidence synthesis'); print('='*96); print('\nRegion evidence matrix'); print(matrix.to_string(index=False)); print('\nEvidence summary'); print(evidence.to_string(index=False)); print('\nDecision questions'); print(decisions.to_string(index=False)); print('\nMain result:'); print(out/'region_evidence_matrix.csv')

if __name__=='__main__': main()
