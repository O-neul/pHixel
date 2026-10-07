# Scripts 15 and 16

Both scripts use only NumPy, pandas, Matplotlib, and the Python standard
library. They do not use statsmodels.

## 15. Multivariate Lab decomposition and ASCA

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_multivariate_lab_asca.ps1
```

Primary output:

```text
outputs/multivariate_lab/effect_composition.csv
```

Use the rows where:

```text
scaling = standardized
```

for the main comparison among Pad, Background, and Contrast.

Additional outputs:

```text
multivariate_effect_shares.csv
sscp_matrices_long.csv
manova_descriptive_statistics.csv
asca_variance.csv
asca_loadings.csv
asca_scores.csv
figures/
```

The MANOVA statistics are descriptive only. The script does not attach formal
p-values to Pillai trace or Wilks lambda because the residual combines the
three-way interaction with pure error.

ASCA loading signs are arbitrary. A loading vector and its negative represent
the same component.

## 16. Additive versus interaction model comparison

```powershell
.\run_model_structure_comparison.ps1
```

Models:

```text
main_effects
additive_nuisance
ph_dependent_interactions
```

Primary comparison:

```text
ph_dependent_interactions - additive_nuisance
```

Negative error differences favor the interaction model.

Cross-validation schemes:

```text
leave_one_cell_out
cyclic_light_block
```

`cyclic_light_block` holds out one light observation from every strip in each
fold while keeping every light level represented in training. Five folds test
every image exactly once.

This analysis is not LOPO. It tests model structure within the observed phone,
pH, and light levels. The earlier LOPO classification results remain the
evidence for unseen-phone generalization.

Primary outputs:

```text
outputs/model_structure_comparison/
├─ cv_summary.csv
├─ paired_model_comparison.csv
├─ cv_metrics_by_ph.csv
├─ cv_metrics_by_phone.csv
├─ cv_metrics_by_light.csv
├─ full_data_fit_by_channel.csv
├─ design_rank_audit.csv
└─ figures/
```

Bootstrap confidence intervals resample strip-level paired differences within
each phone, preserving the fixed three-phone composition.
