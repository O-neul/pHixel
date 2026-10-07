# 14 without statsmodels

The import error came from an incompatible `statsmodels` installation. This
replacement does not import statsmodels.

## Why direct ANOVA is appropriate here

The data form a complete balanced design:

```text
13 pH × 3 phones × 5 lights = 195 cells
```

There is one observation in every cell. Therefore the sums of squares for the
three main effects and all two-way interactions can be calculated directly
from marginal means.

## Model

```text
Y = pH + phone + light
  + pH×phone
  + pH×light
  + phone×light
  + residual
```

Degrees of freedom:

```text
pH             12
phone           2
light           4
pH×phone       24
pH×light       48
phone×light     8
residual        96
total          194
```

The residual contains the unseparated three-way interaction and pure error.

## Run

Replace the existing files with:

```text
scripts/14_factorial_effect_analysis.py
run_factorial_effect_analysis.ps1
```

Then run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_factorial_effect_analysis.ps1
```

## Dependency

Only SciPy is used for F-distribution p-values. It is normally already present
because scikit-learn depends on SciPy. If needed:

```powershell
python -m pip install scipy
```

## Main outputs

```text
outputs/factorial_effects/
├─ anova_long.csv
├─ feature_effect_summary.csv
├─ effect_composition.csv
├─ region_effect_summary.csv
├─ model_fit_summary.csv
├─ paper_effect_size_table.csv
├─ analysis_audit.json
└─ analysis_notes.txt
```
