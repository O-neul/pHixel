# 08 Feature-space nuisance augmentation

This is the first direct test of whether train-only empirical nuisance
recombination improves acidic-vs-alkaline classification on an unseen phone.

## Methods

- `baseline`: original train features only
- `gaussian_matched`: original rows plus Gaussian noise whose covariance is
  matched to empirical nuisance deltas
- `nuisance_recombined`: original rows plus train-only phone/light/interaction
  recombinations

## Default primary settings

- pH 7 excluded
- feature groups: `pad_only`, `contrast_only`
- 4 augmented rows per original train row
- phone scale: 1.0
- light scale: 1.0
- interaction scale: 0.5
- residual scale: 0.0
- Logistic Regression
- scaler fitted on original train rows only
- C selected on validation only

## Run the one-split smoke test

```powershell
.\run_feature_nuisance_smoke_one_split.ps1
```

## Main outputs

- `test_comparison.csv`
- `augmentation_summary.csv`
- `validation_candidates.csv`
- per-method predictions, confusion matrices, coefficients, and pH-specific metrics

## Primary interpretation

Check all of the following, not only average accuracy:

- test balanced accuracy and macro-F1
- pH 8 accuracy
- pH 6 and pH 9 preservation
- whether nuisance outperforms matched Gaussian noise
- whether improvement occurs for `contrast_only`, not only `pad_only`

This first run is exploratory. Fix the augmentation settings before running the
final multi-split experiment.
