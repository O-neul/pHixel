# 17. Leave-one-strip-out spline validation

## Purpose

This is the next validation after the cell-level model comparison.

Each fold removes all five lighting images belonging to one physical strip:

```text
one phone × one pH × five lights
```

The held-out strip is predicted from the remaining pH levels.

## Models

```text
spline_additive
spline(pH) + phone + light + phone×light

phone_spline
spline_additive + phone×spline(pH)

full_spline_interactions
phone_spline + light×spline(pH)
```

The primary scientific comparison is:

```text
phone_spline - spline_additive
```

Negative error differences favor the phone-specific smooth pH trajectory.

The comparison between `full_spline_interactions` and `phone_spline` checks
whether pH-dependent light trajectories provide additional value beyond the
phone-specific pH curve.

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_strip_holdout_spline.ps1
```

## Spline sensitivity

The default run uses cubic B-splines with basis counts:

```text
4, 5, 6
```

The primary setting is 5.

A reliable conclusion should have the same direction across multiple basis
counts. Large changes across basis counts indicate spline-flexibility
sensitivity and should be reported.

## Main outputs

```text
outputs/strip_holdout_spline/
├─ cv_predictions.csv
├─ fold_metrics.csv
├─ cv_summary.csv
├─ metrics_by_ph.csv
├─ metrics_by_phone.csv
├─ paired_model_comparison.csv
├─ spline_sensitivity_summary.csv
├─ design_rank_audit.csv
├─ analysis_audit.json
├─ analysis_notes.txt
└─ figures/
```

## First files to inspect

```text
paired_model_comparison.csv
cv_summary.csv
metrics_by_ph.csv
spline_sensitivity_summary.csv
```

## Interpretation

For paired comparisons:

```text
negative difference
= numerator model has lower held-out-strip error

bootstrap interval entirely below zero
= improvement is reasonably consistent across the 39 strips
```

The bootstrap resamples strips within each phone, preserving the fixed
three-phone composition.

## Important limitation

This is the strongest validation available without collecting more images, but
it is not equivalent to testing a second independent strip at the same
phone × pH combination.

There is only one physical strip per phone × pH cell. Therefore this analysis
tests whether neighboring pH levels support interpolation of an unobserved cell,
not whether two separate strips at an identical pH produce the same result.

Boundary pH values 1 and 13 may have larger errors because prediction is based
on a one-sided smooth trajectory.
