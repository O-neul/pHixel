# 18. Strip-holdout failure diagnostics

## Purpose

Script 17 showed that `phone_spline` did not improve average held-out-strip
prediction. Script 18 diagnoses why.

It reads the predictions already produced by script 17:

```text
outputs/strip_holdout_spline/cv_predictions.csv
```

No new model is fitted.

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_strip_holdout_diagnostics.ps1
```

## Primary comparison

```text
phone_spline - spline_additive
```

Interpretation:

```text
negative difference = phone_spline better
positive difference = spline_additive better
```

## Diagnostics

### pH-specific differences

```text
summary_by_ph.csv
```

There are only three held-out strips at each pH, one per phone. These results
are descriptive.

### Phone-specific differences

```text
summary_by_phone.csv
```

Shows whether deterioration is concentrated in one phone.

### Boundary sensitivity

```text
subset_sensitivity.csv
```

Subsets include:

```text
all pH
boundary pH 1 and 13
interior pH 2 through 12
strict interior pH 3 through 11
highlighted pH 8 and 12
non-highlighted pH
```

If the phone spline becomes better after removing pH 1 and 13, the failure may
mainly reflect boundary prediction. If it remains worse in pH 2 through 12,
the generalization problem is not explained by boundary behavior alone.

### Robust location summaries

```text
robust_location_summary.csv
```

Reports:

```text
mean
median
10% trimmed mean
standard deviation
minimum and maximum
proportion of strips favoring phone_spline
bootstrap interval for the mean
```

A positive mean with a negative median indicates that a minority of very large
failures dominate the average.

### Channel-specific diagnosis

```text
channel_difference_summary.csv
```

Shows whether deterioration is driven mainly by L*, a*, or b*.

### Extreme strips

```text
worst_strips.csv
best_strips.csv
```

`worst_strips.csv` lists strips with the largest positive paired differences.

### Outlier concentration

```text
outlier_concentration.csv
```

Shows what proportion of total deterioration is explained by the largest 1, 3,
5, and 10 failing strips.

### Spline flexibility

```text
basis_sensitivity.csv
```

Compares all-pH mean, trimmed mean, and interior-pH mean across basis counts
4, 5, and 6.

## Main figures

```text
figures/
├─ *_difference_by_ph_basis5.png
├─ *_sorted_strip_differences_basis5.png
├─ *_channel_differences_by_ph_basis5.png
├─ *_basis_sensitivity.png
└─ *_subset_sensitivity_basis5.png
```

## Important interpretation limit

There is one physical strip per phone x pH cell. This diagnostic can identify
which observed cells cause spline failure, but it cannot separate true
phone-specific response irregularity from strip-specific variation.
