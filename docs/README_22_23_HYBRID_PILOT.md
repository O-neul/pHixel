# Final small hybrid empirical augmentation pilot

This package is intentionally split into **two Python scripts**.

That is preferable to one giant script because the augmentation rule should be
fixed before any classifier test result is inspected.

## 22 — Build the augmentation library

```powershell
python .\scripts\22_build_hybrid_augmentation_library.py
```

It uses:

```text
outputs/factorial_analysis/analysis_factorial_table.csv
```

and creates:

```text
outputs/hybrid_augmentation_library/
├─ augmentation_library.json
├─ background_condition_means.csv
├─ background_pairwise_deltas.csv
├─ contrast_phone_means.csv
├─ contrast_global_pairwise_deltas.csv
├─ contrast_phone_rcs_curves.csv
├─ contrast_trajectory_pairwise_deltas.csv
├─ library_summary.csv
└─ analysis_audit.json
```

The pH-dependent trajectory is estimated only for pH 2-12.

At pH 1 and 13, the pH-dependent trajectory term is disabled. This is fixed
from analyses 18-19, not selected after pilot performance is observed.

## 23 — Run the small LOPO CNN pilot

The pilot compares four methods:

```text
baseline
ColorJitter
global_empirical
hybrid_empirical
```

`global_empirical` is a controlled ablation in the same implementation:

```text
global background nuisance
+ pH-independent phone contrast shift
```

`hybrid_empirical` adds:

```text
+ pH-dependent phone trajectory shape at pH 2-12 only
```

This is **not claimed to reproduce the old empirical augmentation code
bit-for-bit**. It is a matched pH-independent comparator for the new hybrid
method.

### Default pilot size

The runner uses:

```text
2 seeds: 42, 123
3 validation lights: first / middle / last
all holdout phones
4 methods
```

For 3 phones and 5 lights:

```text
3 x 3 x 2 x 4 = 72 model fits
```

The model is pretrained ResNet18:

```text
head-only: 4 epochs
layer4 + head: up to 8 epochs
early stopping patience: 3
BatchNorm statistics frozen
```

No geometry augmentation is used because the color patch location is
scientifically meaningful.

## Required target manifest

Script 23 needs a manifest containing:

```text
phone
light
pH
a path to the already-prepared local-context crop
```

It searches aliases automatically for factor columns.

For the local-context crop path it searches:

```text
local_context_path
local_context_file
context_path
context_file
crop_path
crop_file
preprocessed_path
processed_path
```

If the correct manifest is not auto-detected, run:

```powershell
python .\scripts\23_run_hybrid_augmentation_pilot.py `
  --manifest ".\YOUR\MANIFEST.csv" `
  --image-col "YOUR_CONTEXT_IMAGE_COLUMN"
```

Do **not** point the pilot at full raw photographs. The input should be the same
local-context representation used in the previous CNN experiment.

## ROI mask

If the manifest contains a mask path (`roi_mask_path`,
`context_roi_mask_path`, or `mask_path`), it is used.

Otherwise the script uses a fixed feathered right-side ROI mask because the
existing local-context representation places the background sample on the left
and the reaction ROI on the right.

The fallback rectangle is fixed at:

```text
x = 0.52 to 0.98
y = 0.08 to 0.92
```

with 16-pixel feathering.

If an exact ROI mask is available, prefer it.

## First run: dry validation

Before spending compute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_hybrid_pilot_dry.ps1
```

This validates:

```text
capture analysis table
augmentation library
target manifest
image files
fold sizes
selected validation lights
ROI mask mode
```

and trains no models.

## Full small pilot

After the dry run passes:

```powershell
.\run_hybrid_pilot.ps1
```

## Pre-specified success rule

The pilot is considered successful only if `hybrid_empirical`:

1. has higher overall mean LOPO macro-F1 than `baseline`;
2. has higher overall mean LOPO macro-F1 than `global_empirical`;
3. beats `baseline` in at least two-thirds of holdout phones;
4. beats `global_empirical` in at least two-thirds of holdout phones.

ColorJitter is a secondary comparator.

Do not change augmentation strengths after inspecting test results.

Fixed strengths:

```text
augmentation probability      0.70
background strength           0.50
global contrast strength      0.50
trajectory strength           0.50
```

## Main outputs

```text
outputs/hybrid_augmentation_pilot/
├─ pilot_config.json
├─ fold_audit.csv
├─ pilot_runs.csv
├─ test_predictions.csv
├─ training_history.csv
├─ method_summary.csv
├─ by_phone_summary.csv
├─ paired_comparisons.csv
└─ pilot_decision.csv
```

After this pilot, stop model/augmentation experimentation and write the paper
regardless of whether the pilot succeeds.
