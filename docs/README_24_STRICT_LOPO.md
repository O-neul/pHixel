# Strict target-free LOPO rerun

## Why this rerun exists

The previous hybrid pilot excluded the holdout phone from CNN training, but its
single global augmentation library had been estimated from the full 195-image
capture table. Therefore the holdout phone's Lab characteristics were available
to the augmentation procedure.

That previous experiment is still interpretable as a **calibration-assisted**
scenario:

- target-phone images are not used to train the CNN;
- target-phone color characteristics are known to the augmentation rule.

It does **not** establish fully unseen-device, target-free generalization.

This package adds a stricter rerun.

## What script 24 changes

For each LOPO fold, script 24 rebuilds the empirical augmentation library from
the **exact classifier-training cells only**.

With the current 3-phone x 5-light binary design:

```text
train = 2 source phones x 4 training lights x 12 pH levels = 96 cells
val   = 2 source phones x 1 validation light x 12 pH levels = 24 images
test  = 1 holdout phone x 5 lights x 12 pH levels = 60 images
```

The fold library also uses exactly those 96 training cells.

Therefore every fold must have:

```text
n_library_rows = 96
target_phone_rows_in_library = 0
validation_light_rows_in_library = 0
pH7_rows_in_library = 0
```

For two source phones and four training lights the deterministic library sizes
are:

```text
background pairwise deltas = 8 conditions x 7 directed alternatives = 56
global phone contrast deltas = 2 directed vectors
pH-dependent trajectory deltas = 10 internal binary pH levels x 2 directions = 20
```

The internal trajectory levels are pH 2-6 and 8-12. pH 7 is excluded from the
binary task, and pH 1/13 keep the trajectory term OFF.

## ROI handling

The script uses the exact manifest coordinates when these columns exist:

```text
roi_model_x1
roi_model_y1
roi_model_x2
roi_model_y2
```

Expected dry-run output:

```text
Mask mode: manifest_roi_model_box
STRICT AUDIT: PASS
```

## Run order

Copy the Python script to `scripts/` and the two PowerShell files to the project
root.

First:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_strict_lopo_dry.ps1
```

Only if all 9 folds pass the audit, run:

```powershell
.\run_strict_lopo_FULL.ps1
```

## Do not overwrite the previous pilot

The strict rerun writes to:

```text
outputs/hybrid_augmentation_pilot_strict_lopo
```

Keep the previous `outputs/hybrid_augmentation_pilot` directory. The two
experiments answer different questions:

1. previous pilot: target characteristics known, target images excluded from CNN
   training (calibration-assisted);
2. strict rerun: target characteristics completely unknown during training and
   augmentation-library estimation (target-free unseen-device LOPO).

If the strict hybrid pilot still improves over baseline/global empirical, the
claim becomes much stronger: source-domain structure alone can improve
generalization to an unseen phone.

If the gain disappears, the result is also useful: the earlier benefit depended
on target-device characterization, supporting a calibration-assisted deployment
strategy rather than target-free zero-shot deployment.
