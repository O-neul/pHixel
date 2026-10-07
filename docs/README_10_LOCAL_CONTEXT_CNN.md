# 10 Local-context pretrained CNN smoke test

## Before training

The earlier `audit_grid` output contains only 75 images:

```text
3 phones × 5 lights × 5 selected pH values = 75
```

Training requires all 195 source images. The included PowerShell script first
reruns 09 with:

```text
--selection all
--letterbox-fill white
```

and verifies that the resulting manifest has 195 rows.

## Compared methods

- `baseline`
- `color_jitter`
- `nuisance_structured`

All methods use the same binary LOPO split:

- acidic: pH 1-6
- alkaline: pH 8-13
- pH 7 excluded
- train: 96
- validation: 24
- test: 60

## Model

- ImageNet pretrained ResNet18
- backbone frozen by default
- backbone BatchNorm statistics frozen
- only final classifier trained
- ImageNet normalization
- no class weights
- checkpoint selected using validation macro-F1 and balanced accuracy

The first run may download the official ResNet18 ImageNet weights.

## Empirical nuisance augmentation

The native local crop is transformed before white letterboxing:

```text
background nuisance delta -> whole crop
contrast nuisance delta   -> feathered ROI only
```

Default settings match the accepted preview:

```text
phone scale       = 1.0
light scale       = 1.0
interaction scale = 0.5
residual scale    = 0.0
probability       = 0.8
```

## Run

```powershell
.\run_local_context_cnn_smoke.ps1
```

## Outputs

```text
outputs/local_context_cnn_smoke/<run>/
├─ audit.json
├─ test_comparison.csv
├─ nuisance_input_preview/
├─ baseline/
├─ color_jitter/
└─ nuisance_structured/
```

Each method directory contains:

- best checkpoint
- training history
- train/validation/test predictions
- confusion matrices
- classification reports
- test metrics by pH

## First checks

1. Inspect `nuisance_input_preview/`.
2. Confirm that no method predicts only one class.
3. Compare test balanced accuracy and macro-F1.
4. Check pH 6, 8, and 9 separately.
5. Do not tune settings against this held-out-phone test result.
