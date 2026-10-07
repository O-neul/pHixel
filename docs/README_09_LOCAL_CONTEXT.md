# 09 Local-context input audit

The uploaded nuisance-preview code has only one spatial box: the LabelMe ROI.
Its background delta is applied to the whole image. Therefore this script
defines a local background box explicitly.

## Default geometry

- ROI: LabelMe `roi`
- background: same size as ROI
- preferred side: left
- horizontal gap: 0.25 × ROI width
- final context: union of ROI and background, plus 15% margin
- model input: 224 × 224 letterbox with crop median padding

If the left background box does not fit, the script automatically tries the
right side and records the fallback.

## Run

```powershell
.\run_local_context_audit.ps1
```

## Outputs

```text
outputs/local_context_audit/
├─ overlays/
├─ crops_raw/
├─ model_inputs/
├─ contact_sheets/
├─ local_context_manifest.csv
├─ audit_summary.csv
└─ local_context_config.json
```

## Visual acceptance criteria

Every model input should contain:

- the reaction/strip ROI
- a nearby local background patch
- a small amount of surrounding context

Every model input should exclude:

- handwritten pH numbers
- phone model labels
- light labels
- distant scene layout

The script does not use OCR. Contact-sheet inspection is mandatory before
training.

## Geometry adjustment examples

Move the background farther left:

```powershell
--bg-gap-ratio 0.5
```

Use a narrower background patch:

```powershell
--bg-width-ratio 0.75
```

Add less context:

```powershell
--context-margin-ratio 0.08
```

Force the right side:

```powershell
--bg-side right
```
