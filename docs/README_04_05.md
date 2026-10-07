# 04–05: decomposition summary and nuisance preview

## Step 04 — summarize all 15 decomposition runs

Run from the project root:

```powershell
python .\scripts\04_summarize_decomposition.py
```

Expected input:

```text
outputs/decomposition/capture_lopo_holdout_*/
```

Expected output:

```text
outputs/decomposition_summary/
├── fit_summary_all_folds.csv
├── component_level_vectors.csv
├── component_magnitude_all_folds.csv
├── nuisance_distribution_all_folds.csv
├── fold_summary.csv
├── audit_summary.json
└── plots/
```

Important checks:

- `matched_run_count = 15`
- `successful_run_count = 15`
- `fold_summary.csv` has 15 rows
- `fit_summary_all_folds.csv` has 135 rows (`15 folds × 9 color variables`)
- inspect the minimum R² and residual magnitudes before preview generation

## Step 05 — create ROI-aware previews for one fold

Choose one split and the decomposition directory created from exactly the same split. Example:

```powershell
python .\scripts\05_preview_nuisance_recombination.py `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --decomposition-dir ".\outputs\decomposition\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\previews\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --max-images 12 `
  --variants-per-image 4 `
  --seed 42 `
  --sampler structured `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --feather-radius 24
```

The script enforces the following:

- only `train` rows are selected as source images;
- the nuisance library must contain exactly the same image IDs as the train split;
- the split recorded in `decomposition_config.json` must match `--split-file`;
- the held-out phone and validation light are therefore not used;
- background nuisance is applied to the whole image;
- pad-minus-background contrast nuisance is applied only to the LabelMe `roi` rectangle;
- the ROI border is feathered to avoid a hard rectangular edge.

Outputs:

```text
outputs/previews/<fold>/
├── individual/
├── contact_sheets/
├── preview_metadata.csv
├── preview_summary.csv
└── preview_config.json
```

Start by checking `contact_sheets`. Then inspect these fields in `preview_summary.csv`:

- `pad_rgb_gamut_clipped_ratio`
- `outside_rgb_gamut_clipped_ratio`
- `pad_total_delta_E`
- `bg_delta_E`
- `contrast_delta_E`

The initial preview intentionally excludes residual transfer (`residual_scale=0.0`). Residual recombination should be tested only after the structured phone/light/interaction preview looks realistic.

## Sampler meanings

- `structured`: independently samples a target phone and light, then uses the interaction estimated for that phone–light cell. Recommended first.
- `independent`: independently samples phone, light, and interaction effects. More diverse but less physically constrained.
- `swap_row`: takes all target components from one observed train row. Most conservative.

## Initial decision rule

Proceed to a training augmentation class when:

1. ROI boundaries are not visible;
2. the reaction-pad color remains plausible for the original pH;
3. no systematic saturation or severe clipping appears;
4. the transformation is visible but not destructive;
5. results are stable across multiple source phones, lights, and pH levels.
