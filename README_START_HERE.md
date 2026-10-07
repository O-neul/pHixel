# Empirical Nuisance Decomposition for Robust pH Classification

## Research question

Can phone and illumination effects measured from a small repeated-capture experiment be decomposed and recombined as empirical nuisance augmentation, then combined with prediction-consistency learning to improve pH classification on unseen phones and lights?

This starter project intentionally begins with data auditing and leakage-safe decomposition. It does **not** immediately train a CNN. The nuisance effects must be estimated only from each split's training rows.

## What data to keep

### Required

1. **195 original direct-capture images**
   - 3 phones × 5 lights × 13 pH levels.
   - Keep the untouched originals, not only crops or preview images.
2. **`captures_final_features.csv`**
   - Must identify image, phone, light, pH and reaction-pad/background color features.
   - Lab columns are preferred. RGB columns are accepted and converted to Lab.
3. **ROI annotations or masks**
   - LabelMe JSON, masks, or reproducible ROI coordinates for reaction pad and background.
   - The first three scripts can run from the feature CSV alone, but ROI-aware image augmentation and XAI later require these annotations.

### Strongly recommended for reproducibility

- The script that created `captures_final_features.csv`.
- A short capture protocol document: phone names, light labels, pH preparation, capture order, distance/exposure control.
- A table connecting every image file to `phone_id`, `light_id`, `ph_nominal`, and `strip_id`.

### Optional external stress test

- QUB `three_smartphones` raw images and its manifest/splits.
- Keep it under `data/raw/external_qub/`.
- QUB is not required for the main direct-capture domain-generalization experiment. It should be treated as a stronger cross-dataset transfer test.

### Do not carry into the new core project unless needed as a reference

- Old pairwise preview images.
- Old generated pairwise-shift CSVs; they can be regenerated.
- Old QUB model checkpoints and logs.
- Duplicate/renamed copies of the same capture image.
- `_bg.jpg` files from QUB unless the external paired XAI test is retained.

## Project structure

```text
nuisance_dg_starter/
├── config/
│   └── column_map.example.json
├── data/
│   ├── raw/
│   │   ├── captures/
│   │   │   ├── captures_final_features.csv
│   │   │   ├── images/                 # 195 originals
│   │   │   └── annotations/            # LabelMe JSON or masks
│   │   ├── external_qub/               # optional
│   │   └── data_inventory.json
│   ├── interim/                         # generated ROI masks/crops later
│   ├── processed/
│   │   └── capture_manifest.csv
│   └── splits/
├── docs/
├── outputs/
│   ├── audit/
│   ├── decomposition/
│   ├── previews/
│   ├── models/
│   └── evaluation/
├── scripts/
│   ├── 00_import_required_data.py
│   ├── 01_build_canonical_manifest.py
│   ├── 02_make_lopo_splits.py
│   └── 03_fit_nuisance_decomposition.py
├── src/
│   └── color_utils.py
├── requirements.txt
└── run_first_steps.ps1
```

## First execution sequence

Run the following from the project root in PowerShell.

### 0. Import only the required data

```powershell
python .\scripts\00_import_required_data.py `
  --features "C:\path\to\captures_final_features.csv" `
  --images "C:\path\to\direct_capture_images" `
  --annotations "C:\path\to\labelme_json_or_masks"
```

The annotation option may be omitted temporarily. Add `--external-qub` only when retaining the optional external test.

### 1. Build and audit a canonical manifest

First try automatic column detection:

```powershell
python .\scripts\01_build_canonical_manifest.py `
  --features ".\data\raw\captures\captures_final_features.csv" `
  --image-dir ".\data\raw\captures\images"
```

Expected balanced counts:

```text
rows = 195
phones = 3
lights = 5
pH levels = 13
expected full factorial rows = 195
```

If a feature column is not detected, copy `config/column_map.example.json`, fill in the exact CSV column names, and run:

```powershell
Copy-Item .\config\column_map.example.json .\config\column_map.json
notepad .\config\column_map.json

python .\scripts\01_build_canonical_manifest.py `
  --features ".\data\raw\captures\captures_final_features.csv" `
  --image-dir ".\data\raw\captures\images" `
  --column-map ".\config\column_map.json"
```

Outputs:

- `data/processed/capture_manifest.csv`
- `outputs/audit/capture_manifest_audit.json`

The canonical manifest contains:

```text
image_id, image_path, phone_id, phone_label,
light_id, light_label, ph_nominal, label_class, strip_id,
pad_L, pad_a, pad_b,
bg_L, bg_a, bg_b,
contrast_L, contrast_a, contrast_b
```

`contrast_*` is reaction-pad Lab minus background Lab.

### 2. Generate leakage-safe LOPO splits

```powershell
python .\scripts\02_make_lopo_splits.py
```

With 3 phones and 5 lights, this creates 15 files:

```text
3 held-out test phones × 5 validation lights
```

For each split:

- **test:** all 65 images from one held-out phone.
- **validation:** one held-out light from the two source phones, 26 images.
- **train:** the remaining two phones and four lights, 104 images.

The held-out phone never enters training or nuisance estimation. The validation light also never enters training or nuisance estimation.

### 3. Fit the first nuisance decomposition

Choose one split for the first diagnostic run. Replace the phone/light IDs with an actual generated filename.

```powershell
python .\scripts\03_fit_nuisance_decomposition.py `
  --split-file ".\data\splits\capture_lopo_holdout_<PHONE>_vallight_<LIGHT>.csv" `
  --output-dir ".\outputs\decomposition\holdout_<PHONE>_vallight_<LIGHT>"
```

For each Lab channel and each region, the script fits the balanced decomposition:

```text
observed color
= grand mean
+ pH effect
+ phone effect
+ light effect
+ phone × light interaction
+ residual
```

It fits this model to **train rows only** and saves:

- `component_effects.csv`
- `row_decomposition.csv`
- `fit_summary.csv`
- `nuisance_vector_library.csv`
- `decomposition_config.json`

The `nuisance_vector_library.csv` is the starting material for the later recombination augmentation. It keeps phone, light, interaction, and residual Lab vectors separate instead of copying a single observed A→B shift.

## Why the first split is structured this way

The main test is unseen-phone robustness. Validation is deliberately an unseen light among source phones, so model selection does not use the held-out phone. This gives an early warning when a method fits known phones but cannot tolerate an unobserved illumination.

Later experiments can add:

1. unseen-light evaluation across all phones;
2. joint phone-and-light stress tests;
3. recombined nuisance augmentation;
4. original/augmented consistency loss;
5. ROI-aware transforms and XAI;
6. optional QUB cross-dataset transfer.

## Important limitations of the current data

- There are only three phones. Every LOPO training fold estimates phone variability from only two source phones.
- There is one image for each phone-light-pH cell, so within-cell image noise cannot be estimated independently from residual variation.
- Strong claims about arbitrary smartphones are not justified. The defensible target is improvement on held-out devices/lights within this controlled capture design, followed by an optional external stress test.
- The pH color itself and nuisance effects may interact. The initial additive model is a transparent baseline, not an assumption that must remain fixed. The row residuals and pad/background contrast are retained to diagnose this.
