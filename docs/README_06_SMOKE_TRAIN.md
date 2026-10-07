# 06 — One-split smoke training

`06_smoke_train_one_split.py` runs the following methods under the same split,
normalization, model initialization seed, optimizer, and evaluation procedure:

1. `baseline`
2. `colorjitter`
3. `nuisance`
4. `nuisance_consistency`

The nuisance library must be the output of script 03 for the exact supplied
split. Validation and test images are not augmented. The final train metric is
also calculated from an unaugmented loader.

## Recommended first command

From the project root:

```powershell
python .\scripts\06_smoke_train_one_split.py `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --decomposition-dir ".\outputs\decomposition\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\smoke\capture_lopo_holdout_galaxy_s24plus_vallight_L01_seed42" `
  --methods "baseline,colorjitter,nuisance,nuisance_consistency" `
  --model resnet18 `
  --epochs 5 `
  --patience 5 `
  --batch-size 8 `
  --image-size 224 `
  --num-workers 0 `
  --seed 42 `
  --nuisance-probability 0.5 `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --feather-radius 24 `
  --consistency-weight 0.5
```

For a faster code-only check, use `--model tiny_cnn`. Do not interpret that
quick run as a final model comparison.

## Outputs

```text
outputs/smoke/<run>/
├── normalization_stats.json
├── split_audit.json
├── smoke_comparison.csv
├── baseline/
├── colorjitter/
├── nuisance/
└── nuisance_consistency/
```

Each method folder contains:

- `best_model.pt`
- `train_log.csv`
- `metrics_summary.csv`
- `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv`
- confusion matrices and classification reports
- `run_config.json`
- `augmentation_stats.json` for nuisance methods

## What counts as a successful smoke test

- all four methods finish without exceptions or NaN losses;
- the split audit reports 104 train, 26 validation, and 65 test rows;
- train/test phones do not overlap;
- train/validation lights do not overlap;
- nuisance application rate is close to 0.5 for `nuisance`;
- nuisance application rate is close to 1.0 for `nuisance_consistency`;
- all method folders and `smoke_comparison.csv` are created.

Five epochs on one split are not sufficient for the research conclusion.
