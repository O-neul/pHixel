$ErrorActionPreference = "Stop"

Write-Host "Step 1: build canonical manifest"
python .\scripts\01_build_canonical_manifest.py `
  --features ".\data\raw\captures\captures_final_features.csv" `
  --image-dir ".\data\raw\captures\images"

Write-Host "Step 2: create LOPO splits"
python .\scripts\02_make_lopo_splits.py

Write-Host "Generated split files:"
Get-ChildItem .\data\splits\capture_lopo_holdout_*.csv |
  Select-Object Name

Write-Host "Choose one split file, then run 03_fit_nuisance_decomposition.py as shown in README_START_HERE.md."
