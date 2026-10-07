$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "24. STRICT target-free LOPO dry-run"
Write-Host "============================================================"
Write-Host ""
Write-Host "This run builds a NEW augmentation library inside each fold."
Write-Host "The holdout phone and validation light must contribute 0 rows."
Write-Host ""

python .\scripts\24_run_hybrid_augmentation_pilot_strict_lopo.py `
  --manifest ".\outputs\local_context_all_whitepad\local_context_manifest.csv" `
  --image-col "model_input_path" `
  --factorial-table ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\hybrid_augmentation_pilot_strict_lopo" `
  --methods baseline colorjitter global_empirical hybrid_empirical `
  --seeds 42 123 `
  --validation-light-count 3 `
  --image-size 224 `
  --batch-size 16 `
  --head-epochs 4 `
  --finetune-epochs 8 `
  --patience 3 `
  --augmentation-prob 0.70 `
  --background-strength 0.50 `
  --global-contrast-strength 0.50 `
  --trajectory-strength 0.50 `
  --knot-count 4 `
  --trajectory-min-ph 2 `
  --trajectory-max-ph 12 `
  --mask-feather-px 16 `
  --dry-run

if ($LASTEXITCODE -ne 0) {
    throw "STRICT LOPO dry-run failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "CHECK BEFORE FULL RUN"
Write-Host "============================================================"
Write-Host "Every fold should show:"
Write-Host "  n_train = 96"
Write-Host "  n_val = 24"
Write-Host "  n_test = 60"
Write-Host "  n_library_rows = 96"
Write-Host "  target_phone_rows_in_library = 0"
Write-Host "  validation_light_rows_in_library = 0"
Write-Host "  n_bg_vectors = 56"
Write-Host "  n_global_contrast_vectors = 2"
Write-Host "  n_trajectory_vectors = 20"
Write-Host "  Mask mode = manifest_roi_model_box"
Write-Host "  STRICT AUDIT: PASS"
