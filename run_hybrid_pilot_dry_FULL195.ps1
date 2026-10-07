$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "22. Build fixed hybrid augmentation library"
Write-Host "============================================================"

python .\scripts\22_build_hybrid_augmentation_library.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\hybrid_augmentation_library" `
  --knot-count 4 `
  --trajectory-min-ph 2 `
  --trajectory-max-ph 12

if ($LASTEXITCODE -ne 0) {
    throw "22_build_hybrid_augmentation_library.py failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "23. Dry-run with the FULL 195-image local-context manifest"
Write-Host "============================================================"

python .\scripts\23_run_hybrid_augmentation_pilot.py `
  --manifest ".\outputs\local_context_all_whitepad\local_context_manifest.csv" `
  --image-col "model_input_path" `
  --library ".\outputs\hybrid_augmentation_library\augmentation_library.json" `
  --output-dir ".\outputs\hybrid_augmentation_pilot" `
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
  --mask-feather-px 16 `
  --dry-run

if ($LASTEXITCODE -ne 0) {
    throw "23_run_hybrid_augmentation_pilot.py dry-run failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "DRY-RUN COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Before the full pilot, check that every fold has:"
Write-Host "  n_train = 96"
Write-Host "  n_val   = 24"
Write-Host "  n_test  = 60"
Write-Host ""
Write-Host "Also confirm:"
Write-Host "  Manifest = outputs\local_context_all_whitepad\local_context_manifest.csv"
Write-Host "  Image column = model_input_path"
Write-Host "  Phones = 3"
Write-Host "  Validation lights = L01, L03, L05"
