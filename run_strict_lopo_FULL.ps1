$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "24. STRICT target-free LOPO FINAL rerun"
Write-Host "============================================================"
Write-Host ""
Write-Host "IMPORTANT:"
Write-Host "  - target phone is excluded from CNN training"
Write-Host "  - target phone is excluded from augmentation-library estimation"
Write-Host "  - validation light is excluded from augmentation-library estimation"
Write-Host "  - all previous augmentation/training settings remain fixed"
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
  --mask-feather-px 16

if ($LASTEXITCODE -ne 0) {
    throw "STRICT LOPO full rerun failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "STRICT LOPO RERUN COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Main outputs:"
Write-Host "  .\outputs\hybrid_augmentation_pilot_strict_lopo\fold_and_library_audit.csv"
Write-Host "  .\outputs\hybrid_augmentation_pilot_strict_lopo\method_summary.csv"
Write-Host "  .\outputs\hybrid_augmentation_pilot_strict_lopo\by_phone_summary.csv"
Write-Host "  .\outputs\hybrid_augmentation_pilot_strict_lopo\paired_comparisons.csv"
Write-Host "  .\outputs\hybrid_augmentation_pilot_strict_lopo\pilot_decision.csv"
Write-Host ""
Write-Host "Do NOT retune settings after inspecting these results."
