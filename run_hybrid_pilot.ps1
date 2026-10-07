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
Write-Host "23. Run fixed small LOPO augmentation pilot"
Write-Host "============================================================"
Write-Host ""
Write-Host "The pilot will auto-search outputs/ for a CSV manifest that"
Write-Host "contains phone/light/pH plus a precomputed local-context path."
Write-Host "If auto-detection picks the wrong file, run the Python script"
Write-Host "directly with --manifest and, if needed, --image-col."
Write-Host ""

python .\scripts\23_run_hybrid_augmentation_pilot.py `
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
  --mask-feather-px 16

if ($LASTEXITCODE -ne 0) {
    throw "23_run_hybrid_augmentation_pilot.py failed."
}

Write-Host ""
Write-Host "Final pilot outputs:"
Write-Host ".\outputs\hybrid_augmentation_pilot\method_summary.csv"
Write-Host ".\outputs\hybrid_augmentation_pilot\by_phone_summary.csv"
Write-Host ".\outputs\hybrid_augmentation_pilot\paired_comparisons.csv"
Write-Host ".\outputs\hybrid_augmentation_pilot\pilot_decision.csv"
