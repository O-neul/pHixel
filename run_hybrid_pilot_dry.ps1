$ErrorActionPreference = "Stop"

python .\scripts\22_build_hybrid_augmentation_library.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\hybrid_augmentation_library" `
  --knot-count 4 `
  --trajectory-min-ph 2 `
  --trajectory-max-ph 12

if ($LASTEXITCODE -ne 0) {
    throw "22_build_hybrid_augmentation_library.py failed."
}

python .\scripts\23_run_hybrid_augmentation_pilot.py `
  --library ".\outputs\hybrid_augmentation_library\augmentation_library.json" `
  --output-dir ".\outputs\hybrid_augmentation_pilot" `
  --methods baseline colorjitter global_empirical hybrid_empirical `
  --seeds 42 123 `
  --validation-light-count 3 `
  --dry-run

if ($LASTEXITCODE -ne 0) {
    throw "23_run_hybrid_augmentation_pilot.py dry run failed."
}
