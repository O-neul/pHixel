$ErrorActionPreference = "Stop"

Rscript .\scripts\27_gam_leave_one_strip_out.R `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\gam_strip_holdout" `
  --specs all

if ($LASTEXITCODE -ne 0) {
    throw "27_gam_leave_one_strip_out.R failed."
}

python .\scripts\27_summarize_gam_strip_holdout.py `
  --holdout-dir ".\outputs\gam_strip_holdout" `
  --rcs-subsets ".\outputs\boundary_stabilized_models\subset_comparisons.csv" `
  --bootstrap-reps 3000 `
  --seed 42 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "27_summarize_gam_strip_holdout.py failed."
}

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\gam_strip_holdout\subset_comparisons.csv"
Write-Host ".\outputs\gam_strip_holdout\gam_rcs_consistency.csv"
Write-Host ".\outputs\gam_strip_holdout\analysis_notes.txt"
Write-Host ".\outputs\gam_strip_holdout\figures"
