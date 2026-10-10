$ErrorActionPreference = "Stop"

Rscript .\scripts\28_gam_inverse_ph_prediction.R `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\gam_inverse_ph" `
  --specs all `
  --grid-step 0.05

if ($LASTEXITCODE -ne 0) {
    throw "28_gam_inverse_ph_prediction.R failed."
}

python .\scripts\28_summarize_inverse_ph.py `
  --inverse-dir ".\outputs\gam_inverse_ph" `
  --holdout-dir ".\outputs\gam_strip_holdout" `
  --curve-dir ".\outputs\gam_curves" `
  --bootstrap-reps 3000 `
  --seed 42 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "28_summarize_inverse_ph.py failed."
}

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\gam_inverse_ph\subset_comparisons.csv"
Write-Host ".\outputs\gam_inverse_ph\summary_by_model.csv"
Write-Host ".\outputs\gam_inverse_ph\analysis_notes.txt"
Write-Host ".\outputs\gam_inverse_ph\figures"
