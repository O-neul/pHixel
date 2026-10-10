$ErrorActionPreference = "Stop"

Rscript .\scripts\26_gam_phone_specific_curves.R `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\gam_curves" `
  --n-sim 10000 `
  --seed 20261009

if ($LASTEXITCODE -ne 0) {
    throw "26_gam_phone_specific_curves.R failed."
}

python .\scripts\26_plot_gam_curves.py `
  --gam-dir ".\outputs\gam_curves" `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "26_plot_gam_curves.py failed."
}

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\gam_curves\analysis_audit.txt"
Write-Host ".\outputs\gam_curves\smooth_terms.csv"
Write-Host ".\outputs\gam_curves\variance_components.csv"
Write-Host ".\outputs\gam_curves\difference_regions.csv"
Write-Host ".\outputs\gam_curves\figures"
