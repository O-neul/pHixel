$ErrorActionPreference = "Stop"

python .\scripts\20_summarize_asca_interactions.py `
  --variance ".\outputs\multivariate_lab\asca_variance.csv" `
  --loadings ".\outputs\multivariate_lab\asca_loadings.csv" `
  --scores ".\outputs\multivariate_lab\asca_scores.csv" `
  --output-dir ".\outputs\asca_interaction_summary" `
  --scaling standardized `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "20_summarize_asca_interactions.py failed."
}

Write-Host ""
Write-Host "ASCA loading summary"
Import-Csv ".\outputs\asca_interaction_summary\loading_summary.csv" |
Format-Table -AutoSize

Write-Host ""
Write-Host "Peak pH x phone interactions"
Import-Csv ".\outputs\asca_interaction_summary\phone_peak_interactions.csv" |
Format-Table -AutoSize
