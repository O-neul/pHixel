$ErrorActionPreference = "Stop"

python .\scripts\15_multivariate_lab_asca.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\multivariate_lab" `
  --scalings raw standardized `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "15_multivariate_lab_asca.py failed."
}

Write-Host ""
Write-Host "Standardized multivariate effect composition"

Import-Csv ".\outputs\multivariate_lab\effect_composition.csv" |
Where-Object {
  $_.scaling -eq "standardized"
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    pH_signal = [double]$_.pH_signal_share
    nuisance_main = [double]$_.nuisance_main_share
    pH_dependent_interaction =
      [double]$_.pH_dependent_interaction_share
    residual = [double]$_.residual_share
    total = [double]$_.all_share_sum
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\multivariate_lab\effect_composition.csv"
Write-Host ".\outputs\multivariate_lab\asca_loadings.csv"
Write-Host ".\outputs\multivariate_lab\asca_scores.csv"
Write-Host ".\outputs\multivariate_lab\figures"
