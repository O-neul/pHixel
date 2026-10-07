$ErrorActionPreference = "Stop"

python .\scripts\12_build_factorial_analysis_table.py `
  --input ".\data\processed\capture_manifest.csv" `
  --output-dir ".\outputs\factorial_analysis" `
  --expected-rows 195 `
  --expected-phones 3 `
  --expected-lights 5 `
  --expected-ph "1:13" `
  --strict

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The audit failed. Outputs were still written to:"
    Write-Host ".\outputs\factorial_analysis"
    throw "12_build_factorial_analysis_table.py failed its strict audit."
}

Write-Host ""
Write-Host "Audit items"
Import-Csv ".\outputs\factorial_analysis\design_audit_items.csv" |
Select-Object audit_item, status, observed, expected |
Format-Table -AutoSize

Write-Host ""
Write-Host "Feature ranges"
Import-Csv ".\outputs\factorial_analysis\feature_quality_summary.csv" |
Select-Object feature, n, missing, minimum, mean, standard_deviation, maximum |
Format-Table -AutoSize

Write-Host ""
Write-Host "Main analysis table:"
Write-Host ".\outputs\factorial_analysis\analysis_factorial_table.csv"
