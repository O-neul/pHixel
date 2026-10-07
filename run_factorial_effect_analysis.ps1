$ErrorActionPreference = "Stop"

python .\scripts\14_factorial_effect_analysis.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\factorial_effects"

if ($LASTEXITCODE -ne 0) {
    throw "14_factorial_effect_analysis.py failed."
}

Write-Host ""
Write-Host "Region effect summary"

Import-Csv `
  ".\outputs\factorial_effects\region_effect_summary.csv" |
Format-Table -AutoSize

Write-Host ""
Write-Host "Feature effect composition"

Import-Csv `
  ".\outputs\factorial_effects\effect_composition.csv" |
ForEach-Object {
  [PSCustomObject]@{
    feature = $_.feature
    pH_signal = [double]$_.pH_signal_ratio
    nuisance_main = [double]$_.nuisance_main_ratio
    pH_dependent_interaction =
      [double]$_.pH_dependent_interaction_ratio
    residual = [double]$_.residual_ratio
    total = [double]$_.all_ratio_sum
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Interaction effects"

Import-Csv `
  ".\outputs\factorial_effects\feature_effect_summary.csv" |
ForEach-Object {
  [PSCustomObject]@{
    feature = $_.feature
    pH_x_phone_eta_sq = [double]$_.pH_x_phone_eta_sq
    pH_x_phone_partial_eta_sq =
      [double]$_.pH_x_phone_partial_eta_sq
    pH_x_phone_p = [double]$_.pH_x_phone_p_value
    pH_x_light_eta_sq = [double]$_.pH_x_light_eta_sq
    pH_x_light_partial_eta_sq =
      [double]$_.pH_x_light_partial_eta_sq
    pH_x_light_p = [double]$_.pH_x_light_p_value
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Main result:"
Write-Host ".\outputs\factorial_effects\feature_effect_summary.csv"
