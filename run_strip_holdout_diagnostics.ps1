$ErrorActionPreference = "Stop"

python .\scripts\18_diagnose_strip_holdout_failures.py `
  --input ".\outputs\strip_holdout_spline\cv_predictions.csv" `
  --output-dir ".\outputs\strip_holdout_diagnostics" `
  --primary-basis-count 5 `
  --numerator-model "phone_spline" `
  --denominator-model "spline_additive" `
  --highlight-ph "8,12" `
  --trim-fraction 0.10 `
  --bootstrap-reps 3000 `
  --seed 42 `
  --top-n 10 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "18_diagnose_strip_holdout_failures.py failed."
}

Write-Host ""
Write-Host "Primary robust summary"

Import-Csv `
  ".\outputs\strip_holdout_diagnostics\robust_location_summary.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    mean = [double]$_.mean_difference
    median = [double]$_.median_difference
    trimmed_mean = [double]$_.trimmed_mean_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
    proportion_phone_spline_better =
      [double]$_.proportion_numerator_better
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Boundary and interior sensitivity"

Import-Csv `
  ".\outputs\strip_holdout_diagnostics\subset_sensitivity.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
Where-Object {
  $_.subset -in @(
    "all_ph",
    "boundary_ph_1_13",
    "interior_ph_2_12",
    "strict_interior_ph_3_11",
    "highlighted_ph",
    "non_highlighted_ph"
  )
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    subset = $_.subset
    n_strips = [int]$_.n_strips
    mean = [double]$_.mean_difference
    median = [double]$_.median_difference
    trimmed_mean = [double]$_.trimmed_mean_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Channel differences"

Import-Csv `
  ".\outputs\strip_holdout_diagnostics\channel_difference_summary.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    channel = $_.channel
    mean_rmse_difference = [double]$_.mean_difference
    median_rmse_difference = [double]$_.median_difference
    trimmed_mean_difference =
      [double]$_.trimmed_mean_difference
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Largest deterioration strips"

Import-Csv `
  ".\outputs\strip_holdout_diagnostics\worst_strips.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
Select-Object `
  region,
  rank,
  strip_id,
  phone,
  ph_nominal,
  mean_deltaE76_difference,
  rmse_L_difference,
  rmse_a_difference,
  rmse_b_difference |
Format-Table -AutoSize

Write-Host ""
Write-Host "Negative differences favor phone_spline."
Write-Host "Main result:"
Write-Host ".\outputs\strip_holdout_diagnostics\subset_sensitivity.csv"
