$ErrorActionPreference = "Stop"

python .\scripts\17_leave_one_strip_out_spline_models.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\strip_holdout_spline" `
  --basis-counts 4 5 6 `
  --primary-basis-count 5 `
  --degree 3 `
  --models spline_additive phone_spline full_spline_interactions `
  --bootstrap-reps 3000 `
  --seed 42 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "17_leave_one_strip_out_spline_models.py failed."
}

Write-Host ""
Write-Host "Primary cross-validation summary: basis count 5"

Import-Csv ".\outputs\strip_holdout_spline\cv_summary.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    model = $_.model
    standardized_rmse = [double]$_.standardized_rmse
    mean_deltaE76 = [double]$_.mean_deltaE76
    median_deltaE76 = [double]$_.median_deltaE76
    q90_deltaE76 = [double]$_.q90_deltaE76
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Primary paired comparisons: basis count 5"

Import-Csv `
  ".\outputs\strip_holdout_spline\paired_model_comparison.csv" |
Where-Object {
  [int]$_.basis_count -eq 5
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    metric = $_.metric
    comparison = $_.comparison
    mean_difference = [double]$_.mean_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
    proportion_numerator_better =
      [double]$_.proportion_strips_numerator_better
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Spline sensitivity"

Import-Csv `
  ".\outputs\strip_holdout_spline\spline_sensitivity_summary.csv" |
Where-Object {
  $_.model -eq "phone_spline"
} |
ForEach-Object {
  [PSCustomObject]@{
    basis_count = [int]$_.basis_count
    region = $_.region
    model = $_.model
    standardized_rmse = [double]$_.standardized_rmse
    mean_deltaE76 = [double]$_.mean_deltaE76
    phone_spline_minus_additive =
      [double]$_.mean_difference
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Negative differences favor the numerator model."
Write-Host "Main result:"
Write-Host ".\outputs\strip_holdout_spline\paired_model_comparison.csv"
