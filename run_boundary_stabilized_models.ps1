$ErrorActionPreference = "Stop"

python .\scripts\19_boundary_stabilized_strip_holdout.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\boundary_stabilized_models" `
  --knot-counts 3 4 5 `
  --primary-knot-count 4 `
  --lambda-grid 0.01 0.1 1 10 100 1000 `
  --inner-folds 5 `
  --bootstrap-reps 3000 `
  --seed 42 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "19_boundary_stabilized_strip_holdout.py failed."
}

Write-Host ""
Write-Host "Primary paired comparisons"
Import-Csv ".\outputs\boundary_stabilized_models\paired_comparisons.csv" |
Where-Object { [int]$_.knot_count -eq 4 } |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    comparison = $_.comparison
    mean_difference = [double]$_.mean_difference
    median_difference = [double]$_.median_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
    proportion_numerator_better = [double]$_.proportion_numerator_better
  }
} | Format-Table -AutoSize

Write-Host ""
Write-Host "Primary ridge RCS boundary/interior comparison"
Import-Csv ".\outputs\boundary_stabilized_models\subset_comparisons.csv" |
Where-Object {
  [int]$_.knot_count -eq 4 -and
  $_.numerator_model -eq "rcs_phone_ridge" -and
  $_.denominator_model -eq "rcs_additive"
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    subset = $_.subset
    mean_difference = [double]$_.mean_difference
    median_difference = [double]$_.median_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
    proportion_phone_rcs_better = [double]$_.proportion_numerator_better
  }
} | Format-Table -AutoSize

Write-Host ""
Write-Host "Negative differences favor rcs_phone_ridge."
Write-Host "Main result: .\outputs\boundary_stabilized_models\subset_comparisons.csv"
