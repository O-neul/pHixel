$ErrorActionPreference = "Stop"

python .\scripts\16_compare_additive_interaction_models.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\model_structure_comparison" `
  --schemes leave_one_cell_out cyclic_light_block `
  --models main_effects additive_nuisance ph_dependent_interactions `
  --bootstrap-reps 2000 `
  --seed 42 `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "16_compare_additive_interaction_models.py failed."
}

Write-Host ""
Write-Host "Cross-validation summary"

Import-Csv ".\outputs\model_structure_comparison\cv_summary.csv" |
ForEach-Object {
  [PSCustomObject]@{
    scheme = $_.scheme
    region = $_.region
    model = $_.model
    standardized_rmse = [double]$_.standardized_rmse
    mean_deltaE76 = [double]$_.mean_deltaE76
    q90_deltaE76 = [double]$_.q90_deltaE76
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Paired full-minus-additive comparison"

Import-Csv `
  ".\outputs\model_structure_comparison\paired_model_comparison.csv" |
ForEach-Object {
  [PSCustomObject]@{
    scheme = $_.scheme
    region = $_.region
    metric = $_.metric
    mean_difference = [double]$_.mean_difference
    ci_lower = [double]$_.bootstrap_ci_lower
    ci_upper = [double]$_.bootstrap_ci_upper
    probability_below_zero =
      [double]$_.bootstrap_probability_difference_below_zero
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Negative differences favor the pH-dependent interaction model."
Write-Host "Main result:"
Write-Host ".\outputs\model_structure_comparison\paired_model_comparison.csv"
