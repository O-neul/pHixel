$ErrorActionPreference = "Stop"

python .\scripts\13_plot_color_profiles.py `
  --input `
    ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\color_profiles" `
  --uncertainty "range" `
  --highlight-ph "8,12" `
  --dpi 180

if ($LASTEXITCODE -ne 0) {
    throw "13_plot_color_profiles.py failed."
}

Write-Host ""
Write-Host "Largest phone separation by region"

Import-Csv `
  ".\outputs\color_profiles\tables\domain_dispersion_by_ph.csv" |
Where-Object {
  $_.domain_type -eq "phone"
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    pH = [double]$_.ph_nominal
    max_pairwise_deltaE = [double]$_.maximum_pairwise_deltaE76
    mean_pairwise_deltaE = [double]$_.mean_pairwise_deltaE76
    centroid_rms_distance = [double]$_.centroid_rms_distance
  }
} |
Sort-Object region, @{Expression = "max_pairwise_deltaE"; Descending = $true} |
Group-Object region |
ForEach-Object {
  $_.Group | Select-Object -First 5
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Largest pH x phone interaction RMS"

Import-Csv `
  ".\outputs\color_profiles\tables\interaction_strength_by_ph.csv" |
Where-Object {
  $_.domain_type -eq "phone"
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    component = $_.component
    pH = [double]$_.ph_nominal
    interaction_rms = [double]$_.interaction_rms
  }
} |
Sort-Object region, component, `
  @{Expression = "interaction_rms"; Descending = $true} |
Group-Object region, component |
ForEach-Object {
  $_.Group | Select-Object -First 3
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Inspect figures:"
Write-Host ".\outputs\color_profiles\figures\phone_profiles"
Write-Host ".\outputs\color_profiles\figures\phone_interaction_residuals"
Write-Host ".\outputs\color_profiles\figures\lab_separation"
