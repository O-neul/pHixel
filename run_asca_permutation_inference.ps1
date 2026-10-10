$ErrorActionPreference = "Stop"

python .\scripts\25_asca_permutation_inference.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\asca_permutation" `
  --scalings raw standardized `
  --n-permutations 9999 `
  --n-bootstrap 2000 `
  --seed 20261009

if ($LASTEXITCODE -ne 0) {
    throw "25_asca_permutation_inference.py failed."
}

Write-Host ""
Write-Host "Standardized restricted / Freedman-Lane tests"

Import-Csv ".\outputs\asca_permutation\permutation_tests.csv" |
Where-Object {
  $_.scaling -eq "standardized" -and $_.scheme -ne "unrestricted"
} |
ForEach-Object {
  [PSCustomObject]@{
    region = $_.region
    term = $_.term
    share = [double]$_.trace_share
    pseudo_F = [double]$_.pseudo_f
    p = [double]$_.p_value
    p_holm = [double]$_.p_holm_within_region
  }
} |
Format-Table -AutoSize

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\asca_permutation\permutation_tests.csv"
Write-Host ".\outputs\asca_permutation\loading_bootstrap.csv"
Write-Host ".\outputs\asca_permutation\direction_angle_bootstrap.csv"
Write-Host ".\outputs\asca_permutation\analysis_notes.txt"
