$ErrorActionPreference = "Stop"

python .\scripts\13_plot_color_profiles_paper.py `
  --input ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\color_profiles_v2" `
  --plot-style "both" `
  --dpi 220

if ($LASTEXITCODE -ne 0) {
    throw "13_plot_color_profiles_paper.py failed."
}

Write-Host ""
Write-Host "Paper figures:"
Write-Host ".\outputs\color_profiles_v2\paper"
Write-Host "Interaction figures:"
Write-Host ".\outputs\color_profiles_v2\interaction_residuals"
Write-Host "Lab separation figures:"
Write-Host ".\outputs\color_profiles_v2\lab_separation"
