$ErrorActionPreference = "Stop"

python .\scripts\21_build_final_evidence_synthesis.py `
  --region-effects ".\outputs\factorial_effects\region_effect_summary.csv" `
  --multivariate ".\outputs\multivariate_lab\effect_composition.csv" `
  --cell-model-comparison ".\outputs\model_structure_comparison\paired_model_comparison.csv" `
  --strip-diagnostics ".\outputs\strip_holdout_diagnostics\subset_sensitivity.csv" `
  --boundary-stabilized ".\outputs\boundary_stabilized_models\subset_comparisons.csv" `
  --asca ".\outputs\asca_interaction_summary\loading_summary.csv" `
  --output-dir ".\outputs\final_evidence_synthesis" `
  --primary-knot-count 4

if ($LASTEXITCODE -ne 0) {
    throw "21_build_final_evidence_synthesis.py failed."
}

Write-Host ""
Write-Host "Final evidence summary"
Import-Csv ".\outputs\final_evidence_synthesis\evidence_summary.csv" |
Format-Table -AutoSize

Write-Host ""
Write-Host "Decision questions"
Import-Csv ".\outputs\final_evidence_synthesis\decision_questions.csv" |
Format-Table -AutoSize

Write-Host ""
Write-Host "Main result: .\outputs\final_evidence_synthesis\region_evidence_matrix.csv"
